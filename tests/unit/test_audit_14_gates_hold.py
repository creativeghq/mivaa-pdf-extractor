"""Guards for the mivaa#14 fixes (MV-1 …).

MV-1 is the sharpest silent zero in this codebase so far, and worth restating
because the shape recurs: both RAG entry points retrieved context with
`multi_vector_search`, whose rows are PRODUCT-shaped —
`{id, product_name, description, metadata, score, ...}`. The context builder read
`chunk.get('content', chunk.get('text', ''))`. Neither key exists on that shape.

So every RAG answer was synthesised from relevance headers with empty bodies,
under the instruction "answer based ONLY on the provided context".

Nothing could see it:
  * the `if not chunks:` guard PASSED — rows existed, they were the wrong shape
  * the Claude call succeeded, so it was billed and logged as a success
  * the response reported N chunks retrieved
  * the answer was a polite refusal or a hallucination from the question alone,
    both of which read as ordinary model behaviour

`ops.silent_zero` cannot catch it either: the operation self-reports success.

Static, over source text — CI installs pytest alone, so nothing here imports `app`.

Every case was watched to FAIL against the pre-fix tree.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
RAG = APP / "services" / "search" / "rag_service.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def _func(path: Path, name: str) -> str:
    src = _read(path)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    raise AssertionError(f"{name} not found in {path.name} — re-point this guard")


RAG_ENTRY_POINTS = ["query_document", "advanced_rag_query"]


@pytest.mark.parametrize("fn", RAG_ENTRY_POINTS)
def test_rag_context_is_not_read_off_the_product_shaped_path(fn: str):
    body = _strip_comments(_func(RAG, fn))

    assert "multi_vector_search" not in body, (
        f"{fn} retrieves context from multi_vector_search again. Its rows carry no "
        "`content` and no `text` key, so the context is empty on every call and the "
        "model is asked to answer from nothing."
    )
    assert "_retrieve_context_chunks" in body, (
        f"{fn} no longer goes through the chunk retrieval path"
    )


@pytest.mark.parametrize("fn", RAG_ENTRY_POINTS)
def test_rag_entry_points_distinguish_a_fault_from_an_empty_result(fn: str):
    """A retrieval fault and "this document has nothing relevant" must not be the
    same answer — one of them should retry (pipeline convention 1)."""
    body = _strip_comments(_func(RAG, fn))
    assert "ContextRetrievalError" in body, (
        f"{fn} no longer separates a retrieval fault from a genuine empty result"
    )


def test_the_retrieval_helper_refuses_to_search_every_tenant():
    body = _strip_comments(_func(RAG, "_retrieve_context_chunks"))
    assert "if not workspace_id" in body, (
        "the retrieval helper no longer requires a workspace — without it the RPC "
        "would be asked for chunks with no tenancy predicate"
    )
    assert "search_rag_context_chunks" in body, (
        "the helper no longer calls the document-scoped RPC. The two sibling "
        "functions cannot filter by document, and the product path applies "
        "document_id as a score BOOST, which constrains nothing."
    )
    assert "raise ContextRetrievalError" in body, (
        "the helper swallows failures into an empty list again"
    )
    assert re.search(r"except Exception[\s\S]{0,200}raise ContextRetrievalError", body), (
        "a database failure is no longer re-raised as a typed error, so an outage "
        "looks identical to a document with no relevant chunks"
    )


def test_the_retrieved_context_is_fenced_as_data():
    """MV-11. Chunk text comes out of supplier PDFs — untrusted content being
    interpolated into a prompt (invariant 9). It was undelimited, and masked only
    because MV-1 meant the context was empty: fixing retrieval is what activates
    the injection surface, which is why the two ship together."""
    body = _strip_comments(_func(RAG, "_build_fenced_context"))
    assert "DATA ONLY" in body, "the retrieved-excerpt fence lost its DATA-only marker"
    assert "BEGIN RETRIEVED DOCUMENT EXCERPTS" in body and "END RETRIEVED" in body, (
        "the fence no longer delimits where untrusted excerpts start and end"
    )

    for fn in RAG_ENTRY_POINTS:
        entry = _strip_comments(_func(RAG, fn))
        assert "_build_fenced_context" in entry, (
            f"{fn} assembles its own context again instead of using the fenced builder"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Paid spend, tenancy, and the silent zeros
# ═══════════════════════════════════════════════════════════════════════════

DFS = APP / "services" / "integrations" / "dataforseo_unified_client.py"
PPX = APP / "services" / "integrations" / "perplexity_price_search_service.py"
LINK = APP / "services" / "discovery" / "entity_linking_service.py"
DISC = APP / "services" / "discovery" / "product_discovery_service.py"


def test_dataforseo_debits_before_the_upstream_call():
    """MV-3. This is the LAST layer that could gate DataForSEO spend — #352 A18, the
    MIVAA route check, #361 EG-4 and #365 AD-13 each confirmed no other layer does."""
    body = _strip_comments(_func(DFS, "_call"))
    assert "_charge_for_call" in body, "the debit is gone from the DataForSEO call path"
    charge_at = body.index("_charge_for_call")
    post_at = min((body.index(m) for m in ("client.post", "client.get") if m in body),
                  default=len(body))
    assert charge_at < post_at, (
        "the debit runs AFTER the request — invariant 10 requires it before, and no "
        "work on debit failure"
    )


def test_a_metering_fault_does_not_become_free_spend():
    body = _strip_comments(_func(DFS, "_charge_for_call"))
    assert "return False" in body, (
        "an exception in the debit path no longer refuses the call, so a billing "
        "outage converts straight into free provider spend"
    )


def test_perplexity_debits_before_it_searches():
    body = _strip_comments(_func(PPX, "_perplexity_call"))
    assert "_reserve_spend" in body, "the Perplexity search no longer reserves credits"
    assert body.index("_reserve_spend") < body.index("client.post"), (
        "credits are reserved after the provider call again"
    )


def test_every_paid_failure_path_still_records_the_call():
    """MV-5. Timeouts, request failures and non-200s all returned before _log_usage, so
    the one class of paid call you would most want in the cost table never appeared."""
    body = _strip_comments(_func(PPX, "_perplexity_call"))
    assert body.count("_log_failed_call") >= 3, (
        "a paid Perplexity failure path returns without recording the call"
    )
    dfs = _strip_comments(_func(DFS, "_call"))
    parse_at = dfs.index("json parse")
    assert "_log_cost" in dfs[max(0, parse_at - 400):parse_at + 400], (
        "the JSON-parse failure returns without _log_cost again — that paid call "
        "vanishes entirely"
    )


def test_the_dataforseo_retry_keeps_its_method():
    """MV-6. The 5xx retry issued client.post() regardless of the original method, so
    retrying a GET silently became a POST to a paid API."""
    body = _strip_comments(_func(DFS, "_call"))
    retry_at = body.index("retry")
    window = body[retry_at:retry_at + 700]
    assert 'method == "GET"' in window, (
        "the retry no longer branches on the original method"
    )


def test_entity_linking_reconciles_its_two_ids():
    """MV-12 — seventh instance of the two-unchecked-ids class, and MIVAA has no RLS."""
    body = _strip_comments(_func(LINK, "link_product_entities"))
    assert "_assert_product_belongs_to_document" in body, (
        "link_product_entities writes relationships by product_id after reading by "
        "document_id, without proving they belong together"
    )


def test_linking_failure_is_not_a_count_of_zero():
    """MV-14. A DB outage and 'nothing eligible to link' both returned 0, so nothing
    retried and products stayed silently unlinked."""
    src = _strip_comments(_read(LINK))
    assert "LINKING_FAILED" in src, "the explicit linking-failure marker is gone"
    for fn in ("_link_images_to_chunks", "link_chunks_to_products"):
        try:
            body = _strip_comments(_func(LINK, fn))
        except AssertionError:
            continue
        if "except Exception" in body:
            assert "return 0" not in body.split("except Exception")[-1], (
                f"{fn} swallows a failure into a zero count again"
            )


def test_product_rehydration_keeps_the_workspace_predicate():
    """MV-13. The initial vector search is scoped; the association hops were not, so a
    stale or crafted association row bridges a scoped hit into another tenant."""
    body = _strip_comments(_func(RAG, "multi_vector_search"))
    assert body.count("eq('workspace_id', workspace_id)") >= 3, (
        "an association/product rehydration hop dropped its workspace filter"
    )


def test_discovery_says_when_it_could_not_tell():
    """MV-16. 'Vision retry failed (non-fatal)' then `return catalog` made a broken
    discovery indistinguishable from a genuinely product-free document."""
    src = _strip_comments(_read(DISC))
    assert "discovery_status" in src, "the discovery outcome marker is gone"
    assert "vision_retry_failed" in src, (
        "a failed vision retry no longer marks the catalog, so zero products reads as "
        "a clean verdict"
    )


def test_model_output_does_not_become_a_product_field_on_its_own_say_so():
    """MV-15. `material_metadata_fields` decides what a field IS. A key the model
    invented is data worth keeping, but it is not a field."""
    body = _strip_comments(_func(DISC, "_partition_by_registry"))
    assert "field_registry" in body, "the registry partition no longer consults the registry"
    assert "is_loaded" in body, (
        "the partition can now raise mid-pipeline instead of degrading to its previous "
        "behaviour"
    )
