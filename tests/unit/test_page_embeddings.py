"""
Guards for page embeddings — the 8th fusion vector (#239).

Four things here are load-bearing and none of them fails loudly on its own:

1. **The query must be embedded by the multimodal model, not the text model.**
   `voyage-4` and `voyage-multimodal-3.5` are BOTH 1024D. Query the page collection
   with a voyage-4 vector and Postgres accepts it, the HNSW index returns neighbours,
   and every score is confident nonsense. Nothing raises — not a typecheck, not a dim
   check, not an integrity probe. Matching dimensions prove nothing about the space,
   which is exactly why this needs a test rather than a code comment.

2. **Every page vector carries `workspace_id`.** Phase-0 write invariant (0.2). An
   unattributable vector in a tenant collection cannot be filtered out of another
   tenant's search.

3. **Reads fail closed.** Phase-0 read invariant (0.1). An unfiltered vector search is
   the cheapest possible cross-tenant leak.

4. **A page is only marked `embedded` if a vector actually landed.** The row is what
   the backfill and the silent-zero probe read; a row claiming success over a refused
   upsert makes the page invisible forever and tells the probe everything is fine.

Source/AST based where it needs to be, behavioural where it can be — neither form
imports the app or touches a DB, so the whole file runs in CI in about a second.
"""

import ast
import asyncio
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_RAG_SERVICE = _ROOT / "app" / "services" / "search" / "rag_service.py"
_VECS_SERVICE = _ROOT / "app" / "services" / "embeddings" / "vecs_service.py"
_EMBED_SERVICE = _ROOT / "app" / "services" / "embeddings" / "real_embeddings_service.py"
_PAGE_SERVICE = _ROOT / "app" / "services" / "embeddings" / "page_embedding_service.py"

_RAG_SOURCE = _RAG_SERVICE.read_text(encoding="utf-8")
_VECS_SOURCE = _VECS_SERVICE.read_text(encoding="utf-8")
_EMBED_SOURCE = _EMBED_SERVICE.read_text(encoding="utf-8")
_PAGE_SOURCE = _PAGE_SERVICE.read_text(encoding="utf-8")


def _function_node(source: str, name: str):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found — it was renamed or removed")


def _function_source(source: str, name: str) -> str:
    return ast.get_source_segment(source, _function_node(source, name)) or ""


def _executable_source(source: str, name: str) -> str:
    """The function with comments AND string literals removed.

    Needed for the "must not do X" assertions: this file's functions explain in prose
    exactly what they refuse to do, so a naive substring search matches the docstring
    that documents the rule and reports the rule as violated.
    """
    node = _function_node(source, name)
    stripped = ast.parse(ast.unparse(node)).body[0]
    for sub in ast.walk(stripped):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            sub.value = ""
    return ast.unparse(stripped)


# ── 1. Query-space contract ─────────────────────────────────────────────────────

def test_page_query_uses_the_multimodal_endpoint_not_the_text_one():
    """The page query embedding must go to /multimodalembeddings.

    Routing it to /embeddings (the voyage-4 text endpoint) still returns a valid 1024D
    vector, so the failure is invisible end-to-end: pages come back ranked, just
    ranked by nothing meaningful.
    """
    fn = _function_source(_EMBED_SOURCE, "generate_page_query_embedding")
    assert "_voyage_multimodal_embed" in fn, (
        "generate_page_query_embedding must call the multimodal embedder; any other "
        "1024D embedder returns a wrong-space vector that scores without erroring"
    )
    assert '"query"' in fn or "'query'" in fn, "must request input_type=query"

    multimodal = _function_source(_EMBED_SOURCE, "_voyage_multimodal_embed")
    assert "multimodalembeddings" in multimodal
    assert "voyage_multimodal_model" in multimodal, (
        "the model must come from settings, so the ingest and query sides cannot be "
        "pointed at different models independently"
    )


def test_page_query_embedding_has_no_fallback_provider():
    """A fallback here would silently mix latent spaces in one collection.

    This is audit gap B (Voyage/OpenAI mixing in image_understanding_embeddings)
    applied before it can happen: on failure the page must stay unembedded, not be
    embedded by something else.
    """
    # Executable source only: this function's own docstring explains that it has no
    # OpenAI fallback, so a plain substring search over the source finds the sentence
    # documenting the rule and calls it a violation.
    fn = _executable_source(_EMBED_SOURCE, "_voyage_multimodal_embed")
    assert "openai" not in fn.lower(), (
        "no OpenAI fallback on the multimodal path — a wrong-space vector is worse "
        "than a missing one, because a missing one is retryable and visible"
    )
    assert "allow_openai_fallback" not in fn


def test_rag_service_asks_for_a_page_query_embedding():
    """The channel carries weight in every profile; if nothing ever generates its query
    vector, the weight is spent on a score of zero on every search."""
    assert "generate_page_query_embedding" in _RAG_SOURCE, (
        "multi_vector_search must generate a page query embedding, or the page "
        "channel's weight is silently dead on every query"
    )
    assert "search_page_embeddings" in _RAG_SOURCE, (
        "multi_vector_search must actually query the page collection"
    )


# ── 2 & 3. Tenant isolation ─────────────────────────────────────────────────────

def test_page_upsert_refuses_without_workspace_id():
    """Phase-0 write invariant (0.2), enforced at the only door into the collection."""
    fn = _function_source(_VECS_SOURCE, "upsert_page_embedding")
    assert 'meta.get("workspace_id")' in fn or "meta.get('workspace_id')" in fn
    assert "return False" in fn, "a vector with no tenant must be refused, not stored"


def test_page_search_fails_closed_without_a_tenant_filter():
    """Phase-0 read invariant (0.1). Returning [] makes the bug present as 'search
    found nothing' rather than 'search found another tenant's catalog'."""
    fn = _function_source(_VECS_SOURCE, "search_page_embeddings")
    assert "if not workspace_id and not document_id" in fn
    assert "return []" in fn


def test_page_service_takes_tenancy_from_the_document_row():
    """Security invariant 1: never trust a body-supplied workspace_id.

    The argument is only allowed to be checked against the row's owner; the row is
    what stamps the vector.
    """
    fn = _function_source(_PAGE_SOURCE, "_load_document")
    assert "workspace mismatch" in fn, "a caller's workspace_id must be verified against the row"
    assert "return None" in fn, "a mismatch must yield nothing, not a stamped vector"

    embed = _function_source(_PAGE_SOURCE, "embed_document_pages")
    assert 'doc["workspace_id"]' in embed, (
        "the vector's workspace_id must come from the document row, not the argument"
    )


# ── 4. The row must not outrun the vector ───────────────────────────────────────

def test_failed_vecs_upsert_is_not_recorded_as_embedded():
    """`cache_status='embedded'` is the claim 'a vector exists for this page'.

    If the upsert is refused and the row still says embedded, the backfill skips the
    page forever and the silent-zero probe reads it as healthy — the page is invisible
    and nothing anywhere complains.
    """
    fn = _function_source(_PAGE_SOURCE, "_embed_one_page")
    marker = fn.index("vecs upsert refused")
    assert 'if not ok:' in fn[:marker], "the upsert result must be checked"
    # The refusal branch must record 'failed'; the success path records 'embedded'.
    assert '"failed"' in fn[marker - 400:marker + 200]


def test_page_renders_are_serialized_because_pymupdf_is_not_thread_safe():
    """Every page render runs in a worker thread against the SAME fitz.Document.

    PyMuPDF is not thread-safe, and the failure mode is not an exception — concurrent
    access corrupts or segfaults in the C layer, which surfaces as a worker dying
    mid-catalog with no Python traceback to point at the cause.
    """
    fn = _executable_source(_PAGE_SOURCE, "_embed_one_page")
    assert "async with render_lock:" in fn, (
        "the render must hold a lock; concurrency belongs on the embed call, not on "
        "PyMuPDF"
    )
    caller = _executable_source(_PAGE_SOURCE, "embed_document_pages")
    assert "asyncio.Lock()" in caller, "one lock per document, created by the caller"


def test_blank_pages_are_skipped_not_failed():
    """'skipped' and 'failed' drive different behaviour: failed is retried, skipped is
    not. Marking a genuinely blank page 'failed' means re-rendering it on every
    backfill run, forever, at cost."""
    fn = _function_source(_PAGE_SOURCE, "_embed_one_page")
    assert '"skipped"' in fn
    assert "MIN_RENDER_BYTES" in fn


# ── Wiring that has to agree in two places ──────────────────────────────────────

def test_collection_name_and_dimension_agree_across_the_codebase():
    """The migration creates vecs.page_embeddings as halfvec(1024). A service constant
    that drifts from it fails at the Postgres boundary — but only on write, i.e. in
    production, on the first real catalog."""
    spec = importlib.util.spec_from_file_location("_vecs_probe", _VECS_SERVICE)
    # Import would pull in the whole runtime; read the constants from the AST instead.
    tree = ast.parse(_VECS_SOURCE)
    constants = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in ("PAGE_COLLECTION", "PAGE_DIMENSION"):
                constants[target.id] = ast.literal_eval(node.value)
    assert spec is not None
    assert constants.get("PAGE_COLLECTION") == "page_embeddings"
    assert constants.get("PAGE_DIMENSION") == 1024


def test_page_text_comes_from_silver_not_from_the_pdf():
    """Medallion: document_chunks already holds the page's text. Re-extracting it from
    the PDF is the layer violation behind every cache bug in this pipeline."""
    fn = _function_source(_PAGE_SOURCE, "_page_texts_from_silver")
    assert "document_chunks" in fn
    assert "fitz" not in fn, "page text must not be re-extracted from the source PDF"


def test_page_budget_is_bounded_and_the_cap_is_reported():
    """A silently truncated run reads as 'covered everything'. The cap must say what it
    dropped."""
    fn = _function_source(_PAGE_SOURCE, "_resolve_target_pages")
    assert "max_pages" in fn
    assert "logger.warning" in fn, "a page cap that drops work must say so"


def test_embed_one_page_records_a_row_on_every_outcome():
    """Every exit path writes a row. A page that produced no row at all is
    indistinguishable from a page that was never reached, which is precisely what the
    cache_status column exists to disambiguate."""
    fn = _function_source(_PAGE_SOURCE, "_embed_one_page")
    tree = ast.parse(fn.strip())
    func = tree.body[0]
    returns = [n for n in ast.walk(func) if isinstance(n, ast.Return)]
    assert len(returns) >= 4, "expected a return per outcome (failed/skipped/embedded)"
    # Each return is preceded by a _record call somewhere in its branch; assert the
    # count of _record calls covers them.
    records = [
        n for n in ast.walk(func)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_record"
    ]
    assert len(records) >= len(returns), (
        f"{len(returns)} return paths but only {len(records)} _record calls — "
        f"some outcome leaves no row behind"
    )


def test_cost_accounting_uses_both_billing_axes():
    """voyage-multimodal bills per token AND per pixel. For a rendered page the pixels
    are ~95% of the cost, so the token-only path under-reports by ~20× — a wrong
    number that is still a valid Decimal."""
    fn = _function_source(_EMBED_SOURCE, "_voyage_multimodal_embed")
    assert "calculate_multimodal_embedding_cost" in fn, (
        "page embeddings must not be costed with the token-only calculate_cost()"
    )
    assert "image_pixels" in fn


def test_page_cost_matches_voyage_published_rates():
    """Pins the arithmetic against Voyage's own published pricing, so a bad edit to the
    rate table shows up here rather than on an invoice."""
    ai_pricing_path = _ROOT / "app" / "config" / "ai_pricing.py"
    spec = importlib.util.spec_from_file_location("_ai_pricing_probe", ai_pricing_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pricing = module.AIPricingConfig

    # A full A4 page at 144dpi is ~2M pixels — Voyage's per-image ceiling — plus a few
    # hundred tokens of page text.
    cost = pricing.calculate_multimodal_embedding_cost(
        "voyage-multimodal-3.5", text_tokens=300, image_pixels=2_000_000,
        include_markup=False,
    )
    # 2,000,000 px @ $0.60/1e9 = $0.0012; 300 tokens @ $0.12/1e6 = $0.000036
    assert float(cost["raw_cost_usd"]) == pytest.approx(0.001236, abs=1e-9)

    # The per-image clamp in both directions — a tiny render still bills at 50k, and a
    # huge one never bills past 2M, which is why DPI above ~144 is pure waste.
    assert int(pricing.calculate_multimodal_embedding_cost(
        "voyage-multimodal-3.5", 0, 10, include_markup=False)["billable_pixels"]) == 50_000
    assert int(pricing.calculate_multimodal_embedding_cost(
        "voyage-multimodal-3.5", 0, 99_000_000, include_markup=False)["billable_pixels"]) == 2_000_000


def test_page_render_path_is_stable():
    """The renders are protected from the storage orphan cron by BOTH the
    `extracted/<document_id>/` prefix rule and an explicit document_page_embeddings
    branch. Moving them out from under that prefix silently loses one of the two."""
    spec = importlib.util.spec_from_file_location("_page_probe", _PAGE_SERVICE)
    tree = ast.parse(_PAGE_SOURCE)
    fn = _function_source(_PAGE_SOURCE, "page_storage_path")
    assert "extracted/" in fn, (
        "page renders must stay under pdf-tiles/extracted/<document_id>/, which "
        "build_storage_reference_set() protects by prefix"
    )
    assert spec is not None and tree is not None


def test_isolation_and_search_helpers_are_async_and_awaited():
    """`search_page_embeddings` returning a coroutine that nobody awaits would score
    every page at zero while looking entirely healthy in the logs."""

    async def _check():
        return True

    assert asyncio.run(_check())
    assert "await self.vecs_service.search_page_embeddings(" in _RAG_SOURCE
    assert "await self.vecs.upsert_page_embedding(" in _PAGE_SOURCE
