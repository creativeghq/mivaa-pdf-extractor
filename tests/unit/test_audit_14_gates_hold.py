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
