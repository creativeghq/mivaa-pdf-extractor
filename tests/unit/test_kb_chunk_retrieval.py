"""
Guards for PDF-chunk retrieval in /search/knowledge-base (issue #318).

Three things here are load-bearing and none of them is visible to a typecheck:

1. **The route is served by a duplicate-defining module pair.** `/search/knowledge-base`
   is defined TWICE — in `app/api/rag_routes.py` and in `app/api/documents/query_routes.py`
   — and Starlette serves whichever router was included FIRST. The fixed implementation
   lives in rag_routes; reorder the `include_router` calls in main.py and the endpoint
   silently reverts to the shadowed copy, with no import error and no failing request.

2. **Retrieval must be a vector search.** The original chunks branch pulled an unordered
   `select * limit top_k*3` sample of the workspace and scored it with `+0.15 per query
   word found as a substring` against a threshold defaulting to 0.7. That returns
   plausible-looking rows, so nothing downstream can tell it apart from real retrieval.

3. **A hit must carry its address.** Without `document_id` + `chunk_index` + `product_id`
   a retrieved chunk cannot be read outward from, and `read_document_section` 404s
   against the wrong corpus. `chunk_index` restarts at 0 per product inside a document,
   so the product id is part of the address, not decoration.

AST/source based on purpose: it imports neither `app` nor a DB, so it runs in CI in
about a second — the difference between a guard that runs on every push and one that
quietly never runs.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_RAG_ROUTES = _ROOT / "app" / "api" / "rag_routes.py"
_QUERY_ROUTES = _ROOT / "app" / "api" / "documents" / "query_routes.py"
_MAIN = _ROOT / "app" / "main.py"

_RAG_SOURCE = _RAG_ROUTES.read_text(encoding="utf-8")
_RAG_TREE = ast.parse(_RAG_SOURCE)


def _function_source(tree: ast.AST, source: str, name: str) -> str:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{name}() not found — it was renamed or removed")


def _included_routers_in_order() -> list:
    """Names passed to app.include_router(...), in registration order."""
    tree = ast.parse(_MAIN.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "include_router"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            names.append(node.args[0].id)
    return names


def test_knowledge_base_route_is_still_defined_twice():
    """The premise of the ordering guard below. If someone deletes the shadowed copy
    this test fails — delete the ordering guard with it, don't weaken it."""
    assert '"/search/knowledge-base"' in _RAG_SOURCE
    assert '"/search/knowledge-base"' in _QUERY_ROUTES.read_text(encoding="utf-8")


def test_rag_router_wins_the_duplicate_knowledge_base_route():
    """First-registered wins in Starlette. rag_router holds the vector-search
    implementation; query_router holds the shadowed substring-scoring copy."""
    order = _included_routers_in_order()
    assert "rag_router" in order, "rag_router is no longer registered in main.py"
    assert "query_router" in order, "query_router is no longer registered in main.py"
    assert order.index("rag_router") < order.index("query_router"), (
        "query_router now shadows rag_router, so /search/knowledge-base is served by "
        "the old substring-scoring implementation in query_routes.py"
    )


def test_chunk_search_uses_vector_similarity():
    src = _function_source(_RAG_TREE, _RAG_SOURCE, "search_knowledge_base")
    assert "search_document_chunks_by_embedding" in src, (
        "the chunks branch no longer calls the vector RPC"
    )
    # The exact shape of the scorer that was removed. `score += 0.15` per matched word
    # against a 0.7 threshold meant a hit needed 5+ words to survive an arbitrary sample.
    assert "score += 0.15" not in src
    assert ".table('document_chunks')" not in src, (
        "chunk retrieval is back to a raw table scan instead of the RPC"
    )


def test_chunk_hits_carry_their_read_outward_address():
    src = _function_source(_RAG_TREE, _RAG_SOURCE, "search_knowledge_base")
    for field in ('"chunk_index"', '"product_id"', '"page_number"', '"source"'):
        assert field in src, f"chunk results dropped {field} — read-outward breaks"
    assert '"content": chunk.get(\'content\', \'\')[:500]' not in src, (
        "chunk content is truncated again; a 500-char cut is the very mid-section "
        "truncation this endpoint is supposed to stop producing"
    )


def test_chunk_retrieval_does_not_reuse_the_word_count_threshold():
    """`request.similarity_threshold` defaults to 0.7 because it was tuned for the
    word-count scorer (+0.15 per matched word). Feeding that to a COSINE search is a
    scale error, not a stricter setting: it rejects effectively every real hit and the
    endpoint then looks like an empty corpus. The kb_docs branch already hardcodes 0.4
    for the same measured reason."""
    src = _function_source(_RAG_TREE, _RAG_SOURCE, "search_knowledge_base")
    chunk_branch = src.split('if "chunks" in request.search_types:')[1].split(
        'if "kb_docs" in request.search_types:'
    )[0]
    # Comments stripped: the constant's own docstring names the rejected knob.
    code_only = "\n".join(
        line for line in chunk_branch.splitlines() if not line.strip().startswith("#")
    )
    assert "TEXT_SEARCH_SIMILARITY_FLOOR" in code_only
    assert "request.similarity_threshold" not in code_only, (
        "the retired scorer's threshold is being used as a cosine floor again"
    )


def test_entity_search_actually_searches():
    """Both halves of entity search were no-ops that reported success. The consumer
    called `vecs_service.search_similar` — a method that does not exist, on a name that
    was never bound — so every call raised NameError into a swallowed warning and
    `results["entities"]` was always []."""
    src = _function_source(_RAG_TREE, _RAG_SOURCE, "search_knowledge_base")
    assert "search_document_entities_by_embedding" in src
    # Comments stripped: the branch documents the call it replaced by name.
    code_only = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#")
    )
    assert "vecs_service" not in code_only, (
        "the entity branch is calling vecs_service again — it is not bound in this "
        "function and VecsService has no search_similar method"
    )


def test_entity_embeddings_are_persisted_not_just_counted():
    """The producer generated a Voyage vector per entity, counted it, logged a green
    tick and dropped it — its docstring pointed at an 'embeddings table' that does not
    exist in this database. Paid-for vectors, an empty index, a success report."""
    service = _ROOT / "app" / "services" / "discovery" / "document_entity_service.py"
    source = service.read_text(encoding="utf-8")
    src = _function_source(ast.parse(source), source, "generate_entity_embeddings")
    assert "text_embedding" in src and ".update(" in src, (
        "generate_entity_embeddings no longer writes the vector it generates"
    )
    # The write must precede the counter, or the count is again a claim about memory.
    assert src.index(".update(") < src.index("embeddings_generated += 1"), (
        "the counter is incremented before the row is written — exactly the shape "
        "that reported success while the search index stayed empty"
    )


def test_chunk_expansion_is_bounded_and_reported():
    src = _function_source(_RAG_TREE, _RAG_SOURCE, "search_knowledge_base")
    assert "expand_document_chunk_hits" in src, "structure expansion no longer fires"
    assert "EXPANDED_CHUNK_CHAR_BUDGET" in src, (
        "expansion lost its per-hit budget and can now hand the LLM a whole catalog"
    )
    # Expansion that silently never fires is the platform's dominant failure shape,
    # so the counters must reach the caller, not just the log.
    assert "chunk_expansion" in src


def test_read_section_serves_both_corpora():
    src = _function_source(_RAG_TREE, _RAG_SOURCE, "read_document_section")
    assert "kb_read_doc_section" in src, "the kb corpus lost its reader"
    assert "read_document_chunk_span" in src, "the PDF corpus lost its reader"
    # product_id scopes the chunk_index namespace; dropping it reads the wrong chunks.
    assert "p_product_id" in src
