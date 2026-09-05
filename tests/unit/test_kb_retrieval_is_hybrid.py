"""KB retrieval runs two channels, ships their counts, and is measured.

Until 2026-09-05 the corpus the agent searches (kb_doc_chunks, ~9.8k sections) had a
vector column and nothing else: no lexical index, so a question carrying an exact term
(a framework name, a model code, a Greek word the embedding handles poorly) depended
entirely on cosine similarity. The fix is one RPC, `kb_hybrid_doc_chunks`, that runs a
vector channel and an English/Greek tsvector channel over ONE gated candidate set and
fuses them by rank (RRF). These tests pin the three things that would silently undo it:

  * the kb_docs branch going back to `kb_match_doc_chunks` (vector-only) or dropping
    `query_text` — the lexical channel is off and nothing errors;
  * the per-channel counts leaving `search_metadata` — a dead lexical channel then looks
    exactly like a healthy one;
  * the eval runner losing its gate or building its query vector some other way than
    `kb_query_vector` (input_type=query is decided there, once).

Source-scanning, stdlib only: MIVAA CI installs pytest and nothing from `app`.
"""
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"
RAG_ROUTES = APP / "api" / "rag_routes.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _kb_docs_branch(src: str) -> str:
    start = src.index('if "kb_docs" in request.search_types:')
    end = src.index('branch_status["kb_docs"] = ', start)
    return src[start:end]


def test_the_kb_docs_branch_runs_both_channels():
    branch = _kb_docs_branch(_read(RAG_ROUTES))
    assert 'rpc("kb_hybrid_doc_chunks"' in branch, (
        "the kb_docs branch no longer calls kb_hybrid_doc_chunks — KB search is vector-only again"
    )
    assert '"query_text": request.query' in branch, (
        "the raw question is not passed as query_text — with NULL the lexical channel is off "
        "and the RPC degrades to vector-only without an error"
    )
    assert 'rpc("kb_match_doc_chunks"' not in branch, (
        "kb_match_doc_chunks is the vector-only wrapper; the agent path must call the hybrid function"
    )


def test_channel_counts_ship_with_the_response():
    src = _read(RAG_ROUTES)
    assert '"kb_channels": kb_channel_stats' in src, (
        "search_metadata.kb_channels is gone — a lexical channel that never returns a rank "
        "is then indistinguishable from one that does not exist"
    )
    branch = _kb_docs_branch(src)
    for key in ('"vector_only"', '"lexical_only"', '"both"'):
        assert key in branch, f"kb_channel_stats no longer counts {key}"


def test_the_eval_runner_is_internal_only_and_embeds_the_one_way():
    src = _read(RAG_ROUTES)
    i = src.index('@router.post("/kb-eval/run")')
    body = src[i: src.index("\n@router.", i + 10)]
    assert "_require_internal_kb_caller(http_request" in body, (
        "the KB eval runner lost its gate — /api/rag is outside the JWT middleware, so this "
        "check IS the gate"
    )
    assert "kb_query_vector(" in body, (
        "the eval runner must embed through kb_query_vector: it is the only place "
        "input_type=query is decided, and a document-mode vector scores plausibly and wrong"
    )
    assert '"kb_retrieval_eval_score"' in body, "the eval runner must score in SQL, not in Python"
    assert "resolve_workspace_id(" not in body, (
        "the runner is cron-secret gated; resolve_workspace_id would 401 the cron path"
    )


def test_the_rechunk_route_shares_the_gate():
    src = _read(RAG_ROUTES)
    i = src.index('@router.post("/kb-docs/rechunk")')
    body = src[i: src.index("\n@router.", i + 10)]
    assert "_require_internal_kb_caller(http_request" in body, (
        "rechunk and the eval runner must share one gate definition"
    )
    assert src.count("def _require_internal_kb_caller(") == 1
