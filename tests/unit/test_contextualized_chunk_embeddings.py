"""
Contextualized chunk embeddings: windows stay under the model's budget, every vector
lands on its own chunk, and a misaligned reply is refused rather than zipped.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_APP = Path(__file__).resolve().parents[2] / "app"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _APP / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc = _load("contextual_chunks_under_test", "services/embeddings/contextual_chunks.py")


def _reply(request, drop_last_of_first=False):
    data = [
        {"index": d, "data": [{"index": c, "embedding": [float(i)] + [0.0] * 1023}
                              for c, i in enumerate(w)]}
        for d, w in enumerate(request)
    ]
    if drop_last_of_first:
        data[0]["data"] = data[0]["data"][:-1]
    return {"data": data}


def test_a_long_document_is_split_into_windows_under_budget():
    texts = cc.prepare(["x" * (3000 + i) for i in range(40)])
    requests = cc.plan_requests(texts)
    windows = [w for r in requests for w in r]
    assert len(windows) > 1
    assert [i for w in windows for i in w] == list(range(40)), "windows must keep chunk order"
    for w in windows:
        assert sum(cc.estimate_tokens(texts[i]) for i in w) <= cc.WINDOW_TOKENS
    for r in requests:
        assert sum(cc.estimate_tokens(texts[i]) for w in r for i in w) <= cc.REQUEST_TOKENS


def test_every_vector_lands_on_its_own_chunk():
    texts = cc.prepare(["y" * 5000 for _ in range(12)])
    for req in cc.plan_requests(texts):
        for i, emb in cc.assign_vectors(req, _reply(req)):
            assert emb[0] == float(i)


def test_a_short_window_reply_is_refused():
    req = cc.plan_requests(cc.prepare(["a", "b", "c"]))[0]
    with pytest.raises(ValueError):
        cc.assign_vectors(req, _reply(req, drop_last_of_first=True))


def test_a_wrong_width_vector_is_refused():
    req = cc.plan_requests(cc.prepare(["a"]))[0]
    with pytest.raises(ValueError):
        cc.assign_vectors(req, {"data": [{"index": 0, "data": [{"index": 0, "embedding": [0.1] * 512}]}]})


def test_empty_and_oversized_chunks_are_made_sendable():
    out = cc.prepare(["", None, "z" * 100_000])
    assert out[0] == out[1] == "no text content"
    assert len(out[2]) == cc.MAX_CHUNK_CHARS


def test_the_chunk_model_is_priced():
    pricing = _load("ai_pricing_under_test", "config/ai_pricing.py")
    cost = pricing.AIPricingConfig.calculate_cost(
        model="voyage-context-4", input_tokens=1_000_000, output_tokens=0, include_markup=False,
    )
    assert float(cost["raw_cost_usd"]) == pytest.approx(0.12)


def _calls_in(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}


@pytest.mark.parametrize("rel", ["services/kb/kb_chunk_service.py", "services/search/rag_service.py"])
def test_chunk_writers_use_the_contextualized_model(rel):
    assert "generate_contextualized_embeddings" in _calls_in(_APP / rel)


def test_the_chunk_model_default_is_voyage_context_4():
    tree = ast.parse((_APP / "config.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "voyage_chunk_model":
            default = next(kw.value.value for kw in node.value.keywords if kw.arg == "default")
            assert default == "voyage-context-4"
            return
    raise AssertionError("config.py declares no voyage_chunk_model")


def _fn(rel, name):
    tree = ast.parse((_APP / rel).read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def _call_lines(node, attr):
    return [n.lineno for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr]


def test_kb_rechunk_swaps_chunks_only_through_the_atomic_rpc_after_embedding():
    fn = _fn("services/kb/kb_chunk_service.py", "rechunk_doc")
    assert not _call_lines(fn, "delete") and not _call_lines(fn, "insert"), (
        "kb_doc_chunks must be replaced by replace_kb_doc_chunks in one transaction, "
        "never by a delete plus separate inserts"
    )
    embed_line = _call_lines(fn, "generate_contextualized_embeddings")[0]
    empty_branch = {n.lineno for i in ast.walk(fn) if isinstance(i, ast.If) and ast.unparse(i.test) == "not chunks"
                    for n in ast.walk(i) if hasattr(n, "lineno")}
    rpc_lines = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "rpc"
                 and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "replace_kb_doc_chunks"]
    assert rpc_lines
    for line in rpc_lines:
        assert line in empty_branch or line > embed_line, (
            f"chunks replaced on line {line} before the embed on line {embed_line}: "
            "a failed embed would leave the document with no searchable chunks"
        )


def test_the_source_hash_matches_the_sql_function():
    mod = _load("chunk_source_hash_under_test", "services/kb/chunk_source_hash.py")
    # Values computed live by public.kb_doc_chunk_source_hash on 2026-10-17.
    assert mod.source_hash("Πλακάκια 60×120", "Γρανίτης R10 — matte") == "497382c3ed25803120ea5d87b8873467"
    assert mod.source_hash(None, None) == "68b329da9893e34099c7d8ad5cb9c940"


def test_pdf_chunks_are_embedded_once_per_document_not_per_batch():
    fn = _fn("services/search/rag_service.py", "index_pdf_content")
    loops = [n for n in ast.walk(fn) if isinstance(n, (ast.For, ast.AsyncFor))]
    in_loop = {line for loop in loops for line in _call_lines(loop, "generate_contextualized_embeddings")}
    calls = _call_lines(fn, "generate_contextualized_embeddings")
    assert len(calls) == 1 and not in_loop, (
        "the whole document goes to voyage-context-4 in one call; a call inside the batch "
        "loop gives each chunk only its batch as context"
    )
    assert _call_lines(fn, "_generate_text_embedding"), (
        "a chunk the contextualized call could not embed must still get a vector of its own"
    )


def test_transient_errors_are_retried():
    assert cc.is_retryable(429) and cc.is_retryable(503)
    assert not cc.is_retryable(400) and not cc.is_retryable(401)
    assert cc.MAX_ATTEMPTS >= 2
    assert cc.backoff_seconds(1) > cc.backoff_seconds(0) > 0


def test_the_chunk_model_setting_refuses_a_model_outside_the_query_space():
    tree = ast.parse((_APP / "config.py").read_text(encoding="utf-8"))
    validator = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and any(
            isinstance(d, ast.Call) and getattr(d.func, "id", "") == "field_validator"
            and any(isinstance(a, ast.Constant) and a.value == "voyage_chunk_model" for a in d.args)
            for d in n.decorator_list)
    )
    allowed = next(n.value for n in ast.walk(validator)
                   if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "allowed")
    assert {e.value for e in allowed.elts} == {"voyage-context-4"}
    assert any(isinstance(n, ast.Raise) for n in ast.walk(validator))
