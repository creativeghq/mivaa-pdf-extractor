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


def test_kb_rechunk_embeds_before_it_deletes_the_existing_chunks():
    tree = ast.parse((_APP / "services/kb/kb_chunk_service.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "rechunk_doc")
    embed_line = next(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute) and n.func.attr == "generate_contextualized_embeddings")
    empty_branch = {n.lineno for i in ast.walk(fn) if isinstance(i, ast.If) and ast.unparse(i.test) == "not chunks"
                    for n in ast.walk(i) if hasattr(n, "lineno")}
    deletes = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and n.func.attr == "delete"]
    assert deletes
    for line in deletes:
        assert line in empty_branch or line > embed_line, (
            f"kb_doc_chunks deleted on line {line} before the embed on line {embed_line}: "
            "a failed embed would leave the document with no searchable chunks"
        )
