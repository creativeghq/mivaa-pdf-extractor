"""An image's aspect texts go to Voyage in ONE request; the per-aspect call is only the fallback."""

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "app/services/embeddings/real_embeddings_service.py"


def _method():
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    return next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_generate_specialized_aspect_embeddings"
    )


def _calls(fn, attr):
    return [c for c in ast.walk(fn) if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == attr]


def test_the_aspects_are_batched_before_any_single_call():
    fn = _method()
    batch = _calls(fn, "generate_batch_embeddings")
    single = _calls(fn, "_generate_text_embedding")
    assert len(batch) == 1 and len(single) == 1
    assert batch[0].lineno < single[0].lineno


def test_the_single_call_sits_inside_the_loop_as_a_fallback_for_a_missing_vector():
    fn = _method()
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For) and _calls(n, "_generate_text_embedding")]
    assert loops
    source = ast.unparse(loops[0])
    assert "batch_vec or await self._generate_text_embedding" in source


def test_a_batch_of_the_wrong_length_is_discarded_not_zipped():
    source = ast.unparse(_method())
    assert "len(batched) != len(present)" in source
