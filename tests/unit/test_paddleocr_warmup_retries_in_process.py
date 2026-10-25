"""A wedged first prediction reloads the pipeline in-process; only exhausting every attempt crashes the container."""

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "modal_app/paddleocr_vl.py"


def _load_method():
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "load")


def test_every_attempt_but_the_last_reloads_a_fresh_pipeline():
    load = _load_method()
    loops = [n for n in load.body if isinstance(n, ast.For)]
    assert len(loops) == 1
    reloads = [
        n for n in ast.walk(loops[0])
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "self.pipeline" for t in n.targets)
        and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "PaddleOCRVL"
    ]
    assert reloads, "a wedged attempt must reload the pipeline before trying again"


def test_the_container_only_crashes_after_the_attempts_are_used_up():
    load = _load_method()
    loop = next(n for n in load.body if isinstance(n, ast.For))
    inside = [n for n in ast.walk(loop) if isinstance(n, ast.Raise)]
    assert not inside, "raising inside the loop crashes the container on the first wedge (Modal then backs off ~10 min)"
    after = [n for n in load.body[load.body.index(loop) + 1:] if isinstance(n, ast.Raise)]
    assert after, "when every attempt wedges the container must still recycle"


def test_a_fast_failure_or_a_success_leaves_the_container_up():
    load = _load_method()
    loop = next(n for n in load.body if isinstance(n, ast.For))
    returns = [n for n in ast.walk(loop) if isinstance(n, ast.Return)]
    assert len(returns) == 2
