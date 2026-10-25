"""A running job pings the PaddleOCR Modal app so its container is not scaled down mid-job."""

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "app/services/tracking/job_heartbeat.py"


def _tree():
    return ast.parse(_SRC.read_text(encoding="utf-8"))


def _method(name):
    return next(n for n in ast.walk(_tree()) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_the_loop_pings_inside_the_heartbeat_thread_and_only_while_the_job_runs():
    loop = ast.unparse(_method("_thread_loop"))
    assert "self._keep_models_warm()" in loop
    assert loop.index("self._is_job_terminal()") < loop.index("self._keep_models_warm()")


def test_the_ping_is_spaced_wider_than_the_heartbeat_and_inside_the_container_idle_window():
    consts = {
        t.id: n.value.value for n in _tree().body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
        for t in n.targets if isinstance(t, ast.Name)
    }
    assert 15 <= consts["MODEL_KEEPALIVE_SECONDS"] <= 60


def test_a_failed_ping_never_reaches_the_job():
    body = _method("_keep_models_warm")
    handlers = [h for h in ast.walk(body) if isinstance(h, ast.ExceptHandler)]
    assert handlers and all(not any(isinstance(n, ast.Raise) for n in ast.walk(h)) for h in handlers)
