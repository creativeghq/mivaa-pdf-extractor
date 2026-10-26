"""An unhealthy Modal endpoint at job start gets patient retries before the job is failed."""

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "app/api/rag_routes.py"


def _source():
    return _SRC.read_text(encoding="utf-8")


def test_the_retry_budget_covers_a_modal_crash_loop_back_off():
    tree = ast.parse(_source())
    consts = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
              for t in n.targets if isinstance(t, ast.Name)}
    assert consts["WARMUP_RETRY_ATTEMPTS"] >= 2


def test_the_retry_runs_after_the_first_probe_and_before_the_job_is_failed():
    src = _source()
    probe = src.index("health_results[name] = ok")
    retry = src.index("warm-up retry")
    verdict = src.index("all_healthy = all(")
    assert probe < retry < verdict


def test_a_retry_that_raises_marks_the_endpoint_unhealthy_not_the_job_crashed():
    src = _source()
    block = src[src.index("warm-up retry"):src.index("all_healthy = all(")]
    assert "except Exception as retry_err" in block and "ok = False" in block
