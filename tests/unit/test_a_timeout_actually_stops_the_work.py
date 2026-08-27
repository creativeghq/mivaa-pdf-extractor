"""A timeout must stop the work, not just stop waiting for it (mivaa#22 M9-5).

`asyncio.wait_for(loop.run_in_executor(pool, fn, ...), timeout=...)` cancels the
COROUTINE waiting on the future. It cannot touch the thread, because Python has no
mechanism to interrupt one. So a malformed PDF declared timed out kept consuming CPU and
memory for as long as it liked; enough of them exhausted the worker pool; and the job was
marked timed out while the work that caused it was still running — which is also why the
recovery machinery could not reason about it.

A process can be killed. That is the whole difference.

THESE ARE BEHAVIOUR TESTS, NOT SOURCE ASSERTIONS
------------------------------------------------
Most guards in this suite read source, because MIVAA's own CI installs pytest and nothing
else. `killable.py` is stdlib-only, so it can actually be RUN — and "the process is
killed" is a claim worth proving rather than pattern-matching.

They run through `_killable_probe.py` as a SUBPROCESS rather than as plain test
functions, and the reason is worth knowing before anyone tries to simplify it: `spawn`
re-imports `__main__` in the child, and under `python -m pytest` that IS pytest, so every
spawn from inside a pytest process re-runs the session and exits 1. That is
multiprocessing behaving as documented — it is why `if __name__ == "__main__":` is
mandatory there — and production is unaffected because the service runs under the
guarded `uvicorn` console script.

Switching the module to `fork` would have made these plain test functions. It would also
have meant choosing the start method for the convenience of the test rather than the
safety of the thing under test, and `fork` inherits the parent's threads and locks.

WATCHED TO FAIL: the source cases were run against the pre-fix tree and fired. The
behaviour cases were run against a deliberately broken copy of the module — `_kill`
reduced to a bare `proc.terminate()` with no escalation, and the poll loop replaced by a
single blocking `get` — and caught both.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
PROCESSOR = APP / "services" / "pdf" / "pdf_processor.py"


def _strip_comments(src: str) -> str:
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def _body(name: str) -> str:
    src = PROCESSOR.read_text(encoding="utf-8")
    node = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    return _strip_comments(ast.get_source_segment(src, node) or "")


# -------------------------------------------------------------------------
# Behaviour — the part that could not be pattern-matched
# -------------------------------------------------------------------------

PROBE = Path(__file__).parent / "_killable_probe.py"

#: The behaviours the probe proves, in the order it prints them.
PROBE_CASES = [
    "timeout_kills",
    "result_passthrough",
    "large_payload",
    "child_exception",
    "crash_detected",
    "unpicklable_refused",
]


def _run_probe():
    """`killable` uses `spawn`, and `spawn` re-imports `__main__` in the child. Under
    `python -m pytest` that IS pytest, so every spawn from inside a pytest process
    re-runs the session and exits 1.

    That is multiprocessing behaving as documented, not a defect — it is why
    `if __name__ == "__main__":` is mandatory there — and production is unaffected
    because the service runs under the guarded `uvicorn` console script. The probe
    therefore has its own guarded entrypoint and is run as a subprocess.

    Switching the module to `fork` so these could be plain test functions would have
    meant choosing the start method for the convenience of the test rather than the
    safety of the thing under test.
    """
    proc = subprocess.run(
        [sys.executable, str(PROBE)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(ROOT),
    )
    return proc


@pytest.fixture(scope="module")
def probe_result():
    return _run_probe()


@pytest.mark.parametrize("case", PROBE_CASES)
def test_killable_behaviour(case: str, probe_result):
    """One test per proven behaviour, so a failure names which property broke rather
    than reporting that 'the probe failed'."""
    line = next(
        (ln for ln in probe_result.stdout.splitlines() if ln.split(" ", 1)[-1].startswith(case)),
        None,
    )
    assert line is not None, (
        f"the probe did not report {case!r}.\n"
        f"stdout:\n{probe_result.stdout}\n"
        f"stderr:\n{probe_result.stderr[-2000:]}"
    )
    assert line.startswith("PASS"), line


def test_the_probe_exercised_every_case():
    """A probe that silently stopped reporting would let the parametrized cases above
    pass on absence. This pins the roster."""
    src = PROBE.read_text(encoding="utf-8")
    for case in PROBE_CASES:
        assert f'"{case}"' in src, f"{case} is no longer exercised by the probe"


# -------------------------------------------------------------------------
# Source — the wiring
# -------------------------------------------------------------------------

def test_the_extraction_path_no_longer_wraps_an_executor_in_wait_for():
    body = _body("_process_pdf_file")
    assert "asyncio.wait_for(" not in body or "run_in_executor" not in body.split("asyncio.wait_for(")[1][:300], (
        "markdown extraction is back to `wait_for` around `run_in_executor` (#22 M9-5) "
        "— that cancels the coroutine and leaves the thread running"
    )
    assert "run_killable" in body, "the killable runner is gone"


def test_the_timeout_and_the_crash_are_handled_separately():
    body = _body("_process_pdf_file")
    assert "KillableTimeout" in body and "KillableCrashed" in body, (
        "a killed-on-deadline worker and a worker that died on its own are being "
        "collapsed into one outcome again"
    )


def test_the_runner_escalates_to_kill():
    """SIGTERM alone is a request. A C extension spinning inside PyMuPDF may not honour
    it, which is precisely the case this exists for."""
    src = _strip_comments((APP / "utils" / "killable.py").read_text(encoding="utf-8"))
    assert ".terminate()" in src and ".kill()" in src, (
        "the escalation from terminate to kill is gone — a worker that ignores SIGTERM "
        "survives the timeout again"
    )


def test_the_runner_uses_spawn():
    """`fork` inherits the parent's threads and locks — including a half-held logging
    lock — into a child that never runs their owners, producing hangs indistinguishable
    from the timeouts this module exists to stop."""
    src = _strip_comments((APP / "utils" / "killable.py").read_text(encoding="utf-8"))
    assert 'get_context("spawn")' in src


def test_there_is_no_thread_fallback():
    src = _strip_comments((APP / "utils" / "killable.py").read_text(encoding="utf-8"))
    assert "ThreadPoolExecutor" not in src, (
        "a thread fallback has appeared in the killable runner — it would restore the "
        "unenforceable timeout silently, which is worse than not having the module"
    )
