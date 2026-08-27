"""Behaviour probe for `app/utils/killable.py` — run as a script, not collected.

WHY THIS IS A SEPARATE ENTRYPOINT INSTEAD OF PLAIN TEST FUNCTIONS
----------------------------------------------------------------
`killable` uses the `spawn` start method, and `spawn` re-imports `__main__` in the child.
Under `python -m pytest`, `__main__` IS pytest — so the child re-runs the whole test
session, fails, and exits 1. Every spawn from inside a pytest process dies that way.

That is a property of multiprocessing, not a defect in the module: it is the reason
`if __name__ == "__main__":` is mandatory in the multiprocessing docs. Production is
unaffected because the service runs under the `uvicorn` console script, whose `__main__`
is guarded (checked in the module docstring).

So the probe gets its own guarded `__main__` and the test module runs it as a subprocess.
The alternative — switching the module to `fork` so the tests are easier to write — would
have meant choosing the start method for the convenience of the test rather than the
safety of the thing being tested, and `fork` inherits the parent's threads and locks.

Filename starts with `_` so pytest's `python_files = test_*.py` never collects it.

Stdlib only, deliberately: MIVAA's own CI installs pytest and nothing else.
"""

import operator
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load():
    """Imported by its REAL name, not loaded by path under a synthetic one.

    `spawn` pickles the child target by qualified name, so the child must be able to
    import the module that defines it. A path-loaded module lives only in the parent's
    `sys.modules`, and the child dies with `ModuleNotFoundError` — which is exactly what
    the first version of this probe did, while faithfully reporting it as a crashed
    worker.

    `app.utils` is stdlib-only at import time, so this works in MIVAA's CI too.
    """
    sys.path.insert(0, str(ROOT))
    import app.utils.killable as module

    return module


def main() -> int:
    k = _load()
    failures = 0

    def report(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"{'PASS' if ok else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")

    # 1. Work that ignores the deadline is KILLED, not merely abandoned. `time.sleep`
    #    is the honest stand-in for a PDF that will not finish: it polls no flag and
    #    checks for no cancellation, so nothing short of killing it will stop it.
    started = time.monotonic()
    try:
        k.run_killable(time.sleep, 600, timeout=3, label="sleeper")
        report("timeout_kills", False, "no timeout raised")
    except k.KillableTimeout:
        elapsed = time.monotonic() - started
        report("timeout_kills", elapsed < 30, f"took {elapsed:.1f}s for a 3s deadline")

    # 2. The ordinary path still returns.
    report("result_passthrough", k.run_killable(operator.add, 2, 3, timeout=60) == 5)

    # 3. A large payload does not deadlock. The classic multiprocessing trap: a child
    #    putting a big object blocks until the parent reads it, so a parent that joins
    #    BEFORE draining hangs — on exactly the big documents this path exists for.
    big = k.run_killable(operator.mul, "x", 8 * 1024 * 1024, timeout=60)
    report("large_payload", len(big) == 8 * 1024 * 1024)

    # 4. The worker's own exception reaches the caller with its type intact.
    try:
        k.run_killable(operator.truediv, 1, 0, timeout=60, label="divider")
        report("child_exception", False, "no error raised")
    except RuntimeError as e:
        report("child_exception", "ZeroDivisionError" in str(e), str(e)[:80])

    # 5. A child that DIES is reported at once and not as a timeout. A single blocking
    #    `queue.get(timeout=...)` would wait out the whole deadline and then mislabel
    #    the cause, so an OOM kill at two seconds becomes a "timeout" five minutes later
    #    and the document is retried as merely slow. `os._exit` is the faithful
    #    simulation: the process vanishes without unwinding, like SIGKILL.
    started = time.monotonic()
    try:
        # 30s, not 300s: the gap being proven is 0.5s vs "the whole deadline", and 30
        # is decisive without costing CI half a minute on every green run — a regression
        # here is the only thing that pays the full wait.
        k.run_killable(os._exit, 137, timeout=30, label="oom")
        report("crash_detected", False, "crash not detected")
    except k.KillableCrashed as e:
        elapsed = time.monotonic() - started
        report(
            "crash_detected",
            elapsed < 10 and "137" in str(e),
            f"took {elapsed:.1f}s against a 30s deadline; exitcode reported="
            f"{'137' in str(e)}",
        )

    # 6. Unpicklable arguments are refused BEFORE a process is started — and not
    #    silently run in a thread instead, which would restore the exact defect this
    #    module removes on the day somebody adds a new option.
    try:
        k.run_killable(operator.add, lambda x: x, 1, timeout=60)
        report("unpicklable_refused", False, "accepted an unpicklable argument")
    except TypeError as e:
        report("unpicklable_refused", "not picklable" in str(e))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
