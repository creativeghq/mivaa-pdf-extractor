"""Run one function in a subprocess that can actually be killed.

WHAT IT REPLACES
----------------
`asyncio.wait_for(loop.run_in_executor(pool, fn, ...), timeout=...)` (#22 M9-5).

`wait_for` cancels the *coroutine* waiting on the future. It does not touch the thread,
because Python has no mechanism to interrupt one. So a malformed PDF that is declared
timed out keeps consuming CPU and memory for as long as it likes; enough of them exhaust
the worker pool, and the job is marked timed out while the work that caused it is still
running — which also means the recovery machinery is reasoning about a job whose work has
not stopped.

A process can be killed. That is the whole difference, and it is why this is a subprocess
rather than a nicer wrapper around the thread pool.

WHY spawn
---------
`fork` inherits the parent's threads, locks and open handles — including a half-held
logging lock or a Supabase client's connection pool — into a child that never runs their
owners. That produces hangs that look exactly like the timeouts this module exists to
stop. `spawn` costs a fresh interpreter and a re-import (seconds), which is noise next to
a PDF extraction and cheap next to debugging a deadlock.

`spawn` re-imports `__main__` in the child, so it is only safe if the entrypoint is
import-safe. Checked: production runs `uvicorn app.main:app` (systemd ExecStart), so
`__main__` is uvicorn's console script and `app/main.py` is imported as `app.main` — its
`if __name__ == "__main__": main()` guard never fires in the child. If the service is
ever started as `python app/main.py`, this stops being true and the child will try to
boot a second server.

WHY IT LIVES IN app/utils AND NOT NEXT TO ITS CALLER
----------------------------------------------------
`spawn` pickles the child target BY QUALIFIED NAME, so the child must be able to
`import` the module that defines it. That rules out loading this file by path under a
synthetic module name — the child raises `ModuleNotFoundError` for a name that exists
only in the parent's `sys.modules`.

That also rules out every package in this app whose `__init__` pulls a third-party
dependency, because MIVAA's own CI installs pytest and NOTHING ELSE:

  * `app/services/__init__.py` imports the Supabase client.
  * `app/utils/__init__.py` imports `.logging`, which does `from app.config import
    get_settings` — and `app.config` needs `pydantic_settings`.

`app/__init__.py` contains a docstring and two strings. No imports at all. So this
module sits at the top of the package, which is odd placement for a utility and is the
price of the child being able to import it with nothing installed.

This was got wrong once, in the way that matters: `app.utils` was checked LOCALLY, where
every dependency is present, and it imported fine. CI has no dependencies, every spawned
child died with `ModuleNotFoundError: pydantic_settings`, and the probe reported it
faithfully as a crashed worker. `test_a_timeout_actually_stops_the_work` now walks this
module's whole import chain and fails if anything third-party enters it, so the next
person does not need to remember.

WHY THE PAYLOAD IS PICKLE-CHECKED UP FRONT
------------------------------------------
The thread pool was chosen originally "to avoid pickle issues with pydantic Settings and
other non-picklable objects", and the caller already strips the three known offenders. If
a fourth appears, this raises a clear error at the boundary rather than deep inside
multiprocessing — and it does NOT fall back to a thread. A silent fallback would restore
the exact defect this module removes, on the day somebody adds a new option, with nothing
to show for it.
"""

from __future__ import annotations

import logging
import multiprocessing
import pickle
import queue as queue_mod
import time
from typing import Any, Callable, Tuple

logger = logging.getLogger(__name__)

#: How long to let a terminated child clean up before escalating to SIGKILL.
TERMINATE_GRACE_SECONDS = 5.0

#: Queue poll slice. Sets how quickly a DEAD child is noticed, not how long the work
#: may run — the deadline is still the caller's `timeout`.
_POLL_SECONDS = 0.5


class KillableTimeout(TimeoutError):
    """The work exceeded its deadline and the process running it was killed.

    A TimeoutError subclass so existing `except asyncio.TimeoutError` / `except
    TimeoutError` handlers keep working, but distinguishable for callers that want to
    record that the work really did stop.
    """


class KillableCrashed(RuntimeError):
    """The child exited without producing a result or an exception.

    This case did not previously exist as a distinguishable outcome: a thread cannot be
    OOM-killed on its own, so an out-of-memory PDF took the whole worker down. In a
    subprocess it is visible, reportable and retryable, and the exit code says which
    signal did it.
    """


def _child(fn: Callable[..., Any], args: Tuple[Any, ...], out: Any) -> None:  # pragma: no cover
    """Runs in the subprocess. Puts exactly one (ok, payload) tuple."""
    try:
        out.put((True, fn(*args)))
    except BaseException as exc:  # noqa: BLE001 - the parent re-raises a faithful copy
        # The exception itself may not be picklable (some carry sockets or file handles),
        # so send a description rather than risk the child dying while reporting a death.
        out.put((False, f"{type(exc).__name__}: {exc}"))


def run_killable(
    fn: Callable[..., Any],
    *args: Any,
    timeout: float,
    label: str = "work",
) -> Any:
    """Run `fn(*args)` in a subprocess; kill it if it exceeds `timeout`.

    Returns whatever `fn` returned. Raises `KillableTimeout` if it ran too long (and the
    process is dead by the time this raises), `KillableCrashed` if the child died without
    answering, or a `RuntimeError` carrying the child's own exception text.

    Synchronous by design. The caller is an async function and wraps this in
    `run_in_executor` — that thread now blocks on a process it can kill, instead of being
    the thing that cannot be stopped.
    """
    try:
        pickle.dumps(args)
    except Exception as e:
        raise TypeError(
            f"{label}: arguments are not picklable, so they cannot cross a process "
            f"boundary ({e}). Strip the offending value at the call site — do NOT fall "
            "back to a thread, which is what made a timeout unenforceable."
        ) from e

    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_child, args=(fn, args, out), daemon=True)
    proc.start()

    try:
        # Drain the queue BEFORE joining. A child putting a large payload blocks until
        # the parent reads it, so joining first deadlocks on exactly the big documents
        # this path exists for.
        #
        # Polled rather than one `out.get(timeout=timeout)`, so a child that DIES is
        # noticed when it dies. A single blocking get waits out the full deadline and
        # then reports a timeout — so an out-of-memory kill at two seconds would be
        # reported five minutes later, under the wrong name, and retried as though the
        # document were merely slow.
        deadline = time.monotonic() + timeout
        result = None
        while True:
            try:
                result = out.get(timeout=min(_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
                break
            except queue_mod.Empty:
                if not proc.is_alive():
                    # Dead, and it left nothing behind. Give the queue one last look:
                    # the child may have put its answer and exited in the same breath.
                    try:
                        result = out.get_nowait()
                        break
                    except queue_mod.Empty:
                        raise KillableCrashed(
                            f"{label} worker exited with code {proc.exitcode} without "
                            "producing a result — killed by the OS (out of memory) or "
                            "crashed in a C extension"
                        ) from None
                if time.monotonic() >= deadline:
                    logger.error(
                        "%s exceeded %.0fs — terminating the process that is still "
                        "running it", label, timeout,
                    )
                    _kill(proc, label)
                    raise KillableTimeout(
                        f"{label} timed out after {timeout:.0f}s; the process running "
                        "it was terminated"
                    ) from None

        ok, payload = result

        proc.join(timeout=TERMINATE_GRACE_SECONDS)
        if proc.is_alive():
            # It answered and then lingered. Not an error for the caller — the result is
            # in hand — but it must not be left behind.
            logger.warning("%s returned but its process is still alive; killing", label)
            _kill(proc, label)

        if not ok:
            raise RuntimeError(f"{label} failed in its worker process: {payload}")
        return payload

    finally:
        if proc.is_alive():  # pragma: no cover - belt and braces on an early raise
            _kill(proc, label)
        out.close()
        out.join_thread()


def _kill(proc: Any, label: str) -> None:
    """SIGTERM, then SIGKILL. Nothing is left running behind a timeout."""
    proc.terminate()
    proc.join(timeout=TERMINATE_GRACE_SECONDS)
    if proc.is_alive():
        logger.error("%s did not stop on terminate; sending kill", label)
        proc.kill()
        proc.join(timeout=TERMINATE_GRACE_SECONDS)
    if proc.exitcode is not None and proc.exitcode < 0:
        logger.info("%s worker stopped by signal %d", label, -proc.exitcode)
