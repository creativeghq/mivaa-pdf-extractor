"""Shared async locks must work across the per-job event loops (#494 first real ingest).

Each background job runs `asyncio.new_event_loop()` in its own thread; an asyncio primitive
on a process-wide object raised "bound to a different event loop" and failed three of six
concurrent ingests.
"""
import asyncio
import re
import sys
import threading
from pathlib import Path

from app.utils.cross_loop_lock import CrossLoopLock, CrossLoopSemaphore

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"

#: Instances created per job, never shared between loops.
_PER_JOB = {"services/tracking/progress_tracker.py": "one ProgressTracker per job"}


def _contend(primitive, workers: int, hold: float):
    peak, live, guard, errors = [0], [0], threading.Lock(), []

    async def job():
        async with primitive:
            with guard:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            await asyncio.sleep(hold)
            with guard:
                live[0] -= 1

    def run():
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(job())
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            loop.close()

    threads = [threading.Thread(target=run) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    return peak[0], errors


def test_lock_excludes_across_event_loops():
    peak, errors = _contend(CrossLoopLock(poll_seconds=0.001), workers=4, hold=0.02)
    assert errors == [] and peak == 1


def test_semaphore_caps_across_event_loops():
    peak, errors = _contend(CrossLoopSemaphore(2, poll_seconds=0.001), workers=5, hold=0.02)
    assert errors == [] and peak == 2


def test_no_shared_object_holds_an_asyncio_primitive():
    sys.path.insert(0, str(_ROOT / "scripts"))
    try:
        from comment_budget import blank_comments
    finally:
        sys.path.pop(0)
    # An attribute (self./cls.) or a module-level name; function locals live and die in one loop.
    pattern = re.compile(
        r"^(?:\s*(?:self|cls)\.\w+|[A-Za-z_]\w*)\s*(?::[^=\n]+)?=\s*"
        r"asyncio\.(?:Lock|Semaphore|BoundedSemaphore|Condition|Event)\(",
        re.M,
    )
    offenders = []
    for path in _APP.rglob("*.py"):
        rel = path.relative_to(_APP).as_posix()
        if rel in _PER_JOB:
            continue
        code = blank_comments(path.read_text(encoding="utf-8"))
        for match in pattern.finditer(code):
            offenders.append(f"{rel}:{code[:match.start()].count(chr(10)) + 1}")
    assert not offenders, (
        "asyncio primitive on an object or module that outlives one event loop; use "
        f"app.utils.cross_loop_lock instead: {offenders}"
    )
