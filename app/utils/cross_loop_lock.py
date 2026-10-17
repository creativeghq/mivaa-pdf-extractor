"""Async lock and semaphore that are safe to share between event loops.

Every background job runs its own event loop in its own thread (`run_async_in_background`),
so an asyncio primitive on a process-wide singleton binds to the first loop that contends for
it and raises "bound to a different event loop" in every other job. These keep a thread-safe
counter and poll asynchronously, so a waiter never blocks its own loop.
"""

import asyncio
import threading


class CrossLoopSemaphore:
    def __init__(self, value: int = 1, poll_seconds: float = 0.01) -> None:
        if value < 1:
            raise ValueError("value must be >= 1")
        self._free = value
        self._guard = threading.Lock()
        self._poll = poll_seconds

    def _try_acquire(self) -> bool:
        with self._guard:
            if self._free > 0:
                self._free -= 1
                return True
            return False

    async def acquire(self) -> None:
        while not self._try_acquire():
            await asyncio.sleep(self._poll)

    def release(self) -> None:
        with self._guard:
            self._free += 1

    def locked(self) -> bool:
        with self._guard:
            return self._free == 0

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *exc) -> None:
        self.release()


class CrossLoopLock(CrossLoopSemaphore):
    def __init__(self, poll_seconds: float = 0.01) -> None:
        super().__init__(1, poll_seconds)
