"""
Adaptive concurrency controller — AIMD backpressure.

When an upstream endpoint (Qwen HF Inference, SLIG, etc.) cannot scale up its
replicas (no available GPU, or rate-limited), we scale DOWN our in-flight
request rate instead of queuing requests that will time out. Classic AIMD:

    Additive Increase: +1 concurrency slot after N consecutive successes
    Multiplicative Decrease: /2 concurrency slots after M consecutive failures

This is purely in-process — no Redis, no DB state. Each Stage-3 run starts
fresh at `initial`. The controller is reusable for any rate-limited endpoint.

Usage:

    qwen_concurrency = AdaptiveConcurrency(initial=3, minimum=1, maximum=8)

    async with qwen_concurrency.slot():
        try:
            result = await qwen_call(...)
            qwen_concurrency.record_success()
        except (APITimeoutError, APIConnectionError, RateLimitError):
            qwen_concurrency.record_failure()
            raise
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)


class AdaptiveConcurrency:
    """Backpressure controller using a condition-variable gate.

    We do not use `asyncio.Semaphore` because its counter is not runtime-resizable.
    Instead we track `_in_flight` against `_limit` and wake waiters on release
    or limit-change.
    """

    def __init__(
        self,
        initial: int = 3,
        minimum: int = 1,
        maximum: int = 8,
        failure_threshold: int = 2,
        success_threshold: int = 10,
        name: str = "qwen",
    ):
        if not (minimum >= 1 and initial >= minimum and maximum >= initial):
            raise ValueError(
                f"AdaptiveConcurrency: require 1 <= minimum ({minimum}) <= "
                f"initial ({initial}) <= maximum ({maximum})"
            )

        self._limit = initial
        self._min = minimum
        self._max = maximum
        self._failure_threshold = failure_threshold
        self._success_threshold = success_threshold
        self._name = name

        self._in_flight = 0
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        # Lazy-init the Condition on first async use.
        #
        # Why: in Python 3.9, `asyncio.Condition()` binds to the event loop
        # returned by `events.get_event_loop()` at construction time. When the
        # module-level `endpoint_controller` singleton is built at *import*
        # time (before uvicorn has started any request loop), the Condition
        # latches to either a throwaway loop or no loop at all. Once uvicorn
        # creates its per-request loop and code calls `slot()`, the Condition's
        # internal Futures belong to the wrong loop and raise:
        #
        #     RuntimeError: got Future <Future pending> attached to a different loop
        #
        # That's MIVAA-56Z — 42 events in 35 seconds during the Stage 3 batch.
        # Deferring creation until the first coroutine actually enters `slot()`
        # guarantees the Condition binds to the running request loop.
        self._cond: Optional[asyncio.Condition] = None

    def _get_cond(self) -> asyncio.Condition:
        """Return the Condition, creating it lazily inside the caller's
        running event loop on first use.
        """
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    @property
    def limit(self) -> int:
        """Current concurrency limit (for logging/observability)."""
        return self._limit

    @property
    def in_flight(self) -> int:
        """Current in-flight count."""
        return self._in_flight

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Acquire a concurrency slot for the duration of the `async with` block.

        Blocks (cooperatively) until `in_flight < limit`. Releases on exit even
        if the protected code raises.
        """
        cond = self._get_cond()
        async with cond:
            while self._in_flight >= self._limit:
                await cond.wait()
            self._in_flight += 1
        try:
            yield
        finally:
            async with cond:
                self._in_flight -= 1
                cond.notify_all()

    def record_success(self) -> None:
        """Signal that an in-flight call succeeded.

        After `success_threshold` consecutive successes we grow the limit by 1
        (additive increase), capped at `maximum`.
        """
        # This method is sync so that callers inside `try/except` blocks don't
        # need to await. Internal condition interaction is fire-and-forget.
        self._consecutive_successes += 1
        self._consecutive_failures = 0

        if (
            self._consecutive_successes >= self._success_threshold
            and self._limit < self._max
        ):
            old = self._limit
            self._limit += 1
            self._consecutive_successes = 0
            logger.info(
                "🐰 %s adaptive concurrency: %d → %d (after %d consecutive successes)",
                self._name, old, self._limit, self._success_threshold,
            )
            # Wake one waiter so the new slot is used immediately.
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._notify_one())
            except RuntimeError:
                pass  # No running loop — limit change still took effect.

    def record_failure(self) -> None:
        """Signal that an in-flight call failed with a backpressure-relevant error.

        Only call this for errors that indicate upstream overload:
            - APITimeoutError
            - RateLimitError (429)
            - Service Unavailable (503)
            - ConnectionError

        DO NOT call for semantic errors (400 Bad Request, empty JSON response,
        invalid image format, etc.) — those are not concurrency-related.

        After `failure_threshold` consecutive failures we halve the limit
        (multiplicative decrease), floored at `minimum`.
        """
        self._consecutive_failures += 1
        self._consecutive_successes = 0

        if (
            self._consecutive_failures >= self._failure_threshold
            and self._limit > self._min
        ):
            old = self._limit
            self._limit = max(self._min, self._limit // 2)
            self._consecutive_failures = 0
            logger.warning(
                "🐢 %s adaptive concurrency: %d → %d "
                "(after %d consecutive failures — endpoint overloaded)",
                self._name, old, self._limit, self._failure_threshold,
            )
            # Note: we do NOT forcibly interrupt in-flight callers. They finish
            # naturally; new slots will simply be unavailable until in_flight
            # drops below the new limit.

    def force_minimum(self) -> None:
        """Immediately shrink the concurrency limit to `minimum`.

        Called when the upstream endpoint is known-dead (warmup failed, health
        check failed, URL not configured). Prevents the controller from handing
        out slots that would just pile up against a broken endpoint.
        """
        if self._limit != self._min:
            old = self._limit
            self._limit = self._min
            logger.warning(
                "🛑 %s adaptive concurrency forced to minimum: %d → %d "
                "(endpoint dead/unconfigured)",
                self._name, old, self._limit,
            )

    async def _notify_one(self) -> None:
        """Wake one waiter after a limit increase."""
        cond = self._get_cond()
        async with cond:
            cond.notify(1)

    def stats(self) -> dict:
        """Snapshot of current state for logging / progress reports."""
        return {
            "name": self._name,
            "limit": self._limit,
            "in_flight": self._in_flight,
            "min": self._min,
            "max": self._max,
            "consecutive_failures": self._consecutive_failures,
            "consecutive_successes": self._consecutive_successes,
        }
