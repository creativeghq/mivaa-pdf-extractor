"""Adaptive concurrency controller — AIMD backpressure.

Usage:

    slig_concurrency = AdaptiveConcurrency(initial=3, minimum=1, maximum=8)

    async with slig_concurrency.slot():
        try:
            result = await slig_call(...)
            slig_concurrency.record_success()
        except (APITimeoutError, APIConnectionError, RateLimitError):
            slig_concurrency.record_failure()
            raise
"""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class AdaptiveConcurrency:
    """Backpressure controller: `_in_flight` against a resizable `_limit`.

    Shared by every job, and each job runs its own event loop in its own thread, so the
    gate is a thread lock plus a short async poll. Any asyncio primitive here is bound to
    the first loop that waits on it and raises in every other job.
    """

    _POLL_MIN_SECONDS = 0.005
    _POLL_MAX_SECONDS = 0.1

    def __init__(
        self,
        initial: int = 3,
        minimum: int = 1,
        maximum: int = 8,
        failure_threshold: int = 2,
        success_threshold: int = 10,
        name: str = "endpoint",
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
        self._guard = threading.Lock()

    def _try_take(self) -> bool:
        with self._guard:
            if self._in_flight < self._limit:
                self._in_flight += 1
                return True
            return False

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
        delay = self._POLL_MIN_SECONDS
        while not self._try_take():
            await asyncio.sleep(delay)
            delay = min(delay * 2, self._POLL_MAX_SECONDS)
        try:
            yield
        finally:
            with self._guard:
                self._in_flight -= 1

    def record_success(self) -> None:
        """Signal that an in-flight call succeeded.

        After `success_threshold` consecutive successes we grow the limit by 1
        (additive increase), capped at `maximum`.
        """
        with self._guard:
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

    def record_failure(self) -> None:
        """Signal that an in-flight call failed with a backpressure-relevant error."""
        with self._guard:
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
        with self._guard:
            if self._limit != self._min:
                old = self._limit
                self._limit = self._min
                logger.warning(
                    "🛑 %s adaptive concurrency forced to minimum: %d → %d "
                    "(endpoint dead/unconfigured)",
                    self._name, old, self._limit,
                )

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
