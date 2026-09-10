"""Unified endpoint controller for the inference endpoints."""

import asyncio
import logging
from typing import Any, Dict, Optional

from app.services.core.adaptive_concurrency import AdaptiveConcurrency

logger = logging.getLogger(__name__)


# Short endpoint keys → app names. Both inference endpoints are Modal-hosted
# (SLIG visual embeddings + PaddleOCR-VL structural pass) as of 2026-06-14 —
# Modal owns autoscaling, so these names are just for logging + gate lookup.
ENDPOINT_NAMES: Dict[str, str] = {
    "slig": "slig",
    "paddleocr": "paddleocr-vl",
}

VALID_ENDPOINT_KEYS = frozenset(ENDPOINT_NAMES.keys())


class EndpointController:
    """Unified controller — lifecycle + backpressure + scaler cooperation.

    Singleton. Instantiate once per process; every module imports the same
    `endpoint_controller` instance from the bottom of this file.
    """

    def __init__(self):
        # Per-endpoint AdaptiveConcurrency gates.
        self.slig  = AdaptiveConcurrency(name="slig",  initial=8, minimum=2, maximum=16, failure_threshold=3, success_threshold=15)
        self.paddleocr = AdaptiveConcurrency(name="paddleocr", initial=4, minimum=1, maximum=8,  failure_threshold=2, success_threshold=10)

        # Track warm state per endpoint so warm_all is idempotent.
        self._warmed: Dict[str, bool] = {k: False for k in VALID_ENDPOINT_KEYS}
        self._warm_lock = asyncio.Lock()

        # Audit fix #19: active-job tracking for scale-to-zero coordination.
        # Each register_job_start increments; register_job_done decrements.
        # scale_all_to_zero queries DB-side for current 'processing' jobs as
        # the source of truth (in-memory count would be lost across worker
        # restarts), but in-memory is the fast path.
        import threading
        self._active_jobs: set = set()
        self._active_jobs_lock = threading.Lock()

    def register_job_start(self, job_id: str) -> None:
        """Register a job as active. Suppresses scale-to-zero from other jobs."""
        try:
            with self._active_jobs_lock:
                self._active_jobs.add(job_id)
        except Exception:
            pass

    def register_job_done(self, job_id: str) -> None:
        """Unregister a job. The next scale-to-zero call may now proceed."""
        try:
            with self._active_jobs_lock:
                self._active_jobs.discard(job_id)
        except Exception:
            pass

    def _get_active_job_count(self) -> int:
        """Return the count of OTHER active jobs.

        Sum of in-memory tracking + DB-side count of 'processing' background_jobs
        with fresh heartbeat (<2 min). DB query is best-effort; if it fails we
        fall back to in-memory only. In-memory may undercount across worker
        restarts but never overcounts.
        """
        with self._active_jobs_lock:
            in_mem = len(self._active_jobs)
        try:
            from app.services.core.supabase_client import get_supabase_client
            sb = get_supabase_client()
            from datetime import datetime, timedelta
            cutoff = (datetime.utcnow() - timedelta(minutes=2)).isoformat()
            resp = sb.client.table("background_jobs")\
                .select("id", count="exact")\
                .eq("status", "processing")\
                .gte("last_heartbeat", cutoff).execute()
            db_count = int(resp.count or 0)
            # Subtract one if we're called from inside the only active job.
            return max(in_mem, db_count - in_mem)
        except Exception:
            return in_mem

    # ────────────────────────────────────────────────────────────────────
    # Accessors
    # ────────────────────────────────────────────────────────────────────

    def get_gate(self, endpoint: str) -> AdaptiveConcurrency:
        """Look up the concurrency gate by short name ('slig'/'paddleocr')."""
        if endpoint not in VALID_ENDPOINT_KEYS:
            raise ValueError(
                f"Unknown endpoint key {endpoint!r}. Valid keys: {sorted(VALID_ENDPOINT_KEYS)}"
            )
        return getattr(self, endpoint)

    def record_success(self, endpoint: str) -> None:
        """Signal a successful call. Grows the gate toward its maximum."""
        self.get_gate(endpoint).record_success()

    def record_failure(self, endpoint: str) -> None:
        """Signal a backpressure-relevant failure. Shrinks the gate.

        Only call for overload-class errors (timeout, 503, connection error,
        rate limit). DO NOT call for semantic errors (400, invalid payload,
        empty response) — those are not capacity signals.
        """
        self.get_gate(endpoint).record_failure()

    def record_overload_exception(self, endpoint: str, exc: BaseException) -> bool:
        """Classify an exception and record failure if it's overload-class.

        Returns True if the exception was treated as a backpressure signal.
        Useful in `except` blocks where you want one-line error classification:

            except Exception as e:
                endpoint_controller.record_overload_exception("paddleocr", e)
                raise
        """
        name = type(exc).__name__
        is_overload = (
            "Timeout" in name
            or "Connection" in name
            or "RateLimit" in name
            or "ReadError" in name
            or "RemoteProtocol" in name
        )
        # Check for HTTP status codes on errors that carry a response
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None) if resp is not None else None
        if status is not None and status in (429, 500, 502, 503, 504):
            is_overload = True

        if is_overload:
            self.record_failure(endpoint)
        return is_overload

    # ────────────────────────────────────────────────────────────────────
    # Warmup — parallel, at job start
    # ────────────────────────────────────────────────────────────────────

    async def warm_all(self, job_id: str) -> Dict[str, bool]:
        """Warm up the structural endpoints (SLIG + PaddleOCR) in parallel.

        Args:
            job_id: for logging correlation.

        Returns:
            Dict[endpoint_name, success_bool] — quick at-a-glance result.
        """
        async with self._warm_lock:
            from app.services.embeddings.endpoint_registry import endpoint_registry

            warm_specs = [
                ("slig",  endpoint_registry.get_slig_manager()),
                ("paddleocr", endpoint_registry.get_paddleocr_manager()),
            ]
            results = await asyncio.gather(
                *[self._warm_one(key, mgr, job_id) for key, mgr in warm_specs],
                return_exceptions=False,
            )

            # Pair results back to their keys by position (VALID_ENDPOINT_KEYS is
            # an unordered frozenset — zipping it with the ordered results list
            # would mislabel outcomes).
            outcome = {key: res for (key, _), res in zip(warm_specs, results)}
            healthy = [k for k, v in outcome.items() if v]
            degraded = [k for k, v in outcome.items() if not v]

            logger.info(
                "🔥 EndpointController.warm_all(%s): healthy=%s degraded=%s",
                job_id, healthy, degraded,
            )

            return outcome

    async def _warm_one(self, key: str, manager: Any, job_id: str) -> bool:
        """Resume + warmup a single endpoint."""
        gate = self.get_gate(key)

        if manager is None:
            logger.warning(
                "   ⚠️ %s manager not registered — gate forced to minimum (fallback-only)",
                key,
            )
            gate.force_minimum()
            return False

        # Fast path: a prior warmup() succeeded — but the flag is in-memory and
        # outlives the underlying endpoint state. Re-probe before trusting:
        # HF could have scaled to zero between jobs, the container could be
        # wedged, or the process could have restarted with a stale flag. The
        # 2026-05-01 audit fix was specifically about closing this regression.
        if getattr(manager, 'warmup_completed', False):
            try:
                probe_ok = await asyncio.to_thread(manager._test_inference)
            except Exception as probe_err:
                logger.debug("   %s _test_inference raised %s — re-warming", key, probe_err)
                probe_ok = False
            if probe_ok:
                self._warmed[key] = True
                logger.debug("   ↪️  %s already warmed up (probe OK) — skipping re-warmup", key)
                return True
            logger.warning(
                "   ⚠️ %s warmup_completed=True but live probe failed — re-warming",
                key,
            )
            try:
                setattr(manager, 'warmup_completed', False)
            except Exception:
                pass
            self._warmed[key] = False
            # fall through to the status peek + warmup logic below

        # Pre-check: ask HF for the endpoint's actual status. If it's not
        # running (paused, failed, scaledToZero, deploying), don't burn the
        # 360s warmup polling loop — we know it won't respond.
        try:
            def _peek_status() -> Optional[str]:
                ep = manager._get_endpoint() if hasattr(manager, '_get_endpoint') else None
                if ep is None:
                    return None
                try:
                    ep.fetch()
                    return getattr(ep, 'status', None)
                except Exception:
                    return None
            status = await asyncio.to_thread(_peek_status)
            if status is not None and status != 'running':
                logger.warning(
                    "   ⚠️ %s status=%s (not running) — skipping re-probe, gate forced to minimum",
                    key, status,
                )
                gate.force_minimum()
                return False
        except Exception as peek_err:
            logger.debug("Status peek for %s failed (continuing): %s", key, peek_err)

        try:
            # warmup() is sync, uses requests. Wrap in a thread so parallel
            # warmup of 4 endpoints actually runs in parallel.
            ok: bool = await asyncio.to_thread(manager.warmup)
            if ok:
                self._warmed[key] = True
                logger.info("   ✅ %s warmed up", key)
                return True
            else:
                logger.warning(
                    "   ⚠️ %s warmup returned False — gate forced to minimum",
                    key,
                )
                gate.force_minimum()
                return False
        except Exception as e:
            logger.warning(
                "   ⚠️ %s warmup raised %s: %s — gate forced to minimum",
                key, type(e).__name__, e,
            )
            gate.force_minimum()
            return False

    async def prepare_for_processing(
        self,
        reason: str = "job_start",
        min_replica: int = 1,
        max_replica: int = 4,
        scale_to_zero_timeout: int = 30,
    ) -> Dict[str, bool]:
        """Pre-warm posture for an active job. No-op now: both endpoints (SLIG +
        PaddleOCR-VL) are Modal-hosted and Modal owns autoscaling — there is no
        per-job min_replica bump to make. Kept (with its signature) as the first
        half of the per-job state machine so callers don't change:
            prepare_for_processing → ... job runs ... → scale_all_to_zero

        Returns ``{key: True}`` for each endpoint (host-managed = ready).
        """
        logger.debug(
            "prepare_for_processing(reason=%s): Modal-managed autoscaling — no-op",
            reason,
        )
        return {k: True for k in ENDPOINT_NAMES}

    async def scale_all_to_zero(self, reason: str = "cleanup", force: bool = False) -> Dict[str, bool]:
        """Force every HF endpoint to min_replica=0, with concurrent-job guard.

        Args:
            reason: short tag for log correlation, e.g. "pdf_job_<id>",
                    "agent_<id>", "admin_cancel", "periodic_idle".
            force: if True, bypass active-job-count guard.

        Returns:
            Dict[endpoint_key, success_bool] — same shape as warm_all().
        """
        # Active-job guard.
        if not force:
            active_count = self._get_active_job_count()
            if active_count > 0:
                logger.info(
                    "⏸ scale_all_to_zero(reason=%s): SKIPPED — %d other job(s) still running. "
                    "Use force=True to override.",
                    reason, active_count
                )
                return {k: False for k in ENDPOINT_NAMES}

        from app.services.embeddings.endpoint_registry import endpoint_registry

        manager_getters = {
            "slig":  endpoint_registry.get_slig_manager,
            "paddleocr": endpoint_registry.get_paddleocr_manager,
        }

        outcome: Dict[str, bool] = {}
        for key in ENDPOINT_NAMES:
            scaled = False
            mgr = None
            try:
                mgr = manager_getters[key]()
            except Exception:
                mgr = None

            # Modal: manager.scale_to_zero() is a no-op that returns True (Modal
            # drains the GPU container on its own scaledown_window). We just reset
            # our warm flags so the next job re-probes /health.
            if mgr is not None and hasattr(mgr, "scale_to_zero"):
                try:
                    scaled = bool(await asyncio.to_thread(mgr.scale_to_zero))
                    if scaled:
                        logger.info("   📉 %s scale-to-zero (Modal idle clock, reason=%s)", key, reason)
                except Exception as e:
                    logger.warning("   ⚠️ %s scale_to_zero failed: %s", key, e)

            outcome[key] = scaled
            self._warmed[key] = False
            if mgr is not None and hasattr(mgr, "warmup_completed"):
                try:
                    mgr.warmup_completed = False
                except Exception:
                    pass

        succeeded = [k for k, v in outcome.items() if v]
        failed    = [k for k, v in outcome.items() if not v]
        logger.info(
            "📉 EndpointController.scale_all_to_zero(reason=%s): scaled=%s failed=%s",
            reason, succeeded, failed,
        )
        return outcome

    async def request_scale_up(self, endpoint: str, desired_replicas: int) -> bool:
        """No-op since 2026-06-14: both endpoints are Modal-hosted and Modal
        autoscales on demand (up to ``max_containers``) — there is no manual
        replica request to make. Kept so any legacy caller is harmless. The
        adaptive concurrency gate still handles in-run overload."""
        return False

    # ────────────────────────────────────────────────────────────────────
    # Observability
    # ────────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Single snapshot of the gates for progress reports + logs."""
        return {
            "slig":   self.slig.stats(),
            "paddleocr":  self.paddleocr.stats(),
            "warmed": dict(self._warmed),
        }

    def log_stats(self, prefix: str = "🎛️  EndpointController") -> None:
        """Pretty-print gate state at stage boundaries."""
        s = self.stats()
        logger.info(
            "%s: slig=%d/%d (in=%d) | paddleocr=%d/%d (in=%d)",
            prefix,
            s["slig"]["limit"], s["slig"]["max"], s["slig"]["in_flight"],
            s["paddleocr"]["limit"], s["paddleocr"]["max"], s["paddleocr"]["in_flight"],
        )


# ════════════════════════════════════════════════════════════════════════
# Module-level singleton. Import this everywhere:
#
#     from app.services.core.endpoint_controller import endpoint_controller
# ════════════════════════════════════════════════════════════════════════

endpoint_controller = EndpointController()
