"""
Unified endpoint controller for all HuggingFace Inference Endpoints.

This is the single coordination point for:
  - lifecycle     (resume / warmup / pause of the 4 HF endpoints)
  - backpressure  (AdaptiveConcurrency gate per endpoint, AIMD)
  - supply/demand closed loop with EndpointAutoScaler
  - observability (one stats() call for everything)

Why:
  We have 4 HF endpoints (Qwen, SLIG, YOLO, Chandra). Each used to have its
  own scattered warmup path, its own (or no) concurrency gate, and its own
  call-site pattern. When one endpoint got overloaded, nothing shrank our
  in-flight request rate; we just piled requests into a saturated queue and
  timed them out.

  This controller:
  - Warms all 4 endpoints in parallel at job start (saves 60-180s vs serial).
  - Gives each endpoint its own AdaptiveConcurrency slot, tuned to its shape.
  - Per-endpoint failures shrink that endpoint only — a Qwen meltdown does
    NOT drag SLIG/YOLO/Chandra down with it.
  - If HF scales replicas up (via EndpointAutoScaler), the controller raises
    the concurrency cap to match. If HF cannot scale up, the controller
    AIMD-shrinks our in-flight rate to match whatever capacity exists.

Call-site pattern (same for all 4 endpoints):

    from app.services.core.endpoint_controller import endpoint_controller

    async with endpoint_controller.qwen.slot():
        try:
            result = await qwen_call(...)
            endpoint_controller.record_success("qwen")
        except (APITimeoutError, APIConnectionError, httpx.TimeoutException):
            endpoint_controller.record_failure("qwen")
            raise

That's the entire integration — one import, one `async with`, two signals.
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

from app.services.core.adaptive_concurrency import AdaptiveConcurrency

logger = logging.getLogger(__name__)


# Mapping between our short names and the HuggingFace endpoint names used
# by the auto-scaler. Resolved from `Settings.*_endpoint_name` so the user
# can rename endpoints on HF without editing code (e.g. the 2026-04-29
# rename of `mh-qwen332binstruct` → `qwen3-6-35b-fp8` and
# `mh-chandra` → `chandra-ocr-2`).
def _resolve_hf_endpoint_names() -> Dict[str, str]:
    try:
        from app.config import get_settings
        s = get_settings()
        return {
            "qwen":    s.qwen_endpoint_name,
            "slig":    s.slig_endpoint_name,
            "yolo":    s.yolo_endpoint_name,
            "chandra": s.chandra_endpoint_name,
        }
    except Exception:
        # Fallback to the historical defaults so import-time access still
        # works in test contexts that don't configure Settings.
        return {
            "qwen":    "qwen3-6-35b-fp8",
            "slig":    "mh-slig",
            "yolo":    "mh-yolo",
            "chandra": "chandra-ocr-2",
        }


HF_ENDPOINT_NAMES: Dict[str, str] = _resolve_hf_endpoint_names()

VALID_ENDPOINT_KEYS = frozenset(HF_ENDPOINT_NAMES.keys())


class EndpointController:
    """Unified controller — lifecycle + backpressure + scaler cooperation.

    Singleton. Instantiate once per process; every module imports the same
    `endpoint_controller` instance from the bottom of this file.
    """

    def __init__(self):
        # Per-endpoint AdaptiveConcurrency gates.
        #
        # Tuning rationale:
        #   - Qwen (mh-qwen332binstruct): heavy VLM on a single GPU replica by
        #     default. 4 concurrent is the honest ceiling for one replica;
        #     8 is achievable only with 2+ replicas.
        #   - SLIG (mh-slig): lightweight text-guided embeddings, fast responses.
        #     16 concurrent is fine; failures here usually mean network, not load.
        #   - YOLO (mh-yolo): layout detection, ~2-5s per page, single-page calls.
        #     12 concurrent is fine; HF replica typically handles 2-3 RPS.
        #   - Chandra (mh-chandra): OCR, highly variable latency depending on
        #     image complexity. 8 concurrent max; 1 minimum. Most runs, it's
        #     disabled — the controller force_minimum()s if the manager is None.
        self.qwen    = AdaptiveConcurrency(name="qwen",    initial=4, minimum=1, maximum=8,  failure_threshold=2, success_threshold=10)
        self.slig    = AdaptiveConcurrency(name="slig",    initial=8, minimum=2, maximum=16, failure_threshold=3, success_threshold=15)
        self.yolo    = AdaptiveConcurrency(name="yolo",    initial=6, minimum=2, maximum=12, failure_threshold=3, success_threshold=15)
        self.chandra = AdaptiveConcurrency(name="chandra", initial=4, minimum=1, maximum=8,  failure_threshold=2, success_threshold=10)

        # Track warm state per endpoint so warm_all is idempotent.
        self._warmed: Dict[str, bool] = {k: False for k in VALID_ENDPOINT_KEYS}
        self._warm_lock = asyncio.Lock()

    # ────────────────────────────────────────────────────────────────────
    # Accessors
    # ────────────────────────────────────────────────────────────────────

    def get_gate(self, endpoint: str) -> AdaptiveConcurrency:
        """Look up the concurrency gate by short name ('qwen'/'slig'/...)."""
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
                endpoint_controller.record_overload_exception("chandra", e)
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
        """Warm up all 4 HF endpoints in parallel.

        Each endpoint manager's `warmup()` method is sync (it uses `requests`,
        not httpx); we run them in threads so they can overlap without blocking
        the event loop.

        Any endpoint whose warmup fails (manager missing, URL empty, all resume
        attempts failed, warmup probe timeout) has its concurrency gate forced
        to `minimum=1`. That prevents the pipeline from handing out slots that
        would just pile up against a broken endpoint — calls will either
        succeed slowly or fail fast into their application-level fallback
        (e.g. Claude for Qwen, EasyOCR for Chandra).

        Args:
            job_id: for logging correlation.

        Returns:
            Dict[endpoint_name, success_bool] — quick at-a-glance result.
        """
        async with self._warm_lock:
            from app.services.embeddings.endpoint_registry import endpoint_registry

            # Serialize access to the auto-scaler's status so we can cooperate.
            auto_scaler = self._get_auto_scaler()

            tasks = [
                self._warm_one("qwen",    endpoint_registry.get_qwen_manager(),    job_id),
                self._warm_one("slig",    endpoint_registry.get_slig_manager(),    job_id),
                self._warm_one("yolo",    endpoint_registry.get_yolo_manager(),    job_id),
                self._warm_one("chandra", endpoint_registry.get_chandra_manager(), job_id),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)

            outcome = dict(zip(VALID_ENDPOINT_KEYS, results))
            healthy = [k for k, v in outcome.items() if v]
            degraded = [k for k, v in outcome.items() if not v]

            logger.info(
                "🔥 EndpointController.warm_all(%s): healthy=%s degraded=%s",
                job_id, healthy, degraded,
            )

            # Cooperate with the auto-scaler: if it has a fresh replica count
            # from the HF API, use it to raise concurrency caps above default.
            if auto_scaler is not None:
                self._align_caps_with_replica_counts(auto_scaler)

            return outcome

    async def _warm_one(self, key: str, manager: Any, job_id: str) -> bool:
        """Resume + warmup a single endpoint.

        Any failure force_minimum()s the corresponding gate.

        Optimization: rag_routes already ran parallel warmup before calling
        warm_all (so warmup_completed=True for the successful ones). For
        managers whose prior warmup FAILED, calling manager.warmup() again
        re-runs the full 360s polling loop unnecessarily — we already know
        the endpoint is broken. Short-circuit by checking the endpoint
        status BEFORE entering the polling loop.
        """
        gate = self.get_gate(key)

        if manager is None:
            logger.warning(
                "   ⚠️ %s manager not registered — gate forced to minimum (fallback-only)",
                key,
            )
            gate.force_minimum()
            return False

        # Fast path: a prior warmup() already succeeded → no need to re-probe.
        if getattr(manager, 'warmup_completed', False):
            self._warmed[key] = True
            logger.debug("   ↪️  %s already warmed up — skipping re-probe", key)
            return True

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

    # ────────────────────────────────────────────────────────────────────
    # Auto-scaler cooperation
    # ────────────────────────────────────────────────────────────────────

    def _get_auto_scaler(self) -> Optional[Any]:
        """Safely fetch the global auto-scaler if initialized."""
        try:
            from app.services.embeddings.endpoint_auto_scaler import get_auto_scaler
            return get_auto_scaler()
        except Exception:
            return None

    def _align_caps_with_replica_counts(self, auto_scaler: Any) -> None:
        """Raise per-gate `maximum` to match actual HF replica counts.

        If the auto-scaler has already scaled Qwen to 2 replicas, we can
        safely run at 2 × base_max concurrent (8 → 16). This is the "supply
        side" of the control loop — when HF grants more capacity, we let our
        demand-side gate use it. AIMD still handles in-run overload; this is
        just the ceiling.
        """
        try:
            status = auto_scaler.get_status()
            current = status.get("current_replicas", {}) or {}

            # Map HF endpoint names → our short keys via HF_ENDPOINT_NAMES
            for short_key, hf_name in HF_ENDPOINT_NAMES.items():
                replicas = current.get(hf_name, 0)
                if replicas <= 0:
                    continue

                gate = self.get_gate(short_key)
                base_max = gate._max
                # Scale cap by replica count, but never above 4× the base
                # (defensive ceiling so a misreported replica count can't
                # let us DDoS an endpoint).
                new_max = min(base_max * replicas, base_max * 4)
                if new_max > gate._max:
                    logger.info(
                        "📈 %s cap raised: %d → %d (HF has %d replica(s))",
                        short_key, gate._max, new_max, replicas,
                    )
                    gate._max = new_max
        except Exception as e:
            # Cooperation is best-effort; never fail the pipeline on scaler noise.
            logger.debug("Auto-scaler alignment skipped: %s", e)

    async def prepare_for_processing(
        self,
        reason: str = "job_start",
        min_replica: int = 1,
        max_replica: int = 4,
        scale_to_zero_timeout: int = 30,
    ) -> Dict[str, bool]:
        """Pre-warm posture for an active job: ensure every HF endpoint has
        `min_replica >= 1` (so we always have one warm replica) with a
        `max_replica` ceiling for burst, and `scaleToZeroTimeout` set so
        that the moment we later call `scale_all_to_zero`, HF's idle clock
        starts ticking properly.

        Calling this is the FIRST half of the per-job state machine:
            prepare_for_processing → ... job runs ... → scale_all_to_zero

        Args:
            reason: log-correlation tag, e.g. "pdf_job_<id>", "scrape_<sid>"
            min_replica: floor while job is active (default 1)
            max_replica: ceiling (default 4)
            scale_to_zero_timeout: minutes of idle before HF drains the last
                replica AFTER min is dropped to 0 (default 30)

        Returns:
            Dict[endpoint_key, success_bool] for each of the 4 endpoints.
        """
        try:
            from app.config import get_settings
            settings = get_settings()
            config_getters = {
                "qwen":    settings.get_qwen_config,
                "slig":    settings.get_slig_config,
                "yolo":    settings.get_yolo_config,
                "chandra": settings.get_chandra_config,
            }
        except Exception as e:
            logger.warning(
                "prepare_for_processing(reason=%s): could not load settings: %s",
                reason, e,
            )
            return {k: False for k in HF_ENDPOINT_NAMES}

        outcome: Dict[str, bool] = {}
        for key in HF_ENDPOINT_NAMES:
            ok = False
            try:
                cfg = config_getters[key]() or {}
                ep_name = cfg.get("endpoint_name") or HF_ENDPOINT_NAMES[key]
                ns = cfg.get("namespace", "basiliskan")
                token = cfg.get("hf_token") or cfg.get("endpoint_token")
                if not (ep_name and token):
                    logger.warning(
                        "prepare_for_processing: %s missing endpoint_name/token, skipping",
                        key,
                    )
                    outcome[key] = False
                    continue

                def _do(name=ep_name, namespace=ns, tok=token,
                        min_r=min_replica, max_r=max_replica,
                        sttz=scale_to_zero_timeout) -> bool:
                    from huggingface_hub import get_inference_endpoint
                    ep = get_inference_endpoint(
                        name=name, namespace=namespace, token=tok,
                    )
                    ep.fetch()
                    # Paused endpoints (we pause() at job terminal for
                    # instant drain) need resume() before update() — HF
                    # rejects scaling changes on paused endpoints.
                    if ep.status == "paused":
                        try:
                            ep.resume().wait(timeout=120)
                            ep.fetch()
                        except Exception as resume_err:
                            logger.warning(
                                f"   ⚠️ resume() before prepare failed for {name}: "
                                f"{resume_err} — proceeding to update anyway"
                            )
                    raw = getattr(ep, "raw", {}) or {}
                    current_scaling = (raw.get("compute", {}) or {}).get("scaling", {}) or {}
                    cur_min = current_scaling.get("minReplica")
                    cur_max = current_scaling.get("maxReplica")
                    cur_sttz = current_scaling.get("scaleToZeroTimeout")
                    # No-op when already at desired posture (avoid HF API write
                    # rate-limit hits when multiple jobs queue up).
                    if cur_min == min_r and cur_max == max_r and cur_sttz == sttz:
                        return True
                    ep.update(
                        min_replica=min_r,
                        max_replica=max_r,
                        scale_to_zero_timeout=sttz,
                    )
                    return True

                ok = await asyncio.to_thread(_do)
                if ok:
                    logger.info(
                        "   📈 %s prepared for job (min=%d, max=%d, scaleToZero=%dmin, reason=%s)",
                        key, min_replica, max_replica, scale_to_zero_timeout, reason,
                    )
            except Exception as e:
                logger.warning(
                    "   ⚠️ prepare_for_processing %s failed: %s (reason=%s)",
                    key, e, reason,
                )
            outcome[key] = ok

        success_count = sum(1 for v in outcome.values() if v)
        logger.info(
            "📈 EndpointController.prepare_for_processing(reason=%s): "
            "ready=%d/%d endpoints (min=%d, max=%d, scaleToZeroTimeout=%dmin)",
            reason, success_count, len(outcome),
            min_replica, max_replica, scale_to_zero_timeout,
        )
        return outcome

    async def scale_all_to_zero(self, reason: str = "cleanup") -> Dict[str, bool]:
        """Force every HF endpoint to min_replica=0.

        Sequential by design — easier to debug and 4 calls is not enough volume
        to justify parallel HF API hits. Manager path first; if a manager is
        not registered (partial warmup, crashed mid-job, never warmed), falls
        back to a direct HF API update so we still pin min_replica=0.

        After scale-down, resets the in-memory `_warmed` flags AND informs the
        auto-scaler that current_replicas=0, so the next job re-warms cleanly
        and the scaler doesn't think there's still capacity to use.

        Args:
            reason: short tag for log correlation, e.g. "pdf_job_<id>",
                    "agent_<id>", "admin_cancel", "periodic_idle".

        Returns:
            Dict[endpoint_key, success_bool] — same shape as warm_all().
        """
        from app.services.embeddings.endpoint_registry import endpoint_registry

        try:
            from app.config import get_settings
            settings = get_settings()
            config_getters = {
                "qwen":    settings.get_qwen_config,
                "slig":    settings.get_slig_config,
                "yolo":    settings.get_yolo_config,
                "chandra": settings.get_chandra_config,
            }
        except Exception as e:
            logger.warning("scale_all_to_zero(reason=%s): could not load settings for HF fallback: %s", reason, e)
            config_getters = {}

        manager_getters = {
            "qwen":    endpoint_registry.get_qwen_manager,
            "slig":    endpoint_registry.get_slig_manager,
            "yolo":    endpoint_registry.get_yolo_manager,
            "chandra": endpoint_registry.get_chandra_manager,
        }

        outcome: Dict[str, bool] = {}
        for key in HF_ENDPOINT_NAMES:
            scaled = False

            mgr = None
            try:
                mgr = manager_getters[key]()
            except Exception:
                mgr = None

            if mgr is not None and hasattr(mgr, "scale_to_zero"):
                try:
                    ok = await asyncio.to_thread(mgr.scale_to_zero)
                    if ok:
                        scaled = True
                        logger.info("   📉 %s scaled to zero (manager, reason=%s)", key, reason)
                except Exception as e:
                    logger.warning("   ⚠️ %s manager scale_to_zero failed: %s", key, e)

            if not scaled and key in config_getters:
                try:
                    cfg = config_getters[key]() or {}
                    ep_name = cfg.get("endpoint_name") or HF_ENDPOINT_NAMES[key]
                    ns = cfg.get("namespace", "basiliskan")
                    token = cfg.get("hf_token") or cfg.get("endpoint_token")
                    if ep_name and token:
                        def _do_direct_scale(name=ep_name, namespace=ns, tok=token) -> bool:
                            # Force-drain via pause() — kills the replica in
                            # seconds. Setting min_replica=0 alone left the
                            # replica running for up to 30 min idle (per
                            # scaleToZeroTimeout), wasting GPU billing on
                            # every job tail. pause() is the only HF API
                            # primitive that drains immediately. Next job's
                            # prepare_for_processing(min=1) resumes it.
                            from huggingface_hub import get_inference_endpoint
                            ep = get_inference_endpoint(name=name, namespace=namespace, token=tok)
                            ep.fetch()
                            if ep.status == "paused":
                                return True
                            ep.pause()
                            return True

                        ok = await asyncio.to_thread(_do_direct_scale)
                        if ok:
                            scaled = True
                            logger.info("   📉 %s scaled to zero (direct HF, reason=%s)", key, reason)
                except Exception as e:
                    logger.warning("   ⚠️ %s direct HF scale_to_zero failed: %s", key, e)

            outcome[key] = scaled
            self._warmed[key] = False

        try:
            auto_scaler = self._get_auto_scaler()
            if auto_scaler is not None:
                for short_key, hf_name in HF_ENDPOINT_NAMES.items():
                    if outcome.get(short_key):
                        try:
                            auto_scaler._current_replicas[hf_name] = 0
                        except Exception:
                            pass
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
        """Ask the auto-scaler to add replicas for one endpoint.

        Useful at job start when you know you're about to hit an endpoint
        hard — e.g. the main pipeline can call `request_scale_up('qwen', 2)`
        before Stage 3 kicks off. If HF has capacity, we get more replicas;
        if not, this is a no-op and the adaptive gate still keeps us safe.

        Returns True on success, False on failure/no-op.
        """
        auto_scaler = self._get_auto_scaler()
        if auto_scaler is None or not getattr(auto_scaler, "enabled", False):
            return False

        hf_name = HF_ENDPOINT_NAMES.get(endpoint)
        if hf_name is None:
            return False

        try:
            # scale_endpoint is sync + uses HF API; run in thread.
            ok = await asyncio.to_thread(
                auto_scaler.scale_endpoint, hf_name, desired_replicas
            )
            if ok:
                self._align_caps_with_replica_counts(auto_scaler)
            return bool(ok)
        except Exception as e:
            logger.warning("request_scale_up(%s, %d) failed: %s", endpoint, desired_replicas, e)
            return False

    # ────────────────────────────────────────────────────────────────────
    # Observability
    # ────────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Single snapshot of all 4 gates for progress reports + logs."""
        return {
            "qwen":    self.qwen.stats(),
            "slig":    self.slig.stats(),
            "yolo":    self.yolo.stats(),
            "chandra": self.chandra.stats(),
            "warmed":  dict(self._warmed),
        }

    def log_stats(self, prefix: str = "🎛️  EndpointController") -> None:
        """Pretty-print gate state at stage boundaries."""
        s = self.stats()
        logger.info(
            "%s: qwen=%d/%d (in=%d) | slig=%d/%d (in=%d) | yolo=%d/%d (in=%d) | chandra=%d/%d (in=%d)",
            prefix,
            s["qwen"]["limit"], s["qwen"]["max"], s["qwen"]["in_flight"],
            s["slig"]["limit"], s["slig"]["max"], s["slig"]["in_flight"],
            s["yolo"]["limit"], s["yolo"]["max"], s["yolo"]["in_flight"],
            s["chandra"]["limit"], s["chandra"]["max"], s["chandra"]["in_flight"],
        )


# ════════════════════════════════════════════════════════════════════════
# Module-level singleton. Import this everywhere:
#
#     from app.services.core.endpoint_controller import endpoint_controller
# ════════════════════════════════════════════════════════════════════════

endpoint_controller = EndpointController()
