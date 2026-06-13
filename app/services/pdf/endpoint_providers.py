"""
Provider abstraction for model-serving endpoints behind an OpenAI-compatible
``/v1/chat/completions`` route.

The **inference** call (POST an ``image_url`` + prompt, read ``choices[0].message
.content``) is identical across providers — they all run vLLM and speak the
OpenAI Chat API. Only the **lifecycle** differs:

* **HuggingFace Inference Endpoints** (:class:`HuggingFaceEndpointProvider`) —
  explicit SDK lifecycle: resume a paused / scaled-to-zero endpoint, poll until
  HF allocates a GPU replica, force scale-to-zero at job terminal. Needs
  ``huggingface_hub`` + a write token.

* **Modal** (:class:`ModalEndpointProvider`) — the platform owns autoscaling.
  A scaled-down app auto-wakes on the first request and auto-drains after the
  ``scaledown_window`` configured at deploy time. So "resume"/"warmup" is just a
  health probe until the container is up, and "scale to zero" is a no-op (Modal
  does it on its own idle clock). Works for any self-managed vLLM behind a
  stable URL guarded by an ``--api-key`` bearer.

A model picks its provider via ``*_PROVIDER`` config (``huggingface`` | ``modal``),
so a single model can move between GPU hosts with one env var and no code change.
The :class:`SuryaEndpointManager` holds a provider and delegates every lifecycle
call to it; the inference / retry / parse / metrics logic is provider-agnostic.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests

try:
    from huggingface_hub import get_inference_endpoint
    HF_HUB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    HF_HUB_AVAILABLE = False
    logging.warning("huggingface_hub not available - HF endpoint pause/resume disabled")

logger = logging.getLogger(__name__)


# ====================================================================== #
# Base
# ====================================================================== #
class EndpointProvider(ABC):
    """Lifecycle + auth strategy for one OpenAI-compatible vLLM endpoint.

    Subclasses implement :meth:`resolve_base_url`, :meth:`resume_if_needed`,
    :meth:`warmup` and :meth:`scale_to_zero`. The shared :meth:`health_check`
    (GET ``/health``) and :meth:`_warmup_by_probe` loop work for every provider
    because vLLM exposes an unauthenticated ``/health`` route.
    """

    provider_name: str = "base"

    def __init__(
        self,
        *,
        label: str,
        token: str = "",
        warmup_timeout: int = 300,
        health_path: str = "/health",
        health_timeout: int = 15,
    ):
        self.label = label                      # short key for logs, e.g. "surya"
        self._token = token or ""
        self.warmup_timeout = warmup_timeout
        self.health_path = health_path
        self.health_timeout = health_timeout

        # Lifecycle state — read by the manager (get_stats) and the controller
        # (which resets warmup_completed after a scale-down).
        self.warmup_completed: bool = False
        self.resume_count: int = 0
        self.last_resume_time: Optional[float] = None
        self.total_uptime: float = 0.0

    # ---- inference plumbing (used by the manager) ---------------------- #
    @abstractmethod
    def resolve_base_url(self) -> str:
        """Base URL WITHOUT the ``/v1/...`` route — the manager appends it.

        Returns an empty string when the endpoint is not configured/resolvable.
        """

    def auth_token(self) -> str:
        return self._token or ""

    def auth_header(self) -> Dict[str, str]:
        tok = self.auth_token()
        return {"Authorization": f"Bearer {tok}"} if tok else {}

    # ---- lifecycle ----------------------------------------------------- #
    @abstractmethod
    def resume_if_needed(self) -> bool:
        """Ensure the endpoint is up before an inference call. True = ready."""

    @abstractmethod
    def warmup(self) -> bool:
        """Bring the endpoint to a known-warm state. True = warm."""

    @abstractmethod
    def scale_to_zero(self) -> bool:
        """Drain to $0 at job terminal. True = drained (or deferred to platform)."""

    # ---- shared health probe ------------------------------------------- #
    def health_check(self) -> bool:
        """Liveness probe via GET ``/health``. vLLM's ``/health`` is open even
        when the chat route would 401/400, so this is the canonical readiness
        signal across HF and Modal."""
        try:
            base = (self.resolve_base_url() or "").rstrip("/")
            if not base:
                return False
            health_url = base if base.endswith(self.health_path) else base + self.health_path
            resp = requests.get(health_url, headers=self.auth_header(), timeout=self.health_timeout)
            return resp.status_code == 200
        except Exception as e:  # noqa: BLE001 - probe must never raise
            logger.debug("   %s /health probe exception: %s", self.label, e)
            return False

    def _warmup_by_probe(self) -> bool:
        """Probe GET ``/health`` until 200 or ``warmup_timeout``. Shared warmup
        body for any provider that auto-wakes on traffic (Modal) and for HF
        after its SDK resume returns."""
        if self.warmup_completed:
            return True
        logger.info(
            "🔥 Warming up %s endpoint (provider=%s, max %ds)...",
            self.label, self.provider_name, self.warmup_timeout,
        )
        start = time.time()
        attempt = 0
        base_delay, max_delay = 2, 15
        while (time.time() - start) < self.warmup_timeout:
            attempt += 1
            if self.health_check():
                logger.info(
                    "✅ %s endpoint warmed up in %.1fs (%d attempts)",
                    self.label, time.time() - start, attempt,
                )
                self.warmup_completed = True
                self.last_resume_time = time.time()
                return True
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            remaining = self.warmup_timeout - (time.time() - start)
            time.sleep(min(delay, remaining) if remaining > 0 else 0)
        logger.error("❌ %s warmup failed after %.1fs", self.label, time.time() - start)
        return False

    def stats(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "resume_count": self.resume_count,
            "warmup_completed": self.warmup_completed,
            "total_uptime_seconds": self.total_uptime,
        }


# ====================================================================== #
# Modal (and any self-managed vLLM behind a stable URL)
# ====================================================================== #
class ModalEndpointProvider(EndpointProvider):
    """Modal-hosted vLLM. Modal owns autoscaling:

    * a scaled-down app **auto-wakes** on the first request,
    * it **auto-drains** the GPU container after ``scaledown_window`` (set at
      deploy time in ``modal_app/surya_vllm.py``).

    So warmup == health-probe-until-up, resume == warmup (the probe wakes it),
    and scale-to-zero is a no-op handled by Modal's idle clock. The URL is
    static (the ``modal deploy`` output) and the bearer token is the vLLM
    ``--api-key`` injected from the ``surya-vllm-api-key`` Modal secret.
    """

    provider_name = "modal"

    def __init__(
        self,
        *,
        base_url: str,
        token: str = "",
        label: str = "surya",
        warmup_timeout: int = 300,
        health_timeout: int = 15,
    ):
        super().__init__(
            label=label,
            token=token,
            warmup_timeout=warmup_timeout,
            health_timeout=health_timeout,
        )
        self._base_url = (base_url or "").strip().rstrip("/")

    def resolve_base_url(self) -> str:
        return self._base_url

    def warmup(self) -> bool:
        return self._warmup_by_probe()

    def resume_if_needed(self) -> bool:
        # Warm already? Cheap re-validate. Cold? The probe loop both waits for
        # Modal to cold-start a container AND confirms readiness.
        if self.warmup_completed:
            if self.health_check():
                return True
            self.warmup_completed = False
        return self.warmup()

    def scale_to_zero(self) -> bool:
        # Modal drains the container on its own idle clock (scaledown_window).
        # A `@modal.web_server` endpoint exposes no manual drain primitive, so
        # this is intentionally a no-op that reports success — the cost goal is
        # met by the short scaledown_window configured at deploy time.
        if self.last_resume_time:
            self.total_uptime += time.time() - self.last_resume_time
        self.warmup_completed = False
        logger.info(
            "📉 %s (modal): scale-to-zero deferred to Modal scaledown_window", self.label
        )
        return True


# ====================================================================== #
# HuggingFace Inference Endpoints
# ====================================================================== #
class HuggingFaceEndpointProvider(EndpointProvider):
    """HuggingFace Inference Endpoint lifecycle (SDK-driven).

    Carries the resume / poll-for-GPU / scale-to-zero state machine that used
    to live inline in :class:`SuryaEndpointManager`. Reuses the base
    :meth:`health_check` probe (vLLM ``/health``) for liveness.
    """

    provider_name = "huggingface"

    def __init__(
        self,
        *,
        endpoint_url: str,
        token: str,
        endpoint_name: Optional[str],
        namespace: Optional[str],
        warmup_timeout: int = 300,
        max_resume_retries: int = 3,
        label: str = "surya",
        health_timeout: int = 15,
    ):
        super().__init__(
            label=label,
            token=token,
            warmup_timeout=warmup_timeout,
            health_timeout=health_timeout,
        )
        self._base_url = (endpoint_url or "").strip()
        self.endpoint_name = endpoint_name
        self.namespace = namespace
        self.max_resume_retries = max_resume_retries

        self._endpoint = None
        self._can_pause_resume = HF_HUB_AVAILABLE and bool(endpoint_name) and bool(namespace)

        if self._can_pause_resume:
            logger.info(
                "✅ HF provider for %s initialized with pause/resume: endpoint=%s, namespace=%s",
                label, endpoint_name, namespace,
            )
        else:
            logger.warning(
                "⚠️ HF provider for %s WITHOUT pause/resume: hf_hub=%s, name=%s, ns=%s",
                label, HF_HUB_AVAILABLE, endpoint_name, namespace,
            )

    # ---- endpoint handle ---------------------------------------------- #
    def _get_endpoint(self):
        if not self._can_pause_resume:
            return None
        if self._endpoint is None:
            try:
                self._endpoint = get_inference_endpoint(
                    name=self.endpoint_name, namespace=self.namespace, token=self._token,
                )
                logger.info("✅ Connected to HF endpoint: %s", self.endpoint_name)
            except Exception as e:  # noqa: BLE001
                logger.error("❌ Failed to get HF endpoint instance: %s", e)
                return None
        return self._endpoint

    def resolve_base_url(self) -> str:
        """Static ``endpoint_url`` when set, else a live HF SDK lookup (covers an
        unset ``SURYA_ENDPOINT_URL`` secret)."""
        url = (self._base_url or "").strip()
        if url.startswith(("http://", "https://")):
            return url
        ep = self._get_endpoint()
        if ep is not None:
            try:
                ep.fetch()
                live_url = getattr(ep, "url", None)
                if live_url and live_url.startswith(("http://", "https://")):
                    self._base_url = live_url
                    logger.info("   🔗 %s endpoint URL resolved dynamically: %s", self.label, live_url)
                    return live_url
            except Exception as e:  # noqa: BLE001
                logger.warning("   ⚠️ %s live URL resolve failed: %s", self.label, e)
        return ""

    # ---- lifecycle ----------------------------------------------------- #
    def resume_if_needed(self) -> bool:
        if not self._can_pause_resume:
            return True
        endpoint = self._get_endpoint()
        if not endpoint:
            return False
        try:
            endpoint.fetch()

            if endpoint.status == "running":
                if self.health_check():
                    logger.info("✅ %s endpoint running and healthy", self.label)
                    return True
                logger.warning("⚠️ %s reports running but /health failed — re-warming", self.label)
                self.warmup_completed = False
                return self.warmup()

            if endpoint.status == "initializing":
                logger.info("⏳ %s endpoint initializing, waiting...", self.label)
                return self._wait_for_running(endpoint)

            if endpoint.status in ("paused", "scaledToZero"):
                logger.info("🔄 Resuming %s endpoint (status: %s)...", self.label, endpoint.status)
                from app.services.embeddings.hf_errors import is_hf_billing_error, HFBillingError
                for attempt in range(self.max_resume_retries):
                    try:
                        endpoint.resume().wait(timeout=90)
                        self.resume_count += 1
                        self.last_resume_time = time.time()
                        self.warmup_completed = False
                        logger.info(
                            "✅ %s endpoint resumed (attempt %d/%d)",
                            self.label, attempt + 1, self.max_resume_retries,
                        )
                        if not self.warmup():
                            logger.error("❌ %s endpoint warmup failed", self.label)
                            return False
                        return True
                    except Exception as e:  # noqa: BLE001
                        if is_hf_billing_error(e):
                            logger.error("💳 HF billing error on %s resume — aborting: %s", self.label, e)
                            raise HFBillingError(self.endpoint_name or self.label, self.namespace, original=e) from e
                        logger.warning("⚠️ %s resume attempt %d failed: %s", self.label, attempt + 1, e)
                        if attempt < self.max_resume_retries - 1:
                            time.sleep(2 ** attempt)
                        else:
                            raise

            logger.error("❌ %s endpoint in unexpected state: %s", self.label, endpoint.status)
            return False
        except Exception as e:  # noqa: BLE001
            logger.error("❌ Failed to resume %s endpoint: %s", self.label, e)
            return False

    def _wait_for_running(self, endpoint) -> bool:
        start_time = time.time()
        poll_interval = 5
        last_ready_replica = 0
        last_progress_time = time.time()
        NO_PROGRESS_TIMEOUT = 90

        while (time.time() - start_time) < self.warmup_timeout:
            try:
                endpoint.fetch()
                if endpoint.status == "running":
                    logger.info("✅ %s endpoint ready after %.1fs", self.label, time.time() - start_time)
                    self.last_resume_time = time.time()
                    return self.warmup()
                if endpoint.status in ("failed", "error"):
                    logger.error("❌ %s endpoint failed: %s", self.label, endpoint.status)
                    return False

                try:
                    compute = getattr(endpoint, "compute", {}) or {}
                    current_ready = (compute.get("readyReplica") if isinstance(compute, dict) else None) or 0
                    target = (compute.get("targetReplica") if isinstance(compute, dict) else None) or "?"
                except Exception:
                    current_ready, target = 0, "?"

                if current_ready > last_ready_replica:
                    last_ready_replica = current_ready
                    last_progress_time = time.time()
                elif (time.time() - last_progress_time) > NO_PROGRESS_TIMEOUT:
                    logger.error(
                        "❌ HuggingFace cannot allocate GPU capacity for %s endpoint '%s' — "
                        "no progress in %ds (state=%s, ready=0/%s). HF infra issue, not a code "
                        "bug; switch this model to Modal (SURYA_PROVIDER=modal) or set "
                        "min_replica=1 on the HF endpoint to avoid cold starts.",
                        self.label, self.endpoint_name, NO_PROGRESS_TIMEOUT, endpoint.status, target,
                    )
                    return False

                logger.info(
                    "   ⏳ %s still %s (%.0fs, ready=%s/%s)",
                    self.label, endpoint.status, time.time() - start_time, current_ready, target,
                )
                time.sleep(poll_interval)
            except Exception as e:  # noqa: BLE001
                logger.warning("⚠️ Error checking %s status: %s", self.label, e)
                time.sleep(poll_interval)

        logger.error("❌ Timeout waiting for %s endpoint (max %ds)", self.label, self.warmup_timeout)
        return False

    def warmup(self) -> bool:
        if self.warmup_completed:
            return True
        if self._can_pause_resume:
            endpoint = self._get_endpoint()
            if endpoint:
                try:
                    endpoint.fetch()
                    if endpoint.status in ("paused", "scaledToZero"):
                        logger.warning("⚠️ %s endpoint is %s — resuming", self.label, endpoint.status)
                        endpoint.resume().wait(timeout=90)
                        self.resume_count += 1
                        self.last_resume_time = time.time()
                except Exception as e:  # noqa: BLE001
                    logger.warning("⚠️ Failed to check/resume %s during warmup: %s", self.label, e)
        return self._warmup_by_probe()

    def scale_to_zero(self) -> bool:
        if not self._can_pause_resume:
            return False
        endpoint = self._get_endpoint()
        if not endpoint:
            return False
        try:
            endpoint.fetch()
            if endpoint.status in ("scaledToZero", "paused"):
                return True
            logger.info("📉 Scaling %s endpoint to zero (was: %s)", self.label, endpoint.status)
            endpoint.scale_to_zero()
            if self.last_resume_time:
                self.total_uptime += time.time() - self.last_resume_time
            self.warmup_completed = False
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("❌ Failed to scale %s endpoint to zero: %s", self.label, e)
            return False


# ====================================================================== #
# Factory
# ====================================================================== #
def build_endpoint_provider(config: Dict[str, Any], *, label: str) -> EndpointProvider:
    """Build the right provider from a model config dict.

    Reads ``config['provider']`` (``huggingface`` | ``modal``; default
    ``huggingface``). For ``modal`` it uses ``modal_url`` + ``modal_api_key``;
    for ``huggingface`` it uses ``endpoint_url`` + ``hf_token`` +
    ``endpoint_name`` + ``namespace``.
    """
    provider = (config.get("provider") or "huggingface").strip().lower()
    warmup_timeout = int(config.get("warmup_timeout", 300))

    if provider == "modal":
        return ModalEndpointProvider(
            base_url=config.get("modal_url", "") or config.get("endpoint_url", ""),
            token=config.get("modal_api_key", ""),
            label=label,
            warmup_timeout=warmup_timeout,
        )

    return HuggingFaceEndpointProvider(
        endpoint_url=config.get("endpoint_url", ""),
        token=config.get("hf_token", ""),
        endpoint_name=config.get("endpoint_name"),
        namespace=config.get("namespace"),
        warmup_timeout=warmup_timeout,
        max_resume_retries=int(config.get("max_resume_retries", 3)),
        label=label,
    )
