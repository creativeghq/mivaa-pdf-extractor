"""
Provider abstraction for the structural-pass endpoint (PaddleOCR-VL on Modal).

The :class:`PaddleOCRManager` holds an :class:`EndpointProvider` and delegates
every lifecycle call (warmup / resume / scale-to-zero / health) to it, keeping
the inference + parse + metrics logic provider-agnostic. Today there is one
provider — :class:`ModalEndpointProvider` — because Modal owns autoscaling: a
scaled-down app auto-wakes on the first request and auto-drains after the
``scaledown_window`` configured at deploy time. So "resume"/"warmup" is just a
health probe until the container is up, and "scale to zero" is a no-op (Modal's
idle clock handles it).

The abstraction is kept (rather than inlined) so a second host can be added
later as another ``EndpointProvider`` with no change to the manager.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


# ====================================================================== #
# Base
# ====================================================================== #
class EndpointProvider(ABC):
    """Lifecycle + auth strategy for one HTTP endpoint.

    Subclasses implement :meth:`resolve_base_url`, :meth:`resume_if_needed`,
    :meth:`warmup` and :meth:`scale_to_zero`. The shared :meth:`health_check`
    (GET ``/health``) and :meth:`_warmup_by_probe` loop work for any provider
    whose endpoint exposes an unauthenticated ``/health`` route.
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
        self.label = label                      # short key for logs, e.g. "paddleocr"
        self._token = token or ""
        self.warmup_timeout = warmup_timeout
        self.health_path = health_path
        self.health_timeout = health_timeout

        # Lifecycle state — read by the manager (get_stats) and the controller
        # (which resets warmup_completed after a scale-down).
        self.warmup_completed: bool = False
        self.resume_count: int = 0
        self.last_resume_time: float | None = None
        self.total_uptime: float = 0.0

    # ---- inference plumbing (used by the manager) ---------------------- #
    @abstractmethod
    def resolve_base_url(self) -> str:
        """Base URL WITHOUT the route — the manager appends ``/parse`` etc.

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
        """Liveness probe via GET ``/health`` (open even when the main route
        would 401), the canonical readiness signal."""
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
        """Warm the endpoint with ONE patient probe (not a probe loop).

        A scaled-to-zero Modal container cold-starts in ~250s; a single GET
        /health with a long timeout (requests follows Modal's 303 cold-start
        redirect) rides that boot and returns 200 when ready. Using one request
        instead of a loop of short probes means warmup never sprays / floods the
        endpoint while it boots. Liveness elsewhere uses the short health_check.
        """
        if self.warmup_completed:
            return True
        logger.info(
            "🔥 Warming up %s endpoint (provider=%s) — single patient probe (max %ds)...",
            self.label, self.provider_name, self.warmup_timeout,
        )
        start = time.time()
        try:
            base = (self.resolve_base_url() or "").rstrip("/")
            if not base:
                logger.error("❌ %s warmup: base URL unresolved", self.label)
                return False
            url = base if base.endswith(self.health_path) else base + self.health_path
            resp = requests.get(
                url, headers=self.auth_header(), timeout=self.warmup_timeout
            )
            if resp.status_code == 200:
                logger.info(
                    "✅ %s endpoint warmed up in %.1fs (single probe)",
                    self.label, time.time() - start,
                )
                self.warmup_completed = True
                self.last_resume_time = time.time()
                return True
            logger.error(
                "❌ %s warmup got HTTP %s after %.1fs",
                self.label, resp.status_code, time.time() - start,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(
                "❌ %s warmup probe failed after %.1fs: %s",
                self.label, time.time() - start, e,
            )
        return False

    def stats(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "resume_count": self.resume_count,
            "warmup_completed": self.warmup_completed,
            "total_uptime_seconds": self.total_uptime,
        }


# ====================================================================== #
# Modal (and any self-managed HTTP endpoint behind a stable URL)
# ====================================================================== #
class ModalEndpointProvider(EndpointProvider):
    """Modal-hosted endpoint. Modal owns autoscaling:

    * a scaled-down app **auto-wakes** on the first request,
    * it **auto-drains** the GPU container after ``scaledown_window`` (set at
      deploy time in ``modal_app/paddleocr_vl.py``).

    So warmup == health-probe-until-up, resume == warmup (the probe wakes it),
    and scale-to-zero is a no-op handled by Modal's idle clock. The URL is
    static (the ``modal deploy`` output) and the bearer is the value of the
    ``paddleocr-api-key`` Modal secret.
    """

    provider_name = "modal"

    def __init__(
        self,
        *,
        base_url: str,
        token: str = "",
        label: str = "paddleocr",
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
        # A web endpoint exposes no manual drain primitive, so this is a no-op
        # that reports success — the cost goal is met by the scaledown_window.
        if self.last_resume_time:
            self.total_uptime += time.time() - self.last_resume_time
        self.warmup_completed = False
        logger.info(
            "📉 %s (modal): scale-to-zero deferred to Modal scaledown_window", self.label
        )
        return True


# ====================================================================== #
# Factory
# ====================================================================== #
def build_endpoint_provider(config: Dict[str, Any], *, label: str) -> EndpointProvider:
    """Build the endpoint provider from a model config dict.

    Currently Modal-only (``config['provider']`` is expected to be ``modal``).
    Uses ``modal_url`` (falling back to ``endpoint_url``) + ``modal_api_key``.
    """
    warmup_timeout = int(config.get("warmup_timeout", 300))
    return ModalEndpointProvider(
        base_url=config.get("modal_url", "") or config.get("endpoint_url", ""),
        token=config.get("modal_api_key", ""),
        label=label,
        warmup_timeout=warmup_timeout,
    )
