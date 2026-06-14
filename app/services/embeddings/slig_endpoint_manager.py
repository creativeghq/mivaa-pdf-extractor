"""
SLIG (SigLIP2) visual-embedding endpoint manager.

Thin lifecycle wrapper over a :class:`ModalEndpointProvider` — Modal owns
autoscaling, so ``warmup`` = a GET ``/health`` probe and ``scale_to_zero`` is
Modal's idle clock (a no-op here). The actual inference (image / text / zero-shot
/ similarity over POST ``/infer``) lives in :class:`SLIGClient`; this manager only
provides the uniform lifecycle interface that the EndpointController + ``warm_all``
orchestrator call across every Modal-hosted endpoint.

Migrated off HuggingFace Inference Endpoints 2026-06-14 (parity verified,
cosine = 1.0 vs the HF endpoint). All ``huggingface_hub`` pause/resume/scale code
is gone — SLIG is the `slig` Modal app (``modal_app/slig.py``).
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from app.services.pdf.endpoint_providers import EndpointProvider, build_endpoint_provider

logger = logging.getLogger(__name__)


class SLIGEndpointManager:
    """Lifecycle manager for the SLIG visual-embedding endpoint (Modal-hosted)."""

    def __init__(
        self,
        provider: EndpointProvider,
        model_name: str = "basiliskan/slig",
        enabled: bool = True,
    ):
        self.provider = provider
        self.model_name = model_name
        self.enabled = enabled
        self.last_used: Optional[float] = None
        self.inference_count: int = 0
        logger.info(
            "✅ SLIG manager initialized (provider=%s, model=%s)",
            provider.provider_name, model_name,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SLIGEndpointManager":
        """Build a manager (+ Modal provider) from ``Settings.get_slig_config()``."""
        provider = build_endpoint_provider(config, label="slig")
        return cls(
            provider=provider,
            model_name=config.get("model_name", "basiliskan/slig"),
            enabled=config.get("enabled", True),
        )

    # ------------------------------------------------------------------ #
    # Lifecycle — delegated to the provider (controller + warm_all call these)
    # ------------------------------------------------------------------ #
    def resume_if_needed(self) -> bool:
        return self.provider.resume_if_needed()

    def warmup(self) -> bool:
        return self.provider.warmup()

    def scale_to_zero(self) -> bool:
        return self.provider.scale_to_zero()

    def pause_if_idle(self) -> bool:
        # Modal drains on its own idle clock — nothing to pause. Kept as a no-op
        # so any legacy caller is harmless.
        return True

    def _test_inference(self) -> bool:
        """Liveness probe (GET /health) — delegates to the provider. Kept under
        this name because the warmup orchestrator + controller call it uniformly
        across managers."""
        return self.provider.health_check()

    @property
    def warmup_completed(self) -> bool:
        return self.provider.warmup_completed

    @warmup_completed.setter
    def warmup_completed(self, value: bool) -> None:
        self.provider.warmup_completed = bool(value)

    @property
    def endpoint_url(self) -> str:
        return self.provider.resolve_base_url()

    @property
    def endpoint_name(self) -> Optional[str]:
        return getattr(self.provider, "endpoint_name", None)

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    def mark_used(self):
        self.last_used = time.time()
        self.inference_count += 1

    def get_stats(self) -> Dict[str, Any]:
        return {
            "provider": self.provider.provider_name,
            "model_name": self.model_name,
            "inference_count": self.inference_count,
            "warmup_completed": self.provider.warmup_completed,
            "total_uptime_seconds": self.provider.total_uptime,
            "last_used": datetime.fromtimestamp(self.last_used).isoformat() if self.last_used else None,
            "enabled": self.enabled,
        }
