"""PaddleOCR-VL structural-pass endpoint manager."""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from app.services.pdf.endpoint_providers import EndpointProvider, build_endpoint_provider
from app.services.pdf.paddleocr_pipeline import (
    PaddleRegion,
    parse_parse_response,
    regions_to_reading_text,
)

logger = logging.getLogger(__name__)


def _log_paddleocr_gpu_cost(
    task: str,
    latency_ms: int,
    job_id: Optional[str],
    image_id: Optional[str],
    product_id: Optional[str],
    outcome: str = "success",
) -> None:
    """Log the GPU-seconds cost of a PaddleOCR call to ai_usage_logs."""
    try:
        from datetime import datetime, timezone
        from app.config.ai_pricing import ai_pricing
        from app.services.core.supabase_client import get_supabase_client, repeatable_insert

        secs = max(latency_ms / 1000.0, 0.001)
        cost_data = ai_pricing.calculate_time_based_cost(
            model="paddleocr-vl", inference_seconds=secs
        )
        billed = float(cost_data.get("billed_cost_usd", 0.0))
        repeatable_insert(get_supabase_client().client, "ai_usage_logs", {
            "operation_type": task,
            "model_name": "paddleocr-vl",
            "input_tokens": 0,
            "output_tokens": 0,
            "raw_cost_usd": billed,
            "markup_multiplier": 1.0,
            "billed_cost_usd": billed,
            "job_id": job_id,
            "product_id": product_id,
            "image_id": image_id,
            "module_slug": "pdf_pipeline",
            "metadata": {
                "latency_ms": latency_ms,
                "billing": "time_based",
                "gpu_hourly_usd": 1.0,
                "outcome": outcome,
                # `outcome` already carries the answer; `success` is the key
                # `ops.silent_zero_provider` actually reads. Derived from the one field
                # rather than passed separately, so the two cannot disagree.
                "success": outcome == "success",
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as log_err:  # noqa: BLE001
        # WARNING, not DEBUG: a dropped row here under-reports GPU spend, and at DEBUG
        # that loss leaves no trace anywhere it would ever be read.
        logger.warning("PaddleOCR GPU cost log failed (non-fatal): %s", log_err)


class PaddleOCRResponseError(RuntimeError):
    """Raised when PaddleOCR returns output that cannot be parsed into regions.

    Callers catch this and mark the page ``ocr_failed`` (retryable) rather than
    persist an empty layout.
    """


class PaddleOCRConfigError(RuntimeError):
    """Raised on a NON-retryable endpoint error (401/403/404).

    A wrong bearer key (401/403) or wrong URL (404) is a configuration problem,
    not a transient one — retrying just multiplies doomed requests. The manager
    raises this immediately (no retry) and Stage 1 aborts the whole job, so one
    misconfiguration surfaces as a single fast failure instead of ~100 calls.
    """


class PaddleOCRManager:
    """Manages the PaddleOCR-VL structural pass over a Modal endpoint provider."""

    # Transient-failure retries (HTTP 5xx / timeouts). PaddleOCR is deterministic
    # — a retry just re-issues the same call.
    _MAX_ATTEMPTS = 3

    def __init__(
        self,
        provider: EndpointProvider,
        model_name: str = "paddleocr-vl",
        inference_timeout: int = 180,
        warmup_timeout: int = 300,
        max_image_pixels: int = 8_000_000,
        enabled: bool = True,
    ):
        self.provider = provider
        self.model_name = model_name
        self.inference_timeout = inference_timeout
        self.warmup_timeout = warmup_timeout
        self.max_image_pixels = max_image_pixels
        self.enabled = enabled

        self.last_used: Optional[float] = None
        self.inference_count: int = 0

        logger.info(
            "✅ PaddleOCR manager initialized (provider=%s, model=%s)",
            provider.provider_name, model_name,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PaddleOCRManager":
        """Build a manager (+ provider) from ``Settings.get_paddleocr_config()``."""
        provider = build_endpoint_provider(config, label="paddleocr")
        return cls(
            provider=provider,
            model_name=config.get("model_name", "paddleocr-vl"),
            inference_timeout=config.get("inference_timeout", 180),
            warmup_timeout=config.get("warmup_timeout", 300),
            max_image_pixels=config.get("max_image_pixels", 8_000_000),
            enabled=True,
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
    def endpoint_name(self) -> Optional[str]:
        return getattr(self.provider, "endpoint_name", None)

    @property
    def endpoint_url(self) -> str:
        return self.provider.resolve_base_url()

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    @property
    def resume_count(self) -> int:
        return self.provider.resume_count

    @property
    def total_uptime(self) -> float:
        return self.provider.total_uptime

    def mark_used(self):
        self.last_used = time.time()
        self.inference_count += 1

    def get_stats(self) -> Dict[str, Any]:
        return {
            "provider": self.provider.provider_name,
            "endpoint_name": self.endpoint_name,
            "model_name": self.model_name,
            "resume_count": self.provider.resume_count,
            "inference_count": self.inference_count,
            "total_uptime_seconds": self.provider.total_uptime,
            "total_uptime_hours": self.provider.total_uptime / 3600,
            "warmup_completed": self.provider.warmup_completed,
            "last_used": datetime.fromtimestamp(self.last_used).isoformat() if self.last_used else None,
            "enabled": self.enabled,
        }

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def run_structural_pass(
        self,
        image_input: Any,
        caller: str = "ad_hoc",
        page_number: Optional[int] = None,
        job_id: Optional[str] = None,
        document_id: Optional[str] = None,
        product_id: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full-page structural pass (``page`` mode).

        Raises:
            PaddleOCRResponseError: a non-empty response that parsed to zero
                regions, after retries — the page is marked ocr_failed (retryable).
            requests.HTTPError: HTTP/endpoint-health failures fail fast.
        """
        if not self.enabled:
            raise Exception("PaddleOCR endpoint is disabled")
        if not self.provider.resume_if_needed():
            raise Exception(f"Failed to resume PaddleOCR endpoint (provider={self.provider.provider_name})")

        image_bytes = _coerce_image_to_png_bytes(image_input, self.max_image_pixels)

        last_error: Optional[Exception] = None
        for attempt_idx in range(1, self._MAX_ATTEMPTS + 1):
            start_time = time.time()
            try:
                payload = self._do_parse(image_bytes, mode="page")
            except requests.HTTPError as he:
                last_error = he
                status = getattr(getattr(he, "response", None), "status_code", None)
                non_retryable = status in (401, 403, 404)
                self._emit_metric(
                    caller=caller, page_number=page_number, image_id=None,
                    job_id=job_id, document_id=document_id, attempt_number=attempt_idx,
                    outcome="failed_config_error" if non_retryable else "failed_http_error",
                    region_count=None, chars_count=None,
                    failure_mode_head=str(he)[:200],
                    latency_ms=int((time.time() - start_time) * 1000),
                )
                # An HTTP error still consumed GPU-seconds up to the point it failed
                # (a timeout consumes the FULL budget). Bill it. The 401/403/404 case
                # below is the one exception worth noting: it fails before any GPU
                # work, so it is logged with a near-zero latency and is harmless.
                _log_paddleocr_gpu_cost(
                    task="pdf_structural_pass",
                    latency_ms=int((time.time() - start_time) * 1000),
                    job_id=job_id,
                    image_id=image_id,
                    product_id=product_id,
                    outcome="failed_config_error" if non_retryable else "failed_http_error",
                )
                if non_retryable:
                    # 401/403 = wrong bearer key; 404 = wrong URL. Retrying is
                    # pointless and floods the endpoint — fail fast so Stage 1
                    # aborts the whole job instead of doing pages × _MAX_ATTEMPTS.
                    raise PaddleOCRConfigError(
                        f"PaddleOCR endpoint misconfigured (HTTP {status}) — check "
                        f"PADDLEOCR_MODAL_API_KEY / PADDLEOCR_MODAL_URL. {str(he)[:160]}"
                    ) from he
                if attempt_idx < self._MAX_ATTEMPTS:
                    time.sleep(2 ** (attempt_idx - 1))
                    continue
                raise

            latency_ms = int((time.time() - start_time) * 1000)
            raw_regions = payload.get("regions")
            regions: List[PaddleRegion] = parse_parse_response(payload)
            # Two distinct zero-region causes, previously collapsed into one.
            _dropped_every_region = bool(raw_regions) and not regions
            if raw_regions is None or _dropped_every_region:
                last_error = PaddleOCRResponseError(
                    f"PaddleOCR /parse returned no usable regions "
                    f"({'all regions dropped in parsing' if _dropped_every_region else 'no regions key'}). "
                    f"head={str(payload)[:300]!r}"
                )
                self._emit_metric(
                    caller=caller, page_number=page_number, image_id=None,
                    job_id=job_id, document_id=document_id, attempt_number=attempt_idx,
                    outcome="failed_all_regions_dropped" if _dropped_every_region else "failed_no_regions",
                    region_count=0, chars_count=0,
                    failure_mode_head=str(payload)[:200], latency_ms=latency_ms,
                )
                # A dropped-everything attempt still consumed real GPU-seconds.
                _log_paddleocr_gpu_cost(
                    task="pdf_structural_pass",
                    latency_ms=latency_ms,
                    job_id=job_id,
                    image_id=image_id,
                    product_id=product_id,
                    outcome="failed_all_regions_dropped" if _dropped_every_region else "failed_no_regions",
                )
                if attempt_idx < self._MAX_ATTEMPTS:
                    time.sleep(2 ** (attempt_idx - 1))
                    continue
                raise last_error

            generated_text = regions_to_reading_text(regions)
            self._on_success(start_time)
            self._emit_metric(
                caller=caller, page_number=page_number, image_id=None,
                job_id=job_id, document_id=document_id, attempt_number=attempt_idx,
                outcome="success" if attempt_idx == 1 else "success_after_retry",
                region_count=len(regions), chars_count=len(generated_text),
                failure_mode_head=None, latency_ms=latency_ms,
            )
            logger.info(
                "✅ PaddleOCR structural pass: %d regions / %d chars in %dms "
                "(attempt=%d, provider=%s)",
                len(regions), len(generated_text), latency_ms, attempt_idx,
                self.provider.provider_name,
            )
            # GPU-seconds cost → ai_usage_logs (rolls up into total_ai_cost_usd).
            # paddleocr_metrics above is endpoint telemetry; this is billing.
            _log_paddleocr_gpu_cost(
                task="pdf_structural_pass",
                latency_ms=latency_ms,
                job_id=job_id,
                image_id=image_id,
                product_id=product_id,
            )
            return {
                "regions": regions,
                "generated_text": generated_text,
                "raw": payload,
                "attempts_made": attempt_idx,
            }

        # Should be unreachable — both branches raise on the last attempt.
        assert last_error is not None
        raise last_error

    # run_block_ocr (block-mode per-crop OCR) was REMOVED 2026-07-04 (S1-2): it had
    # no callers — Phase-3 per-image OCR goes through run_structural_pass (page mode)
    # in ocr_service, not block mode. The Modal /parse endpoint still supports
    # mode="block" if a future caller needs it.

    def _do_parse(self, image_bytes: bytes, mode: str) -> Dict[str, Any]:
        """Single ``/parse`` call. Returns the decoded JSON.

        Base URL + bearer come from the active provider (Modal). Raises
        ``requests.HTTPError`` on HTTP failure.
        """
        import base64

        base_url = self.provider.resolve_base_url().rstrip("/")
        if not base_url:
            raise requests.HTTPError(
                f"PaddleOCR endpoint URL unresolved (provider={self.provider.provider_name})"
            )
        api_url = base_url + "/parse"
        headers = {"Content-Type": "application/json", **self.provider.auth_header()}
        body = {
            "image_b64": base64.b64encode(image_bytes).decode("ascii"),
            "mode": mode,
        }
        response = requests.post(api_url, headers=headers, json=body, timeout=self.inference_timeout)
        response.raise_for_status()
        return response.json()

    def _on_success(self, start_time: float) -> None:
        self.last_used = time.time()
        self.inference_count += 1
        self.provider.total_uptime += time.time() - start_time

    def _emit_metric(
        self,
        caller: str,
        page_number: Optional[int],
        image_id: Optional[str],
        job_id: Optional[str],
        document_id: Optional[str],
        attempt_number: int,
        outcome: str,
        region_count: Optional[int],
        chars_count: Optional[int],
        failure_mode_head: Optional[str],
        latency_ms: int,
    ) -> None:
        """Best-effort insert to ``paddleocr_metrics``. Never raises."""
        try:
            from app.services.core.supabase_client import get_supabase_client
            sb = get_supabase_client()
            sb.client.table("paddleocr_metrics").insert({
                "caller": caller,
                "page_number": page_number,
                "image_id": image_id,
                "job_id": job_id,
                "document_id": document_id,
                "attempt_number": attempt_number,
                "outcome": outcome,
                "region_count": region_count,
                "chars_count": chars_count,
                "failure_mode_head": failure_mode_head,
                "latency_ms": latency_ms,
                "provider": self.provider.provider_name,
            }).execute()
        except Exception as metric_err:  # noqa: BLE001
            logger.debug("paddleocr_metrics insert failed (non-fatal): %s", metric_err)


def _coerce_image_to_png_bytes(image_input: Any, max_pixels: int) -> bytes:
    """Normalize path / bytes / PIL.Image to PNG bytes, downscaling to
    ``max_pixels`` if larger. Coordinate-safe: the Modal app reports pixel boxes
    on the image it receives and the parser normalizes by that image's size, so
    downscaling does not move the normalized boxes.
    """
    from io import BytesIO
    from PIL import Image

    if isinstance(image_input, str):
        # Detach from the file so the OS descriptor is released immediately.
        # A bare Image.open(path) holds the fd open until GC; under the parallel
        # per-crop OCR pass that exhausts the ulimit ([Errno 24] Too many open files).
        with Image.open(image_input) as _src:
            _src.load()
            img = _src.copy()
    elif isinstance(image_input, bytes):
        img = Image.open(BytesIO(image_input))  # BytesIO holds no OS fd
    else:
        img = image_input

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / float(w * h)) ** 0.5
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
