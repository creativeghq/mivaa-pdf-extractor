"""
PaddleOCR-VL structural-pass endpoint manager.

Drives the PaddleOCR-VL pipeline hosted on Modal — the two-stage document parser
(PP-DocLayoutV2 detector + 0.9B VLM) that produces the page layout + OCR text +
figure boxes per page, with tight RT-DETR crop boxes and a dedicated reading order.

The manager speaks the Modal app's custom ``/parse`` contract (NOT OpenAI chat):

    POST /parse  {"image_b64": "...", "mode": "page"|"block"}
      → {"regions":[{"bbox":[x0,y0,x1,y1] px,"label","content","order"}],
         "width","height"}

Lifecycle (warmup / scale-to-zero / health) is delegated to a
:class:`~app.services.pdf.endpoint_providers.ModalEndpointProvider` — Modal owns
autoscaling, so warmup is a ``/health`` probe and scale-to-zero is Modal's idle
clock. The inference / parse / metrics logic lives here.

Two modes (the only two the pipeline uses):
* ``page`` — full-page structural pass (default). Returns :class:`PaddleRegion`
  list with layout label + OCR content + 0..1 bbox + reading order.
* ``block`` — OCR a single cropped block image to text (per-image OCR).
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.services.pdf.endpoint_providers import EndpointProvider, build_endpoint_provider
from app.services.pdf.paddleocr_pipeline import (
    PaddleRegion,
    parse_parse_response,
    regions_to_reading_text,
)

logger = logging.getLogger(__name__)


def _fire_and_forget_async(coro) -> None:
    """Run an async coroutine to completion off the calling thread.

    ``run_structural_pass`` / ``run_block_ocr`` are SYNC (they use ``time.sleep``)
    and are dispatched via ``asyncio.to_thread`` from the async stages — so the
    calling thread has no running event loop, and we must not block it on a DB
    write. This spins up a throwaway daemon thread with its own event loop, runs
    the coroutine there, and returns immediately. Best-effort: any failure is
    swallowed so GPU-cost logging can never break the structural pass.
    """
    import asyncio
    import threading

    def _runner() -> None:
        try:
            asyncio.new_event_loop().run_until_complete(coro)
        except Exception as bg_err:  # noqa: BLE001
            logger.debug("PaddleOCR cost-log thread failed (non-fatal): %s", bg_err)

    try:
        threading.Thread(target=_runner, daemon=True).start()
    except Exception as spawn_err:  # noqa: BLE001
        logger.debug("PaddleOCR cost-log thread spawn failed (non-fatal): %s", spawn_err)


def _log_paddleocr_gpu_cost(
    task: str,
    latency_ms: int,
    job_id: Optional[str],
    image_id: Optional[str],
    product_id: Optional[str],
) -> None:
    """Log the GPU-seconds cost of a successful PaddleOCR call to ai_usage_logs.

    PaddleOCR-VL runs on a Modal GPU endpoint — billing is per GPU-second
    (time-based), NOT per token. model="paddleocr-vl" routes to PADDLEOCR_PRICING
    (time-based) in ai_pricing.calculate_time_based_cost. Best-effort — never
    raises into the structural pass.
    """
    try:
        from app.services.core.ai_call_logger import AICallLogger

        _fire_and_forget_async(
            AICallLogger().log_time_based_call(
                task=task,
                model="paddleocr-vl",
                latency_ms=latency_ms,
                confidence_score=0.85,
                confidence_breakdown={},
                job_id=job_id,
                image_id=image_id,
                product_id=product_id,
            )
        )
    except Exception as log_err:  # noqa: BLE001
        logger.debug("PaddleOCR GPU cost log failed (non-fatal): %s", log_err)


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

        Returns::

            {
              "regions": List[PaddleRegion],   # 0..1 bbox, label + content + order
              "generated_text": str,           # reading-order plain text
              "raw": dict,                      # raw /parse response
              "attempts_made": int,
            }

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
            regions: List[PaddleRegion] = parse_parse_response(payload)
            if not regions and (payload.get("regions") is None):
                # Endpoint returned a malformed payload (no regions key) — retry.
                last_error = PaddleOCRResponseError(
                    f"PaddleOCR /parse returned no 'regions' key. head={str(payload)[:300]!r}"
                )
                self._emit_metric(
                    caller=caller, page_number=page_number, image_id=None,
                    job_id=job_id, document_id=document_id, attempt_number=attempt_idx,
                    outcome="failed_no_regions", region_count=0, chars_count=0,
                    failure_mode_head=str(payload)[:200], latency_ms=latency_ms,
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

    def run_block_ocr(
        self,
        image_input: Any,
        caller: str = "block_ocr",
        image_id: Optional[str] = None,
        job_id: Optional[str] = None,
        document_id: Optional[str] = None,
        product_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """OCR a single cropped block image to text (``block`` mode).

        Returns ``{"generated_text": str, "raw": dict, "attempts_made": int}``.
        Used for per-image OCR on text-bearing product crops.
        """
        if not self.enabled:
            raise Exception("PaddleOCR endpoint is disabled")
        if not self.provider.resume_if_needed():
            raise Exception("Failed to resume PaddleOCR endpoint")

        image_bytes = _coerce_image_to_png_bytes(image_input, self.max_image_pixels)
        start_time = time.time()
        payload = self._do_parse(image_bytes, mode="block")
        latency_ms = int((time.time() - start_time) * 1000)
        text = str(payload.get("text") or "")
        if not text:
            regions = parse_parse_response(payload)
            text = regions_to_reading_text(regions)
        self._on_success(start_time)
        self._emit_metric(
            caller=caller, page_number=None, image_id=image_id,
            job_id=job_id, document_id=document_id, attempt_number=1,
            outcome="success", region_count=1 if text else 0, chars_count=len(text),
            failure_mode_head=None, latency_ms=latency_ms,
        )
        # GPU-seconds cost → ai_usage_logs (rolls up into total_ai_cost_usd).
        _log_paddleocr_gpu_cost(
            task="pdf_ocr_paddleocr",
            latency_ms=latency_ms,
            job_id=job_id,
            image_id=image_id,
            product_id=product_id,
        )
        return {"generated_text": text, "raw": payload, "attempts_made": 1}

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
