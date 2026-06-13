"""
Surya-2 structural-pass endpoint manager.

Drives the Surya-2 (``datalab-to/surya-ocr-2``) vision-language model — the
single model that produces the page layout + OCR text + figure boxes in one
``/v1/chat/completions`` call. Surya is the pipeline's structural backbone: it
replaced the YOLO (figure/region boxes) + Chandra (text + boxes) split and the
``merge_layout`` step.

The manager is **provider-agnostic**. The inference / retry / parse / metrics
logic lives here; the endpoint **lifecycle** (warmup, resume, scale-to-zero) is
delegated to an :class:`~app.services.pdf.endpoint_providers.EndpointProvider`:

* ``SURYA_PROVIDER=huggingface`` → :class:`HuggingFaceEndpointProvider` (SDK
  resume / poll-for-GPU / scale-to-zero).
* ``SURYA_PROVIDER=modal`` → :class:`ModalEndpointProvider` (Modal owns
  autoscaling; warmup is a health probe, scale-to-zero is Modal's idle clock).

Switching GPU hosts is one env var — no code change — because both providers
serve the identical OpenAI-compatible vLLM contract.

Two inference modes (the only two the pipeline uses):
* ``high_accuracy_bbox`` — full-page structural pass (default). Returns
  :class:`SuryaBlock` list with layout label + OCR'd HTML + 0..1 bbox per block.
* ``block`` — OCR a single cropped block image to text (per-image OCR).
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import requests

from app.services.pdf.endpoint_providers import (
    EndpointProvider,
    build_endpoint_provider,
)
from app.services.pdf.surya_blocks import (
    BLOCK_PROMPT,
    HIGH_ACCURACY_BBOX_PROMPT,
    PROMPT_TYPE_BLOCK,
    PROMPT_TYPE_HIGH_ACCURACY_BBOX,
    SuryaBlock,
    html_to_text,
    parse_full_page_html,
)

logger = logging.getLogger(__name__)


class SuryaResponseError(RuntimeError):
    """Raised when Surya returns output that cannot be parsed into blocks.

    The full-page parser refuses to silently return zero blocks for a
    non-empty response — that would corrupt the page's layout cache. Callers
    catch this and mark the page ``ocr_failed`` (retryable) rather than persist
    an empty layout.
    """


class SuryaEndpointManager:
    """Manages the Surya-2 structural pass over a pluggable endpoint provider."""

    # Retry-with-jitter on parse failure. Surya is near-deterministic at temp 0;
    # if it freelances (prose, repeated tokens) the wider sampling on retry
    # breaks the sticky state. Same rationale as the Chandra retry ladder.
    _RETRY_TEMPERATURES: Tuple[float, ...] = (0.0, 0.2, 0.4)

    def __init__(
        self,
        provider: EndpointProvider,
        model_name: str = "surya-ocr-2",
        inference_timeout: int = 180,
        warmup_timeout: int = 300,
        max_tokens: int = 8000,
        max_image_pixels: int = 2_000_000,
        enabled: bool = True,
    ):
        """
        Args:
            provider: Endpoint lifecycle + auth strategy (HF or Modal).
            model_name: Model id sent in the chat request. Must match what the
                served vLLM endpoint advertises (``--served-model-name``).
            inference_timeout: Per-call timeout (full-page passes can take ~30-90s
                on dense pages).
            warmup_timeout: Max warmup time after a cold start.
            max_tokens: Default completion cap for the full-page HTML pass.
            max_image_pixels: Page renders larger than this are downscaled before
                send (Surya's supported input range tops out ~2MP; bboxes are
                normalized 0-1000 so downscaling does not move coordinates).
            enabled: Master enable.
        """
        self.provider = provider
        self.model_name = model_name
        self.inference_timeout = inference_timeout
        self.warmup_timeout = warmup_timeout
        self.max_tokens = max_tokens
        self.max_image_pixels = max_image_pixels
        self.enabled = enabled

        self.last_used: Optional[float] = None
        self.inference_count: int = 0

        logger.info(
            "✅ Surya manager initialized (provider=%s, model=%s)",
            provider.provider_name, model_name,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SuryaEndpointManager":
        """Build a manager (+ provider) from ``Settings.get_surya_config()``."""
        provider = build_endpoint_provider(config, label="surya")
        return cls(
            provider=provider,
            model_name=config.get("model_name", "surya-ocr-2"),
            inference_timeout=config.get("inference_timeout", 180),
            warmup_timeout=config.get("warmup_timeout", 300),
            max_tokens=config.get("max_tokens", 8000),
            max_image_pixels=config.get("max_image_pixels", 2_000_000),
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
        this name because the warmup orchestrator + endpoint controller call it
        on every manager uniformly (alongside the SLIG manager's own probe)."""
        return self.provider.health_check()

    # The controller resets warmup_completed after a scale-down and reads it to
    # decide whether to re-warm — proxy it to the provider's flag.
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
        """Live base URL from the active provider (HF dynamic resolution / Modal
        static URL). Read by health-check wiring that expects a manager attr."""
        return self.provider.resolve_base_url()

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    # Lifecycle counters live on the provider; expose them for stats consumers
    # (ocr_service.get_endpoint_stats, admin dashboards) that read them off the
    # manager.
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
    ) -> Dict[str, Any]:
        """Full-page structural pass (``high_accuracy_bbox``).

        Returns::

            {
              "blocks": List[SuryaBlock],   # 0..1 bbox, layout label + OCR HTML
              "generated_text": str,        # reading-order plain text
              "raw_html": str,              # raw model output
              "attempts_made": int,
              "final_temperature": float,
            }

        Raises:
            SuryaResponseError: non-empty output that parsed to zero blocks,
                after all retries — the page is marked ocr_failed (retryable).
            requests.HTTPError: HTTP/endpoint-health failures fail fast (no retry).
        """
        return self._run(
            image_input,
            prompt_type=PROMPT_TYPE_HIGH_ACCURACY_BBOX,
            caller=caller,
            page_number=page_number,
            job_id=job_id,
            document_id=document_id,
        )

    def run_block_ocr(
        self,
        image_input: Any,
        caller: str = "block_ocr",
        image_id: Optional[str] = None,
        job_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """OCR a single cropped block image to text (``block`` mode).

        Returns ``{"generated_text": str, "raw_html": str, "attempts_made": int}``.
        Used for per-image OCR on text-bearing product crops (higher resolution
        on small text than the page-level pass).
        """
        return self._run(
            image_input,
            prompt_type=PROMPT_TYPE_BLOCK,
            caller=caller,
            page_number=None,
            job_id=job_id,
            document_id=document_id,
            image_id=image_id,
        )

    def _run(
        self,
        image_input: Any,
        prompt_type: str,
        caller: str,
        page_number: Optional[int],
        job_id: Optional[str],
        document_id: Optional[str],
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise Exception("Surya endpoint is disabled")
        if not self.provider.resume_if_needed():
            raise Exception(f"Failed to resume Surya endpoint (provider={self.provider.provider_name})")

        image_bytes = _coerce_image_to_png_bytes(image_input, self.max_image_pixels)

        last_error: Optional[SuryaResponseError] = None
        for attempt_idx, temperature in enumerate(self._RETRY_TEMPERATURES, start=1):
            start_time = time.time()
            try:
                raw_html, _raw_response = self._do_single_inference(
                    image_bytes=image_bytes,
                    prompt_type=prompt_type,
                    temperature=temperature,
                )
            except requests.HTTPError as he:
                self._emit_metric(
                    caller=caller, page_number=page_number, image_id=image_id,
                    job_id=job_id, document_id=document_id, attempt_number=attempt_idx,
                    temperature=temperature, outcome="failed_http_error",
                    blocks_count=None, chars_count=None,
                    failure_mode_head=str(he)[:200],
                    latency_ms=int((time.time() - start_time) * 1000),
                )
                raise  # endpoint-health issue — retry won't help

            latency_ms = int((time.time() - start_time) * 1000)

            if prompt_type == PROMPT_TYPE_HIGH_ACCURACY_BBOX:
                blocks = parse_full_page_html(raw_html)
                if not blocks and raw_html.strip():
                    # Non-empty output that yielded no blocks = malformed; retry.
                    last_error = SuryaResponseError(
                        f"Surya full-page output parsed to 0 blocks. head={raw_html[:300]!r}"
                    )
                    self._emit_metric(
                        caller=caller, page_number=page_number, image_id=image_id,
                        job_id=job_id, document_id=document_id, attempt_number=attempt_idx,
                        temperature=temperature, outcome="failed_no_blocks",
                        blocks_count=0, chars_count=len(raw_html),
                        failure_mode_head=raw_html[:200], latency_ms=latency_ms,
                    )
                    logger.warning(
                        "⚠️ Surya attempt %d/%d parsed 0 blocks at temp=%s",
                        attempt_idx, len(self._RETRY_TEMPERATURES), temperature,
                    )
                    continue

                generated_text = "\n\n".join(b.text for b in blocks if b.text).strip()
                self._on_success(start_time)
                self._emit_metric(
                    caller=caller, page_number=page_number, image_id=image_id,
                    job_id=job_id, document_id=document_id, attempt_number=attempt_idx,
                    temperature=temperature,
                    outcome="success" if attempt_idx == 1 else "success_after_retry",
                    blocks_count=len(blocks), chars_count=len(generated_text),
                    failure_mode_head=None, latency_ms=latency_ms,
                )
                logger.info(
                    "✅ Surya structural pass: %d blocks / %d chars in %dms "
                    "(attempt=%d, temp=%s, provider=%s)",
                    len(blocks), len(generated_text), latency_ms, attempt_idx,
                    temperature, self.provider.provider_name,
                )
                return {
                    "blocks": blocks,
                    "generated_text": generated_text,
                    "raw_html": raw_html,
                    "attempts_made": attempt_idx,
                    "final_temperature": temperature,
                }

            # block mode — plain HTML for one crop.
            text = html_to_text(raw_html)
            self._on_success(start_time)
            self._emit_metric(
                caller=caller, page_number=page_number, image_id=image_id,
                job_id=job_id, document_id=document_id, attempt_number=attempt_idx,
                temperature=temperature,
                outcome="success" if attempt_idx == 1 else "success_after_retry",
                blocks_count=1 if text else 0, chars_count=len(text),
                failure_mode_head=None, latency_ms=latency_ms,
            )
            return {
                "generated_text": text,
                "raw_html": raw_html,
                "attempts_made": attempt_idx,
                "final_temperature": temperature,
            }

        # Retries exhausted on a parse failure.
        assert last_error is not None
        raise last_error

    def _do_single_inference(
        self,
        image_bytes: bytes,
        prompt_type: str,
        temperature: float,
    ) -> Tuple[str, Dict[str, Any]]:
        """Single Surya call. Returns ``(raw_text, raw_response_json)``.

        Base URL + bearer auth come from the active provider, so this is
        identical whether the endpoint lives on HuggingFace or Modal.

        Raises ``requests.HTTPError`` on HTTP failure.
        """
        import base64

        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        prompt = (
            HIGH_ACCURACY_BBOX_PROMPT
            if prompt_type == PROMPT_TYPE_HIGH_ACCURACY_BBOX
            else BLOCK_PROMPT
        )
        base_url = self.provider.resolve_base_url().rstrip("/")
        if not base_url:
            raise requests.HTTPError(
                f"Surya endpoint URL unresolved (provider={self.provider.provider_name})"
            )
        api_url = base_url + "/v1/chat/completions"
        headers = {"Content-Type": "application/json", **self.provider.auth_header()}
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "stream": False,
            "temperature": temperature,
            "top_p": 0.1 if temperature == 0.0 else 0.95,
            "max_tokens": self.max_tokens if prompt_type == PROMPT_TYPE_HIGH_ACCURACY_BBOX else 2048,
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=self.inference_timeout)
        response.raise_for_status()
        result = response.json()

        choices = result.get("choices") or []
        if not choices:
            raise SuryaResponseError(f"Surya response has no choices: {str(result)[:300]}")
        content = (choices[0].get("message") or {}).get("content") or ""
        return content, result

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
        temperature: float,
        outcome: str,
        blocks_count: Optional[int],
        chars_count: Optional[int],
        failure_mode_head: Optional[str],
        latency_ms: int,
    ) -> None:
        """Best-effort insert to ``surya_ocr_metrics``. Never raises."""
        try:
            from app.services.core.supabase_client import get_supabase_client
            sb = get_supabase_client()
            sb.client.table("surya_ocr_metrics").insert({
                "caller": caller,
                "page_number": page_number,
                "image_id": image_id,
                "job_id": job_id,
                "document_id": document_id,
                "attempt_number": attempt_number,
                "temperature": temperature,
                "outcome": outcome,
                "blocks_count": blocks_count,
                "chars_count": chars_count,
                "failure_mode_head": failure_mode_head,
                "latency_ms": latency_ms,
                "provider": self.provider.provider_name,
            }).execute()
        except Exception as metric_err:  # noqa: BLE001
            logger.debug("surya_ocr_metrics insert failed (non-fatal): %s", metric_err)


def _coerce_image_to_png_bytes(image_input: Any, max_pixels: int) -> bytes:
    """Normalize path / bytes / PIL.Image to PNG bytes, downscaling to
    ``max_pixels`` if larger (Surya's supported input range tops out ~2MP).

    Downscaling is coordinate-safe: Surya emits bboxes normalized 0-1000, so a
    smaller input image yields the same normalized boxes.
    """
    from io import BytesIO
    from PIL import Image

    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, bytes):
        img = Image.open(BytesIO(image_input))
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
