"""
Surya-2 structural-pass Inference Endpoint Manager.

Manages the HuggingFace Inference Endpoint lifecycle for Surya-2
(``datalab-to/surya-ocr-2``) — the single vision-language model that produces
the page layout + OCR text + figure boxes in one ``/v1/chat/completions`` call.
Surya is the pipeline's structural backbone: it replaced the YOLO (figure/region
boxes) + Chandra (text + boxes) two-model split and the ``merge_layout`` step.

Lifecycle (resume / warmup / scale-to-zero) mirrors
:class:`ChandraEndpointManager` so this manager drops into the same endpoint
registry + controller and the same per-job cost-control flow:
- Endpoint scaled to zero: $0/hour.
- Auto-wakes on first request; we resume + health-probe before trusting it.
- Force scale-to-zero at job terminal.

Two inference modes are exposed (the only two the pipeline uses):
- ``high_accuracy_bbox`` — full-page structural pass (default). Returns
  :class:`SuryaBlock` list with layout label + OCR'd HTML + 0..1 bbox per block.
- ``block`` — OCR a single cropped block image to text (per-image OCR).
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.services.pdf.surya_blocks import (
    BLOCK_PROMPT,
    HIGH_ACCURACY_BBOX_PROMPT,
    PROMPT_TYPE_BLOCK,
    PROMPT_TYPE_HIGH_ACCURACY_BBOX,
    SuryaBlock,
    html_to_text,
    parse_full_page_html,
)

try:
    from huggingface_hub import get_inference_endpoint
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logging.warning("huggingface_hub not available - Surya pause/resume disabled")

logger = logging.getLogger(__name__)


class SuryaResponseError(RuntimeError):
    """Raised when Surya returns output that cannot be parsed into blocks.

    The full-page parser refuses to silently return zero blocks for a
    non-empty response — that would corrupt the page's layout cache. Callers
    catch this and mark the page ``ocr_failed`` (retryable) rather than persist
    an empty layout.
    """


class SuryaEndpointManager:
    """Manages the HuggingFace Inference Endpoint lifecycle for Surya-2."""

    # Retry-with-jitter on parse failure. Surya is near-deterministic at temp 0;
    # if it freelances (prose, repeated tokens) the wider sampling on retry
    # breaks the sticky state. Same rationale as the Chandra retry ladder.
    _RETRY_TEMPERATURES: Tuple[float, ...] = (0.0, 0.2, 0.4)

    def __init__(
        self,
        endpoint_url: str,
        hf_token: str,
        endpoint_name: Optional[str] = None,
        namespace: Optional[str] = None,
        model_name: str = "surya-ocr-2",
        inference_timeout: int = 180,
        warmup_timeout: int = 300,
        max_resume_retries: int = 3,
        max_tokens: int = 8000,
        max_image_pixels: int = 2_000_000,
        enabled: bool = True,
    ):
        """
        Args:
            endpoint_url: Full URL of the Surya inference endpoint.
            hf_token: HuggingFace API token (write perms for pause/resume).
            endpoint_name: Endpoint name for pause/resume (e.g. 'surya').
            namespace: HuggingFace namespace/username (e.g. 'basiliskan').
            model_name: Model id sent in the chat request. Must match the name
                the served vLLM/llama.cpp endpoint advertises.
            inference_timeout: Per-call timeout (full-page passes can take ~30-90s
                on dense pages).
            warmup_timeout: Max warmup time after a cold start.
            max_resume_retries: Resume attempts on a paused/scaled-to-zero endpoint.
            max_tokens: Default completion cap for the full-page HTML pass.
            max_image_pixels: Page renders larger than this are downscaled before
                send (Surya's supported input range tops out ~2MP; bboxes are
                normalized so downscaling does not move coordinates).
            enabled: Master enable.
        """
        self.endpoint_url = endpoint_url
        self.hf_token = hf_token
        self.endpoint_name = endpoint_name
        self.namespace = namespace
        self.model_name = model_name
        self.inference_timeout = inference_timeout
        self.warmup_timeout = warmup_timeout
        self.max_resume_retries = max_resume_retries
        self.max_tokens = max_tokens
        self.max_image_pixels = max_image_pixels
        self.enabled = enabled

        self.last_used: Optional[float] = None
        self.last_resume_time: Optional[float] = None
        self.total_uptime: float = 0.0
        self.resume_count: int = 0
        self.inference_count: int = 0
        self.warmup_completed: bool = False

        self._endpoint = None
        self._can_pause_resume = HF_HUB_AVAILABLE and bool(endpoint_name) and bool(namespace)

        if self._can_pause_resume:
            logger.info(
                f"✅ Surya Endpoint Manager initialized with pause/resume: "
                f"endpoint={endpoint_name}, namespace={namespace}"
            )
        else:
            logger.warning(
                f"⚠️ Surya Endpoint Manager initialized WITHOUT pause/resume: "
                f"hf_hub_available={HF_HUB_AVAILABLE}, endpoint_name={endpoint_name}, "
                f"namespace={namespace}"
            )

    # ------------------------------------------------------------------ #
    # Endpoint lifecycle (mirrors ChandraEndpointManager)
    # ------------------------------------------------------------------ #

    def _get_endpoint(self):
        if not self._can_pause_resume:
            return None
        if self._endpoint is None:
            try:
                self._endpoint = get_inference_endpoint(
                    name=self.endpoint_name,
                    namespace=self.namespace,
                    token=self.hf_token,
                )
                logger.info(f"✅ Connected to Surya endpoint: {self.endpoint_name}")
            except Exception as e:
                logger.error(f"❌ Failed to get Surya endpoint instance: {e}")
                return None
        return self._endpoint

    def resume_if_needed(self) -> bool:
        """Resume the endpoint if paused/scaled-to-zero; health-probe if running."""
        if not self._can_pause_resume:
            return True

        endpoint = self._get_endpoint()
        if not endpoint:
            return False

        try:
            endpoint.fetch()

            if endpoint.status == "running":
                # Don't blindly trust "running" — the container can be up while
                # inference is wedged. A cheap /health probe catches this.
                if self._test_inference():
                    logger.info("✅ Surya endpoint running and healthy")
                    return True
                logger.warning("⚠️ Surya reports running but /health failed — re-warming")
                self.warmup_completed = False
                return self.warmup()

            if endpoint.status == "initializing":
                logger.info("⏳ Surya endpoint initializing, waiting...")
                return self._wait_for_running(endpoint)

            if endpoint.status in ("paused", "scaledToZero"):
                logger.info(f"🔄 Resuming Surya endpoint (status: {endpoint.status})...")
                from app.services.embeddings.hf_errors import is_hf_billing_error, HFBillingError
                for attempt in range(self.max_resume_retries):
                    try:
                        endpoint.resume().wait(timeout=90)
                        self.resume_count += 1
                        self.last_resume_time = time.time()
                        self.warmup_completed = False
                        logger.info(
                            f"✅ Surya endpoint resumed "
                            f"(attempt {attempt + 1}/{self.max_resume_retries})"
                        )
                        if not self.warmup():
                            logger.error("❌ Surya endpoint warmup failed")
                            return False
                        return True
                    except Exception as e:
                        if is_hf_billing_error(e):
                            logger.error(f"💳 HF billing error on Surya resume — aborting: {e}")
                            raise HFBillingError(
                                self.endpoint_name or "surya", self.namespace, original=e
                            ) from e
                        logger.warning(f"⚠️ Surya resume attempt {attempt + 1} failed: {e}")
                        if attempt < self.max_resume_retries - 1:
                            time.sleep(2 ** attempt)
                        else:
                            raise

            logger.error(f"❌ Surya endpoint in unexpected state: {endpoint.status}")
            return False

        except Exception as e:
            logger.error(f"❌ Failed to resume Surya endpoint: {e}")
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
                    logger.info(f"✅ Surya endpoint ready after {time.time() - start_time:.1f}s")
                    self.last_resume_time = time.time()
                    return self.warmup()
                if endpoint.status in ("failed", "error"):
                    logger.error(f"❌ Surya endpoint failed: {endpoint.status}")
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
                        f"❌ HuggingFace cannot allocate GPU capacity for Surya endpoint "
                        f"'{self.endpoint_name}' — no progress in {NO_PROGRESS_TIMEOUT}s "
                        f"(state={endpoint.status}, ready=0/{target}). HF infra issue, not a "
                        f"code bug; set min_replica=1 on the endpoint to avoid cold starts."
                    )
                    return False

                logger.info(
                    f"   ⏳ Surya still {endpoint.status} "
                    f"({time.time() - start_time:.0f}s, ready={current_ready}/{target})"
                )
                time.sleep(poll_interval)
            except Exception as e:
                logger.warning(f"⚠️ Error checking Surya status: {e}")
                time.sleep(poll_interval)

        logger.error(f"❌ Timeout waiting for Surya endpoint (max {self.warmup_timeout}s)")
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
                        logger.warning(f"⚠️ Surya endpoint is {endpoint.status} — resuming")
                        endpoint.resume().wait(timeout=90)
                        self.resume_count += 1
                        self.last_resume_time = time.time()
                except Exception as e:
                    logger.warning(f"⚠️ Failed to check/resume Surya during warmup: {e}")

        logger.info(f"🔥 Warming up Surya endpoint (max {self.warmup_timeout}s)...")
        start_time = time.time()
        attempt = 0
        base_delay, max_delay = 2, 15

        while (time.time() - start_time) < self.warmup_timeout:
            attempt += 1
            if self._test_inference():
                logger.info(
                    f"✅ Surya endpoint warmed up in {time.time() - start_time:.1f}s "
                    f"({attempt} attempts)"
                )
                self.warmup_completed = True
                return True
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            remaining = self.warmup_timeout - (time.time() - start_time)
            time.sleep(min(delay, remaining) if remaining > 0 else 0)

        logger.error(f"❌ Surya warmup failed after {time.time() - start_time:.1f}s")
        return False

    def _resolve_endpoint_url(self) -> str:
        """Return ``self.endpoint_url``, falling back to live HF SDK lookup when
        it is empty/not a URL (covers an unset ``SURYA_ENDPOINT_URL`` secret)."""
        url = (self.endpoint_url or "").strip()
        if url.startswith(("http://", "https://")):
            return url
        ep = self._get_endpoint()
        if ep is not None:
            try:
                ep.fetch()
                live_url = getattr(ep, "url", None)
                if live_url and live_url.startswith(("http://", "https://")):
                    self.endpoint_url = live_url
                    logger.info(f"   🔗 Surya endpoint URL resolved dynamically: {live_url}")
                    return live_url
            except Exception as e:
                logger.warning(f"   ⚠️ Surya live URL resolve failed: {e}")
        return ""

    def _test_inference(self) -> bool:
        """Liveness probe via GET /health (the chat route returns 400 when
        scaled-to-zero even though the container is up)."""
        try:
            url = self._resolve_endpoint_url()
            if not url:
                return False
            base = url.rstrip("/")
            health_url = base if base.endswith("/health") else base + "/health"
            response = requests.get(
                health_url, headers={"Authorization": f"Bearer {self.hf_token}"}, timeout=15
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"   Surya /health probe exception: {e}")
            return False

    def scale_to_zero(self) -> bool:
        """Force-drain to 0 replicas at job terminal — instant $0/h, URL stays
        alive so the next request auto-wakes it."""
        if not self._can_pause_resume:
            return False
        endpoint = self._get_endpoint()
        if not endpoint:
            return False
        try:
            endpoint.fetch()
            if endpoint.status in ("scaledToZero", "paused"):
                return True
            logger.info(f"📉 Scaling Surya endpoint to zero (was: {endpoint.status})")
            endpoint.scale_to_zero()
            if self.last_resume_time:
                self.total_uptime += time.time() - self.last_resume_time
            self.warmup_completed = False
            return True
        except Exception as e:
            logger.error(f"❌ Failed to scale Surya endpoint to zero: {e}")
            return False

    def mark_used(self):
        self.last_used = time.time()
        self.inference_count += 1

    def get_stats(self) -> Dict[str, Any]:
        return {
            "endpoint_name": self.endpoint_name,
            "resume_count": self.resume_count,
            "inference_count": self.inference_count,
            "total_uptime_seconds": self.total_uptime,
            "total_uptime_hours": self.total_uptime / 3600,
            "warmup_completed": self.warmup_completed,
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
        result = self._run(
            image_input,
            prompt_type=PROMPT_TYPE_BLOCK,
            caller=caller,
            page_number=None,
            job_id=job_id,
            document_id=document_id,
            image_id=image_id,
        )
        return result

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
        if self._can_pause_resume and not self.resume_if_needed():
            raise Exception("Failed to resume Surya endpoint")

        image_bytes = _coerce_image_to_png_bytes(image_input, self.max_image_pixels)

        last_error: Optional[SuryaResponseError] = None
        for attempt_idx, temperature in enumerate(self._RETRY_TEMPERATURES, start=1):
            start_time = time.time()
            try:
                raw_html, raw_response = self._do_single_inference(
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
                        f"⚠️ Surya attempt {attempt_idx}/{len(self._RETRY_TEMPERATURES)} "
                        f"parsed 0 blocks at temp={temperature}"
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
                    f"✅ Surya structural pass: {len(blocks)} blocks / "
                    f"{len(generated_text)} chars in {latency_ms}ms "
                    f"(attempt={attempt_idx}, temp={temperature})"
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

        Raises ``requests.HTTPError`` on HTTP failure.
        """
        import base64

        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        prompt = (
            HIGH_ACCURACY_BBOX_PROMPT
            if prompt_type == PROMPT_TYPE_HIGH_ACCURACY_BBOX
            else BLOCK_PROMPT
        )
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        api_url = self._resolve_endpoint_url().rstrip("/") + "/v1/chat/completions"
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
        self.total_uptime += time.time() - start_time

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
            }).execute()
        except Exception as metric_err:
            logger.debug(f"surya_ocr_metrics insert failed (non-fatal): {metric_err}")


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
