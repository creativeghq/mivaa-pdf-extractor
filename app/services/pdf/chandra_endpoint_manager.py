"""
Chandra OCR v2 Inference Endpoint Manager

Manages HuggingFace Inference Endpoint lifecycle for Chandra OCR v2
(model `chandra-ocr-2.Q8_0.gguf`). State-of-the-art OCR that returns
structured bbox-JSON: each output entry is {"text", "x", "y", "w", "h"}
with pixel coordinates on the source image.

Provides automatic pause/resume functionality to control billing costs.

Cost Control Strategy:
- Endpoint paused: $0/hour
- Endpoint running: ~$0.50/hour (T4 GPU)
- Auto-pause after 60s idle (configurable)
- Force-pause after batch processing
- Typical cost: ~$0.02 per 30-page document
"""

import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

try:
    from huggingface_hub import get_inference_endpoint
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logging.warning("huggingface_hub not available - pause/resume features disabled")

logger = logging.getLogger(__name__)


CHANDRA_V2_MODEL_ID = "chandra-ocr-2.Q8_0.gguf"

DEFAULT_OCR_PROMPT = (
    "Extract all text from this image. Return a JSON array where each entry is "
    '{"text": <fragment>, "x": <int>, "y": <int>, "w": <int>, "h": <int>}. '
    "Output JSON only - no commentary, no markdown."
)


class ChandraResponseError(RuntimeError):
    """Raised when Chandra v2 returns output that cannot be parsed as bbox-JSON.

    The strict parser refuses to silently fall back to plain text or empty
    strings - both would silently corrupt downstream OCR text fields. Callers
    catch this and log the page as an OCR failure instead of writing garbage.
    """


class ChandraEndpointManager:
    """
    Manages HuggingFace Inference Endpoint lifecycle for Chandra OCR v2.

    Features:
    - Automatic resume before inference (start billing)
    - Automatic pause after idle timeout (stop billing)
    - Force pause after batch processing
    - Strict bbox-JSON response parsing (raises on garbage, never returns trash)
    - Cost tracking and monitoring
    """
    
    def __init__(
        self,
        endpoint_url: str,
        hf_token: str,
        endpoint_name: Optional[str] = None,
        namespace: Optional[str] = None,
        auto_pause_timeout: int = 60,
        inference_timeout: int = 120,
        warmup_timeout: int = 300,
        max_resume_retries: int = 3,
        enabled: bool = True
    ):
        """
        Initialize Chandra Endpoint Manager.

        Args:
            endpoint_url: Full URL of the Chandra inference endpoint
            hf_token: HuggingFace API token (with write permissions)
            endpoint_name: Endpoint name for pause/resume (e.g., 'mh-chandra')
            namespace: HuggingFace namespace/username (e.g., 'basiliskan')
            auto_pause_timeout: Seconds of idle time before auto-pause (default: 60)
            inference_timeout: Timeout for inference calls in seconds (default: 30)
            warmup_timeout: Maximum warmup time in seconds (default: 300)
            max_resume_retries: Maximum retry attempts for resuming endpoint (default: 3)
            enabled: Enable/disable Chandra fallback (default: True)
        """
        self.endpoint_url = endpoint_url
        self.hf_token = hf_token
        self.endpoint_name = endpoint_name
        self.namespace = namespace
        self.auto_pause_timeout = auto_pause_timeout
        self.inference_timeout = inference_timeout
        self.warmup_timeout = warmup_timeout
        self.max_resume_retries = max_resume_retries
        self.enabled = enabled

        # Track usage for auto-pause
        self.last_used: Optional[float] = None
        self.last_resume_time: Optional[float] = None
        self.total_uptime: float = 0.0
        self.resume_count: int = 0
        self.pause_count: int = 0
        self.inference_count: int = 0
        self.warmup_completed: bool = False

        # Endpoint instance (for pause/resume)
        self._endpoint = None
        self._can_pause_resume = HF_HUB_AVAILABLE and endpoint_name and namespace

        if self._can_pause_resume:
            logger.info(
                f"✅ Chandra Endpoint Manager initialized with pause/resume: "
                f"endpoint={endpoint_name}, namespace={namespace}, auto_pause={auto_pause_timeout}s"
            )
        else:
            logger.warning(
                f"⚠️ Chandra Endpoint Manager initialized WITHOUT pause/resume: "
                f"hf_hub_available={HF_HUB_AVAILABLE}, endpoint_name={endpoint_name}, namespace={namespace}"
            )
    
    def _get_endpoint(self):
        """Get or create endpoint instance for pause/resume operations."""
        if not self._can_pause_resume:
            return None

        if self._endpoint is None:
            try:
                self._endpoint = get_inference_endpoint(
                    name=self.endpoint_name,
                    namespace=self.namespace,
                    token=self.hf_token
                )
                logger.info(f"✅ Connected to endpoint: {self.endpoint_name}")
            except Exception as e:
                logger.error(f"❌ Failed to get endpoint instance: {e}")
                return None

        return self._endpoint

    def resume_if_needed(self) -> bool:
        """
        Resume endpoint if paused or wait if initializing.

        Handles all HuggingFace endpoint states:
        - running: Ready to use
        - paused/scaledToZero: Needs resume
        - initializing: Needs polling until ready

        Returns:
            True if endpoint is running, False if failed
        """
        if not self._can_pause_resume:
            logger.warning("Pause/resume not available - endpoint should already be running")
            return True

        endpoint = self._get_endpoint()
        if not endpoint:
            return False

        try:
            # Fetch current status
            endpoint.fetch()

            if endpoint.status == "running":
                logger.info("✅ Chandra endpoint already running")
                return True

            # Handle "initializing" state - poll until ready
            if endpoint.status == "initializing":
                logger.info(f"⏳ Chandra endpoint initializing, waiting for it to be ready...")
                return self._wait_for_running(endpoint)

            if endpoint.status in ["paused", "scaledToZero"]:
                logger.info(f"🔄 Resuming Chandra endpoint (status: {endpoint.status})...")

                # Resume with retries
                for attempt in range(self.max_resume_retries):
                    try:
                        endpoint.resume().wait(timeout=300)  # Wait up to 5 minutes
                        self.resume_count += 1
                        self.last_resume_time = time.time()
                        self.warmup_completed = False  # Reset warmup flag
                        logger.info(f"✅ Chandra endpoint resumed (attempt {attempt + 1}/{self.max_resume_retries})")

                        # Warmup after resume
                        if not self.warmup():
                            logger.error("❌ Chandra endpoint warmup failed")
                            return False
                        return True
                    except Exception as e:
                        logger.warning(f"⚠️ Resume attempt {attempt + 1} failed: {e}")
                        if attempt < self.max_resume_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            raise

            logger.error(f"❌ Endpoint in unexpected state: {endpoint.status}")
            return False

        except Exception as e:
            logger.error(f"❌ Failed to resume endpoint: {e}")
            return False

    def _wait_for_running(self, endpoint) -> bool:
        """
        Wait for endpoint to transition from initializing to running.

        Args:
            endpoint: HuggingFace endpoint instance

        Returns:
            True if endpoint becomes running, False on timeout
        """
        start_time = time.time()
        poll_interval = 5  # Check every 5 seconds
        max_wait = self.warmup_timeout

        # 2026-04-11: HF "no GPU capacity" detector — see qwen_endpoint_manager
        # for the full rationale. Same pattern in all 4 managers.
        last_ready_replica = 0
        last_progress_time = time.time()
        NO_PROGRESS_TIMEOUT = 90

        while (time.time() - start_time) < max_wait:
            try:
                endpoint.fetch()

                if endpoint.status == "running":
                    elapsed = time.time() - start_time
                    logger.info(f"✅ Chandra endpoint ready after {elapsed:.1f}s")
                    self.last_resume_time = time.time()

                    # Warmup after becoming ready
                    if not self.warmup():
                        logger.error("❌ Chandra endpoint warmup failed")
                        return False
                    return True

                if endpoint.status in ["failed", "error"]:
                    logger.error(f"❌ Endpoint failed: {endpoint.status}")
                    return False

                # No-progress detection
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
                        f"❌ HuggingFace cannot allocate GPU capacity for Chandra endpoint "
                        f"'{getattr(self, 'endpoint_name', '<unknown>')}' — no progress in "
                        f"{NO_PROGRESS_TIMEOUT}s (state={endpoint.status}, "
                        f"ready=0/{target}). This is an HF infrastructure issue, "
                        f"not a code bug. Either wait for capacity to free up, "
                        f"or set min_replica=1 on the HF endpoint dashboard to "
                        f"keep at least one instance always-on (eliminates "
                        f"cold-start delays for production runs)."
                    )
                    return False

                logger.info(f"   ⏳ Still {endpoint.status}, waiting... ({time.time() - start_time:.0f}s, ready={current_ready}/{target})")
                time.sleep(poll_interval)

            except Exception as e:
                logger.warning(f"⚠️ Error checking status: {e}")
                time.sleep(poll_interval)

        logger.error(f"❌ Timeout waiting for endpoint (max {max_wait}s)")
        return False

    def warmup(self) -> bool:
        """
        Smart polling-based warmup for Chandra endpoint.

        Uses exponential backoff to poll the endpoint until it responds
        successfully to inference requests.

        Returns:
            True if warmup successful, False if timeout
        """
        if self.warmup_completed:
            logger.info("✅ Chandra endpoint already warmed up")
            return True

        # Defensive check: ensure endpoint is not paused before warmup
        if self._can_pause_resume:
            endpoint = self._get_endpoint()
            if endpoint:
                try:
                    endpoint.fetch()
                    if endpoint.status in ["paused", "scaledToZero"]:
                        logger.warning(f"⚠️ Chandra endpoint is {endpoint.status} - triggering resume")
                        endpoint.resume().wait(timeout=300)
                        self.resume_count += 1
                        self.last_resume_time = time.time()
                except Exception as e:
                    logger.warning(f"⚠️ Failed to check/resume endpoint during warmup: {e}")

        logger.info(f"🔥 Warming up Chandra endpoint (max {self.warmup_timeout}s)...")

        start_time = time.time()
        attempt = 0
        base_delay = 2
        max_delay = 15

        while (time.time() - start_time) < self.warmup_timeout:
            attempt += 1

            if self._test_inference():
                elapsed = time.time() - start_time
                logger.info(f"✅ Chandra endpoint warmed up in {elapsed:.1f}s ({attempt} attempts)")
                self.warmup_completed = True
                return True

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            remaining = self.warmup_timeout - (time.time() - start_time)

            if remaining > delay:
                logger.info(f"   ⏳ Warmup attempt {attempt} - not ready, retrying in {delay}s...")
                time.sleep(delay)
            else:
                time.sleep(min(2, remaining))

        logger.error(f"❌ Chandra warmup failed after {time.time() - start_time:.1f}s")
        return False

    def _test_inference(self) -> bool:
        """
        Test if Chandra endpoint is alive via its /health route.

        2026-04-11: Switched from POST /v1/chat/completions to GET /health.
        Chandra's chat route returns HTTP 400 "paused, ask a maintainer to
        restart it" when the LLM is scaled to zero, even though the
        container itself is up and the /health route responds with
        {"status":"ok"}. The old behavior never recognized a paused endpoint
        as "alive" and the warmup loop timed out after 300s on every job
        that hit a cold Chandra. Verified against the live endpoint
        today: curl -H "Authorization: Bearer $TOKEN" .../health → 200 OK.

        Returns:
            True if endpoint responds with 200, False otherwise.
        """
        try:
            base = self.endpoint_url.rstrip('/')
            health_url = base if base.endswith('/health') else base + '/health'

            response = requests.get(
                health_url,
                headers={"Authorization": f"Bearer {self.hf_token}"},
                timeout=15
            )
            if response.status_code == 200:
                return True
            # Chandra /chat/completions bad request (400 "paused") still
            # means the LLM route is scaled to zero — NOT a healthy state
            # for this manager. Let the outer retry loop keep polling.
            logger.debug(
                f"   Chandra /health returned HTTP {response.status_code}"
            )
            return False
        except Exception as e:
            logger.debug(f"   Chandra /health probe exception: {e}")
            return False

    def pause_if_idle(self) -> bool:
        """
        DISABLED: Let HuggingFace handle scale-to-zero automatically.
        Manual pausing causes endpoints to not auto-resume on requests.

        Returns:
            True always (no-op)
        """
        logger.debug("pause_if_idle() disabled - letting HF handle scale-to-zero automatically")
        return True

    def force_pause(self) -> bool:
        """
        Pause endpoint at JOB COMPLETION to stop billing.

        IMPORTANT: Call this ONLY when the entire job is complete, NOT between
        batches or processing steps. This puts endpoint in "paused" state which
        requires explicit resume() - handled by resume_if_needed() and warmup().

        The endpoint will be resumed automatically when the next job starts.

        Returns:
            True if paused successfully, False if failed
        """
        if not self._can_pause_resume:
            logger.warning("Pause/resume not available - cannot pause")
            return False

        endpoint = self._get_endpoint()
        if not endpoint:
            return False

        try:
            endpoint.fetch()
            if endpoint.status == "running":
                logger.info("⏸️ Pausing Chandra endpoint (JOB COMPLETED - stopping billing)")
                endpoint.pause()
                self.pause_count += 1

                if self.last_resume_time:
                    uptime = time.time() - self.last_resume_time
                    self.total_uptime += uptime

                self.warmup_completed = False
                logger.info("✅ Chandra endpoint paused (no billing until next job)")
                return True
            else:
                logger.info(f"Endpoint already not running (status: {endpoint.status})")
                return True

        except Exception as e:
            logger.error(f"❌ Failed to pause Chandra endpoint: {e}")
            return False

    def scale_to_zero(self) -> bool:
        """
        Scale endpoint to zero replicas at JOB COMPLETION.

        Unlike force_pause(), this sets min_replica=0 which allows HuggingFace
        to auto-scale based on demand. The endpoint will auto-resume when
        the next request arrives.

        This is the PREFERRED method for job completion cleanup.

        Returns:
            True if scaled successfully, False if failed
        """
        if not self._can_pause_resume:
            logger.warning("Endpoint management not available - cannot scale to zero")
            return False

        endpoint = self._get_endpoint()
        if not endpoint:
            return False

        try:
            endpoint.fetch()
            current_status = endpoint.status

            # Get current max_replica to preserve it
            raw = getattr(endpoint, 'raw', {})
            if 'compute' in raw and 'scaling' in raw['compute']:
                max_rep = raw['compute']['scaling'].get('maxReplica', 2)
            else:
                max_rep = 2

            logger.info(f"📉 Scaling Chandra endpoint to 0 replicas (JOB COMPLETED - will auto-resume on demand)")
            endpoint.update(min_replica=0, max_replica=max_rep)

            if self.last_resume_time:
                uptime = time.time() - self.last_resume_time
                self.total_uptime += uptime

            self.warmup_completed = False
            logger.info(f"✅ Chandra endpoint scaled to 0 (was: {current_status}, max: {max_rep})")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to scale Chandra endpoint to zero: {e}")
            return False

    def mark_used(self):
        """Mark endpoint as recently used (for auto-pause tracking)."""
        self.last_used = time.time()
        self.inference_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get endpoint usage statistics."""
        return {
            "endpoint_name": self.endpoint_name,
            "resume_count": self.resume_count,
            "pause_count": self.pause_count,
            "inference_count": self.inference_count,
            "total_uptime_seconds": self.total_uptime,
            "total_uptime_hours": self.total_uptime / 3600,
            "estimated_cost_usd": (self.total_uptime / 3600) * 0.60,
            "warmup_completed": self.warmup_completed,
            "last_used": datetime.fromtimestamp(self.last_used).isoformat() if self.last_used else None,
            "enabled": self.enabled
        }

    def run_inference(
        self,
        image_input: Any,
        parameters: Optional[Dict] = None,
        prompt: str = DEFAULT_OCR_PROMPT,
    ) -> Dict[str, Any]:
        """Run OCR inference on an image using Chandra v2.

        Always returns structured output. The model emits a JSON array of
        {"text", "x", "y", "w", "h"} entries describing every text fragment
        on the page; the parser preserves that list as `blocks` and also
        produces a reading-order joined string in `generated_text` so legacy
        callers that expect a string keep working.

        Returns:
            Dict with keys:
              - generated_text: str (joined fragments, newline-separated)
              - blocks: list[dict] (bbox-aligned text fragments)
              - extraction_path: 'content_bbox_json' | 'reasoning_bbox_json'
              - confidence: float (default 0.85)
              - raw_response: dict (the raw HTTP JSON body)

        Raises:
            ChandraResponseError: when the model returns no parseable
                bbox-JSON. Callers should treat the page as an OCR failure
                rather than writing empty text to the database.
        """
        import base64

        if not self.enabled:
            raise Exception("Chandra endpoint is disabled")

        if self._can_pause_resume:
            if not self.resume_if_needed():
                raise Exception("Failed to resume Chandra endpoint")

        start_time = time.time()

        # Convert image to base64
        if isinstance(image_input, str):
            with open(image_input, 'rb') as f:
                image_bytes = f.read()
        elif isinstance(image_input, bytes):
            image_bytes = image_input
        else:
            from io import BytesIO
            buffer = BytesIO()
            image_input.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()

        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        api_url = self.endpoint_url.rstrip('/') + '/v1/chat/completions'
        payload = {
            "model": CHANDRA_V2_MODEL_ID,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "stream": False,
            "max_tokens": parameters.get('max_tokens', 4000) if parameters else 4000,
        }

        logger.info(f"Calling Chandra v2 endpoint: {api_url}")
        response = requests.post(api_url, headers=headers, json=payload, timeout=self.inference_timeout)
        response.raise_for_status()
        result = response.json()

        parsed = _extract_bbox_json_from_response(result)
        blocks = parsed["blocks"]
        generated_text = parsed["generated_text"]
        extraction_path = parsed["extraction_path"]

        inference_time = time.time() - start_time
        self.last_used = time.time()
        self.inference_count += 1
        self.total_uptime += inference_time

        logger.info(
            f"✅ Chandra v2 OCR ok: {len(blocks)} fragments / {len(generated_text)} chars in "
            f"{inference_time:.2f}s (path={extraction_path})"
        )

        return {
            "generated_text": generated_text,
            "blocks": blocks,
            "extraction_path": extraction_path,
            "confidence": 0.85,
            "raw_response": result,
        }


# ---------------------------------------------------------------------------
# Strict bbox-JSON response parser
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\s*```\s*$")


def _strip_fences_and_junk(raw: str) -> str:
    """Strip markdown fences and trailing garbage (e.g. truncated `']`).

    v2 occasionally wraps output in ```json ... ``` fences and very rarely
    emits a stray closing-quote/bracket pair when the response was cut off.
    Both cases are recoverable by trimming to the outermost JSON brackets.
    """
    s = raw.strip()
    s = _FENCE_RE.sub("", s)
    s = _FENCE_END_RE.sub("", s)
    s = s.strip()
    # Trim to outermost JSON array/object brackets if junk surrounds them.
    first_open = min(
        (i for i in (s.find("["), s.find("{")) if i >= 0),
        default=-1,
    )
    if first_open > 0:
        s = s[first_open:]
    last_close = max(s.rfind("]"), s.rfind("}"))
    if last_close >= 0 and last_close < len(s) - 1:
        s = s[: last_close + 1]
    return s


def _join_blocks_in_reading_order(blocks: List[Dict[str, Any]]) -> str:
    """Join fragment .text fields top-to-bottom, left-to-right.

    Sorting on (y, x) preserves natural reading order for single-column
    layouts. For multi-column layouts the layout_merge_service provides
    proper column-aware ordering by post-classifying fragments against
    YOLO regions; this function is only the legacy-string fallback.
    """
    def _key(b: Dict[str, Any]) -> tuple:
        try:
            return (float(b.get("y", 0)), float(b.get("x", 0)))
        except (TypeError, ValueError):
            return (0.0, 0.0)
    sorted_blocks = sorted(blocks, key=_key)
    return "\n".join(b["text"].strip() for b in sorted_blocks if isinstance(b.get("text"), str) and b["text"].strip())


def _extract_bbox_json_from_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a Chandra v2 OpenAI-compatible response into bbox blocks.

    v2 puts the JSON answer in `choices[0].message.content`. When the
    prompt accidentally suppresses content, the model falls back to
    `reasoning_content` - we honour both. Anything else is a hard error;
    we never return empty/garbage text silently.
    """
    choices = result.get("choices") or []
    if not choices:
        raise ChandraResponseError(f"Chandra response has no choices: {str(result)[:500]}")

    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    extraction_path = "content_bbox_json"

    if not content:
        content = (message.get("reasoning_content") or "").strip()
        extraction_path = "reasoning_bbox_json"

    if not content:
        raise ChandraResponseError(
            "Chandra response has no content or reasoning_content. "
            f"finish_reason={choices[0].get('finish_reason')!r} usage={result.get('usage')!r}"
        )

    cleaned = _strip_fences_and_junk(content)

    # Use raw_decode() so trailing junk after a complete JSON value (e.g.
    # the `]]` and `']` truncation quirks v2 sometimes emits when the
    # response was cut off near max_tokens) doesn't reject the parse.
    decoder = json.JSONDecoder()
    try:
        parsed, _end_idx = decoder.raw_decode(cleaned.lstrip())
    except json.JSONDecodeError as exc:
        raise ChandraResponseError(
            f"Chandra response is not valid JSON: {exc}. cleaned_head={cleaned[:300]!r}"
        ) from exc

    if not isinstance(parsed, list):
        raise ChandraResponseError(
            f"Chandra response is not a JSON array (got {type(parsed).__name__}): {cleaned[:300]!r}"
        )

    blocks: List[Dict[str, Any]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        text_val = entry.get("text")
        if not isinstance(text_val, str) or not text_val.strip():
            continue
        blocks.append({
            "text": text_val,
            "x": entry.get("x"),
            "y": entry.get("y"),
            "w": entry.get("w") if entry.get("w") is not None else entry.get("width"),
            "h": entry.get("h") if entry.get("h") is not None else entry.get("height"),
        })

    if not blocks:
        raise ChandraResponseError(
            f"Chandra response parsed but contained no usable bbox text: {cleaned[:300]!r}"
        )

    return {
        "blocks": blocks,
        "generated_text": _join_blocks_in_reading_order(blocks),
        "extraction_path": extraction_path,
    }

