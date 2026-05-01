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
                from app.services.embeddings.hf_errors import is_hf_billing_error, HFBillingError
                for attempt in range(self.max_resume_retries):
                    try:
                        endpoint.resume().wait(timeout=90)  # P2-3: 90s cap
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
                        # FAST-FAIL on HF billing errors — retries can't fix this.
                        if is_hf_billing_error(e):
                            logger.error(
                                f"💳 HF billing error on Chandra resume — aborting all retries: {e}"
                            )
                            raise HFBillingError("mh-chandra", self.namespace, original=e) from e
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

        # HF "no GPU capacity" detector — see qwen_endpoint_manager
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
                        endpoint.resume().wait(timeout=90)  # P2-3: 90s cap
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

    def _resolve_endpoint_url(self) -> str:
        """Return self.endpoint_url, falling back to dynamic HF SDK lookup
        when it's empty / not a real URL.

        Same pattern as YOLO/Qwen managers — covers the case where
        deploy.yml writes an empty `Environment=CHANDRA_ENDPOINT_URL=` line
        (because the GH Secret is unset). Reads the live URL via
        `get_inference_endpoint(name, namespace).url` and caches it.
        """
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
                    logger.info(
                        f"   🔗 Chandra endpoint URL resolved dynamically: {live_url}"
                    )
                    return live_url
            except Exception as e:
                logger.warning(f"   ⚠️ Chandra live URL resolve failed: {e}")
        return ""

    def _test_inference(self) -> bool:
        """
        Test if Chandra endpoint is alive via its /health route.

        Use GET /health, not POST /v1/chat/completions: the chat route returns
        HTTP 400 "paused, ask a maintainer to restart it" when the LLM is
        scaled to zero even though the container is up, so probing it would
        miss paused-but-alive endpoints and stall warmup.

        Returns:
            True if endpoint responds with 200, False otherwise.
        """
        try:
            url = self._resolve_endpoint_url()
            if not url:
                logger.debug("   Chandra _test_inference: no resolvable endpoint URL")
                return False
            base = url.rstrip('/')
            health_url = base if base.endswith('/health') else base + '/health'

            response = requests.get(
                health_url,
                headers={"Authorization": f"Bearer {self.hf_token}"},
                timeout=15
            )
            if response.status_code == 200:
                return True
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

    def scale_to_zero(self) -> bool:
        """Force-drain endpoint to 0 replicas IMMEDIATELY at job terminal.

        Uses `endpoint.scale_to_zero()` — replica killed in seconds, $0/h
        immediately. Endpoint URL stays alive so the next inference
        request auto-wakes it (no explicit resume needed).
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

            if current_status in ("scaledToZero", "paused"):
                logger.info(f"Chandra endpoint already at zero ({current_status}) — no-op")
                return True

            logger.info(f"📉 Scaling Chandra endpoint to zero NOW (was: {current_status}) — instant $0/h, URL stays alive")
            endpoint.scale_to_zero()

            if self.last_resume_time:
                uptime = time.time() - self.last_resume_time
                self.total_uptime += uptime

            self.warmup_completed = False
            logger.info(f"✅ Chandra endpoint scaled to zero — billing stopped, auto-wakes on next request")
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

    # Retry policy (audit fix #1, #7): sample at temperature 0 first so OCR is
    # deterministic; if v2 freelances ("The image is..." prose) at temp=0, jitter
    # the next attempts to break the sticky-prose state. ~50% prose rate at temp=0
    # observed in production drops to <5% after retry-with-jitter.
    _RETRY_TEMPERATURES = (0.0, 0.1, 0.2)

    def run_inference(
        self,
        image_input: Any,
        parameters: Optional[Dict] = None,
        prompt: str = DEFAULT_OCR_PROMPT,
        caller: str = "ad_hoc",
        image_id: Optional[str] = None,
        job_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run OCR inference with retry-with-jitter on parse failures.

        Returns structured output (see _do_single_inference for full schema)
        plus retry metadata:
          - attempts_made: int (1..3)
          - final_temperature: float

        On every attempt — success OR fail — emits one row to chandra_ocr_metrics
        for observability.

        Raises:
            ChandraResponseError: only after all retries exhaust on parse failures.
            requests.HTTPError: HTTP errors (5xx, timeouts) fail fast — no retry,
                because they indicate endpoint health issues that retry can't fix.
        """
        if not self.enabled:
            raise Exception("Chandra endpoint is disabled")

        if self._can_pause_resume:
            if not self.resume_if_needed():
                raise Exception("Failed to resume Chandra endpoint")

        # Pre-encode image once; reused across retries.
        image_bytes = _coerce_image_to_bytes(image_input)

        last_parse_error: Optional[ChandraResponseError] = None
        last_failure_head: Optional[str] = None
        last_failure_outcome: Optional[str] = None

        for attempt_idx, temperature in enumerate(self._RETRY_TEMPERATURES, start=1):
            start_time = time.time()
            try:
                result, blocks, generated_text, extraction_path = self._do_single_inference(
                    image_bytes=image_bytes,
                    prompt=prompt,
                    parameters=parameters,
                    temperature=temperature,
                )
            except ChandraResponseError as cre:
                latency_ms = int((time.time() - start_time) * 1000)
                last_parse_error = cre
                err_str = str(cre)
                last_failure_head = err_str[:200]
                # Classify failure for the metrics row.
                if "is not valid JSON" in err_str and "cleaned_head=" in err_str:
                    head_lower = err_str.lower()
                    if "the image" in head_lower or "this image" in head_lower:
                        last_failure_outcome = "failed_prose"
                    else:
                        last_failure_outcome = "failed_malformed_json"
                else:
                    last_failure_outcome = "failed_other"

                self._emit_metric(
                    image_id=image_id, job_id=job_id, document_id=document_id,
                    caller=caller, attempt_number=attempt_idx, temperature=temperature,
                    outcome=last_failure_outcome,
                    blocks_count=None, chars_count=None,
                    failure_mode_head=last_failure_head, latency_ms=latency_ms,
                )
                logger.warning(
                    f"⚠️ Chandra v2 attempt {attempt_idx}/{len(self._RETRY_TEMPERATURES)} "
                    f"failed at temp={temperature}: {err_str[:120]}"
                )
                continue
            except requests.HTTPError as he:
                latency_ms = int((time.time() - start_time) * 1000)
                self._emit_metric(
                    image_id=image_id, job_id=job_id, document_id=document_id,
                    caller=caller, attempt_number=attempt_idx, temperature=temperature,
                    outcome="failed_http_error",
                    blocks_count=None, chars_count=None,
                    failure_mode_head=str(he)[:200], latency_ms=latency_ms,
                )
                # Fail fast — HTTP errors indicate endpoint health, retry won't help.
                raise

            # Success path.
            inference_time = time.time() - start_time
            self.last_used = time.time()
            self.inference_count += 1
            self.total_uptime += inference_time

            outcome = "success" if attempt_idx == 1 else "success_after_retry"
            self._emit_metric(
                image_id=image_id, job_id=job_id, document_id=document_id,
                caller=caller, attempt_number=attempt_idx, temperature=temperature,
                outcome=outcome, blocks_count=len(blocks), chars_count=len(generated_text),
                failure_mode_head=None, latency_ms=int(inference_time * 1000),
            )
            logger.info(
                f"✅ Chandra v2 OCR ok: {len(blocks)} fragments / {len(generated_text)} chars in "
                f"{inference_time:.2f}s (path={extraction_path}, attempt={attempt_idx}, temp={temperature})"
            )
            return {
                "generated_text": generated_text,
                "blocks": blocks,
                "extraction_path": extraction_path,
                "confidence": 0.85,
                "raw_response": result,
                "attempts_made": attempt_idx,
                "final_temperature": temperature,
            }

        # All retries exhausted — raise the last parse error so caller can handle.
        assert last_parse_error is not None
        raise last_parse_error

    def _do_single_inference(
        self,
        image_bytes: bytes,
        prompt: str,
        parameters: Optional[Dict],
        temperature: float,
    ) -> tuple:
        """Single Chandra v2 call. Returns (raw_result, blocks, generated_text, extraction_path).

        Raises ChandraResponseError on parse failure (caller handles retry).
        Raises requests.HTTPError on HTTP failure (caller fails fast).
        """
        import base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        api_url = self.endpoint_url.rstrip('/') + '/v1/chat/completions'
        payload = {
            "model": CHANDRA_V2_MODEL_ID,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an OCR engine. Your only job is to extract "
                        "text fragments from images and emit a JSON array of "
                        '{"text", "x", "y", "w", "h"} entries. '
                        "Never describe images. Never write prose. "
                        "Output JSON only — no markdown, no commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                },
            ],
            "stream": False,
            "temperature": temperature,
            "max_tokens": parameters.get('max_tokens', 4000) if parameters else 4000,
        }

        logger.info(f"Calling Chandra v2 endpoint: {api_url} (temp={temperature})")
        response = requests.post(api_url, headers=headers, json=payload, timeout=self.inference_timeout)
        response.raise_for_status()
        result = response.json()

        parsed = _extract_bbox_json_from_response(result)
        return result, parsed["blocks"], parsed["generated_text"], parsed["extraction_path"]

    def _emit_metric(
        self,
        image_id: Optional[str],
        job_id: Optional[str],
        document_id: Optional[str],
        caller: str,
        attempt_number: int,
        temperature: float,
        outcome: str,
        blocks_count: Optional[int],
        chars_count: Optional[int],
        failure_mode_head: Optional[str],
        latency_ms: int,
    ) -> None:
        """Best-effort insert to chandra_ocr_metrics. Never raises."""
        try:
            from app.services.core.supabase_client import get_supabase_client
            sb = get_supabase_client()
            sb.client.table("chandra_ocr_metrics").insert({
                "image_id": image_id,
                "job_id": job_id,
                "document_id": document_id,
                "caller": caller,
                "attempt_number": attempt_number,
                "temperature": temperature,
                "outcome": outcome,
                "blocks_count": blocks_count,
                "chars_count": chars_count,
                "failure_mode_head": failure_mode_head,
                "latency_ms": latency_ms,
            }).execute()
        except Exception as metric_err:
            # Telemetry must never break the OCR path.
            logger.debug(f"chandra_ocr_metrics insert failed (non-fatal): {metric_err}")


def _coerce_image_to_bytes(image_input: Any) -> bytes:
    """Normalize string-path / bytes / PIL.Image inputs to PNG bytes."""
    if isinstance(image_input, str):
        with open(image_input, 'rb') as f:
            return f.read()
    if isinstance(image_input, bytes):
        return image_input
    from io import BytesIO
    buffer = BytesIO()
    image_input.save(buffer, format='PNG')
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Strict bbox-JSON response parser
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\s*```\s*$")


def _find_balanced_close(s: str, open_pos: int) -> int:
    """Find the position of the bracket that closes the array/object at open_pos.

    String-state-aware: brackets inside JSON strings don't count. Returns -1 if no
    balanced close is found within the input.

    Replaces the old `rfind(']')`/`rfind('}')` trim, which mis-trimmed v2 outputs
    like `[{"text":"","x":0,"y":0,"w":1000,"h":1000}\\']'` (the rfind picked the `]`
    inside the trailing `\\']'` garbage rather than the real closing bracket at
    char 51). The walker correctly stops at the real close.
    """
    if open_pos < 0 or open_pos >= len(s):
        return -1
    open_c = s[open_pos]
    close_c = "]" if open_c == "[" else "}" if open_c == "{" else ""
    if not close_c:
        return -1

    depth = 0
    in_string = False
    escape_next = False
    for i in range(open_pos, len(s)):
        c = s[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\" and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _strip_fences_and_junk(raw: str) -> str:
    """Strip markdown fences and trim trailing garbage after a complete JSON value.

    v2 quirks this handles:
      - ` ```json [...] ``` ` markdown fences
      - leading prose before the JSON: `Here is the result: [...]`
      - trailing escaped-quote junk: `[...]\\']`  (model truncates near max_tokens)
      - trailing bare-quote junk:    `[...]"]`

    Returns the substring from the first `[`/`{` to the matching balanced close.
    String-state-aware (won't mis-count brackets inside `"text"` values).
    """
    s = raw.strip()
    s = _FENCE_RE.sub("", s)
    s = _FENCE_END_RE.sub("", s)
    s = s.strip()

    # Find the first opening bracket of an array or object.
    candidates = [i for i in (s.find("["), s.find("{")) if i >= 0]
    if not candidates:
        return s
    first_open = min(candidates)
    if first_open > 0:
        s = s[first_open:]

    balanced_close = _find_balanced_close(s, 0)
    if balanced_close >= 0:
        return s[: balanced_close + 1]

    # No balanced close — leave as-is so json.loads can produce a precise error.
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

