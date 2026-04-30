"""
Qwen Vision Model - HuggingFace Inference Endpoint Manager

Manages HuggingFace Inference Endpoint lifecycle for the configured Qwen
vision model (id resolved from `Settings.qwen_model`, default
`Qwen/Qwen3.6-35B-A3B-FP8`). Provides automatic pause/resume functionality
to control billing costs.

CRITICAL: Endpoint is paused by default (no billing). Only resumes when
vision analysis is needed, then auto-pauses after idle timeout to prevent
unnecessary billing.

Cost Control Strategy:
- Endpoint paused: $0/hour
- Endpoint running: ~$4.50/hour on A100 (changes when hardware/model swap)
- Auto-pause after 60s idle (configurable)
- Force-pause after batch processing
- Warmup required: ~60 seconds before first inference
- Update `app/config/ai_pricing.py:qwen3-vl-32b` if hardware changes
"""

import os
import time
import logging
import httpx
import base64
from typing import Optional, Dict, Any, List
from datetime import datetime

try:
    from huggingface_hub import get_inference_endpoint
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logging.warning("huggingface_hub not available - pause/resume features disabled")

logger = logging.getLogger(__name__)


class QwenEndpointManager:
    """
    Manages HuggingFace Inference Endpoint lifecycle for the configured Qwen vision model.
    
    Features:
    - Automatic resume before inference (start billing)
    - Automatic pause after idle timeout (stop billing)
    - Force pause after batch processing
    - Warmup handling (60s required before first inference)
    - Error handling and retry logic
    - Cost tracking and monitoring
    """
    
    def __init__(
        self,
        endpoint_url: str,
        endpoint_token: str,
        endpoint_name: str = "qwen3-6-35b-fp8",
        namespace: str = "basiliskan",
        auto_pause_timeout: int = 60,
        inference_timeout: int = 180,
        warmup_timeout: int = 60,
        max_resume_retries: int = 3,
        enabled: bool = True,
        model: Optional[str] = None,
    ):
        """
        Initialize Qwen endpoint manager.
        
        Args:
            endpoint_url: Full URL of the Qwen inference endpoint
            endpoint_token: HuggingFace API token (with write permissions)
            endpoint_name: Endpoint name for pause/resume (e.g., 'mh-qwen332binstruct')
            namespace: HuggingFace namespace/username (e.g., 'basiliskan')
            auto_pause_timeout: Seconds of idle time before auto-pause (default: 60)
            inference_timeout: Timeout for inference calls in seconds (default: 180)
            warmup_timeout: Warmup time in seconds (default: 60)
            max_resume_retries: Maximum retry attempts for resuming endpoint (default: 3)
            enabled: Enable/disable Qwen endpoint management (default: True)
        """
        self.endpoint_url = endpoint_url
        self.endpoint_token = endpoint_token
        self.endpoint_name = endpoint_name
        self.namespace = namespace
        # Resolved model id served by the endpoint. Required for the
        # OpenAI-compat chat-completions warmup test — vLLM 404s when the
        # `model` field doesn't match what `/v1/models` reports. Falls back
        # to settings.qwen_model when caller doesn't pass one.
        if model:
            self.model = model
        else:
            try:
                from app.config import get_settings
                self.model = get_settings().qwen_model
            except Exception:
                self.model = "Qwen/Qwen3.6-35B-A3B-FP8"
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
        
        if not self._can_pause_resume:
            logger.warning("⚠️ Qwen endpoint pause/resume not available - endpoint will run continuously")
        
        logger.info(f"✅ Qwen Endpoint Manager initialized: {endpoint_name}")
    
    def _get_endpoint(self):
        """Get or create endpoint instance for pause/resume operations."""
        if not self._can_pause_resume:
            return None

        if self._endpoint is None:
            try:
                self._endpoint = get_inference_endpoint(
                    name=self.endpoint_name,
                    namespace=self.namespace,
                    token=self.endpoint_token
                )
                logger.info(f"✅ Connected to Qwen endpoint: {self.endpoint_name}")
            except Exception as e:
                logger.error(f"❌ Failed to get Qwen endpoint: {e}")
                return None

        return self._endpoint

    def _refresh_url_from_endpoint(self) -> None:
        """Pull the live URL from the HF endpoint object so we never use a stale hardcoded URL.

        ⚠️ HF SDK note: depending on endpoint state, `endpoint.url` may return
        the bare host (e.g. `gbz6krk3i2is85b0.us-east-1.aws.endpoints.huggingface.cloud`)
        without an `https://` scheme. We MUST normalize it here, otherwise
        every consumer (httpx health check, OpenAI client, etc.) raises
        "Request URL is missing an 'http://' or 'https://' protocol" or
        APIConnectionError. (2026-04-10 fix)
        """
        endpoint = self._get_endpoint()
        if not endpoint:
            return
        try:
            live_url = getattr(endpoint, "url", None)
            if live_url:
                if not live_url.startswith(("http://", "https://")):
                    live_url = "https://" + live_url.lstrip("/")
                if live_url != self.endpoint_url:
                    logger.info(f"🔗 Qwen endpoint URL updated: {live_url}")
                    self.endpoint_url = live_url
        except Exception as e:
            logger.debug(f"Could not refresh endpoint URL: {e}")
    
    def is_running(self) -> bool:
        """
        Quick non-blocking check: is the endpoint currently accepting requests?

        Only does a single HF status fetch (~1s HTTP call). Does NOT trigger
        resume or warmup — use resume_if_needed() for that.

        Returns:
            True if status == "running", False for any other state or error.
        """
        if not self._can_pause_resume:
            return True  # no management available — assume running

        endpoint = self._get_endpoint()
        if not endpoint:
            return False

        try:
            endpoint.fetch()
            status = endpoint.status
            logger.debug(f"Qwen endpoint status: {status}")
            return status == "running"
        except Exception as e:
            logger.warning(f"Could not check Qwen endpoint status: {e}")
            return False

    def resume_if_needed(self) -> bool:
        """
        Resume endpoint if it's paused or wait if initializing.

        Handles all HuggingFace endpoint states:
        - running: Ready to use
        - paused/scaledToZero: Needs resume
        - initializing: Needs polling until ready

        CRITICAL: This starts billing! Only call when vision analysis is needed.

        Returns:
            True if endpoint is running or successfully resumed, False if failed
        """
        if not self._can_pause_resume:
            logger.info("Pause/resume not available - assuming endpoint is running")
            return True

        endpoint = self._get_endpoint()
        if not endpoint:
            return False

        try:
            # Fetch current status
            endpoint.fetch()

            if endpoint.status == "running":
                logger.info("✅ Qwen endpoint already running")
                self._refresh_url_from_endpoint()
                return True

            # Handle "initializing" state - poll until ready
            if endpoint.status == "initializing":
                logger.info(f"⏳ Qwen endpoint initializing, waiting for it to be ready...")
                return self._wait_for_running(endpoint)

            if endpoint.status in ["paused", "scaledToZero"]:
                logger.info(f"🔄 Resuming Qwen endpoint (status: {endpoint.status})...")

                # Resume with retries - FIXED: Added .wait() to block until running
                from app.services.embeddings.hf_errors import is_hf_billing_error, HFBillingError
                for attempt in range(self.max_resume_retries):
                    try:
                        endpoint.resume().wait(timeout=90)  # P2-3: 90s cap
                        self.resume_count += 1
                        self.last_resume_time = time.time()
                        self.warmup_completed = False  # Reset warmup flag
                        logger.info(f"✅ Qwen endpoint resumed (attempt {attempt + 1}/{self.max_resume_retries})")
                        self._refresh_url_from_endpoint()

                        # Smart polling-based warmup (calls self.warmup())
                        if not self.warmup():
                            logger.error("❌ Qwen endpoint warmup failed after polling")
                            return False

                        return True
                    except Exception as e:
                        # FAST-FAIL on HF billing errors — retries can't fix this.
                        if is_hf_billing_error(e):
                            logger.error(
                                f"💳 HF billing error on Qwen resume — aborting all retries: {e}"
                            )
                            raise HFBillingError(self.endpoint_name, self.namespace, original=e) from e
                        logger.error(f"❌ Resume attempt {attempt + 1} failed: {e}")
                        if attempt < self.max_resume_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            return False

            logger.warning(f"⚠️ Unexpected Qwen endpoint status: {endpoint.status}")
            return False

        except Exception as e:
            logger.error(f"❌ Failed to check/resume Qwen endpoint: {e}")
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

        # 2026-04-11: HF "no GPU capacity" detector. If readyReplica hasn't
        # advanced for NO_PROGRESS_TIMEOUT seconds, HuggingFace can't allocate
        # GPUs for us right now (cold-start stall, region capacity exhaustion,
        # etc.). Fail loud immediately instead of burning the full max_wait
        # window. The caller (resume_if_needed → rag_routes warmup orchestrator)
        # propagates this False as a job failure with a clear, actionable
        # error message.
        last_ready_replica = 0
        last_progress_time = time.time()
        NO_PROGRESS_TIMEOUT = 90  # seconds with no allocation progress = HF capacity issue

        while (time.time() - start_time) < max_wait:
            try:
                endpoint.fetch()

                if endpoint.status == "running":
                    elapsed = time.time() - start_time
                    logger.info(f"✅ Qwen endpoint ready after {elapsed:.1f}s")
                    self.last_resume_time = time.time()
                    self._refresh_url_from_endpoint()

                    # Warmup after becoming ready
                    if not self.warmup():
                        logger.error("❌ Qwen endpoint warmup failed")
                        return False
                    return True

                if endpoint.status in ["failed", "error"]:
                    logger.error(f"❌ Endpoint failed: {endpoint.status}")
                    return False

                # No-progress detection — see comment above the loop.
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
                        f"❌ HuggingFace cannot allocate GPU capacity for Qwen endpoint "
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
        """Smart polling-based warmup - stops as soon as endpoint responds.

        2026-04-11 rework: pre-fetch the live URL from HF SDK AND inline
        the resume logic (fetch + resume().wait()) directly. We can NOT
        call `self.resume_if_needed()` here because that method calls
        `self.warmup()` at the end — mutual recursion = infinite loop.
        """
        if self.warmup_completed:
            logger.info("✅ Qwen endpoint already warmed up")
            return True

        # Step 1: pull the live URL from HF SDK so self.endpoint_url
        # isn't empty when _test_inference runs.
        self._refresh_url_from_endpoint()

        # Step 2: inline resume — fetch the endpoint and if it's in any
        # state other than 'running', wait it into 'running'. Wraps HF
        # SDK calls in try/except so warmup still proceeds to the test
        # loop even if the SDK path fails.
        if self._can_pause_resume:
            endpoint = self._get_endpoint()
            if endpoint:
                try:
                    endpoint.fetch()
                    current_status = getattr(endpoint, 'status', None)
                    logger.info(f"🔍 Qwen endpoint status at warmup: {current_status}")
                    if current_status in ("paused", "scaledToZero", "pending"):
                        logger.warning(
                            f"⚠️ Qwen endpoint is {current_status} — "
                            f"calling endpoint.resume() and waiting up to 300s"
                        )
                        try:
                            endpoint.resume().wait(timeout=90)  # P2-3: 90s cap
                            self.resume_count += 1
                            self.last_resume_time = time.time()
                            logger.info("✅ Qwen endpoint resume() returned")
                        except Exception as resume_err:
                            logger.warning(
                                f"⚠️ Qwen endpoint resume() raised: {resume_err}"
                            )
                        # Refresh URL after resume in case HF reassigned it.
                        self._refresh_url_from_endpoint()
                    elif current_status == "initializing":
                        logger.info(
                            f"⏳ Qwen endpoint is initializing — "
                            f"the test loop below will poll until it's ready"
                        )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to fetch/resume Qwen endpoint: {e}")

        logger.info(f"🔥 Warming up Qwen endpoint (max {self.warmup_timeout}s)...")
        start_time = time.time()
        attempt = 0

        while (time.time() - start_time) < self.warmup_timeout:
            attempt += 1
            if self._test_inference():
                elapsed = time.time() - start_time
                logger.info(f"✅ Qwen warmed up in {elapsed:.1f}s ({attempt} attempts)")
                self.warmup_completed = True
                return True

            delay = min(2 * (2 ** (attempt - 1)), 15)
            remaining = self.warmup_timeout - (time.time() - start_time)
            if remaining > delay:
                logger.info(f"   ⏳ Attempt {attempt} - retrying in {delay}s...")
                time.sleep(delay)
            else:
                time.sleep(min(2, remaining))

        logger.error(f"❌ Qwen warmup timed out after {self.warmup_timeout}s")
        return False
    
    def _test_inference(self) -> bool:
        """Quick test if Qwen can handle requests."""
        try:
            import requests
            # If endpoint_url isn't a real URL (empty / unset GH Secret),
            # try to populate it from the live HF endpoint object before
            # giving up. We always know endpoint_name + namespace, so the
            # SDK can resolve the URL deterministically.
            if not self.endpoint_url or not self.endpoint_url.startswith(("http://", "https://")):
                self._refresh_url_from_endpoint()
            if not self.endpoint_url or not self.endpoint_url.startswith(("http://", "https://")):
                logger.warning(
                    f"   Qwen warmup test skipped: endpoint_url not configured "
                    f"and HF SDK lookup returned no URL (got {self.endpoint_url!r})"
                )
                return False

            base_url = self.endpoint_url.rstrip('/')
            if base_url.endswith('/v1'):
                url = f"{base_url}/chat/completions"
            else:
                url = f"{base_url}/v1/chat/completions"

            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.endpoint_token}", "Content-Type": "application/json"},
                json={"model": self.model, "messages": [{"role": "user", "content": "OK"}], "max_tokens": 2},
                timeout=15
            )
            if response.status_code != 200:
                logger.warning(f"   Qwen warmup test failed: HTTP {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"   Qwen warmup test exception: {e}")
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
        """
        Force-drain endpoint to 0 replicas IMMEDIATELY at job terminal.

        Uses `endpoint.scale_to_zero()` — replica killed in seconds, $0/h
        immediately. Endpoint URL stays alive so the next inference
        request auto-wakes it (no explicit resume needed).

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

            if current_status in ("scaledToZero", "paused"):
                logger.info(f"Qwen endpoint already at zero ({current_status}) — no-op")
                return True

            logger.info(f"📉 Scaling Qwen endpoint to zero NOW (was: {current_status}) — instant $0/h, URL stays alive")
            endpoint.scale_to_zero()

            if self.last_resume_time:
                uptime = time.time() - self.last_resume_time
                self.total_uptime += uptime

            self.warmup_completed = False
            logger.info(f"✅ Qwen endpoint scaled to zero — billing stopped, auto-wakes on next request")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to scale Qwen endpoint to zero: {e}")
            return False

    async def analyze_image(
        self,
        image_base64: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> Optional[str]:
        """
        Analyze image using Qwen vision model.

        Args:
            image_base64: Base64-encoded image
            prompt: Text prompt for analysis
            max_tokens: Maximum tokens to generate (default: 1024)
            temperature: Temperature for generation (default: 0.1)

        Returns:
            Analysis result as string, or None if failed
        """
        if not self.enabled:
            logger.warning("Qwen endpoint is disabled")
            return None

        # Resume endpoint if needed
        if not self.resume_if_needed():
            logger.error("Failed to resume Qwen endpoint")
            return None

        try:
            async with httpx.AsyncClient(timeout=self.inference_timeout) as client:
                response = await client.post(
                    self.endpoint_url,
                    headers={
                        "Authorization": f"Bearer {self.endpoint_token}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "inputs": {
                            "text": prompt,
                            "image": image_base64
                        },
                        "parameters": {
                            "max_tokens": max_tokens,
                            "temperature": temperature
                        }
                    }
                )

                if response.status_code != 200:
                    logger.error(f"Qwen endpoint error {response.status_code}: {response.text}")
                    return None

                result = response.json()
                self.inference_count += 1
                self.last_used = time.time()

                return result.get("generated_text") or result.get("output")

        except Exception as e:
            logger.error(f"❌ Qwen inference failed: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for cost tracking."""
        return {
            "endpoint_name": self.endpoint_name,
            "resume_count": self.resume_count,
            "pause_count": self.pause_count,
            "inference_count": self.inference_count,
            "total_uptime_hours": self.total_uptime / 3600,
            "estimated_cost_usd": (self.total_uptime / 3600) * 1.20,  # ~$1.20/hour
            "last_used": datetime.fromtimestamp(self.last_used).isoformat() if self.last_used else None,
            "warmup_completed": self.warmup_completed
        }

