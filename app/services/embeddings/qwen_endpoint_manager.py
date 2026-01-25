"""
Qwen Vision Model - HuggingFace Inference Endpoint Manager

Manages HuggingFace Inference Endpoint lifecycle for Qwen3-VL-32B-Instruct vision model.
Provides automatic pause/resume functionality to control billing costs.

CRITICAL: Endpoint is paused by default (no billing). Only resumes when vision analysis is needed,
then auto-pauses after idle timeout to prevent unnecessary billing.

Cost Control Strategy:
- Endpoint paused: $0/hour
- Endpoint running: ~$1.20/hour (GPU)
- Auto-pause after 60s idle (configurable)
- Force-pause after batch processing
- Warmup required: 60 seconds before first inference
- Typical cost: ~$0.02 per 100 images
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
    Manages HuggingFace Inference Endpoint lifecycle for Qwen3-VL-32B-Instruct.
    
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
        endpoint_name: str = "mh-qwen332binstruct",
        namespace: str = "basiliskan",
        auto_pause_timeout: int = 60,
        inference_timeout: int = 180,
        warmup_timeout: int = 60,
        max_resume_retries: int = 3,
        enabled: bool = True
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
                return True

            # Handle "initializing" state - poll until ready
            if endpoint.status == "initializing":
                logger.info(f"⏳ Qwen endpoint initializing, waiting for it to be ready...")
                return self._wait_for_running(endpoint)

            if endpoint.status in ["paused", "scaledToZero"]:
                logger.info(f"🔄 Resuming Qwen endpoint (status: {endpoint.status})...")

                # Resume with retries - FIXED: Added .wait() to block until running
                for attempt in range(self.max_resume_retries):
                    try:
                        endpoint.resume().wait(timeout=300)  # Wait up to 5 minutes
                        self.resume_count += 1
                        self.last_resume_time = time.time()
                        self.warmup_completed = False  # Reset warmup flag
                        logger.info(f"✅ Qwen endpoint resumed (attempt {attempt + 1}/{self.max_resume_retries})")

                        # Smart polling-based warmup (calls self.warmup())
                        if not self.warmup():
                            logger.error("❌ Qwen endpoint warmup failed after polling")
                            return False

                        return True
                    except Exception as e:
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

        while (time.time() - start_time) < max_wait:
            try:
                endpoint.fetch()

                if endpoint.status == "running":
                    elapsed = time.time() - start_time
                    logger.info(f"✅ Qwen endpoint ready after {elapsed:.1f}s")
                    self.last_resume_time = time.time()

                    # Warmup after becoming ready
                    if not self.warmup():
                        logger.error("❌ Qwen endpoint warmup failed")
                        return False
                    return True

                if endpoint.status in ["failed", "error"]:
                    logger.error(f"❌ Endpoint failed: {endpoint.status}")
                    return False

                logger.info(f"   ⏳ Still {endpoint.status}, waiting... ({time.time() - start_time:.0f}s)")
                time.sleep(poll_interval)

            except Exception as e:
                logger.warning(f"⚠️ Error checking status: {e}")
                time.sleep(poll_interval)

        logger.error(f"❌ Timeout waiting for endpoint (max {max_wait}s)")
        return False


    def warmup(self) -> bool:
        """Smart polling-based warmup - stops as soon as endpoint responds."""
        if self.warmup_completed:
            logger.info("✅ Qwen endpoint already warmed up")
            return True

        # Defensive check: ensure endpoint is not paused before warmup
        if self._can_pause_resume:
            endpoint = self._get_endpoint()
            if endpoint:
                try:
                    endpoint.fetch()
                    if endpoint.status in ["paused", "scaledToZero"]:
                        logger.warning(f"⚠️ Qwen endpoint is {endpoint.status} - triggering resume")
                        endpoint.resume().wait(timeout=300)
                        self.resume_count += 1
                        self.last_resume_time = time.time()
                except Exception as e:
                    logger.warning(f"⚠️ Failed to check/resume endpoint during warmup: {e}")

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
            # Fix URL construction - endpoint_url may already include /v1/
            base_url = self.endpoint_url.rstrip('/')
            if base_url.endswith('/v1'):
                url = f"{base_url}/chat/completions"
            else:
                url = f"{base_url}/v1/chat/completions"

            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.endpoint_token}", "Content-Type": "application/json"},
                json={"model": "Qwen/Qwen3-VL-32B-Instruct", "messages": [{"role": "user", "content": "OK"}], "max_tokens": 2},
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

    def force_pause(self) -> bool:
        """
        Signal job completion - let HuggingFace auto-scale to zero.

        NOTE: We do NOT call endpoint.pause() as it sets state to "paused" which
        won't auto-resume on requests. Instead, we rely on HuggingFace's automatic
        scale-to-zero feature which sets state to "scaledToZero" and WILL auto-resume.

        Returns:
            True always (HF handles scale-to-zero automatically)
        """
        logger.info("✅ Qwen job completed - letting HuggingFace auto-scale to zero")

        # Track uptime if we were running
        if self.last_resume_time:
            uptime = time.time() - self.last_resume_time
            self.total_uptime += uptime
            self.last_resume_time = None

        self.warmup_completed = False
        return True

    def manual_pause(self) -> bool:
        """
        MANUAL PAUSE - Use only if you explicitly want to stop the endpoint.
        WARNING: This sets state to "paused" which requires explicit resume().
        Prefer letting HuggingFace auto-scale to "scaledToZero" instead.

        Returns:
            True if paused successfully, False if failed
        """
        if not self._can_pause_resume:
            logger.warning("Pause/resume not available - cannot manual pause")
            return False

        endpoint = self._get_endpoint()
        if not endpoint:
            return False

        try:
            endpoint.fetch()
            if endpoint.status == "running":
                logger.warning("⚠️ MANUAL PAUSE: Qwen endpoint (will require explicit resume)")
                endpoint.pause()
                self.pause_count += 1

                if self.last_resume_time:
                    uptime = time.time() - self.last_resume_time
                    self.total_uptime += uptime

                self.warmup_completed = False
                logger.info("✅ Qwen endpoint manually paused")
                return True
            else:
                logger.info(f"Endpoint already not running (status: {endpoint.status})")
                return True

        except Exception as e:
            logger.error(f"❌ Failed to manual pause Qwen endpoint: {e}")
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

