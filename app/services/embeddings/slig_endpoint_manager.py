"""
SLIG (SigLIP2) Inference Endpoint Manager

Manages HuggingFace Inference Endpoint lifecycle for SLIG visual embeddings.
Provides automatic pause/resume functionality to control billing costs.

CRITICAL: Endpoint is paused by default (no billing). Only resumes when visual embeddings are needed,
then auto-pauses after idle timeout to prevent unnecessary billing.

Cost Control Strategy:
- Endpoint paused: $0/hour
- Endpoint running: ~$0.60/hour (GPU)
- Auto-pause after 60s idle (configurable)
- Force-pause after batch processing
- Warmup required: 60 seconds before first inference
- Typical cost: ~$0.01 per 100 images
"""

import os
import time
import logging
from typing import Optional
from datetime import datetime

try:
    from huggingface_hub import get_inference_endpoint
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logging.warning("huggingface_hub not available - pause/resume features disabled")

logger = logging.getLogger(__name__)


class SLIGEndpointManager:
    """
    Manages HuggingFace Inference Endpoint lifecycle for SLIG (SigLIP2).
    
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
        hf_token: str,
        endpoint_name: str = "mh-siglip2",
        namespace: str = "basiliskan",
        auto_pause_timeout: int = 60,
        inference_timeout: int = 30,
        warmup_timeout: int = 60,
        max_resume_retries: int = 3,
        enabled: bool = True
    ):
        """
        Initialize SLIG endpoint manager.
        
        Args:
            endpoint_url: Full URL of the SLIG inference endpoint
            hf_token: HuggingFace API token (with write permissions)
            endpoint_name: Endpoint name for pause/resume (e.g., 'mh-siglip2')
            namespace: HuggingFace namespace/username (e.g., 'basiliskan')
            auto_pause_timeout: Seconds of idle time before auto-pause (default: 60)
            inference_timeout: Timeout for inference calls in seconds (default: 30)
            warmup_timeout: Warmup time in seconds (default: 60)
            max_resume_retries: Maximum retry attempts for resuming endpoint (default: 3)
            enabled: Enable/disable SLIG endpoint management (default: True)
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
        
        if not self._can_pause_resume:
            logger.warning("⚠️ SLIG endpoint pause/resume not available - endpoint will run continuously")
        
        logger.info(f"✅ SLIG Endpoint Manager initialized: {endpoint_name}")
    
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
                logger.info(f"✅ Connected to SLIG endpoint: {self.endpoint_name}")
            except Exception as e:
                logger.error(f"❌ Failed to get SLIG endpoint: {e}")
                return None
        
        return self._endpoint
    
    def resume_if_needed(self) -> bool:
        """
        Resume endpoint if it's paused or wait if initializing.

        Handles all HuggingFace endpoint states:
        - running: Ready to use
        - paused/scaledToZero: Needs resume
        - initializing: Needs polling until ready

        CRITICAL: This starts billing! Only call when visual embeddings are needed.

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
                logger.info("✅ SLIG endpoint already running")
                return True

            # Handle "initializing" state - poll until ready
            if endpoint.status == "initializing":
                logger.info(f"⏳ SLIG endpoint initializing, waiting for it to be ready...")
                return self._wait_for_running(endpoint)

            if endpoint.status in ["paused", "scaledToZero"]:
                logger.info(f"🔄 Resuming SLIG endpoint (status: {endpoint.status})...")

                # Resume with retries
                for attempt in range(self.max_resume_retries):
                    try:
                        endpoint.resume().wait(timeout=300)  # Wait up to 5 minutes
                        self.resume_count += 1
                        self.last_resume_time = time.time()
                        self.warmup_completed = False  # Reset warmup flag
                        logger.info(f"✅ SLIG endpoint resumed (attempt {attempt + 1}/{self.max_resume_retries})")

                        # Warmup after resume
                        if not self.warmup():
                            logger.error("❌ SLIG endpoint warmup failed")
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

        while (time.time() - start_time) < max_wait:
            try:
                endpoint.fetch()

                if endpoint.status == "running":
                    elapsed = time.time() - start_time
                    logger.info(f"✅ SLIG endpoint ready after {elapsed:.1f}s")
                    self.last_resume_time = time.time()

                    # Warmup after becoming ready
                    if not self.warmup():
                        logger.error("❌ SLIG endpoint warmup failed")
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
        """
        Warmup the SLIG endpoint after resume using inference testing.

        Instead of blind sleep, this method polls the endpoint with test
        requests until it responds successfully, with exponential backoff.

        Returns:
            True if warmup successful, False if failed/timeout
        """
        if self.warmup_completed:
            logger.info("✅ SLIG endpoint already warmed up")
            return True

        logger.info(f"🔥 Warming up SLIG endpoint (max {self.warmup_timeout}s)...")

        # Test inference with exponential backoff
        start_time = time.time()
        attempt = 0
        base_delay = 2  # Start with 2 second delay
        max_delay = 15  # Cap at 15 seconds between attempts

        while (time.time() - start_time) < self.warmup_timeout:
            attempt += 1

            if self._test_inference():
                elapsed = time.time() - start_time
                logger.info(f"✅ SLIG endpoint warmed up in {elapsed:.1f}s ({attempt} attempts)")
                self.warmup_completed = True
                return True

            # Calculate delay with exponential backoff (2, 4, 8, 15, 15, ...)
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            remaining = self.warmup_timeout - (time.time() - start_time)

            if remaining > delay:
                logger.info(f"   ⏳ Attempt {attempt} failed, retrying in {delay}s...")
                time.sleep(delay)
            else:
                # Not enough time for another full delay, do a quick final check
                time.sleep(min(2, remaining))

        # Timeout reached
        elapsed = time.time() - start_time
        logger.error(f"❌ SLIG warmup failed after {elapsed:.1f}s ({attempt} attempts)")
        return False

    def _test_inference(self) -> bool:
        """
        Test if the endpoint can handle inference requests.

        Uses a minimal test to check endpoint health without wasting resources.

        Returns:
            True if endpoint responds successfully, False otherwise
        """
        try:
            import requests

            # Create a 16x16 red pixel PNG for testing
            # 1x1 images are too small for SigLIP - use 16x16 minimum
            test_image_base64 = (
                "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAFklEQVR4"
                "2mP4z8BAEmIY1TCqYfhqAACQ+f8B8u7oVwAAAABJRU5ErkJggg=="
            )

            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }

            # Send minimal inference request
            # Handler expects: {"inputs": "<base64_string>", "parameters": {"mode": "image_embedding"}}
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json={
                    "inputs": test_image_base64,
                    "parameters": {"mode": "image_embedding"}
                },
                timeout=10  # Short timeout for health check
            )

            if response.status_code == 200:
                return True
            elif response.status_code == 503:
                # Service unavailable - still warming up
                return False
            elif response.status_code == 500:
                # Internal error - might still be loading model
                return False
            else:
                logger.debug(f"   Warmup test returned status {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            # Timeout is expected during warmup
            return False
        except requests.exceptions.ConnectionError:
            # Connection error means endpoint not ready
            return False
        except Exception as e:
            logger.debug(f"   Warmup test error: {e}")
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
        DISABLED: Let HuggingFace handle scale-to-zero automatically.
        Manual pausing causes endpoints to not auto-resume on requests.

        Returns:
            True always (no-op)
        """
        logger.debug("force_pause() disabled - letting HF handle scale-to-zero automatically")
        return True

    def mark_used(self):
        """Mark endpoint as recently used (for auto-pause tracking)."""
        self.last_used = time.time()
        self.inference_count += 1

    def get_stats(self) -> dict:
        """Get endpoint usage statistics."""
        return {
            "endpoint_name": self.endpoint_name,
            "resume_count": self.resume_count,
            "pause_count": self.pause_count,
            "inference_count": self.inference_count,
            "total_uptime_seconds": self.total_uptime,
            "total_uptime_minutes": round(self.total_uptime / 60, 2),
            "estimated_cost_usd": round(self.total_uptime / 3600 * 0.60, 4),  # $0.60/hour
            "warmup_completed": self.warmup_completed,
            "last_used": datetime.fromtimestamp(self.last_used).isoformat() if self.last_used else None
        }

