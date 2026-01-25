"""
YOLO DocParser Inference Endpoint Manager

Manages HuggingFace Inference Endpoint lifecycle for YOLO DocParser layout detection model.
Provides automatic pause/resume functionality to control billing costs.

CRITICAL: Endpoint is paused by default (no billing). Only resumes when layout detection is needed,
then auto-pauses after idle timeout to prevent unnecessary billing.

Cost Control Strategy:
- Endpoint paused: $0/hour
- Endpoint running: ~$0.60/hour (GPU)
- Auto-pause after 60s idle (configurable)
- Force-pause after batch processing
- Warmup required: 60 seconds before first inference
- Typical cost: ~$0.01 per 30-page document
"""

import os
import time
import logging
import requests
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from huggingface_hub import get_inference_endpoint
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logging.warning("huggingface_hub not available - pause/resume features disabled")

logger = logging.getLogger(__name__)


class YoloEndpointManager:
    """
    Manages HuggingFace Inference Endpoint lifecycle for YOLO DocParser.
    
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
        endpoint_name: Optional[str] = None,
        namespace: Optional[str] = None,
        auto_pause_timeout: int = 60,
        inference_timeout: int = 30,
        warmup_timeout: int = 60,
        max_resume_retries: int = 3,
        enabled: bool = True
    ):
        """
        Initialize YOLO Endpoint Manager.

        Args:
            endpoint_url: Full URL of the YOLO inference endpoint
            hf_token: HuggingFace API token (with write permissions)
            endpoint_name: Endpoint name for pause/resume (e.g., 'yolo-docparser')
            namespace: HuggingFace namespace/username (e.g., 'basiliskan')
            auto_pause_timeout: Seconds of idle time before auto-pause (default: 60)
            inference_timeout: Timeout for inference calls in seconds (default: 30)
            warmup_timeout: Warmup time in seconds (default: 60)
            max_resume_retries: Maximum retry attempts for resuming endpoint (default: 3)
            enabled: Enable/disable YOLO layout detection (default: True)
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
                f"✅ YOLO Endpoint Manager initialized with pause/resume: "
                f"endpoint={endpoint_name}, namespace={namespace}, auto_pause={auto_pause_timeout}s, warmup={warmup_timeout}s"
            )
        else:
            logger.warning(
                f"⚠️ YOLO Endpoint Manager initialized WITHOUT pause/resume: "
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
                logger.info(f"✅ Connected to YOLO endpoint: {self.endpoint_name}")
            except Exception as e:
                logger.error(f"❌ Failed to get YOLO endpoint: {e}")
                return None

        return self._endpoint

    def resume_if_needed(self) -> bool:
        """
        Resume endpoint if it's paused or wait if initializing.

        Handles all HuggingFace endpoint states:
        - running: Ready to use
        - paused/scaledToZero: Needs resume
        - initializing: Needs polling until ready

        CRITICAL: This starts billing! Only call when layout detection is needed.

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
                logger.info("✅ YOLO endpoint already running")
                return True

            # Handle "initializing" state - poll until ready
            if endpoint.status == "initializing":
                logger.info(f"⏳ YOLO endpoint initializing, waiting for it to be ready...")
                return self._wait_for_running(endpoint)

            if endpoint.status in ["paused", "scaledToZero"]:
                logger.info(f"🔄 Resuming YOLO endpoint (status: {endpoint.status})...")

                # Resume with retries
                for attempt in range(self.max_resume_retries):
                    try:
                        endpoint.resume().wait(timeout=300)  # Wait up to 5 minutes
                        self.resume_count += 1
                        self.last_resume_time = time.time()
                        self.warmup_completed = False  # Reset warmup flag
                        logger.info(f"✅ YOLO endpoint resumed (attempt {attempt + 1}/{self.max_resume_retries})")

                        # Warmup after resume
                        if not self.warmup():
                            logger.error("❌ YOLO endpoint warmup failed")
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
                    logger.info(f"✅ YOLO endpoint ready after {elapsed:.1f}s")
                    self.last_resume_time = time.time()

                    # Warmup after becoming ready
                    if not self.warmup():
                        logger.error("❌ YOLO endpoint warmup failed")
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
        Smart polling-based warmup for YOLO endpoint.

        Uses exponential backoff to poll the endpoint until it responds,
        stopping as soon as ready instead of fixed 60s wait.

        Returns:
            True if warmup successful, False if timeout
        """
        if self.warmup_completed:
            logger.info("✅ YOLO endpoint already warmed up")
            return True

        # Defensive check: ensure endpoint is not paused before warmup
        if self._can_pause_resume:
            endpoint = self._get_endpoint()
            if endpoint:
                try:
                    endpoint.fetch()
                    if endpoint.status in ["paused", "scaledToZero"]:
                        logger.warning(f"⚠️ YOLO endpoint is {endpoint.status} - triggering resume")
                        endpoint.resume().wait(timeout=300)
                        self.resume_count += 1
                        self.last_resume_time = time.time()
                except Exception as e:
                    logger.warning(f"⚠️ Failed to check/resume endpoint during warmup: {e}")

        logger.info(f"🔥 Warming up YOLO endpoint (max {self.warmup_timeout}s)...")

        start_time = time.time()
        attempt = 0
        base_delay = 2
        max_delay = 15

        while (time.time() - start_time) < self.warmup_timeout:
            attempt += 1

            if self._test_inference():
                elapsed = time.time() - start_time
                logger.info(f"✅ YOLO endpoint warmed up in {elapsed:.1f}s ({attempt} attempts)")
                self.warmup_completed = True
                return True

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            remaining = self.warmup_timeout - (time.time() - start_time)

            if remaining > delay:
                logger.info(f"   ⏳ Warmup attempt {attempt} - not ready, retrying in {delay}s...")
                time.sleep(delay)
            else:
                time.sleep(min(2, remaining))

        logger.error(f"❌ YOLO warmup failed after {time.time() - start_time:.1f}s")
        return False

    def _test_inference(self) -> bool:
        """Test if YOLO endpoint can handle requests."""
        try:
            response = requests.get(
                f"{self.endpoint_url.rstrip('/')}/health",
                headers={"Authorization": f"Bearer {self.hf_token}"},
                timeout=10
            )
            # Accept 200 or 404 (no /health route but endpoint responding)
            return response.status_code in [200, 404]
        except:
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
                logger.info("⏸️ Pausing YOLO endpoint (JOB COMPLETED - stopping billing)")
                endpoint.pause()
                self.pause_count += 1

                if self.last_resume_time:
                    uptime = time.time() - self.last_resume_time
                    self.total_uptime += uptime

                self.warmup_completed = False
                logger.info("✅ YOLO endpoint paused (no billing until next job)")
                return True
            else:
                logger.info(f"Endpoint already not running (status: {endpoint.status})")
                return True

        except Exception as e:
            logger.error(f"❌ Failed to pause YOLO endpoint: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get endpoint usage statistics.

        Returns:
            Dictionary with usage stats
        """
        return {
            "resume_count": self.resume_count,
            "pause_count": self.pause_count,
            "inference_count": self.inference_count,
            "total_uptime_seconds": self.total_uptime,
            "total_uptime_hours": self.total_uptime / 3600,
            "estimated_cost_usd": (self.total_uptime / 3600) * 0.60,  # $0.60/hour estimate
            "last_used": datetime.fromtimestamp(self.last_used).isoformat() if self.last_used else None,
            "warmup_completed": self.warmup_completed,
            "enabled": self.enabled
        }

    def mark_used(self):
        """Mark endpoint as recently used (for auto-pause tracking)."""
        self.last_used = time.time()
        self.inference_count += 1


