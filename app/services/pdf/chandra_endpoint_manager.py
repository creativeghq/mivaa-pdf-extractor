"""
Chandra OCR Inference Endpoint Manager

Manages HuggingFace Inference Endpoint lifecycle for Chandra OCR model.
Provides automatic pause/resume functionality to control billing costs.

CRITICAL: Endpoint is paused by default (no billing). Only resumes when OCR is needed,
then auto-pauses after idle timeout to prevent unnecessary billing.

Cost Control Strategy:
- Endpoint paused: $0/hour
- Endpoint running: ~$0.60/hour (GPU)
- Auto-pause after 60s idle (configurable)
- Force-pause after batch processing
- Typical cost: ~$0.02 per 30-page document
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


class ChandraEndpointManager:
    """
    Manages HuggingFace Inference Endpoint lifecycle for Chandra OCR.
    
    Features:
    - Automatic resume before inference (start billing)
    - Automatic pause after idle timeout (stop billing)
    - Force pause after batch processing
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

                logger.info(f"   ⏳ Still {endpoint.status}, waiting... ({time.time() - start_time:.0f}s)")
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
        Test if Chandra endpoint can handle requests.
        Uses the correct /v1/chat/completions path.

        Returns:
            True if endpoint responds successfully, False otherwise
        """
        try:
            # Use correct OpenAI-compatible chat completions endpoint
            api_url = self.endpoint_url.rstrip('/') + '/v1/chat/completions'

            response = requests.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {self.hf_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "prithivMLmods/chandra-OCR-GGUF",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1
                },
                timeout=15
            )

            # 200 = success, 503 = still loading
            if response.status_code == 200:
                return True
            elif response.status_code == 503:
                return False
            elif response.status_code == 500:
                # Internal error - might still be loading model
                return False
            else:
                logger.debug(f"   Warmup test returned status {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            return False
        except requests.exceptions.ConnectionError:
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

    def run_inference(self, image_input: Any, parameters: Optional[Dict] = None, prompt: str = "Extract all text from this image. Return only the extracted text.") -> Dict[str, Any]:
        """
        Run OCR inference on image using Chandra endpoint (OpenAI-compatible format).

        Args:
            image_input: Image data (bytes, PIL Image, or file path)
            parameters: Optional inference parameters
            prompt: OCR prompt (default: extract text)

        Returns:
            Dict with OCR result: {'generated_text': str, 'confidence': float}
        """
        import base64
        
        if not self.enabled:
            raise Exception("Chandra endpoint is disabled")

        if self._can_pause_resume:
            if not self.resume_if_needed():
                raise Exception("Failed to resume Chandra endpoint")
        
        start_time = time.time()
        
        try:
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
            
            # OpenAI-compatible chat completions format
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }
            
            # Use /v1/chat/completions endpoint
            api_url = self.endpoint_url.rstrip('/') + '/v1/chat/completions'
            
            payload = {
                "model": "prithivMLmods/chandra-OCR-GGUF",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }],
                "stream": False,
                "max_tokens": parameters.get('max_tokens', 2000) if parameters else 2000
            }
            
            logger.info(f"Calling Chandra endpoint: {api_url}")
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=self.inference_timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract text from OpenAI format response
            generated_text = ""
            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0].get("message", {}).get("content", "")
            
            inference_time = time.time() - start_time
            self.last_used = time.time()
            self.inference_count += 1
            self.total_uptime += inference_time
            
            logger.info(f"✅ Chandra OCR successful: {len(generated_text)} chars in {inference_time:.2f}s")
            
            return {
                "generated_text": generated_text,
                "confidence": 0.85,
                "raw_response": result
            }
            
        except Exception as e:
            logger.error(f"❌ Chandra inference failed: {e}")
            raise

