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
            max_resume_retries: Maximum retry attempts for resuming endpoint (default: 3)
            enabled: Enable/disable Chandra fallback (default: True)
        """
        self.endpoint_url = endpoint_url
        self.hf_token = hf_token
        self.endpoint_name = endpoint_name
        self.namespace = namespace
        self.auto_pause_timeout = auto_pause_timeout
        self.inference_timeout = inference_timeout
        self.max_resume_retries = max_resume_retries
        self.enabled = enabled

        # Track usage for auto-pause
        self.last_used: Optional[float] = None
        self.last_resume_time: Optional[float] = None
        self.total_uptime: float = 0.0
        self.resume_count: int = 0
        self.pause_count: int = 0
        self.inference_count: int = 0

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
        Resume endpoint if paused.

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

            if endpoint.status in ["paused", "scaledToZero"]:
                logger.info(f"🔄 Resuming Chandra endpoint (status: {endpoint.status})...")

                # Resume with retries
                for attempt in range(self.max_resume_retries):
                    try:
                        endpoint.resume().wait(timeout=300)  # Wait up to 5 minutes
                        self.resume_count += 1
                        self.last_resume_time = time.time()
                        logger.info(f"✅ Chandra endpoint resumed (attempt {attempt + 1}/{self.max_resume_retries})")
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

    def pause_if_idle(self) -> bool:
        """
        Pause endpoint if idle for too long.
        CRITICAL: This prevents billing when not in use!

        Returns:
            True if paused or already paused, False if failed
        """
        if not self._can_pause_resume:
            return False

        if self.last_used is None:
            return False

        idle_time = time.time() - self.last_used

        if idle_time > self.auto_pause_timeout:
            endpoint = self._get_endpoint()
            if not endpoint:
                return False

            try:
                endpoint.fetch()
                if endpoint.status == "running":
                    logger.info(f"⏸️ Auto-pausing Chandra endpoint (idle for {idle_time:.0f}s)")
                    endpoint.pause()
                    self.pause_count += 1

                    # Track uptime
                    if self.last_resume_time:
                        uptime = time.time() - self.last_resume_time
                        self.total_uptime += uptime

                    logger.info(f"✅ Chandra endpoint paused (no billing)")
                    return True

            except Exception as e:
                logger.error(f"❌ Failed to pause endpoint: {e}")
                return False

        return False

    def force_pause(self) -> bool:
        """
        Force pause endpoint immediately.
        Use this after batch processing is complete.

        Returns:
            True if paused successfully, False if failed
        """
        if not self._can_pause_resume:
            logger.warning("Pause/resume not available - cannot force pause")
            return False

        endpoint = self._get_endpoint()
        if not endpoint:
            return False

        try:
            endpoint.fetch()
            if endpoint.status == "running":
                logger.info("⏸️ Force pausing Chandra endpoint")
                endpoint.pause()
                self.pause_count += 1

                # Track uptime
                if self.last_resume_time:
                    uptime = time.time() - self.last_resume_time
                    self.total_uptime += uptime

                logger.info(f"✅ Chandra endpoint paused (no billing)")
                return True
            else:
                logger.info(f"Endpoint already paused (status: {endpoint.status})")
                return True

        except Exception as e:
            logger.error(f"❌ Failed to force pause endpoint: {e}")
            return False

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

