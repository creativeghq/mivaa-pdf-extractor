"""
SLIG (SigLIP2) Cloud Client

Client for HuggingFace Inference Endpoint `mh-slig` (namespace `basiliskan`)
serving the custom HF model `basiliskan/slig`. Underlying architecture is
SigLIP2 SO400M (native 1152D); the endpoint applies a 1152D → 768D projection
head so every downstream consumer sees a uniform 768D output.

Supports 4 modes: zero_shot, image_embedding, text_embedding, similarity.

Includes automatic endpoint lifecycle management (pause/resume) to control costs.

Based on API specification in docs/api/slig.md
"""

import httpx
import logging
import base64
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import io

logger = logging.getLogger(__name__)


class SLIGClient:
    """
    Client for SLIG (SigLIP2) Inference Endpoint.

    Supports 4 modes:
    - zero_shot: Classify image against candidate labels
    - image_embedding: Extract 768D embedding from image
    - text_embedding: Extract 768D embedding from text
    - similarity: Calculate similarity between images and texts

    Includes automatic endpoint lifecycle management:
    - Auto-resume before inference (starts billing)
    - Auto-pause after idle timeout (stops billing)
    - 60s warmup after resume
    """

    def __init__(
        self,
        endpoint_url: str,
        token: str,
        timeout: float = 30.0,
        model_name: str = "basiliskan/slig",
        endpoint_name: str = "mh-slig",
        namespace: str = "basiliskan",
        auto_pause: bool = True,
        auto_pause_timeout: int = 60,
        endpoint_manager: "SLIGEndpointManager" = None  # ✅ Accept pre-created manager
    ):
        """
        Initialize SLIG client.

        Args:
            endpoint_url: HuggingFace Inference Endpoint URL
            token: HuggingFace authentication token
            timeout: Request timeout in seconds
            model_name: Model name for logging
            endpoint_name: Endpoint name for pause/resume
            namespace: HuggingFace namespace/username
            auto_pause: Enable automatic pause/resume (default: True)
            auto_pause_timeout: Seconds of idle time before auto-pause (default: 60)
            endpoint_manager: Pre-created endpoint manager (for singleton pattern)
        """
        self.endpoint_url = endpoint_url.rstrip('/')
        self.token = token
        self.timeout = timeout
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # Initialize endpoint manager for pause/resume
        self._endpoint_manager = None

        # ✅ Use pre-created manager if provided (singleton pattern from registry)
        if endpoint_manager is not None:
            self._endpoint_manager = endpoint_manager
            logger.info("✅ SLIG client using pre-warmed endpoint manager (singleton)")
        elif auto_pause:
            try:
                from app.services.embeddings.slig_endpoint_manager import SLIGEndpointManager
                self._endpoint_manager = SLIGEndpointManager(
                    endpoint_url=endpoint_url,
                    hf_token=token,
                    endpoint_name=endpoint_name,
                    namespace=namespace,
                    auto_pause_timeout=auto_pause_timeout
                )
                logger.info("✅ SLIG endpoint manager initialized (auto-pause enabled)")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize endpoint manager: {e}")
                logger.warning("⚠️ Endpoint will run continuously (no auto-pause)")
        else:
            logger.info("ℹ️ SLIG auto-pause disabled - endpoint will run continuously")
        
    async def _ensure_endpoint_ready(self):
        """Ensure endpoint is running and warmed up before inference."""
        # ✅ If we have an endpoint manager, verify the endpoint is ready
        if self._endpoint_manager:
            # Check warmup status from the manager
            if not self._endpoint_manager.warmup_completed:
                logger.info("⏳ SLIG endpoint not warmed up - checking status...")
                import asyncio

                # Try to warmup using thread pool (blocking HF API calls)
                try:
                    success = await asyncio.to_thread(self._endpoint_manager.warmup)
                    if success:
                        logger.info("✅ SLIG endpoint warmed up successfully")
                    else:
                        logger.warning("⚠️ SLIG warmup failed - inference may fail")
                except Exception as e:
                    logger.warning(f"⚠️ SLIG warmup check failed: {e}")
        # else: No manager means we trust the endpoint is already running (pre-warmed)

    async def _call_endpoint(self, payload: Dict[str, Any]) -> Any:
        """
        Make async HTTP request to SLIG endpoint.
        Automatically handles endpoint resume and warmup.

        Gated through the unified EndpointController so that all SLIG call
        sites downstream automatically participate in AIMD backpressure — one
        choke point, no per-call-site wiring.
        """
        from app.services.core.endpoint_controller import endpoint_controller

        # Ensure endpoint is ready
        await self._ensure_endpoint_ready()

        async with endpoint_controller.slig.slot():
            try:
                # Make inference call
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.endpoint_url,
                        headers=self.headers,
                        json=payload
                    )
                    response.raise_for_status()
                    result = response.json()

                # Mark endpoint as used (for auto-pause tracking)
                if self._endpoint_manager:
                    self._endpoint_manager.mark_used()

                endpoint_controller.record_success("slig")
                return result

            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError):
                # Overload-class failure → shrink SLIG concurrency
                endpoint_controller.record_failure("slig")
                raise
            except httpx.HTTPStatusError as e:
                # Only 5xx / 429 are backpressure signals
                if e.response.status_code in (429, 500, 502, 503, 504):
                    endpoint_controller.record_failure("slig")
                raise
    
    def _image_to_base64(self, image: Image.Image, max_dimension: int = 512) -> str:
        """
        Convert PIL Image to base64 string.

        Automatically resizes large images to prevent 400 Bad Request errors
        from the SLIG endpoint which has payload size limits.

        Args:
            image: PIL Image to convert
            max_dimension: Maximum width/height (default 512 for SigLIP2)

        Returns:
            Base64 encoded image string
        """
        # Resize if needed to prevent 400 errors
        if image.width > max_dimension or image.height > max_dimension:
            original_size = (image.width, image.height)
            image = image.copy()  # Don't modify original
            image.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
            logger.debug(f"📐 Resized image from {original_size} to {image.size} for SLIG")

        # Use JPEG for smaller payload (vs PNG)
        buffered = io.BytesIO()
        # Convert to RGB if needed (JPEG doesn't support alpha)
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=85, optimize=True)
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    async def zero_shot_classification(
        self,
        image: Union[str, Image.Image],
        candidate_labels: List[str]
    ) -> List[Dict[str, float]]:
        """
        Classify image against candidate labels.
        
        Args:
            image: Image URL or PIL Image
            candidate_labels: List of text labels to classify against
            
        Returns:
            List of {"label": str, "score": float} sorted by score descending
        """
        # Convert PIL Image to base64 if needed
        if isinstance(image, Image.Image):
            image = self._image_to_base64(image)
        
        payload = {
            "inputs": image,
            "parameters": {
                "candidate_labels": candidate_labels,
                "mode": "zero_shot"
            }
        }
        
        result = await self._call_endpoint(payload)
        logger.info(f"✅ SLIG zero-shot classification: {len(candidate_labels)} labels")
        return result
    
    async def get_image_embedding(
        self,
        image: Union[str, Image.Image, List[Union[str, Image.Image]]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Extract 768D embedding from image(s).
        
        Args:
            image: Single image (URL or PIL) or list of images
            
        Returns:
            Single embedding [768] or list of embeddings [[768], [768], ...]
        """
        # Handle single image
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        
        # Convert PIL Images to base64
        processed_images = []
        for img in images:
            if isinstance(img, Image.Image):
                processed_images.append(self._image_to_base64(img))
            else:
                processed_images.append(img)
        
        payload = {
            "inputs": processed_images if is_batch else processed_images[0],
            "parameters": {"mode": "image_embedding"}
        }
        
        result = await self._call_endpoint(payload)

        # Extract embeddings from response
        if is_batch:
            embeddings = [item["embedding"] for item in result]
            logger.info(f"✅ SLIG image embeddings: {len(embeddings)} images, 768D each")
            return embeddings
        else:
            embedding = result[0]["embedding"]
            logger.info(f"✅ SLIG image embedding: 768D")
            return embedding

    async def get_text_embedding(
        self,
        text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Extract 768D embedding from text(s).

        Args:
            text: Single text string or list of text strings

        Returns:
            Single embedding [768] or list of embeddings [[768], [768], ...]
        """
        is_batch = isinstance(text, list)

        payload = {
            "inputs": text,
            "parameters": {"mode": "text_embedding"}
        }

        result = await self._call_endpoint(payload)

        # Extract embeddings from response
        if is_batch:
            embeddings = [item["embedding"] for item in result]
            logger.info(f"✅ SLIG text embeddings: {len(embeddings)} texts, 768D each")
            return embeddings
        else:
            embedding = result[0]["embedding"]
            logger.info(f"✅ SLIG text embedding: 768D")
            return embedding

    async def calculate_similarity(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        texts: Union[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Calculate cosine similarity between images and texts.

        Args:
            images: Single image or list of images (URL or PIL)
            texts: Single text or list of texts

        Returns:
            {
                "similarity_scores": [[float, ...], ...],  # [image_count, text_count]
                "image_count": int,
                "text_count": int
            }
        """
        # Handle single vs batch
        is_image_batch = isinstance(images, list)
        is_text_batch = isinstance(texts, list)

        # Convert to lists
        image_list = images if is_image_batch else [images]
        text_list = texts if is_text_batch else [texts]

        # Convert PIL Images to base64
        processed_images = []
        for img in image_list:
            if isinstance(img, Image.Image):
                processed_images.append(self._image_to_base64(img))
            else:
                processed_images.append(img)

        # Build payload — always use plural keys; the SLIG endpoint handler
        # accepts arrays for both images and texts.
        payload = {
            "inputs": {
                "images": processed_images,  # Always array
                "texts": text_list           # Always array
            },
            "parameters": {"mode": "similarity"}
        }

        result = await self._call_endpoint(payload)
        logger.info(
            f"✅ SLIG similarity: {result['image_count']} images × "
            f"{result['text_count']} texts"
        )
        return result

    # Endpoint lifecycle management methods

    def pause_if_idle(self) -> bool:
        """
        Pause endpoint if it's been idle for too long.
        Called automatically by background tasks.

        Returns:
            True if paused or already paused, False if failed
        """
        if self._endpoint_manager:
            return self._endpoint_manager.pause_if_idle()
        return False

    def get_endpoint_stats(self) -> Optional[dict]:
        """
        Get endpoint usage statistics.

        Returns:
            Dictionary with stats or None if manager not available
        """
        if self._endpoint_manager:
            return self._endpoint_manager.get_stats()
        return None

