"""
SLIG (SigLIP2) Cloud Client

Client for HuggingFace Inference Endpoint running SigLIP2-base-patch16-512.
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
        model_name: str = "basiliskan/siglip2",
        endpoint_name: str = "mh-siglip2",
        namespace: str = "basiliskan",
        auto_pause: bool = True,
        auto_pause_timeout: int = 60
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
        if auto_pause:
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
        if self._endpoint_manager:
            # Resume endpoint if paused
            if not self._endpoint_manager.resume_if_needed():
                logger.error("❌ Failed to resume SLIG endpoint")
                raise RuntimeError("Failed to resume SLIG endpoint")

            # Warmup if needed
            if not self._endpoint_manager.warmup():
                logger.error("❌ Failed to warmup SLIG endpoint")
                raise RuntimeError("Failed to warmup SLIG endpoint")

    async def _call_endpoint(self, payload: Dict[str, Any]) -> Any:
        """
        Make async HTTP request to SLIG endpoint.
        Automatically handles endpoint resume and warmup.
        """
        # Ensure endpoint is ready
        await self._ensure_endpoint_ready()

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

        return result
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
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

        # Build payload
        payload = {
            "inputs": {
                "images" if len(processed_images) > 1 else "image":
                    processed_images if len(processed_images) > 1 else processed_images[0],
                "texts" if len(text_list) > 1 else "text":
                    text_list if len(text_list) > 1 else text_list[0]
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

    def pause_endpoint(self) -> bool:
        """
        Manually pause the endpoint to stop billing.
        Use this after batch processing is complete.

        Returns:
            True if paused successfully, False if failed or not available
        """
        if self._endpoint_manager:
            return self._endpoint_manager.force_pause()
        else:
            logger.warning("⚠️ Endpoint manager not available - cannot pause")
            return False

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

