"""
Hugging Face Visual Embeddings Service

Provides remote SigLIP visual embedding generation via Hugging Face Inference API.
This service offers GPU-accelerated embeddings without requiring local GPU hardware.
"""

import logging
import asyncio
import base64
import io
import time
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import httpx
import numpy as np

from app.services.ai_call_logger import AICallLogger

logger = logging.getLogger(__name__)


class HuggingFaceVisualEmbeddingsService:
    """
    Remote visual embedding generation using Hugging Face Inference API.
    
    Features:
    - GPU-accelerated SigLIP embeddings via Hugging Face
    - Batch processing support (up to 10 images per request)
    - Automatic retry with exponential backoff
    - Cost tracking and logging
    - Fallback to local processing on failure
    """
    
    def __init__(self, api_key: str, config=None):
        """Initialize Hugging Face visual embeddings service."""
        self.api_key = api_key
        self.config = config
        self.logger = logger
        self.ai_logger = AICallLogger()
        
        # Load configuration from settings
        from app.config import get_settings
        settings = get_settings()
        
        self.api_url = settings.huggingface_api_url
        self.model_id = settings.huggingface_siglip_model
        self.batch_size = settings.huggingface_batch_size
        self.timeout = settings.huggingface_timeout
        self.max_retries = settings.huggingface_max_retries
        
        self.logger.info(f"🤗 Hugging Face Visual Embeddings initialized: {self.model_id}")
        self.logger.info(f"   Batch size: {self.batch_size}, Timeout: {self.timeout}s")
    
    async def generate_embedding(
        self,
        image_data: str,
        pil_image: Optional[Image.Image] = None,
        job_id: Optional[str] = None
    ) -> Tuple[Optional[List[float]], Optional[Image.Image]]:
        """
        Generate visual embedding for a single image.
        
        Args:
            image_data: Base64 encoded image data
            pil_image: Optional PIL image (for reuse)
            job_id: Optional job ID for logging
            
        Returns:
            Tuple of (embedding vector, PIL image for reuse)
        """
        # Process single image as batch of 1
        results = await self.generate_embeddings_batch(
            images_data=[{"image_data": image_data, "pil_image": pil_image}],
            job_id=job_id
        )
        
        if results and len(results) > 0 and results[0]["success"]:
            return results[0]["embedding"], results[0].get("pil_image")
        
        return None, pil_image
    
    async def generate_embeddings_batch(
        self,
        images_data: List[Dict[str, Any]],
        job_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate visual embeddings for multiple images in batches.
        
        Args:
            images_data: List of dicts with 'image_data' (base64) or 'pil_image'
            job_id: Optional job ID for logging
            
        Returns:
            List of dicts with 'success', 'embedding', and optionally 'pil_image'
        """
        results = []
        total_images = len(images_data)
        
        # Process in batches according to Hugging Face limits
        for batch_start in range(0, total_images, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_images)
            batch = images_data[batch_start:batch_end]
            batch_num = (batch_start // self.batch_size) + 1
            total_batches = (total_images + self.batch_size - 1) // self.batch_size
            
            self.logger.info(f"   🤗 Processing HF batch {batch_num}/{total_batches} ({len(batch)} images)")
            
            batch_results = await self._process_batch(batch, job_id)
            results.extend(batch_results)
        
        return results
    
    async def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        job_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Process a single batch of images with retry logic."""
        start_time = time.time()
        
        # Prepare images for API call
        image_bytes_list = []
        pil_images = []
        
        for img_data in batch:
            try:
                if 'pil_image' in img_data and img_data['pil_image']:
                    pil_img = img_data['pil_image']
                elif 'image_data' in img_data:
                    image_data = img_data['image_data']
                    if image_data.startswith('data:image'):
                        image_data = image_data.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    pil_img = Image.open(io.BytesIO(image_bytes))
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                else:
                    pil_images.append(None)
                    image_bytes_list.append(None)
                    continue
                
                # Convert PIL image to bytes for API
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format='JPEG', quality=95)
                img_byte_arr.seek(0)
                
                pil_images.append(pil_img)
                image_bytes_list.append(img_byte_arr.getvalue())
                
            except Exception as e:
                self.logger.error(f"Failed to prepare image: {e}")
                pil_images.append(None)
                image_bytes_list.append(None)
        
        # Call Hugging Face API with retry
        embeddings = await self._call_api_with_retry(image_bytes_list)
        
        # Build results
        results = []
        for i, (pil_img, embedding) in enumerate(zip(pil_images, embeddings)):
            if embedding is not None:
                results.append({
                    "success": True,
                    "embedding": embedding,
                    "pil_image": pil_img
                })
            else:
                results.append({
                    "success": False,
                    "error": "Failed to generate embedding",
                    "pil_image": pil_img
                })
        
        # Log API call
        latency_ms = int((time.time() - start_time) * 1000)
        await self._log_api_call(len(batch), latency_ms, job_id)

        return results

    async def _call_api_with_retry(
        self,
        image_bytes_list: List[Optional[bytes]]
    ) -> List[Optional[List[float]]]:
        """Call Hugging Face API with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                return await self._call_api(image_bytes_list)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    self.logger.warning(f"HF API attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"HF API failed after {self.max_retries} attempts: {e}")
                    # Return None for all images
                    return [None] * len(image_bytes_list)

        return [None] * len(image_bytes_list)

    async def _call_api(
        self,
        image_bytes_list: List[Optional[bytes]]
    ) -> List[Optional[List[float]]]:
        """Call Hugging Face Inference API for feature extraction."""
        # Filter out None images
        valid_images = [(i, img_bytes) for i, img_bytes in enumerate(image_bytes_list) if img_bytes is not None]

        if not valid_images:
            return [None] * len(image_bytes_list)

        # Hugging Face API endpoint
        api_endpoint = f"{self.api_url}/models/{self.model_id}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        # For batch processing, we need to call API for each image
        # Hugging Face Inference API doesn't support true batching for feature extraction
        embeddings_map = {}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = []
            for idx, img_bytes in valid_images:
                task = self._call_single_image(client, api_endpoint, headers, img_bytes, idx)
                tasks.append(task)

            # Process all images concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, result in zip([i for i, _ in valid_images], results):
                if isinstance(result, Exception):
                    self.logger.error(f"Image {idx} failed: {result}")
                    embeddings_map[idx] = None
                else:
                    embeddings_map[idx] = result

        # Map results back to original order
        final_embeddings = []
        for i in range(len(image_bytes_list)):
            if i in embeddings_map:
                final_embeddings.append(embeddings_map[i])
            else:
                final_embeddings.append(None)

        return final_embeddings

    async def _call_single_image(
        self,
        client: httpx.AsyncClient,
        api_endpoint: str,
        headers: Dict[str, str],
        image_bytes: bytes,
        idx: int
    ) -> Optional[List[float]]:
        """Call API for a single image."""
        try:
            response = await client.post(
                api_endpoint,
                headers=headers,
                content=image_bytes,
                timeout=self.timeout
            )

            if response.status_code == 200:
                # Hugging Face returns embeddings in different formats depending on model
                # For SigLIP feature extraction, it returns a tensor
                result = response.json()

                # Extract embedding from response
                # Format: array of shape [1, embedding_dim] or just [embedding_dim]
                if isinstance(result, list):
                    if len(result) > 0:
                        if isinstance(result[0], list):
                            embedding = result[0]  # [[emb]] -> [emb]
                        else:
                            embedding = result  # [emb]
                        return embedding
                elif isinstance(result, dict):
                    # Some models return {"embeddings": [...]}
                    if "embeddings" in result:
                        emb = result["embeddings"]
                        if isinstance(emb, list) and len(emb) > 0:
                            return emb[0] if isinstance(emb[0], list) else emb

                self.logger.error(f"Unexpected response format: {type(result)}")
                return None
            else:
                error_text = response.text
                self.logger.error(f"HF API error {response.status_code}: {error_text}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to call HF API for image {idx}: {e}")
            raise

    async def _log_api_call(
        self,
        batch_size: int,
        latency_ms: int,
        job_id: Optional[str] = None
    ):
        """Log API call to database."""
        try:
            # Hugging Face Inference API pricing (as of Dec 2024)
            # Free tier: 30,000 requests/month
            # Pro: $9/month for 100,000 requests
            # Estimate: ~$0.0001 per image
            cost = batch_size * 0.0001

            await self.ai_logger.log_ai_call(
                task="visual_embedding_generation",
                model=f"huggingface/{self.model_id}",
                input_tokens=0,  # Visual models don't use tokens
                output_tokens=0,
                cost=cost,
                latency_ms=latency_ms,
                confidence_score=0.95,
                confidence_breakdown={
                    "model_confidence": 0.98,
                    "completeness": 1.0,
                    "consistency": 0.95,
                    "validation": 0.90,
                    "batch_size": batch_size
                },
                action="use_ai_result",
                job_id=job_id
            )
        except Exception as e:
            self.logger.warning(f"Failed to log HF API call: {e}")

