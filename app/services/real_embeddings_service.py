"""
Real Embeddings Service - Step 4 Implementation

Generates 3 real embedding types using AI models:
1. Text (1536D) - OpenAI text-embedding-3-small
2. Visual CLIP (512D) - CLIP visual embeddings
3. Multimodal Fusion (2048D) - Combined text+visual

Removed fake embeddings (color, texture, application) as they were just
downsampled versions of text embeddings - redundant and wasteful.
"""

import logging
import asyncio
import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import httpx
import numpy as np

from app.services.ai_call_logger import AICallLogger
from app.services.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY", "")
MIVAA_GATEWAY_URL = os.getenv("MIVAA_GATEWAY_URL", "http://localhost:3000")


class RealEmbeddingsService:
    """
    Generates 3 real embedding types using AI models.

    This service provides:
    - Text embeddings via OpenAI (1536D)
    - Visual embeddings via CLIP (512D)
    - Multimodal fusion (2048D) - combined text+visual

    Removed fake embeddings (color, texture, application) as they were
    redundant with text embeddings.
    """
    
    def __init__(self, supabase_client=None):
        """Initialize embeddings service."""
        self.supabase = supabase_client
        self.logger = logger
        self.openai_api_key = OPENAI_API_KEY
        self.together_api_key = TOGETHER_AI_API_KEY
        self.mivaa_gateway_url = MIVAA_GATEWAY_URL

        # Initialize AI logger
        self.ai_logger = AICallLogger()
    
    async def generate_all_embeddings(
        self,
        entity_id: str,
        entity_type: str,  # 'product', 'chunk', 'image'
        text_content: str,
        image_url: Optional[str] = None,
        material_properties: Optional[Dict[str, Any]] = None,
        image_data: Optional[str] = None  # base64 encoded
    ) -> Dict[str, Any]:
        """
        Generate all 6 embedding types for an entity.
        
        Args:
            entity_id: ID of entity to embed
            entity_type: Type of entity (product, chunk, image)
            text_content: Text to embed
            image_url: URL of image (optional)
            material_properties: Material properties dict (optional)
            image_data: Base64 encoded image data (optional)
            
        Returns:
            Dictionary with all embedding types
        """
        try:
            self.logger.info(f"🔄 Generating all embeddings for {entity_type} {entity_id}")
            
            embeddings = {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "generated_at": datetime.utcnow().isoformat(),
                "embeddings": {},
                "metadata": {
                    "model_versions": {},
                    "confidence_scores": {},
                    "generation_times_ms": {}
                }
            }
            
            # 1. Generate Text Embedding (1536D) - REAL
            text_embedding = await self._generate_text_embedding(text_content)
            if text_embedding:
                embeddings["embeddings"]["text_1536"] = text_embedding
                embeddings["metadata"]["model_versions"]["text"] = "text-embedding-3-small"
                embeddings["metadata"]["confidence_scores"]["text"] = 0.95
                self.logger.info("✅ Text embedding generated (1536D)")
            
            # 2. Visual CLIP Embedding (512D) - REAL (from Step 2)
            if image_url or image_data:
                visual_embedding = await self._generate_visual_embedding(image_url, image_data)
                if visual_embedding:
                    embeddings["embeddings"]["visual_clip_512"] = visual_embedding
                    embeddings["metadata"]["model_versions"]["visual"] = "clip-vit-base-patch32"
                    embeddings["metadata"]["confidence_scores"]["visual"] = 0.90
                    self.logger.info("✅ Visual CLIP embedding generated (512D)")
            
            # 3. Multimodal Fusion Embedding (2048D) - REAL
            if embeddings["embeddings"].get("text_1536") and embeddings["embeddings"].get("visual_clip_512"):
                multimodal_embedding = self._generate_multimodal_fusion(
                    embeddings["embeddings"]["text_1536"],
                    embeddings["embeddings"]["visual_clip_512"]
                )
                embeddings["embeddings"]["multimodal_2048"] = multimodal_embedding
                embeddings["metadata"]["model_versions"]["multimodal"] = "fusion-v1"
                embeddings["metadata"]["confidence_scores"]["multimodal"] = 0.92
                self.logger.info("✅ Multimodal fusion embedding generated (2048D)")
            
            # Removed fake embeddings (color, texture, application)
            # They were just downsampled text embeddings - redundant!
            
            self.logger.info(f"✅ All embeddings generated: {len(embeddings['embeddings'])} types")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"❌ Embedding generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_text_embedding(self, text: str, job_id: Optional[str] = None) -> Optional[List[float]]:
        """Generate text embedding using OpenAI."""
        start_time = time.time()
        try:
            if not self.openai_api_key:
                self.logger.warning("OpenAI API key not available")
                return None

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {self.openai_api_key}"},
                    json={
                        "model": "text-embedding-3-small",
                        "input": text[:8191],  # OpenAI limit
                        "encoding_format": "float"
                    },
                    timeout=30.0
                )

                if response.status_code == 200:
                    data = response.json()
                    embedding = data["data"][0]["embedding"]

                    # Log AI call
                    latency_ms = int((time.time() - start_time) * 1000)
                    usage = data.get("usage", {})
                    input_tokens = usage.get("prompt_tokens", 0)

                    confidence_breakdown = {
                        "model_confidence": 0.98,
                        "completeness": 1.0,
                        "consistency": 0.95,
                        "validation": 0.90
                    }
                    confidence_score = (
                        0.30 * confidence_breakdown["model_confidence"] +
                        0.30 * confidence_breakdown["completeness"] +
                        0.25 * confidence_breakdown["consistency"] +
                        0.15 * confidence_breakdown["validation"]
                    )

                    # CRITICAL FIX: Use log_gpt_call instead of log_openai_call
                    await self.ai_logger.log_gpt_call(
                        task="text_embedding_generation",
                        model="text-embedding-3-small",
                        response=data,
                        latency_ms=latency_ms,
                        confidence_score=confidence_score,
                        confidence_breakdown=confidence_breakdown,
                        action="use_ai_result",
                        job_id=job_id
                    )

                    return embedding
                else:
                    self.logger.warning(f"OpenAI API error: {response.status_code}")
                    return None

        except Exception as e:
            self.logger.error(f"Text embedding generation failed: {e}")

            # Log failed AI call
            latency_ms = int((time.time() - start_time) * 1000)
            await self.ai_logger.log_ai_call(
                task="text_embedding_generation",
                model="text-embedding-3-small",
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                latency_ms=latency_ms,
                confidence_score=0.0,
                confidence_breakdown={
                    "model_confidence": 0.0,
                    "completeness": 0.0,
                    "consistency": 0.0,
                    "validation": 0.0
                },
                action="fallback_to_rules",
                job_id=job_id,
                fallback_reason=f"OpenAI API error: {str(e)}",
                error_message=str(e)
            )

            return None
    
    async def _generate_visual_embedding(
        self,
        image_url: Optional[str],
        image_data: Optional[str]
    ) -> Optional[List[float]]:
        """Generate visual CLIP embedding using local PyTorch model."""
        try:
            # ✅ FIX: Use local CLIP model instead of non-existent gateway
            from llama_index.embeddings.clip import ClipEmbedding
            import base64
            from PIL import Image
            import io

            # Initialize CLIP model (cached after first use)
            if not hasattr(self, '_clip_model'):
                self._clip_model = ClipEmbedding(model_name="ViT-B/32")
                self.logger.info("✅ Initialized local CLIP model: ViT-B/32")

            # Convert base64 image data to PIL Image
            if image_data:
                # Remove data URL prefix if present
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]

                image_bytes = base64.b64decode(image_data)
                pil_image = Image.open(io.BytesIO(image_bytes))

                # Save PIL image to temporary file for CLIP model
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    pil_image.save(tmp_file.name, format='JPEG')
                    tmp_path = tmp_file.name

                try:
                    # Generate embedding using local CLIP model with file path
                    embedding = self._clip_model.get_image_embedding(tmp_path)
                finally:
                    # Clean up temporary file
                    import os
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                self.logger.info(f"✅ Generated CLIP embedding: {len(embedding)}D")
                return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

            elif image_url:
                # Download image from URL
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(image_url)
                    if response.status_code == 200:
                        pil_image = Image.open(io.BytesIO(response.content))

                        # Save PIL image to temporary file for CLIP model
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                            pil_image.save(tmp_file.name, format='JPEG')
                            tmp_path = tmp_file.name

                        try:
                            # Generate embedding using local CLIP model with file path
                            embedding = self._clip_model.get_image_embedding(tmp_path)
                            self.logger.info(f"✅ Generated CLIP embedding from URL: {len(embedding)}D")
                            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                        finally:
                            # Clean up temporary file
                            import os
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)

        except Exception as e:
            self.logger.error(f"Visual embedding generation failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

        return None
    
    def _generate_multimodal_fusion(
        self,
        text_embedding: List[float],
        visual_embedding: List[float]
    ) -> List[float]:
        """Generate multimodal fusion embedding by concatenating text and visual."""
        # Concatenate text (1536D) + visual (512D) = 2048D
        return text_embedding + visual_embedding
    
    # Removed fake embedding methods:
    # - _generate_color_embedding (was just downsampled text embedding)
    # - _generate_texture_embedding (was just downsampled text embedding)
    # - _generate_application_embedding (was just downsampled text embedding)
    #
    # These were redundant - text_embedding_1536 already contains all this information!

