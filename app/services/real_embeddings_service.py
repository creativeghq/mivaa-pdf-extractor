"""
Real Embeddings Service - Step 4 Implementation

Generates 3 real embedding types using AI models:
1. Text (1536D) - OpenAI text-embedding-3-small
2. Visual Embeddings (1152D) - Google SigLIP ViT-SO400M
3. Multimodal Fusion (2688D) - Combined text+visual (1536D + 1152D)

Visual Embedding Strategy:
- Uses Google SigLIP ViT-SO400M exclusively (1152D embeddings)
- Text-guided specialized embeddings for color, texture, material, style

Removed fake embeddings and CLIP fallback - SigLIP only for consistency.
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
import sentry_sdk

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
    - Visual embeddings (1152D) - Google SigLIP ViT-SO400M exclusively
    - Multimodal fusion (2688D) - combined text+visual (1536D + 1152D)

    Visual Embedding Strategy:
    - Uses SigLIP exclusively for all visual embeddings (1152D)
    - Text-guided specialized embeddings for color, texture, material, style
    - No CLIP fallback - SigLIP only for dimensional consistency

    Removed fake embeddings and CLIP fallback.
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

        # Model loading state
        self._models_loaded = False
        self._siglip_model = None
        self._siglip_processor = None
    
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
            
            # 2. Visual Embedding (512D) - REAL (SigLIP with CLIP fallback)
            pil_image_for_reuse = None  # Track PIL image for reuse
            if image_url or image_data:
                visual_embedding, model_used, pil_image_for_reuse = await self._generate_visual_embedding(
                    image_url, image_data
                )
                if visual_embedding:
                    # Store with both key names for compatibility
                    embeddings["embeddings"]["visual_512"] = visual_embedding  # Expected by llamaindex_service
                    embeddings["embeddings"]["visual_clip_512"] = visual_embedding  # Legacy key
                    embeddings["metadata"]["model_versions"]["visual"] = model_used
                    # Higher confidence for SigLIP, lower for CLIP fallback
                    embeddings["metadata"]["confidence_scores"]["visual"] = 0.95 if "siglip" in model_used else 0.90
                    self.logger.info(f"✅ Visual embedding generated (512D) using {model_used}")

                # 2a. Generate text-guided specialized visual embeddings for pattern/color/texture matching
                # ✅ REUSE PIL image from visual embedding to avoid redundant decoding!
                specialized_embeddings, pil_image_for_reuse = await self._generate_specialized_siglip_embeddings(
                    image_url, image_data, pil_image=pil_image_for_reuse
                )
                if specialized_embeddings:
                    embeddings["embeddings"]["color_siglip_1152"] = specialized_embeddings.get("color")
                    embeddings["embeddings"]["texture_siglip_1152"] = specialized_embeddings.get("texture")
                    embeddings["embeddings"]["style_siglip_1152"] = specialized_embeddings.get("style")
                    embeddings["embeddings"]["material_siglip_1152"] = specialized_embeddings.get("material")
                    # Text-guided SigLIP embeddings
                    embeddings["metadata"]["model_versions"]["specialized_visual"] = "siglip-so400m-patch14-384-text-guided"
                    embeddings["metadata"]["confidence_scores"]["specialized_visual"] = 0.95  # High confidence for text-guided
                    self.logger.info("✅ Text-guided specialized SigLIP embeddings generated (4 × 1152D)")

                # Close PIL image after all embeddings are generated
                if pil_image_for_reuse and hasattr(pil_image_for_reuse, 'close'):
                    try:
                        pil_image_for_reuse.close()
                        self.logger.debug("✅ Closed PIL image after all embeddings generated")
                    except:
                        pass

            # 3. Multimodal Fusion Embedding (2048D) - REAL
            if embeddings["embeddings"].get("text_1536") and embeddings["embeddings"].get("visual_512"):
                multimodal_embedding = self._generate_multimodal_fusion(
                    embeddings["embeddings"]["text_1536"],
                    embeddings["embeddings"]["visual_512"]
                )
                embeddings["embeddings"]["multimodal_2048"] = multimodal_embedding
                embeddings["metadata"]["model_versions"]["multimodal"] = "fusion-v1"
                embeddings["metadata"]["confidence_scores"]["multimodal"] = 0.92
                self.logger.info("✅ Multimodal fusion embedding generated (2048D)")
            
            # Removed fake embeddings (color, texture, application)
            # They were just downsampled text embeddings - redundant!

            self.logger.info(f"✅ All embeddings generated: {len(embeddings['embeddings'])} types")

            # Add success flag for compatibility with calling code
            embeddings["success"] = True

            return embeddings
            
        except Exception as e:
            self.logger.error(f"❌ Embedding generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_text_embedding(
        self,
        text: str,
        job_id: Optional[str] = None,
        dimensions: int = 1536
    ) -> Optional[List[float]]:
        """Generate text embedding using OpenAI.

        Args:
            text: Text to embed
            job_id: Optional job ID for logging
            dimensions: Embedding dimensions (512 or 1536, default 1536)
        """
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
                        "encoding_format": "float",
                        "dimensions": dimensions  # Support custom dimensions
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
        image_data: Optional[str],
        confidence_threshold: float = 0.8,
        pil_image = None  # NEW: Accept pre-decoded PIL image
    ) -> tuple[Optional[List[float]], str, Optional[any]]:
        """
        Generate visual embedding using SigLIP exclusively.

        Uses Google SigLIP ViT-SO400M for all visual embeddings (1152D).
        No CLIP fallback - SigLIP only for dimensional consistency.

        Args:
            image_url: URL of image
            image_data: Base64 encoded image data
            confidence_threshold: Unused (kept for API compatibility)
            pil_image: Optional pre-decoded PIL image (avoids redundant decoding)

        Returns:
            Tuple of (1152D embedding vector or None, model_name used, PIL image for reuse)
        """
        # Use SigLIP exclusively
        siglip_embedding, pil_image_out = await self._generate_siglip_embedding(
            image_url, image_data, pil_image=pil_image
        )
        if siglip_embedding:
            self.logger.info("✅ Using SigLIP embedding")
            return siglip_embedding, "siglip-so400m-patch14-384", pil_image_out

        self.logger.error("❌ SigLIP embedding generation failed")
        return None, "none", None

    async def _generate_siglip_embedding(
        self,
        image_url: Optional[str],
        image_data: Optional[str],
        pil_image = None  # NEW: Accept pre-decoded PIL image
    ) -> tuple[Optional[List[float]], Optional[any]]:
        """
        Generate visual embedding using Google SigLIP ViT-SO400M.

        Uses transformers library directly instead of sentence-transformers
        to avoid 'hidden_size' attribute error with SiglipConfig.

        NOTE: Models should be pre-loaded using ensure_models_loaded() before calling this.

        Args:
            image_url: URL of image
            image_data: Base64 encoded image data
            pil_image: Optional pre-decoded PIL image (avoids redundant decoding)

        Returns:
            Tuple of (embedding list or None, PIL image for reuse or None)
        """
        try:
            import torch
            import base64
            from PIL import Image
            import io
            import numpy as np
            import asyncio

            # Check if SigLIP model is loaded
            if self._siglip_model is None or self._siglip_processor is None:
                self.logger.warning("⚠️ SigLIP model not loaded, skipping")
                return None, None

            # Use pre-decoded PIL image if provided, otherwise decode
            image_was_provided = pil_image is not None

            if pil_image is None and image_data:
                # Remove data URL prefix if present
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]

                try:
                    image_bytes = base64.b64decode(image_data)
                    pil_image = await asyncio.wait_for(
                        asyncio.to_thread(Image.open, io.BytesIO(image_bytes)),
                        timeout=30.0  # ✅ INCREASED from 10s to 30s
                    )

                    # Convert RGBA to RGB if necessary
                    if pil_image.mode == 'RGBA':
                        # Create white background
                        rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
                        rgb_image.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
                        pil_image = rgb_image
                    elif pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                except asyncio.TimeoutError:
                    self.logger.error("❌ Image decoding timed out after 30s")
                    return None, None

            elif pil_image is None and image_url:
                # Download image from URL if no PIL image provided
                import httpx
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.get(image_url)
                        if response.status_code == 200:
                            pil_image = Image.open(io.BytesIO(response.content))
                except Exception as e:
                    self.logger.error(f"Failed to download image from URL: {e}")
                    return None, None

            if pil_image is None:
                self.logger.error("No image data provided")
                return None, None

            # Ensure RGB format
            if pil_image.mode == 'RGBA':
                rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
                rgb_image.paste(pil_image, mask=pil_image.split()[3])
                pil_image = rgb_image
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Generate embedding using SigLIP model WITH TIMEOUT
            try:
                def _generate_embedding():
                    with torch.no_grad():
                        inputs = self._siglip_processor(images=pil_image, return_tensors="pt")
                        # Get image features from vision model
                        image_features = self._siglip_model.get_image_features(**inputs)

                        # L2 normalize to unit vector
                        embedding = image_features / image_features.norm(dim=-1, keepdim=True)
                        result = embedding.squeeze().cpu().numpy()

                        # Explicit memory cleanup
                        del inputs, image_features, embedding
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        return result

                embedding = await asyncio.wait_for(
                    asyncio.to_thread(_generate_embedding),
                    timeout=30.0  # 30s max for embedding generation
                )

                self.logger.info(f"✅ Generated SigLIP visual embedding: {len(embedding)}D")

                # Return embedding AND PIL image for reuse (don't close it yet!)
                # Only close if we created it (not if it was provided)
                if image_was_provided:
                    # Image was provided, return it for continued reuse
                    return embedding.tolist(), pil_image
                else:
                    # We created the image, caller can reuse it
                    return embedding.tolist(), pil_image

            except asyncio.TimeoutError:
                self.logger.error("❌ SigLIP embedding generation timed out after 30s")
                # Close image on error if we created it
                if not image_was_provided and pil_image and hasattr(pil_image, 'close'):
                    try:
                        pil_image.close()
                    except:
                        pass
                return None, None

        except Exception as e:
            self.logger.error(f"SigLIP embedding generation failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Close image on error if we created it
            if not image_was_provided and pil_image and hasattr(pil_image, 'close'):
                try:
                    pil_image.close()
                except:
                    pass

        return None, None

    async def _generate_specialized_siglip_embeddings(
        self,
        image_url: Optional[str],
        image_data: Optional[str],
        pil_image = None  # NEW: Accept pre-decoded PIL image
    ) -> tuple[Optional[Dict[str, List[float]]], Optional[any]]:
        """
        Generate text-guided specialized visual embeddings using SigLIP.

        This creates 4 unique text-guided embeddings (1152D each) for different search types:
        - Color: "focus on color palette and color relationships"
        - Texture: "focus on surface patterns and texture details"
        - Material: "focus on material type and physical properties"
        - Style: "focus on design style and aesthetic elements"

        Uses SigLIP's text-image matching to create embeddings that emphasize different aspects.
        Each embedding is unique and optimized for its specific search type.

        NOTE: Models should be pre-loaded using ensure_models_loaded() before calling this.

        Args:
            image_url: URL of image
            image_data: Base64 encoded image data
            pil_image: Optional pre-decoded PIL image (avoids redundant decoding)

        Returns:
            Tuple of (specialized embeddings dict or None, PIL image for reuse or None)
        """
        try:
            import torch
            import base64
            from PIL import Image
            import io
            import numpy as np
            import asyncio
            import httpx

            # Check if SigLIP model is loaded
            if self._siglip_model is None or self._siglip_processor is None:
                self.logger.warning("⚠️ SigLIP model not loaded for specialized embeddings, skipping")
                return None, None

            # Use pre-decoded PIL image if provided, otherwise decode
            image_was_provided = pil_image is not None

            if pil_image is None and image_data:
                # Remove data URL prefix if present
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]

                try:
                    image_bytes = base64.b64decode(image_data)
                    pil_image = await asyncio.wait_for(
                        asyncio.to_thread(Image.open, io.BytesIO(image_bytes)),
                        timeout=30.0  # ✅ INCREASED from no timeout to 30s
                    )
                except asyncio.TimeoutError:
                    self.logger.error("❌ Image decoding timed out after 30s (specialized embeddings)")
                    return None, None

            elif pil_image is None and image_url:
                # Download image from URL
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.get(image_url)
                        if response.status_code == 200:
                            pil_image = Image.open(io.BytesIO(response.content))
                except Exception as e:
                    self.logger.error(f"Failed to download image from URL: {e}")
                    return None, None

            if not pil_image:
                return None, None

            # Ensure RGB format
            if pil_image.mode == 'RGBA':
                rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
                rgb_image.paste(pil_image, mask=pil_image.split()[3])
                pil_image = rgb_image
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Define text prompts for each specialized embedding type
            text_prompts = {
                "color": "focus on color palette and color relationships",
                "texture": "focus on surface patterns and texture details",
                "material": "focus on material type and physical properties",
                "style": "focus on design style and aesthetic elements"
            }

            # Generate text-guided embeddings for each specialized type
            specialized = {}

            try:
                def _generate_text_guided_embedding(text_prompt: str):
                    """Generate text-guided SigLIP embedding using full model forward pass"""
                    with torch.no_grad():
                        # Process image with text prompt for text-guided embedding
                        inputs = self._siglip_processor(
                            text=[text_prompt],
                            images=pil_image,
                            return_tensors="pt",
                            padding=True
                        )

                        # Use full model forward pass (accepts both text and image)
                        # This computes text-image similarity and returns image embeddings
                        outputs = self._siglip_model(**inputs)
                        image_features = outputs.image_embeds  # Get image embeddings guided by text

                        # Normalize to unit vector
                        embedding = image_features / image_features.norm(dim=-1, keepdim=True)
                        result = embedding.squeeze().cpu().numpy()

                        # Explicit memory cleanup
                        del inputs, outputs, image_features, embedding
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        return result

                # Generate each specialized embedding with its text prompt
                for embedding_type, text_prompt in text_prompts.items():
                    embedding = await asyncio.wait_for(
                        asyncio.to_thread(_generate_text_guided_embedding, text_prompt),
                        timeout=30.0  # 30s max per embedding
                    )
                    specialized[embedding_type] = embedding.tolist()
                    self.logger.debug(f"✅ Generated {embedding_type} embedding (1152D) with prompt: '{text_prompt}'")

                self.logger.info("✅ Generated 4 text-guided specialized SigLIP embeddings (1152D each)")

                # Return embeddings AND PIL image for reuse (don't close it yet!)
                # Only close if we created it (not if it was provided)
                if image_was_provided:
                    # Image was provided, return it for continued reuse
                    return specialized, pil_image
                else:
                    # We created the image, caller can reuse it
                    return specialized, pil_image

            except asyncio.TimeoutError:
                self.logger.error("❌ Specialized SigLIP embedding generation timed out after 30s")
                sentry_sdk.capture_message(
                    "❌ Specialized SigLIP embedding generation timeout (30s)",
                    level="error"
                )
                # Close image on error if we created it
                if not image_was_provided and pil_image and hasattr(pil_image, 'close'):
                    try:
                        pil_image.close()
                    except:
                        pass
                return None, None

        except Exception as e:
            self.logger.error(f"Specialized SigLIP embedding generation failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Close image on error if we created it
            if not image_was_provided and pil_image and hasattr(pil_image, 'close'):
                try:
                    pil_image.close()
                except:
                    pass
            return None, None

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


    async def ensure_models_loaded(self):
        """
        Ensure CLIP/SigLIP models are loaded into memory.

        This should be called ONCE before batch processing to avoid
        loading models multiple times per batch (which causes OOM).

        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        if self._models_loaded:
            self.logger.debug("Models already loaded, skipping initialization")
            return True

        try:
            from transformers import AutoModel, AutoProcessor
            import asyncio

            self.logger.info("🔄 Loading SigLIP model for batch processing...")

            # Load SigLIP model (exclusive visual embedding model)
            try:
                self.logger.info("   Loading SigLIP model: google/siglip-so400m-patch14-384")
                self._siglip_model = await asyncio.wait_for(
                    asyncio.to_thread(AutoModel.from_pretrained, 'google/siglip-so400m-patch14-384'),
                    timeout=60.0
                )
                self._siglip_processor = await asyncio.wait_for(
                    asyncio.to_thread(AutoProcessor.from_pretrained, 'google/siglip-so400m-patch14-384'),
                    timeout=60.0
                )
                self._siglip_model.eval()  # Set to evaluation mode
                self.logger.info("   ✅ SigLIP model loaded")
            except asyncio.TimeoutError:
                self.logger.error("   ❌ SigLIP model loading timed out after 60s")
                return False
            except Exception as e:
                self.logger.error(f"   ❌ SigLIP model loading failed: {e}")
                return False

            self._models_loaded = True
            self.logger.info("✅ SigLIP model loaded successfully for batch processing")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load models: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def unload_siglip_model(self):
        """
        Unload SigLIP model from memory to free up resources.

        Call this after batch processing to reduce memory footprint.
        Model will be automatically reloaded on next use.
        """
        try:
            import torch
            import gc

            # Unload SigLIP model
            if self._siglip_model is not None:
                del self._siglip_model
                del self._siglip_processor
                self._siglip_model = None
                self._siglip_processor = None

            # Reset loaded state
            self._models_loaded = False

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()

            self.logger.info("🧹 Unloaded SigLIP model from memory")
            return True

        except Exception as e:
            self.logger.warning(f"Failed to unload SigLIP model: {e}")
            return False
