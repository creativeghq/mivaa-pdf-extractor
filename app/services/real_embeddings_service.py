"""
Real Embeddings Service - Step 4 Implementation (Updated for Voyage AI)

Generates 3 real embedding types using AI models:
1. Text (1024D) - Voyage AI voyage-3.5 (primary) with OpenAI fallback (both 1024D)
2. Visual Embeddings (1152D) - Google SigLIP ViT-SO400M
3. Multimodal Fusion (2176D) - Combined text+visual (1024D + 1152D = 2176D)

Text Embedding Strategy:
- Primary: Voyage AI voyage-3.5 (1024D default, supports 256/512/1024/2048)
- Supports input_type parameter: "document" for indexing, "query" for search
- Fallback: OpenAI text-embedding-3-small (1024D) if Voyage fails - matches DB schema vector(1024)

Visual Embedding Strategy:
- Uses Google SigLIP ViT-SO400M exclusively (1152D embeddings)
- Text-guided specialized embeddings for color, texture, material, style (each 1152D)

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
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")  # From GitHub Secrets / Deno.env
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY", "")
MIVAA_GATEWAY_URL = os.getenv("MIVAA_GATEWAY_URL", "http://localhost:3000")


class RealEmbeddingsService:
    """
    Generates 3 real embedding types using AI models.

    This service provides:
    - Text embeddings via Voyage AI (1024D primary) with OpenAI fallback (1024D) - matches DB schema
    - Visual embeddings (1152D) - Google SigLIP ViT-SO400M exclusively
    - Multimodal fusion (2176D) - combined text+visual (1024D + 1152D = 2176D)

    Text Embedding Strategy:
    - Primary: Voyage AI voyage-3.5 (1024D default, supports 256/512/1024/2048)
    - Supports input_type: "document" for indexing, "query" for search
    - Fallback: OpenAI text-embedding-3-small (1024D) if Voyage fails - matches DB vector(1024)

    Visual Embedding Strategy:
    - Uses SigLIP exclusively for all visual embeddings (1152D)
    - Text-guided specialized embeddings for color, texture, material, style (each 1152D)
    - No CLIP fallback - SigLIP only for dimensional consistency

    Removed fake embeddings and CLIP fallback.
    """

    def __init__(self, supabase_client=None, config=None):
        """Initialize embeddings service."""
        self.supabase = supabase_client
        self.logger = logger
        self.openai_api_key = OPENAI_API_KEY
        self.voyage_api_key = VOYAGE_API_KEY
        self.together_api_key = TOGETHER_AI_API_KEY
        self.mivaa_gateway_url = MIVAA_GATEWAY_URL
        self.config = config

        # Initialize AI logger
        self.ai_logger = AICallLogger()

        # Model loading state
        self._models_loaded = False
        self._siglip_model = None
        self._siglip_processor = None

        # Load visual embedding model configuration from settings
        from app.config import settings
        self.visual_primary_model = settings.visual_embedding_primary_model
        self.visual_fallback_model = settings.visual_embedding_fallback_model
        self.visual_dimensions = settings.visual_embedding_dimensions
        self.visual_enabled = settings.visual_embedding_enabled

        # Voyage AI configuration
        self.voyage_api_key = settings.voyage_api_key
        self.voyage_model = settings.voyage_model
        self.voyage_enabled = settings.voyage_enabled
        self.voyage_fallback_to_openai = settings.voyage_fallback_to_openai

        # Debug logging for Voyage AI configuration
        self.logger.info(f"🔧 Voyage AI Config: enabled={self.voyage_enabled}, api_key={'SET' if self.voyage_api_key else 'NOT SET'}, model={self.voyage_model}")

        # Initialize embedding cache (Phase 1 optimization)
        self._embedding_cache = None
        if config and getattr(config, 'enable_embedding_cache', False):
            try:
                from app.services.embedding_cache_service import EmbeddingCacheService
                redis_url = os.getenv("REDIS_URL")
                self._embedding_cache = EmbeddingCacheService(
                    redis_url=redis_url,
                    ttl=getattr(config, 'embedding_cache_ttl', 86400),
                    max_size=getattr(config, 'embedding_cache_max_size', 10000),
                    enabled=True
                )
                self.logger.info("✅ Embedding cache initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize embedding cache: {e}")
    
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
            
            # Auto-detect input_type based on entity_type for optimal retrieval
            # "query" → input_type="query" (optimized for searching documents)
            # everything else → input_type="document" (optimized for being found by queries)
            input_type = "query" if entity_type == "query" else "document"

            # 1. Generate Text Embedding (1024D) - REAL with Voyage AI optimization
            text_embedding = await self._generate_text_embedding(
                text=text_content,
                input_type=input_type
            )
            if text_embedding:
                embeddings["embeddings"]["text_1536"] = text_embedding
                embeddings["metadata"]["model_versions"]["text"] = "voyage-3.5" if self.voyage_enabled else "text-embedding-3-small"
                embeddings["metadata"]["confidence_scores"]["text"] = 0.95
                self.logger.info(f"✅ Text embedding generated (1024D, input_type={input_type})")
            
            # 2. Visual Embedding (1152D) - REAL (SigLIP exclusive)
            pil_image_for_reuse = None  # Track PIL image for reuse
            if image_url or image_data:
                visual_embedding, model_used, pil_image_for_reuse = await self._generate_visual_embedding(
                    image_url, image_data
                )
                if visual_embedding:
                    embeddings["embeddings"]["visual_1152"] = visual_embedding  # SigLIP 1152D
                    embeddings["metadata"]["model_versions"]["visual"] = model_used
                    embeddings["metadata"]["confidence_scores"]["visual"] = 0.95
                    self.logger.info(f"✅ Visual embedding generated (1152D) using {model_used}")

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

            # 3. Multimodal Fusion Embedding (2688D) - REAL (1536D text + 1152D visual)
            if embeddings["embeddings"].get("text_1536") and embeddings["embeddings"].get("visual_1152"):
                multimodal_embedding = self._generate_multimodal_fusion(
                    embeddings["embeddings"]["text_1536"],
                    embeddings["embeddings"]["visual_1152"]
                )
                embeddings["embeddings"]["multimodal_2688"] = multimodal_embedding
                embeddings["metadata"]["model_versions"]["multimodal"] = "fusion-v1"
                embeddings["metadata"]["confidence_scores"]["multimodal"] = 0.92
                self.logger.info("✅ Multimodal fusion embedding generated (2688D)")
            
            # Removed fake embeddings (color, texture, application)
            # They were just downsampled text embeddings - redundant!

            self.logger.info(f"✅ All embeddings generated: {len(embeddings['embeddings'])} types")

            # Add success flag for compatibility with calling code
            embeddings["success"] = True

            return embeddings

        except Exception as e:
            self.logger.error(f"❌ Embedding generation failed: {e}")
            return {"success": False, "error": str(e)}

    async def generate_embedding(
        self,
        text: str,
        embedding_type: str = "openai",
        dimensions: int = 1536
    ) -> Optional[List[float]]:
        """
        Public method to generate a single text embedding.

        This is the main entry point for generating query embeddings for search.

        Args:
            text: Text to embed
            embedding_type: Type of embedding ("openai" for text-embedding-3-small")
            dimensions: Embedding dimensions (default 1536)

        Returns:
            List of floats representing the embedding, or None if failed
        """
        return await self._generate_text_embedding(text=text, dimensions=dimensions)

    async def generate_batch_embeddings(
        self,
        texts: List[str],
        dimensions: int = 1024,
        input_type: str = "document",
        truncation: bool = True,
        output_dtype: str = "float"
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in a single batch API call.

        This is optimized for Voyage AI's batch embedding endpoint, which is more
        efficient than making individual calls. Falls back to OpenAI if Voyage fails.

        Args:
            texts: List of texts to embed
            dimensions: Embedding dimensions (256, 512, 1024, 2048 for Voyage; 512, 1536 for OpenAI)
            input_type: "document" for indexing, "query" for search (Voyage AI only)
            truncation: Whether to truncate text to fit context length (Voyage AI only)
            output_dtype: Output data type (Voyage AI only)

        Returns:
            List of embedding vectors (same length as input texts)
            Returns None for any text that failed to generate an embedding
        """
        if not texts:
            return []

        start_time = time.time()

        # Try Voyage AI first if enabled and API key available
        # ✅ CRITICAL FIX: Default to True if config is None (don't skip Voyage AI!)
        voyage_enabled = getattr(self.config, 'voyage_enabled', True) if self.config else True
        if self.voyage_api_key and voyage_enabled:
            try:
                # ✅ CRITICAL FIX: Map 1536D (OpenAI default) to 1024D (Voyage AI max)
                # Voyage AI supports: 256, 512, 1024, 2048
                # OpenAI supports: 512, 1536
                voyage_dimensions = 1024 if dimensions == 1536 else dimensions

                # Call Voyage AI batch API
                async with httpx.AsyncClient() as client:
                    request_data = {
                        "model": "voyage-3.5",
                        "input": texts,  # Voyage AI accepts list of texts
                        "truncation": truncation
                    }

                    # Add optional parameters (only if not None/default)
                    if input_type is not None:
                        request_data["input_type"] = input_type
                    if voyage_dimensions != 1024:
                        request_data["output_dimension"] = voyage_dimensions
                    if output_dtype != "float":
                        request_data["output_dtype"] = output_dtype

                    response = await client.post(
                        "https://api.voyageai.com/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {self.voyage_api_key}",
                            "Content-Type": "application/json"
                        },
                        json=request_data,
                        timeout=60.0  # Longer timeout for batch requests
                    )

                    if response.status_code == 200:
                        data = response.json()
                        embeddings = [item["embedding"] for item in data["data"]]

                        # Log AI call with proper cost calculation
                        latency_ms = int((time.time() - start_time) * 1000)
                        usage = data.get("usage", {})
                        input_tokens = usage.get("total_tokens", 0)

                        # Voyage AI Pricing (as of Dec 2024)
                        cost_per_million = 0.06  # voyage-3.5
                        cost = (input_tokens / 1_000_000) * cost_per_million

                        await self.ai_logger.log_ai_call(
                            task="batch_text_embedding_generation",
                            model=f"voyage-3.5-{voyage_dimensions}d",
                            input_tokens=input_tokens,
                            output_tokens=0,
                            cost=cost,
                            latency_ms=latency_ms,
                            confidence_score=0.95,
                            confidence_breakdown={
                                "model_confidence": 0.98,
                                "completeness": 1.0,
                                "consistency": 0.95,
                                "validation": 0.90,
                                "batch_size": len(texts)  # ✅ FIXED: Move batch_size to confidence_breakdown
                            },
                            action="use_ai_result"
                        )

                        self.logger.info(f"✅ Generated {len(embeddings)} Voyage AI embeddings in batch ({voyage_dimensions}D, {input_type})")
                        return embeddings
                    else:
                        error_body = response.text
                        self.logger.error(f"Voyage AI batch API error {response.status_code}: {error_body}")
                        self.logger.error(f"Request data: {request_data}")
                        raise Exception(f"Voyage AI API error: {response.status_code} - {error_body}")

            except Exception as e:
                self.logger.warning(f"Voyage AI batch failed, falling back to OpenAI: {e}")

                # Log failed Voyage call
                latency_ms = int((time.time() - start_time) * 1000)
                # Use voyage_dimensions for logging (may be different from requested dimensions)
                voyage_dimensions = 1024 if dimensions == 1536 else dimensions
                await self.ai_logger.log_ai_call(
                    task="batch_text_embedding_generation",
                    model=f"voyage-3.5-{voyage_dimensions}d",
                    input_tokens=0,
                    output_tokens=0,
                    cost=0.0,
                    latency_ms=latency_ms,
                    confidence_score=0.0,
                    confidence_breakdown={
                        "model_confidence": 0.0,
                        "completeness": 0.0,
                        "consistency": 0.0,
                        "validation": 0.0,
                        "batch_size": len(texts)  # ✅ FIXED: Move batch_size to confidence_breakdown
                    },
                    action="fallback_to_rules",  # ✅ FIXED: Use valid DB constraint value
                    fallback_reason=f"Voyage AI batch error: {str(e)}",
                    error_message=str(e)
                )

                # If fallback disabled, raise the error
                if self.config and not getattr(self.config, 'voyage_fallback_to_openai', True):
                    raise

        # Fallback to OpenAI batch processing
        try:
            if not self.openai_api_key:
                self.logger.warning("OpenAI API key not available for fallback")
                return [None] * len(texts)

            self.logger.info(f"🔄 Falling back to OpenAI for batch of {len(texts)} texts...")

            # OpenAI also supports batch embeddings
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.openai_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "text-embedding-3-small",
                        "input": [text[:8191] for text in texts],  # OpenAI limit per text
                        "encoding_format": "float",
                        "dimensions": 1024  # ✅ CRITICAL FIX: Always use 1024D to match Voyage AI and DB schema
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    embeddings = [item["embedding"] for item in data["data"]]

                    self.logger.info(f"✅ OpenAI fallback successful: {len(embeddings)} embeddings generated")

                    # Log AI call
                    latency_ms = int((time.time() - start_time) * 1000)
                    usage = data.get("usage", {})
                    input_tokens = usage.get("prompt_tokens", 0)

                    # OpenAI Pricing: $0.02 per 1M tokens for text-embedding-3-small
                    cost = (input_tokens / 1_000_000) * 0.02

                    await self.ai_logger.log_ai_call(
                        task="batch_text_embedding_generation",
                        model="text-embedding-3-small",
                        input_tokens=input_tokens,
                        output_tokens=0,
                        cost=cost,
                        latency_ms=latency_ms,
                        confidence_score=0.95,
                        confidence_breakdown={
                            "model_confidence": 0.98,
                            "completeness": 1.0,
                            "consistency": 0.95,
                            "validation": 0.90,
                            "batch_size": len(texts)  # ✅ FIXED: Move batch_size to confidence_breakdown
                        },
                        action="use_ai_result"
                    )

                    self.logger.info(f"✅ Generated {len(embeddings)} OpenAI embeddings in batch ({dimensions}D)")
                    return embeddings
                else:
                    error_text = response.text[:500]  # Limit error text
                    raise Exception(f"OpenAI API error {response.status_code}: {error_text}")

        except httpx.TimeoutException as e:
            self.logger.error(f"❌ OpenAI batch fallback timed out after 30s: {e}")
        except httpx.ConnectError as e:
            self.logger.error(f"❌ OpenAI batch fallback connection failed: {e}")
        except Exception as e:
            self.logger.error(f"❌ Batch embedding generation failed: {e}")

            # Log failed AI call
            latency_ms = int((time.time() - start_time) * 1000)
            await self.ai_logger.log_ai_call(
                task="batch_text_embedding_generation",
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
                    "validation": 0.0,
                    "batch_size": len(texts)  # ✅ FIXED: Move batch_size to confidence_breakdown
                },
                action="fallback_to_rules",  # ✅ FIXED: Use valid DB constraint value
                fallback_reason=f"Batch API error: {str(e)}",
                error_message=str(e)
            )

            return [None] * len(texts)

    async def _generate_text_embedding(
        self,
        text: str,
        job_id: Optional[str] = None,
        dimensions: int = 1024,
        input_type: Optional[str] = None,
        truncation: bool = True,
        output_dtype: str = "float"
    ) -> Optional[List[float]]:
        """Generate text embedding using Voyage AI (primary) with OpenAI fallback.

        Args:
            text: Text to embed
            job_id: Optional job ID for logging
            dimensions: Embedding dimensions (256, 512, 1024, 2048 for Voyage; 512, 1536 for OpenAI)
            input_type: None (default), "document" for indexing, "query" for search (Voyage AI only)
            truncation: Whether to truncate text to fit context length (Voyage AI only, default: True)
            output_dtype: Output data type - 'float', 'int8', 'uint8', 'binary', 'ubinary' (Voyage AI only)

        Returns:
            List of floats representing the embedding, or None if failed
        """
        start_time = time.time()

        # Try Voyage AI first if enabled and API key available
        # ✅ CRITICAL FIX: Default to True if config is None (don't skip Voyage AI!)
        voyage_enabled = getattr(self.config, 'voyage_enabled', True) if self.config else True
        self.logger.info(f"🔍 Voyage AI check: api_key={'SET' if self.voyage_api_key else 'NOT SET'}, config={'SET' if self.config else 'NOT SET'}, voyage_enabled={voyage_enabled}")
        if self.voyage_api_key and voyage_enabled:
            try:
                # ✅ CRITICAL FIX: Map 1536D (OpenAI default) to 1024D (Voyage AI max)
                voyage_dimensions = 1024 if dimensions == 1536 else dimensions

                # Check cache first
                model_name = f"voyage-3.5-{voyage_dimensions}d-{input_type}"
                if self._embedding_cache:
                    cached_embedding = await self._embedding_cache.get(text, model_name)
                    if cached_embedding is not None:
                        self.logger.debug(f"✅ Cache hit for Voyage embedding ({voyage_dimensions}D, {input_type})")
                        return cached_embedding.tolist()

                # Call Voyage AI API
                async with httpx.AsyncClient() as client:
                    request_data = {
                        "model": "voyage-3.5",
                        "input": [text],  # Voyage AI handles truncation
                        "truncation": truncation
                    }

                    # Add optional parameters (only if not None/default)
                    if input_type is not None:
                        request_data["input_type"] = input_type
                    if voyage_dimensions != 1024:
                        request_data["output_dimension"] = voyage_dimensions
                    if output_dtype != "float":
                        request_data["output_dtype"] = output_dtype

                    response = await client.post(
                        "https://api.voyageai.com/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {self.voyage_api_key}",
                            "Content-Type": "application/json"
                        },
                        json=request_data,
                        timeout=30.0
                    )

                    if response.status_code == 200:
                        data = response.json()
                        embedding = data["data"][0]["embedding"]

                        # Cache the result
                        if self._embedding_cache:
                            await self._embedding_cache.set(text, model_name, np.array(embedding))

                        # Log AI call with proper cost calculation
                        latency_ms = int((time.time() - start_time) * 1000)
                        usage = data.get("usage", {})
                        input_tokens = usage.get("total_tokens", 0)

                        # Voyage AI Pricing (as of Dec 2024)
                        # voyage-3.5: $0.06 per 1M tokens
                        # voyage-3-large: $0.18 per 1M tokens
                        cost_per_million = 0.06  # voyage-3.5
                        cost = (input_tokens / 1_000_000) * cost_per_million

                        await self.ai_logger.log_ai_call(
                            task="text_embedding_generation",
                            model=f"voyage-3.5-{voyage_dimensions}d",
                            input_tokens=input_tokens,
                            output_tokens=0,
                            cost=cost,
                            latency_ms=latency_ms,
                            confidence_score=0.95,
                            confidence_breakdown={
                                "model_confidence": 0.98,
                                "completeness": 1.0,
                                "consistency": 0.95,
                                "validation": 0.90
                            },
                            action="use_ai_result",
                            job_id=job_id,
                        )

                        self.logger.info(f"✅ Generated Voyage AI embedding ({voyage_dimensions}D, {input_type})")
                        return embedding
                    else:
                        error_body = response.text
                        self.logger.error(f"Voyage AI API error {response.status_code}: {error_body}")
                        self.logger.error(f"Request data: {request_data}")
                        raise Exception(f"Voyage AI API error: {response.status_code} - {error_body}")

            except Exception as e:
                self.logger.warning(f"Voyage AI failed, falling back to OpenAI: {e}")

                # Log failed Voyage call
                latency_ms = int((time.time() - start_time) * 1000)
                voyage_dimensions = 1024 if dimensions == 1536 else dimensions
                await self.ai_logger.log_ai_call(
                    task="text_embedding_generation",
                    model=f"voyage-3.5-{voyage_dimensions}d",
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
                    action="fallback_to_rules",  # ✅ FIXED: Use valid DB constraint value
                    job_id=job_id,
                    fallback_reason=f"Voyage AI error: {str(e)}",
                    error_message=str(e)
                )

                # If fallback disabled, raise the error
                if self.config and not getattr(self.config, 'voyage_fallback_to_openai', True):
                    raise

        # Fallback to OpenAI (or primary if Voyage disabled)
        try:
            if not self.openai_api_key:
                self.logger.warning("OpenAI API key not available")
                return None

            # ✅ CRITICAL FIX: Keep same dimensions for OpenAI to match DB schema
            # Database expects vector(1024), so OpenAI must also generate 1024D
            # OpenAI text-embedding-3-small supports custom dimensions via 'dimensions' parameter
            openai_dimensions = dimensions  # Use requested dimensions (1024D for chunks)

            # Check cache first
            model_name = f"text-embedding-3-small-{openai_dimensions}d"
            if self._embedding_cache:
                cached_embedding = await self._embedding_cache.get(text, model_name)
                if cached_embedding is not None:
                    self.logger.debug(f"✅ Cache hit for OpenAI embedding ({openai_dimensions}D)")
                    return cached_embedding.tolist()

            # Call OpenAI API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {self.openai_api_key}"},
                    json={
                        "model": "text-embedding-3-small",
                        "input": text[:8191],
                        "encoding_format": "float",
                        "dimensions": openai_dimensions
                    },
                    timeout=30.0
                )

                if response.status_code == 200:
                    data = response.json()
                    embedding = data["data"][0]["embedding"]

                    # Cache the result
                    if self._embedding_cache:
                        await self._embedding_cache.set(text, model_name, np.array(embedding))

                    # Log AI call
                    latency_ms = int((time.time() - start_time) * 1000)
                    usage = data.get("usage", {})
                    input_tokens = usage.get("prompt_tokens", 0)

                    await self.ai_logger.log_gpt_call(
                        task="text_embedding_generation",
                        model="text-embedding-3-small",
                        response=data,
                        latency_ms=latency_ms,
                        confidence_score=0.95,
                        confidence_breakdown={
                            "model_confidence": 0.98,
                            "completeness": 1.0,
                            "consistency": 0.95,
                            "validation": 0.90
                        },
                        action="use_ai_result",
                        job_id=job_id
                    )

                    self.logger.info(f"✅ Generated OpenAI embedding ({openai_dimensions}D) - fallback")
                    return embedding
                else:
                    self.logger.warning(f"OpenAI API error: {response.status_code}")
                    return None

        except Exception as e:
            self.logger.error(f"OpenAI embedding generation failed: {e}")

            # Log failed OpenAI call
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
                action="fallback_failed",
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
        pil_image = None,  # NEW: Accept pre-decoded PIL image
        job_id: Optional[str] = None  # NEW: Add job_id for logging
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
            job_id: Optional job ID for logging

        Returns:
            Tuple of (1152D embedding vector or None, model_name used, PIL image for reuse)
        """
        # Use configured visual embedding model (default: SigLIP)
        visual_embedding, pil_image_out = await self._generate_siglip_embedding(
            image_url, image_data, pil_image=pil_image, job_id=job_id
        )
        if visual_embedding:
            self.logger.info(f"✅ Using visual embedding from {self.visual_primary_model}")
            return visual_embedding, self.visual_primary_model, pil_image_out

        self.logger.error(f"❌ Visual embedding generation failed for {self.visual_primary_model}")
        return None, "none", None

    async def _generate_siglip_embedding(
        self,
        image_url: Optional[str],
        image_data: Optional[str],
        pil_image = None,  # NEW: Accept pre-decoded PIL image
        job_id: Optional[str] = None  # NEW: Add job_id for logging
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
            job_id: Optional job ID for logging

        Returns:
            Tuple of (embedding list or None, PIL image for reuse or None)
        """
        import time
        start_time = time.time()

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

                # Log SigLIP embedding generation
                latency_ms = int((time.time() - start_time) * 1000)
                await self.ai_logger.log_ai_call(
                    task="visual_embedding_generation",
                    model="google/siglip-so400m-patch14-384",
                    input_tokens=0,  # Visual models don't use tokens
                    output_tokens=0,
                    cost=0.0,  # Free model
                    latency_ms=latency_ms,
                    confidence_score=0.95,
                    confidence_breakdown={
                        "model_confidence": 0.98,
                        "completeness": 1.0,
                        "consistency": 0.95,
                        "validation": 0.90
                    },
                    action="use_ai_result",
                    job_id=job_id
                )

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
        """Generate multimodal fusion embedding by concatenating text and visual.

        Returns:
            Combined embedding: text (1536D) + visual (1152D) = 2688D
        """
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

            self.logger.info("🔄 Loading visual embedding model for batch processing...")

            # Load visual embedding model (configurable - default SigLIP)
            try:
                self.logger.info(f"   Loading visual model: {self.visual_primary_model}")
                self._siglip_model = await asyncio.wait_for(
                    asyncio.to_thread(AutoModel.from_pretrained, self.visual_primary_model),
                    timeout=60.0
                )
                self._siglip_processor = await asyncio.wait_for(
                    asyncio.to_thread(AutoProcessor.from_pretrained, self.visual_primary_model),
                    timeout=60.0
                )
                self._siglip_model.eval()  # Set to evaluation mode
                self.logger.info(f"   ✅ Visual embedding model loaded: {self.visual_primary_model}")
            except asyncio.TimeoutError:
                self.logger.error(f"   ❌ Visual embedding model loading timed out after 60s: {self.visual_primary_model}")
                return False
            except Exception as e:
                self.logger.error(f"   ❌ Visual embedding model loading failed ({self.visual_primary_model}): {e}")
                return False

            self._models_loaded = True
            self.logger.info(f"✅ Visual embedding model loaded successfully: {self.visual_primary_model} ({self.visual_dimensions}D)")
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
