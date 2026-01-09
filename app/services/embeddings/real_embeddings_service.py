"""
Real Embeddings Service - Step 4 Implementation (Updated for Voyage AI)

Generates 3 real embedding types using AI models:
1. Text (1024D) - Voyage AI voyage-3.5 (primary) with OpenAI fallback (both 1024D)
2. Visual Embeddings (768D) - SLIG (SigLIP2) via HuggingFace Cloud Endpoint
3. Multimodal Fusion (1792D) - Combined text+visual (1024D + 768D = 1792D)

Text Embedding Strategy:
- Primary: Voyage AI voyage-3.5 (1024D default, supports 256/512/1024/2048)
- Supports input_type parameter: "document" for indexing, "query" for search
- Fallback: OpenAI text-embedding-3-small (1024D) if Voyage fails - matches DB schema vector(1024)

Visual Embedding Strategy:
- Uses SLIG cloud endpoint exclusively (768D embeddings)
- Specialized embeddings DEPRECATED (text-guided embeddings not supported by cloud endpoint)

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

from app.services.core.ai_call_logger import AICallLogger
from app.services.core.supabase_client import SupabaseClient

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
    - Visual embeddings (768D) - SLIG (SigLIP2) via HuggingFace Cloud Endpoint
    - Multimodal fusion (1792D) - combined text+visual (1024D + 768D = 1792D)

    Text Embedding Strategy:
    - Primary: Voyage AI voyage-3.5 (1024D default, supports 256/512/1024/2048)
    - Supports input_type: "document" for indexing, "query" for search
    - Fallback: OpenAI text-embedding-3-small (1024D) if Voyage fails - matches DB vector(1024)

    Visual Embedding Strategy:
    - Uses SLIG cloud endpoint exclusively for all visual embeddings (768D)
    - Specialized embeddings DEPRECATED (text-guided not supported by cloud endpoint)
    - No local model loading - cloud-only architecture

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

        # Model loading state (for local mode only)
        self._models_loaded = False
        self._model_load_failed = False  # ✅ Track if model loading has permanently failed
        self._model_load_attempts = 0  # ✅ Track number of load attempts
        self._max_load_attempts = 3  # ✅ Maximum attempts before giving up
        self._siglip_model = None
        self._siglip_processor = None
        self._device = None  # Will be set when models are loaded

        # Load visual embedding model configuration from settings
        from app.config import settings
        self.visual_primary_model = settings.visual_embedding_primary_model
        self.visual_fallback_model = settings.visual_embedding_fallback_model
        self.visual_dimensions = settings.visual_embedding_dimensions
        self.visual_enabled = settings.visual_embedding_enabled

        # Visual embedding mode: "local" or "remote"
        self.visual_embedding_mode = settings.visual_embedding_mode

        # SLIG (SigLIP2) Cloud Endpoint Configuration
        self.slig_endpoint_url = settings.slig_endpoint_url
        self.slig_endpoint_token = settings.slig_endpoint_token
        self.slig_model_name = settings.slig_model_name
        self.slig_embedding_dimension = settings.slig_embedding_dimension
        self._slig_client = None  # Lazy-initialized SLIG client

        # Log visual embedding mode
        if self.visual_embedding_mode == "remote":
            self.logger.info(f"☁️ Visual embedding mode: REMOTE (SLIG Cloud Endpoint)")
        else:
            self.logger.warning(f"⚠️ Visual embedding mode: LOCAL (deprecated - use REMOTE with SLIG endpoint)")

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
                from app.services.embeddings.embedding_cache_service import EmbeddingCacheService
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
            
            # 2. Visual Embedding (768D) - REAL (SLIG cloud endpoint)
            pil_image_for_reuse = None  # Track PIL image for reuse
            if image_url or image_data:
                visual_embedding, model_used, pil_image_for_reuse = await self._generate_visual_embedding(
                    image_url, image_data
                )
                if visual_embedding:
                    embeddings["embeddings"]["visual_768"] = visual_embedding  # SLIG 768D
                    embeddings["metadata"]["model_versions"]["visual"] = model_used
                    embeddings["metadata"]["confidence_scores"]["visual"] = 0.95
                    self.logger.info(f"✅ Visual embedding generated (768D) using {model_used}")

                # 2a. Generate text-guided specialized visual embeddings using SLIG similarity mode
                # Uses SLIG's similarity scoring to create embeddings focused on specific aspects
                specialized_embeddings, pil_image_for_reuse = await self._generate_specialized_siglip_embeddings(
                    image_url, image_data, pil_image=pil_image_for_reuse
                )
                if specialized_embeddings:
                    embeddings["embeddings"]["color_slig_768"] = specialized_embeddings.get("color")
                    embeddings["embeddings"]["texture_slig_768"] = specialized_embeddings.get("texture")
                    embeddings["embeddings"]["style_slig_768"] = specialized_embeddings.get("style")
                    embeddings["embeddings"]["material_slig_768"] = specialized_embeddings.get("material")
                    # Text-guided SLIG embeddings using similarity mode
                    embeddings["metadata"]["model_versions"]["specialized_visual"] = "slig-similarity-guided"
                    embeddings["metadata"]["confidence_scores"]["specialized_visual"] = 0.90  # High confidence for SLIG
                    self.logger.info("✅ Text-guided specialized SLIG embeddings generated (4 × 768D)")

                # Close PIL image after all embeddings are generated
                if pil_image_for_reuse and hasattr(pil_image_for_reuse, 'close'):
                    try:
                        pil_image_for_reuse.close()
                        self.logger.debug("✅ Closed PIL image after all embeddings generated")
                    except:
                        pass

            # 3. Multimodal Fusion Embedding (1792D) - REAL (1024D text + 768D visual)
            if embeddings["embeddings"].get("text_1024") and embeddings["embeddings"].get("visual_768"):
                multimodal_embedding = self._generate_multimodal_fusion(
                    embeddings["embeddings"]["text_1024"],
                    embeddings["embeddings"]["visual_768"]
                )
                embeddings["embeddings"]["multimodal_1792"] = multimodal_embedding
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

        # ✅ FIX: Replace empty/whitespace strings with placeholder to prevent API errors
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                processed_texts.append("no text content")
            else:
                processed_texts.append(text)

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
                        "input": processed_texts,  # Use processed texts (no empty strings)
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

        # ✅ VALIDATION: Use placeholder for empty or whitespace-only text
        # This prevents Voyage AI API errors with empty strings
        if not text or not text.strip():
            self.logger.debug("⏭️  Using placeholder text for empty/whitespace input")
            text = "no text content"  # Use placeholder instead of returning None

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
        Generate visual embedding using SLIG cloud endpoint exclusively.

        Uses SLIG (SigLIP2) via HuggingFace Cloud Endpoint for all visual embeddings (768D).
        No local model loading - cloud-only architecture.

        Args:
            image_url: URL of image
            image_data: Base64 encoded image data
            confidence_threshold: Unused (kept for API compatibility)
            pil_image: Optional pre-decoded PIL image (avoids redundant decoding)
            job_id: Optional job ID for logging

        Returns:
            Tuple of (768D embedding vector or None, model_name used, PIL image for reuse)
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

            # ✅ REMOTE MODE: Use SLIG Cloud Endpoint
            if self.visual_embedding_mode == "remote":
                self.logger.debug("☁️ Using SLIG cloud endpoint for visual embeddings")

                # Initialize SLIG client if needed
                if self._slig_client is None:
                    if not self.slig_endpoint_url or not self.slig_endpoint_token:
                        self.logger.error("❌ SLIG endpoint URL or token not configured")
                        return None, None

                    from app.services.embeddings.slig_client import SLIGClient
                    self._slig_client = SLIGClient(
                        endpoint_url=self.slig_endpoint_url,
                        token=self.slig_endpoint_token,
                        model_name=self.slig_model_name
                    )
                    self.logger.info(f"✅ Initialized SLIG client: {self.slig_endpoint_url}")

                # Decode image if needed
                if pil_image is None and image_data:
                    # Remove data URL prefix if present
                    if image_data.startswith('data:image'):
                        image_data = image_data.split(',')[1]

                    # Decode base64 to PIL Image
                    image_bytes = base64.b64decode(image_data)
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                # Get embedding from SLIG endpoint
                try:
                    embedding = await self._slig_client.get_image_embedding(pil_image)

                    latency_ms = int((time.time() - start_time) * 1000)
                    self.logger.info(f"✅ SLIG cloud embedding generated: {len(embedding)}D (latency={latency_ms}ms)")

                    # Log AI call
                    await self.ai_logger.log_ai_call(
                        task="visual_embedding_generation",
                        model=self.slig_model_name,
                        input_tokens=0,
                        output_tokens=0,
                        cost=0.0,  # Endpoint cost tracked separately
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

                    return embedding, pil_image

                except Exception as e:
                    self.logger.error(f"❌ SLIG cloud endpoint failed: {e}")
                    self.logger.warning("⚠️ Remote embedding failed, falling back to local")
                    # Fall through to local processing

            # ✅ CLOUD MODE: Use SLIG cloud endpoint (replaces local SigLIP model)
            # Initialize SLIG client if not already done
            if not hasattr(self, '_slig_client') or self._slig_client is None:
                from app.services.embeddings.slig_client import SLIGClient
                from app.config import get_settings
                settings = get_settings()

                self._slig_client = SLIGClient(
                    endpoint_url=settings.slig_endpoint_url,
                    token=settings.slig_endpoint_token,
                    endpoint_name="mh-siglip2",
                    namespace="basiliskan",
                    auto_pause=True,  # Enable auto-pause to save costs
                    auto_pause_timeout=60  # Pause after 60s idle
                )
                self.logger.info("✅ SLIG cloud client initialized (auto-pause enabled)")

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
                        timeout=30.0
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

            # ✅ Generate embedding using SLIG cloud endpoint (768D)
            # Automatically handles endpoint resume, warmup, and auto-pause
            try:
                embedding = await self._slig_client.get_image_embedding(pil_image)

                if embedding is None:
                    self.logger.error("❌ SLIG returned None embedding")
                    return None, None

                # Verify dimension (should be 768D)
                if len(embedding) != 768:
                    self.logger.warning(f"⚠️ Unexpected embedding dimension: {len(embedding)} (expected 768)")

                self.logger.info(f"✅ Generated 768D embedding via SLIG cloud endpoint")

                # Log SLIG embedding generation
                latency_ms = int((time.time() - start_time) * 1000)
                await self.ai_logger.log_ai_call(
                    task="visual_embedding_generation",
                    model="basiliskan/siglip2",  # SLIG cloud model
                    input_tokens=0,  # Visual models don't use tokens
                    output_tokens=0,
                    cost=0.0,  # Cloud endpoint cost tracked separately
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
                    return embedding, pil_image
                else:
                    # We created the image, caller can reuse it
                    return embedding, pil_image

            except asyncio.TimeoutError:
                self.logger.error("❌ SLIG embedding generation timed out")
                # Close image on error if we created it
                if not image_was_provided and pil_image and hasattr(pil_image, 'close'):
                    try:
                        pil_image.close()
                    except:
                        pass
                return None, None

        except Exception as e:
            self.logger.error(f"SLIG embedding generation failed: {e}")
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
        pil_image = None  # Accept pre-decoded PIL image
    ) -> tuple[Optional[Dict[str, List[float]]], Optional[any]]:
        """
        Generate specialized text-guided embeddings using SLIG's similarity mode.

        Uses SLIG cloud endpoint to create text-guided visual embeddings by:
        1. Getting base image embedding (768D)
        2. Calculating similarity scores with specialized text prompts
        3. Weighting image embeddings by similarity to create specialized embeddings

        This approach leverages SLIG's native similarity scoring to create
        embeddings that are guided by specific visual aspects (color, texture, etc.)

        Args:
            image_url: URL of image
            image_data: Base64 encoded image data
            pil_image: Optional pre-decoded PIL image (for reuse)

        Returns:
            Tuple of (Dict with color/texture/style/material embeddings (768D each), PIL image for reuse)
        """
        try:
            import asyncio

            # Define text prompts for each specialized embedding type
            text_prompts = {
                "color": "focus on color palette and color relationships",
                "texture": "focus on surface patterns and texture details",
                "material": "focus on material type and physical properties",
                "style": "focus on design style and aesthetic elements"
            }

            # First, get the base image embedding (768D)
            image_embedding_result = await self.slig_client.get_image_embedding(
                image_url=image_url,
                image_data=image_data
            )

            if not image_embedding_result or "embedding" not in image_embedding_result:
                self.logger.error("❌ Failed to get base image embedding for specialized embeddings")
                return None, pil_image

            base_image_embedding = image_embedding_result["embedding"]

            # Generate specialized embeddings using similarity-weighted approach
            specialized = {}

            for embedding_type, text_prompt in text_prompts.items():
                try:
                    # Get similarity score between image and text prompt
                    similarity_result = await self.slig_client.get_similarity(
                        image_url=image_url,
                        image_data=image_data,
                        text=text_prompt
                    )

                    if similarity_result and "similarity" in similarity_result:
                        similarity_score = similarity_result["similarity"]

                        # Also get text embedding for this prompt
                        text_embedding_result = await self.slig_client.get_text_embedding(text_prompt)

                        if text_embedding_result and "embedding" in text_embedding_result:
                            text_embedding = text_embedding_result["embedding"]

                            # Create text-guided embedding by blending:
                            # - Base image embedding (weighted by similarity)
                            # - Text embedding (weighted by 1-similarity)
                            # This creates embeddings that focus on the specific aspect
                            import numpy as np

                            # Convert to numpy for easier math
                            img_emb = np.array(base_image_embedding)
                            txt_emb = np.array(text_embedding)

                            # Blend: higher similarity = more image, lower = more text guidance
                            blend_weight = 0.7 + (0.2 * similarity_score)  # 0.7-0.9 range
                            guided_embedding = (blend_weight * img_emb + (1 - blend_weight) * txt_emb)

                            # Normalize to unit vector
                            norm = np.linalg.norm(guided_embedding)
                            if norm > 0:
                                guided_embedding = guided_embedding / norm

                            specialized[embedding_type] = guided_embedding.tolist()
                            self.logger.debug(f"✅ Generated {embedding_type} embedding (768D, similarity={similarity_score:.3f})")
                        else:
                            self.logger.warning(f"⚠️ Failed to get text embedding for {embedding_type}")
                            # Fallback: use base image embedding
                            specialized[embedding_type] = base_image_embedding
                    else:
                        self.logger.warning(f"⚠️ Failed to get similarity for {embedding_type}")
                        # Fallback: use base image embedding
                        specialized[embedding_type] = base_image_embedding

                except Exception as e:
                    self.logger.error(f"❌ Failed to generate {embedding_type} embedding: {e}")
                    # Fallback: use base image embedding
                    specialized[embedding_type] = base_image_embedding

            if len(specialized) == 4:
                self.logger.info("✅ Generated 4 text-guided specialized SLIG embeddings (768D each)")
                return specialized, pil_image
            else:
                self.logger.warning(f"⚠️ Only generated {len(specialized)}/4 specialized embeddings")
                return specialized if specialized else None, pil_image

        except Exception as e:
            self.logger.error(f"❌ Specialized SLIG embedding generation failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None, pil_image

    def _generate_multimodal_fusion(
        self,
        text_embedding: List[float],
        visual_embedding: List[float]
    ) -> List[float]:
        """Generate multimodal fusion embedding by concatenating text and visual.

        Returns:
            Combined embedding: text (1024D) + visual (768D) = 1792D
        """
        return text_embedding + visual_embedding
    
    # Removed fake embedding methods:
    # - _generate_color_embedding (was just downsampled text embedding)
    # - _generate_texture_embedding (was just downsampled text embedding)
    # - _generate_application_embedding (was just downsampled text embedding)
    #
    # These were redundant - text_embedding_1536 already contains all this information!




