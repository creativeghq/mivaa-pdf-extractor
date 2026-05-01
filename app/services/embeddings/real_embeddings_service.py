"""
Real Embeddings Service - Step 4 Implementation (Updated for Voyage AI)

Generates embedding types using AI models:
1. Text (1024D) - Voyage AI voyage-4 (primary) with OpenAI fallback (both 1024D)
2. Visual Embeddings (768D) - SLIG (SigLIP2) via HuggingFace Cloud Endpoint
3. Understanding (1024D) - Qwen vision_analysis JSON → Voyage AI text embedding
4. Multimodal Fusion (1792D) - Combined text+visual (1024D + 768D = 1792D)

Text Embedding Strategy:
- Primary: Voyage AI voyage-4 (1024D default, supports 256/512/1024/2048)
- Supports input_type parameter: "document" for indexing, "query" for search
- Fallback: OpenAI text-embedding-3-small (1024D) if Voyage fails - matches DB schema vector(1024)

Visual Embedding Strategy:
- Uses SLIG cloud endpoint exclusively (768D embeddings)
- Cloud-only architecture - no local model loading

Understanding Embedding Strategy:
- Embeds Qwen3-VL's structured vision_analysis JSON as descriptive text via Voyage AI (1024D)
- Enables spec-based search (e.g., "porcelain tile 60x120cm", "R10 slip rating")
"""

import logging
import asyncio


def _is_valid_vision_analysis_schema(vision_analysis: Any) -> bool:  # type: ignore[name-defined]
    """Reject malformed Qwen vision_analysis payloads (audit fix #33).

    Without this, error envelopes like {"error": "OOM", "message": "..."} would
    survive the dict-extraction in `_vision_analysis_to_text`, produce an
    almost-empty text payload, and Voyage would happily return a degenerate
    embedding that matches every search query.
    """
    if not isinstance(vision_analysis, dict):
        return False
    # An error envelope is the most common failure shape.
    if 'error' in vision_analysis and 'material_type' not in vision_analysis:
        return False
    # Require at least one of the expected describing fields. The schema is
    # forgiving — Qwen sometimes drops keys — but if NONE of these are present
    # the payload can't yield meaningful descriptive text.
    expected_any = ('material_type', 'category', 'colors', 'textures',
                    'finish', 'surface_pattern', 'description')
    return any(k in vision_analysis for k in expected_any)
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
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
MIVAA_GATEWAY_URL = os.getenv("MIVAA_GATEWAY_URL", "http://localhost:3000")


class RealEmbeddingsService:
    """
    Generates embedding types using AI models.

    This service provides:
    - Text embeddings via Voyage AI (1024D primary) with OpenAI fallback (1024D)
    - Visual embeddings (768D) - SLIG (SigLIP2) via HuggingFace Cloud Endpoint
    - Understanding embeddings (1024D) - Qwen vision_analysis → Voyage AI text embedding
    - Multimodal fusion (1792D) - combined text+visual (1024D + 768D = 1792D)

    Concurrency:
    - All outbound embedding calls go through process-wide asyncio semaphores
      (`_slig_semaphore`, `_voyage_semaphore`). 1000-image catalogs no longer
      open 1000 simultaneous HTTP connections to HF / Voyage. Caps come from
      settings.slig_concurrency / settings.voyage_concurrency.
    """

    # Process-wide semaphores; created lazily on first use so the event loop
    # is bound to the running loop, not module import time.
    _slig_semaphore: Optional[asyncio.Semaphore] = None
    _voyage_semaphore: Optional[asyncio.Semaphore] = None

    @classmethod
    def _get_slig_semaphore(cls) -> asyncio.Semaphore:
        if cls._slig_semaphore is None:
            from app.config import get_settings as _gs
            cls._slig_semaphore = asyncio.Semaphore(_gs().slig_concurrency)
        return cls._slig_semaphore

    @classmethod
    def _get_voyage_semaphore(cls) -> asyncio.Semaphore:
        if cls._voyage_semaphore is None:
            from app.config import get_settings as _gs
            cls._voyage_semaphore = asyncio.Semaphore(_gs().voyage_concurrency)
        return cls._voyage_semaphore

    def __init__(self, supabase_client=None, config=None):
        """Initialize embeddings service."""
        self.supabase = supabase_client
        self.logger = logger
        self.openai_api_key = OPENAI_API_KEY
        self.voyage_api_key = VOYAGE_API_KEY
        self.huggingface_api_key = HUGGINGFACE_API_KEY
        self.mivaa_gateway_url = MIVAA_GATEWAY_URL
        self.config = config

        # Initialize AI logger
        self.ai_logger = AICallLogger()

        # Model loading state (for local mode only)
        self._models_loaded = False
        self._model_load_failed = False
        self._model_load_attempts = 0
        self._max_load_attempts = 3
        self._siglip_model = None
        self._siglip_processor = None
        self._device = None

        # ============================================================================
        # SLIG (SigLIP2) Cloud Endpoint Configuration
        # ============================================================================
        # All visual embeddings are generated via SLIG cloud endpoint (no local models)
        from app.config import settings
        self.slig_endpoint_url = settings.slig_endpoint_url
        self.slig_endpoint_token = settings.slig_endpoint_token
        self.slig_model_name = settings.slig_model_name
        self.slig_embedding_dimension = settings.slig_embedding_dimension
        self.slig_enabled = settings.slig_enabled
        self.slig_timeout = settings.slig_timeout
        self.slig_max_retries = settings.slig_max_retries
        self._slig_client = None  # Lazy-initialized SLIG client
        self._voyage_client = None  # Lazy-initialized Voyage AI httpx client

        # Log SLIG configuration
        self.logger.info(f"☁️ Visual Embeddings: SLIG Cloud Endpoint (basiliskan/siglip2, 768D)")

        # Voyage AI configuration
        self.voyage_api_key = settings.voyage_api_key
        self.voyage_model = settings.voyage_model
        self.voyage_enabled = settings.voyage_enabled
        self.voyage_fallback_to_openai = settings.voyage_fallback_to_openai

        # Debug logging for Voyage AI configuration
        self.logger.info(f"🔧 Voyage AI Config: enabled={self.voyage_enabled}, api_key={'SET' if self.voyage_api_key else 'NOT SET'}, model={self.voyage_model}")

    
    async def generate_all_embeddings(
        self,
        entity_id: str,
        entity_type: str,  # 'product', 'chunk', 'image'
        text_content: str,
        image_url: Optional[str] = None,
        material_properties: Optional[Dict[str, Any]] = None,
        image_data: Optional[str] = None,  # base64 encoded
        vision_analysis: Optional[Dict[str, Any]] = None  # Qwen3-VL analysis JSON
    ) -> Dict[str, Any]:
        """
        Generate all embedding types for an entity.

        Args:
            entity_id: ID of entity to embed
            entity_type: Type of entity (product, chunk, image)
            text_content: Text to embed
            image_url: URL of image (optional)
            material_properties: Material properties dict (optional)
            image_data: Base64 encoded image data (optional)
            vision_analysis: Qwen3-VL vision_analysis JSON for understanding embedding (optional)

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
                embeddings["embeddings"]["text_1024"] = text_embedding
                embeddings["metadata"]["model_versions"]["text"] = "voyage-4" if self.voyage_enabled else "text-embedding-3-small"
                embeddings["metadata"]["confidence_scores"]["text"] = 0.95
                self.logger.info(f"✅ Text embedding generated (1024D, input_type={input_type})")
            
            # 2. Visual Embedding (768D) - REAL (SLIG cloud endpoint)
            pil_image_for_reuse = None  # Track PIL image for reuse
            visual_embedding = None
            if image_url or image_data:
                visual_embedding, model_used, pil_image_for_reuse = await self._generate_visual_embedding(
                    image_url, image_data
                )
                if visual_embedding:
                    embeddings["embeddings"]["visual_768"] = visual_embedding  # SLIG 768D
                    embeddings["metadata"]["model_versions"]["visual"] = model_used
                    embeddings["metadata"]["confidence_scores"]["visual"] = 0.95
                    self.logger.info(f"✅ Visual embedding generated (768D) using {model_used}")

                # 2a. Generate text-guided specialized visual embeddings using SLIG similarity mode.
                # Pass the visual_embedding we just computed so the specialized
                # path doesn't re-fetch the same image embedding from SLIG.
                # Saves 1 SLIG call per image (was 1 base + 1 re-base + 4 sim + 4 text = 10/img;
                # now 1 base + 4 sim + 0 text-cached = 5/img).
                specialized_embeddings, pil_image_for_reuse = await self._generate_specialized_siglip_embeddings(
                    image_url,
                    image_data,
                    pil_image=pil_image_for_reuse,
                    base_image_embedding=visual_embedding,
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
                    except Exception:
                        pass

            # 3. Understanding Embedding (1024D) - Qwen vision_analysis → Voyage AI
            if vision_analysis:
                understanding_embedding = await self.generate_understanding_embedding(
                    vision_analysis=vision_analysis,
                    material_properties=material_properties
                )
                if understanding_embedding:
                    embeddings["embeddings"]["understanding_1024"] = understanding_embedding
                    embeddings["metadata"]["model_versions"]["understanding"] = "voyage-4"
                    embeddings["metadata"]["confidence_scores"]["understanding"] = 0.93
                    self.logger.info("✅ Understanding embedding generated (1024D)")

            self.logger.info(f"✅ All embeddings generated: {len(embeddings['embeddings'])} types")

            # Add success flag
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

    async def generate_text_embedding(
        self,
        query: str,
        dimensions: int = 1024
    ) -> Dict[str, Any]:
        """
        Public method to generate a text embedding for search queries.

        Returns dict with {"success": bool, "embedding": list} format
        expected by RAG service.

        Args:
            query: Text query to embed
            dimensions: Embedding dimensions (default 1024)

        Returns:
            Dict with success status and embedding vector
        """
        try:
            embedding = await self._generate_text_embedding(text=query, dimensions=dimensions)
            if embedding:
                return {"success": True, "embedding": embedding}
            return {"success": False, "error": "Text embedding generation returned None"}
        except Exception as e:
            self.logger.error(f"❌ generate_text_embedding failed: {e}")
            return {"success": False, "error": str(e)}

    async def generate_visual_embedding(
        self,
        query: str
    ) -> Dict[str, Any]:
        """
        Public method to generate a visual-space embedding from text query.

        Uses SLIG (SigLIP2) cloud endpoint to convert text into the visual
        embedding space (768D), enabling text-to-image search across visual,
        color, texture, style, and material embeddings.

        Args:
            query: Text query to convert to visual embedding space

        Returns:
            Dict with {"success": bool, "embedding": list} format
        """
        try:
            # Initialize SLIG client if needed
            if self._slig_client is None:
                if not self.slig_enabled or not self.slig_endpoint_url or not self.slig_endpoint_token:
                    self.logger.warning("⚠️ SLIG not configured, cannot generate visual embedding from text")
                    return {"success": False, "error": "SLIG visual embedding service not configured"}

                from app.services.embeddings.slig_client import SLIGClient
                self._slig_client = SLIGClient(
                    endpoint_url=self.slig_endpoint_url,
                    token=self.slig_endpoint_token,
                    timeout=self.slig_timeout,
                )

            # Use SLIG's text embedding to map text into visual space (768D)
            embedding = await self._slig_client.get_text_embedding(query)
            if embedding:
                self.logger.info(f"✅ Visual embedding from text: {len(embedding)}D via SLIG")
                return {"success": True, "embedding": embedding}

            return {"success": False, "error": "SLIG text-to-visual embedding returned None"}
        except Exception as e:
            self.logger.error(f"❌ generate_visual_embedding failed: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _validate_vision_analysis_schema_static(vision_analysis: Any) -> bool:
        return _is_valid_vision_analysis_schema(vision_analysis)

    async def generate_understanding_embedding(
        self,
        vision_analysis: Dict[str, Any],
        material_properties: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None
    ) -> Optional[List[float]]:
        """
        Generate understanding embedding from Qwen3-VL vision_analysis JSON.

        Converts structured vision analysis into descriptive text, then embeds
        via Voyage AI (1024D) to enable spec-based search queries like
        "porcelain tile 60x120cm" or "R10 slip rating".

        Args:
            vision_analysis: Qwen3-VL analysis JSON with material_type, colors, textures, etc.
            material_properties: Optional additional material properties
            job_id: Optional job ID for logging

        Returns:
            1024D embedding vector or None if failed
        """
        try:
            # Audit fix #33: validate Qwen vision_analysis schema before serializing.
            # Without this check a malformed Qwen response (e.g. {"error": "OOM"})
            # would yield near-empty text, which Voyage would happily embed as a
            # degenerate "no text" vector that matches every search query — silent
            # search corruption.
            if not _is_valid_vision_analysis_schema(vision_analysis):
                self.logger.warning(
                    f"⚠️ Malformed vision_analysis schema (keys={list((vision_analysis or {}).keys())[:6]}); "
                    f"refusing to embed degenerate text. Skipping understanding embedding."
                )
                return None

            # Convert vision_analysis JSON to descriptive text
            text = self._vision_analysis_to_text(vision_analysis, material_properties)

            if not text or not text.strip():
                self.logger.warning("⚠️ Empty text from vision_analysis conversion, skipping understanding embedding")
                return None

            self.logger.debug(f"📝 Understanding embedding text ({len(text)} chars): {text[:200]}...")

            # Embed via Voyage AI with input_type="document" for optimal retrieval
            embedding = await self._generate_text_embedding(
                text=text,
                input_type="document",
                job_id=job_id
            )

            if embedding:
                self.logger.info(f"✅ Understanding embedding generated ({len(embedding)}D)")
            return embedding

        except Exception as e:
            self.logger.error(f"❌ Understanding embedding generation failed: {e}")
            return None

    def _vision_analysis_to_text(
        self,
        vision_analysis: Dict[str, Any],
        material_properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Convert Qwen3-VL vision_analysis JSON into descriptive text for embedding.

        Extracts all structured fields and combines into a searchable text representation.

        Args:
            vision_analysis: Qwen3-VL analysis JSON
            material_properties: Optional additional material properties

        Returns:
            Descriptive text string for embedding
        """
        parts = []

        # Material type
        material_type = vision_analysis.get('material_type') or vision_analysis.get('type')
        if material_type:
            parts.append(f"Material: {material_type}.")

        # Category / subcategory
        category = vision_analysis.get('category')
        subcategory = vision_analysis.get('subcategory')
        if category:
            cat_str = f"Category: {category}"
            if subcategory:
                cat_str += f", {subcategory}"
            parts.append(cat_str + ".")

        # Colors
        colors = vision_analysis.get('colors') or vision_analysis.get('color_palette') or vision_analysis.get('dominant_colors')
        if colors:
            if isinstance(colors, list):
                parts.append(f"Colors: {', '.join(str(c) for c in colors)}.")
            elif isinstance(colors, dict):
                color_items = [f"{k}: {v}" for k, v in colors.items()]
                parts.append(f"Colors: {', '.join(color_items)}.")
            else:
                parts.append(f"Colors: {colors}.")

        # Textures
        textures = vision_analysis.get('textures') or vision_analysis.get('texture') or vision_analysis.get('surface_texture')
        if textures:
            if isinstance(textures, list):
                parts.append(f"Textures: {', '.join(str(t) for t in textures)}.")
            else:
                parts.append(f"Texture: {textures}.")

        # Finish
        finish = vision_analysis.get('finish') or vision_analysis.get('surface_finish')
        if finish:
            parts.append(f"Finish: {finish}.")

        # Pattern
        pattern = vision_analysis.get('pattern') or vision_analysis.get('pattern_type')
        if pattern:
            parts.append(f"Pattern: {pattern}.")

        # Description
        description = vision_analysis.get('description') or vision_analysis.get('visual_description')
        if description:
            parts.append(f"Description: {description}.")

        # Applications / usage
        applications = vision_analysis.get('applications') or vision_analysis.get('suitable_for') or vision_analysis.get('usage')
        if applications:
            if isinstance(applications, list):
                parts.append(f"Applications: {', '.join(str(a) for a in applications)}.")
            else:
                parts.append(f"Applications: {applications}.")

        # Dimensions / size
        dimensions = vision_analysis.get('dimensions') or vision_analysis.get('size')
        if dimensions:
            parts.append(f"Dimensions: {dimensions}.")

        # Properties (from vision_analysis)
        properties = vision_analysis.get('properties') or vision_analysis.get('characteristics')
        if properties:
            if isinstance(properties, dict):
                prop_items = [f"{k}: {v}" for k, v in properties.items() if v]
                if prop_items:
                    parts.append(f"Properties: {', '.join(prop_items)}.")
            elif isinstance(properties, list):
                parts.append(f"Properties: {', '.join(str(p) for p in properties)}.")

        # OCR text detected in image
        ocr_text = vision_analysis.get('ocr_text') or vision_analysis.get('detected_text') or vision_analysis.get('text_content')
        if ocr_text:
            if isinstance(ocr_text, list):
                parts.append(f"Text detected: {' '.join(ocr_text)}.")
            else:
                parts.append(f"Text detected: {ocr_text}.")

        # Additional material_properties if provided
        if material_properties:
            mp_parts = []
            for key, value in material_properties.items():
                if value and key not in ('id', 'created_at', 'updated_at', 'document_id', 'image_id'):
                    mp_parts.append(f"{key}: {value}")
            if mp_parts:
                parts.append(f"Material properties: {', '.join(mp_parts)}.")

        return " ".join(parts)

    async def generate_understanding_query_embedding(
        self,
        query: str
    ) -> Dict[str, Any]:
        """
        Generate understanding-space embedding from a text query for search.

        Uses Voyage AI with input_type="query" to create an embedding in the
        same space as understanding embeddings for similarity search.

        Args:
            query: Text query to embed for understanding search

        Returns:
            Dict with {"success": bool, "embedding": list} format
        """
        try:
            embedding = await self._generate_text_embedding(
                text=query,
                input_type="query"
            )
            if embedding:
                return {"success": True, "embedding": embedding}
            return {"success": False, "error": "Understanding query embedding returned None"}
        except Exception as e:
            self.logger.error(f"❌ generate_understanding_query_embedding failed: {e}")
            return {"success": False, "error": str(e)}

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
            dimensions: Embedding dimensions (default 1024 for Voyage AI; 256, 512, 1024, 2048 supported)
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

        # Replace empty/whitespace strings with a placeholder; Voyage rejects empty inputs.
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                processed_texts.append("no text content")
            else:
                processed_texts.append(text)

        # Default voyage_enabled to True when config is missing — Voyage is the primary provider.
        voyage_enabled = getattr(self.config, 'voyage_enabled', True) if self.config else True
        if self.voyage_api_key and voyage_enabled:
            try:
                # Voyage supports {256, 512, 1024, 2048}; map the OpenAI-style 1536 down to 1024.
                voyage_dimensions = 1024 if dimensions == 1536 else dimensions

                # Reuse httpx client for connection keepalive across batches
                if self._voyage_client is None:
                    self._voyage_client = httpx.AsyncClient(
                        timeout=60.0,
                        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                    )

                request_data = {
                    "model": "voyage-4",
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

                response = await self._voyage_client.post(
                    "https://api.voyageai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.voyage_api_key}",
                        "Content-Type": "application/json"
                    },
                    json=request_data,
                )

                if response.status_code == 200:
                    data = response.json()
                    embeddings = [item["embedding"] for item in data["data"]]

                    # Log AI call with proper cost calculation
                    latency_ms = int((time.time() - start_time) * 1000)
                    usage = data.get("usage", {})
                    input_tokens = usage.get("total_tokens", 0)

                    # Voyage AI Pricing (as of Dec 2024)
                    cost_per_million = 0.06  # voyage-4
                    cost = (input_tokens / 1_000_000) * cost_per_million

                    await self.ai_logger.log_ai_call(
                        task="batch_text_embedding_generation",
                        model=f"voyage-4-{voyage_dimensions}d",
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
                            "batch_size": len(texts)
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
                    model=f"voyage-4-{voyage_dimensions}d",
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
                        "batch_size": len(texts)
                    },
                    action="fallback_to_rules",
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
                        "dimensions": 1024  # Pinned to 1024D so the OpenAI fallback matches Voyage + DB schema.
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
                            "batch_size": len(texts)
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
                    "batch_size": len(texts)
                },
                action="fallback_to_rules",
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
            dimensions: Embedding dimensions (default 1024 for Voyage AI; 256, 512, 1024, 2048 supported)
            input_type: None (default), "document" for indexing, "query" for search (Voyage AI only)
            truncation: Whether to truncate text to fit context length (Voyage AI only, default: True)
            output_dtype: Output data type - 'float', 'int8', 'uint8', 'binary', 'ubinary' (Voyage AI only)

        Returns:
            List of floats representing the embedding, or None if failed
        """
        start_time = time.time()

        # Placeholder for empty/whitespace text; Voyage rejects empty inputs.
        if not text or not text.strip():
            self.logger.debug("⏭️  Using placeholder text for empty/whitespace input")
            text = "no text content"

        # Default voyage_enabled to True when config is missing — Voyage is the primary provider.
        voyage_enabled = getattr(self.config, 'voyage_enabled', True) if self.config else True
        self.logger.info(f"🔍 Voyage AI check: api_key={'SET' if self.voyage_api_key else 'NOT SET'}, config={'SET' if self.config else 'NOT SET'}, voyage_enabled={voyage_enabled}")
        if self.voyage_api_key and voyage_enabled:
            try:
                # Voyage caps at 2048D; map OpenAI-style 1536 down to 1024.
                voyage_dimensions = 1024 if dimensions == 1536 else dimensions

                # Throttle outbound Voyage calls — without the semaphore, gather()
                # over 1000 chunks fires 1000 simultaneous HTTPS requests and the
                # API rate-limits us. settings.voyage_concurrency caps it.
                async with self._get_voyage_semaphore(), httpx.AsyncClient() as client:
                    request_data = {
                        "model": self.voyage_model,
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

                    # Audit fix #34: handle 429 rate-limit explicitly. Without this,
                    # 429s fall through to the generic except → silent OpenAI fallback
                    # → bursts of OpenAI cost during a Voyage rate-limit window.
                    # Respect Retry-After header (Voyage sets it). Up to 3 retries
                    # with capped backoff before giving up to fallback.
                    rate_limit_attempt = 0
                    while response.status_code == 429 and rate_limit_attempt < 3:
                        retry_after_raw = response.headers.get("Retry-After", "5")
                        try:
                            retry_after = min(60.0, float(retry_after_raw))
                        except ValueError:
                            retry_after = 5.0
                        self.logger.warning(
                            f"⚠️ Voyage 429 rate-limit (attempt {rate_limit_attempt+1}/3); "
                            f"sleeping {retry_after}s before retry"
                        )
                        await asyncio.sleep(retry_after)
                        response = await client.post(
                            "https://api.voyageai.com/v1/embeddings",
                            headers={
                                "Authorization": f"Bearer {self.voyage_api_key}",
                                "Content-Type": "application/json"
                            },
                            json=request_data,
                            timeout=30.0
                        )
                        rate_limit_attempt += 1

                    if response.status_code == 200:
                        data = response.json()
                        embedding = data["data"][0]["embedding"]

                        # Log AI call with proper cost calculation
                        latency_ms = int((time.time() - start_time) * 1000)
                        usage = data.get("usage", {})
                        input_tokens = usage.get("total_tokens", 0)

                        # Voyage AI Pricing (as of Dec 2024)
                        # voyage-4: $0.06 per 1M tokens
                        # voyage-3-large: $0.18 per 1M tokens
                        cost_per_million = 0.06  # voyage-4
                        cost = (input_tokens / 1_000_000) * cost_per_million

                        await self.ai_logger.log_ai_call(
                            task="text_embedding_generation",
                            model=f"{self.voyage_model}-{voyage_dimensions}d",
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
                    model=f"voyage-4-{voyage_dimensions}d",
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

            # Audit fix #16: pin OpenAI fallback to 1024D regardless of caller's
            # `dimensions` arg. The DB schema is halfvec(1024) and there's no
            # legitimate code path that should request a different dim. Previously
            # legacy callers passing dimensions=1536 caused silent dim-mismatch
            # storage (truncation or type error at write time).
            openai_dimensions = 1024
            if dimensions != 1024:
                self.logger.warning(
                    f"OpenAI fallback received dimensions={dimensions} but pinning to 1024 "
                    f"(DB schema constraint). Caller should be updated."
                )

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
            confidence_threshold: Unused parameter
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
            self.logger.info(f"✅ Using visual embedding from {self.slig_model_name}")
            return visual_embedding, self.slig_model_name, pil_image_out

        self.logger.error(f"❌ Visual embedding generation failed for {self.slig_model_name}")
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

        Uses cloud SLIG endpoint (HuggingFace) — no local model loading.

        NOTE: Requires SLIG_ENDPOINT_URL and SLIG_ENDPOINT_TOKEN to be configured.

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
            import base64
            from PIL import Image
            import io
            import numpy as np
            import asyncio

            # SLIG Cloud Endpoint — visual embeddings are always remote.
            if self.slig_enabled:
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

                try:
                    # Audit fix #14: SLIG dim-mismatch retry. Previously a single
                    # wrong-dim response (transient model swap, partial deploy)
                    # silently aborted with no retry → mass data loss. Now we
                    # retry up to 3x, each with a fresh request — the endpoint
                    # may have just swapped models mid-batch.
                    embedding = None
                    latency_ms = 0
                    for slig_attempt in range(3):
                        async with self._get_slig_semaphore():
                            embedding = await self._slig_client.get_image_embedding(pil_image)
                        latency_ms = int((time.time() - start_time) * 1000)
                        if len(embedding) == self.slig_embedding_dimension:
                            break
                        self.logger.warning(
                            f"⚠️ SLIG dim-mismatch attempt {slig_attempt+1}/3: "
                            f"got {len(embedding)}D, expected {self.slig_embedding_dimension}D"
                        )
                        await asyncio.sleep(0.5 * (2 ** slig_attempt))
                        embedding = None

                    if embedding is None or len(embedding) != self.slig_embedding_dimension:
                        self.logger.error(
                            f"❌ SLIG endpoint returned wrong-dim embedding 3x in a row — "
                            f"refusing to store. The endpoint is likely misconfigured "
                            f"(serving wrong model). Operator action required."
                        )
                        return None, None

                    self.logger.info(f"✅ SLIG embedding generated: {len(embedding)}D (latency={latency_ms}ms)")

                    await self.ai_logger.log_ai_call(
                        task="visual_embedding_generation",
                        model=self.slig_model_name,
                        input_tokens=0,
                        output_tokens=0,
                        cost=0.0,
                        latency_ms=latency_ms,
                        confidence_score=0.95,
                        confidence_breakdown={
                            "model_confidence": 0.98,
                            "completeness": 1.0,
                            "consistency": 0.95,
                            "validation": 0.90,
                            # Each visual call yields ONE 768D vector. The
                            # specialized (color/texture/style/material) calls
                            # log themselves separately — see
                            # _generate_specialized_siglip_embeddings — so the
                            # `vectors_generated` field tells analytics how
                            # many vectors a single document run produced.
                            "vectors_generated": 1,
                            "vector_dimension": self.slig_embedding_dimension,
                            "vector_kind": "visual",
                        },
                        action="use_ai_result",
                        job_id=job_id,
                    )

                    return embedding, pil_image

                except Exception as e:
                    self.logger.error(f"❌ SLIG cloud endpoint failed: {e}")
                    return None, None
            else:
                self.logger.error("❌ SLIG visual embeddings are disabled")
                return None, None



        except Exception as e:
            self.logger.error(f"SLIG embedding generation failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Close image on error if we created it
            if not image_was_provided and pil_image and hasattr(pil_image, 'close'):
                try:
                    pil_image.close()
                except Exception:
                    pass

        return None, None

    # Class-level cache for the 4 fixed text-prompt SLIG embeddings.
    # These prompts are CONSTANTS — embedding them once per pipeline run
    # instead of per-image saves 4 SLIG calls × N images per product.
    # On a 30-image product = 120 fewer SLIG calls per product.
    _SPECIALIZED_TEXT_PROMPTS = {
        "color": "focus on color palette and color relationships",
        "texture": "focus on surface patterns and texture details",
        "material": "focus on material type and physical properties",
        "style": "focus on design style and aesthetic elements",
    }
    _text_prompt_embedding_cache: Dict[str, List[float]] = {}
    _text_prompt_cache_lock: Optional[asyncio.Lock] = None

    @classmethod
    def _get_text_cache_lock(cls) -> asyncio.Lock:
        if cls._text_prompt_cache_lock is None:
            cls._text_prompt_cache_lock = asyncio.Lock()
        return cls._text_prompt_cache_lock

    async def _get_cached_specialized_text_embedding(self, prompt_key: str) -> Optional[List[float]]:
        """Return SLIG text embedding for a fixed specialized prompt, computing
        once per process and caching forever. The prompts are constants so
        re-embedding them per image is pure waste."""
        prompt = self._SPECIALIZED_TEXT_PROMPTS.get(prompt_key)
        if not prompt:
            return None
        cached = RealEmbeddingsService._text_prompt_embedding_cache.get(prompt_key)
        if cached is not None:
            return cached
        async with self._get_text_cache_lock():
            cached = RealEmbeddingsService._text_prompt_embedding_cache.get(prompt_key)
            if cached is not None:
                return cached
            async with self._get_slig_semaphore():
                emb = await self._slig_client.get_text_embedding(prompt)
            if emb and len(emb) == self.slig_embedding_dimension:
                RealEmbeddingsService._text_prompt_embedding_cache[prompt_key] = emb
                return emb
            return None

    async def _generate_specialized_siglip_embeddings(
        self,
        image_url: Optional[str],
        image_data: Optional[str],
        pil_image=None,
        base_image_embedding: Optional[List[float]] = None,
    ) -> tuple[Optional[Dict[str, List[float]]], Optional[any]]:
        """
        Generate 4 text-guided SLIG embeddings (color / texture / style / material).

        All-or-nothing: if any of the 4 fail, this returns None for the whole set
        so the image doesn't end up with partial specialized coverage.

        `base_image_embedding`: pass the already-computed visual_768 embedding
        from the caller so we don't re-compute it. Saves 1 SLIG call per image.
        """
        try:
            import numpy as np

            text_prompts = self._SPECIALIZED_TEXT_PROMPTS

            if base_image_embedding is None:
                async with self._get_slig_semaphore():
                    base_image_embedding = await self._slig_client.get_image_embedding(
                        image=image_data if image_data else image_url
                    )
            if not base_image_embedding or len(base_image_embedding) != self.slig_embedding_dimension:
                self.logger.warning(
                    f"⚠️ Specialized embeddings: base image embedding invalid "
                    f"(len={len(base_image_embedding) if base_image_embedding else 0}, "
                    f"expected={self.slig_embedding_dimension}) — skipping all 4"
                )
                return None, pil_image

            specialized: Dict[str, List[float]] = {}
            image_input = image_data if image_data else image_url

            for embedding_type, text_prompt in text_prompts.items():
                try:
                    async with self._get_slig_semaphore():
                        similarity_result = await self._slig_client.calculate_similarity(
                            images=image_input,
                            texts=text_prompt,
                        )
                    sim_matrix = similarity_result.get("similarity_scores") if similarity_result else None
                    if not sim_matrix or not sim_matrix[0]:
                        raise ValueError(f"empty similarity_scores for {embedding_type}")
                    similarity_score = float(sim_matrix[0][0])

                    # Cached: only computed once per process for the 4 fixed prompts.
                    text_embedding = await self._get_cached_specialized_text_embedding(embedding_type)
                    if not text_embedding or len(text_embedding) != self.slig_embedding_dimension:
                        raise ValueError(
                            f"text embedding wrong size for {embedding_type}: "
                            f"len={len(text_embedding) if text_embedding else 0}"
                        )

                    img_emb = np.array(base_image_embedding, dtype=np.float32)
                    txt_emb = np.array(text_embedding, dtype=np.float32)
                    blend_weight = 0.7 + (0.2 * similarity_score)
                    guided = blend_weight * img_emb + (1 - blend_weight) * txt_emb
                    norm = np.linalg.norm(guided)
                    if norm > 0:
                        guided = guided / norm

                    specialized[embedding_type] = guided.tolist()

                except Exception as e:
                    self.logger.error(
                        f"❌ Specialized {embedding_type} embedding failed — aborting all 4: {e}",
                        exc_info=True,
                    )
                    return None, pil_image

            self.logger.info("✅ Generated 4 text-guided specialized SLIG embeddings (768D each)")
            try:
                await self.ai_logger.log_ai_call(
                    task="visual_specialized_embeddings_batch",
                    model=self.slig_model_name,
                    input_tokens=0,
                    output_tokens=0,
                    cost=0.0,
                    latency_ms=0,
                    confidence_score=0.95,
                    confidence_breakdown={
                        "model_confidence": 0.98,
                        "completeness": 1.0,
                        "consistency": 0.95,
                        "validation": 0.90,
                        # 4 vectors × (1 image_embedding + 1 text_embedding +
                        # 1 calculate_similarity) = up to 12 SLIG calls per
                        # image; we log the count so analytics knows the
                        # actual fan-out behind the single "specialized" log.
                        "vectors_generated": len(specialized),
                        "vector_dimension": self.slig_embedding_dimension,
                        "vector_kinds": list(specialized.keys()),
                    },
                    action="use_ai_result",
                )
            except Exception as log_err:
                self.logger.debug(f"Specialized SLIG aggregate log skipped: {log_err}")
            return specialized, pil_image

        except Exception as e:
            self.logger.error(f"❌ Specialized SLIG embedding generation failed: {e}", exc_info=True)
            return None, pil_image


    # Replaced by SLIG text-guided specialized embeddings (color, texture, style, material)
    # stored in VECS collections at 768D each.




