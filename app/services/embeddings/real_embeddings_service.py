"""
Real Embeddings Service - Step 4 Implementation (Updated for Voyage AI)

Generates embedding types using AI models:
1. Text (1024D) - Voyage AI voyage-4
2. Visual Embeddings (768D) - SLIG (SigLIP2) via HuggingFace Cloud Endpoint
3. Understanding (1024D) - Claude Opus vision_analysis JSON -> Voyage AI text embedding
4. Page (1024D) - voyage-multimodal, whole rendered catalog page (#239)

Text Embedding Strategy:
- Voyage AI voyage-4 (1024D default, supports 256/512/1024/2048)
- Supports input_type parameter: "document" for indexing, "query" for search

THERE IS NO SECOND EMBEDDING PROVIDER, AND ADDING ONE IS A BUG
--------------------------------------------------------------
This service used to fall back to OpenAI `text-embedding-3-small` when Voyage failed.
It was removed (2026-08-08) because a fallback embedder is not a resilience feature -
it is a correctness hazard wearing one.

`text-embedding-3-small` at 1024D and `voyage-4` at 1024D are the same SHAPE and a
different SPACE. A fallback vector is therefore accepted everywhere a real one is:
Postgres stores it, the HNSW index ranks it, cosine similarity returns a confident
number. Nothing raises, nothing logs, and no probe can see it - every artifact
involved is individually well-formed. On the write paths the damage is durable:
mixed-space rows sit in the collection forever, ranking wrongly.

The old design tried to contain this with a per-call `allow_openai_fallback=False`
opt-out, and SEVEN call sites duly passed it. `generate_understanding_query_embedding`
did not - so the collections most carefully purified on the write side could still be
QUERIED with a wrong-space vector. That is the predictable end state of an opt-out:
it holds until someone adds the eighth call site.

So: Voyage or nothing. `None` means NO VECTOR, and every caller handles that - the
work is retryable and the gap is visible. A vector from another model is neither.

(OpenAI is still used elsewhere in this codebase, legitimately: the multi-provider
LLM mention probe compares gpt-4o-mini against haiku/gemini/sonar on purpose. That
is a comparison of models, not a substitution of one for another.)

Visual Embedding Strategy:
- Uses SLIG cloud endpoint exclusively (768D embeddings)
- Cloud-only architecture - no local model loading

Understanding Embedding Strategy:
- Embeds Claude Opus's structured vision_analysis JSON (via Anthropic tool
  use, schema-locked) as descriptive text → Voyage AI (1024D).
- Enables spec-based search (e.g., "porcelain tile 60x120cm", "R10 slip rating").
- Claude is the sole vision producer.
"""

import logging
import asyncio
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import httpx


def _is_valid_vision_analysis_schema(vision_analysis: Any) -> bool:
    """Reject malformed vision_analysis payloads before embedding.

    Without this, error envelopes like {"error": "OOM", "message": "..."} would
    feed into the embedding pipeline, produce an almost-empty text payload,
    and Voyage would return a degenerate embedding that matches every query.

    Quick structural check only — the strict Pydantic schema enforcement
    happens later via `vision_analysis_from_legacy_dict` + `VisionAnalysis`.
    """
    if not isinstance(vision_analysis, dict):
        return False
    # An error envelope is the most common failure shape.
    if 'error' in vision_analysis and 'material_type' not in vision_analysis:
        return False
    # Require at least one of the canonical describing fields. The legacy
    # rows pre-2026-05-01 sometimes dropped keys — we accept those but reject
    # payloads with NONE of these present.
    expected_any = ('material_type', 'category', 'colors', 'textures',
                    'finish', 'surface_pattern', 'description')
    return any(k in vision_analysis for k in expected_any)

from app.services.core.ai_call_logger import AICallLogger
from app.models.vision_analysis import (
    VisionAnalysis,
    SCHEMA_VERSION,
    ASPECT_SERIALIZERS,
    vision_analysis_from_legacy_dict,
)

logger = logging.getLogger(__name__)

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")  # From GitHub Secrets / Deno.env
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
MIVAA_GATEWAY_URL = os.getenv("MIVAA_GATEWAY_URL", "http://localhost:3000")


class RealEmbeddingsService:
    """
    Generates embedding types using AI models.

    This service provides:
    - Text embeddings via Voyage AI (1024D) - no fallback provider, by design
    - Visual embeddings (768D) - SLIG (SigLIP2) via HuggingFace Cloud Endpoint
    - Understanding embeddings (1024D) - Claude vision_analysis → Voyage AI text embedding
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
        # SLIG (SigLIP2) Modal Endpoint Configuration
        # ============================================================================
        # Visual embeddings are served by the `slig` Modal app (modal_app/slig.py).
        # These attrs hold the Modal base URL + bearer; SLIGClient appends /infer.
        from app.config import settings
        self.slig_endpoint_url = settings.slig_modal_url
        self.slig_endpoint_token = settings.slig_modal_api_key
        self.slig_model_name = settings.slig_model_name
        self.slig_embedding_dimension = settings.slig_embedding_dimension
        self.slig_enabled = settings.slig_enabled
        self.slig_timeout = settings.slig_timeout
        self.slig_max_retries = settings.slig_max_retries
        self._slig_client = None  # Lazy-initialized SLIG client
        self._voyage_client = None  # Lazy-initialized Voyage AI httpx client

        # Records which model produced the most recent text embedding. With one
        # provider this is always the configured Voyage model - kept because callers
        # persist it as row provenance, which is how a MODEL change (voyage-4 ->
        # voyage-5) stays detectable even though a PROVIDER change cannot happen.
        self._last_provider: Optional[str] = None

        # Pull voyage_model from config once so call sites don't have to
        # re-read settings on every embedding.
        self.voyage_model = getattr(config, "voyage_model", None) or "voyage-4"

        # Log SLIG configuration
        self.logger.info("☁️ Visual Embeddings: SLIG on Modal (basiliskan/slig — siglip2-base-patch16-512, native 768D)")

        # Voyage AI configuration
        self.voyage_api_key = settings.voyage_api_key
        self.voyage_model = settings.voyage_model
        self.voyage_enabled = settings.voyage_enabled

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
        vision_analysis: Optional[Dict[str, Any]] = None,  # Claude Opus vision_analysis JSON (schema-locked via Anthropic tool use)
        job_id: Optional[str] = None,
        # Mirrors job_id exactly. The workspace was never threaded, so 2134 ai_usage_logs rows
        # landed with no tenant: invisible to per-workspace cost views AND to that table's own
        # `is_workspace_admin(workspace_id)` policy, which cannot match on NULL. Per CALL, never
        # on the instance — this service is reused across tenants (rag_service holds one on
        # self), so instance state would misattribute confidently instead of leaving a blank.
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        product_id: Optional[str] = None,
        image_id: Optional[str] = None,
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
            vision_analysis: Claude Opus vision_analysis JSON for understanding embedding (optional)
            job_id: Optional job ID for cost attribution (rolls up into total_ai_cost_usd)
            product_id: Optional product ID for per-product cost attribution
            image_id: Optional image ID for per-image cost attribution

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
                input_type=input_type,
                job_id=job_id,
                workspace_id=workspace_id,
                user_id=user_id,
                product_id=product_id,
                image_id=image_id,
            )
            if text_embedding:
                embeddings["embeddings"]["text_1024"] = text_embedding
                # Provenance = the model that ACTUALLY produced the vector (S3-3).
                # Was a hardcoded "voyage-4" that lied whenever Settings.voyage_model
                # was set to a different version.
                # The `voyage_enabled is False` arm used to stamp
                # "text-embedding-3-small" here. With the fallback removed that arm is
                # unreachable - Voyage disabled means text_embedding is None and this
                # block never runs - but it was also WRONG on its own terms: it would
                # have labelled a row with a model that did not embed it, which is the
                # one thing provenance exists to prevent.
                embeddings["metadata"]["model_versions"]["text"] = (
                    self._last_provider or self.voyage_model
                )
                embeddings["metadata"]["confidence_scores"]["text"] = 0.95
                self.logger.info(f"✅ Text embedding generated (1024D, input_type={input_type})")
            
            # 2. Visual Embedding (768D) - REAL (SLIG cloud endpoint)
            pil_image_for_reuse = None  # Track PIL image for reuse
            visual_embedding = None
            if image_url or image_data:
                visual_embedding, model_used, pil_image_for_reuse = await self._generate_visual_embedding(
                    image_url, image_data,
                    job_id=job_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    product_id=product_id,
                    image_id=image_id,
                )
                if visual_embedding:
                    embeddings["embeddings"]["visual_768"] = visual_embedding  # SLIG 768D
                    embeddings["metadata"]["model_versions"]["visual"] = model_used
                    embeddings["metadata"]["confidence_scores"]["visual"] = 0.95
                    self.logger.info(f"✅ Visual embedding generated (768D) using {model_used}")

                # Close PIL image after the SLIG visual call. Aspect embeddings
                # post-v2 are derived from VisionAnalysis text, not from the
                # image bytes, so we no longer need to keep the PIL image
                # alive past this point.
                if pil_image_for_reuse and hasattr(pil_image_for_reuse, 'close'):
                    try:
                        pil_image_for_reuse.close()
                        self.logger.debug("✅ Closed PIL image after visual embedding")
                    except Exception:
                        pass

            # 2a. Per-aspect embeddings (color / texture / style / material).
            #
            # Each aspect string is built deterministically from per-image
            # VisionAnalysis fields (colors[], textures[]+finish, style+
            # surface_pattern+applications, material_type+category+subcategory)
            # and Voyage-embedded to 1024D — same model and embedding space
            # as image_understanding_embeddings. The aspect vector encodes
            # what THIS image's color/texture/style/material looks like
            # according to Claude Opus. Skipped when vision_analysis
            # is missing (caller provides it for image entities; not for
            # text-only entities).
            if vision_analysis:
                aspect_embeddings = await self._generate_specialized_aspect_embeddings(
                    vision_analysis=vision_analysis,
                    job_id=job_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    product_id=product_id,
                    image_id=image_id,
                )
                if aspect_embeddings:
                    embeddings["embeddings"]["color_aspect_1024"] = aspect_embeddings.get("color")
                    embeddings["embeddings"]["texture_aspect_1024"] = aspect_embeddings.get("texture")
                    embeddings["embeddings"]["style_aspect_1024"] = aspect_embeddings.get("style")
                    embeddings["embeddings"]["material_aspect_1024"] = aspect_embeddings.get("material")
                    # Aspect vectors are produced by _generate_text_embedding →
                    # self.voyage_model, NOT a fixed "voyage-3" (S3-3 provenance fix).
                    embeddings["metadata"]["model_versions"]["specialized_aspect"] = self.voyage_model
                    embeddings["metadata"]["confidence_scores"]["specialized_aspect"] = 0.95
                    embeddings["metadata"]["schema_versions"] = embeddings["metadata"].get("schema_versions", {})
                    embeddings["metadata"]["schema_versions"]["specialized_aspect"] = SCHEMA_VERSION
                    self.logger.info(
                        f"✅ Per-aspect embeddings generated ({len(aspect_embeddings)} × 1024D Voyage)"
                    )

            # 3. Understanding Embedding (1024D) — vision_analysis JSON → Voyage AI.
            # Source vision_analysis comes from Claude Opus via Anthropic
            # tool use. Returns a dict with embedding +
            # provenance so vecs_service can persist embedding_model and
            # schema_version for fallback-drift detection.
            if vision_analysis:
                # Thread the ids, like every sibling embed call in this method does.
                # This one passed none, so its Voyage cost landed in ai_usage_logs with
                # job_id NULL — and progress_tracker.complete_job sums billed_cost_usd
                # FILTERED BY job_id. One Voyage call per image (the 7th fusion vector,
                # on every image in the catalog) was therefore missing from
                # background_jobs.total_ai_cost_usd and from all per-product attribution.
                ue_result = await self.generate_understanding_embedding(
                    vision_analysis=vision_analysis,
                    material_properties=material_properties,
                    job_id=job_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    product_id=product_id,
                    image_id=image_id,
                )
                if ue_result and ue_result.get("embedding"):
                    embeddings["embeddings"]["understanding_1024"] = ue_result["embedding"]
                    embeddings["metadata"]["model_versions"]["understanding"] = ue_result.get("embedding_model") or self.voyage_model
                    embeddings["metadata"]["schema_versions"] = embeddings["metadata"].get("schema_versions", {})
                    embeddings["metadata"]["schema_versions"]["understanding"] = ue_result.get("schema_version", 1)
                    embeddings["metadata"]["confidence_scores"]["understanding"] = 0.93
                    self.logger.info("✅ Understanding embedding generated (1024D)")

            # `success` must mean "vectors came back", not "no exception was raised".
            # It was set to True unconditionally, so with every provider down this
            # returned success alongside an EMPTY embeddings dict and callers counted
            # the entity as embedded. An empty result is a failure the caller has to be
            # able to see — it is the difference between "retry this" and "done".
            _produced = len(embeddings.get("embeddings") or {})
            embeddings["success"] = _produced > 0
            if not _produced:
                embeddings["error"] = "no_vectors_generated"
                self.logger.error(
                    f"❌ No embeddings generated for {entity_type} {entity_id} — "
                    f"every embedding path returned empty"
                )
            else:
                self.logger.info(f"✅ All embeddings generated: {_produced} types")

            return embeddings

        except Exception as e:
            self.logger.error(f"❌ Embedding generation failed: {e}")
            return {"success": False, "error": str(e)}

    async def generate_embedding(
        self,
        text: str,
        embedding_type: str = "voyage",
        dimensions: int = 1536
    ) -> Optional[List[float]]:
        """
        Public method to generate a single text embedding.

        This is the main entry point for generating query embeddings for search.

        Args:
            text: Text to embed
            embedding_type: Legacy, ignored - Voyage is the only text embedder
            dimensions: Embedding dimensions (default 1536)

        Returns:
            List of floats representing the embedding, or None if failed
        """
        return await self._generate_text_embedding(text=text, dimensions=dimensions)

    async def generate_text_embedding(
        self,
        query: str,
        dimensions: int = 1024,
        job_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        product_id: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Public method to generate a text embedding for search queries.

        Returns dict with {"success", "embedding", "model"} format. The model
        field is the model that actually produced the vector, so callers can
        persist provenance and detect a model change across a collection.

        Args:
            query: Text query to embed
            dimensions: Embedding dimensions (default 1024)
            job_id: Optional job ID for cost attribution
            product_id: Optional product ID for per-product cost attribution
            image_id: Optional image ID for per-image cost attribution
        """
        try:
            # _last_provider is stamped by the embedder at the end of the call.
            self._last_provider = None
            embedding = await self._generate_text_embedding(
                text=query,
                dimensions=dimensions,
                job_id=job_id,
                workspace_id=workspace_id,
                user_id=user_id,
                product_id=product_id,
                image_id=image_id,
            )
            if embedding:
                return {
                    "success": True,
                    "embedding": embedding,
                    # Report the model that ACTUALLY produced the vector, not the
                    # configured default - they differ whenever settings change
                    # between a row being written and being read.
                    "model": self._last_provider or self.voyage_model or "voyage-3.5",
                }
            return {"success": False, "error": "Text embedding generation returned None"}
        except Exception as e:
            self.logger.error(f"❌ generate_text_embedding failed: {e}")
            return {"success": False, "error": str(e)}

    async def generate_visual_embedding(
        self,
        query: str,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Public method to generate a visual-space embedding from text query.

        Uses SLIG (SigLIP2) cloud endpoint to convert text into the visual
        embedding space (768D), enabling text-to-image search across visual,
        color, texture, style, and material embeddings.

        MV2-9: this method hit a billed GPU endpoint and returned WITHOUT calling the
        logger at all — every text-to-visual query in the search path was free as far
        as `ai_usage_logs` was concerned. It is a search entry point, so it is called
        far more often than ingestion; the spend it hid was not a rounding error. The
        attribution args are optional because a couple of callers genuinely have no
        tenant (health probes), and a required arg would have been supplied as `None`
        at those sites anyway — better an honest blank than a confident wrong tenant.

        Args:
            query: Text query to convert to visual embedding space
            workspace_id: Tenant to attribute the GPU spend to
            user_id: Principal to bill
            job_id: Optional job for cost roll-up

        Returns:
            Dict with {"success": bool, "embedding": list} format
        """
        start_time = time.time()

        async def _log(action: str, failure: Optional[str] = None) -> None:
            try:
                await self.ai_logger.log_time_based_call(
                    task="visual_query_embedding_generation",
                    model="slig-768d",
                    latency_ms=int((time.time() - start_time) * 1000),
                    action=action,
                    confidence_score=0.95 if action == "use_ai_result" else 0.0,
                    confidence_breakdown={
                        "model_confidence": 0.98 if action == "use_ai_result" else 0.0,
                        "completeness": 1.0 if action == "use_ai_result" else 0.0,
                        "consistency": 0.95 if action == "use_ai_result" else 0.0,
                        "validation": 0.90 if action == "use_ai_result" else 0.0,
                        "vectors_generated": 1 if action == "use_ai_result" else 0,
                        "vector_kind": "visual_query",
                        **({"failure": failure} if failure else {}),
                    },
                    job_id=job_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                )
            except Exception as log_err:
                self.logger.warning(f"Could not log visual query embedding call: {log_err}")

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
                await _log("use_ai_result")
                return {"success": True, "embedding": embedding}

            await _log("fallback_failed", failure="empty_response")
            return {"success": False, "error": "SLIG text-to-visual embedding returned None"}
        except Exception as e:
            self.logger.error(f"❌ generate_visual_embedding failed: {e}")
            await _log("fallback_failed", failure=type(e).__name__)
            return {"success": False, "error": str(e)}

    @staticmethod
    def _validate_vision_analysis_schema_static(vision_analysis: Any) -> bool:
        return _is_valid_vision_analysis_schema(vision_analysis)

    async def generate_understanding_embedding(
        self,
        vision_analysis: Dict[str, Any],
        material_properties: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        product_id: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate understanding embedding from a structured vision_analysis dict.

        Converts the structured vision analysis (produced by Claude Opus via
        Anthropic tool use) into deterministic descriptive text, then embeds
        via Voyage AI (1024D) to enable spec-based search queries like
        "porcelain tile 60x120cm" or "R10 slip rating".

        Args:
            vision_analysis: vision_analysis JSON (preferred: shape matches
                app.models.vision_analysis.VisionAnalysis; legacy free-form
                dicts also accepted via vision_analysis_from_legacy_dict).
            material_properties: Optional additional material properties.
            job_id: Optional job ID for logging.

        Returns:
            Dict with `embedding` (1024D list) + `embedding_model` + `schema_version`
            on success. Returns None on failure.

            Fails soft: on Voyage failure this returns None and the image gets no
            understanding embedding, leaving the visual + specialized vectors to
            cover the row in fusion search. This used to require an explicit opt-out
            from the OpenAI fallback (audit gap B); that fallback no longer exists,
            so mixing two latent spaces in one collection is now structurally
            impossible rather than per-call discipline.
        """
        from app.models.vision_analysis import (
            VisionAnalysis,
            SCHEMA_VERSION,
            serialize_vision_analysis_to_text,
            vision_analysis_from_legacy_dict,
        )

        try:
            # Coerce to the strict schema. Accepts both new (VisionAnalysis)
            # and legacy free-form dicts; refuses error payloads / missing
            # material_type — same guarantee the legacy
            # _is_valid_vision_analysis_schema gave but with structure.
            if isinstance(vision_analysis, VisionAnalysis):
                va = vision_analysis
            else:
                va = vision_analysis_from_legacy_dict(vision_analysis)
            if va is None:
                self.logger.warning(
                    f"⚠️ Malformed vision_analysis (keys={list((vision_analysis or {}).keys())[:6]}); "
                    f"refusing to embed. Skipping understanding embedding."
                )
                return None

            # Single source of truth for the text serialisation. Same function
            # is used at query time so ingestion and query land in the same
            # Voyage embedding distribution.
            text = serialize_vision_analysis_to_text(va)
            if material_properties:
                # Append additional material properties deterministically.
                mp_parts = sorted(
                    f"{k}: {v}"
                    for k, v in material_properties.items()
                    if v and k not in ("id", "created_at", "updated_at",
                                       "document_id", "image_id")
                )
                if mp_parts:
                    text = f"{text} Material properties: {', '.join(mp_parts)}."

            if not text.strip():
                self.logger.warning(
                    "⚠️ Empty serialised text from VisionAnalysis, "
                    "skipping understanding embedding"
                )
                return None

            self.logger.debug(
                f"📝 Understanding embedding text ({len(text)} chars): {text[:200]}..."
            )

            # Embed via Voyage AI with input_type="document".
            self._last_provider = None  # reset so we can read the actual provider
            embedding = await self._generate_text_embedding(
                text=text,
                input_type="document",
                job_id=job_id,
                workspace_id=workspace_id,
                user_id=user_id,
                product_id=product_id,
                image_id=image_id,
            )

            if not embedding:
                return None

            self.logger.info(
                f"✅ Understanding embedding generated ({len(embedding)}D, "
                f"schema_v{SCHEMA_VERSION})"
            )
            # Provenance accuracy: report the ACTUAL provider that returned the
            # vector (`self._last_provider` is stamped by _generate_text_embedding
            # to the actual model used — voyage-3, voyage-3.5, voyage-4, etc.).
            # Previously this hardcoded "voyage-4" regardless, which lied when
            # Settings.voyage_model was set to a different version. The
            # 2026-05-23 audit caught this drift-detection blind spot.
            return {
                "embedding": embedding,
                "embedding_model": self._last_provider or self.voyage_model or "voyage-3.5",
                "schema_version": SCHEMA_VERSION,
            }

        except Exception as e:
            self.logger.error(f"❌ Understanding embedding generation failed: {e}")
            return None

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

    # ────────────────────────────────────────────────────────────────────────────
    # Page embeddings (#239) — the 8th fusion vector
    # ────────────────────────────────────────────────────────────────────────────

    async def _voyage_multimodal_embed(
        self,
        content: List[Dict[str, str]],
        input_type: str,
        job_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        product_id: Optional[str] = None,
        image_id: Optional[str] = None,
        task: str = "page_embedding_generation",
    ) -> Optional[Dict[str, Any]]:
        """One call to Voyage's multimodal endpoint. Returns vector + real usage.

        `content` is Voyage's interleaved list: `{"type": "text", "text": ...}` and
        `{"type": "image_base64", "image_base64": "data:image/png;base64,..."}` items
        that are embedded TOGETHER into a single vector — that fusion is the whole
        point, and it is why the page vector can match a query against a product name
        that exists only as pixels inside a photograph.

        There is deliberately NO fallback provider. voyage-4 is a text-only model in a
        different latent space, and OpenAI has no equivalent; substituting either on
        failure would push a wrong-space vector into `page_embeddings` and quietly
        corrode every page search. That is audit gap B's lesson, applied before it can
        happen rather than after. On failure we return None and the page stays
        unembedded for the backfill to retry — visible, and recoverable.
        """
        from app.config import settings as _settings

        if not self.voyage_api_key:
            self.logger.warning("⚠️ VOYAGE_API_KEY not set — cannot generate page embedding")
            return None

        model = _settings.voyage_multimodal_model
        expected_dim = _settings.voyage_multimodal_dimension
        start_time = time.time()

        request_data: Dict[str, Any] = {
            "model": model,
            "inputs": [{"content": content}],
            "input_type": input_type,
            "truncation": True,
        }

        try:
            async with self._get_voyage_semaphore(), httpx.AsyncClient() as client:
                async def _post():
                    return await client.post(
                        "https://api.voyageai.com/v1/multimodalembeddings",
                        headers={
                            "Authorization": f"Bearer {self.voyage_api_key}",
                            "Content-Type": "application/json",
                        },
                        json=request_data,
                        # Generous vs the 30s text timeout: the request body carries a
                        # ~2 megapixel PNG, so upload alone can outlast a text call.
                        timeout=90.0,
                    )

                response = await _post()

                # Same 429 discipline as the text path: honour Retry-After instead of
                # letting a rate-limit window look like a hard failure.
                rate_limit_attempt = 0
                throttled_ms = 0
                while response.status_code == 429 and rate_limit_attempt < 3:
                    try:
                        retry_after = min(60.0, float(response.headers.get("Retry-After", "5")))
                    except ValueError:
                        retry_after = 5.0
                    self.logger.warning(
                        f"⚠️ Voyage multimodal 429 (attempt {rate_limit_attempt+1}/3); "
                        f"sleeping {retry_after}s"
                    )
                    # MV2-10, same as the text path: record the throttling so a clean
                    # call and a thrice-throttled one are not the same log row.
                    throttled_ms += int(retry_after * 1000)
                    await asyncio.sleep(retry_after)
                    response = await _post()
                    rate_limit_attempt += 1

                if response.status_code != 200:
                    raise Exception(
                        f"Voyage multimodal API error: {response.status_code} - "
                        f"{response.text[:300]}"
                    )

                data = response.json()
                embedding = data["data"][0]["embedding"]

                if len(embedding) != expected_dim:
                    # Never store a wrong-dim vector: the collection is fixed-width, so
                    # this would fail at the Postgres boundary anyway — but failing here
                    # names the real cause (model/config mismatch) instead of a dim error.
                    self.logger.error(
                        f"❌ Voyage multimodal returned {len(embedding)}D, expected "
                        f"{expected_dim}D (model={model}). Refusing to store."
                    )
                    return None

                usage = data.get("usage", {}) or {}
                text_tokens = int(usage.get("text_tokens") or 0)
                image_pixels = int(usage.get("image_pixels") or 0)
                latency_ms = int((time.time() - start_time) * 1000)

                # Cost is BOTH tokens and pixels; the token-only path would under-report
                # a page by ~20×. See calculate_multimodal_embedding_cost.
                from app.config.ai_pricing import AIPricingConfig
                costs = AIPricingConfig.calculate_multimodal_embedding_cost(
                    model=model, text_tokens=text_tokens, image_pixels=image_pixels,
                )

                await self.ai_logger.log_ai_call(
                    task=task,
                    model=model,
                    input_tokens=text_tokens,
                    output_tokens=0,
                    cost=float(costs["raw_cost_usd"]),
                    latency_ms=latency_ms,
                    confidence_score=0.95,
                    confidence_breakdown={
                        "model_confidence": 0.98,
                        "completeness": 1.0,
                        "consistency": 0.95,
                        "validation": 0.90,
                        "vectors_generated": 1,
                        "vector_dimension": expected_dim,
                        "vector_kind": "page",
                        "rate_limit_retries": rate_limit_attempt,
                        "throttled_ms": throttled_ms,
                        "image_pixels": image_pixels,
                        "billable_pixels": int(costs["billable_pixels"]),
                    },
                    action="use_ai_result",
                    job_id=job_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    product_id=product_id,
                    image_id=image_id,
                )

                self.logger.info(
                    f"✅ Page embedding generated ({expected_dim}D, {model}, "
                    f"{text_tokens} text tokens, {image_pixels} px, {latency_ms}ms)"
                )
                return {
                    "embedding": embedding,
                    "embedding_model": model,
                    "text_tokens": text_tokens,
                    "image_pixels": image_pixels,
                }

        except Exception as e:
            self.logger.error(f"❌ Voyage multimodal embedding failed: {e}")
            latency_ms = int((time.time() - start_time) * 1000)
            try:
                await self.ai_logger.log_ai_call(
                    task=task,
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    cost=0.0,
                    latency_ms=latency_ms,
                    confidence_score=0.0,
                    confidence_breakdown={
                        "model_confidence": 0.0, "completeness": 0.0,
                        "consistency": 0.0, "validation": 0.0,
                    },
                    # NOT "fallback_to_rules": nothing falls back here by design, and
                    # logging a fallback that does not exist would make the dashboards
                    # read as "handled" on a path where the page simply has no vector.
                    action="fallback_failed",
                    job_id=job_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    product_id=product_id,
                    image_id=image_id,
                    fallback_reason="no fallback provider for multimodal embeddings",
                    error_message=str(e),
                )
            except Exception:
                pass
            return None

    async def generate_page_embedding(
        self,
        image_base64: str,
        page_text: Optional[str] = None,
        job_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        product_id: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Embed one rendered catalog page (image + its extracted text) as one vector.

        Args:
            image_base64: PNG/JPEG bytes, base64. A bare payload or a full
                `data:image/png;base64,...` URI both work.
            page_text: The page's text as SILVER already holds it (document_chunks),
                not re-extracted from the PDF. Optional — an image-only page still
                embeds fine, which matters because the pages this vector exists for
                are precisely the ones whose text the OCR pass could not read.

        Returns dict with `embedding` + provenance, or None on failure.
        """
        if not image_base64:
            self.logger.warning("⚠️ generate_page_embedding called without image data")
            return None

        # Voyage wants a data URI. Accept either form so callers can pass raw base64.
        if not image_base64.startswith("data:"):
            image_base64 = f"data:image/png;base64,{image_base64}"

        content: List[Dict[str, str]] = []
        if page_text and page_text.strip():
            # Text first, then the image: Voyage's interleaved order is preserved, and
            # leading with text keeps the page's own words in context if the combined
            # input ever hits the 32k-token truncation limit.
            content.append({"type": "text", "text": page_text.strip()})
        content.append({"type": "image_base64", "image_base64": image_base64})

        return await self._voyage_multimodal_embed(
            content=content,
            input_type="document",
            job_id=job_id,
            workspace_id=workspace_id,
            user_id=user_id,
            product_id=product_id,
            image_id=image_id,
        )

    async def generate_page_query_embedding(self, query: str) -> Dict[str, Any]:
        """Embed a text query into the PAGE vector space for search.

        Must use the same multimodal model the pages were embedded with, with
        `input_type="query"`. Reaching for the ordinary voyage-4 text embedding here
        would be the easy mistake and a silent one: both are 1024D, so the query would
        be accepted by the collection and return confidently-scored nonsense rather
        than erroring. Different model, different space — dimension agreement proves
        nothing.
        """
        result = await self._voyage_multimodal_embed(
            content=[{"type": "text", "text": query}],
            input_type="query",
            task="page_query_embedding_generation",
        )
        if result and result.get("embedding"):
            return {"success": True, "embedding": result["embedding"], "model": result["embedding_model"]}
        return {"success": False, "error": "Page query embedding returned None"}

    async def generate_batch_embeddings(
        self,
        texts: List[str],
        dimensions: int = 1024,
        input_type: str = "document",
        truncation: bool = True,
        output_dtype: str = "float",
        job_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        product_id: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in a single batch API call.

        This is optimized for Voyage AI's batch embedding endpoint, which is more
        efficient than making individual calls. On Voyage failure every entry comes
        back None - there is no second provider (see the module docstring).

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

                # Audit #12 finding 3: this said "voyage-4" unconditionally while the
                # provenance stamp below wrote self.voyage_model, so setting
                # VOYAGE_MODEL to anything else would have batch-indexed rows
                # embedded by voyage-4 and queried with the new model -- both 1024D,
                # so VECS accepts the mixed-space vector and ranks confident
                # nonsense instead of raising. Latent only because the config
                # default happens to match. The single-text path already used
                # self.voyage_model; only the batch path diverged.
                request_data = {
                    "model": self.voyage_model,
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

                    # Validate the response against what we asked for BEFORE any
                    # caller can zip it back onto its inputs. A batch is positional:
                    # caller i gets embeddings[i]. A 100-text batch that came back
                    # with 99 vectors, or came back out of order, silently attached
                    # every chunk to its neighbour's embedding -- stored, indexed and
                    # ranked with nothing raising. Voyage returns an explicit `index`
                    # per item; order is not part of the contract, so sort by it
                    # rather than trusting arrival order.
                    items = data.get("data") or []
                    if len(items) != len(processed_texts):
                        raise ValueError(
                            f"Voyage batch returned {len(items)} embeddings for "
                            f"{len(processed_texts)} inputs -- refusing to misalign them"
                        )
                    try:
                        items = sorted(items, key=lambda it: int(it["index"]))
                    except (KeyError, TypeError, ValueError) as idx_err:
                        raise ValueError(
                            f"Voyage batch response items lack a usable 'index': {idx_err}"
                        )
                    if [int(it["index"]) for it in items] != list(range(len(processed_texts))):
                        raise ValueError(
                            "Voyage batch response indices are not a complete 0..n-1 run"
                        )

                    embeddings = [item.get("embedding") for item in items]
                    wrong = [
                        i for i, e in enumerate(embeddings)
                        if not isinstance(e, list) or len(e) != voyage_dimensions
                    ]
                    if wrong:
                        raise ValueError(
                            f"Voyage batch returned {len(wrong)} vector(s) that are not "
                            f"{voyage_dimensions}D (first offending index {wrong[0]}) -- "
                            f"a wrong-width vector is a wrong SPACE, never a usable vector"
                        )

                    # Log AI call with proper cost calculation
                    latency_ms = int((time.time() - start_time) * 1000)
                    usage = data.get("usage", {})
                    input_tokens = usage.get("total_tokens", 0)

                    # Voyage AI Pricing (as of Dec 2024)
                    cost_per_million = 0.06  # voyage-4
                    cost = (input_tokens / 1_000_000) * cost_per_million

                    await self.ai_logger.log_ai_call(
                        task="batch_text_embedding_generation",
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
                            "validation": 0.90,
                            "batch_size": len(texts)
                        },
                        action="use_ai_result",
                        job_id=job_id,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        product_id=product_id,
                        image_id=image_id,
                    )

                    self.logger.info(f"✅ Generated {len(embeddings)} Voyage AI embeddings in batch ({voyage_dimensions}D, {input_type})")
                    # Stamp the actual provider so downstream provenance writes
                    # (e.g. document_chunks.embedding_model in rag_service.py)
                    # don't lie. Without this, the chunk provenance fix from
                    # 2026-05-23 round-3 was reading stale state from the LAST
                    # single-text call. Drift detection blind spot - fixed post-round-3.
                    self._last_provider = self.voyage_model or "voyage-4"
                    return embeddings
                else:
                    error_body = response.text
                    self.logger.error(f"Voyage AI batch API error {response.status_code}: {error_body}")
                    self.logger.error(f"Request data: {request_data}")
                    raise Exception(f"Voyage AI API error: {response.status_code} - {error_body}")

            except Exception as e:
                self.logger.error(f"Voyage AI batch embedding failed: {e}")

                # Log the failed Voyage call
                latency_ms = int((time.time() - start_time) * 1000)
                voyage_dimensions = 1024 if dimensions == 1536 else dimensions
                await self.ai_logger.log_ai_call(
                    task="batch_text_embedding_generation",
                    model=f"{self.voyage_model}-{voyage_dimensions}d",
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
                    # NOT "fallback_to_rules" - nothing falls back. See the
                    # single-text path for the same reasoning.
                    action="fallback_failed",
                    job_id=job_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    product_id=product_id,
                    image_id=image_id,
                    fallback_reason="no fallback provider for text embeddings",
                    error_message=str(e)
                )
                return [None] * len(texts)

        # Voyage disabled or unkeyed. A list of Nones is the honest answer: every
        # caller already treats a None entry as "this text has no vector", which is
        # recoverable. A vector from a different model would not be.
        self.logger.error(
            "Voyage AI unavailable (disabled or no API key) and there is no "
            f"fallback embedder - returning {len(texts)} empty results"
        )
        return [None] * len(texts)

    async def _generate_text_embedding(
        self,
        text: str,
        job_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        dimensions: int = 1024,
        input_type: Optional[str] = None,
        truncation: bool = True,
        output_dtype: str = "float",
        product_id: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> Optional[List[float]]:
        """Generate a text embedding using Voyage AI. There is no second provider.

        Args:
            text: Text to embed
            job_id: Optional job ID for logging
            dimensions: Embedding dimensions (default 1024 for Voyage AI; 256, 512, 1024, 2048 supported)
            input_type: None (default), "document" for indexing, "query" for search (Voyage AI only)
            truncation: Whether to truncate text to fit context length (Voyage AI only, default: True)
            output_dtype: Output data type - 'float', 'int8', 'uint8', 'binary', 'ubinary'

        Returns:
            List of floats representing the embedding, or None if Voyage failed.
            None means NO VECTOR - callers must handle it. It never means
            "a vector from somewhere else".
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
                    # 429s would otherwise fall through to the generic except and be
                    # reported as a hard failure, losing work to a transient window.
                    # Respect Retry-After header (Voyage sets it). Up to 3 retries
                    # with capped backoff before giving up to fallback.
                    rate_limit_attempt = 0
                    throttled_ms = 0
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
                        # MV2-10: count the throttling. Up to 3 retries collapsed into
                        # one log row, so a clean first-call success and a success after
                        # three throttled attempts were indistinguishable — and the
                        # latency of the second is dominated by sleep, not by Voyage.
                        # Carried into confidence_breakdown below rather than emitted as
                        # extra rows: the retries cost WAITING, not tokens, and a row per
                        # attempt would triple the apparent spend of one embedding.
                        throttled_ms += int(retry_after * 1000)
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

                        # voyage-4: $0.06 per 1M tokens (sole production embedder)
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
                                "validation": 0.90,
                                "rate_limit_retries": rate_limit_attempt,
                                "throttled_ms": throttled_ms,
                            },
                            action="use_ai_result",
                            job_id=job_id,
                            workspace_id=workspace_id,
                            user_id=user_id,
                            product_id=product_id,
                            image_id=image_id,
                        )

                        self.logger.info(f"✅ Generated Voyage AI embedding ({voyage_dimensions}D, {input_type})")
                        self._last_provider = self.voyage_model or "voyage-4"
                        return embedding
                    else:
                        error_body = response.text
                        self.logger.error(f"Voyage AI API error {response.status_code}: {error_body}")
                        self.logger.error(f"Request data: {request_data}")
                        raise Exception(f"Voyage AI API error: {response.status_code} - {error_body}")

            except Exception as e:
                self.logger.error(f"Voyage AI text embedding failed: {e}")

                # Log the failed Voyage call
                latency_ms = int((time.time() - start_time) * 1000)
                voyage_dimensions = 1024 if dimensions == 1536 else dimensions
                await self.ai_logger.log_ai_call(
                    task="text_embedding_generation",
                    model=f"{self.voyage_model}-{voyage_dimensions}d",
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
                    # NOT "fallback_to_rules": nothing falls back. Reporting a
                    # fallback that does not exist makes the dashboards read as
                    # "handled" on a path where the caller simply gets no vector.
                    action="fallback_failed",
                    job_id=job_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    product_id=product_id,
                    image_id=image_id,
                    fallback_reason="no fallback provider for text embeddings",
                    error_message=str(e)
                )
                return None

        # Voyage disabled or unkeyed. There is nothing else to try - see the module
        # docstring for why there is no second provider.
        self.logger.error(
            "Voyage AI unavailable (disabled or no API key) and there is no "
            "fallback embedder - returning None"
        )
        return None

    async def _generate_visual_embedding(
        self,
        image_url: Optional[str],
        image_data: Optional[str],
        confidence_threshold: float = 0.8,
        pil_image = None,  # NEW: Accept pre-decoded PIL image
        job_id: Optional[str] = None,  # NEW: Add job_id for logging
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        product_id: Optional[str] = None,
        image_id: Optional[str] = None,
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
            image_url, image_data, pil_image=pil_image, job_id=job_id, workspace_id=workspace_id,
            product_id=product_id, image_id=image_id,
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
        job_id: Optional[str] = None,  # NEW: Add job_id for logging
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        product_id: Optional[str] = None,
        image_id: Optional[str] = None,
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

                # Audit #12: `image_url` was accepted as a parameter and then never
                # read. An image known only by URL therefore reached
                # get_image_embedding(None) -- no visual_768 vector, while the other
                # channels still reported success, so the image looked embedded and
                # had no visual vector at all. Fetch it, through the shared SSRF
                # guard because the URL can come from a supplier feed or an LLM.
                if pil_image is None and image_url:
                    try:
                        from app.utils.ssrf_guard import assert_safe_url, SSRFError
                        try:
                            assert_safe_url(image_url, allow_schemes=("https",))
                        except SSRFError as ssrf_err:
                            self.logger.error(f"❌ Blocked image_url for SLIG ({ssrf_err})")
                            return None, None
                        import httpx as _httpx
                        async with _httpx.AsyncClient(
                            timeout=30.0, follow_redirects=False
                        ) as _c:
                            _r = await _c.get(image_url)
                        if _r.status_code != 200:
                            self.logger.error(
                                f"❌ image_url fetch for SLIG returned HTTP {_r.status_code}"
                            )
                            return None, None
                        _body = _r.content
                        if len(_body) > 10 * 1024 * 1024:
                            self.logger.error("❌ image_url for SLIG exceeds the 10MB cap")
                            return None, None
                        pil_image = Image.open(io.BytesIO(_body)).convert('RGB')
                    except Exception as _fetch_err:
                        self.logger.error(f"❌ Could not fetch image_url for SLIG: {_fetch_err}")
                        return None, None

                # Never hand SLIG a None image: it produced a confusing downstream
                # failure instead of an honest "this image has no visual vector".
                if pil_image is None:
                    self.logger.error(
                        "❌ SLIG called with no decodable image (no pil_image, "
                        "no image_data, no usable image_url) — returning no vector"
                    )
                    return None, None

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
                            "❌ SLIG endpoint returned wrong-dim embedding 3x in a row — "
                            "refusing to store. The endpoint is likely misconfigured "
                            "(serving wrong model). Operator action required."
                        )
                        # MV2-9: log the FAILURE too. Three GPU calls were made and paid
                        # for on the way here; returning without a log row meant a
                        # misconfigured endpoint burned real money that appeared nowhere —
                        # the spend was invisible precisely when it was pure waste. The
                        # latency is the whole retry loop, which is what was actually
                        # billed. action="fallback_failed" so this cannot be mistaken for
                        # a successful embed in the dashboards.
                        await self.ai_logger.log_time_based_call(
                            task="visual_embedding_generation",
                            model="slig-768d",
                            latency_ms=latency_ms,
                            action="fallback_failed",
                            confidence_score=0.0,
                            confidence_breakdown={
                                "model_confidence": 0.0,
                                "completeness": 0.0,
                                "consistency": 0.0,
                                "validation": 0.0,
                                "vectors_generated": 0,
                                "attempts": 3,
                                "failure": "dimension_mismatch",
                                "expected_dimension": self.slig_embedding_dimension,
                                "endpoint_model": self.slig_model_name,
                            },
                            job_id=job_id,
                            workspace_id=workspace_id,
                            user_id=user_id,
                            product_id=product_id,
                            image_id=image_id,
                        )
                        return None, None

                    self.logger.info(f"✅ SLIG embedding generated: {len(embedding)}D (latency={latency_ms}ms)")

                    # SLIG runs on a HuggingFace GPU endpoint — bill per
                    # GPU-second (time-based), NOT per token. The old
                    # log_ai_call token path yielded $0.00 for this call
                    # (0 tokens) and was a latent billing gap. model="slig-768d"
                    # routes to VISUAL_EMBEDDING_PRICING (time-based) in
                    # ai_pricing.calculate_time_based_cost.
                    await self.ai_logger.log_time_based_call(
                        task="visual_embedding_generation",
                        model="slig-768d",
                        latency_ms=latency_ms,
                        confidence_score=0.95,
                        confidence_breakdown={
                            "model_confidence": 0.98,
                            "completeness": 1.0,
                            "consistency": 0.95,
                            "validation": 0.90,
                            # Each visual call yields ONE 768D vector. The 4
                            # per-aspect Voyage embeddings (color/texture/style/
                            # material) are logged separately by
                            # _generate_specialized_aspect_embeddings under
                            # task="aspect_embeddings_batch".
                            "vectors_generated": 1,
                            "vector_dimension": self.slig_embedding_dimension,
                            "vector_kind": "visual",
                            "endpoint_model": self.slig_model_name,
                        },
                        job_id=job_id,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        product_id=product_id,
                        image_id=image_id,
                    )

                    return embedding, pil_image

                except Exception as e:
                    self.logger.error(f"❌ SLIG cloud endpoint failed: {e}")
                    # Same reasoning as the dim-mismatch arm above: the request was
                    # issued, so GPU time was consumed whatever the exception was.
                    # A timeout on a paid endpoint costs exactly as much as a success.
                    try:
                        await self.ai_logger.log_time_based_call(
                            task="visual_embedding_generation",
                            model="slig-768d",
                            latency_ms=int((time.time() - start_time) * 1000),
                            action="fallback_failed",
                            confidence_score=0.0,
                            confidence_breakdown={
                                "model_confidence": 0.0,
                                "completeness": 0.0,
                                "consistency": 0.0,
                                "validation": 0.0,
                                "vectors_generated": 0,
                                "failure": type(e).__name__,
                                "endpoint_model": self.slig_model_name,
                            },
                            job_id=job_id,
                            workspace_id=workspace_id,
                            user_id=user_id,
                            product_id=product_id,
                            image_id=image_id,
                        )
                    except Exception as log_err:
                        # Never let the cost log turn a soft failure into a hard one.
                        self.logger.warning(f"Could not log failed SLIG call: {log_err}")
                    return None, None
            else:
                self.logger.error("❌ SLIG visual embeddings are disabled")
                return None, None



        except Exception as e:
            self.logger.error(f"SLIG embedding generation failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

        return None, None

    async def _generate_specialized_aspect_embeddings(
        self,
        vision_analysis: Any,
        job_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        product_id: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> Optional[Dict[str, List[float]]]:
        """Generate 4 per-image aspect embeddings (1024D Voyage) from VisionAnalysis.

        Replaces the legacy SLIG-blend trick (`_generate_specialized_siglip_embeddings`)
        which produced 4 vectors that were ~80% identical to the base image
        embedding because they were just blended copies of it with 4 fixed
        global text directions. This new path embeds **per-image** aspect
        text derived from the vision-model's structured output, so the
        4 vectors actually carry independent per-aspect signal.

        Source mapping (see app.models.vision_analysis):
          color    → VisionAnalysis.colors[]
          texture  → VisionAnalysis.textures[] + finish
          style    → VisionAnalysis.style + surface_pattern + applications
          material → VisionAnalysis.material_type + category + subcategory

        Behavior:
          - color/texture/style aspects skip when their source fields are
            empty (returns dict missing that key — caller upserts only the
            ones present). Material always returns text since material_type
            is required.
          - Returns None on hard failure (vision_analysis unparseable, all
            Voyage calls failed). Caller treats None as "skip aspect
            embeddings entirely for this image".
          - Each per-aspect Voyage call is logged via ai_call_logger so
            cost attribution lines up with the rest of the pipeline.

        Cost: 4 short Voyage `voyage-3` text embeddings per image. Aspect
        strings average <30 tokens, so ~$0.0001 per image — much cheaper
        than the ~12 SLIG calls the legacy path made.
        """
        # Normalize input → VisionAnalysis instance. Accepts dict (from
        # cached DB JSON), VisionAnalysis (when called directly from
        # ingestion), or legacy dict shape (from pre-schema rows).
        try:
            if isinstance(vision_analysis, VisionAnalysis):
                va = vision_analysis
            elif isinstance(vision_analysis, dict):
                if not _is_valid_vision_analysis_schema(vision_analysis):
                    self.logger.info(
                        "⚠️ Aspect embeddings: vision_analysis dict failed schema validation — skipping"
                    )
                    return None
                try:
                    va = VisionAnalysis(**vision_analysis)
                except Exception:
                    # Fall back to legacy coercion for older rows whose JSON
                    # predates the strict schema.
                    va = vision_analysis_from_legacy_dict(vision_analysis)
                    if va is None:
                        self.logger.info(
                            "⚠️ Aspect embeddings: legacy vision_analysis coercion failed — skipping"
                        )
                        return None
            else:
                self.logger.info(
                    f"⚠️ Aspect embeddings: unsupported vision_analysis type {type(vision_analysis)} — skipping"
                )
                return None
        except Exception as e:
            self.logger.error(f"❌ Aspect embeddings: VisionAnalysis parse failed: {e}")
            return None

        # Build 4 deterministic aspect strings via the registry. None means
        # the source fields didn't carry enough text for that aspect.
        aspect_texts: Dict[str, Optional[str]] = {
            aspect: serializer(va) for aspect, serializer in ASPECT_SERIALIZERS.items()
        }

        # Embed each non-None aspect string via Voyage 1024D. We allow per-
        # aspect skip (rather than all-or-nothing) because color/texture/
        # style are legitimately optional — we don't want a missing color
        # field to also wipe out a perfectly good material vector.
        #
        # On a Voyage outage the aspect for this image stays unembedded and the
        # backfill cron picks it up next run. It cannot instead be filled by another
        # provider's same-dimension vector - that fallback was removed, so mixed
        # spaces in these four collections are structurally impossible now.
        embeddings: Dict[str, List[float]] = {}
        any_failure = False
        for aspect, text in aspect_texts.items():
            if not text:
                self.logger.debug(f"⏭️ Aspect '{aspect}' skipped — empty source text")
                continue
            try:
                vec = await self._generate_text_embedding(
                    text=text,
                    input_type="document",
                    job_id=job_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    product_id=product_id,
                    image_id=image_id,
                )
                if not vec:
                    self.logger.warning(f"⚠️ Aspect '{aspect}' Voyage embed returned None")
                    any_failure = True
                    continue
                if len(vec) != 1024:
                    self.logger.error(
                        f"❌ Aspect '{aspect}' wrong dim: got {len(vec)}, expected 1024"
                    )
                    any_failure = True
                    continue
                embeddings[aspect] = vec
                self.logger.debug(
                    f"✅ Aspect '{aspect}' embedded: '{text[:60]}{'…' if len(text) > 60 else ''}'"
                )
            except Exception as e:
                self.logger.error(f"❌ Aspect '{aspect}' embed failed: {e}", exc_info=True)
                any_failure = True

        if not embeddings:
            self.logger.warning(
                "⚠️ Aspect embeddings: 0/4 generated (all aspects empty or all failed) — skipping"
            )
            return None

        # Aggregate cost log so analytics shows one row per image rather
        # than four. Confidence breakdown mirrors the understanding-
        # embedding logger so dashboards can stack them.
        try:
            await self.ai_logger.log_ai_call(
                task="aspect_embeddings_batch",
                model=self.voyage_model,  # S3-3: was hardcoded "voyage-3"; aspects use voyage_model
                input_tokens=0,  # voyage doesn't surface token count on text embed
                output_tokens=0,
                cost=0.0,  # cost rolled up by Voyage account-level billing
                latency_ms=0,
                confidence_score=0.95,
                confidence_breakdown={
                    "model_confidence": 0.98,
                    "completeness": 1.0 if not any_failure else 0.7,
                    "consistency": 1.0,
                    "validation": 1.0,
                    "vectors_generated": len(embeddings),
                    "vector_dimension": 1024,
                    "vector_kinds": list(embeddings.keys()),
                    "schema_version": SCHEMA_VERSION,
                },
                action="use_ai_result",
                job_id=job_id,
                workspace_id=workspace_id,
                user_id=user_id,
                product_id=product_id,
                image_id=image_id,
            )
        except Exception as log_err:
            self.logger.debug(f"Aspect aggregate log skipped: {log_err}")

        return embeddings





