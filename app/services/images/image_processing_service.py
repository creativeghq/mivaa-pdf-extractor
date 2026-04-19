"""
Image Processing Service - Handles image extraction, classification, upload, and CLIP generation.

This service encapsulates all image-related operations in the PDF processing pipeline:
1. Extract images from PDF
2. Classify images (material vs non-material) using Qwen/Claude
3. Upload material images to Supabase Storage
4. Save images to database
5. Generate CLIP embeddings
"""

import os
import base64
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from asyncio import Semaphore
import logging

from app.services.core.supabase_client import get_supabase_client
from app.services.embeddings.vecs_service import VecsService
from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
from app.services.images.vision_provider import VisionProvider
from app.services.pdf.pdf_processor import PDFProcessor
# PageConverter removed - using simple PDF page numbers instead
from app.config import get_settings


logger = logging.getLogger(__name__)


def _detect_image_media_type(image_bytes: bytes, file_path: str = "") -> str:
    """Detect actual image media type from magic bytes, falling back to file extension."""
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if image_bytes[:3] == b'\xff\xd8\xff':
        return "image/jpeg"
    if image_bytes[:4] == b'GIF8':
        return "image/gif"
    if image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
        return "image/webp"
    # Fall back to file extension
    ext = file_path.lower().rsplit('.', 1)[-1] if '.' in file_path else ''
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/jpeg")


# ============================================================================
# IMAGE CLASSIFICATION CATEGORY MAPPING
# ============================================================================
# The Qwen vision model returns these categories:
#   - PRODUCT_IMAGE: Clear shot of a material/product (KEEP)
#   - MIXED: Product with context/environment (KEEP)
#   - DECORATIVE: Lifestyle/artistic images without clear product (FILTER)
#   - TECHNICAL_DIAGRAM: Charts, graphs, technical drawings (FILTER)
#
# We map these to is_material=True/False for downstream processing
# ============================================================================

MATERIAL_CATEGORIES = {'PRODUCT_IMAGE', 'MIXED'}  # These ARE material images
NON_MATERIAL_CATEGORIES = {'DECORATIVE', 'TECHNICAL_DIAGRAM'}  # These are NOT


def is_material_classification(classification: str) -> bool:
    """
    Check if a classification indicates a material image.

    Args:
        classification: The classification string from Qwen (e.g., 'PRODUCT_IMAGE', 'DECORATIVE')

    Returns:
        True if this is a material image, False otherwise
    """
    if not classification:
        return False
    category = classification.upper().strip()
    return category in MATERIAL_CATEGORIES


class ImageProcessingService:
    """Service for handling all image processing operations."""

    def __init__(self, workspace_id: str = None):
        """Initialize service."""
        self.supabase_client = get_supabase_client()
        self.vecs_service = VecsService()
        self.embedding_service = RealEmbeddingsService()
        self.pdf_processor = PDFProcessor()
        self.settings = get_settings()
        self.workspace_id = workspace_id or self.settings.default_workspace_id
        self.classification_prompt = None
        self.material_analyzer_prompt = None  # Rich material analysis (feeds understanding embedding)
        self.material_analyzer_system_prompt = None
        self._load_classification_prompt()
        self._load_material_analyzer_prompt()

    def _load_classification_prompt(self) -> None:
        """Load image classification prompt from database."""
        try:
            result = self.supabase_client.client.table('prompts')\
                .select('prompt_text')\
                .eq('prompt_type', 'classification')\
                .eq('stage', 'image_analysis')\
                .eq('category', 'image_classification')\
                .eq('is_active', True)\
                .order('version', desc=True)\
                .limit(1)\
                .execute()

            if result.data and len(result.data) > 0:
                self.classification_prompt = result.data[0]['prompt_text']
                logger.info("✅ Loaded classification prompt from database")
            else:
                logger.warning("⚠️ Classification prompt not found in database. Add via /admin/ai-configs - classification will fail!")
                self.classification_prompt = None

        except Exception as e:
            logger.error(f"❌ Failed to load classification prompt from database: {e}")
            self.classification_prompt = None

    def _load_material_analyzer_prompt(self) -> None:
        """
        Load the rich material analysis prompt from the database.

        This prompt is used AFTER classification confirms an image is a material,
        to extract structured properties (material type, color, texture, finish,
        applications) which are then:
          1. Stored on `document_images.vision_analysis` (JSONB)
          2. Passed to `RealEmbeddingsService.generate_all_embeddings(vision_analysis=...)`
             so the Voyage understanding embedding (1024D) gets generated and
             written to `vecs.image_understanding_embeddings`.

        Without this prompt loaded, the understanding branch is skipped and the
        7-vector fusion search degrades to 6 vectors.
        """
        try:
            result = self.supabase_client.client.table('prompts')\
                .select('prompt_text, system_prompt')\
                .eq('name', 'Material Image Analyzer')\
                .eq('is_active', True)\
                .limit(1)\
                .execute()

            if result.data and len(result.data) > 0:
                self.material_analyzer_prompt = result.data[0].get('prompt_text')
                self.material_analyzer_system_prompt = result.data[0].get('system_prompt')
                logger.info("✅ Loaded Material Image Analyzer prompt from database")
            else:
                logger.warning(
                    "⚠️ 'Material Image Analyzer' prompt not found in database. "
                    "Understanding embeddings will be skipped — search will use 6 of 7 vectors."
                )
                self.material_analyzer_prompt = None
        except Exception as e:
            logger.error(f"❌ Failed to load Material Image Analyzer prompt from database: {e}")
            self.material_analyzer_prompt = None

    async def classify_images(
        self,
        extracted_images: List[Dict[str, Any]],
        confidence_threshold: float = 0.6,  # OPTIMIZED: Lowered from 0.7 to reduce validation calls
        primary_model: str = "Qwen/Qwen3-VL-32B-Instruct",  # PRIMARY: Qwen3-VL-32B (reliable, high accuracy)
        validation_model: str = "claude-sonnet-4-7",  # FALLBACK: Claude Sonnet 4.5 (highest quality)
        batch_size: int = 15,  # NEW: Process images in batches to prevent OOM
        job_id: Optional[str] = None  # NEW: Job ID for AI cost tracking
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Classify images as material or non-material using Qwen Vision models.

        SUPPORTED MODELS:
        - Qwen/Qwen3-VL-32B-Instruct: PRIMARY - High accuracy, reliable ($0.50/1M input, $1.50/1M output)
        - claude-sonnet-4-7: FALLBACK - Highest quality vision model (for failures/low confidence)

        MEMORY OPTIMIZATIONS:
        - Processes images in batches (default: 15) to prevent OOM crashes
        - Explicit garbage collection after each batch
        - Cleanup of base64 strings after API calls
        - Lower confidence threshold (0.6) to reduce expensive validation calls

        Args:
            extracted_images: List of extracted image data
            confidence_threshold: Threshold for validation (default: 0.6)
            primary_model: Primary classification model (default: Qwen3-VL-32B)
            validation_model: Fallback model for failures/low confidence (default: Claude Sonnet 4.5)
            batch_size: Number of images to process per batch (default: 15)
            job_id: Optional job ID for AI cost tracking/aggregation

        Returns:
            Tuple of (material_images, non_material_images)
        """
        import gc  # ✅ NEW: For explicit garbage collection
        import time
        import traceback

        classification_start_time = time.time()

        logger.info(f"🤖 Starting AI-based image classification for {len(extracted_images)} images...")
        logger.info(f"   Strategy: {primary_model} (fast filter) → {validation_model} (validation for uncertain cases)")
        logger.info(f"   Batch size: {batch_size} images per batch (memory optimization)")
        logger.info(f"   Confidence threshold: {confidence_threshold} (lower = fewer validation calls)")

        # Import AI services
        from app.services.core.ai_client_service import get_ai_client_service
        import httpx
        import json

        ai_service = get_ai_client_service()

        # Get HuggingFace endpoint configuration from settings
        from app.config import get_settings
        from app.services.embeddings.qwen_endpoint_manager import QwenEndpointManager

        settings = get_settings()
        qwen_config = settings.get_qwen_config()

        huggingface_api_key = qwen_config["endpoint_token"]
        # ⚠️ DO NOT use qwen_config["endpoint_url"] here. Per config.py:320-324
        # that field is "only a fallback" — the real URL is resolved dynamically
        # from HF at runtime by the QwenEndpointManager during warmup/resume
        # (qwen_endpoint_manager.py:131-132 logs `🔗 Qwen endpoint URL updated`).
        # If we read from settings the URL is empty/stale and OpenAI client
        # raises APIConnectionError on every classification → Stage 3 fails
        # universally → falls back to Claude. Read it from the LIVE manager
        # below after `qwen_manager` is initialized.
        qwen_endpoint_url = None  # populated from qwen_manager.endpoint_url after init

        if not huggingface_api_key:
            logger.error("❌ CRITICAL: HUGGINGFACE_API_KEY not configured!")
            logger.error("   Image classification will fail. Please set HUGGINGFACE_API_KEY.")
            raise ValueError("HUGGINGFACE_API_KEY not configured")

        # Initialize Qwen endpoint manager - prefer warmed-up manager from registry
        qwen_manager = None
        try:
            from app.services.embeddings.endpoint_registry import endpoint_registry
            registry_qwen = endpoint_registry.get_qwen_manager()
            if registry_qwen is not None:
                qwen_manager = registry_qwen
                logger.info("✅ Using warmed-up Qwen manager from registry")
                # Scale to max replicas for batch image processing
                if hasattr(qwen_manager, 'scale_to_max'):
                    try:
                        qwen_manager.scale_to_max()
                        logger.info("📈 Scaled Qwen to max replicas for batch processing")
                    except Exception as scale_err:
                        logger.warning(f"⚠️ Could not scale Qwen to max: {scale_err}")
        except Exception as e:
            logger.debug(f"Registry Qwen manager not available: {e}")

        # Fallback: Create new manager if registry not available
        if qwen_manager is None:
            qwen_manager = QwenEndpointManager(
                endpoint_url="",  # resolved dynamically by manager from HF
                endpoint_name=qwen_config["endpoint_name"],
                namespace=qwen_config["namespace"],
                endpoint_token=huggingface_api_key,
                enabled=qwen_config["enabled"]
            )
            logger.info("ℹ️ Created new Qwen manager (registry not available)")

        # ⚠️ Pull the LIVE endpoint URL from the manager, not from settings.
        # The manager updates its URL during warmup/resume (qwen_endpoint_manager.py:131-132).
        # Settings may be empty or stale and will cause OpenAI client APIConnectionError.
        qwen_endpoint_url = qwen_manager.endpoint_url
        if not qwen_endpoint_url or not qwen_endpoint_url.startswith(("http://", "https://")):
            logger.error(
                f"❌ CRITICAL: qwen_manager.endpoint_url is invalid ({qwen_endpoint_url!r}). "
                "Manager warmup probably did not run before classify_images was called."
            )
        else:
            logger.info(f"✅ Using live Qwen endpoint URL from manager: {qwen_endpoint_url}")

        # Check if Qwen is enabled and assume it's ready (warmup done at job start)
        qwen_endpoint_available = qwen_config["enabled"] and bool(
            qwen_endpoint_url and qwen_endpoint_url.startswith(("http://", "https://"))
        )
        if qwen_endpoint_available:
            logger.info("✅ Qwen endpoint configured (warmup done at job start)")
        else:
            logger.info("ℹ️ Qwen endpoint disabled - will use Claude for all classifications")

        async def classify_image_with_vision_model(image_path: str, model: str, base64_data: str = None) -> Dict[str, Any]:
            """Fast classification using vision model (Qwen via HuggingFace Inference Endpoint)."""
            import time
            from app.services.core.ai_call_logger import AICallLogger

            # ✅ FIX: If Qwen endpoint is not available, return error to trigger Claude fallback
            if not qwen_endpoint_available:
                return {
                    'is_material': False,
                    'confidence': 0.0,
                    'reason': 'Qwen endpoint unavailable - endpoint failed to resume',
                    'model': f'{model.split("/")[-1]}_api_error',
                    'error': 'Endpoint failed to resume'
                }

            start_time = time.time()
            # If base64_data is provided, we use it directly. Otherwise, we read from disk.
            image_base64 = base64_data
            try:
                if not image_base64:
                    with open(image_path, 'rb') as f:
                        image_bytes = f.read()
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                        del image_bytes

                # Use database prompt - NO FALLBACK
                if self.classification_prompt:
                    classification_prompt = self.classification_prompt
                else:
                    error_msg = "CRITICAL: Classification prompt not found in database. Add via /admin/ai-configs with prompt_type='classification', stage='image_analysis', category='image_classification'"
                    logger.error(f"❌ {error_msg}")
                    raise ValueError(error_msg)

                # ✅ Use OpenAI client for HuggingFace endpoint (llamacpp provides OpenAI-compatible API)
                from openai import AsyncOpenAI

                # Guard: refuse to build a URL without a scheme. A missing scheme
                # here is the root cause of MIVAA-507 "UnsupportedProtocol: Request
                # URL is missing an 'http://' or 'https://' protocol" (296 events).
                if not qwen_endpoint_url or not qwen_endpoint_url.startswith(("http://", "https://")):
                    raise RuntimeError(
                        f"Qwen endpoint URL not configured (got {qwen_endpoint_url!r}). "
                        "Set QWEN_ENDPOINT_URL env var or wait for qwen_endpoint_manager.resume() to populate it."
                    )

                # Ensure base_url ends with /v1/ for OpenAI client compatibility
                # The client will append /chat/completions to this base
                if not qwen_endpoint_url.endswith('/v1/') and not qwen_endpoint_url.endswith('/v1'):
                    base_url = qwen_endpoint_url.rstrip('/') + '/v1/'
                else:
                    base_url = qwen_endpoint_url if qwen_endpoint_url.endswith('/') else qwen_endpoint_url + '/'

                logger.info(f"🔧 Qwen OpenAI client base_url: {base_url}")

                # Tightened client config:
                #   timeout=60  — fast-fail rather than wait 90s for an overloaded replica
                #   max_retries=0 — kill openai SDK retries. Our app-level retry loop +
                #                    AdaptiveConcurrency + Claude fallback handle resilience.
                #                    The SDK's default max_retries=2 used to turn one
                #                    timeout into 3×90s=270s of wasted queue pressure.
                client = AsyncOpenAI(
                    base_url=base_url,  # https://...huggingface.cloud/v1/
                    api_key=huggingface_api_key,
                    timeout=60.0,
                    max_retries=0,
                )

                # ✅ Call endpoint with retry logic for 503 errors
                import random
                max_retries = 3
                response = None
                
                for retry_attempt in range(max_retries):
                    try:
                        response = await client.chat.completions.create(
                            model=model,
                            messages=[{
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_base64}"
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": classification_prompt
                                    }
                                ]
                            }],
                            max_tokens=512,
                            temperature=0.1,
                            stream=False
                        )
                        break  # Success, exit retry loop
                    except Exception as retry_e:
                        error_str = str(retry_e)
                        # Retry on 503 Service Unavailable
                        if '503' in error_str or 'Service Unavailable' in error_str:
                            if retry_attempt < max_retries - 1:
                                delay = min(2 ** retry_attempt + random.uniform(0, 1), 15)
                                logger.warning(f"⏳ Qwen 503 error, retrying in {delay:.1f}s (attempt {retry_attempt + 1}/{max_retries})")
                                import asyncio
                                await asyncio.sleep(delay)
                                continue
                            # All retries exhausted for 503 — return graceful fallback (no Sentry noise)
                            logger.warning(f"⚠️ Qwen 503 persists after {max_retries} retries for {image_path} — falling back to Claude")
                            return {
                                'is_material': False,
                                'confidence': 0.0,
                                'reason': f'Qwen endpoint unavailable (503) after {max_retries} retries',
                                'model': f'{model.split("/")[-1]}_503_fallback',
                                'error': error_str
                            }
                        raise  # Re-raise for non-503 errors

                if response is None:
                    raise Exception("Qwen request failed after all retries")

                # ✅ Parse response from OpenAI format
                result_text = response.choices[0].message.content

                # ✅ FIX: Handle empty/invalid responses
                # Qwen sometimes returns whitespace or single characters instead of JSON
                result_text_stripped = result_text.strip() if result_text else ""

                if not result_text_stripped or len(result_text_stripped) < 10:
                    logger.warning(f"⚠️ Invalid response from {model} for {image_path}")
                    logger.warning(f"   Content length: {len(result_text) if result_text else 0}, Stripped: {len(result_text_stripped)}")
                    logger.warning(f"   Content preview: {repr(result_text[:100]) if result_text else 'None'}")
                    return {
                        'is_material': False,
                        'confidence': 0.0,
                        'reason': f'Invalid response from vision model (length: {len(result_text_stripped)})',
                        'model': f'{model.split("/")[-1]}_invalid_response'
                    }

                # Clean up response text (remove markdown code blocks if present)
                result_text = result_text_stripped
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.startswith("```"):
                    result_text = result_text[3:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                result_text = result_text.strip()

                # Final validation before JSON parsing
                if not result_text.startswith('{'):
                    logger.warning(f"⚠️ Response doesn't start with '{{' for {image_path}")
                    logger.warning(f"   Content: {repr(result_text[:100])}")
                    return {
                        'is_material': False,
                        'confidence': 0.0,
                        'reason': f'Response not JSON format: {result_text[:50]}',
                        'model': f'{model.split("/")[-1]}_not_json'
                    }

                result = json.loads(result_text)

                # Extract model name for logging
                model_short = model.split('/')[-1] if '/' in model else model

                # Log Qwen endpoint call (HuggingFace Inference Endpoint)
                ai_logger = AICallLogger()
                latency_ms = int((time.time() - start_time) * 1000)

                # Get usage from OpenAI response
                usage = response.usage if hasattr(response, 'usage') else None
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0

                # Qwen pricing (HuggingFace Endpoint)
                # Qwen3-VL-32B: $0.40/1M input, $0.40/1M output
                cost = (input_tokens / 1_000_000) * 0.40 + (output_tokens / 1_000_000) * 0.40

                # Convert OpenAI response to dict for logging
                response_dict = {
                    'choices': [{'message': {'content': result_text}}],
                    'usage': {'prompt_tokens': input_tokens, 'completion_tokens': output_tokens}
                }

                # Qwen runs on a HuggingFace endpoint (time-based GPU billing).
                # log_qwen_call uses token-based math which records $0 for these
                # endpoints. log_time_based_call computes cost from latency × $/hr.
                await ai_logger.log_time_based_call(
                    task="image_classification",
                    model=model_short,
                    latency_ms=latency_ms,
                    confidence_score=result.get('confidence', 0.5),
                    confidence_breakdown={
                        "model_confidence": result.get('confidence', 0.5),
                        "completeness": 1.0,
                        "consistency": 0.95,
                        "validation": 0.90,
                    },
                    action="use_ai_result",
                    job_id=job_id,
                )

                # Determine is_material using category mapping
                # The prompt returns classification categories like PRODUCT_IMAGE, DECORATIVE, etc.
                # We map these to is_material boolean using the module-level mapping
                classification_category = result.get('classification', '')
                if classification_category:
                    is_material = is_material_classification(classification_category)
                else:
                    # Fallback to direct is_material field if no classification category
                    is_material = result.get('is_material', False)

                return {
                    'is_material': is_material,
                    'confidence': result.get('confidence', 0.5),
                    'reason': result.get('reason', 'Unknown'),
                    'classification': classification_category,  # Include category for debugging
                    'model': model_short
                }

            except Exception as e:
                model_short = model.split('/')[-1] if '/' in model else model
                error_str = str(e)
                api_status = getattr(getattr(e, 'response', None), 'status_code', None)
                api_body = getattr(getattr(e, 'response', None), 'text', '')[:300] if hasattr(e, 'response') else ''
                logger.error(
                    f"❌ CLASSIFICATION FAILED for {image_path} | model={model_short} "
                    f"| {type(e).__name__}: {error_str[:200]}"
                    + (f" | HTTP {api_status}: {api_body}" if api_status else "")
                )

                # Signal the controller so the Qwen gate can shrink on repeated
                # failures. Only treat overload-class errors as backpressure
                # signals — semantic / bad-payload errors are filtered out.
                exc_name = type(e).__name__
                is_overload = (
                    "Timeout" in exc_name
                    or "Connection" in exc_name
                    or "RateLimit" in exc_name
                    or (api_status is not None and api_status in (429, 500, 502, 503, 504))
                )
                if is_overload:
                    try:
                        from app.services.core.endpoint_controller import endpoint_controller
                        endpoint_controller.record_failure("qwen")
                    except Exception:
                        pass  # Never let observability fail the primary path

                return {
                    'is_material': False,
                    'confidence': 0.0,
                    'reason': f'{model_short} failed: {str(e)}',
                    'model': f'{model_short}_failed',
                    'error': str(e)
                }
            finally:
                # ✅ NEW: Explicit cleanup to free memory
                if not base64_data and image_base64 is not None:
                    del image_base64

        async def validate_with_claude(image_path: str, base64_data: str = None) -> Dict[str, Any]:
            """Validate uncertain cases with Claude Sonnet."""
            image_base64 = base64_data
            detected_media_type = "image/jpeg"
            try:
                if not image_base64:
                    with open(image_path, 'rb') as f:
                        image_bytes = f.read()
                        detected_media_type = _detect_image_media_type(image_bytes, image_path)
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                        del image_bytes
                else:
                    # Detect from decoded base64 when data already provided
                    try:
                        sample = base64.b64decode(image_base64[:64])
                        detected_media_type = _detect_image_media_type(sample, image_path)
                    except Exception:
                        pass

                # Use database prompt - NO FALLBACK
                if self.classification_prompt:
                    classification_prompt = self.classification_prompt
                else:
                    error_msg = "CRITICAL: Classification prompt not found in database. Add via /admin/ai-configs with prompt_type='classification', stage='image_analysis', category='image_classification'"
                    logger.error(f"❌ {error_msg}")
                    raise ValueError(error_msg)

                from app.services.core.claude_helper import tracked_claude_call_async
                response = await tracked_claude_call_async(
                    task="image_classification_vision",
                    model="claude-sonnet-4-7",
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": detected_media_type, "data": image_base64}},
                            {"type": "text", "text": classification_prompt}
                        ]
                    }],
                )

                # Diagnose the "Expecting value: line 1 column 1 (char 0)" bug — that
                # error means Claude returned an empty string, a safety refusal, or a
                # non-text content block. Log what actually arrived so we can tell the
                # three cases apart instead of guessing.
                result_text = response.content[0].text if response.content else ""
                stop_reason = getattr(response, "stop_reason", "unknown")

                if not result_text or not result_text.strip():
                    logger.warning(
                        "⚠️ Claude returned empty content for %s | stop_reason=%s | "
                        "usage=%s | content_blocks=%d",
                        image_path, stop_reason,
                        getattr(response, "usage", None),
                        len(response.content or []),
                    )
                    return {
                        'is_material': False,
                        'confidence': 0.0,
                        'reason': f'Claude returned empty response (stop_reason={stop_reason})',
                        'model': 'claude_empty_response',
                    }

                try:
                    result = json.loads(result_text)
                except json.JSONDecodeError as parse_err:
                    logger.warning(
                        "⚠️ Claude returned non-JSON for %s | stop_reason=%s | "
                        "raw[:300]=%r | error=%s",
                        image_path, stop_reason, result_text[:300], parse_err,
                    )
                    return {
                        'is_material': False,
                        'confidence': 0.0,
                        'reason': f'Claude returned non-JSON: {parse_err}',
                        'model': 'claude_not_json',
                    }

                # Use the category mapping function for consistent classification
                is_material = is_material_classification(result.get('classification', ''))

                return {
                    'is_material': is_material,
                    'confidence': result.get('confidence', 0.9),
                    'reason': result.get('reason', 'Claude validation'),
                    'classification': result['classification'],
                    'model': 'claude'
                }

            except Exception as e:
                logger.warning(f"⚠️ Claude validation failed for {image_path}: {e}")
                return {
                    'is_material': False,
                    'confidence': 0.0,
                    'reason': f'Claude failed: {str(e)}',
                    'model': 'claude_failed'
                }
            finally:
                # ✅ NEW: Explicit cleanup to free memory
                if not base64_data and image_base64 is not None:
                    del image_base64

        # ✅ TIER-BASED RATE LIMITING: Dynamically adjust based on vision model tier
        from app.config.rate_limits import CLAUDE_CONCURRENCY, CURRENT_TIER
        from app.services.core.endpoint_controller import endpoint_controller

        logger.info(f"🎯 Rate Limiting Configuration:")
        logger.info(f"   Vision Tier: {CURRENT_TIER.tier} (${CURRENT_TIER.total_spend} spent)")
        logger.info(f"   LLM Rate Limit: {CURRENT_TIER.llm_rpm} RPM ({CURRENT_TIER.llm_rps:.1f} RPS)")
        endpoint_controller.log_stats("   Endpoint gates")

        # Qwen concurrency is gated through the unified EndpointController.
        # The controller owns the AIMD state across ALL calls (not just this
        # classify pass), so a bad Qwen day earlier in the pipeline is already
        # reflected when we get here. Claude fallback stays on a fixed semaphore
        # — Anthropic's rate limits are published and stable, not a
        # single-replica bottleneck.
        claude_semaphore = Semaphore(CLAUDE_CONCURRENCY)

        async def classify_with_two_stage(img_data):
            image_path = img_data.get('path')
            filename = img_data.get('filename', 'unknown')
            image_base64 = None

            # ✅ FIX 3: Detailed path validation with Supabase fallback
            if not image_path:
                logger.error(f"❌ Image path is None for {filename}")
                return None

            if not os.path.exists(image_path):
                # Try fallback: Download from Supabase if URL is available
                storage_url = img_data.get('storage_url') or img_data.get('url') or img_data.get('public_url')
                if storage_url:
                    logger.info(f"   📥 Image missing locally ({filename}), downloading from storage for classification...")
                    try:
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            resp = await client.get(storage_url)
                            if resp.status_code == 200:
                                image_base64 = base64.b64encode(resp.content).decode('utf-8')
                                logger.info(f"   ✅ Downloaded {len(resp.content)} bytes from storage")
                            else:
                                logger.error(f"   ❌ Storage download failed: HTTP {resp.status_code}")
                    except Exception as e:
                        logger.error(f"   ❌ Error downloading from storage: {e}")

                if not image_base64:
                    logger.error(f"❌ Image file does not exist locally and could not be downloaded: {image_path}")
                    logger.error(f"   Filename: {filename}")
                    return None

            # STAGE 1: Fast primary model classification with app-level retry.
            # The unified EndpointController gates in-flight Qwen requests and
            # learns the endpoint's real capacity across the whole pipeline.
            # Timeout/503 errors signal overload and shrink the cap; successes
            # grow it back. Cap also scales with HF replica count via the
            # auto-scaler cooperation path.
            async with endpoint_controller.qwen.slot():
                primary_result = None
                max_retries = 2  # Was 3. With max_retries=0 on the openai client
                                  # and adaptive concurrency gating the burst, 2 app-level
                                  # attempts is enough without wasting queue pressure.
                retry_delay = 1.0

                for attempt in range(max_retries):
                    primary_result = await classify_image_with_vision_model(image_path, primary_model, base64_data=image_base64)

                    # Signal the controller. Timeout/503/connection errors →
                    # shrink Qwen concurrency. Clean success → grow back toward max.
                    result_model = primary_result.get('model', '')
                    if '_503_fallback' in result_model or '_api_error' in result_model:
                        endpoint_controller.record_failure("qwen")
                    elif primary_result.get('confidence', 0) > 0 and '_invalid_response' not in result_model:
                        endpoint_controller.record_success("qwen")

                    # Only retry on endpoint-level API errors (not semantic failures).
                    should_retry = (
                        primary_result.get('retry_recommended', False) or
                        '_api_error' in result_model
                    )
                    if not should_retry or attempt == max_retries - 1:
                        break

                    wait_time = retry_delay * (2 ** attempt) + (0.1 * attempt)
                    qwen_gate = endpoint_controller.qwen
                    logger.warning(
                        f"   🔄 Retrying classification for {filename} "
                        f"(attempt {attempt + 2}/{max_retries}) after {wait_time:.1f}s... "
                        f"[qwen_limit={qwen_gate.limit}, in_flight={qwen_gate.in_flight}]"
                    )
                    await asyncio.sleep(wait_time)

            # ✅ NEW: Check if primary model failed (invalid response or exception)
            primary_failed = (
                '_invalid_response' in primary_result.get('model', '') or
                '_not_json' in primary_result.get('model', '') or
                '_empty_response' in primary_result.get('model', '') or
                '_api_error' in primary_result.get('model', '') or
                '_failed' in primary_result.get('model', '') or
                '_503_fallback' in primary_result.get('model', '')
            )

            # STAGE 2: If confidence is low OR primary failed, validate with secondary model
            if primary_result['confidence'] < confidence_threshold or primary_failed:
                if primary_failed:
                    logger.warning(f"   🔄 Primary model failed for {filename}, using fallback: {validation_model}")
                else:
                    logger.debug(f"   🔍 Low confidence ({primary_result['confidence']:.2f}) - validating with {validation_model}: {filename}")

                async with claude_semaphore:
                    # Use Claude or Qwen-32B for validation with retry
                    validation_result = None
                    for attempt in range(max_retries):
                        if 'claude' in validation_model.lower():
                            validation_result = await validate_with_claude(image_path, base64_data=image_base64)
                        else:
                            validation_result = await classify_image_with_vision_model(image_path, validation_model, base64_data=image_base64)

                        # Check if we should retry
                        should_retry = (
                            validation_result.get('retry_recommended', False) or
                            '_api_error' in validation_result.get('model', '')
                        )

                        if not should_retry or attempt == max_retries - 1:
                            break

                        wait_time = retry_delay * (2 ** attempt) + (0.1 * attempt)
                        logger.warning(f"   🔄 Retrying validation for {filename} (attempt {attempt + 2}/{max_retries}) after {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)

                    # ✅ NEW: If validation also failed, use Claude as final fallback
                    validation_failed = (
                        '_invalid_response' in validation_result.get('model', '') or
                        '_not_json' in validation_result.get('model', '') or
                        '_empty_response' in validation_result.get('model', '') or
                        '_api_error' in validation_result.get('model', '')
                    )

                    if validation_failed and 'claude' not in validation_model.lower():
                        logger.warning(f"   🔄 Validation model also failed for {filename}, using Claude as final fallback")
                        validation_result = await validate_with_claude(image_path, base64_data=image_base64)

                img_data['ai_classification'] = validation_result
            else:
                img_data['ai_classification'] = primary_result

            # ✅ NEW: Explicitly clear the base64 string to free memory
            if image_base64:
                del image_base64

            return img_data

        # ✅ NEW: Process images in batches to prevent OOM
        material_images = []
        non_material_images = []
        total_images = len(extracted_images)

        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            batch_images = extracted_images[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (total_images + batch_size - 1) // batch_size

            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch_images)} images)")

            # Classify batch
            batch_start_time = time.time()
            classification_tasks = [classify_with_two_stage(img_data) for img_data in batch_images]
            classified_batch = await asyncio.gather(*classification_tasks, return_exceptions=True)
            batch_duration = time.time() - batch_start_time

            # ✅ FIX 4: Detect suspiciously fast classification
            expected_min_time = len(batch_images) * 0.5  # At least 0.5 seconds per image
            if batch_duration < expected_min_time:
                logger.warning(f"⚠️ SUSPICIOUS: Batch {batch_num} completed in {batch_duration:.2f}s for {len(batch_images)} images")
                logger.warning(f"   Expected minimum: {expected_min_time:.2f}s ({len(batch_images)} images × 0.5s)")
                logger.warning(f"   This may indicate API failures or skipped classifications")

            # Filter batch results
            failed_count = 0
            for img_data in classified_batch:
                if img_data is None:
                    failed_count += 1
                    logger.warning(f"   ⚠️ Skipping image: returned None (path validation failed)")
                    continue

                if isinstance(img_data, Exception):
                    failed_count += 1
                    logger.error(f"   ❌ Classification exception: {type(img_data).__name__}: {str(img_data)}")
                    continue

                classification = img_data.get('ai_classification', {})

                # ✅ FIX 5: Log classification details for debugging
                if not classification:
                    logger.warning(f"   ⚠️ No classification data for {img_data.get('filename')}")
                    failed_count += 1
                    continue

                # ✅ FIX: Handle classification failures more gracefully
                # If classification failed or returned empty, assume it's material (safer approach)
                # This ensures we don't miss material images due to API issues
                if 'error' in classification or '_failed' in classification.get('model', '') or '_empty_response' in classification.get('model', ''):
                    logger.warning(f"   ⚠️ Classification uncertain for {img_data.get('filename')}: {classification.get('reason')}")
                    logger.warning(f"   → Treating as MATERIAL (safe default) to ensure CLIP embeddings are generated")
                    # Override classification to treat as material with low confidence
                    img_data['ai_classification'] = {
                        'is_material': True,
                        'confidence': 0.3,  # Low confidence to indicate uncertainty
                        'reason': f"Fallback: {classification.get('reason', 'Classification failed')}",
                        'model': 'fallback_material',
                        'original_error': classification.get('reason', 'Unknown')
                    }
                    material_images.append(img_data)
                    failed_count += 1
                    continue

                if classification.get('is_material', False):
                    material_images.append(img_data)
                    logger.debug(f"   ✅ Material: {img_data.get('filename')} - confidence: {classification.get('confidence', 0):.2f}")
                else:
                    non_material_images.append(img_data)
                    logger.debug(f"   🚫 Filtered out: {img_data.get('filename')} - {classification.get('reason')} (confidence: {classification.get('confidence', 0):.2f})")

            # ✅ NEW: Explicit garbage collection after each batch
            del classification_tasks
            del classified_batch
            gc.collect()

            logger.info(f"   ✅ Batch {batch_num}/{total_batches} complete: {len(material_images)} material, {len(non_material_images)} filtered, {failed_count} failed")

        classification_duration = time.time() - classification_start_time

        logger.info(f"✅ AI classification complete:")
        logger.info(f"   Total time: {classification_duration:.2f}s")
        logger.info(f"   Material images: {len(material_images)}")
        logger.info(f"   Non-material images filtered out: {len(non_material_images)}")

        total_classified = len(material_images) + len(non_material_images)
        total_input = len(extracted_images)

        if total_classified > 0:
            logger.info(f"   Classification accuracy: {len(material_images) / total_classified * 100:.1f}% kept")

        # ✅ FIX 6: Critical validation - detect complete classification failure
        if total_classified == 0 and total_input > 0:
            logger.error("❌ CRITICAL FAILURE: ALL images failed classification!")
            logger.error(f"   Input images: {total_input}")
            logger.error(f"   Successfully classified: 0")
            logger.error(f"   This indicates a systemic issue with the AI classification service")
            logger.error(f"   Possible causes:")
            logger.error(f"   1. HuggingFace Endpoint token invalid or expired")
            logger.error(f"   2. HuggingFace Endpoint service unavailable")
            logger.error(f"   3. Image files deleted before classification")
            logger.error(f"   4. Network connectivity issues")
            logger.error(f"   5. Model name incorrect or model unavailable")

            # Log first few image paths for debugging
            logger.error(f"   Sample image paths:")
            for i, img in enumerate(extracted_images[:3]):
                logger.error(f"     {i+1}. {img.get('path')} (exists: {os.path.exists(img.get('path', ''))})")

            raise Exception(
                f"Image classification completely failed: 0/{total_input} images classified. "
                f"Check HuggingFace Endpoint token and service availability."
            )

        # ✅ FIX 7: Warning if classification rate is suspiciously low
        if total_classified < total_input * 0.5:
            logger.warning(f"⚠️ WARNING: Low classification success rate")
            logger.warning(f"   Successfully classified: {total_classified}/{total_input} ({total_classified/total_input*100:.1f}%)")
            logger.warning(f"   Failed: {total_input - total_classified}")

        # NOTE: Removed between-batch pause_if_idle - endpoints pause only at full job completion
        # This prevents expensive re-warmup cycles when processing multiple products

        return material_images, non_material_images

    async def upload_images_to_storage(
        self,
        material_images: List[Dict[str, Any]],
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Upload material images to Supabase Storage.

        Args:
            material_images: List of material image data
            document_id: Document ID for storage path

        Returns:
            List of uploaded images with storage URLs
        """
        logger.info(f"📤 Uploading {len(material_images)} material images to Supabase Storage...")

        upload_semaphore = Semaphore(10)  # 10 concurrent uploads

        async def upload_single_image(img_data):
            """Upload a single material image to Supabase Storage"""
            try:
                image_path = img_data.get('path')
                if not image_path or not os.path.exists(image_path):
                    logger.warning(f"Image file not found: {image_path}")
                    return None

                # Upload to Supabase Storage
                upload_result = await self.pdf_processor._upload_image_to_storage(
                    image_path,
                    document_id,
                    {
                        'filename': img_data.get('filename'),
                        'size_bytes': img_data.get('size_bytes'),
                        'format': img_data.get('format'),
                        'dimensions': img_data.get('dimensions'),
                        'width': img_data.get('width'),
                        'height': img_data.get('height')
                    },
                    None  # No enhanced path
                )

                if upload_result and upload_result.get('success'):
                    public_url = upload_result.get('public_url')
                    storage_path = upload_result.get('storage_path')

                    # Set storage metadata
                    img_data['storage_url'] = public_url
                    img_data['storage_path'] = storage_path
                    img_data['storage_uploaded'] = True  # ✅ FIX: Set storage_uploaded flag
                    img_data['storage_bucket'] = upload_result.get('bucket', 'pdf-tiles')

                    # Debug logging
                    logger.debug(f"✅ Upload successful for {img_data.get('filename')}")
                    logger.debug(f"   storage_url: {public_url[:100] if public_url else 'None'}")
                    logger.debug(f"   storage_path: {storage_path}")
                    logger.debug(f"   img_data keys after upload: {list(img_data.keys())}")

                    return img_data
                else:
                    logger.warning(f"Failed to upload image: {img_data.get('filename')} - Error: {upload_result.get('error') if upload_result else 'No result'}")
                    return None

            except Exception as e:
                logger.error(f"Error uploading image {img_data.get('filename')}: {e}")
                return None

        async def upload_with_semaphore(img_data):
            async with upload_semaphore:
                return await upload_single_image(img_data)

        upload_tasks = [upload_with_semaphore(img_data) for img_data in material_images]
        uploaded_images = await asyncio.gather(*upload_tasks, return_exceptions=True)

        # Filter out failed uploads
        successful_uploads = [img for img in uploaded_images if img is not None and not isinstance(img, Exception)]

        logger.info(f"✅ Upload complete: {len(successful_uploads)} material images uploaded to storage")

        return successful_uploads

    async def _get_embedding_checkpoint(self, document_id: str) -> Optional[int]:
        """
        Get the last successfully processed image index for embedding generation.

        Args:
            document_id: Document ID

        Returns:
            Last processed index or None if no checkpoint exists
        """
        try:
            # Use the canonical has_slig_embedding flag (maintained by vecs_service)
            # to count how many images already have visual embeddings.
            result = self.supabase_client.client.table('document_images')\
                .select('id')\
                .eq('document_id', document_id)\
                .eq('has_slig_embedding', True)\
                .execute()

            if result.data:
                return len(result.data)
            return 0
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to get embedding checkpoint: {e}")
            return 0

    # ------------------------------------------------------------------ #
    # Icon-candidate detection (Stage 3 split)                            #
    # ------------------------------------------------------------------ #
    #
    # After classification, some images that look like product specs
    # (R-rating badges, PEI icons, slip-resistance symbols, packaging
    # icons, etc.) get routed to the icon extraction pipeline INSTEAD of
    # the regular image embedding pipeline. They get OCR + Claude → spec
    # metadata, NOT visual SLIG / specialized SLIG / understanding vectors.
    # This keeps the visual VECS collections clean of icon junk while
    # capturing the structured spec data into product.metadata.
    #
    # Detection rules (all must hold):
    #   1. width  < ICON_MAX_DIM and height < ICON_MAX_DIM
    #   2. ICON_MIN_ASPECT <= width/height <= ICON_MAX_ASPECT
    #   3. ≥ ICON_MIN_PER_PAGE such images on the same page_number
    #
    # The 3rd rule is the strictest — a single small image on a page is
    # almost certainly a logo or thumbnail, but a row of N small images
    # together is the spec icon strip. Catalogs with ceramic specs
    # typically have 5-8 icons in a single row at the bottom of the page.
    #
    # Two sources feed the icon candidate pool:
    #   (a) `material_images`     — Qwen classified them as PRODUCT_IMAGE/MIXED
    #                               but they're actually small spec icons
    #   (b) `non_material_images` — Qwen classified them as DECORATIVE
    #                               (logos, headers, etc.); the DECORATIVE
    #                               override re-routes them to icon extraction
    #                               IF they meet the size + grid rules
    #
    # Anything that fails all 3 rules stays in its original bucket.
    ICON_MAX_DIM = 200          # px — both width and height must be below this
    ICON_MIN_ASPECT = 0.5       # width/height ≥ 0.5 (not super-tall)
    ICON_MAX_ASPECT = 2.0       # width/height ≤ 2.0 (not super-wide)
    ICON_MIN_PER_PAGE = 3       # at least N icon-shaped images on the same page

    @classmethod
    def _is_icon_shaped(cls, img_data: Dict[str, Any]) -> bool:
        """Check if a single image meets the size + aspect ratio rules.

        Page-grouping (rule 3) is checked separately by the caller because it
        needs all candidates from the page to compare against.
        """
        width = img_data.get('width') or 0
        height = img_data.get('height') or 0
        if width <= 0 or height <= 0:
            return False
        if width >= cls.ICON_MAX_DIM or height >= cls.ICON_MAX_DIM:
            return False
        aspect = width / height
        return cls.ICON_MIN_ASPECT <= aspect <= cls.ICON_MAX_ASPECT

    @staticmethod
    def _classification_is_decorative(img_data: Dict[str, Any]) -> bool:
        """True if Qwen classified this image as DECORATIVE.

        Used by the DECORATIVE override path: a small grid of decorative-
        classified images on the same page is more likely a strip of spec
        icons than actual decoration, so we re-route them to icon extraction.
        """
        classification = img_data.get('ai_classification') or {}
        category = (classification.get('classification') or '').upper().strip()
        return category == 'DECORATIVE'

    def _split_material_and_icon_candidates(
        self,
        material_images: List[Dict[str, Any]],
        non_material_images: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split classified images into 3 routing buckets:
          - regular_material_images : full Stage 3 pipeline (visual SLIG +
                                      4 specialized + understanding embedding)
          - icon_candidates         : icon extraction pipeline only
                                      (OCR + Claude → spec metadata, NO embeddings)
          - remaining_non_material  : pure decoration / technical diagrams
                                      (logos, headers, page borders) — dropped
                                      from further processing as before

        Detection: a candidate is icon-shaped if (width, height < ICON_MAX_DIM)
        AND aspect ratio in [ICON_MIN_ASPECT, ICON_MAX_ASPECT]. The page-grouping
        rule (≥ ICON_MIN_PER_PAGE per page) is enforced as the second pass.

        Args:
            material_images: Images Qwen classified as PRODUCT_IMAGE / MIXED
            non_material_images: Optional — images Qwen classified as
                DECORATIVE / TECHNICAL_DIAGRAM. When provided, decorative
                images that meet the icon shape + grid rules are re-routed
                to the icon path (the "DECORATIVE override").

        Returns:
            (regular_material_images, icon_candidates, remaining_non_material)
        """
        non_material_images = non_material_images or []

        # Pass 1: tag every image with whether it's icon-shaped
        material_shaped = [(img, self._is_icon_shaped(img)) for img in material_images]
        decorative_shaped = [
            (img, self._is_icon_shaped(img) and self._classification_is_decorative(img))
            for img in non_material_images
        ]

        # Pass 2: group by page and check the per-page count gate
        # (combine both pools for the count, since a spec strip can span both
        # classifier verdicts — Qwen often calls some icons PRODUCT_IMAGE and
        # adjacent ones DECORATIVE).
        from collections import defaultdict
        page_counts: Dict[int, int] = defaultdict(int)
        for img, is_icon in material_shaped + decorative_shaped:
            if is_icon:
                page = img.get('page_number')
                if page is not None:
                    page_counts[page] += 1

        pages_with_icon_grid = {
            page for page, count in page_counts.items() if count >= self.ICON_MIN_PER_PAGE
        }

        # Pass 3: assign each image to a final bucket
        regular_material: List[Dict[str, Any]] = []
        icon_candidates: List[Dict[str, Any]] = []
        remaining_non_material: List[Dict[str, Any]] = []

        for img, is_shaped in material_shaped:
            if is_shaped and img.get('page_number') in pages_with_icon_grid:
                icon_candidates.append(img)
            else:
                regular_material.append(img)

        for img, is_shaped_and_decorative in decorative_shaped:
            if is_shaped_and_decorative and img.get('page_number') in pages_with_icon_grid:
                icon_candidates.append(img)
            else:
                remaining_non_material.append(img)

        if icon_candidates:
            logger.info(
                f"   🔖 Icon split: {len(regular_material)} regular material, "
                f"{len(icon_candidates)} icon candidates "
                f"(from {len(material_images)} material + {len(non_material_images)} non-material), "
                f"{len(remaining_non_material)} remain as non-material"
            )
        else:
            logger.debug(
                f"   🔖 Icon split: no icon candidates detected "
                f"({len(regular_material)} regular material, "
                f"{len(remaining_non_material)} non-material)"
            )

        return regular_material, icon_candidates, remaining_non_material

    # ------------------------------------------------------------------ #
    # Material analysis (vision_analysis JSON) — feeds understanding emb #
    # ------------------------------------------------------------------ #
    #
    # The flow is:
    #   1. Try Qwen3-VL via the warmed-up endpoint, with N retries on
    #      transient (5xx / connection) failures.
    #   2. If Qwen returns nothing parseable after retries, fall back to
    #      Claude Sonnet 4.7 which is highly reliable for strict JSON.
    #   3. Parse the response into a dict, validate against the expected
    #      schema, and return None if both providers failed.
    #
    # Result is stored on `document_images.vision_analysis` (JSONB) and
    # passed to `RealEmbeddingsService.generate_all_embeddings(vision_analysis=)`
    # so the Voyage understanding (1024D) embedding gets generated and saved
    # to `vecs.image_understanding_embeddings`.

    # Top-level fields the Material Image Analyzer prompt is expected to
    # return. We treat the analysis as valid if AT LEAST `_MIN_REQUIRED_FIELDS`
    # of these are present and non-null. This is intentionally tolerant —
    # the goal is to catch outright garbage, not to enforce perfect schema
    # compliance from a vision model.
    _EXPECTED_VISION_ANALYSIS_FIELDS = (
        'material_type',
        'material_subtype',
        'color_palette',
        'primary_color_hex',
        'texture',
        'pattern',
        'finish',
        'design_style',
        'applications',
        'physical_properties',
        'quality_assessment',
        'confidence',
    )
    _MIN_REQUIRED_VISION_FIELDS = 4  # at least this many non-null keys

    @staticmethod
    def _parse_vision_analysis_json(raw: str, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Parse a vision-analysis raw response into a dict.

        Tolerates:
          - Plain JSON
          - ```json ... ``` markdown fences
          - Extra prose around a single JSON object (extracts first {...})

        Returns None if no parseable JSON object can be recovered.
        """
        import json as _json
        import re as _re

        if not raw:
            return None

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = _re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = _re.sub(r'\s*```$', '', cleaned)

        try:
            return _json.loads(cleaned)
        except _json.JSONDecodeError:
            pass

        # Fallback: extract the first {...} block, tolerating leading prose.
        match = _re.search(r'\{[\s\S]*\}', cleaned)
        if not match:
            logger.warning(
                f"   ⚠️ Vision analysis response for {image_id} not parseable as JSON; "
                f"first 200 chars: {raw[:200]!r}"
            )
            return None
        try:
            return _json.loads(match.group(0))
        except _json.JSONDecodeError as parse_err:
            logger.warning(
                f"   ⚠️ Vision analysis JSON parse failed for {image_id}: {parse_err}"
            )
            return None

    @classmethod
    def _validate_vision_analysis(
        cls, vision_analysis: Any, image_id: str, source: str
    ) -> Optional[Dict[str, Any]]:
        """
        Validate that a parsed vision_analysis dict looks usable.

        Returns the dict on success, None on validation failure. Logs the
        reason loudly so partial / malformed responses are visible to ops.
        """
        if not isinstance(vision_analysis, dict) or not vision_analysis:
            logger.warning(
                f"   ⚠️ {source}: vision_analysis for {image_id} is not a non-empty dict — "
                f"got {type(vision_analysis).__name__}"
            )
            return None

        present_fields = [
            f for f in cls._EXPECTED_VISION_ANALYSIS_FIELDS
            if vision_analysis.get(f) not in (None, "", [], {})
        ]
        if len(present_fields) < cls._MIN_REQUIRED_VISION_FIELDS:
            logger.warning(
                f"   ⚠️ {source}: vision_analysis for {image_id} has only "
                f"{len(present_fields)}/{cls._MIN_REQUIRED_VISION_FIELDS} required "
                f"fields populated (got: {present_fields}). Marking invalid."
            )
            return None

        logger.info(
            f"   🔬 {source}: vision_analysis valid for {image_id} "
            f"({len(present_fields)}/{len(cls._EXPECTED_VISION_ANALYSIS_FIELDS)} fields populated)"
        )
        return vision_analysis

    async def _try_qwen_material_analysis(
        self,
        image_base64: str,
        image_id: str,
        max_retries: int = 2,
    ) -> Optional[Dict[str, Any]]:
        """
        Primary path: call Qwen3-VL with the Material Image Analyzer prompt.

        Retries up to `max_retries` times on transient errors. Returns a
        validated dict, or None if Qwen could not produce a usable response.
        """
        try:
            from app.services.embeddings.endpoint_registry import endpoint_registry
            from app.services.embeddings.qwen_endpoint_manager import QwenEndpointManager
            from app.config import get_settings
            from openai import AsyncOpenAI

            settings = get_settings()
            qwen_config = settings.get_qwen_config()
            huggingface_api_key = qwen_config["endpoint_token"]

            if not huggingface_api_key or not qwen_config.get("enabled"):
                logger.warning(
                    f"   ⚠️ Qwen endpoint not configured/enabled — skipping Qwen path "
                    f"for material analysis of {image_id}"
                )
                return None

            try:
                qwen_manager = endpoint_registry.get_qwen_manager()
            except Exception:
                qwen_manager = None
            if qwen_manager is None:
                qwen_manager = QwenEndpointManager(
                    endpoint_url="",  # resolved dynamically by manager from HF
                    endpoint_name=qwen_config["endpoint_name"],
                    namespace=qwen_config["namespace"],
                    endpoint_token=huggingface_api_key,
                    enabled=qwen_config["enabled"],
                )

            if not await asyncio.to_thread(qwen_manager.resume_if_needed):
                logger.warning(
                    f"   ⚠️ Failed to resume Qwen endpoint for material analysis of {image_id}"
                )
                return None

            # ⚠️ Pull the LIVE endpoint URL from the manager, not from settings.
            # The classification path already handles this correctly (line ~263).
            # Settings["endpoint_url"] is typically unset in prod and was causing
            # APIConnectionError ("Connection error.") on every material analysis
            # call, which fell through to the Claude Sonnet fallback for EVERY
            # image — defeating the whole purpose of having Qwen. Root cause of
            # the "vision_provider = claude_fallback for every image" regression.
            qwen_endpoint_url = qwen_manager.endpoint_url
            if not qwen_endpoint_url or not qwen_endpoint_url.startswith(("http://", "https://")):
                logger.error(
                    f"   ❌ CRITICAL: qwen_manager.endpoint_url is invalid "
                    f"({qwen_endpoint_url!r}) for material analysis of {image_id}. "
                    f"Manager warmup probably did not run before this call."
                )
                return None

            base_url = qwen_endpoint_url.rstrip('/')
            if not base_url.endswith('/v1'):
                base_url = base_url + '/v1'
            base_url = base_url + '/'
            logger.info(
                f"   🔧 Qwen material analysis base_url: {base_url} (for {image_id})"
            )

            client = AsyncOpenAI(
                base_url=base_url,
                api_key=huggingface_api_key,
                timeout=120.0,
            )

            # 2026-04-11: fix MIVAA-NG for Qwen path — sniff real media type
            # from decoded base64 so PNG icons don't get sent as image/jpeg.
            # Qwen's OpenAI-compat endpoint is more forgiving than Claude but
            # has still returned 400s on mismatched data URLs in the past.
            qwen_detected_media_type = "image/jpeg"
            try:
                import base64 as _b64
                sample = _b64.b64decode(image_base64[:64], validate=False)[:16]
                qwen_detected_media_type = _detect_image_media_type(sample)
            except Exception:
                pass

            messages = []
            if self.material_analyzer_system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.material_analyzer_system_prompt,
                })
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{qwen_detected_media_type};base64,{image_base64}"},
                    },
                    {"type": "text", "text": self.material_analyzer_prompt},
                ],
            })

            for attempt in range(max_retries + 1):
                try:
                    response = await client.chat.completions.create(
                        model="Qwen/Qwen3-VL-32B-Instruct",
                        messages=messages,
                        max_tokens=1024,
                        temperature=0.1,
                        stream=False,
                    )
                    raw = response.choices[0].message.content if response.choices else None
                    if not raw:
                        logger.warning(
                            f"   ⚠️ Qwen returned empty material analysis for {image_id} "
                            f"(attempt {attempt + 1}/{max_retries + 1})"
                        )
                        if attempt < max_retries:
                            await asyncio.sleep(1.5 ** attempt)
                            continue
                        return None

                    parsed = self._parse_vision_analysis_json(raw, image_id)
                    validated = self._validate_vision_analysis(parsed, image_id, source=VisionProvider.QWEN.value)
                    if validated is not None:
                        return validated

                    # Parsed but failed validation — retry once in case of
                    # a one-off bad sample.
                    if attempt < max_retries:
                        logger.info(
                            f"   🔁 Retrying Qwen material analysis for {image_id} "
                            f"(attempt {attempt + 2}/{max_retries + 1}) — "
                            f"previous result failed validation"
                        )
                        await asyncio.sleep(1.0)
                        continue
                    return None

                except Exception as call_err:
                    err_str = str(call_err)
                    is_transient = (
                        '503' in err_str
                        or '502' in err_str
                        or '504' in err_str
                        or 'timeout' in err_str.lower()
                        or 'Service Unavailable' in err_str
                        or 'Connection' in err_str
                    )
                    if is_transient and attempt < max_retries:
                        backoff = min(1.5 ** attempt + 0.5, 6.0)
                        logger.warning(
                            f"   ⏳ Qwen material analysis transient error for {image_id} "
                            f"(attempt {attempt + 1}/{max_retries + 1}): {err_str[:150]} — "
                            f"retrying in {backoff:.1f}s"
                        )
                        await asyncio.sleep(backoff)
                        continue
                    logger.warning(
                        f"   ⚠️ Qwen material analysis failed for {image_id} "
                        f"(attempt {attempt + 1}/{max_retries + 1}): {err_str[:200]}"
                    )
                    return None

            return None

        except Exception as e:
            logger.warning(
                f"   ⚠️ Qwen material analysis path failed for {image_id}: {e}",
                exc_info=True,
            )
            return None

    async def _try_claude_material_analysis(
        self,
        image_base64: str,
        image_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Fallback path: call Claude Sonnet 4.7 with the same prompt when
        Qwen has failed. Claude is more reliable for strict JSON output
        and acts as a safety net so we don't lose the understanding
        embedding for an image just because Qwen had a bad day.
        """
        try:
            from app.services.core.ai_client_service import get_ai_client_service

            if not self.material_analyzer_prompt:
                return None

            ai_service = get_ai_client_service()
            if not getattr(ai_service, 'anthropic_async', None):
                logger.warning(
                    f"   ⚠️ Anthropic client not available — cannot run Claude fallback "
                    f"for material analysis of {image_id}"
                )
                return None

            # 2026-04-11: fix MIVAA-NG — detect the real media type from the
            # decoded base64 magic bytes instead of hardcoding "image/jpeg".
            # Claude's API rejects with
            # "image was specified using image/jpeg but appears to be image/png"
            # when the mime type lies, and ~60% of our icon + YOLO-region
            # crops are PNG. Sniff first 16 decoded bytes — cheap, no new
            # deps. Falls back to image/jpeg on decode error.
            detected_media_type = "image/jpeg"
            try:
                import base64 as _b64
                sample = _b64.b64decode(image_base64[:64], validate=False)[:16]
                detected_media_type = _detect_image_media_type(sample)
            except Exception:
                pass

            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": detected_media_type,
                        "data": image_base64,
                    },
                },
                {"type": "text", "text": self.material_analyzer_prompt},
            ]

            kwargs = {
                "model": "claude-sonnet-4-7",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": content}],
            }
            if self.material_analyzer_system_prompt:
                kwargs["system"] = self.material_analyzer_system_prompt

            # Wrap into helper signature; helper rebuilds the api kwargs internally
            from app.services.core.claude_helper import tracked_claude_call_async
            response = await tracked_claude_call_async(
                task="image_material_analysis_fallback",
                model=kwargs["model"],
                max_tokens=kwargs["max_tokens"],
                messages=kwargs["messages"],
                system=kwargs.get("system"),
            )

            # Anthropic SDK returns a list of content blocks; concatenate
            # any text blocks.
            text_parts: List[str] = []
            for block in response.content:
                block_type = getattr(block, 'type', None)
                if block_type == 'text':
                    text_parts.append(getattr(block, 'text', '') or '')
            raw = ''.join(text_parts).strip()
            if not raw:
                logger.warning(
                    f"   ⚠️ Claude fallback returned empty material analysis for {image_id}"
                )
                return None

            parsed = self._parse_vision_analysis_json(raw, image_id)
            return self._validate_vision_analysis(parsed, image_id, source=VisionProvider.CLAUDE_FALLBACK.value)

        except Exception as e:
            logger.warning(
                f"   ⚠️ Claude fallback for material analysis failed for {image_id}: {e}",
                exc_info=True,
            )
            return None

    async def _analyze_material_image(
        self,
        image_base64: str,
        image_id: str,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Run rich material analysis on a confirmed-material image and return
        a structured `vision_analysis` JSON.

        This is the input the Voyage understanding embedding consumes — it
        captures material type, color, texture, finish, applications, etc.
        as structured properties, which then become a 1024D embedding in
        `vecs.image_understanding_embeddings`.

        Strategy:
          1. Try Qwen3-VL (warm endpoint, with retries) — primary path.
          2. On failure, fall back to Claude Sonnet 4.7 — secondary path.
          3. If both fail, return (None, FAILED) so the caller can record
             the failure in job-level stats and the image gets all other
             vectors except `understanding_1024`.

        Returns:
            (vision_analysis_dict, source) where source is the string value
            of a `VisionProvider` enum member: QWEN | CLAUDE_FALLBACK | SKIPPED | FAILED.
            Only QWEN and CLAUDE_FALLBACK are persistable (DB CHECK constraint);
            SKIPPED and FAILED are in-memory only — the caller never writes a
            row when one of those is returned.
        """
        if not self.material_analyzer_prompt:
            logger.debug(
                f"   ⏭️ Skipping material analysis for {image_id}: "
                f"Material Image Analyzer prompt not loaded"
            )
            return None, VisionProvider.SKIPPED.value

        # Primary: Qwen
        qwen_result = await self._try_qwen_material_analysis(
            image_base64=image_base64,
            image_id=image_id,
            max_retries=2,
        )
        if qwen_result is not None:
            return qwen_result, VisionProvider.QWEN.value

        # Fallback: Claude
        logger.info(
            f"   🩹 Falling back to Claude for material analysis of {image_id} "
            f"(Qwen returned no usable result)"
        )
        claude_result = await self._try_claude_material_analysis(
            image_base64=image_base64,
            image_id=image_id,
        )
        if claude_result is not None:
            return claude_result, VisionProvider.CLAUDE_FALLBACK.value

        logger.error(
            f"   ❌ Material analysis failed for {image_id} via BOTH Qwen and Claude — "
            f"image will be missing the understanding embedding"
        )
        return None, VisionProvider.FAILED.value

    async def _process_single_image_with_retry(
        self,
        img_data: Dict[str, Any],
        document_id: str,
        workspace_id: str,
        idx: int,
        total: int,
        max_retries: int = 3,
        material_category: Optional[str] = None,
    ) -> Tuple[bool, bool, Optional[str], Dict[str, Any]]:
        """
        Process a single image with retry logic.

        Args:
            img_data: Image data
            document_id: Document ID
            workspace_id: Workspace ID
            idx: Image index
            total: Total images
            max_retries: Maximum retry attempts
            material_category: Material category from upload (tiles, heatpump, wood, etc.)

        Returns:
            Tuple of (image_saved, embedding_generated, error_message)
        """
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                # 🔍 BBOX TRACE: Log bbox right before save_single_image
                bbox_value = img_data.get('bbox')
                bbox_len = len(bbox_value) if isinstance(bbox_value, (list, tuple)) else 'N/A'
                logger.info(
                    f"   🔍 [BBOX TRACE] Before save_single_image - {img_data.get('filename')}: "
                    f"bbox_len={bbox_len}, bbox={bbox_value[:5] if isinstance(bbox_value, (list, tuple)) and len(bbox_value) >= 5 else bbox_value}, "
                    f"id(img_data)={id(img_data)}"
                )
                # Check if bbox looks like an embedding (768 elements)
                if isinstance(bbox_value, (list, tuple)) and len(bbox_value) > 10:
                    logger.error(
                        f"   ❌ [BBOX TRACE] CORRUPTION DETECTED! bbox has {len(bbox_value)} elements "
                        f"(expected 4). First 5: {bbox_value[:5]}"
                    )
                    # Log all keys in img_data to find where corruption came from
                    logger.error(f"   ❌ [BBOX TRACE] img_data keys: {list(img_data.keys())}")

                # Save to database with material_category for proper categorization
                # (ai_classification is already in img_data from classify_images)
                image_id = await self.supabase_client.save_single_image(
                    image_info=img_data,
                    document_id=document_id,
                    workspace_id=workspace_id,
                    image_index=idx,
                    category='product',  # Fallback category if material_category not provided
                    extraction_method=img_data.get('extraction_method', 'pymupdf'),
                    bbox=img_data.get('bbox'),
                    detection_confidence=img_data.get('detection_confidence'),
                    product_name=img_data.get('product_name'),
                    material_category=material_category  # Pass material_category for proper categorization
                )

                if not image_id:
                    last_error = "Failed to save image to database"
                    retry_count += 1
                    if retry_count < max_retries:
                        await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    continue

                img_data['id'] = image_id
                logger.info(f"   ✅ Saved image {idx + 1}/{total} to DB: {image_id}")

                # Generate CLIP embeddings
                image_path = img_data.get('path')
                if not image_path or not os.path.exists(image_path):
                    logger.warning(f"   ⚠️ Image file not found for CLIP generation: {image_path}")
                    return (True, False, "Image file not found", {
                        'image_id': image_id,
                        'visual_slig': False,
                        'color_slig': False,
                        'texture_slig': False,
                        'style_slig': False,
                        'material_slig': False,
                        'understanding': False,
                        'vision_analysis_source': VisionProvider.SKIPPED.value,
                    })

                logger.info(f"   🎨 Generating CLIP embeddings for image {idx + 1}/{total}")

                # Read image once and prepare two encodings:
                #   - `image_base64_raw`  : raw base64 (what Qwen / Material Image Analyzer expects)
                #   - `image_base64_data` : `data:image/jpeg;base64,...` data URL (what the
                #                            embeddings service / SLIG client expect)
                with open(image_path, 'rb') as img_file:
                    image_bytes = img_file.read()
                    image_base64_raw = base64.b64encode(image_bytes).decode('utf-8')
                    image_base64_data = f"data:image/jpeg;base64,{image_base64_raw}"

                # Run rich material analysis (Qwen3-VL primary, Claude Sonnet 4.7
                # fallback) to produce the structured `vision_analysis` JSON. This
                # is the input the Voyage understanding embedding consumes — without
                # it, the 1024D understanding branch is skipped and we lose the 7th
                # vector. The (vision_analysis, source) tuple lets us track which
                # provider produced the result for job-level stats.
                vision_analysis, vision_analysis_source = await self._analyze_material_image(
                    image_base64=image_base64_raw,
                    image_id=image_id,
                )

                # Persist vision_analysis (and provenance) to document_images so the
                # rest of the platform can read it (admin UI, search filters,
                # downstream services). Best-effort — we never want a JSONB write
                # failure to block embedding generation.
                #
                # Defensive gate: only persist if BOTH vision_analysis is present
                # AND the source value is in the persistable set (`qwen` or
                # `claude_fallback`). The DB CHECK constraint enforces the same
                # rule, so writing a non-persistable value would fail anyway.
                try:
                    persistable_source = VisionProvider(vision_analysis_source).is_persistable()
                except ValueError:
                    persistable_source = False
                if vision_analysis and persistable_source:
                    try:
                        self.supabase_client.client.table('document_images').update({
                            'vision_analysis': vision_analysis,
                            'vision_provider': vision_analysis_source,
                        }).eq('id', image_id).execute()
                    except Exception as va_persist_err:
                        logger.warning(
                            f"   ⚠️ Failed to persist vision_analysis on document_images "
                            f"for {image_id}: {va_persist_err}"
                        )

                # Generate all embeddings — passing vision_analysis enables the
                # Voyage understanding (1024D) branch in real_embeddings_service.
                embedding_result = await self.embedding_service.generate_all_embeddings(
                    entity_id=image_id,
                    entity_type="image",
                    text_content="",
                    image_data=image_base64_data,
                    material_properties={},
                    vision_analysis=vision_analysis,
                )

                if not embedding_result or not embedding_result.get('success'):
                    last_error = "Failed to generate CLIP embeddings"
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"   ⚠️ Retry {retry_count}/{max_retries} for image {image_id}")
                        await asyncio.sleep(2 ** retry_count)
                    continue

                embeddings = embedding_result.get('embeddings', {})
                model_used = embedding_result.get('model_used', 'unknown')

                # 🔍 BBOX TRACE: Check if bbox changed after embedding generation
                bbox_after = img_data.get('bbox')
                bbox_after_len = len(bbox_after) if isinstance(bbox_after, (list, tuple)) else 'N/A'
                if isinstance(bbox_after, (list, tuple)) and len(bbox_after) > 10:
                    logger.error(
                        f"   ❌ [BBOX TRACE] CORRUPTION AFTER EMBEDDING! bbox has {len(bbox_after)} elements. "
                        f"This suggests embedding_service modified img_data!"
                    )
                    logger.error(f"   ❌ [BBOX TRACE] embeddings keys: {list(embeddings.keys())}")

                # Save visual SLIG embedding (768D from SigLIP2 via SLIG cloud endpoint).
                # Producer key is `visual_768` — see real_embeddings_service.generate_all_embeddings.
                # VECS is the single source of truth for image embeddings; the
                # has_slig_embedding boolean flag on document_images is updated
                # automatically by vecs_service after a successful upsert.
                visual_embedding = embeddings.get('visual_768')
                if visual_embedding:
                    try:
                        await self.vecs_service.upsert_image_embedding(
                            image_id=image_id,
                            siglip_embedding=visual_embedding,
                            metadata={
                                'document_id': document_id,
                                'workspace_id': workspace_id,
                                'page_number': img_data.get('page_number', 1),
                                'quality_score': img_data.get('quality_score', 0.5)
                            }
                        )
                        logger.debug(f"   ✅ Saved visual embedding to VECS for {image_id}")
                    except Exception as vecs_error:
                        logger.error(f"   ❌ Failed to save visual embedding to VECS: {vecs_error}")
                        last_error = f"Failed to save visual embedding: {vecs_error}"
                        retry_count += 1
                        if retry_count < max_retries:
                            await asyncio.sleep(2 ** retry_count)
                        continue

                # Save specialized embeddings (SLIG 768D, emitted by real_embeddings_service
                # under the keys color_slig_768 / texture_slig_768 / style_slig_768 / material_slig_768).
                specialized_embeddings = {}
                if embeddings.get('color_slig_768'):
                    specialized_embeddings['color'] = embeddings.get('color_slig_768')
                if embeddings.get('texture_slig_768'):
                    specialized_embeddings['texture'] = embeddings.get('texture_slig_768')
                if embeddings.get('style_slig_768'):
                    specialized_embeddings['style'] = embeddings.get('style_slig_768')
                if embeddings.get('material_slig_768'):
                    specialized_embeddings['material'] = embeddings.get('material_slig_768')

                # Save understanding embedding to VECS if present (1024D from Voyage AI)
                understanding_embedding = embeddings.get('understanding_1024')
                if understanding_embedding:
                    try:
                        await self.vecs_service.upsert_understanding_embedding(
                            image_id=image_id,
                            embedding=understanding_embedding,
                            metadata={
                                'document_id': document_id,
                                'workspace_id': workspace_id,
                                'page_number': img_data.get('page_number', 1)
                            }
                        )
                        logger.debug(f"   ✅ Saved understanding embedding (1024D) to VECS for {image_id}")
                    except Exception as understanding_error:
                        logger.warning(f"   ⚠️ Failed to save understanding embedding to VECS: {understanding_error}")

                if specialized_embeddings:
                    # Save to VECS collections — single source of truth.
                    # The has_color_slig / has_texture_slig / has_style_slig /
                    # has_material_slig flags on document_images are updated by
                    # vecs_service automatically after each successful upsert.
                    await self.vecs_service.upsert_specialized_embeddings(
                        image_id=image_id,
                        embeddings=specialized_embeddings,
                        metadata={
                            'document_id': document_id,
                            'page_number': img_data.get('page_number', 1)
                        }
                    )

                    # ✨ Stage 3.5 - Convert visual embeddings to text metadata
                    try:
                        from app.services.metadata.visual_metadata_service import VisualMetadataService

                        logger.info(f"   🎨 Stage 3.5: Converting visual embeddings to text metadata for {image_id}")
                        visual_metadata_service = VisualMetadataService(workspace_id=workspace_id)

                        # Prepare embeddings for conversion (SLIG 768D — canonical schema).
                        embeddings_for_conversion = {}
                        if embeddings.get('color_slig_768'):
                            embeddings_for_conversion['color_slig_768'] = embeddings.get('color_slig_768')
                        if embeddings.get('texture_slig_768'):
                            embeddings_for_conversion['texture_slig_768'] = embeddings.get('texture_slig_768')
                        if embeddings.get('material_slig_768'):
                            embeddings_for_conversion['material_slig_768'] = embeddings.get('material_slig_768')
                        if embeddings.get('style_slig_768'):
                            embeddings_for_conversion['style_slig_768'] = embeddings.get('style_slig_768')

                        if embeddings_for_conversion:
                            visual_metadata_result = await visual_metadata_service.process_image_visual_metadata(
                                image_id=image_id,
                                embeddings=embeddings_for_conversion
                            )

                            if visual_metadata_result.get('success'):
                                logger.info(f"   ✅ Visual metadata extracted and saved for {image_id}")
                            else:
                                logger.warning(f"   ⚠️ Visual metadata extraction failed: {visual_metadata_result.get('error')}")
                        else:
                            logger.debug(f"   ℹ️ No SigLIP embeddings available for visual metadata extraction")

                    except Exception as visual_meta_error:
                        logger.warning(f"   ⚠️ Visual metadata extraction failed (non-critical): {visual_meta_error}")

                # Per-image vector inventory — what actually got into VECS for this
                # image. Used by save_images_and_generate_clips to aggregate job-level
                # stats and surface them in the admin UI.
                per_image_stats = {
                    'image_id': image_id,
                    'visual_slig': bool(visual_embedding),
                    'color_slig': 'color' in specialized_embeddings,
                    'texture_slig': 'texture' in specialized_embeddings,
                    'style_slig': 'style' in specialized_embeddings,
                    'material_slig': 'material' in specialized_embeddings,
                    'understanding': bool(understanding_embedding),
                    'vision_analysis_source': vision_analysis_source,
                }

                total_embeddings = (
                    (1 if visual_embedding else 0)
                    + len(specialized_embeddings)
                    + (1 if understanding_embedding else 0)
                )
                logger.info(
                    f"   ✅ Image {image_id}: saved {total_embeddings} embeddings "
                    f"(visual={per_image_stats['visual_slig']}, "
                    f"specialized={len(specialized_embeddings)}/4, "
                    f"understanding={per_image_stats['understanding']}, "
                    f"vision_analysis={vision_analysis_source})"
                )
                return (True, True, None, per_image_stats)

            except Exception as e:
                last_error = str(e)
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"   ⚠️ Retry {retry_count}/{max_retries} for image {idx + 1}: {e}")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"   ❌ Failed after {max_retries} retries for image {idx + 1}: {e}")

        return (False, False, last_error, {
            'image_id': img_data.get('id'),
            'visual_slig': False,
            'color_slig': False,
            'texture_slig': False,
            'style_slig': False,
            'material_slig': False,
            'understanding': False,
            'vision_analysis_source': VisionProvider.FAILED.value,
        })

    # ------------------------------------------------------------------ #
    # Icon candidate processing — OCR + Claude → spec metadata, NO VECS  #
    # ------------------------------------------------------------------ #

    async def _process_icon_candidate(
        self,
        img_data: Dict[str, Any],
        document_id: str,
        workspace_id: str,
        idx: int,
        total: int,
    ) -> Tuple[bool, bool, Optional[str], Dict[str, Any]]:
        """
        Process a single icon-candidate image.

        Steps:
          1. save_single_image — write the document_images row with category='icon_metadata'
          2. ocr_service.extract_icon_metadata — OCR + Claude with the
             `Icon-Based Metadata Extraction` prompt
          3. UPDATE document_images.metadata['icon_metadata'] with the
             extracted IconMetadata items (audit trail)
          4. NO embedding generation, NO VECS writes, NO SLIG calls.

        The Stage 4 product consolidation step reads metadata['icon_metadata']
        across all images for a product and rolls them up into the flat
        top-level keys on products.metadata that match material_metadata_fields.

        Returns:
            (image_saved, embedding_generated, error_message, per_image_stats)
            where embedding_generated is always False (icons get no embeddings)
            and per_image_stats reports zeros for every embedding type plus
            an `icon_metadata_count` field showing how many spec items were
            extracted from this icon.
        """
        per_image_stats = {
            'image_id': None,
            'visual_slig': False,
            'color_slig': False,
            'texture_slig': False,
            'style_slig': False,
            'material_slig': False,
            'understanding': False,
            'vision_analysis_source': VisionProvider.SKIPPED.value,
            'is_icon_candidate': True,
            'icon_metadata_count': 0,
        }

        try:
            # Step 1: save the document_images row, marked as an icon
            image_id = await self.supabase_client.save_single_image(
                image_info=img_data,
                document_id=document_id,
                workspace_id=workspace_id,
                image_index=idx,
                category='icon_metadata',
                extraction_method=img_data.get('extraction_method', 'pymupdf'),
                bbox=img_data.get('bbox'),
                detection_confidence=img_data.get('detection_confidence'),
                product_name=img_data.get('product_name'),
                material_category='icon_metadata',
            )
            if not image_id:
                return (False, False, 'Failed to save icon image to database', per_image_stats)

            img_data['id'] = image_id
            per_image_stats['image_id'] = image_id

            # Step 2: OCR + Claude — extract structured spec items from the icon
            image_path = img_data.get('path')
            if not image_path or not os.path.exists(image_path):
                logger.warning(
                    f"   ⚠️ Icon image file not found at {image_path} for {image_id}"
                )
                return (True, False, 'Icon image file not found', per_image_stats)

            try:
                from app.services.pdf.ocr_service import get_ocr_service
                ocr_service = get_ocr_service()
                icon_metadata_items = await ocr_service.extract_icon_metadata(
                    image=image_path,
                    workspace_id=workspace_id,
                    use_ai=True,
                )
            except Exception as ocr_err:
                logger.warning(
                    f"   ⚠️ Icon OCR/AI extraction failed for {image_id}: {ocr_err}"
                )
                return (True, False, f'Icon extraction failed: {ocr_err}', per_image_stats)

            if not icon_metadata_items:
                logger.info(
                    f"   ℹ️ Icon {image_id} (page {img_data.get('page_number')}): "
                    f"no spec items extracted"
                )
                return (True, False, None, per_image_stats)

            # Step 3: persist the extracted items as JSONB on document_images.metadata
            # for the audit trail. Stage 4 reads this to roll up onto the product.
            try:
                serialized_items = [
                    {
                        'field_name': item.field_name,
                        'value': item.value,
                        'confidence': item.confidence,
                        'bbox': item.bbox,
                        'icon_type': item.icon_type,
                    }
                    for item in icon_metadata_items
                ]
                # Read current metadata, merge, write back. We use a single
                # update because we know icon images don't compete with other
                # writers (no embedding pipeline, no vision_analysis update).
                existing = self.supabase_client.client.table('document_images')\
                    .select('metadata')\
                    .eq('id', image_id)\
                    .single()\
                    .execute()
                existing_meta = (existing.data or {}).get('metadata') or {}
                if not isinstance(existing_meta, dict):
                    existing_meta = {}
                existing_meta['icon_metadata'] = serialized_items

                self.supabase_client.client.table('document_images').update({
                    'metadata': existing_meta,
                }).eq('id', image_id).execute()

                per_image_stats['icon_metadata_count'] = len(serialized_items)
                field_names = sorted({item.field_name for item in icon_metadata_items})
                logger.info(
                    f"   🔖 Icon {image_id} (page {img_data.get('page_number')}): "
                    f"extracted {len(serialized_items)} spec items "
                    f"({', '.join(field_names[:6])}{'...' if len(field_names) > 6 else ''})"
                )
                return (True, False, None, per_image_stats)

            except Exception as persist_err:
                logger.warning(
                    f"   ⚠️ Failed to persist icon_metadata on document_images "
                    f"for {image_id}: {persist_err}"
                )
                # Treat as a partial success — the image is saved, items were
                # extracted, but we couldn't persist them. Better to retry the
                # whole image at the next backfill than to silently lose data.
                return (True, False, f'Icon metadata persist failed: {persist_err}', per_image_stats)

        except Exception as e:
            logger.error(
                f"   ❌ Unexpected error processing icon candidate at idx {idx}: {e}",
                exc_info=True,
            )
            return (False, False, str(e), per_image_stats)

    async def save_images_and_generate_clips(
        self,
        material_images: List[Dict[str, Any]],
        document_id: str,
        workspace_id: str,
        batch_size: int = 20,
        max_retries: int = 3,
        material_category: Optional[str] = None,
        job_id: Optional[str] = None,  # NEW: Job ID for AI cost tracking
        tracker: Optional[Any] = None,  # NEW: ProgressTracker for per-image events
        progress_label: Optional[str] = None,  # e.g. "Stage 3: Processing images for {product}"
        icon_candidates: Optional[List[Dict[str, Any]]] = None,  # NEW: spec icons → OCR + Claude path
    ) -> Dict[str, Any]:
        """
        Save images to database and generate CLIP embeddings with batching and retry logic.

        This method implements:
        1. Batch processing (default: 20 images per batch)
        2. Retry logic with exponential backoff (up to 3 retries per image)
        3. Checkpoint recovery (resume from last successful batch)
        4. Detailed error tracking (log which images fail and why)
        5. Per-image progress events to ProgressTracker (visible in admin UI)
        6. Per-vector statistics aggregation (visual + 4 specialized + understanding)
        7. Icon candidate processing — when `icon_candidates` is provided,
           those images are routed to OCR + Claude for spec extraction and
           are NOT given any visual embeddings.

        Args:
            material_images: List of regular material image data (full Stage 3 pipeline)
            document_id: Document ID
            workspace_id: Workspace ID
            batch_size: Number of images to process per batch (default: 20)
            max_retries: Maximum retry attempts per image (default: 3)
            material_category: Material category from upload (tiles, heatpump, wood, etc.)
            job_id: Optional job ID for AI cost tracking/aggregation
            tracker: Optional ProgressTracker — when provided, per-image progress
                     events are pushed via update_detailed_progress() so the admin
                     UI can show "Image 12/50: generating embeddings".
            progress_label: Optional label prefix shown in progress updates.
            icon_candidates: Optional list of icon-shaped images that should be
                             routed to the icon extraction path (OCR + Claude →
                             spec metadata, NO visual embeddings). When None or
                             empty, only the regular material path runs.

        Returns:
            Dict with counts, failed images, and per-vector stats: {
                images_saved,
                clip_embeddings_generated,
                failed_images,
                vector_stats: {
                    visual_slig, color_slig, texture_slig, style_slig,
                    material_slig, understanding,
                    vision_analysis_<provider> (one per VisionProvider value),
                    icon_candidates_processed,
                    icon_metadata_extracted,
                    icon_extraction_failed,
                }
            }
        """
        icon_candidates = icon_candidates or []

        logger.info(f"💾 Saving {len(material_images)} material images to database and generating CLIP embeddings...")
        if icon_candidates:
            logger.info(f"   🔖 Plus {len(icon_candidates)} icon candidates → OCR + Claude path (no embeddings)")
        logger.info(f"   📦 Batch size: {batch_size}, Max retries: {max_retries}")

        images_saved_count = 0
        clip_embeddings_count = 0
        failed_images = []

        # Per-vector aggregation — drives the admin UI's "what actually populated"
        # display and lets us answer "how many images got the understanding
        # embedding?" without round-tripping to VECS.
        #
        # vision_analysis_* keys are derived from the VisionProvider enum so
        # adding a new provider value (e.g. a new fallback model) only requires
        # adding it to the enum — the aggregation keys appear automatically.
        vector_stats = {
            'visual_slig': 0,
            'color_slig': 0,
            'texture_slig': 0,
            'style_slig': 0,
            'material_slig': 0,
            'understanding': 0,
            # Icon pipeline counters — populated below if icon_candidates is set.
            'icon_candidates_processed': 0,
            'icon_metadata_extracted': 0,  # # of icons that returned ≥1 spec item
            'icon_extraction_failed': 0,   # # of icons whose OCR/Claude failed
        }
        for vp in VisionProvider:
            vector_stats[f'vision_analysis_{vp.value}'] = 0

        # Check checkpoint - get number of images already processed
        checkpoint_index = await self._get_embedding_checkpoint(document_id)
        if checkpoint_index > 0:
            logger.info(f"   ⏭️ Resuming from checkpoint: {checkpoint_index} images already have embeddings")
            # Skip already processed images
            material_images = material_images[checkpoint_index:]
            if not material_images:
                logger.info(f"   ✅ All images already processed!")
                return {
                    'images_saved': checkpoint_index,
                    'clip_embeddings_generated': checkpoint_index,
                    'failed_images': [],
                    'vector_stats': vector_stats,
                }

        # Process in batches
        total_images = len(material_images)
        total_with_checkpoint = total_images + checkpoint_index
        label_prefix = progress_label or "Stage 3: Processing images"

        # Initial progress event so the UI shows "0/N" before the first image
        # finishes (otherwise the user sees no movement until the first image
        # completes, which can be 10-20s into the run).
        if tracker is not None:
            try:
                await tracker.update_detailed_progress(
                    current_step=f"{label_prefix} (0/{total_with_checkpoint})",
                    progress_current=checkpoint_index,
                    progress_total=total_with_checkpoint,
                )
            except Exception as tracker_init_err:
                logger.debug(f"   ⚠️ Tracker init update failed (non-critical): {tracker_init_err}")

        # 2026-04-10: parallelized this loop with a per-image semaphore.
        # The previous sequential per-image pattern caused Qwen3-VL endpoint
        # auto-pause: each image's Qwen material-analysis call was followed
        # by ~15s of save+update+embed orchestration before the next call,
        # so Qwen sat idle for 15s gaps and the 60s auto-pause eventually
        # fired mid-job (Stage 3.5 Material analysis fallback to Claude).
        # Per-image work is fully isolated (no cross-image state), so we
        # run POST_PROCESSING_CONCURRENCY in flight via a semaphore. This
        # keeps Qwen continuously busy with N concurrent calls (well under
        # the 4-replica capacity), eliminates the auto-pause window, and
        # cuts wall clock from sequential ~16s/img to ~16s/N amortized.
        # The outer batching loop is preserved for memory bounds.
        POST_PROCESSING_CONCURRENCY = 8
        post_processing_sem = asyncio.Semaphore(POST_PROCESSING_CONCURRENCY)

        # Dedicated counter for "completed so far" so the per-image progress
        # label is monotonic regardless of which task finishes first. We
        # increment it inside _process_one after the heavy work, on the
        # event loop, where increments are atomic under asyncio's
        # single-threaded cooperative scheduling.
        completed_counter = {'value': checkpoint_index}

        async def _process_one(img_data, global_idx):
            """Per-image task: heavy work behind semaphore, aggregation after.

            Aggregation runs on the event loop after each await, so increments
            on the shared counters/dicts/lists are safe under asyncio's
            single-threaded cooperative scheduling — no Lock needed.
            """
            nonlocal images_saved_count, clip_embeddings_count

            async with post_processing_sem:
                image_saved, embedding_generated, error, per_image_stats = (
                    await self._process_single_image_with_retry(
                        img_data=img_data,
                        document_id=document_id,
                        workspace_id=workspace_id,
                        idx=global_idx,
                        total=total_with_checkpoint,
                        max_retries=max_retries,
                        material_category=material_category,
                    )
                )

            if image_saved:
                images_saved_count += 1
            if embedding_generated:
                clip_embeddings_count += 1

            # Aggregate per-vector counts
            for vec_key in (
                'visual_slig', 'color_slig', 'texture_slig',
                'style_slig', 'material_slig', 'understanding',
            ):
                if per_image_stats.get(vec_key):
                    vector_stats[vec_key] += 1

            # Aggregate vision_analysis provenance counts. The source value is
            # always one of VisionProvider's string values (the orchestrator
            # uses the enum). Defensive: an unknown value gets bucketed as
            # SKIPPED so we don't silently lose the count.
            source = per_image_stats.get('vision_analysis_source', VisionProvider.SKIPPED.value)
            source_key = f'vision_analysis_{source}'
            if source_key in vector_stats:
                vector_stats[source_key] += 1
            else:
                vector_stats[f'vision_analysis_{VisionProvider.SKIPPED.value}'] += 1

            if error:
                failed_images.append({
                    'index': global_idx,
                    'image_id': per_image_stats.get('image_id'),
                    'path': img_data.get('path'),
                    'page_number': img_data.get('page_number'),
                    'error': error,
                })

            # Per-image progress event — pushed to background_jobs so the
            # admin UI updates as we go. update_detailed_progress() debounces
            # internally (MIN_SYNC_INTERVAL = 2s), so we can call it from
            # every concurrent slot without overwhelming the database.
            completed_counter['value'] += 1
            if tracker is not None:
                try:
                    step_label = (
                        f"{label_prefix} ({completed_counter['value']}/{total_with_checkpoint}) "
                        f"— v:{vector_stats['visual_slig']} "
                        f"c:{vector_stats['color_slig']} "
                        f"t:{vector_stats['texture_slig']} "
                        f"s:{vector_stats['style_slig']} "
                        f"m:{vector_stats['material_slig']} "
                        f"u:{vector_stats['understanding']}"
                    )
                    await tracker.update_detailed_progress(
                        current_step=step_label,
                        progress_current=completed_counter['value'],
                        progress_total=total_with_checkpoint,
                    )
                except Exception as tracker_err:
                    logger.debug(f"   ⚠️ Tracker update failed (non-critical): {tracker_err}")

        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            batch = material_images[batch_start:batch_end]

            logger.info(
                f"   📦 Processing batch {batch_start // batch_size + 1}/"
                f"{(total_images + batch_size - 1) // batch_size} "
                f"({batch_start + 1}-{batch_end}/{total_images}) "
                f"with concurrency {POST_PROCESSING_CONCURRENCY}"
            )

            # Build and run per-image tasks for this batch in parallel.
            # return_exceptions=True is a safety net — _process_single_image_with_retry
            # already handles its own retries and returns (False, False, error, stats)
            # rather than raising, but we keep it so an unexpected exception in
            # one image cannot take down the whole batch.
            tasks = [
                _process_one(img_data, batch_start + idx + checkpoint_index)
                for idx, img_data in enumerate(batch)
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Log batch completion
            logger.info(f"   ✅ Batch {batch_start // batch_size + 1} complete: {len(batch)} images processed")

        # ──────────────────────────────────────────────────────────────── #
        # Icon candidate processing                                         #
        # ──────────────────────────────────────────────────────────────── #
        # Icons are processed AFTER the regular material loop so they
        # don't compete for the Qwen endpoint while embeddings are running.
        # Each icon gets a single Claude call (the icon prompt is small and
        # the OCR step is local), so we run them with a smaller concurrency
        # cap to avoid hammering the Anthropic rate limit.
        if icon_candidates:
            ICON_CONCURRENCY = 4
            icon_sem = asyncio.Semaphore(ICON_CONCURRENCY)
            icon_completed = {'value': 0}
            icon_total = len(icon_candidates)
            icon_label = (progress_label or "Stage 3: Processing images") + " — icons"

            async def _process_one_icon(img_data, icon_idx):
                """Per-icon task: OCR + Claude → spec metadata, behind a semaphore."""
                nonlocal images_saved_count
                async with icon_sem:
                    image_saved, _embedding_generated, error, per_icon_stats = (
                        await self._process_icon_candidate(
                            img_data=img_data,
                            document_id=document_id,
                            workspace_id=workspace_id,
                            idx=icon_idx,
                            total=icon_total,
                        )
                    )

                if image_saved:
                    images_saved_count += 1

                vector_stats['icon_candidates_processed'] += 1
                if per_icon_stats.get('icon_metadata_count', 0) > 0:
                    vector_stats['icon_metadata_extracted'] += 1
                if error and per_icon_stats.get('icon_metadata_count', 0) == 0:
                    vector_stats['icon_extraction_failed'] += 1

                if error:
                    failed_images.append({
                        'index': icon_idx,
                        'image_id': per_icon_stats.get('image_id'),
                        'path': img_data.get('path'),
                        'page_number': img_data.get('page_number'),
                        'error': f'[icon] {error}',
                    })

                # Per-icon progress event so the UI shows icons advancing
                # alongside the regular material counters.
                icon_completed['value'] += 1
                if tracker is not None:
                    try:
                        step_label = (
                            f"{icon_label} ({icon_completed['value']}/{icon_total}) "
                            f"— v:{vector_stats['visual_slig']} "
                            f"c:{vector_stats['color_slig']} "
                            f"t:{vector_stats['texture_slig']} "
                            f"s:{vector_stats['style_slig']} "
                            f"m:{vector_stats['material_slig']} "
                            f"u:{vector_stats['understanding']} "
                            f"i:{vector_stats['icon_metadata_extracted']}"
                        )
                        await tracker.update_detailed_progress(
                            current_step=step_label,
                            progress_current=total_with_checkpoint + icon_completed['value'],
                            progress_total=total_with_checkpoint + icon_total,
                        )
                    except Exception as tracker_err:
                        logger.debug(f"   ⚠️ Tracker icon update failed (non-critical): {tracker_err}")

            logger.info(
                f"   🔖 Processing {icon_total} icon candidates with concurrency {ICON_CONCURRENCY}"
            )
            for icon_batch_start in range(0, icon_total, batch_size):
                icon_batch_end = min(icon_batch_start + batch_size, icon_total)
                icon_batch = icon_candidates[icon_batch_start:icon_batch_end]
                icon_tasks = [
                    _process_one_icon(img_data, icon_batch_start + idx)
                    for idx, img_data in enumerate(icon_batch)
                ]
                await asyncio.gather(*icon_tasks, return_exceptions=True)
                logger.info(
                    f"   ✅ Icon batch {icon_batch_start // batch_size + 1} complete: "
                    f"{len(icon_batch)} icons processed"
                )

        # Final summary
        logger.info(f"✅ Image processing complete:")
        logger.info(f"   Images saved to DB: {images_saved_count + checkpoint_index}")
        logger.info(f"   CLIP embeddings generated: {clip_embeddings_count + checkpoint_index}")
        logger.info(
            f"   Vectors written: visual={vector_stats['visual_slig']}, "
            f"color={vector_stats['color_slig']}, texture={vector_stats['texture_slig']}, "
            f"style={vector_stats['style_slig']}, material={vector_stats['material_slig']}, "
            f"understanding={vector_stats['understanding']}"
        )
        provenance_summary = ', '.join(
            f"{vp.value}={vector_stats[f'vision_analysis_{vp.value}']}"
            for vp in VisionProvider
        )
        logger.info(f"   Vision analysis provenance: {provenance_summary}")
        if vector_stats['icon_candidates_processed'] > 0:
            logger.info(
                f"   Icon extraction: {vector_stats['icon_metadata_extracted']}/"
                f"{vector_stats['icon_candidates_processed']} icons produced spec items "
                f"(failed: {vector_stats['icon_extraction_failed']})"
            )

        if failed_images:
            logger.warning(f"   ⚠️ Failed images: {len(failed_images)}")
            for failed in failed_images[:5]:  # Log first 5 failures
                logger.warning(f"      - Image {failed['index']} (page {failed['page_number']}): {failed['error']}")
            if len(failed_images) > 5:
                logger.warning(f"      ... and {len(failed_images) - 5} more")

        return {
            'images_saved': images_saved_count + checkpoint_index,
            'clip_embeddings_generated': clip_embeddings_count + checkpoint_index,
            'failed_images': failed_images,
            'vector_stats': vector_stats,
        }
