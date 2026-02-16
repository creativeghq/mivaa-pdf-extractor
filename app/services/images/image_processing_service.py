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
from app.services.pdf.pdf_processor import PDFProcessor
# PageConverter removed - using simple PDF page numbers instead
from app.config import get_settings


logger = logging.getLogger(__name__)


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
        self._load_classification_prompt()

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

    async def classify_images(
        self,
        extracted_images: List[Dict[str, Any]],
        confidence_threshold: float = 0.6,  # OPTIMIZED: Lowered from 0.7 to reduce validation calls
        primary_model: str = "Qwen/Qwen3-VL-32B-Instruct",  # PRIMARY: Qwen3-VL-32B (reliable, high accuracy)
        validation_model: str = "claude-sonnet-4-20250514",  # FALLBACK: Claude Sonnet 4.5 (highest quality)
        batch_size: int = 15,  # NEW: Process images in batches to prevent OOM
        job_id: Optional[str] = None  # NEW: Job ID for AI cost tracking
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Classify images as material or non-material using Qwen Vision models.

        SUPPORTED MODELS:
        - Qwen/Qwen3-VL-32B-Instruct: PRIMARY - High accuracy, reliable ($0.50/1M input, $1.50/1M output)
        - claude-sonnet-4-20250514: FALLBACK - Highest quality vision model (for failures/low confidence)

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
        qwen_endpoint_url = qwen_config["endpoint_url"]

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
                endpoint_url=qwen_endpoint_url,
                endpoint_name=qwen_config["endpoint_name"],
                namespace=qwen_config["namespace"],
                endpoint_token=huggingface_api_key,
                enabled=qwen_config["enabled"]
            )
            logger.info("ℹ️ Created new Qwen manager (registry not available)")

        # Check if Qwen is enabled and assume it's ready (warmup done at job start)
        qwen_endpoint_available = qwen_config["enabled"]
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

                # Ensure base_url ends with /v1/ for OpenAI client compatibility
                # The client will append /chat/completions to this base
                if not qwen_endpoint_url.endswith('/v1/') and not qwen_endpoint_url.endswith('/v1'):
                    base_url = qwen_endpoint_url.rstrip('/') + '/v1/'
                else:
                    base_url = qwen_endpoint_url if qwen_endpoint_url.endswith('/') else qwen_endpoint_url + '/'

                logger.info(f"🔧 Qwen OpenAI client base_url: {base_url}")

                client = AsyncOpenAI(
                    base_url=base_url,  # https://...huggingface.cloud/v1/
                    api_key=huggingface_api_key,
                    timeout=90.0
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
                        raise  # Re-raise if not 503 or last attempt
                
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

                await ai_logger.log_qwen_call(
                    task="image_classification",
                    model=model_short,
                    response=response_dict,
                    latency_ms=latency_ms,
                    confidence_score=result.get('confidence', 0.5),
                    confidence_breakdown={
                        "model_confidence": result.get('confidence', 0.5),
                        "completeness": 1.0,
                        "consistency": 0.95,
                        "validation": 0.90
                    },
                    action="use_ai_result",
                    job_id=job_id  # Track cost per job
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
                # ✅ FIX 2: Enhanced error logging with full details
                model_short = model.split('/')[-1] if '/' in model else model
                logger.error(f"❌ CLASSIFICATION FAILED for {image_path}")
                logger.error(f"   Model: {model}")
                logger.error(f"   Error Type: {type(e).__name__}")
                logger.error(f"   Error Message: {str(e)}")

                # ✅ NEW: Log response_data if available (for debugging empty responses)
                try:
                    if 'response_data' in locals():
                        logger.error(f"   Response Data: {json.dumps(response_data, indent=2)[:500]}")
                        if 'choices' in response_data and len(response_data['choices']) > 0:
                            content = response_data['choices'][0].get('message', {}).get('content', '')
                            logger.error(f"   Content Length: {len(content)} chars")
                            logger.error(f"   Content Preview: {content[:200]}")
                except Exception as log_err:
                    logger.error(f"   Could not log response data: {log_err}")

                logger.error(f"   Stack Trace:\n{traceback.format_exc()}")

                # Check if it's an API error
                if hasattr(e, 'response'):
                    logger.error(f"   API Response Status: {getattr(e.response, 'status_code', 'N/A')}")
                    logger.error(f"   API Response Body: {getattr(e.response, 'text', 'N/A')[:500]}")

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

                response = await ai_service.anthropic_async.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                            {"type": "text", "text": classification_prompt}
                        ]
                    }]
                )

                result_text = response.content[0].text
                result = json.loads(result_text)

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
        from app.config.rate_limits import VISION_CONCURRENCY, CLAUDE_CONCURRENCY, CURRENT_TIER

        logger.info(f"🎯 Rate Limiting Configuration:")
        logger.info(f"   Vision Tier: {CURRENT_TIER.tier} (${CURRENT_TIER.total_spend} spent)")
        logger.info(f"   LLM Rate Limit: {CURRENT_TIER.llm_rpm} RPM ({CURRENT_TIER.llm_rps:.1f} RPS)")
        logger.info(f"   Vision Concurrency: {VISION_CONCURRENCY} concurrent requests")
        logger.info(f"   Claude Concurrency: {CLAUDE_CONCURRENCY} concurrent requests")

        # Two-stage classification with tier-based semaphores for rate limiting
        vision_semaphore = Semaphore(VISION_CONCURRENCY)  # Dynamic based on tier
        claude_semaphore = Semaphore(CLAUDE_CONCURRENCY)  # Conservative for Claude

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

            # STAGE 1: Fast primary model classification with retry logic
            async with vision_semaphore:
                primary_result = None
                max_retries = 3
                retry_delay = 1.0  # Start with 1 second

                for attempt in range(max_retries):
                    primary_result = await classify_image_with_vision_model(image_path, primary_model, base64_data=image_base64)

                    # ✅ CRITICAL: Check if we should retry (API errors, service unavailable)
                    should_retry = (
                        primary_result.get('retry_recommended', False) or
                        '_api_error' in primary_result.get('model', '')
                    )

                    if not should_retry or attempt == max_retries - 1:
                        break

                    # Exponential backoff with jitter
                    wait_time = retry_delay * (2 ** attempt) + (0.1 * attempt)
                    logger.warning(f"   🔄 Retrying classification for {filename} (attempt {attempt + 2}/{max_retries}) after {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)

            # ✅ NEW: Check if primary model failed (invalid response)
            primary_failed = (
                '_invalid_response' in primary_result.get('model', '') or
                '_not_json' in primary_result.get('model', '') or
                '_empty_response' in primary_result.get('model', '') or
                '_api_error' in primary_result.get('model', '')
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
            logger.error(f"   1. Together.AI API key invalid or expired")
            logger.error(f"   2. Together.AI service unavailable")
            logger.error(f"   3. Image files deleted before classification")
            logger.error(f"   4. Network connectivity issues")
            logger.error(f"   5. Model name incorrect or model unavailable")

            # Log first few image paths for debugging
            logger.error(f"   Sample image paths:")
            for i, img in enumerate(extracted_images[:3]):
                logger.error(f"     {i+1}. {img.get('path')} (exists: {os.path.exists(img.get('path', ''))})")

            raise Exception(
                f"Image classification completely failed: 0/{total_input} images classified. "
                f"Check Together.AI API key and service availability."
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
            result = self.supabase_client.client.table('document_images')\
                .select('id')\
                .eq('document_id', document_id)\
                .not_.is_('visual_clip_embedding_512', 'null')\
                .execute()

            if result.data:
                return len(result.data)  # Number of images with embeddings
            return 0
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to get embedding checkpoint: {e}")
            return 0

    async def _process_single_image_with_retry(
        self,
        img_data: Dict[str, Any],
        document_id: str,
        workspace_id: str,
        idx: int,
        total: int,
        max_retries: int = 3,
        material_category: Optional[str] = None
    ) -> Tuple[bool, bool, Optional[str]]:
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
                    return (True, False, "Image file not found")

                logger.info(f"   🎨 Generating CLIP embeddings for image {idx + 1}/{total}")

                # Read image and convert to base64
                with open(image_path, 'rb') as img_file:
                    image_bytes = img_file.read()
                    image_base64 = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

                # Generate all embeddings
                embedding_result = await self.embedding_service.generate_all_embeddings(
                    entity_id=image_id,
                    entity_type="image",
                    text_content="",
                    image_data=image_base64,
                    material_properties={}
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

                # Save visual CLIP embedding (768D from SigLIP2)
                # ✅ FIXED: Changed from 'visual_512' to 'visual_768' to match real_embeddings_service output
                visual_embedding = embeddings.get('visual_768')
                if visual_embedding:
                    # Save to embeddings table for tracking
                    try:
                        # Save visual CLIP embedding to document_images
                        # Note: Column is named visual_clip_embedding_512 but stores 768D SigLIP embeddings
                        update_data = {
                            "visual_clip_embedding_512": visual_embedding
                        }
                        self.supabase_client.client.table('document_images').update(update_data).eq('id', image_id).execute()
                        logger.debug(f"   ✅ Saved visual CLIP embedding (768D SigLIP) to document_images for {image_id}")
                    except Exception as emb_error:
                        logger.error(f"   ❌ Failed to save visual embedding to document_images: {emb_error}")
                        last_error = f"Failed to save visual embedding: {emb_error}"
                        retry_count += 1
                        if retry_count < max_retries:
                            await asyncio.sleep(2 ** retry_count)
                        continue

                    # Save to VECS collection for fast similarity search
                    try:
                        await self.vecs_service.upsert_image_embedding(
                            image_id=image_id,
                            siglip_embedding=visual_embedding,  # ✅ FIXED: Changed from clip_embedding to siglip_embedding
                            metadata={
                                'document_id': document_id,
                                'workspace_id': workspace_id,  # ✅ ADDED: Include workspace_id in metadata
                                'page_number': img_data.get('page_number', 1),
                                'quality_score': img_data.get('quality_score', 0.5)
                            }
                        )
                        logger.debug(f"   ✅ Saved visual embedding to VECS for {image_id}")
                    except Exception as vecs_error:
                        logger.warning(f"   ⚠️ Failed to save to VECS: {vecs_error}")

                # Save specialized embeddings (support both old 512D and new 1152D SigLIP)
                specialized_embeddings = {}

                # Check for new SigLIP embeddings (1152D) first
                if embeddings.get('color_siglip_1152'):
                    specialized_embeddings['color'] = embeddings.get('color_siglip_1152')
                elif embeddings.get('color_512'):
                    specialized_embeddings['color'] = embeddings.get('color_512')

                if embeddings.get('texture_siglip_1152'):
                    specialized_embeddings['texture'] = embeddings.get('texture_siglip_1152')
                elif embeddings.get('texture_512'):
                    specialized_embeddings['texture'] = embeddings.get('texture_512')

                if embeddings.get('style_siglip_1152'):
                    specialized_embeddings['style'] = embeddings.get('style_siglip_1152')

                if embeddings.get('material_siglip_1152'):
                    specialized_embeddings['material'] = embeddings.get('material_siglip_1152')
                elif embeddings.get('material_512'):
                    specialized_embeddings['material'] = embeddings.get('material_512')

                if embeddings.get('application_512'):
                    specialized_embeddings['application'] = embeddings.get('application_512')

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
                    # Save to VECS collections
                    await self.vecs_service.upsert_specialized_embeddings(
                        image_id=image_id,
                        embeddings=specialized_embeddings,
                        metadata={
                            'document_id': document_id,
                            'page_number': img_data.get('page_number', 1)
                        }
                    )

                    # Save specialized embeddings to document_images
                    update_data = {}
                    for emb_type, emb_vector in specialized_embeddings.items():
                        try:
                            # Map embedding type to column name
                            column_map = {
                                "color": "color_embedding_256",
                                "texture": "texture_embedding_256",
                                "application": "application_embedding_512"
                            }
                            column_name = column_map.get(emb_type)
                            if column_name:
                                update_data[column_name] = emb_vector
                                logger.debug(f"   ✅ Adding {emb_type} embedding to document_images for {image_id}")
                        except Exception as emb_error:
                            logger.warning(f"   ⚠️ Failed to prepare {emb_type} embedding: {emb_error}")

                    # Update document_images with all specialized embeddings
                    if update_data:
                        try:
                            self.supabase_client.client.table('document_images').update(update_data).eq('id', image_id).execute()
                            logger.debug(f"   ✅ Saved {len(update_data)} specialized embeddings to document_images for {image_id}")
                        except Exception as update_error:
                            logger.warning(f"   ⚠️ Failed to save specialized embeddings to document_images: {update_error}")

                    # ✨ NEW: Stage 3.5 - Convert visual embeddings to text metadata
                    try:
                        from app.services.metadata.visual_metadata_service import VisualMetadataService

                        logger.info(f"   🎨 Stage 3.5: Converting visual embeddings to text metadata for {image_id}")
                        visual_metadata_service = VisualMetadataService(workspace_id=workspace_id)

                        # Prepare embeddings for conversion (use SigLIP 1152D embeddings)
                        embeddings_for_conversion = {}
                        if embeddings.get('color_siglip_1152'):
                            embeddings_for_conversion['color_siglip_1152'] = embeddings.get('color_siglip_1152')
                        if embeddings.get('texture_siglip_1152'):
                            embeddings_for_conversion['texture_siglip_1152'] = embeddings.get('texture_siglip_1152')
                        if embeddings.get('material_siglip_1152'):
                            embeddings_for_conversion['material_siglip_1152'] = embeddings.get('material_siglip_1152')
                        if embeddings.get('style_siglip_1152'):
                            embeddings_for_conversion['style_siglip_1152'] = embeddings.get('style_siglip_1152')

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

                total_embeddings = 1 + len(specialized_embeddings)
                logger.info(f"   ✅ Generated and saved {total_embeddings} CLIP embeddings for image {image_id}")
                return (True, True, None)

            except Exception as e:
                last_error = str(e)
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"   ⚠️ Retry {retry_count}/{max_retries} for image {idx + 1}: {e}")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"   ❌ Failed after {max_retries} retries for image {idx + 1}: {e}")

        return (False, False, last_error)

    async def save_images_and_generate_clips(
        self,
        material_images: List[Dict[str, Any]],
        document_id: str,
        workspace_id: str,
        batch_size: int = 20,
        max_retries: int = 3,
        material_category: Optional[str] = None,
        job_id: Optional[str] = None  # NEW: Job ID for AI cost tracking
    ) -> Dict[str, Any]:
        """
        Save images to database and generate CLIP embeddings with batching and retry logic.

        This method implements:
        1. Batch processing (default: 20 images per batch)
        2. Retry logic with exponential backoff (up to 3 retries per image)
        3. Checkpoint recovery (resume from last successful batch)
        4. Detailed error tracking (log which images fail and why)

        Args:
            material_images: List of material image data
            document_id: Document ID
            workspace_id: Workspace ID
            batch_size: Number of images to process per batch (default: 20)
            max_retries: Maximum retry attempts per image (default: 3)
            material_category: Material category from upload (tiles, heatpump, wood, etc.)
            job_id: Optional job ID for AI cost tracking/aggregation

        Returns:
            Dict with counts and failed images: {
                images_saved,
                clip_embeddings_generated,
                failed_images: [{index, path, error}]
            }
        """
        logger.info(f"💾 Saving {len(material_images)} material images to database and generating CLIP embeddings...")
        logger.info(f"   📦 Batch size: {batch_size}, Max retries: {max_retries}")

        images_saved_count = 0
        clip_embeddings_count = 0
        failed_images = []

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
                    'failed_images': []
                }

        # Process in batches
        total_images = len(material_images)
        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            batch = material_images[batch_start:batch_end]

            logger.info(f"   📦 Processing batch {batch_start // batch_size + 1}/{(total_images + batch_size - 1) // batch_size} ({batch_start + 1}-{batch_end}/{total_images})")

            # Process batch images with retry logic
            for idx, img_data in enumerate(batch):
                global_idx = batch_start + idx + checkpoint_index

                image_saved, embedding_generated, error = await self._process_single_image_with_retry(
                    img_data=img_data,
                    document_id=document_id,
                    workspace_id=workspace_id,
                    idx=global_idx,
                    total=total_images + checkpoint_index,
                    max_retries=max_retries,
                    material_category=material_category
                )

                if image_saved:
                    images_saved_count += 1
                if embedding_generated:
                    clip_embeddings_count += 1

                if error:
                    failed_images.append({
                        'index': global_idx,
                        'path': img_data.get('path'),
                        'page_number': img_data.get('page_number'),
                        'error': error
                    })

            # Log batch completion
            logger.info(f"   ✅ Batch {batch_start // batch_size + 1} complete: {len(batch)} images processed")

        # Final summary
        logger.info(f"✅ Image processing complete:")
        logger.info(f"   Images saved to DB: {images_saved_count + checkpoint_index}")
        logger.info(f"   CLIP embeddings generated: {clip_embeddings_count + checkpoint_index}")

        if failed_images:
            logger.warning(f"   ⚠️ Failed images: {len(failed_images)}")
            for failed in failed_images[:5]:  # Log first 5 failures
                logger.warning(f"      - Image {failed['index']} (page {failed['page_number']}): {failed['error']}")
            if len(failed_images) > 5:
                logger.warning(f"      ... and {len(failed_images) - 5} more")

        return {
            'images_saved': images_saved_count + checkpoint_index,
            'clip_embeddings_generated': clip_embeddings_count + checkpoint_index,
            'failed_images': failed_images
        }
