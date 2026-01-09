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
from app.utils.page_converter import PageConverter, PageNumber  
from app.config import get_settings


logger = logging.getLogger(__name__)


class ImageProcessingService:
    """Service for handling all image processing operations."""

    def __init__(self):
        """Initialize service."""
        self.supabase_client = get_supabase_client()
        self.vecs_service = VecsService()
        self.embedding_service = RealEmbeddingsService()
        self.pdf_processor = PDFProcessor()
        self.settings = get_settings()

    async def classify_images(
        self,
        extracted_images: List[Dict[str, Any]],
        confidence_threshold: float = 0.6,  # OPTIMIZED: Lowered from 0.7 to reduce validation calls
        primary_model: str = "Qwen/Qwen3-VL-32B-Instruct",  # PRIMARY: Qwen3-VL-32B (reliable, high accuracy)
        validation_model: str = "claude-sonnet-4-20250514",  # FALLBACK: Claude Sonnet 4.5 (highest quality)
        batch_size: int = 15  # NEW: Process images in batches to prevent OOM
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

        # Get HuggingFace API key for all cloud endpoints
        huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')

        if not huggingface_api_key:
            logger.error("❌ CRITICAL: HUGGINGFACE_API_KEY environment variable not set!")
            logger.error("   Image classification will fail. Please set HUGGINGFACE_API_KEY.")
            raise ValueError("HUGGINGFACE_API_KEY not configured")

        async def classify_image_with_vision_model(image_path: str, model: str, base64_data: str = None) -> Dict[str, Any]:
            """Fast classification using vision model (Qwen via TogetherAI)."""
            import time
            from app.services.core.ai_call_logger import AICallLogger

            start_time = time.time()
            # If base64_data is provided, we use it directly. Otherwise, we read from disk.
            image_base64 = base64_data
            try:
                if not image_base64:
                    with open(image_path, 'rb') as f:
                        image_bytes = f.read()
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                        del image_bytes

                classification_prompt = """Analyze this image and classify it as:
1. MATERIAL: Shows building/interior materials (tiles, wood, fabric, stone, metal, flooring, wallpaper, etc.) - either close-up texture or in application
2. NOT_MATERIAL: Faces, logos, charts, diagrams, text, decorative graphics, abstract patterns

Respond ONLY with JSON:
{"is_material": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}"""

                async with httpx.AsyncClient(timeout=90.0) as client:  # ✅ Increased timeout from 30s to 90s for vision models
                    response = await client.post(
                        "https://api.together.xyz/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {huggingface_api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": model,
                            "messages": [{
                                "role": "user",
                                "content": [
                                    # ✅ CRITICAL FIX: Text must come BEFORE image for Qwen models
                                    # This matches Together AI documentation format
                                    {"type": "text", "text": classification_prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                                ]
                            }],
                            "max_tokens": 512,
                            "temperature": 0.1
                        }
                    )

                    response_data = response.json()

                    # ✅ CRITICAL FIX: Check for API errors before accessing 'choices'
                    # TogetherAI returns {"error": {...}} when service is unavailable or rate limited
                    if 'error' in response_data:
                        error_info = response_data['error']
                        error_msg = error_info.get('message', 'Unknown error')
                        error_type = error_info.get('type', 'unknown')

                        logger.error(f"❌ TogetherAI API Error for {image_path}")
                        logger.error(f"   Error Type: {error_type}")
                        logger.error(f"   Error Message: {error_msg}")
                        logger.error(f"   HTTP Status: {response.status_code}")
                        logger.error(f"   Response Data: {json.dumps(response_data, indent=2)}")

                        return {
                            'is_material': False,
                            'confidence': 0.0,
                            'reason': f'TogetherAI API error: {error_type} - {error_msg}',
                            'model': f'{model.split("/")[-1]}_api_error',
                            'error': error_msg,
                            'retry_recommended': error_type in ['service_unavailable', 'rate_limit_exceeded']
                        }

                    # ✅ CRITICAL FIX: Validate 'choices' exists in response
                    if 'choices' not in response_data or not response_data['choices']:
                        logger.error(f"❌ Invalid TogetherAI response for {image_path}")
                        logger.error(f"   Response missing 'choices' key")
                        logger.error(f"   Response Data: {json.dumps(response_data, indent=2)}")

                        return {
                            'is_material': False,
                            'confidence': 0.0,
                            'reason': 'Invalid API response: missing choices',
                            'model': f'{model.split("/")[-1]}_invalid_response',
                            'error': 'Response missing choices key'
                        }

                    result_text = response_data['choices'][0]['message']['content']

                    # ✅ FIX: Handle empty/invalid responses from Together AI
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

                    # Log TogetherAI call (Qwen models)
                    ai_logger = AICallLogger()
                    latency_ms = int((time.time() - start_time) * 1000)
                    usage = response_data.get('usage', {})
                    input_tokens = usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0)

                    # Qwen pricing (HuggingFace Endpoint)
                    # Qwen3-VL-32B: $0.40/1M input, $0.40/1M output (32B only, 8B removed)
                    cost = (input_tokens / 1_000_000) * 0.40 + (output_tokens / 1_000_000) * 0.40

                    await ai_logger.log_together_call(
                        task="image_classification",
                        model=model_short,
                        response=response_data,
                        latency_ms=latency_ms,
                        confidence_score=result.get('confidence', 0.5),
                        confidence_breakdown={
                            "model_confidence": result.get('confidence', 0.5),
                            "completeness": 1.0,
                            "consistency": 0.95,
                            "validation": 0.90
                        },
                        action="use_ai_result"
                    )

                    return {
                        'is_material': result.get('is_material', False),
                        'confidence': result.get('confidence', 0.5),
                        'reason': result.get('reason', 'Unknown'),
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

                classification_prompt = """Analyze this image and classify it into ONE of these categories:

1. **material_closeup**: Close-up photo showing material texture, surface, pattern, or finish (tiles, wood, fabric, stone, metal, etc.)
2. **material_in_situ**: Material shown in application/context (bathroom with tiles, furniture with fabric, room with flooring, etc.)
3. **non_material**: NOT material-related (faces, logos, decorative graphics, charts, diagrams, text, random images)

Respond ONLY with this JSON format:
{
    "classification": "material_closeup" | "material_in_situ" | "non_material",
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}"""

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

                is_material = result['classification'] in ['material_closeup', 'material_in_situ']

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

        # ✅ TIER-BASED RATE LIMITING: Dynamically adjust based on TogetherAI tier
        # Import rate limit configuration
        from app.config.rate_limits import VISION_CONCURRENCY, CLAUDE_CONCURRENCY, CURRENT_TIER

        logger.info(f"🎯 Rate Limiting Configuration:")
        logger.info(f"   TogetherAI Tier: {CURRENT_TIER.tier} (${CURRENT_TIER.total_spend} spent)")
        logger.info(f"   LLM Rate Limit: {CURRENT_TIER.llm_rpm} RPM ({CURRENT_TIER.llm_rps:.1f} RPS)")
        logger.info(f"   Vision Concurrency: {VISION_CONCURRENCY} concurrent requests")
        logger.info(f"   Claude Concurrency: {CLAUDE_CONCURRENCY} concurrent requests")

        # Two-stage classification with tier-based semaphores for rate limiting
        together_semaphore = Semaphore(VISION_CONCURRENCY)  # Dynamic based on tier
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
            async with together_semaphore:
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

                if upload_result:
                    img_data['storage_url'] = upload_result.get('storage_url')
                    img_data['storage_path'] = upload_result.get('storage_path')
                    return img_data
                else:
                    logger.warning(f"Failed to upload image: {img_data.get('filename')}")
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
        max_retries: int = 3
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

        Returns:
            Tuple of (image_saved, embedding_generated, error_message)
        """
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                # Save to database with category='product' for material images
                # (ai_classification is already in img_data from classify_images)
                image_id = await self.supabase_client.save_single_image(
                    image_info=img_data,
                    document_id=document_id,
                    workspace_id=workspace_id,
                    image_index=idx,
                    category='product',  # ✅ All images in this flow are material images
                    extraction_method=img_data.get('extraction_method', 'pymupdf'),
                    bbox=img_data.get('bbox'),
                    detection_confidence=img_data.get('detection_confidence'),
                    product_name=img_data.get('product_name')
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

                # Save visual CLIP embedding
                visual_embedding = embeddings.get('visual_512')
                if visual_embedding:
                    # Save to embeddings table for tracking
                    try:
                        # Save visual CLIP embedding to document_images
                        update_data = {
                            "visual_clip_embedding_512": visual_embedding
                        }
                        self.supabase_client.client.table('document_images').update(update_data).eq('id', image_id).execute()
                        logger.debug(f"   ✅ Saved visual CLIP embedding (512D) to document_images for {image_id}")
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
        max_retries: int = 3
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
                    max_retries=max_retries
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
