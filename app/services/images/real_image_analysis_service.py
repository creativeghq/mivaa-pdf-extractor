"""
Real Image Analysis Service - Stage 4 Implementation

This service provides real image analysis using:
1. Configurable vision model (default: Qwen3-VL-8B) for detailed image analysis (superior OCR, table/diagram understanding)
2. Claude 4.5 Sonnet Vision for validation
3. CLIP embeddings for visual similarity (512D)
4. Material property extraction
5. Real quality scoring

Replaces mock data with actual AI model calls.
"""

import logging
import asyncio
import base64
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import os

import httpx
from PIL import Image
import io
import anthropic

from app.services.core.ai_call_logger import AICallLogger
from app.services.core.supabase_client import SupabaseClient, get_supabase_client
from app.services.core.ai_client_service import get_ai_client_service

logger = logging.getLogger(__name__)

# Get API keys from environment - will be loaded from settings in __init__
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


@dataclass
class ImageAnalysisResult:
    """Result of real image analysis"""
    image_id: str
    vision_analysis: Dict[str, Any]  # Vision model analysis (configurable)
    claude_validation: Optional[Dict[str, Any]]  # Claude 4.5 Sonnet validation (optional, async)
    clip_embedding: List[float]  # 512D CLIP embedding
    material_properties: Dict[str, Any]  # Extracted properties
    quality_score: float  # Real quality score (0.0-1.0)
    confidence_score: float  # Confidence in analysis (0.0-1.0)
    processing_time_ms: float
    timestamp: str
    needs_claude_validation: bool = False  # NEW: True if quality score < threshold


class RealImageAnalysisService:
    """
    Provides real image analysis using vision models and CLIP embeddings.

    This service replaces mock data with actual AI model calls:
    - Configurable vision model (default: Qwen3-VL-8B): Detailed image analysis
    - Claude 4.5 Sonnet Vision: Validation and enrichment
    - CLIP: Visual embeddings for similarity search
    """
    
    def __init__(self, supabase_client=None, embedding_service=None, workspace_id: str = None):
        self.logger = logger

        # Load HuggingFace endpoint configuration from settings
        from app.config import get_settings
        from app.services.embeddings.qwen_endpoint_manager import QwenEndpointManager

        settings = get_settings()
        self.workspace_id = workspace_id or settings.default_workspace_id
        qwen_config = settings.get_qwen_config()

        self.qwen_endpoint_url = qwen_config["endpoint_url"]
        self.qwen_endpoint_token = qwen_config["endpoint_token"]
        self.anthropic_url = "https://api.anthropic.com/v1"
        self.clip_model = "clip-vit-base-patch32"
        self.workspace_id = workspace_id
        self.supabase = get_supabase_client()

        # Initialize AI logger
        self.ai_logger = AICallLogger()

        # Use provided embedding service (with loaded models) or create new instance
        self._embeddings_service = embedding_service

        # Initialize Qwen endpoint manager for auto-resume/pause
        self.qwen_manager = QwenEndpointManager(
            endpoint_url=self.qwen_endpoint_url,
            endpoint_name=qwen_config["endpoint_name"],
            namespace=qwen_config["namespace"],
            endpoint_token=self.qwen_endpoint_token,
            enabled=qwen_config["enabled"]
        )

        # Load prompts from database
        self._load_prompts_from_database()

    def _load_prompts_from_database(self) -> None:
        """Load image analysis prompts from database.

        Loads:
        - Version 3: Vision model analysis prompt
        - Version 4: Claude Vision validation prompt
        """
        try:
            query = self.supabase.client.table('prompts')\
                .select('prompt_text, version')\
                .eq('prompt_type', 'extraction')\

            # Only filter by workspace_id if it's a valid non-None value
            if self.workspace_id and self.workspace_id != "None":
                query = query.eq('workspace_id', self.workspace_id)

            result = query\
                .eq('stage', 'image_analysis')\
                .eq('category', 'products')\
                .eq('is_custom', False)\
                .in_('version', [3, 4])\
                .execute()

            if result.data and len(result.data) > 0:
                self.vision_prompt = None
                self.claude_prompt = None

                for row in result.data:
                    version = row['version']
                    prompt = row['prompt_text']

                    if version == 3:
                        self.vision_prompt = prompt
                        logger.info("✅ Using DATABASE prompt for vision model analysis")
                    elif version == 4:
                        self.claude_prompt = prompt
                        logger.info("✅ Using DATABASE prompt for Claude Vision validation")

                # Log warnings if prompts not found - will error at runtime
                if not self.vision_prompt:
                    logger.warning("⚠️ Vision model prompt (version 3) not found in database. Add via /admin/ai-configs - extraction will fail!")
                if not self.claude_prompt:
                    logger.warning("⚠️ Claude Vision prompt (version 4) not found in database. Add via /admin/ai-configs - validation will fail!")
            else:
                logger.warning("⚠️ No image analysis prompts found in database. Add via /admin/ai-configs - extraction will fail!")
                self.vision_prompt = None
                self.claude_prompt = None

        except Exception as e:
            logger.error(f"❌ Failed to load prompts from database: {e}")
            self.vision_prompt = None
            self.claude_prompt = None

    async def analyze_image(
        self,
        image_url: str,
        image_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ImageAnalysisResult:
        """
        Perform real image analysis using vision models and CLIP.
        
        Args:
            image_url: URL to image
            image_id: Unique image identifier
            context: Optional context for analysis
            
        Returns:
            ImageAnalysisResult with real analysis data
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"🖼️ Starting real image analysis for {image_id}")
            
            # Step 1: Get image data
            image_data = await self._download_image(image_url)
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Step 2: Run parallel analysis
            vision_task = self._analyze_with_vision_model(image_base64, context)
            claude_task = self._analyze_with_claude(image_url, context)
            clip_task = self._generate_clip_embedding(image_base64)

            vision_result, claude_result, clip_embedding = await asyncio.gather(
                vision_task,
                claude_task,
                clip_task,
                return_exceptions=True
            )

            # Handle errors
            if isinstance(vision_result, Exception):
                self.logger.error(f"Vision model analysis failed: {vision_result}")
                vision_result = {"error": str(vision_result)}
            if isinstance(claude_result, Exception):
                self.logger.error(f"Claude analysis failed: {claude_result}")
                claude_result = {"error": str(claude_result)}
            if isinstance(clip_embedding, Exception):
                self.logger.error(f"CLIP embedding failed: {clip_embedding}")
                clip_embedding = []
            
            # Step 3: Extract material properties
            material_properties = self._extract_material_properties(
                vision_result,
                claude_result
            )

            # Step 4: Calculate real quality score
            quality_score = self._calculate_quality_score(
                vision_result,
                claude_result,
                clip_embedding,
                material_properties
            )

            # Step 5: Calculate confidence
            confidence_score = self._calculate_confidence(
                vision_result,
                claude_result,
                material_properties
            )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            result = ImageAnalysisResult(
                image_id=image_id,
                vision_analysis=vision_result,
                claude_validation=claude_result,
                clip_embedding=clip_embedding,
                material_properties=material_properties,
                quality_score=quality_score,
                confidence_score=confidence_score,
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow().isoformat()
            )
            
            self.logger.info(f"✅ Image analysis complete: quality={quality_score:.2f}, confidence={confidence_score:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Image analysis failed: {e}")
            raise
    
    async def analyze_image_from_base64(
        self,
        image_base64: str,
        image_id: str,
        context: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> ImageAnalysisResult:
        """
        Perform VISION-ONLY image analysis with quality scoring.

        NEW ARCHITECTURE (per user requirements):
        - Use ONLY vision model (configurable) for sync processing
        - Calculate quality score based on vision model confidence
        - Queue Claude validation ONLY if vision score < threshold (0.7)
        - Keep ALL 5 CLIP embeddings

        This prevents OOM crashes by removing dual-model sync processing.

        Args:
            image_base64: Base64-encoded image data
            image_id: Unique image identifier
            context: Optional context for analysis
            job_id: Optional job ID for tracking
            document_id: Optional document ID for queuing Claude validation

        Returns:
            ImageAnalysisResult with vision model analysis + CLIP embeddings
        """
        start_time = datetime.now()

        try:
            self.logger.info(f"🖼️ [VISION-ONLY] Starting image analysis for {image_id}")

            # Step 1: Run vision model + CLIP in parallel (NO CLAUDE)
            vision_task = self._analyze_with_vision_model(image_base64, context, job_id)
            clip_task = self._generate_clip_embedding(image_base64)

            vision_result, clip_embedding = await asyncio.gather(
                vision_task,
                clip_task,
                return_exceptions=True
            )

            # Handle errors
            if isinstance(vision_result, Exception):
                self.logger.error(f"Vision model analysis failed: {vision_result}")
                vision_result = {"error": str(vision_result), "confidence": 0.0}
            if isinstance(clip_embedding, Exception):
                self.logger.error(f"CLIP embedding failed: {clip_embedding}")
                clip_embedding = []

            # Step 2: Extract material properties from vision model ONLY
            material_properties = self._extract_material_properties_from_vision(vision_result)

            # Step 3: Calculate quality score based on vision model confidence
            vision_confidence = vision_result.get('confidence', 0.0)
            quality_score = self._calculate_vision_quality_score(
                vision_result,
                clip_embedding,
                material_properties
            )

            # Step 4: Check if Claude validation is needed
            CLAUDE_THRESHOLD = 0.7  # Queue Claude only if score < 0.7
            needs_claude_validation = quality_score < CLAUDE_THRESHOLD

            if needs_claude_validation:
                self.logger.warning(
                    f"⚠️ Image {image_id} has low quality score ({quality_score:.2f} < {CLAUDE_THRESHOLD}). "
                    f"Queuing for Claude validation."
                )

                # Queue for async Claude validation
                if document_id:
                    try:
                        from app.services.ai_validation.claude_validation_service import ClaudeValidationService
                        validation_service = ClaudeValidationService()
                        await validation_service.queue_image_for_validation(
                            image_id=image_id,
                            document_id=document_id,
                            vision_quality_score=quality_score,
                            priority=5
                        )
                    except Exception as e:
                        self.logger.error(f"❌ Failed to queue image for Claude validation: {e}")
            else:
                self.logger.info(
                    f"✅ Image {image_id} has good quality score ({quality_score:.2f} >= {CLAUDE_THRESHOLD}). "
                    f"No Claude validation needed."
                )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            result = ImageAnalysisResult(
                image_id=image_id,
                vision_analysis=vision_result,
                claude_validation=None,  # No Claude in sync processing
                clip_embedding=clip_embedding,
                material_properties=material_properties,
                quality_score=quality_score,
                confidence_score=vision_confidence,
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow().isoformat(),
                needs_claude_validation=needs_claude_validation  # New field
            )

            self.logger.info(
                f"✅ [VISION-ONLY] Image analysis complete: "
                f"quality={quality_score:.2f}, confidence={vision_confidence:.2f}, "
                f"needs_claude={needs_claude_validation}"
            )
            return result

        except Exception as e:
            self.logger.error(f"❌ Image analysis failed: {e}")
            raise

    async def _download_image(self, image_url: str) -> bytes:
        """Download image from URL"""
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=30.0)
            response.raise_for_status()
            return response.content
    


    async def _analyze_with_vision_model(
        self,
        image_base64: str,
        context: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze image with configurable vision model (default: Qwen3-VL-8B)"""
        start_time = time.time()
        try:
            if not self.qwen_endpoint_token:
                raise ValueError("HUGGINGFACE_API_KEY not set - cannot perform vision model analysis")

            # Use database prompt - NO FALLBACK
            if self.vision_prompt:
                prompt = self.vision_prompt
                logger.info("✅ Using DATABASE prompt for vision model analysis")
            else:
                error_msg = "CRITICAL: Vision model prompt not found in database. Add prompt via /admin/ai-configs with prompt_type='extraction', stage='image_analysis', category='vision_model'"
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            # Resume Qwen endpoint if needed (CRITICAL: Must be called before inference)
            # Wrapped in asyncio.to_thread() — resume_if_needed() is synchronous and
            # calls endpoint.resume().wait() which can block for up to 60-90 seconds.
            # Without to_thread(), this would freeze the entire async event loop,
            # stalling all concurrent image processing tasks.
            import asyncio
            if not await asyncio.to_thread(self.qwen_manager.resume_if_needed):
                self.logger.error("❌ Failed to resume Qwen endpoint - falling back to Claude")
                return await self._analyze_with_claude(image_base64, context, job_id)

            # ✅ FIX: Add retry logic for vision model empty responses
            max_retries = 3
            last_error = None

            for attempt in range(1, max_retries + 1):
                try:
                    # Use centralized httpx client
                    ai_service = get_ai_client_service()
                    response = await ai_service.httpx.post(
                            self.qwen_endpoint_url,
                            headers={
                                "Authorization": f"Bearer {self.qwen_endpoint_token}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": "Qwen/Qwen3-VL-8B-Instruct",  # Model name from Together AI
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": prompt
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                                }
                                            }
                                        ]
                                    }
                                ],
                                "max_tokens": 1024,
                                "temperature": 0.1,
                                "top_p": 0.9
                            }
                        )

                    if response.status_code != 200:
                        error_text = response.text

                        # CRITICAL FIX: Handle 5xx errors with retry logic
                        # Together.ai API occasionally returns 503/500 during high load or internal issues
                        if response.status_code in [500, 503]:
                            if attempt < max_retries:
                                error_name = "Internal Server Error" if response.status_code == 500 else "Service Unavailable"
                                self.logger.warning(f"Vision API {response.status_code} {error_name} (attempt {attempt}/{max_retries}), retrying with backoff...")
                                import asyncio
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            else:
                                self.logger.error(f"Vision API {response.status_code} after {max_retries} attempts: {error_text}")
                                raise RuntimeError(f"Vision API error {response.status_code} after {max_retries} attempts")

                        # CRITICAL FIX (MIVAA-8B): Handle 400 Input validation errors
                        # This happens when image is too large, corrupted, or unsupported format
                        if response.status_code == 400:
                            self.logger.warning(f"Vision API 400 Input validation error: {error_text[:200]}")
                            self.logger.warning(f"Image size: {len(image_base64)} bytes (base64)")

                            # Don't retry 400 errors - they won't succeed
                            # Fall back to rule-based analysis
                            import sentry_sdk
                            with sentry_sdk.push_scope() as scope:
                                scope.set_context("vision_validation_error", {
                                    "error": error_text[:500],
                                    "image_size_bytes": len(image_base64),
                                    "status_code": 400
                                })
                                sentry_sdk.capture_message(
                                    f"Vision API input validation error - falling back to rules",
                                    level="warning"
                                )

                            # Return None to trigger fallback to rule-based analysis
                            return None

                        # For other errors (4xx, etc.), log and raise immediately
                        self.logger.error(f"Vision API error {response.status_code}: {error_text}")
                        raise RuntimeError(f"Vision API returned error {response.status_code}: {error_text}")

                    result = response.json()
                    content = result["choices"][0]["message"]["content"]

                    # Parse JSON from response
                    try:
                        # Clean up response - remove markdown code blocks if present
                        content = content.strip()

                        # ROBUST JSON EXTRACTION - Handle all vision model response formats
                        # Format 1: "Here is the JSON:\n\n```json\n{...}\n```\n\nExplanation..."
                        # Format 2: "Here is the JSON:\n\n```\n{...}\n```\n\nExplanation..."
                        # Format 3: "```json\n{...}\n```"
                        # Format 4: "{...}" (pure JSON)

                        # Step 1: Extract from markdown code blocks if present
                        if "```" in content:
                            # Try ```json first (more specific)
                            if "```json" in content:
                                json_start = content.find("```json") + 7
                                json_end = content.find("```", json_start)
                                if json_end > json_start:
                                    content = content[json_start:json_end].strip()
                            else:
                                # Try generic ``` blocks
                                # Find first ``` and extract until next ```
                                first_backtick = content.find("```")
                                if first_backtick != -1:
                                    # Skip past the ``` and any language identifier (e.g., "json\n")
                                    json_start = first_backtick + 3
                                    # Skip any text on the same line as ``` (like "json")
                                    newline_after_backtick = content.find("\n", json_start)
                                    if newline_after_backtick != -1 and newline_after_backtick - json_start < 10:
                                        # If there's a newline within 10 chars, skip to it (handles "```json\n")
                                        json_start = newline_after_backtick + 1

                                    # Find closing ```
                                    json_end = content.find("```", json_start)
                                    if json_end > json_start:
                                        content = content[json_start:json_end].strip()

                        # Step 2: Find the actual JSON object boundaries
                        # Look for the first { and last }
                        first_brace = content.find('{')
                        last_brace = content.rfind('}')

                        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                            # Extract only the JSON portion
                            json_text = content[first_brace:last_brace + 1]
                        else:
                            # No braces found, try parsing as-is
                            json_text = content

                        # Step 3: Clean up any remaining issues
                        json_text = json_text.strip()

                        # Check if content is empty
                        if not json_text:
                            error_msg = f"Vision model returned empty response (attempt {attempt}/{max_retries})"
                            self.logger.warning(error_msg)
                            if attempt < max_retries:
                                # Exponential backoff: 2^attempt seconds
                                import asyncio
                                await asyncio.sleep(2 ** attempt)
                                continue
                            else:
                                raise RuntimeError(f"Vision model returned empty response after {max_retries} attempts")

                        # Step 3.5: Sanitize common vision model JSON mistakes
                        # Fix: ["item1", "item2", or "item3"] -> ["item1", "item2", "item3"]
                        # This handles cases where model puts unquoted words like "or" or "and" in arrays
                        import re

                        # Pattern 1: Remove unquoted "or" or "and" between array items
                        # Matches: ", or " or ", and " (with optional extra spaces)
                        # Example: ["vinyl", "linoleum", or " rubber"] -> ["vinyl", "linoleum", " rubber"]
                        json_text = re.sub(r'",\s+(or|and)\s+"', '", "', json_text)

                        # Pattern 2: Handle case where conjunction appears after comma without space
                        # Matches: ",or " or ",and "
                        json_text = re.sub(r'",(or|and)\s+"', '", "', json_text)

                        # Pattern 3: Handle case with space before comma
                        # Matches: " , or " or " , and "
                        json_text = re.sub(r'"\s*,\s*(or|and)\s+"', '", "', json_text)

                        # Step 4: Parse JSON
                        analysis = json.loads(json_text)

                        self.logger.info(f"✅ Vision model analysis successful on attempt {attempt}")

                        # Log AI call
                        latency_ms = int((time.time() - start_time) * 1000)
                        usage = result.get("usage", {})
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)

                        confidence_breakdown = {
                            "model_confidence": 0.90,
                            "completeness": analysis.get("confidence", 0.85),
                            "consistency": 0.88,
                            "validation": 0.80
                        }
                        confidence_score = (
                            0.30 * confidence_breakdown["model_confidence"] +
                            0.30 * confidence_breakdown["completeness"] +
                            0.25 * confidence_breakdown["consistency"] +
                            0.15 * confidence_breakdown["validation"]
                        )

                        await self.ai_logger.log_qwen_call(
                            task="image_vision_analysis",
                            model="qwen3-vl-8b",
                            response=result,
                            latency_ms=latency_ms,
                            confidence_score=confidence_score,
                            confidence_breakdown=confidence_breakdown,
                            action="use_ai_result",
                            job_id=job_id
                        )

                        return {
                            "model": "qwen3-vl-8b-vision",
                            "analysis": analysis,
                            "success": True
                        }
                    except json.JSONDecodeError as e:
                        # CRITICAL FIX (MIVAA-7Y, MIVAA-87): Enhanced JSON parsing with multiple fallback strategies
                        error_msg = f"Failed to parse vision model response as JSON (attempt {attempt}/{max_retries}): {e}"
                        self.logger.warning(error_msg)
                        self.logger.warning(f"Extracted JSON text (first 300 chars): {json_text[:300]}")
                        self.logger.warning(f"Original content (first 300 chars): {result['choices'][0]['message']['content'][:300]}")

                        # FALLBACK STRATEGY 1: Try to fix common JSON issues
                        try:
                            # Fix trailing commas
                            fixed_json = re.sub(r',\s*}', '}', json_text)
                            fixed_json = re.sub(r',\s*]', ']', fixed_json)
                            # Fix single quotes to double quotes
                            fixed_json = fixed_json.replace("'", '"')
                            # Try parsing fixed JSON
                            analysis = json.loads(fixed_json)
                            self.logger.info(f"✅ Qwen analysis successful after JSON repair on attempt {attempt}")

                            # Log AI call
                            latency_ms = int((time.time() - start_time) * 1000)
                            usage = result.get("usage", {})
                            input_tokens = usage.get("prompt_tokens", 0)
                            output_tokens = usage.get("completion_tokens", 0)

                            confidence_breakdown = {
                                "model_confidence": 0.85,
                                "completeness": 0.80,
                                "consistency": 0.75,
                                "validation": 0.70
                            }

                            await self.ai_logger.log_ai_call(
                                task="image_vision_analysis",
                                model="qwen3-vl-8b-vision",
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                cost=(input_tokens * 0.0000002) + (output_tokens * 0.0000006),
                                latency_ms=latency_ms,
                                confidence_score=0.75,
                                confidence_breakdown=confidence_breakdown,
                                action="use_ai_result_after_repair",
                                job_id=job_id
                            )

                            return {
                                "model": "qwen3-vl-8b-vision",
                                "analysis": analysis,
                                "success": True
                            }
                        except json.JSONDecodeError:
                            pass  # Continue to retry logic

                        if attempt < max_retries:
                            import asyncio
                            await asyncio.sleep(2 ** attempt)
                            continue
                        else:
                            # Log full details on final failure
                            self.logger.error(f"JSON parsing failed after {max_retries} attempts")
                            self.logger.error(f"Extracted JSON text: {json_text}")
                            self.logger.error(f"Parse error: {e}")

                            # Send to Sentry with full context
                            import sentry_sdk
                            with sentry_sdk.push_scope() as scope:
                                scope.set_context("json_parsing_error", {
                                    "extracted_json": json_text[:500],
                                    "original_content": result['choices'][0]['message']['content'][:500],
                                    "parse_error": str(e),
                                    "attempts": max_retries
                                })
                                sentry_sdk.capture_message(
                                    f"Vision model JSON parsing failed after {max_retries} attempts: {e}",
                                    level="error"
                                )

                            raise RuntimeError(f"Vision model returned invalid JSON after {max_retries} attempts: {e}")

                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        self.logger.warning(f"Vision model attempt {attempt}/{max_retries} failed: {e}, retrying...")
                        import asyncio
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        raise

            # If we get here, all retries failed
            if last_error:
                raise last_error

        except Exception as e:
            self.logger.error(f"Vision model analysis failed: {e}")

            # Log failed AI call
            latency_ms = int((time.time() - start_time) * 1000)
            await self.ai_logger.log_ai_call(
                task="image_vision_analysis",
                model="qwen3-vl-8b",  # Actual Together AI model identifier
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
                fallback_reason=f"Vision API error: {str(e)}",
                error_message=str(e)
            )

            raise RuntimeError(f"Vision model analysis failed: {str(e)}") from e
        # NOTE: Removed between-batch auto_pause - endpoints pause only at full job completion

    async def _analyze_with_claude(
        self,
        image_url: str,
        context: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze image with Claude 4.5 Sonnet Vision"""
        start_time = time.time()
        try:
            # Use centralized AI client service
            ai_service = get_ai_client_service()
            client = ai_service.anthropic

            # Use database prompt - NO FALLBACK
            if self.claude_prompt:
                prompt = self.claude_prompt
                logger.info("✅ Using DATABASE prompt for Claude Vision validation")
            else:
                error_msg = "CRITICAL: Claude Vision prompt not found in database. Add prompt via /admin/ai-configs with prompt_type='extraction', stage='image_analysis', category='products', version=4"
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            # Call Claude Vision API
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": image_url
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            content = response.content[0].text.strip()

            try:
                # CRITICAL FIX (MIVAA-87): Enhanced JSON extraction with multiple strategies
                # Strategy 1: Find JSON between first { and last }
                first_brace = content.find('{')
                last_brace = content.rfind('}')

                if first_brace != -1 and last_brace != -1:
                    json_text = content[first_brace:last_brace + 1]

                    try:
                        validation = json.loads(json_text)
                    except json.JSONDecodeError as e:
                        # Strategy 2: Try to fix common JSON issues
                        self.logger.warning(f"Initial JSON parse failed, attempting repair: {e}")

                        # Fix trailing commas
                        fixed_json = re.sub(r',\s*}', '}', json_text)
                        fixed_json = re.sub(r',\s*]', ']', fixed_json)
                        # Fix single quotes to double quotes (but not in strings)
                        fixed_json = re.sub(r"(?<!\\)'", '"', fixed_json)
                        # Remove comments
                        fixed_json = re.sub(r'//.*?\n', '\n', fixed_json)
                        fixed_json = re.sub(r'/\*.*?\*/', '', fixed_json, flags=re.DOTALL)

                        try:
                            validation = json.loads(fixed_json)
                            self.logger.info("✅ Claude JSON repaired successfully")
                        except json.JSONDecodeError as e2:
                            # Strategy 3: Extract JSON using regex
                            self.logger.warning(f"JSON repair failed, trying regex extraction: {e2}")
                            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                            if json_match:
                                validation = json.loads(json_match.group(0))
                                self.logger.info("✅ Claude JSON extracted via regex")
                            else:
                                raise e2
                else:
                    # No JSON found, raise error
                    raise json.JSONDecodeError("No JSON object found", content, 0)

                # Log AI call
                latency_ms = int((time.time() - start_time) * 1000)
                confidence_breakdown = {
                    "model_confidence": 0.95,
                    "completeness": validation.get("confidence", 0.90),
                    "consistency": 0.93,
                    "validation": 0.88
                }
                confidence_score = (
                    0.30 * confidence_breakdown["model_confidence"] +
                    0.30 * confidence_breakdown["completeness"] +
                    0.25 * confidence_breakdown["consistency"] +
                    0.15 * confidence_breakdown["validation"]
                )

                await self.ai_logger.log_claude_call(
                    task="image_vision_validation",
                    model="claude-sonnet-4-5-20250929",
                    response=response,
                    latency_ms=latency_ms,
                    confidence_score=confidence_score,
                    confidence_breakdown=confidence_breakdown,
                    action="use_ai_result",
                    job_id=job_id
                )

                return {
                    "model": "claude-sonnet-4-5-20250929",
                    "validation": validation,
                    "success": True
                }
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse Claude response as JSON: {e}")
                self.logger.debug(f"Raw response (first 500 chars): {content[:500]}")

                # Send to Sentry with full context
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_context("claude_json_parsing_error", {
                        "raw_content": content[:500],
                        "parse_error": str(e),
                        "error_position": f"line {e.lineno} column {e.colno}" if hasattr(e, 'lineno') else "unknown"
                    })
                    sentry_sdk.capture_message(
                        f"Claude JSON parsing failed: {e}",
                        level="error"
                    )

                raise RuntimeError(f"Claude returned invalid JSON: {e}")

        except Exception as e:
            self.logger.error(f"Claude analysis failed: {e}")

            # Log failed AI call
            latency_ms = int((time.time() - start_time) * 1000)
            await self.ai_logger.log_ai_call(
                task="image_vision_validation",
                model="claude-sonnet-4-5-20250929",
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
                fallback_reason=f"Claude API error: {str(e)}",
                error_message=str(e)
            )

            raise RuntimeError(f"Claude 4.5 Sonnet Vision analysis failed: {str(e)}") from e

    async def _analyze_with_claude_base64(
        self,
        image_base64: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze image with Claude 4.5 Sonnet Vision from base64 data"""
        try:
            # Use centralized AI client service
            ai_service = get_ai_client_service()
            client = ai_service.anthropic

            # Use database prompt - NO FALLBACK
            if self.claude_prompt:
                prompt = self.claude_prompt
                logger.info("✅ Using DATABASE prompt for Claude Vision validation (base64)")
            else:
                error_msg = "CRITICAL: Claude Vision prompt not found in database. Add prompt via /admin/ai-configs with prompt_type='extraction', stage='image_analysis', category='products', version=4"
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            # Call Claude Vision API with base64 image
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            content = response.content[0].text.strip()

            try:
                # Handle Claude's tendency to add extra text after JSON
                # Find the last closing brace to extract only the JSON portion
                last_brace = content.rfind('}')
                if last_brace != -1:
                    json_text = content[:last_brace + 1]
                    validation = json.loads(json_text)
                else:
                    # No JSON found, raise error
                    raise json.JSONDecodeError("No JSON object found", content, 0)

                return {
                    "model": "claude-sonnet-4-5-20250929",
                    "validation": validation,
                    "success": True
                }
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse Claude response as JSON: {e}")
                self.logger.debug(f"Raw response (first 500 chars): {content[:500]}")
                raise RuntimeError(f"Claude returned invalid JSON: {e}")

        except Exception as e:
            self.logger.error(f"Claude analysis failed: {e}")
            raise RuntimeError(f"Claude 4.5 Sonnet Vision analysis failed: {str(e)}") from e

    async def _generate_clip_embedding(self, image_base64: str) -> List[float]:
        """Generate SigLIP embedding for image using RealEmbeddingsService"""
        try:
            # Use RealEmbeddingsService directly instead of HTTP call
            from .real_embeddings_service import RealEmbeddingsService

            # Create new instance only if not provided in constructor
            if self._embeddings_service is None:
                self._embeddings_service = RealEmbeddingsService()

            # Generate visual embedding using SigLIP
            # Returns tuple: (embedding_list, model_name, pil_image)
            result = await self._embeddings_service._generate_visual_embedding(
                image_url=None,
                image_data=image_base64
            )

            # Unpack tuple: (embedding_list, model_name, pil_image)
            if result and isinstance(result, tuple) and len(result) == 3:
                visual_embedding, model_name, pil_image = result

                # Close PIL image if returned (we don't need it here)
                if pil_image and hasattr(pil_image, 'close'):
                    try:
                        pil_image.close()
                    except:
                        pass

                # SigLIP returns 1152D embeddings
                if visual_embedding and isinstance(visual_embedding, list) and len(visual_embedding) == 1152:
                    self.logger.info(f"✅ Generated SigLIP embedding: {len(visual_embedding)}D using {model_name}")
                    return visual_embedding
                else:
                    actual_len = len(visual_embedding) if visual_embedding else 0
                    self.logger.error(f"SigLIP embedding has wrong dimensions: expected 1152D, got {actual_len}D from {model_name}")
                    raise RuntimeError(f"Failed to generate valid SigLIP embedding: expected 1152D, got {actual_len}D")
            else:
                self.logger.error(f"SigLIP embedding generation returned invalid format: expected tuple(list, str, PIL), got {type(result).__name__}")
                raise RuntimeError("Failed to generate valid SigLIP embedding: invalid return format")

        except Exception as e:
            self.logger.error(f"❌ SigLIP embedding generation failed: {e}")
            raise RuntimeError(f"CLIP embedding generation failed: {str(e)}") from e

    def _extract_material_properties(
        self,
        vision_result: Dict[str, Any],
        claude_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract material properties from analysis results"""
        properties = {
            "color": None,
            "finish": None,
            "pattern": None,
            "texture": None,
            "composition": None,
            "confidence": 0.0
        }

        # Extract from vision model analysis
        if vision_result.get("success") and vision_result.get("analysis"):
            analysis = vision_result["analysis"]
            properties["color"] = analysis.get("colors", [None])[0] if analysis.get("colors") else None
            properties["texture"] = analysis.get("textures", [None])[0] if analysis.get("textures") else None
            properties["composition"] = analysis.get("properties", {}).get("composition")
            properties["finish"] = analysis.get("properties", {}).get("finish")
            properties["pattern"] = analysis.get("properties", {}).get("pattern")
            properties["confidence"] = analysis.get("confidence", 0.0)

        # Enhance with Claude validation
        if claude_result.get("success") and claude_result.get("validation"):
            validation = claude_result["validation"]
            if not properties["color"] and validation.get("visual_properties", {}).get("color_palette"):
                properties["color"] = validation["visual_properties"]["color_palette"][0]
            if not properties["finish"] and validation.get("visual_properties", {}).get("surface_finish"):
                properties["finish"] = validation["visual_properties"]["surface_finish"]
            if not properties["pattern"] and validation.get("visual_properties", {}).get("pattern_type"):
                properties["pattern"] = validation["visual_properties"]["pattern_type"]

            # Update confidence with Claude's assessment
            claude_confidence = validation.get("confidence", 0.0)
            if claude_confidence > properties["confidence"]:
                properties["confidence"] = claude_confidence

        return properties

    def _extract_material_properties_from_vision(
        self,
        vision_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract material properties from vision model analysis ONLY.

        This is the method for vision-only processing (no Claude).
        """
        properties = {
            "color": None,
            "finish": None,
            "pattern": None,
            "texture": None,
            "composition": None,
            "confidence": 0.0
        }

        # Extract from vision model analysis
        if vision_result.get("success") and vision_result.get("analysis"):
            analysis = vision_result["analysis"]
            properties["color"] = analysis.get("colors", [None])[0] if analysis.get("colors") else None
            properties["texture"] = analysis.get("textures", [None])[0] if analysis.get("textures") else None
            properties["composition"] = analysis.get("properties", {}).get("composition")
            properties["finish"] = analysis.get("properties", {}).get("finish")
            properties["pattern"] = analysis.get("properties", {}).get("pattern")
            properties["confidence"] = analysis.get("confidence", 0.0)
        elif "error" not in vision_result:
            # If vision model succeeded but no analysis field, try direct access
            properties["color"] = vision_result.get("colors", [None])[0] if vision_result.get("colors") else None
            properties["texture"] = vision_result.get("textures", [None])[0] if vision_result.get("textures") else None
            properties["composition"] = vision_result.get("properties", {}).get("composition")
            properties["finish"] = vision_result.get("properties", {}).get("finish")
            properties["pattern"] = vision_result.get("properties", {}).get("pattern")
            properties["confidence"] = vision_result.get("confidence", 0.0)

        return properties

    def _calculate_vision_quality_score(
        self,
        vision_result: Dict[str, Any],
        clip_embedding: List[float],
        material_properties: Dict[str, Any]
    ) -> float:
        """
        Calculate quality score based on vision model analysis ONLY.

        NEW SCORING (Vision-only):
        - Vision model confidence: 60% weight
        - Material properties completeness: 30% weight
        - CLIP embedding validity: 10% weight

        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        weight_count = 0

        # Vision model confidence (60% weight)
        if vision_result.get("success"):
            vision_conf = vision_result.get("analysis", {}).get("confidence", 0.0)
            if vision_conf == 0.0:
                # Try direct access
                vision_conf = vision_result.get("confidence", 0.0)
            score += vision_conf * 0.6
            weight_count += 0.6
        elif "error" not in vision_result:
            # If no explicit success field, try to get confidence directly
            vision_conf = vision_result.get("confidence", 0.0)
            if vision_conf > 0:
                score += vision_conf * 0.6
                weight_count += 0.6

        # Material properties completeness (30% weight)
        if material_properties:
            properties_filled = sum(1 for v in material_properties.values() if v is not None and v != 0.0)
            properties_score = properties_filled / 6.0  # 6 properties total
            score += properties_score * 0.3
            weight_count += 0.3

        # CLIP embedding validity (10% weight)
        if clip_embedding and len(clip_embedding) > 0:
            # Check if embedding is not all zeros
            non_zero_count = sum(1 for v in clip_embedding if abs(v) > 0.001)
            if non_zero_count > len(clip_embedding) * 0.1:  # At least 10% non-zero
                score += 1.0 * 0.1
                weight_count += 0.1

        # Normalize score
        if weight_count > 0:
            return min(1.0, score / weight_count)

        return 0.5  # Default score if no data available

    def _calculate_quality_score(
        self,
        vision_result: Dict[str, Any],
        claude_result: Dict[str, Any],
        clip_embedding: List[float],
        material_properties: Dict[str, Any]
    ) -> float:
        """Calculate real quality score based on analysis results"""
        score = 0.0
        weight_count = 0

        # Vision model confidence (40% weight)
        if vision_result.get("success"):
            vision_conf = vision_result.get("analysis", {}).get("confidence", 0.0)
            score += vision_conf * 0.4
            weight_count += 0.4

        # Claude quality assessment (40% weight)
        if claude_result.get("success"):
            quality_assessment = claude_result.get("validation", {}).get("quality_assessment", {})
            overall_quality = quality_assessment.get("overall_quality", 0.0)
            score += overall_quality * 0.4
            weight_count += 0.4

        # Material properties completeness (20% weight)
        if material_properties:
            properties_filled = sum(1 for v in material_properties.values() if v is not None and v != 0.0)
            properties_score = properties_filled / 6.0  # 6 properties total
            score += properties_score * 0.2
            weight_count += 0.2

        # Normalize score
        if weight_count > 0:
            return min(1.0, score / weight_count)

        return 0.5

    def _calculate_confidence(
        self,
        vision_result: Dict[str, Any],
        claude_result: Dict[str, Any],
        material_properties: Dict[str, Any]
    ) -> float:
        """Calculate confidence score based on model agreement"""
        confidences = []

        # Vision model confidence
        if vision_result.get("success"):
            confidences.append(vision_result.get("analysis", {}).get("confidence", 0.0))

        # Claude confidence
        if claude_result.get("success"):
            confidences.append(claude_result.get("validation", {}).get("confidence", 0.0))

        # Material properties confidence
        if material_properties:
            confidences.append(material_properties.get("confidence", 0.0))

        # Return average confidence
        if confidences:
            return sum(confidences) / len(confidences)

        return 0.5


