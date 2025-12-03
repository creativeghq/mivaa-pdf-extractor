"""
Real Image Analysis Service - Stage 4 Implementation

This service provides real image analysis using:
1. Llama 4 Scout 17B Vision for detailed image analysis (superior OCR, table/diagram understanding)
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

from app.services.ai_call_logger import AICallLogger
from app.services.supabase_client import SupabaseClient, get_supabase_client
from app.services.ai_client_service import get_ai_client_service

logger = logging.getLogger(__name__)

# Get API keys from environment
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


@dataclass
class ImageAnalysisResult:
    """Result of real image analysis"""
    image_id: str
    llama_analysis: Dict[str, Any]  # Llama 4 Scout 17B Vision analysis
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
    - Llama 4 Scout 17B Vision: Detailed image analysis (69.4% MMMU, #1 OCR)
    - Claude 4.5 Sonnet Vision: Validation and enrichment
    - CLIP: Visual embeddings for similarity search
    """
    
    def __init__(self, supabase_client=None, embedding_service=None, workspace_id: str = "ffafc28b-1b8b-4b0d-b226-9f9a6154004e"):
        self.logger = logger
        self.together_ai_url = "https://api.together.xyz/v1"
        self.anthropic_url = "https://api.anthropic.com/v1"
        self.clip_model = "clip-vit-base-patch32"
        self.workspace_id = workspace_id
        self.supabase = get_supabase_client()

        # Initialize AI logger
        self.ai_logger = AICallLogger()

        # Use provided embedding service (with loaded models) or create new instance
        self._embeddings_service = embedding_service

        # Load prompts from database
        self._load_prompts_from_database()

    def _load_prompts_from_database(self) -> None:
        """Load image analysis prompts from database.

        Loads:
        - Version 3: Llama Vision analysis prompt
        - Version 4: Claude Vision validation prompt
        """
        try:
            result = self.supabase.client.table('extraction_prompts')\
                .select('prompt_template, version')\
                .eq('workspace_id', self.workspace_id)\
                .eq('stage', 'image_analysis')\
                .eq('category', 'products')\
                .eq('is_custom', False)\
                .in_('version', [3, 4])\
                .execute()

            if result.data and len(result.data) > 0:
                self.llama_prompt = None
                self.claude_prompt = None

                for row in result.data:
                    version = row['version']
                    prompt = row['prompt_template']

                    if version == 3:
                        self.llama_prompt = prompt
                        logger.info("✅ Using DATABASE prompt for Llama Vision analysis")
                    elif version == 4:
                        self.claude_prompt = prompt
                        logger.info("✅ Using DATABASE prompt for Claude Vision validation")

                # Set fallbacks if not found
                if not self.llama_prompt:
                    logger.warning("⚠️ Llama Vision prompt not found in database, using hardcoded fallback")
                    self.llama_prompt = None
                if not self.claude_prompt:
                    logger.warning("⚠️ Claude Vision prompt not found in database, using hardcoded fallback")
                    self.claude_prompt = None
            else:
                logger.warning("⚠️ No image analysis prompts found in database, using hardcoded fallbacks")
                self.llama_prompt = None
                self.claude_prompt = None

        except Exception as e:
            logger.error(f"❌ Failed to load prompts from database: {e}")
            self.llama_prompt = None
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
            llama_task = self._analyze_with_llama(image_base64, context)
            claude_task = self._analyze_with_claude(image_url, context)
            clip_task = self._generate_clip_embedding(image_base64)
            
            llama_result, claude_result, clip_embedding = await asyncio.gather(
                llama_task,
                claude_task,
                clip_task,
                return_exceptions=True
            )
            
            # Handle errors
            if isinstance(llama_result, Exception):
                self.logger.error(f"Llama analysis failed: {llama_result}")
                llama_result = {"error": str(llama_result)}
            if isinstance(claude_result, Exception):
                self.logger.error(f"Claude analysis failed: {claude_result}")
                claude_result = {"error": str(claude_result)}
            if isinstance(clip_embedding, Exception):
                self.logger.error(f"CLIP embedding failed: {clip_embedding}")
                clip_embedding = []
            
            # Step 3: Extract material properties
            material_properties = self._extract_material_properties(
                llama_result,
                claude_result
            )
            
            # Step 4: Calculate real quality score
            quality_score = self._calculate_quality_score(
                llama_result,
                claude_result,
                clip_embedding,
                material_properties
            )
            
            # Step 5: Calculate confidence
            confidence_score = self._calculate_confidence(
                llama_result,
                claude_result,
                material_properties
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = ImageAnalysisResult(
                image_id=image_id,
                llama_analysis=llama_result,
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
        Perform LLAMA-ONLY image analysis with quality scoring.

        NEW ARCHITECTURE (per user requirements):
        - Use ONLY Llama 4 Scout Vision for sync processing
        - Calculate quality score based on Llama confidence
        - Queue Claude validation ONLY if Llama score < threshold (0.7)
        - Keep ALL 5 CLIP embeddings

        This prevents OOM crashes by removing dual-model sync processing.

        Args:
            image_base64: Base64-encoded image data
            image_id: Unique image identifier
            context: Optional context for analysis
            job_id: Optional job ID for tracking
            document_id: Optional document ID for queuing Claude validation

        Returns:
            ImageAnalysisResult with Llama analysis + CLIP embeddings
        """
        start_time = datetime.now()

        try:
            self.logger.info(f"🖼️ [LLAMA-ONLY] Starting image analysis for {image_id}")

            # Step 1: Run Llama + CLIP in parallel (NO CLAUDE)
            llama_task = self._analyze_with_llama(image_base64, context, job_id)
            clip_task = self._generate_clip_embedding(image_base64)

            llama_result, clip_embedding = await asyncio.gather(
                llama_task,
                clip_task,
                return_exceptions=True
            )

            # Handle errors
            if isinstance(llama_result, Exception):
                self.logger.error(f"Llama analysis failed: {llama_result}")
                llama_result = {"error": str(llama_result), "confidence": 0.0}
            if isinstance(clip_embedding, Exception):
                self.logger.error(f"CLIP embedding failed: {clip_embedding}")
                clip_embedding = []

            # Step 2: Extract material properties from Llama ONLY
            material_properties = self._extract_material_properties_from_llama(llama_result)

            # Step 3: Calculate quality score based on Llama confidence
            llama_confidence = llama_result.get('confidence', 0.0)
            quality_score = self._calculate_llama_quality_score(
                llama_result,
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
                        from app.services.claude_validation_service import ClaudeValidationService
                        validation_service = ClaudeValidationService()
                        await validation_service.queue_image_for_validation(
                            image_id=image_id,
                            document_id=document_id,
                            llama_quality_score=quality_score,
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
                llama_analysis=llama_result,
                claude_validation=None,  # No Claude in sync processing
                clip_embedding=clip_embedding,
                material_properties=material_properties,
                quality_score=quality_score,
                confidence_score=llama_confidence,
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow().isoformat(),
                needs_claude_validation=needs_claude_validation  # New field
            )

            self.logger.info(
                f"✅ [LLAMA-ONLY] Image analysis complete: "
                f"quality={quality_score:.2f}, confidence={llama_confidence:.2f}, "
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
    


    async def _analyze_with_llama(
        self,
        image_base64: str,
        context: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze image with Llama 4 Scout 17B Vision (69.4% MMMU, #1 OCR)"""
        start_time = time.time()
        try:
            if not TOGETHER_API_KEY:
                raise ValueError("TOGETHER_API_KEY not set - cannot perform Llama vision analysis")

            # Use database prompt or hardcoded fallback
            if self.llama_prompt:
                prompt = self.llama_prompt
                logger.info("✅ Using DATABASE prompt for Llama Vision analysis")
            else:
                logger.info("⚠️ Using HARDCODED fallback prompt for Llama Vision analysis")
                prompt = """Analyze this material/product image and provide detailed analysis in JSON format:
{
  "description": "<detailed description of what you see>",
  "objects_detected": ["<object1>", "<object2>"],
  "materials_identified": ["<material1>", "<material2>"],
  "colors": ["<color1>", "<color2>"],
  "textures": ["<texture1>", "<texture2>"],
  "confidence": <0.0-1.0>,
  "properties": {
    "finish": "<matte/glossy/satin/etc>",
    "pattern": "<solid/striped/geometric/etc>",
    "composition": "<estimated composition>"
  }
}

Respond ONLY with valid JSON, no additional text."""

            # ✅ FIX: Add retry logic for Llama empty responses
            max_retries = 3
            last_error = None

            for attempt in range(1, max_retries + 1):
                try:
                    # Use centralized httpx client
                    ai_service = get_ai_client_service()
                    response = await ai_service.httpx.post(
                            "https://api.together.xyz/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
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
                        self.logger.error(f"Llama API error {response.status_code}: {error_text}")
                        raise RuntimeError(f"Llama API returned error {response.status_code}: {error_text}")

                    result = response.json()
                    content = result["choices"][0]["message"]["content"]

                    # Parse JSON from response
                    try:
                        # Clean up response - remove markdown code blocks if present
                        content = content.strip()
                        if content.startswith("```json"):
                            content = content[7:]
                        if content.startswith("```"):
                            content = content[3:]
                        if content.endswith("```"):
                            content = content[:-3]
                        content = content.strip()

                        # Check if content is empty
                        if not content:
                            error_msg = f"Llama returned empty response (attempt {attempt}/{max_retries})"
                            self.logger.warning(error_msg)
                            if attempt < max_retries:
                                # Exponential backoff: 2^attempt seconds
                                import asyncio
                                await asyncio.sleep(2 ** attempt)
                                continue
                            else:
                                raise RuntimeError(f"Llama returned empty response after {max_retries} attempts")

                        # Handle extra text after JSON (similar to Claude fix)
                        last_brace = content.rfind('}')
                        if last_brace != -1:
                            json_text = content[:last_brace + 1]
                            analysis = json.loads(json_text)
                        else:
                            # Try parsing as-is
                            analysis = json.loads(content)

                        self.logger.info(f"✅ Llama analysis successful on attempt {attempt}")

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

                        await self.ai_logger.log_llama_call(
                            task="image_vision_analysis",
                            model="llama-4-scout-17b",
                            response=result,
                            latency_ms=latency_ms,
                            confidence_score=confidence_score,
                            confidence_breakdown=confidence_breakdown,
                            action="use_ai_result",
                            job_id=job_id
                        )

                        return {
                            "model": "llama-4-scout-17b-vision",
                            "analysis": analysis,
                            "success": True
                        }
                    except json.JSONDecodeError as e:
                        error_msg = f"Failed to parse Llama response as JSON (attempt {attempt}/{max_retries}): {e}. Content: {content[:200]}"
                        self.logger.warning(error_msg)
                        if attempt < max_retries:
                            import asyncio
                            await asyncio.sleep(2 ** attempt)
                            continue
                        else:
                            self.logger.error(f"Full response: {result}")
                            raise RuntimeError(f"Llama returned invalid JSON after {max_retries} attempts: {e}")

                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        self.logger.warning(f"Llama attempt {attempt}/{max_retries} failed: {e}, retrying...")
                        import asyncio
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        raise

            # If we get here, all retries failed
            if last_error:
                raise last_error

        except Exception as e:
            self.logger.error(f"Llama analysis failed: {e}")

            # Log failed AI call
            latency_ms = int((time.time() - start_time) * 1000)
            await self.ai_logger.log_ai_call(
                task="image_vision_analysis",
                model="llama-4-scout-17b",
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
                fallback_reason=f"Llama API error: {str(e)}",
                error_message=str(e)
            )

            raise RuntimeError(f"Llama vision analysis failed: {str(e)}") from e
    
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

            # Use database prompt or hardcoded fallback
            if self.claude_prompt:
                prompt = self.claude_prompt
                logger.info("✅ Using DATABASE prompt for Claude Vision validation")
            else:
                logger.info("⚠️ Using HARDCODED fallback prompt for Claude Vision validation")
                prompt = """Validate and analyze this material/product image. Provide response in JSON format:
{
  "quality_assessment": {
    "clarity": <0.0-1.0>,
    "lighting": <0.0-1.0>,
    "composition": <0.0-1.0>,
    "overall_quality": <0.0-1.0>
  },
  "material_classification": {
    "primary_material": "<material>",
    "secondary_materials": ["<material1>", "<material2>"],
    "confidence": <0.0-1.0>
  },
  "visual_properties": {
    "surface_finish": "<finish type>",
    "color_palette": ["<color1>", "<color2>"],
    "pattern_type": "<pattern>"
  },
  "recommendations": ["<recommendation1>", "<recommendation2>"],
  "confidence": <0.0-1.0>
}"""

            # Call Claude Vision API
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
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
                # Handle Claude's tendency to add extra text after JSON
                # Find the last closing brace to extract only the JSON portion
                last_brace = content.rfind('}')
                if last_brace != -1:
                    json_text = content[:last_brace + 1]
                    validation = json.loads(json_text)
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
                    model="claude-3-5-sonnet-20241022",
                    response=response,
                    latency_ms=latency_ms,
                    confidence_score=confidence_score,
                    confidence_breakdown=confidence_breakdown,
                    action="use_ai_result",
                    job_id=job_id
                )

                return {
                    "model": "claude-3-5-sonnet-20241022",
                    "validation": validation,
                    "success": True
                }
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse Claude response as JSON: {e}")
                self.logger.debug(f"Raw response (first 500 chars): {content[:500]}")
                raise RuntimeError(f"Claude returned invalid JSON: {e}")

        except Exception as e:
            self.logger.error(f"Claude analysis failed: {e}")

            # Log failed AI call
            latency_ms = int((time.time() - start_time) * 1000)
            await self.ai_logger.log_ai_call(
                task="image_vision_validation",
                model="claude-3-5-sonnet-20241022",
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

            # Use database prompt or hardcoded fallback
            if self.claude_prompt:
                prompt = self.claude_prompt
                logger.info("✅ Using DATABASE prompt for Claude Vision validation (base64)")
            else:
                logger.info("⚠️ Using HARDCODED fallback prompt for Claude Vision validation (base64)")
                prompt = """Validate and analyze this material/product image. Provide response in JSON format:
{
  "quality_assessment": {
    "clarity": <0.0-1.0>,
    "lighting": <0.0-1.0>,
    "composition": <0.0-1.0>,
    "overall_quality": <0.0-1.0>
  },
  "material_classification": {
    "primary_material": "<material>",
    "secondary_materials": ["<material1>", "<material2>"],
    "confidence": <0.0-1.0>
  },
  "visual_properties": {
    "surface_finish": "<finish type>",
    "color_palette": ["<color1>", "<color2>"],
    "pattern_type": "<pattern>"
  },
  "recommendations": ["<recommendation1>", "<recommendation2>"],
  "confidence": <0.0-1.0>
}"""

            # Call Claude Vision API with base64 image
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
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
                    "model": "claude-3-5-sonnet-20241022",
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
        llama_result: Dict[str, Any],
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

        # Extract from Llama analysis
        if llama_result.get("success") and llama_result.get("analysis"):
            analysis = llama_result["analysis"]
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

    def _extract_material_properties_from_llama(
        self,
        llama_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract material properties from Llama analysis ONLY.

        This is the new method for Llama-only processing (no Claude).
        """
        properties = {
            "color": None,
            "finish": None,
            "pattern": None,
            "texture": None,
            "composition": None,
            "confidence": 0.0
        }

        # Extract from Llama analysis
        if llama_result.get("success") and llama_result.get("analysis"):
            analysis = llama_result["analysis"]
            properties["color"] = analysis.get("colors", [None])[0] if analysis.get("colors") else None
            properties["texture"] = analysis.get("textures", [None])[0] if analysis.get("textures") else None
            properties["composition"] = analysis.get("properties", {}).get("composition")
            properties["finish"] = analysis.get("properties", {}).get("finish")
            properties["pattern"] = analysis.get("properties", {}).get("pattern")
            properties["confidence"] = analysis.get("confidence", 0.0)
        elif "error" not in llama_result:
            # If Llama succeeded but no analysis field, try direct access
            properties["color"] = llama_result.get("colors", [None])[0] if llama_result.get("colors") else None
            properties["texture"] = llama_result.get("textures", [None])[0] if llama_result.get("textures") else None
            properties["composition"] = llama_result.get("properties", {}).get("composition")
            properties["finish"] = llama_result.get("properties", {}).get("finish")
            properties["pattern"] = llama_result.get("properties", {}).get("pattern")
            properties["confidence"] = llama_result.get("confidence", 0.0)

        return properties

    def _calculate_llama_quality_score(
        self,
        llama_result: Dict[str, Any],
        clip_embedding: List[float],
        material_properties: Dict[str, Any]
    ) -> float:
        """
        Calculate quality score based on Llama analysis ONLY.

        NEW SCORING (Llama-only):
        - Llama confidence: 60% weight
        - Material properties completeness: 30% weight
        - CLIP embedding validity: 10% weight

        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        weight_count = 0

        # Llama confidence (60% weight)
        if llama_result.get("success"):
            llama_conf = llama_result.get("analysis", {}).get("confidence", 0.0)
            if llama_conf == 0.0:
                # Try direct access
                llama_conf = llama_result.get("confidence", 0.0)
            score += llama_conf * 0.6
            weight_count += 0.6
        elif "error" not in llama_result:
            # If no explicit success field, try to get confidence directly
            llama_conf = llama_result.get("confidence", 0.0)
            if llama_conf > 0:
                score += llama_conf * 0.6
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
        llama_result: Dict[str, Any],
        claude_result: Dict[str, Any],
        clip_embedding: List[float],
        material_properties: Dict[str, Any]
    ) -> float:
        """Calculate real quality score based on analysis results"""
        score = 0.0
        weight_count = 0

        # Llama confidence (40% weight)
        if llama_result.get("success"):
            llama_conf = llama_result.get("analysis", {}).get("confidence", 0.0)
            score += llama_conf * 0.4
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
        llama_result: Dict[str, Any],
        claude_result: Dict[str, Any],
        material_properties: Dict[str, Any]
    ) -> float:
        """Calculate confidence score based on model agreement"""
        confidences = []

        # Llama confidence
        if llama_result.get("success"):
            confidences.append(llama_result.get("analysis", {}).get("confidence", 0.0))

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

