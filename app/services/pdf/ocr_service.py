"""
OCR Service for multi-modal text extraction from images.

Single-tier: Chandra v2 cloud endpoint only (HuggingFace, GPU, structured bbox-JSON).

Pytesseract was removed 2026-05-01 — it had been silently broken on production
(TESSDATA_PREFIX unset, no en.traineddata installed) and even when "working" it
produced bbox-less text that was indistinguishable to layout-merge from
UNCLASSIFIED orphans, silently degrading layout-aware chunking. Chandra v2 now
has retry-with-jitter (3 attempts, temps 0.0/0.1/0.2) which raises the success
rate to >95% and eliminates the need for a fallback.
"""

import asyncio
import logging
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from dataclasses import dataclass, field

from app.services.pdf.chandra_endpoint_manager import (
    ChandraEndpointManager,
    ChandraResponseError,
)

# Import endpoint registry for using warmed-up managers
try:
    from app.services.embeddings.endpoint_registry import endpoint_registry
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class IconMetadata:
    """Data class for icon-based metadata extraction results."""
    field_name: str  # e.g., 'slip_resistance', 'fire_rating', 'certification'
    value: str  # e.g., 'R11', 'A1', 'CE'
    confidence: float
    bbox: Optional[List[int]] = None  # [x1, y1, x2, y2]
    icon_type: Optional[str] = None  # 'slip_resistance_icon', 'fire_rating_icon', etc.


@dataclass
class OCRResult:
    """Data class for OCR extraction results.

    `blocks` carries the per-fragment bbox list from Chandra v2 (each entry is
    {text, x, y, w, h} in image pixel coordinates). Consumers that need
    bbox-aware processing (icon-metadata extraction, layout-merge fallback)
    must read `blocks`, NOT the legacy single-`bbox` field.

    `method` values:
      - 'chandra'         — successful OCR
      - 'chandra_failed'  — all retries exhausted; text is "" and blocks is []
                            (explicit failure marker; downstream can distinguish
                            from "page genuinely had no text")
    """
    text: str
    confidence: float
    bbox: Optional[List[int]] = None  # [x1, y1, x2, y2] — legacy, generally None
    language: Optional[str] = None
    method: Optional[str] = None
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    attempts_made: int = 0  # populated by _call_chandra; useful for telemetry


@dataclass
class OCRConfig:
    """Configuration for OCR processing (Chandra v2 only)."""
    languages: List[str] = None
    use_gpu: bool = False
    confidence_threshold: float = 0.5
    preprocessing_enabled: bool = True

    # Chandra OCR Endpoint Settings
    chandra_enabled: bool = True
    chandra_endpoint_url: str = ""
    chandra_hf_token: str = ""
    chandra_endpoint_name: str = ""
    chandra_namespace: str = ""
    chandra_confidence_threshold: float = 0.7
    chandra_auto_pause_timeout: int = 60
    chandra_inference_timeout: int = 30
    chandra_max_resume_retries: int = 3

    def __post_init__(self):
        if self.languages is None:
            self.languages = ['en']


class ImagePreprocessor:
    """Image preprocessing utilities for better OCR accuracy."""
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """
        Apply image enhancements to improve OCR accuracy.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Enhanced image as numpy array
        """
        try:
            # Convert to PIL Image for enhancement
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(2.0)
            
            # Apply slight blur to reduce noise
            pil_image = pil_image.filter(ImageFilter.MedianFilter(size=3))
            
            # Convert back to numpy array
            enhanced = np.array(pil_image)
            
            # Convert back to BGR if original was BGR
            if len(image.shape) == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
                
            return enhanced
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {str(e)}")
            return image
    
    @staticmethod
    def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for optimal OCR performance.

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            return cleaned

        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}")
            return image


class OCRService:
    """
    OCR Service providing text extraction from images.

    Two-tier chain:
    1. Chandra v2 cloud endpoint (high quality, GPU, structured bbox-JSON)
    2. Pytesseract (local, deterministic, last-resort fallback)
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        """
        Initialize OCR Service.

        Args:
            config: OCR configuration settings
        """
        self.config = config or OCRConfig()
        self.preprocessor = ImagePreprocessor()
        self._initialized = False

        # Initialize Chandra Endpoint Manager - prefer warmed-up manager from registry
        self.chandra_manager: Optional[ChandraEndpointManager] = None
        if self.config.chandra_enabled:
            # Try registry first for warmed-up manager
            if REGISTRY_AVAILABLE:
                try:
                    registry_manager = endpoint_registry.get_chandra_manager()
                    if registry_manager is not None:
                        self.chandra_manager = registry_manager
                        logger.info("✅ Using warmed-up Chandra manager from registry")
                except Exception as e:
                    logger.debug(f"Registry Chandra manager not available: {e}")

            # Fallback: Create new manager if registry not available
            if self.chandra_manager is None and self.config.chandra_endpoint_url and self.config.chandra_hf_token:
                try:
                    self.chandra_manager = ChandraEndpointManager(
                        endpoint_url=self.config.chandra_endpoint_url,
                        hf_token=self.config.chandra_hf_token,
                        endpoint_name=self.config.chandra_endpoint_name,
                        namespace=self.config.chandra_namespace,
                        auto_pause_timeout=self.config.chandra_auto_pause_timeout,
                        inference_timeout=self.config.chandra_inference_timeout,
                        max_resume_retries=self.config.chandra_max_resume_retries,
                        enabled=self.config.chandra_enabled
                    )
                    logger.info("ℹ️ Created new Chandra manager (registry not available)")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize Chandra endpoint manager: {e}")
                    self.chandra_manager = None

        logger.info(f"OCR Service initialized with languages: {self.config.languages}")
        self._initialized = True

    def initialize(self) -> None:
        """Initialize OCR service. Chandra is ready from __init__; Tesseract is built-in."""
        self._initialized = True
        logger.info("OCR Service initialized (Chandra primary, Tesseract fallback)")
    

    def _call_chandra(
        self,
        image: np.ndarray,
        caller: str = "ad_hoc",
        image_id: Optional[str] = None,
        job_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[OCRResult]:
        """Call Chandra v2 with retry-with-jitter.

        Returns:
            - One OCRResult with method='chandra' on success (text + blocks populated)
            - One OCRResult with method='chandra_failed' on retry-exhaustion
              (explicit failure marker — text="", blocks=[]; consumers must check
              method, NOT just emptiness, to distinguish failure from "no text")

        Never raises. HTTP errors are caught and converted to chandra_failed
        results so callers don't need to wrap in try/except.
        """
        if not self.chandra_manager:
            return [OCRResult(
                text="", confidence=0.0, method='chandra_failed', blocks=[],
                attempts_made=0,
            )]

        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image

        import time as _time
        _chandra_start = _time.time()
        try:
            chandra_result = self.chandra_manager.run_inference(
                pil_image,
                caller=caller,
                image_id=image_id,
                job_id=job_id,
                document_id=document_id,
            )
            from app.services.core.endpoint_controller import endpoint_controller
            endpoint_controller.record_success("chandra")
        except ChandraResponseError as parse_err:
            # All retries exhausted — return explicit failure marker.
            logger.warning(f"❌ Chandra OCR exhausted retries for {caller}: {parse_err}")
            return [OCRResult(
                text="", confidence=0.0, method='chandra_failed', blocks=[],
                attempts_made=len(self.chandra_manager._RETRY_TEMPERATURES),
            )]
        except Exception as chandra_err:
            from app.services.core.endpoint_controller import endpoint_controller
            endpoint_controller.record_overload_exception("chandra", chandra_err)
            logger.warning(f"❌ Chandra HTTP/endpoint error for {caller}: {chandra_err}")
            return [OCRResult(
                text="", confidence=0.0, method='chandra_failed', blocks=[],
                attempts_made=0,
            )]

        # Track GPU time-based cost (L4 endpoint, $0.80/hr) — sync-safe.
        try:
            import asyncio as _asyncio
            from app.services.core.ai_call_logger import AICallLogger
            _chandra_latency_ms = int((_time.time() - _chandra_start) * 1000)
            _log_coro = AICallLogger().log_time_based_call(
                task="pdf_ocr_chandra",
                model="chandra-ocr-2",
                latency_ms=_chandra_latency_ms,
                confidence_score=0.85,
                confidence_breakdown={},
            )
            try:
                _loop = _asyncio.get_running_loop()
                _loop.create_task(_log_coro)
            except RuntimeError:
                _asyncio.run(_log_coro)
        except Exception as _log_err:
            logger.debug(f"Chandra logging failed (non-fatal): {_log_err}")

        chandra_text = chandra_result.get('generated_text', '') or ''
        blocks = chandra_result.get('blocks', []) or []
        attempts_made = chandra_result.get('attempts_made', 1)

        if chandra_text.strip() or blocks:
            return [OCRResult(
                text=chandra_text,
                confidence=0.85,
                method='chandra',
                blocks=blocks,
                attempts_made=attempts_made,
            )]
        # Parsed successfully but the page genuinely has no text.
        return [OCRResult(
            text="", confidence=0.85, method='chandra', blocks=[],
            attempts_made=attempts_made,
        )]

    def extract_text_from_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        use_preprocessing: bool = None,
        caller: str = "ad_hoc",
        image_id: Optional[str] = None,
        job_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[OCRResult]:
        """
        Extract text from image using Chandra v2 (single-tier, with retry).

        Returns:
            List of OCRResult. Always non-empty:
              - Success: one OCRResult with method='chandra', text + blocks
              - Failure (retries exhausted): one OCRResult with method='chandra_failed'
              - No-text (page parsed clean but has no text): one OCRResult with
                method='chandra' and text="" / blocks=[]

            Callers MUST check `result.method == 'chandra_failed'` to distinguish
            real failure from "no text on page" — both have empty text.
        """
        image = self._load_image(image_input)

        if use_preprocessing or (use_preprocessing is None and self.config.preprocessing_enabled):
            image = self.preprocessor.enhance_image(image)
            image = self.preprocessor.preprocess_for_ocr(image)

        return self._call_chandra(
            image, caller=caller, image_id=image_id,
            job_id=job_id, document_id=document_id,
        )
    
    def extract_text_simple(
        self, 
        image_input: Union[str, Path, np.ndarray, Image.Image]
    ) -> str:
        """
        Extract text from image and return as simple concatenated string.
        
        Args:
            image_input: Image file path, numpy array, or PIL Image
            
        Returns:
            Extracted text as a single string
        """
        results = self.extract_text_from_image(image_input)
        
        # Concatenate all text results
        text_parts = [result.text for result in results if result.text.strip()]
        return ' '.join(text_parts)
    
    def get_text_with_confidence(
        self, 
        image_input: Union[str, Path, np.ndarray, Image.Image],
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Extract text with confidence metrics and metadata.
        
        Args:
            image_input: Image file path, numpy array, or PIL Image
            min_confidence: Minimum confidence threshold for results
            
        Returns:
            Dictionary with extracted text, confidence metrics, and metadata
        """
        results = self.extract_text_from_image(image_input)
        
        # Filter by confidence
        filtered_results = [r for r in results if r.confidence >= min_confidence]
        
        if not filtered_results:
            return {
                'text': '',
                'confidence': 0.0,
                'word_count': 0,
                'regions': 0,
                'methods_used': [],
                'metadata': {}
            }
        
        # Calculate metrics
        all_text = ' '.join([r.text for r in filtered_results])
        avg_confidence = sum(r.confidence for r in filtered_results) / len(filtered_results)
        methods_used = list(set(r.method for r in filtered_results if r.method))
        
        return {
            'text': all_text,
            'confidence': avg_confidence,
            'word_count': len(all_text.split()),
            'regions': len(filtered_results),
            'methods_used': methods_used,
            'metadata': {
                'languages': self.config.languages,
                'preprocessing_used': self.config.preprocessing_enabled,
                'results': [
                    {
                        'text': r.text,
                        'confidence': r.confidence,
                        'bbox': r.bbox,
                        'method': r.method
                    } for r in filtered_results
                ]
            }
        }
    
    def _load_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Load and convert image to numpy array.
        
        Args:
            image_input: Image in various formats
            
        Returns:
            Image as numpy array
            
        Raises:
            ValueError: If image format is not supported
        """
        try:
            if isinstance(image_input, (str, Path)):
                # Load from file path
                image_path = Path(image_input)
                if not image_path.exists():
                    raise ValueError(f"Image file not found: {image_path}")

                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")

            elif isinstance(image_input, np.ndarray):
                # Already a numpy array
                image = image_input.copy()

            elif isinstance(image_input, Image.Image):
                # Convert PIL Image to numpy array
                image = np.array(image_input)
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image: {str(e)}")
            raise ValueError(f"Invalid image input: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on OCR service.

        Returns:
            Dictionary with health status and capabilities
        """
        status = {
            'initialized': self._initialized,
            'chandra_available': self.chandra_manager is not None,
            'languages': self.config.languages,
            'gpu_enabled': self.config.use_gpu,
        }
        status['healthy'] = status['chandra_available']
        return status

    async def extract_icon_metadata(
        self,
        image: Union[str, np.ndarray, Image.Image],
        workspace_id: str = None,
        use_ai: bool = True
    ) -> List[IconMetadata]:
        """
        Extract metadata from icons and symbols in images using AI with database prompts.

        This method:
        1. Extracts text from image using OCR
        2. Loads icon extraction prompt from database
        3. Uses AI (Claude/GPT) to interpret OCR results and identify technical specifications
        4. Returns structured metadata with confidence scores

        Args:
            image: Input image (path, numpy array, or PIL Image)
            workspace_id: Workspace ID for loading custom prompts
            use_ai: Whether to use AI interpretation (True) or regex patterns (False)

        Returns:
            List of IconMetadata objects with extracted values
        """
        # Use default workspace ID from config if not provided
        from app.config import get_settings
        workspace_id = workspace_id or get_settings().default_workspace_id

        try:
            from app.services.core.supabase_client import get_supabase_client

            # Load image
            if isinstance(image, str):
                img = cv2.imread(image)
            elif isinstance(image, Image.Image):
                img = np.array(image)
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = image

            # Preprocess image for better icon detection
            preprocessed = self.preprocessor.preprocess_for_ocr(img)

            # Extract all text with bounding boxes. Use the synchronous
            # extract_text_from_image() (returns List[OCRResult] with
            # .text / .confidence / .bbox) and run it in a thread so the
            # CPU-bound OCR call doesn't block the event loop.
            ocr_results = await asyncio.to_thread(
                self.extract_text_from_image,
                preprocessed,
                False,  # use_preprocessing=False — we already preprocessed above
            )

            # Filter out chandra_failed markers — they have no usable text.
            ocr_results = [
                r for r in (ocr_results or [])
                if r.method != 'chandra_failed' and (r.text.strip() or r.blocks)
            ]
            if not ocr_results:
                logger.warning("No text extracted from image (Chandra failed or page empty)")
                return []

            # Build per-fragment OCR data from Chandra's bbox blocks.
            # Previously we used `result.bbox` which was always None — Claude was
            # asked to reason about icon positions with no positional info.
            # Now we pass the per-fragment {text, x, y, w, h} list that Chandra
            # actually produced (audit fix #22).
            ocr_data = []
            for result in ocr_results:
                if result.blocks:
                    for block in result.blocks:
                        ocr_data.append({
                            'text': block.get('text', ''),
                            'confidence': result.confidence,
                            'bbox': {
                                'x': block.get('x'),
                                'y': block.get('y'),
                                'w': block.get('w'),
                                'h': block.get('h'),
                            },
                        })
                elif result.text.strip():
                    # No bbox available — emit a single-entry fallback.
                    ocr_data.append({
                        'text': result.text,
                        'confidence': result.confidence,
                        'bbox': None,
                    })

            if use_ai:
                # Load prompt from database
                supabase = get_supabase_client()
                prompt_result = supabase.client.table('prompts') \
                    .select('prompt_text') \
                    .eq('workspace_id', workspace_id) \
                    .eq('prompt_type', 'extraction') \
                    .eq('stage', 'image_analysis') \
                    .eq('category', 'icon_metadata') \
                    .eq('is_active', True) \
                    .order('version', desc=True) \
                    .limit(1) \
                    .execute()

                if not prompt_result.data:
                    logger.warning("No icon extraction prompt found in database, using fallback")
                    return []

                prompt_template = prompt_result.data[0]['prompt_text']

                # Imports at top of the branch so `json` is bound before
                # any use below — placing them after the f-string would make
                # `json` a function-local and raise UnboundLocalError.
                import json
                import re
                import anthropic
                import os

                # Build full prompt with OCR data
                full_prompt = f"{prompt_template}\n\n**OCR Results:**\n\n```json\n{json.dumps(ocr_data, indent=2)}\n```\n\nAnalyze the OCR results and extract icon-based metadata. Return ONLY valid JSON."

                # Call AI (Claude preferred for vision tasks) — tracked
                from app.services.core.claude_helper import tracked_claude_call
                response = tracked_claude_call(
                    task="ocr_icon_metadata_extraction",
                    model="claude-opus-4-7",
                    max_tokens=2048,
                    messages=[{
                        "role": "user",
                        "content": full_prompt
                    }],
                )

                # Parse AI response
                response_text = response.content[0].text.strip()

                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group(0))

                    # Convert to IconMetadata objects
                    extracted_metadata = []
                    for item in result_data.get('icon_metadata', []):
                        metadata = IconMetadata(
                            field_name=item['field_name'],
                            value=item['value'],
                            confidence=item['confidence'],
                            bbox=item.get('bbox'),
                            icon_type=item.get('icon_type')
                        )
                        extracted_metadata.append(metadata)
                        logger.info(f"✅ Extracted icon metadata: {item['field_name']}={item['value']} (confidence: {item['confidence']:.2f})")

                    return extracted_metadata
                else:
                    logger.error("Failed to parse AI response as JSON")
                    return []

            else:
                # Fallback: Use regex patterns (legacy mode)
                logger.warning("Using legacy regex pattern matching for icon extraction")
                return []

        except Exception as e:
            logger.error(f"Error extracting icon metadata: {e}")
            return []

    def get_endpoint_stats(self) -> Dict[str, Any]:
        """
        Get Chandra endpoint usage statistics.

        Returns:
            Dictionary with endpoint stats (uptime, calls, etc.)
        """
        if self.chandra_manager:
            return {
                'enabled': True,
                'total_uptime': self.chandra_manager.total_uptime,
                'inference_count': self.chandra_manager.inference_count,
                'resume_count': self.chandra_manager.resume_count,
                'pause_count': self.chandra_manager.pause_count,
                'last_used': self.chandra_manager.last_used
            }
        return {'enabled': False}


# Global instance
_ocr_service: Optional[OCRService] = None


def get_ocr_service(config: Optional[OCRConfig] = None) -> OCRService:
    """
    Get the global OCR service instance.
    
    Args:
        config: OCR configuration (used only on first call)
        
    Returns:
        OCRService instance
    """
    global _ocr_service
    
    if _ocr_service is None:
        _ocr_service = OCRService(config)
    
    return _ocr_service


def initialize_ocr_service(config: Optional[OCRConfig] = None) -> None:
    """
    Initialize the global OCR service.
    
    Args:
        config: OCR configuration settings
    """
    service = get_ocr_service(config)
    service.initialize()
