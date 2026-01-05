"""
PDF Processing Service - Integration with existing PyMuPDF4LLM functionality

This service wraps the existing extractor.py functionality to work with the 
production FastAPI application structure, providing async interfaces and 
proper error handling while leveraging the proven PDF extraction code.
"""

import asyncio
import base64
import inspect
import logging
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiofiles
import httpx
from dataclasses import dataclass

# Import image processing libraries (headless OpenCV)
try:
    import cv2
    CV2_AVAILABLE = True
    logging.info("OpenCV (headless) loaded successfully")
except ImportError as e:
    logging.error(f"OpenCV (headless) not available: {e}. Image processing features will be limited.")
    CV2_AVAILABLE = False
    cv2 = None

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from PIL.ExifTags import TAGS
import imageio

try:
    from skimage import filters, morphology, measure
    SKIMAGE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"scikit-image not available: {e}. Some advanced image processing features will be disabled.")
    SKIMAGE_AVAILABLE = False
from scipy import ndimage


# ============================================================================
# HELPER FUNCTION: Download Image from Supabase URL to Base64
# ============================================================================

async def download_image_to_base64(image_url: str) -> str:
    """
    Download image from Supabase URL and convert to base64.

    Used for AI classification and Qwen Vision analysis.
    CLIP embeddings can use URLs directly.

    Args:
        image_url: Public Supabase Storage URL

    Returns:
        Base64-encoded image string (without data URI prefix)

    Raises:
        Exception: If download fails or URL is invalid
    """
    import httpx
    import base64

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(image_url)
            if response.status_code == 200:
                return base64.b64encode(response.content).decode('utf-8')
            else:
                raise Exception(f"Failed to download image from {image_url}: HTTP {response.status_code}")
    except Exception as e:
        raise Exception(f"Error downloading image to base64: {str(e)}")


# Import existing extraction functions
try:
    # Try to import from the proper location first
    from ..core.extractor import extract_pdf_to_markdown, extract_pdf_to_markdown_with_doc, extract_pdf_tables, extract_json_and_images
except ImportError:
    # Fall back to the root level extractor if it exists
    try:
        from app.core.extractor import extract_pdf_to_markdown, extract_pdf_to_markdown_with_doc, extract_pdf_tables, extract_json_and_images
    except ImportError as e:
        # Log the error and provide a fallback
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to import extractor functions: {e}")
        # Define placeholder functions that will raise NotImplementedError
        def extract_pdf_to_markdown(*args, **kwargs):
            raise NotImplementedError("PDF extraction functions not available")
        def extract_pdf_to_markdown_with_doc(*args, **kwargs):
            raise NotImplementedError("PDF extraction functions not available")
        def extract_pdf_tables(*args, **kwargs):
            raise NotImplementedError("PDF table extraction functions not available")
        def extract_json_and_images(*args, **kwargs):
            raise NotImplementedError("PDF image extraction functions not available")

# Import OCR service
from app.services.pdf.ocr_service import get_ocr_service, OCRConfig

# Import Supabase client for storage
from app.services.core.supabase_client import get_supabase_client

# Import custom exceptions
from app.utils.exceptions import (
    PDFProcessingError,
    PDFValidationError,
    PDFExtractionError,
    PDFDownloadError,
    PDFSizeError,
    PDFTimeoutError,
    PDFStorageError,
    PDFFormatError
)

# Import unified chunking service (Step 6)

# Import unified chunking service (Step 6)
from app.services.chunking.unified_chunking_service import UnifiedChunkingService, ChunkingConfig, ChunkingStrategy

# Import worker for process isolation
from app.services.pdf.pdf_worker import execute_pdf_extraction_job

# Import PageConverter for centralized page number management
from app.utils.page_converter import PageConverter, PageNumber


@dataclass
class PDFProcessingResult:
    """Result of PDF processing operation"""
    document_id: str
    markdown_content: str
    extracted_images: List[Dict[str, Any]]
    ocr_text: str  # Combined OCR text from all images
    ocr_results: List[Dict[str, Any]]  # Detailed OCR results per image
    metadata: Dict[str, Any]
    processing_time: float
    page_count: int
    word_count: int
    character_count: int
    multimodal_enabled: bool = False
    temp_dir: Optional[str] = None  # Temp directory to cleanup when job completes
    page_chunks: Optional[List[Dict[str, Any]]] = None  # ✅ NEW: Page-aware text data from PyMuPDF4LLM


class PDFProcessor:
    """
    Core PDF processing service using existing PyMuPDF4LLM functionality.
    
    This class provides async interfaces to the existing extractor.py functions,
    adding proper error handling, logging, and integration with the FastAPI app.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the PDF processor with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize unified chunking service (Step 6)
        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID,  # Use hybrid strategy by default
            max_chunk_size=1000,
            min_chunk_size=100,
            overlap_size=100,
            preserve_structure=True,
            split_on_sentences=True,
            split_on_paragraphs=True,
            respect_hierarchy=True
        )
        self.chunking_service = UnifiedChunkingService(chunking_config)
        
        # Default processing options
        self.default_timeout = self.config.get('timeout_seconds', 7200)  # 2 hours for large PDFs with OCR
        self.max_file_size = self.config.get('max_file_size_mb', 50) * 1024 * 1024  # Convert to bytes
        self.temp_dir_base = self.config.get('temp_dir', tempfile.gettempdir())

        # Initialize THREAD pool executor for async processing
        # NOTE: Using ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickle issues
        # with pydantic Settings and other non-picklable objects
        max_workers = self.config.get('max_workers', 2)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.logger.info("PDFProcessor initialized with config: %s (max_workers=%d for memory efficiency)", self.config, max_workers)

    def __del__(self):
        """Cleanup resources when the processor is destroyed."""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=True)
            self.logger.debug("ThreadPoolExecutor shutdown completed")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for PDF processing capabilities.

        Returns:
            Dict with health status and details
        """
        try:
            # Check if required dependencies are available
            dependencies = {
                "pymupdf": True,  # Should be available if we got this far
                "opencv": CV2_AVAILABLE,
                "skimage": SKIMAGE_AVAILABLE,
                "pil": True,  # PIL is imported successfully
                "numpy": True,  # numpy is imported successfully
            }

            # Check if executor is available
            executor_healthy = hasattr(self, 'executor') and self.executor is not None

            # Check temp directory access
            temp_dir_accessible = os.access(self.temp_dir_base, os.W_OK)

            # Overall health status
            all_critical_deps = dependencies["pymupdf"] and dependencies["pil"] and dependencies["numpy"]
            overall_healthy = all_critical_deps and executor_healthy and temp_dir_accessible

            return {
                "status": "healthy" if overall_healthy else "degraded",
                "dependencies": dependencies,
                "executor_available": executor_healthy,
                "temp_directory_accessible": temp_dir_accessible,
                "max_file_size_mb": self.max_file_size // (1024 * 1024),
                "timeout_seconds": self.default_timeout,
                "max_workers": getattr(self.executor, '_max_workers', 'unknown') if executor_healthy else 0,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def process_pdf_from_bytes(
        self,
        pdf_bytes: bytes,
        document_id: Optional[str] = None,
        processing_options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ) -> PDFProcessingResult:
        """
        Process PDF from bytes and return markdown + images.
        
        Args:
            pdf_bytes: Raw PDF file bytes
            document_id: Optional document identifier
            processing_options: Processing configuration options
            
        Returns:
            PDFProcessingResult with extracted content and metadata
            
        Raises:
            PDFProcessingError: If processing fails
            ProcessingTimeoutError: If processing exceeds timeout
        """
        start_time = time.time()
        document_id = document_id or str(uuid.uuid4())
        processing_options = processing_options or {}
        
        self.logger.info("Starting PDF processing for document %s", document_id)
        
        # Validate file size
        if len(pdf_bytes) > self.max_file_size:
            raise PDFSizeError(f"PDF file too large: {len(pdf_bytes)} bytes (max: {self.max_file_size})")
        
        temp_dir = None
        try:
            # Create temporary directory for processing
            temp_dir = self._create_temp_directory(document_id)
            self.logger.info(f"📁 Created temp directory: {temp_dir}")
            
            # Save PDF bytes to temporary file
            temp_pdf_path = os.path.join(temp_dir, f"{document_id}.pdf")
            self.logger.info(f"💾 Saving PDF to: {temp_pdf_path}")
            
            # Optimization: Write to file in executor to ensure pdf_bytes scope ends and memory is freed
            # This prevents holding the full PDF in memory while processing
            def write_pdf_to_disk(path, data):
                with open(path, 'wb') as f:
                    f.write(data)
            
            await asyncio.get_event_loop().run_in_executor(None, write_pdf_to_disk, temp_pdf_path, pdf_bytes)
            
            # Explicitly release memory
            file_size = len(pdf_bytes)
            del pdf_bytes
            
            self.logger.info(f"✅ PDF saved successfully, file size: {file_size} bytes")
            
            # Process with timeout
            timeout = processing_options.get('timeout_seconds', self.default_timeout)
            
            try:
                result = await asyncio.wait_for(
                    self._process_pdf_file(temp_pdf_path, document_id, processing_options, progress_callback),
                    timeout=timeout
                )

                processing_time = time.time() - start_time
                result.processing_time = processing_time

                self.logger.info(
                    "PDF processing completed for document %s in %.2f seconds",
                    document_id, processing_time
                )

                # Store temp_dir in result for reference (cleanup handled by admin cron job)
                result.temp_dir = temp_dir

                return result

            except asyncio.TimeoutError:
                # NOTE: Cleanup moved to admin panel cron job
                raise PDFTimeoutError(f"PDF processing timed out after {timeout} seconds")

        except Exception as e:
            # NOTE: Cleanup moved to admin panel cron job
            self.logger.error("PDF processing failed for document %s: %s", document_id, str(e))
            if isinstance(e, (PDFProcessingError, PDFTimeoutError)):
                raise
            raise PDFProcessingError(f"Unexpected error during PDF processing: {str(e)}") from e
    
    async def process_pdf_from_url(
        self,
        pdf_url: str,
        document_id: Optional[str] = None,
        processing_options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ) -> PDFProcessingResult:
        """
        Process PDF from URL and return markdown + images.
        
        Args:
            pdf_url: URL to PDF file
            document_id: Optional document identifier
            processing_options: Processing configuration options
            
        Returns:
            PDFProcessingResult with extracted content and metadata
            
        Raises:
            PDFDownloadError: If PDF download fails
            PDFProcessingError: If processing fails
        """
        document_id = document_id or str(uuid.uuid4())
        self.logger.info("Downloading PDF from URL for document %s: %s", document_id, pdf_url)
        
        try:
            # Download PDF with timeout
            timeout = processing_options.get('download_timeout', 30) if processing_options else 30
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(pdf_url)
                response.raise_for_status()
                
                # Validate content type
                content_type = response.headers.get('content-type', '').lower()
                if 'application/pdf' not in content_type and not pdf_url.lower().endswith('.pdf'):
                    self.logger.warning("Unexpected content type for PDF: %s", content_type)
                
                pdf_bytes = response.content
                
        except httpx.HTTPError as e:
            raise PDFDownloadError(f"Failed to download PDF from {pdf_url}: {str(e)}") from e
        except Exception as e:
            raise PDFDownloadError(f"Unexpected error downloading PDF: {str(e)}") from e
        
        # Process the downloaded PDF
        return await self.process_pdf_from_bytes(pdf_bytes, document_id, processing_options, progress_callback)
    
    async def _process_pdf_file(
        self,
        pdf_path: str,
        document_id: str,
        processing_options: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> PDFProcessingResult:
        """
        Internal method to process PDF file using existing extractor functions.

        This method runs the existing synchronous extractor functions in a thread pool
        to maintain async compatibility.
        """
        loop = asyncio.get_event_loop()

        try:
            # Extract markdown content using existing function with timeout
            # Set a reasonable timeout for markdown extraction (5 minutes for large PDFs)
            markdown_timeout = processing_options.get('markdown_timeout', 300)  # 5 minutes default

            # Extract markdown content using isolated worker process
            # Progress callback is limited with ProcessPool, we notify start here
            if progress_callback:
                try:
                    if not inspect.iscoroutinefunction(progress_callback):
                        progress_callback(progress=10, details={"current_step": "Starting PDF extraction in isolated process"})
                except Exception:
                    pass

            try:
                # Execute in thread pool
                # ✅ NEW: Worker now returns (markdown_content, metadata, page_chunks)
                # Filter out non-serializable objects (services, trackers) before passing to worker
                filtered_options = {
                    k: v for k, v in processing_options.items()
                    if k not in ('checkpoint_recovery_service', 'progress_tracker', 'job_id')
                }
                markdown_content, metadata, page_chunks = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        execute_pdf_extraction_job,
                        pdf_path,
                        filtered_options
                    ),
                    timeout=markdown_timeout
                )
            except asyncio.TimeoutError:
                self.logger.error(f"Markdown extraction timed out after {markdown_timeout} seconds")
                raise PDFTimeoutError(f"Markdown extraction timed out after {markdown_timeout} seconds. The PDF may be corrupted or too complex.")
            
            # Extract images if requested
            extracted_images = []
            extraction_stats = {'vision_guided_count': 0, 'pymupdf_count': 0, 'failed_count': 0, 'total_pages': 0}

            if processing_options.get('extract_images', True):
                # ✅ NEW: Pass job_id and checkpoint_recovery_service if available in processing_options
                job_id = processing_options.get('job_id')
                checkpoint_recovery_service = processing_options.get('checkpoint_recovery_service')
                progress_tracker = processing_options.get('progress_tracker')

                extracted_images, extraction_stats = await self._extract_images_async(
                    pdf_path,
                    document_id,
                    processing_options,
                    progress_callback,
                    job_id=job_id,
                    checkpoint_recovery_service=checkpoint_recovery_service,
                    progress_tracker=progress_tracker
                )
            
            # Calculate content metrics
            content_metrics = self._calculate_content_metrics(markdown_content)
            
            # Initialize OCR results
            ocr_text = ""
            ocr_results = []

            # Intelligent multimodal detection
            manual_multimodal = processing_options.get('enable_multimodal', None)
            if manual_multimodal is not None:
                # User explicitly set multimodal preference
                multimodal_enabled = manual_multimodal
                multimodal_reason = "manual_override"
            else:
                # Auto-detect if multimodal processing would be beneficial
                multimodal_enabled = self._should_use_multimodal(extracted_images, markdown_content)
                multimodal_reason = "auto_detected"

            ocr_languages = processing_options.get('ocr_languages', ['en'])  # Define outside conditional

            self.logger.info(f"🔍 Multimodal: {'enabled' if multimodal_enabled else 'disabled'} "
                           f"({multimodal_reason}, {len(extracted_images)} images available)")

            # Process images with OCR if multimodal is enabled
            if multimodal_enabled and extracted_images:
                self.logger.info(f"Processing {len(extracted_images)} images with OCR")
                ocr_text, ocr_results = await self._process_images_with_ocr(
                    extracted_images, ocr_languages, progress_callback
                )
                
                # Enhance extracted images with OCR data
                for i, image_data in enumerate(extracted_images):
                    if i < len(ocr_results):
                        image_data['ocr_result'] = ocr_results[i]
            
            # Update content metrics to include OCR text
            if ocr_text:
                ocr_word_count = len(ocr_text.split())
                content_metrics['word_count'] += ocr_word_count
                content_metrics['character_count'] += len(ocr_text)

            return PDFProcessingResult(
                document_id=document_id,
                markdown_content=markdown_content,
                extracted_images=extracted_images,
                ocr_text=ocr_text,
                ocr_results=ocr_results,
                metadata={
                    **metadata,
                    **content_metrics,
                    'processing_options': processing_options,
                    'timestamp': datetime.utcnow().isoformat(),
                    'multimodal_enabled': multimodal_enabled,
                    'ocr_enabled': multimodal_enabled and bool(extracted_images),
                    'ocr_languages': ocr_languages if multimodal_enabled else [],
                    'ocr_text_length': len(ocr_text) if ocr_text else 0,
                    # ✅ NEW: Include extraction stats for job tracking
                    'extraction_stats': extraction_stats
                },
                processing_time=0.0,  # Will be set by caller
                page_count=metadata.get('page_count', 0),
                word_count=content_metrics['word_count'],
                character_count=content_metrics['character_count'],
                multimodal_enabled=multimodal_enabled,
                page_chunks=page_chunks  # ✅ NEW: Include page-aware data
            )
            
        except Exception as e:
            self.logger.error("Error processing PDF file %s: %s", pdf_path, str(e))
            raise PDFExtractionError(f"Failed to parse PDF content: {str(e)}") from e
    


    def _should_use_multimodal(self, extracted_images: List[Dict], markdown_content: str) -> bool:
        """
        Intelligent detection of whether multimodal processing would be beneficial.

        Args:
            extracted_images: List of extracted images
            markdown_content: Extracted text content

        Returns:
            bool: True if multimodal processing is recommended

        Note:
            OCR uses 3-phase intelligent filtering (OpenCV → CLIP → EasyOCR),
            so enabling multimodal doesn't mean all images get OCR.
            Only images with text patterns AND technical content get full OCR.
        """
        try:
            # Criteria for multimodal processing
            has_images = len(extracted_images) > 0
            many_images = len(extracted_images) >= 3
            very_many_images = len(extracted_images) >= 50  # Product catalogs
            low_text_content = len(markdown_content.strip()) < 500
            moderate_text_content = len(markdown_content.strip()) < 2000

            # Calculate image-to-text ratio
            text_length = len(markdown_content.strip())
            image_count = len(extracted_images)
            images_per_1000_chars = (image_count / max(text_length, 1)) * 1000 if text_length > 0 else image_count

            # Decision logic
            if not has_images:
                return False  # No images → no multimodal needed

            # ✅ NEW: Enable for image-heavy documents (product catalogs)
            # Catalogs have many images with technical specs/labels that need OCR
            if very_many_images:
                self.logger.info(f"   📚 Catalog detected: {image_count} images → enabling OCR")
                return True

            # ✅ NEW: Enable if high image-to-text ratio (visual-heavy documents)
            # Example: 100 images + 10,000 chars = 10 images per 1000 chars → likely catalog
            if images_per_1000_chars > 5 and image_count >= 10:
                self.logger.info(f"   📊 High image density: {images_per_1000_chars:.1f} images/1000 chars → enabling OCR")
                return True

            if many_images and low_text_content:
                return True  # Many images + little text → likely visual document

            if has_images and moderate_text_content:
                return True  # Some images + moderate text → could benefit from multimodal

            if len(extracted_images) >= 1 and low_text_content:
                return True  # Any images + very little text → likely needs OCR

            # Text-heavy document with few images → multimodal not needed
            self.logger.info(f"   📄 Text-heavy document: {text_length} chars, {image_count} images → OCR not needed")
            return False

        except Exception as e:
            self.logger.warning(f"Multimodal detection failed: {e}, defaulting to False")
            return False

    async def _extract_images_async(
        self,
        pdf_path: str,
        document_id: str,
        processing_options: Dict[str, Any],
        progress_callback: Optional[callable] = None,
        job_id: Optional[str] = None,
        checkpoint_recovery_service: Optional[Any] = None,
        progress_tracker: Optional[Any] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Enhanced async image extraction with Supabase Storage upload.

        NOW USES STREAMING EXTRACTION to prevent OOM on large PDFs.

        Features:
        - STREAMING extraction in small batches (prevents memory accumulation)
        - Process and upload images immediately (don't accumulate in memory)
        - Aggressive garbage collection between batches
        - Image format conversion and optimization
        - Advanced metadata extraction (EXIF, dimensions, quality metrics)
        - Upload to Supabase Storage instead of local storage
        - Quality assessment and duplicate detection
        - ✅ NEW: Full checkpoint and progress tracking integration
        - ✅ NEW: Vision-guided extraction with fallback logging

        Returns:
            Tuple of (images_list, extraction_stats)
            - images_list: List of image dictionaries
            - extraction_stats: Dict with vision_guided_count, pymupdf_count, failed_count, total_pages
        """
        try:
            import fitz
            import gc

            # Create output directory for images
            output_dir = self._create_temp_directory(f"{document_id}_images")
            image_dir = os.path.join(output_dir, 'images')
            os.makedirs(image_dir, exist_ok=True)

            # 🚀 CRITICAL: Use VERY small batch size for low-memory systems
            # Process 2-3 pages at a time to prevent OOM
            batch_size = processing_options.get('image_batch_size', 2)

            # ✅ OPTIMIZATION: Get page_list for focused extraction
            page_list = processing_options.get('page_list')  # List of page numbers (1-indexed)

            # Open PDF to get page count
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()

            # ✅ NEW: Use PageConverter for centralized page number management
            converter = PageConverter.from_pdf_path(pdf_path)
            
            self.logger.info(
                f"📐 Detected PDF layout: {converter.pages_per_sheet} page(s) per sheet"
            )
            self.logger.info(
                f"   Physical pages: {converter.total_pdf_pages}, "
                f"Catalog pages: {converter.total_catalog_pages}"
            )

            # Determine which pages to process
            if page_list:
                # page_list contains catalog pages (1-indexed)
                # Validate and convert to array indices
                valid_catalog_pages = converter.validate_page_range(page_list)
                pages_to_process = [
                    converter.from_catalog_page(catalog_page).array_index
                    for catalog_page in valid_catalog_pages
                ]
                
                if len(valid_catalog_pages) < len(page_list):
                    self.logger.info(
                        f"   Filtered {len(page_list) - len(valid_catalog_pages)} "
                        f"out-of-bounds catalog pages"
                    )
            else:
                pages_to_process = list(range(total_pages))


            self.logger.info(f"🔄 STREAMING IMAGE EXTRACTION: {len(pages_to_process)} pages in batches of {batch_size}")

            # Process pages in small batches with aggressive memory cleanup
            loop = asyncio.get_event_loop()
            all_images = []

            # Track extraction method for metrics
            extraction_stats = {
                'vision_guided_count': 0,
                'pymupdf_count': 0,
                'failed_count': 0,
                'total_pages': len(pages_to_process)
            }

            for batch_num, batch_start in enumerate(range(0, len(pages_to_process), batch_size)):
                batch_end = min(batch_start + batch_size, len(pages_to_process))
                batch_pages = pages_to_process[batch_start:batch_end]

                self.logger.info(f"   📦 Batch {batch_num + 1}: Processing pages {batch_pages}")

                # Extract images for THIS BATCH
                batch_extracted_images = await loop.run_in_executor(
                    None,
                    self._extract_batch_images,
                    pdf_path,
                    image_dir,
                    batch_pages,
                    job_id,
                    document_id,
                    converter  # ✅ NEW: Pass PageConverter for page number validation
                )

                # Update extraction stats based on extracted images
                for img in batch_extracted_images:
                    method = img.get('detection_method', 'pymupdf')
                    if method == 'vision_guided':
                        extraction_stats['vision_guided_count'] += 1
                    else:
                        extraction_stats['pymupdf_count'] += 1

                # Process extracted images IMMEDIATELY (don't accumulate)
                batch_images = await self._process_batch_images(
                    image_dir,
                    document_id,
                    batch_extracted_images,
                    processing_options,
                    progress_callback,
                    batch_num,
                    len(pages_to_process) // batch_size + 1
                )

                all_images.extend(batch_images)

                # AGGRESSIVE MEMORY CLEANUP after each batch
                gc.collect()

                self.logger.info(f"   ✅ Batch {batch_num + 1} complete: {len(batch_images)} images processed, {len(all_images)} total")

            self.logger.info(
                f"✅ STREAMING EXTRACTION COMPLETE: {len(all_images)} total images "
                f"(vision: {extraction_stats['vision_guided_count']}, "
                f"pymupdf: {extraction_stats['pymupdf_count']})"
            )

            return all_images, extraction_stats

        except Exception as e:
            raise PDFExtractionError(f"Streaming image extraction failed: {str(e)}") from e

    def _extract_batch_images(
        self,
        pdf_path: str,
        image_dir: str,
        batch_pages: List[int],
        job_id: Optional[str] = None,
        document_id: Optional[str] = None,
        converter: Optional[PageConverter] = None  # ✅ NEW: PageConverter for validation
    ) -> List[Dict[str, Any]]:
        """
        Extract images from a specific batch of pages with vision-guided extraction + PyMuPDF fallback.

        ✅ FULLY INTEGRATED with metadata propagation

        Architecture:
        1. Try vision-guided extraction first
        2. Fall back to PyMuPDF if vision fails or returns nothing
        
        Args:
            pdf_path: Path to PDF file
            image_dir: Directory to save extracted images
            batch_pages: List of page indices (0-based) to process
            job_id: Optional job ID for tracking
            document_id: Optional document ID
            converter: Optional PageConverter for page number validation

        Returns:
            List of extracted image data dictionaries
        """
        import fitz
        import gc
        from app.config import get_settings

        settings = get_settings()

        # Try vision-guided extraction first (if enabled)
        if settings.vision_guided_enabled:
            try:
                self.logger.info(f"   🔍 [Job: {job_id}] Attempting vision-guided extraction for pages {batch_pages}")
                vision_images = self._extract_batch_images_with_vision(
                    pdf_path, image_dir, batch_pages, job_id, document_id, converter  # ✅ Pass converter
                )
                if vision_images:
                    self.logger.info(
                        f"   ✅ [Job: {job_id}] Vision-guided extraction successful: "
                        f"{len(vision_images)} images extracted"
                    )
                    return vision_images
            except Exception as e:
                self.logger.warning(
                    f"   ⚠️ [Job: {job_id}] Vision-guided extraction failed: {e}. Falling back to PyMuPDF..."
                )

        # Fall back to PyMuPDF extraction
        self.logger.info(f"   📄 [Job: {job_id}] Using PyMuPDF extraction for pages {batch_pages}")
        return self._extract_batch_images_with_pymupdf(
            pdf_path, image_dir, batch_pages, job_id, document_id
        )

    def _extract_batch_images_with_vision(
        self,
        pdf_path: str,
        image_dir: str,
        batch_pages: List[int],
        job_id: Optional[str] = None,
        document_id: Optional[str] = None,
        converter: Optional[PageConverter] = None  # ✅ NEW: PageConverter
    ) -> List[Dict[str, Any]]:
        """
        Extract images using VisionGuidedExtractor with bounding boxes.

        ✅ INTEGRATED with job tracking and error logging

        Args:
            batch_pages: List of page indices (0-based) to process
            converter: PageConverter for page number validation and conversion

        Returns list of extracted images with vision metadata (bounding boxes, confidence).
        Raises exception if vision extraction fails.
        """
        from app.services.discovery.vision_guided_extractor import VisionGuidedExtractor
        from app.config import get_settings
        import asyncio

        settings = get_settings()
        extractor = VisionGuidedExtractor()

        extracted_images = []

        # Process each page with vision extraction
        for page_idx in batch_pages:
            try:
                # ✅ NEW: Convert array index to all page number formats
                if converter:
                    page = converter.from_array_index(page_idx)
                    catalog_page = page.catalog_page
                    pdf_page = page.pdf_page
                else:
                    # Fallback if converter not available
                    catalog_page = page_idx + 1
                    pdf_page = page_idx + 1

                self.logger.info(
                    f"   🔍 [Job: {job_id}] Vision extraction for catalog page {catalog_page} "
                    f"(PDF page {pdf_page}, index {page_idx})"
                )

                # Run async vision extraction in sync context
                result = asyncio.run(
                    extractor.extract_products_from_page(
                        pdf_path=pdf_path,
                        page_num=page_idx,  # PyMuPDF array index
                        catalog_page=catalog_page  # ✅ NEW: Pass catalog page for metadata
                    )
                )

                if not result.get('success', False):
                    self.logger.warning(
                        f"   ❌ [Job: {job_id}] Vision extraction failed for page {page_idx + 1}"
                    )
                    continue

                # Check confidence threshold
                confidence = result.get('confidence_score', 0.0)
                detections_count = len(result.get('detections', []))

                # ✅ FIX: Only warn if page has content but low confidence
                # Don't warn for empty pages (confidence = 0.0, detections = 0)
                if confidence < settings.vision_guided_confidence_threshold:
                    if detections_count == 0:
                        # Empty page - this is normal, not an error
                        self.logger.debug(
                            f"   📄 [Job: {job_id}] Page {page_idx + 1}: No products detected (empty page)"
                        )
                    else:
                        # Has detections but low confidence - this is concerning
                        self.logger.warning(
                            f"   ⚠️ [Job: {job_id}] Page {page_idx + 1}: confidence {confidence:.2f} "
                            f"below threshold {settings.vision_guided_confidence_threshold:.2f} "
                            f"({detections_count} detections with low confidence)"
                        )
                    continue

                # Extract detected product images with bounding boxes
                detections = result.get('detections', [])
                for detection_idx, detection in enumerate(detections):
                    try:
                        # Get bounding box coordinates
                        bbox = detection.get('bbox', [])

                        # Save image with vision metadata using precise cropping
                        image_filename = f"page_{page_idx + 1}_vision_detection_{detection_idx}.jpg"
                        image_path = os.path.join(image_dir, image_filename)

                        # ✅ FIX: Use precise cropping with PREMIUM quality (3.0x zoom)
                        crop_result = asyncio.run(extractor.crop_and_save_image(
                            pdf_path=pdf_path,
                            page_num=page_idx,
                            bbox=bbox,
                            output_path=image_path,
                            zoom=3.0  # 216 DPI for premium quality
                        ))

                        if not crop_result.get('success'):
                            self.logger.warning(f"   ⚠️ Failed to crop image for detection {detection_idx}")
                            continue

                        # ✅ NEW: Include complete page number metadata
                        img_meta = {
                            'filename': image_filename,
                            'path': image_path,
                            'page_number': page_idx + 1,  # 1-indexed for legacy compatibility
                            'catalog_page_number': catalog_page,  # ✅ NEW
                            'pdf_page_number': pdf_page,  # ✅ NEW
                            'array_index': page_idx,  # ✅ NEW
                            'bbox': bbox,
                            'confidence': detection.get('confidence', 0.0),
                            'detection_confidence': detection.get('confidence', 0.0),
                            'product_name': detection.get('product_name', 'Unknown'),
                            'description': detection.get('description', ''),
                            'detection_method': 'vision_guided',
                            'extraction_method': 'vision_guided',
                            'vision_provider': settings.vision_guided_provider,
                            'vision_model': settings.vision_guided_model,
                            'width': crop_result.get('width', 0),
                            'height': crop_result.get('height', 0),
                            'size_bytes': os.path.getsize(image_path) if os.path.exists(image_path) else 0
                        }

                        extracted_images.append(img_meta)

                        self.logger.debug(
                            f"   ✅ Extracted: {detection.get('product_name')} "
                            f"(confidence: {detection.get('confidence'):.2f})"
                        )

                    except Exception as detection_error:
                        self.logger.error(f"   ❌ Failed to process detection {detection_idx}: {detection_error}")
                        continue

            except Exception as page_error:
                self.logger.error(f"   ❌ Vision extraction failed for page {page_idx + 1}: {page_error}")
                continue

        return extracted_images

    def _extract_batch_images_with_pymupdf(
        self,
        pdf_path: str,
        image_dir: str,
        batch_pages: List[int],
        job_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract images using PyMuPDF (fallback method).

        ✅ INTEGRATED with job tracking and error logging

        Uses PyMuPDF directly to extract images with minimal memory footprint.

        NEW: If no embedded images found, renders entire page as image (for scanned PDFs).

        Returns:
            List of extracted image data
        """
        import fitz
        import gc
        import os

        doc = fitz.open(pdf_path)
        extracted_images = []

        try:
            for page_idx in batch_pages:
                if page_idx >= len(doc):
                    continue

                page = doc[page_idx]
                image_list = page.get_images(full=True)

                # ✅ IMPROVED LOGGING: Differentiate between pages with/without images
                if len(image_list) > 0:
                    self.logger.info(
                        f"   📄 [Job: {job_id}] PyMuPDF: Page {page_idx + 1} has {len(image_list)} embedded images"
                    )
                else:
                    self.logger.debug(
                        f"   📄 [Job: {job_id}] PyMuPDF: Page {page_idx + 1} has no embedded images (text-only or scanned page)"
                    )

                # ============================================================
                # CASE 1: Extract embedded images (normal PDFs)
                # ============================================================
                if len(image_list) > 0:
                    for img_idx, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]

                            # Save image to disk
                            image_filename = f"page_{page_idx + 1}_image_{img_idx}.{image_ext}"
                            image_path = os.path.join(image_dir, image_filename)

                            with open(image_path, "wb") as img_file:
                                img_file.write(image_bytes)

                            # Populate image metadata
                            extracted_images.append({
                                'path': image_path,
                                'filename': image_filename,
                                'page_number': page_idx + 1,
                                'extraction_method': 'pymupdf',
                                'format': image_ext,
                                'size_bytes': len(image_bytes),
                                'width': base_image.get('width'),
                                'height': base_image.get('height')
                            })

                            # Immediately free memory
                            del image_bytes, base_image

                            self.logger.debug(
                                f"   ✅ [Job: {job_id}] Extracted PyMuPDF image: {image_filename}"
                            )

                        except Exception as e:
                            self.logger.error(
                                f"   ❌ [Job: {job_id}] Failed to extract image {img_idx} "
                                f"from page {page_idx + 1}: {e}"
                            )
                            continue

                # ============================================================
                # CASE 2: No embedded images - use OCR to extract text from scanned page
                # (Common for scanned PDFs where each page IS an image)
                # ============================================================
                else:
                    try:
                        self.logger.info(
                            f"   📸 [Job: {job_id}] No embedded images on page {page_idx + 1} - "
                            f"using OCR to extract text from scanned page"
                        )

                        # Render page to high-res image for OCR
                        zoom = 2.0  # 144 DPI (72 * 2)
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat, alpha=False)

                        # Convert pixmap to PIL Image for OCR
                        from PIL import Image
                        import io
                        img_data = pix.tobytes('png')
                        pil_image = Image.open(io.BytesIO(img_data))

                        # Free pixmap memory immediately
                        pix = None

                        # Use OCR service to extract text
                        try:
                            from app.services.pdf.ocr_service import OCRService
                            ocr_service = OCRService()

                            # Extract text with bounding boxes
                            ocr_results = ocr_service.extract_text_from_image(
                                pil_image,
                                use_preprocessing=True
                            )

                            # Combine all text
                            page_text = " ".join([
                                r.text.strip()
                                for r in ocr_results
                                if r.text.strip() and r.confidence > 0.3
                            ])

                            if page_text:
                                # Save OCR text to a text file alongside images
                                text_filename = f"page_{page_idx + 1}_ocr.txt"
                                text_path = os.path.join(image_dir, text_filename)
                                with open(text_path, 'w', encoding='utf-8') as f:
                                    f.write(page_text)

                                self.logger.info(
                                    f"   ✅ [Job: {job_id}] OCR extracted {len(page_text)} characters "
                                    f"from page {page_idx + 1}"
                                )
                            else:
                                self.logger.warning(
                                    f"   ⚠️ [Job: {job_id}] OCR found no text on page {page_idx + 1}"
                                )

                            # Also save the full page image for reference
                            full_page_filename = f"page_{page_idx + 1}_scanned.jpg"
                            full_page_path = os.path.join(image_dir, full_page_filename)
                            pil_image.save(full_page_path, "JPEG", quality=85)

                            extracted_images.append({
                                'path': full_page_path,
                                'filename': full_page_filename,
                                'page_number': page_idx + 1,
                                'extraction_method': 'pymupdf_scanned_page',
                                'format': 'jpg',
                                'size_bytes': os.path.getsize(full_page_path),
                                'width': pil_image.width,
                                'height': pil_image.height
                            })

                        except Exception as ocr_error:
                            self.logger.error(f"   ❌ [Job: {job_id}] OCR Service failure: {ocr_error}")
                        finally:
                            pil_image = None

                    except Exception as e:
                        self.logger.error(
                            f"   ❌ [Job: {job_id}] Failed to process scanned page {page_idx + 1}: {e}"
                        )

                # Free page memory
                page = None
                gc.collect()

            self.logger.info(
                f"   ✅ [Job: {job_id}] PyMuPDF extraction complete: {len(extracted_images)} images extracted"
            )

        finally:
            doc.close()
            gc.collect()

        return extracted_images

    async def _process_batch_images(
        self,
        image_dir: str,
        document_id: str,
        extracted_images: List[Dict[str, Any]],
        processing_options: Dict[str, Any],
        progress_callback: Optional[callable],
        batch_num: int,
        total_batches: int
    ) -> List[Dict[str, Any]]:
        """
        Process and upload images for a single batch.

        ✅ INTEGRATED with metadata preservation:
        - Takes the actual image objects (with bboxes, labels)
        - Merges extraction metadata with processing results

        Args:
            image_dir: Directory containing images
            document_id: Document ID
            extracted_images: List of images from extraction stage
            processing_options: Processing configuration
            progress_callback: Callback for progress updates
            batch_num: Current batch number
            total_batches: Total number of batches

        Returns:
            List of processed and uploaded image data
        """
        import gc
        batch_images = []

        if not extracted_images:
            return batch_images

        self.logger.info(f"   📸 Processing {len(extracted_images)} images from batch {batch_num + 1}/{total_batches}")

        for idx, img_info in enumerate(extracted_images):
            image_path = img_info.get('path')
            filename = img_info.get('filename')

            if not image_path or not os.path.exists(image_path):
                self.logger.warning(f"   ⚠️ Image file not found: {image_path}")
                continue

            try:
                # Process and upload image (ALWAYS uploads to Supabase)
                processed_info = await self._process_extracted_image(
                    image_path,
                    document_id,
                    processing_options
                )

                if processed_info:
                    # ✅ MERGE metadata: Combine extraction metadata with processing info
                    # Extraction info (confidence, bbox, product_name) is preserved
                    merged_info = {**img_info, **processed_info}
                    batch_images.append(merged_info)

                # ALWAYS delete local file immediately after upload
                try:
                    os.remove(image_path)
                    self.logger.debug(f"   🗑️ Deleted local file: {filename}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete {image_path}: {e}")

                # Free memory after each image
                gc.collect()

            except Exception as e:
                self.logger.warning(f"Failed to process image {filename}: {e}")
                # Still try to delete the file to prevent accumulation
                try:
                    os.remove(image_path)
                except:
                    pass
                continue

        return batch_images

    
    async def _test_connection(self) -> dict:
        """
        Test basic functionality for health checks.
        
        Returns:
            Dict with test results
        """
        try:
            # Test basic imports and functionality
            import pymupdf4llm
            import tempfile
            import os
            
            # Create a minimal test to verify the service is working
            test_result = {
                "pymupdf4llm_available": True,
                "tempfile_access": os.access(tempfile.gettempdir(), os.W_OK),
                "thread_pool_active": self.executor is not None
            }
            
            return test_result
            
        except Exception as e:
            self.logger.error(f"Health check test failed: {str(e)}")
            return {
                "error": str(e),
                "pymupdf4llm_available": False,
                "tempfile_access": False,
                "thread_pool_active": False
            }
    
    def _create_temp_directory(self, document_id: str) -> str:
        """Create temporary directory for processing."""
        temp_dir = os.path.join(self.temp_dir_base, f"pdf_processor_{document_id}")
        os.makedirs(temp_dir, exist_ok=True)
        self.logger.debug("Created temporary directory: %s", temp_dir)
        return temp_dir
    
    def _cleanup_temp_files(self, temp_dir: str) -> None:
        """Clean up temporary files after processing."""
        try:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                self.logger.debug("Cleaned up temporary directory: %s", temp_dir)
        except Exception as e:
            self.logger.warning("Failed to cleanup temporary directory %s: %s", temp_dir, str(e))
    
    def _calculate_content_metrics(self, content: str) -> Dict[str, int]:
        """Calculate word count, character count, etc."""
        if not content:
            return {'word_count': 0, 'character_count': 0, 'line_count': 0}
        
        lines = content.split('\n')
        words = content.split()
        
        return {
            'word_count': len(words),
            'character_count': len(content),
            'line_count': len(lines)
        }
    
    async def _process_extracted_image(
        self,
        image_path: str,
        document_id: str,
        processing_options: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single extracted image with advanced capabilities and upload to Supabase Storage.

        NEW ARCHITECTURE:
        - ALWAYS uploads to Supabase immediately
        - Returns image data with storage_url for subsequent processing
        - No skip_upload parameter (deprecated)

        Features:
        - Format conversion and optimization
        - Metadata extraction (EXIF, technical specs)
        - Quality assessment
        - Image enhancement options
        - Upload to Supabase Storage instead of local storage
        - Duplicate detection preparation
        - Memory-efficient processing with explicit cleanup
        """
        import gc

        try:
            # Load image with PIL for metadata and basic processing
            with Image.open(image_path) as pil_image:
                # Extract basic metadata
                basic_info = {
                    'filename': os.path.basename(image_path),
                    'path': image_path,  # Keep local path for processing
                    'size_bytes': os.path.getsize(image_path),
                    'format': pil_image.format or 'UNKNOWN',
                    'mode': pil_image.mode,
                    'dimensions': pil_image.size,
                    'width': pil_image.width,
                    'height': pil_image.height
                }

                # Extract EXIF metadata if available
                exif_data = self._extract_exif_metadata(pil_image)
                
                # Load with OpenCV for advanced analysis (if available)
                quality_metrics = {'overall_score': 0.5}  # Default fallback
                image_hash = 'unavailable'
                enhanced_path = None

                if CV2_AVAILABLE:
                    try:
                        cv_image = cv2.imread(image_path)
                        if cv_image is not None:
                            # Calculate quality metrics
                            quality_metrics = self._calculate_image_quality(cv_image)

                            # Calculate image hash for duplicate detection
                            image_hash = self._calculate_image_hash(cv_image)

                            # Apply enhancements if requested
                            if processing_options.get('enhance_images', False):
                                enhanced_path = self._enhance_image(
                                    cv_image,
                                    image_path,
                                    processing_options
                                )
                        else:
                            self.logger.warning("Could not load image with OpenCV: %s", image_path)
                    except Exception as e:
                        self.logger.warning(f"OpenCV analysis failed for {image_path}: {e}")
                else:
                    self.logger.debug("OpenCV not available, using basic image analysis")

                # Convert format if requested (common for both CV2 available and not available)
                converted_path = None
                target_format = processing_options.get('target_format')
                if target_format and target_format.upper() != basic_info['format']:
                    converted_path = self._convert_image_format(
                        pil_image,
                        image_path,
                        target_format
                    )

                # Upload image to Supabase Storage
                # NOTE: skip_upload parameter is deprecated - we ALWAYS upload now
                upload_result = await self._upload_image_to_storage(
                    image_path,
                    document_id,
                    basic_info,
                    converted_path or enhanced_path
                )

                # DETAILED LOGGING: Log upload result for debugging
                self.logger.info(f"📤 Upload result for {basic_info['filename']}:")
                self.logger.info(f"   success: {upload_result.get('success')}")
                self.logger.info(f"   public_url: {upload_result.get('public_url')}")
                self.logger.info(f"   storage_path: {upload_result.get('storage_path')}")
                self.logger.info(f"   page_number: {upload_result.get('page_number')}")  # ✅ Log page number
                self.logger.info(f"   error: {upload_result.get('error')}")

                if not upload_result.get('success'):
                    self.logger.error(f"❌ CRITICAL: Failed to upload image to storage: {upload_result.get('error')}")
                    self.logger.error(f"   Image path: {image_path}")
                    self.logger.error(f"   Document ID: {document_id}")
                    # Continue processing even if upload fails, but mark it
                    upload_result = {'success': False, 'error': 'Upload failed', 'public_url': None}

                # ✅ CRITICAL FIX: Skip AI analysis during extraction to prevent blocking
                # AI analysis will be performed AFTER chunks/images are saved to database
                # This ensures data persistence even if AI processing fails or hangs
                real_analysis_data = {
                    'quality_score': quality_metrics['overall_score'],
                    'confidence_score': 0.5,
                    'analysis_pending': True,  # Flag to indicate analysis needs to be done later
                    'image_url': upload_result.get('public_url'),  # Store URL for later analysis
                    'document_id': document_id
                }

                self.logger.info(f"✅ Image uploaded to Supabase Storage: {basic_info['filename']}")

                # Combine all metadata with storage information
                # AI analysis results will be added later via async update
                result = {
                    **basic_info,
                    'exif': exif_data,
                    'quality_score': real_analysis_data.get('quality_score', quality_metrics['overall_score']),
                    'confidence_score': real_analysis_data.get('confidence_score', 0.5),
                    'quality_metrics': quality_metrics,
                    'image_hash': image_hash,
                    'enhanced_path': enhanced_path,
                    'converted_path': converted_path,
                    'processing_timestamp': datetime.utcnow().isoformat(),
                    # Storage information
                    'storage_uploaded': upload_result.get('success', False),
                    'storage_url': upload_result.get('public_url'),
                    'storage_path': upload_result.get('storage_path'),
                    'storage_bucket': upload_result.get('bucket', 'pdf-tiles'),
                        # This ensures the correct page number is saved to the database
                    'page_number': upload_result.get('page_number'),
                    # Analysis status (deferred for async processing)
                    'analysis_pending': real_analysis_data.get('analysis_pending', False),
                    'analysis_image_url': real_analysis_data.get('image_url'),
                    'analysis_document_id': real_analysis_data.get('document_id')
                }

                # Explicit memory cleanup for large images
                gc.collect()

                return result

        except Exception as e:
            self.logger.error("Error processing image %s: %s", image_path, str(e))
            gc.collect()  # Cleanup even on error
            return None

    async def _upload_image_to_storage(
        self,
        image_path: str,
        document_id: str,
        image_info: Dict[str, Any],
        processed_path: str = None
    ) -> Dict[str, Any]:
        """
        Upload extracted image to Supabase Storage.

        Args:
            image_path: Path to the original image file
            document_id: Document ID for organizing images
            image_info: Basic image information
            processed_path: Path to processed/enhanced image (if available)

        Returns:
            Dictionary with upload result
        """
        try:
            # Use processed image if available, otherwise use original
            upload_path = processed_path if processed_path and os.path.exists(processed_path) else image_path

            # Validate file exists
            if not os.path.exists(upload_path):
                raise FileNotFoundError(f"Image file not found: {upload_path}")

            # Read image file as bytes
            with open(upload_path, 'rb') as f:
                image_data = f.read()

            # Validate image_data is bytes
            if not isinstance(image_data, bytes):
                raise TypeError(f"Expected bytes, got {type(image_data).__name__}: {image_data}")

            self.logger.debug(f"Read {len(image_data)} bytes from {upload_path}")

            # Get Supabase client
            supabase_client = get_supabase_client()

            # Extract page number from filename if possible
            filename = os.path.basename(upload_path)
            page_number = None

            # Try to extract page number from filename patterns like "page_1_image_0.png"
            import re
            page_match = re.search(r'page[_-]?(\d+)', filename, re.IGNORECASE)
            if page_match:
                page_number = int(page_match.group(1))

            # Upload to Supabase Storage
            upload_result = await supabase_client.upload_image_file(
                image_data=image_data,
                filename=filename,
                document_id=document_id,
                page_number=page_number
            )

            if upload_result.get('success'):
                self.logger.info(f"Successfully uploaded image to storage: {upload_result.get('public_url')}")

                # NOTE: Temporary file cleanup moved to admin panel cron job

            return upload_result

        except Exception as e:
            self.logger.error(f"Failed to upload image to storage: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }



    def _extract_exif_metadata(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Extract EXIF metadata from PIL Image."""
        exif_data = {}
        try:
            if hasattr(pil_image, '_getexif') and pil_image._getexif() is not None:
                exif = pil_image._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
        except Exception as e:
            self.logger.debug("Could not extract EXIF data: %s", str(e))
        
        return exif_data
    
    def _calculate_image_quality(self, cv_image: np.ndarray) -> Dict[str, float]:
        """
        Calculate various image quality metrics using OpenCV and scikit-image.
        """
        if not CV2_AVAILABLE:
            return {
                'sharpness': 0.5,
                'contrast': 0.5,
                'brightness': 0.5,
                'noise_level': 0.5,
                'overall_score': 0.5
            }

        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize

            # Calculate contrast using standard deviation
            contrast_score = min(gray.std() / 128.0, 1.0)  # Normalize
            
            # Calculate brightness (mean intensity)
            brightness = gray.mean() / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Penalize extreme brightness
            
            # Calculate noise level using high-frequency content
            noise_level = filters.gaussian(gray, sigma=1).std()
            noise_score = max(0, 1.0 - noise_level / 50.0)  # Lower noise = higher score
            
            # Calculate overall quality score (weighted average)
            overall_score = (
                sharpness_score * 0.3 +
                contrast_score * 0.25 +
                brightness_score * 0.25 +
                noise_score * 0.2
            )
            
            return {
                'sharpness': float(sharpness_score),
                'contrast': float(contrast_score),
                'brightness': float(brightness_score),
                'noise': float(noise_score),
                'overall_score': float(overall_score)
            }
            
        except Exception as e:
            self.logger.error("Error calculating image quality: %s", str(e))
            return {
                'sharpness': 0.5,
                'contrast': 0.5,
                'brightness': 0.5,
                'noise': 0.5,
                'overall_score': 0.5
            }
    
    def _calculate_image_hash(self, cv_image: np.ndarray) -> str:
        """Calculate perceptual hash for duplicate detection."""
        if not CV2_AVAILABLE:
            return 'opencv_unavailable'

        try:
            # Resize to 8x8 for hash calculation
            small = cv2.resize(cv_image, (8, 8))
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            
            # Calculate average
            avg = gray_small.mean()
            
            # Create hash based on whether each pixel is above/below average
            hash_bits = []
            for row in gray_small:
                for pixel in row:
                    hash_bits.append('1' if pixel > avg else '0')
            
            # Convert to hexadecimal
            hash_str = hex(int(''.join(hash_bits), 2))[2:]
            return hash_str.zfill(16)  # Pad to 16 characters
            
        except Exception as e:
            self.logger.error("Error calculating image hash: %s", str(e))
            return "0000000000000000"
    
    def _enhance_image(
        self,
        cv_image: np.ndarray,
        original_path: str,
        processing_options: Dict[str, Any]
    ) -> Optional[str]:
        """Apply image enhancements and save enhanced version."""
        if not CV2_AVAILABLE:
            self.logger.warning("OpenCV not available, cannot enhance image")
            return None

        try:
            enhanced = cv_image.copy()
            
            # Apply sharpening if requested
            if processing_options.get('sharpen', True):
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # Apply contrast enhancement if requested
            if processing_options.get('enhance_contrast', True):
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Apply noise reduction if requested
            if processing_options.get('denoise', True):
                enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            
            # Save enhanced image
            base_name = os.path.splitext(original_path)[0]
            enhanced_path = f"{base_name}_enhanced.jpg"
            cv2.imwrite(enhanced_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return enhanced_path
            
        except Exception as e:
            self.logger.error("Error enhancing image: %s", str(e))
            return None
    
    def _convert_image_format(
        self,
        pil_image: Image.Image,
        original_path: str,
        target_format: str
    ) -> Optional[str]:
        """Convert image to target format with optimization."""
        try:
            base_name = os.path.splitext(original_path)[0]
            target_format = target_format.upper()
            
            # Determine file extension and save parameters
            if target_format == 'JPEG':
                converted_path = f"{base_name}_converted.jpg"
                save_kwargs = {'quality': 95, 'optimize': True}
            elif target_format == 'PNG':
                converted_path = f"{base_name}_converted.png"
                save_kwargs = {'optimize': True}
            elif target_format == 'WEBP':
                converted_path = f"{base_name}_converted.webp"
                save_kwargs = {'quality': 95, 'method': 6}
            else:
                self.logger.warning("Unsupported target format: %s", target_format)
                return None
            
            # Convert and save
            if pil_image.mode in ('RGBA', 'LA') and target_format == 'JPEG':
                # Convert RGBA to RGB for JPEG
                rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
                rgb_image.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode == 'RGBA' else None)
                rgb_image.save(converted_path, target_format, **save_kwargs)
            else:
                pil_image.save(converted_path, target_format, **save_kwargs)
            
            return converted_path
            
        except Exception as e:
            self.logger.error("Error converting image format: %s", str(e))
            return None
    
    def _remove_duplicate_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate images based on perceptual hash similarity."""
        if len(images) <= 1:
            return images
        
        unique_images = []
        seen_hashes = set()
        
        for image in images:
            image_hash = image.get('image_hash', '')
            
            # Check for exact hash matches
            if image_hash and image_hash not in seen_hashes:
                seen_hashes.add(image_hash)
                unique_images.append(image)
            elif not image_hash:
                # Keep images without hashes (fallback)
                unique_images.append(image)
        
        self.logger.info(
            "Duplicate removal: %d original images, %d unique images",
            len(images), len(unique_images)
        )
        
        return unique_images
    
    def _opencv_fast_text_detection(self, image_path: str) -> Dict[str, Any]:
        """
        Ultra-fast text detection using OpenCV edge detection and contour analysis.
        
        This is Phase 1 of the OCR filtering pipeline. It uses simple computer vision
        techniques to detect text-like patterns without running expensive OCR.
        
        Speed: ~0.1 seconds per image (300x faster than EasyOCR)
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict with:
                - has_text: Boolean indicating if text patterns detected
                - text_contours_count: Number of text-like contours found
                - confidence: Confidence score (0.0-1.0)
                - method: Detection method used
        """
        try:
            import cv2
            import numpy as np
            
            # Load image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                self.logger.warning(f"Failed to load image for OpenCV detection: {image_path}")
                return {
                    'has_text': True,  # Default to True to avoid false negatives
                    'text_contours_count': 0,
                    'confidence': 0.0,
                    'method': 'opencv_edge_detection',
                    'error': 'failed_to_load_image'
                }
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
            
            # Apply edge detection (Canny algorithm)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours (shapes in the image)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count text-like contours
            text_like_contours = 0
            total_contours = len(contours)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                
                # Text characteristics:
                # - Aspect ratio between 0.1 and 10 (not too wide, not too tall)
                # - Minimum size (width > 10px, height > 5px)
                # - Maximum size (not the entire image)
                # - Reasonable area (not too small, not too large)
                area = w * h
                image_area = img.shape[0] * img.shape[1]
                area_ratio = area / image_area if image_area > 0 else 0
                
                if (0.1 < aspect_ratio < 10 and 
                    w > 10 and h > 5 and 
                    w < img.shape[1] * 0.9 and h < img.shape[0] * 0.9 and
                    0.0001 < area_ratio < 0.5):
                    text_like_contours += 1
            
            # Decision threshold: If fewer than 10 text-like shapes, probably no text
            has_text = text_like_contours >= 10
            
            # Calculate confidence (normalize to 0-1)
            confidence = min(text_like_contours / 50, 1.0)
            
            self.logger.debug(
                f"OpenCV text detection: {text_like_contours} text-like contours "
                f"(total: {total_contours}) -> {'HAS TEXT' if has_text else 'NO TEXT'}"
            )
            
            return {
                'has_text': has_text,
                'text_contours_count': text_like_contours,
                'total_contours': total_contours,
                'confidence': confidence,
                'method': 'opencv_edge_detection'
            }
            
        except Exception as e:
            self.logger.error(f"OpenCV text detection failed: {str(e)}")
            # On error, default to True to avoid false negatives
            return {
                'has_text': True,
                'text_contours_count': 0,
                'confidence': 0.0,
                'method': 'opencv_edge_detection',
                'error': str(e)
            }

    async def _should_image_have_ocr(self, image_path: str) -> Dict[str, Any]:
        """
        AI-powered decision on whether an image needs OCR processing.
        
        Uses CLIP embeddings to classify images as:
        - RELEVANT: Product specs, dimensions, technical data, material properties
        - IRRELEVANT: Historical photos, biographies, decorative images, mood boards
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict with 'should_process' (bool), 'reason' (str), 'confidence' (float)
        """
        try:
            from PIL import Image
            import torch
            
            # Load image
            if not os.path.exists(image_path):
                return {'should_process': False, 'reason': 'file_not_found', 'confidence': 0.0}
            
            image = Image.open(image_path).convert('RGB')
            
            # Initialize SigLIP model for OCR filtering (cached after first use)
            # Use transformers directly instead of sentence-transformers to avoid 'hidden_size' error
            from transformers import AutoModel, AutoProcessor
            import torch
            import numpy as np

            if not hasattr(self, '_siglip_model_for_ocr'):
                self.logger.info("🔄 Loading SigLIP2 model for OCR filtering: google/siglip2-so400m-patch14-384")
                self._siglip_model_for_ocr = AutoModel.from_pretrained('google/siglip2-so400m-patch14-384')
                self._siglip_processor_for_ocr = AutoProcessor.from_pretrained('google/siglip2-so400m-patch14-384')
                self._siglip_model_for_ocr.eval()
                self.logger.info("✅ Initialized SigLIP2 model for OCR filtering")

            # Define text prompts for classification
            relevant_prompts = [
                "product specification table with dimensions and measurements",
                "technical data sheet with material properties and numbers",
                "dimension annotations and measurements on product image",
                "material property chart or graph with data",
                "product label with technical specifications and text",
                "technical drawing with dimension callouts and numbers",
                "size chart or measurement guide with numbers",
                "product features list with specifications",
                "CAD drawing with dimension annotations",
                "engineering diagram with measurements"
            ]

            irrelevant_prompts = [
                "historical photograph of people without technical content",
                "biography or portrait photo with captions",
                "decorative mood board or lifestyle image",
                "artistic photography without text or labels",
                "interior design scene without specifications or measurements",
                "pure product photo without text labels or annotations",
                "texture or pattern sample without text",
                "company history or timeline image",
                "inspirational quote or decorative text",
                "brand logo or company name only",
                "artistic typography without technical information"
            ]

            # Get image embedding using transformers directly
            with torch.no_grad():
                inputs = self._siglip_processor_for_ocr(images=image, return_tensors="pt")
                image_features = self._siglip_model_for_ocr.get_image_features(**inputs)

                # L2 normalize to unit vector
                image_embedding_tensor = image_features / image_features.norm(dim=-1, keepdim=True)
                image_embedding_raw = image_embedding_tensor.squeeze().cpu().numpy()

            image_embedding = image_embedding_raw.tolist() if hasattr(image_embedding_raw, 'tolist') else list(image_embedding_raw)

            # Get text embeddings for all prompts using transformers directly
            all_prompts = relevant_prompts + irrelevant_prompts
            text_embeddings = []

            with torch.no_grad():
                for prompt in all_prompts:
                    text_inputs = self._siglip_processor_for_ocr(text=prompt, return_tensors="pt")
                    text_features = self._siglip_model_for_ocr.get_text_features(**text_inputs)

                    # L2 normalize to unit vector
                    text_emb_tensor = text_features / text_features.norm(dim=-1, keepdim=True)
                    text_emb_raw = text_emb_tensor.squeeze().cpu().numpy()
                    text_emb_list = text_emb_raw.tolist() if hasattr(text_emb_raw, 'tolist') else list(text_emb_raw)
                    text_embeddings.append(text_emb_list)
            
            # Calculate similarities
            image_embedding_tensor = torch.tensor(image_embedding).unsqueeze(0)
            text_embeddings_tensor = torch.tensor(text_embeddings)
            
            # Cosine similarity
            similarities = torch.nn.functional.cosine_similarity(
                image_embedding_tensor,
                text_embeddings_tensor
            )
            
            # Split similarities
            relevant_similarities = similarities[:len(relevant_prompts)]
            irrelevant_similarities = similarities[len(relevant_prompts):]
            
            # Calculate scores
            relevant_score = relevant_similarities.max().item()
            irrelevant_score = irrelevant_similarities.max().item()
            
            # Decision logic
            # If relevant score is significantly higher, process with OCR
            score_diff = relevant_score - irrelevant_score
            
            # STRICT THRESHOLDS: Only process images with high confidence of technical content
            # relevant_score > 0.35 = Must have strong similarity to technical prompts (was 0.25)
            # score_diff > 0.15 = Must be significantly more technical than decorative (was 0.05)
            if relevant_score > 0.35 and score_diff > 0.15:
                # Image likely contains technical/specification content
                should_process = True
                reason = f"technical_content (relevant: {relevant_score:.3f}, irrelevant: {irrelevant_score:.3f})"
                confidence = relevant_score
            else:
                # Image is likely decorative/historical
                should_process = False
                reason = f"decorative_content (relevant: {relevant_score:.3f}, irrelevant: {irrelevant_score:.3f})"
                confidence = irrelevant_score
            
            return {
                'should_process': should_process,
                'reason': reason,
                'confidence': confidence,
                'relevant_score': relevant_score,
                'irrelevant_score': irrelevant_score
            }
            
        except Exception as e:
            self.logger.warning(f"OCR classification failed for {image_path}: {e}")
            # Default to processing if classification fails (safe fallback)
            return {
                'should_process': True,
                'reason': f'classification_error: {str(e)}',
                'confidence': 0.5
            }

    async def _process_images_with_ocr(
        self,
        extracted_images: List[Dict[str, Any]],
        ocr_languages: List[str],
        progress_callback: Optional[Callable] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process extracted images with OCR using the OCR service.
        Now includes AI-powered filtering to only process relevant images.

        Args:
            extracted_images: List of image dictionaries with metadata
            ocr_languages: List of language codes for OCR processing
            progress_callback: Optional callback to report progress

        Returns:
            Tuple of (combined_ocr_text, ocr_results_list)
        """
        try:
            ocr_service = get_ocr_service(OCRConfig(languages=ocr_languages))

            combined_ocr_text = ""
            ocr_results = []
            total_images = len(extracted_images)
            
            # PHASE 1: OpenCV Fast Text Detection (0.1s per image)
            self.logger.info(f"🔍 Phase 1: OpenCV fast text detection on {total_images} images")
            opencv_passed = []
            opencv_skipped = []
            
            for idx, image_data in enumerate(extracted_images):
                image_path = image_data.get('path')
                if not image_path or not os.path.exists(image_path):
                    continue
                
                # Use OpenCV to quickly detect text patterns
                opencv_result = self._opencv_fast_text_detection(image_path)
                
                if opencv_result['has_text']:
                    opencv_passed.append({
                        'data': image_data,
                        'opencv_result': opencv_result
                    })
                    self.logger.debug(
                        f"  ✅ Image {idx+1}/{total_images}: OpenCV detected {opencv_result['text_contours_count']} text-like contours"
                    )
                else:
                    opencv_skipped.append({
                        'data': image_data,
                        'opencv_result': opencv_result,
                        'skip_reason': 'opencv_no_text',
                        'metadata': {
                            'ocr_status': 'skipped',
                            'skip_reason': 'opencv_no_text',
                            'text_contours_count': opencv_result['text_contours_count'],
                            'can_reprocess': True
                        }
                    })
                    self.logger.debug(
                        f"  ⏭️  Image {idx+1}/{total_images}: OpenCV SKIPPED - only {opencv_result['text_contours_count']} text-like contours"
                    )
            
            self.logger.info(
                f"🎯 Phase 1 Results: {len(opencv_passed)} images with text patterns, "
                f"{len(opencv_skipped)} skipped (no text detected)"
            )
            
            # PHASE 2: CLIP AI Classification (0.5s per image) - only on images that passed Phase 1
            self.logger.info(f"🤖 Phase 2: CLIP AI classification on {len(opencv_passed)} images")
            images_to_process = []
            clip_skipped = []
            
            for idx, item in enumerate(opencv_passed):
                image_data = item['data']
                image_path = image_data.get('path')
                
                # Use CLIP AI to classify if text is technical or decorative
                clip_decision = await self._should_image_have_ocr(image_path)
                
                if clip_decision['should_process']:
                    images_to_process.append({
                        'data': image_data,
                        'opencv_result': item['opencv_result'],
                        'clip_decision': clip_decision
                    })
                    self.logger.debug(
                        f"  ✅ Image {idx+1}/{len(opencv_passed)}: {clip_decision['reason']}"
                    )
                else:
                    clip_skipped.append({
                        'data': image_data,
                        'opencv_result': item['opencv_result'],
                        'clip_decision': clip_decision,
                        'skip_reason': 'clip_decorative',
                        'metadata': {
                            'ocr_status': 'skipped',
                            'skip_reason': 'clip_decorative',
                            'relevant_score': clip_decision.get('relevant_score', 0),
                            'irrelevant_score': clip_decision.get('irrelevant_score', 0),
                            'can_reprocess': True
                        }
                    })
                    self.logger.debug(
                        f"  ⏭️  Image {idx+1}/{len(opencv_passed)}: CLIP SKIPPED - {clip_decision['reason']}"
                    )
            
            # Combine all skipped images
            images_skipped = opencv_skipped + clip_skipped
            
            self.logger.info(
                f"🎯 Phase 2 Results: {len(images_to_process)} technical images, "
                f"{len(clip_skipped)} decorative images skipped"
            )
            self.logger.info(
                f"📊 Total Filtering: {len(images_to_process)}/{total_images} images will be processed with OCR "
                f"({len(images_skipped)} skipped: {len(opencv_skipped)} no text, {len(clip_skipped)} decorative)"
            )
            
            # PHASE 3: Full EasyOCR Processing (30s per image) - only on images that passed both filters
            self.logger.info(f"📝 Phase 3: Running EasyOCR on {len(images_to_process)} filtered images")
            
            for idx, item in enumerate(images_to_process):
                image_data = item['data']
                ocr_decision = item['clip_decision']
                image_path = image_data.get('path')

                try:
                    # Report progress
                    if progress_callback:
                        progress_percent = 25 + int((idx / len(images_to_process)) * 15) if len(images_to_process) > 0 else 25
                        if inspect.iscoroutinefunction(progress_callback):
                            await progress_callback(
                                progress=progress_percent,
                                details={
                                    "current_step": f"Processing image {idx + 1}/{len(images_to_process)} with OCR (AI-filtered)",
                                    "images_processed": idx + 1,
                                    "total_images": len(images_to_process),
                                    "images_skipped": len(images_skipped)
                                }
                            )
                        else:
                            progress_callback(
                                progress=progress_percent,
                                details={
                                    "current_step": f"Processing image {idx + 1}/{len(images_to_process)} with OCR (AI-filtered)",
                                    "images_processed": idx + 1,
                                    "total_images": len(images_to_process),
                                    "images_skipped": len(images_skipped)
                                }
                            )

                    # Process image with OCR using the correct method
                    loop = asyncio.get_event_loop()
                    ocr_result_list = await loop.run_in_executor(
                        None,
                        ocr_service.extract_text_from_image,
                        image_path
                    )

                    if ocr_result_list:
                        # Combine all extracted text from the image
                        extracted_text = " ".join([result.text for result in ocr_result_list])
                        avg_confidence = sum([result.confidence for result in ocr_result_list]) / len(ocr_result_list) if ocr_result_list else 0.0

                        combined_ocr_text += extracted_text + "\n"
                        ocr_results.append({
                            'image_path': image_path,
                            'text': extracted_text,
                            'confidence': avg_confidence,
                            'language': ocr_languages[0] if ocr_languages else 'en',
                            'regions_detected': len(ocr_result_list),
                            'ai_classification': ocr_decision
                        })
                        
                        self.logger.info(f"  ✅ OCR extracted {len(ocr_result_list)} text regions from image {idx+1}")
                    else:
                        self.logger.info(f"  ℹ️  No text found in image {idx+1}")

                except Exception as e:
                    self.logger.warning("OCR processing failed for image %s: %s", image_path, str(e))
                    ocr_results.append({
                        'image_path': image_path,
                        'text': '',
                        'confidence': 0.0,
                        'language': 'unknown',
                        'error': str(e),
                        'ai_classification': ocr_decision
                    })
            
            # Add skipped images to results with metadata
            for item in images_skipped:
                image_data = item['data']
                skip_reason = item.get('skip_reason', 'unknown')
                metadata = item.get('metadata', {})
                
                # Build comprehensive metadata for skipped images
                skip_metadata = {
                    'ocr_status': 'skipped',
                    'skip_reason': skip_reason,
                    'can_reprocess': True
                }
                
                # Add OpenCV results if available
                if 'opencv_result' in item:
                    skip_metadata['opencv_detection'] = {
                        'text_contours_count': item['opencv_result'].get('text_contours_count', 0),
                        'confidence': item['opencv_result'].get('confidence', 0.0)
                    }
                
                # Add CLIP results if available
                if 'clip_decision' in item:
                    skip_metadata['clip_classification'] = {
                        'relevant_score': item['clip_decision'].get('relevant_score', 0.0),
                        'irrelevant_score': item['clip_decision'].get('irrelevant_score', 0.0),
                        'reason': item['clip_decision'].get('reason', '')
                    }
                
                ocr_results.append({
                    'image_path': image_data.get('path'),
                    'text': '',
                    'confidence': 0.0,
                    'language': 'skipped',
                    'regions_detected': 0,
                    'skipped': True,
                    'skip_metadata': skip_metadata
                })
            
            self.logger.info(f"✅ OCR Processing Complete: {len(images_to_process)} processed, {len(images_skipped)} skipped")

            return combined_ocr_text.strip(), ocr_results

        except Exception as e:
            self.logger.error("Error in OCR processing: %s", str(e))
            return "", []


