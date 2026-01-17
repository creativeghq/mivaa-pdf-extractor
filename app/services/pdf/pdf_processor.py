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
import imagehash  # For perceptual hash deduplication (Layer 4)

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
# ✅ REMOVED extract_pdf_tables - now using TableExtractor class from table_extraction.py
try:
    # Try to import from the proper location first
    from ..core.extractor import extract_pdf_to_markdown, extract_pdf_to_markdown_with_doc, extract_json_and_images
except ImportError:
    # Fall back to the root level extractor if it exists
    try:
        from app.core.extractor import extract_pdf_to_markdown, extract_pdf_to_markdown_with_doc, extract_json_and_images
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

# PageConverter removed - using simple PDF page numbers instead


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

        # ⚡ OPTIMIZED: Initialize THREAD pool executor for async processing
        # NOTE: Using ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickle issues
        # with pydantic Settings and other non-picklable objects
        # Default increased from 2 to 4 - modern CPUs handle this well
        # For I/O-bound operations (PDF reading), higher concurrency improves throughput
        max_workers = self.config.get('max_workers', 4)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.logger.info("PDFProcessor initialized with config: %s (max_workers=%d)", self.config, max_workers)

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
            # ✅ FIX: Make markdown extraction conditional
            # If extract_text is False, skip markdown extraction (for image-only processing)
            extract_text = processing_options.get('extract_text', True)

            markdown_content = ""
            metadata = {}
            page_chunks = None

            if extract_text:
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
            else:
                # Image-only processing - get minimal metadata without text extraction
                self.logger.info("📄 Skipping text extraction (extract_text=False) - extracting images only")
                import fitz
                doc = fitz.open(pdf_path)
                metadata = {
                    'page_count': len(doc),
                    'file_size': os.path.getsize(pdf_path)
                }
                doc.close()
            
            # Extract images if requested
            extracted_images = []
            extraction_stats = {'pymupdf_count': 0, 'failed_count': 0, 'total_pages': 0}

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

        Returns:
            Tuple of (images_list, extraction_stats)
            - images_list: List of image dictionaries
            - extraction_stats: Dict with pymupdf_count, failed_count, total_pages
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

            self.logger.info(f"📐 PDF has {total_pages} pages")

            # Determine which pages to process
            if page_list:

                self.logger.info(f"   📋 Received page_list (1-based PDF pages): {page_list}")

                # Validate pages are within bounds
                pages_to_process = []
                invalid_pages = []
                for pdf_page in page_list:
                    if pdf_page > 0 and pdf_page <= total_pages:
                        pages_to_process.append(pdf_page - 1)  # Convert to 0-based
                    else:
                        invalid_pages.append(pdf_page)

                self.logger.info(f"   ✅ Valid PDF pages converted to array indices (0-based): {pages_to_process}")

                if invalid_pages:
                    self.logger.warning(
                        f"   ⚠️ Filtered {len(invalid_pages)} out-of-bounds PDF pages: {invalid_pages}"
                    )
                    self.logger.warning(f"      PDF has {total_pages} pages total")
            else:
                pages_to_process = list(range(total_pages))
                self.logger.info(f"   📋 No page_list provided - processing all {total_pages} pages")


            self.logger.info(f"🔄 STREAMING IMAGE EXTRACTION: {len(pages_to_process)} pages in batches of {batch_size}")
            self.logger.info(f"   Pages to process (array indices): {pages_to_process}")

            # Process pages in small batches with aggressive memory cleanup
            loop = asyncio.get_event_loop()
            all_images = []

            # Track extraction method for metrics (4-layer cascade)
            extraction_stats = {
                'embedded_count': 0,
                'yolo_crop_count': 0,
                'full_render_count': 0,
                'pymupdf_count': 0,  # Keep for backward compatibility
                'failed_count': 0,
                'total_pages': len(pages_to_process),
                'duplicates_removed': 0
            }

            for batch_num, batch_start in enumerate(range(0, len(pages_to_process), batch_size)):
                batch_end = min(batch_start + batch_size, len(pages_to_process))
                batch_pages = pages_to_process[batch_start:batch_end]

                self.logger.info(f"   📦 Batch {batch_num + 1}: Processing pages {batch_pages}")

                # Extract images for THIS BATCH (now async with 4-layer cascade)
                batch_extracted_images = await self._extract_batch_images(
                    pdf_path,
                    image_dir,
                    batch_pages,
                    job_id,
                    document_id
                )

                # Update extraction stats based on extracted images (4-layer tracking)
                for img in batch_extracted_images:
                    layer = img.get('extraction_layer', 'embedded')
                    if layer == 'embedded':
                        extraction_stats['embedded_count'] += 1
                    elif layer == 'yolo_crop':
                        extraction_stats['yolo_crop_count'] += 1
                    elif layer == 'full_render':
                        extraction_stats['full_render_count'] += 1

                    # Backward compatibility
                    extraction_stats['pymupdf_count'] += 1

                    # Track duplicates
                    if img.get('is_duplicate', False):
                        extraction_stats['duplicates_removed'] += 1

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
                f"✅ 4-LAYER EXTRACTION COMPLETE: {len(all_images)} total images "
                f"(embedded: {extraction_stats['embedded_count']}, "
                f"yolo_crop: {extraction_stats['yolo_crop_count']}, "
                f"full_render: {extraction_stats['full_render_count']}, "
                f"duplicates_removed: {extraction_stats['duplicates_removed']})"
            )

            return all_images, extraction_stats

        except Exception as e:
            raise PDFExtractionError(f"Streaming image extraction failed: {str(e)}") from e

    async def _extract_batch_images(
        self,
        pdf_path: str,
        image_dir: str,
        batch_pages: List[int],
        job_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract images from a specific batch of pages with 4-layer cascade.

        ✅ 4-LAYER EXTRACTION CASCADE:
        1. Layer 1: Embedded images (PyMuPDF) - Fast, extracts actual embedded images
        2. Layer 2: YOLO-guided cropping - Detects IMAGE regions and crops from rendered page
        3. Layer 3: Full page render - Fallback for scanned PDFs and vector graphics
        4. Layer 4: Perceptual hash deduplication - Remove duplicates across all layers

        Args:
            pdf_path: Path to PDF file
            image_dir: Directory to save extracted images
            batch_pages: List of page indices (0-based) to process
            job_id: Optional job ID for tracking
            document_id: Optional document ID

        Returns:
            List of extracted image data dictionaries (deduplicated)
        """
        from app.config import get_settings
        settings = get_settings()

        all_images = []

        self.logger.info(
            f"   🎯 [Job: {job_id}] Starting 4-layer extraction for pages {batch_pages}"
        )

        # Layer 1: Try embedded images first (fastest)
        self.logger.info(f"   📄 [Job: {job_id}] Layer 1: Extracting embedded images...")
        embedded_images = self._extract_batch_images_with_pymupdf(
            pdf_path, image_dir, batch_pages, job_id, document_id
        )

        # Mark extraction layer
        for img in embedded_images:
            img['extraction_layer'] = 'embedded'

        all_images.extend(embedded_images)

        self.logger.info(
            f"   ✅ [Job: {job_id}] Layer 1 complete: {len(embedded_images)} embedded images"
        )

        # Layer 2: Try YOLO-guided extraction (if enabled and no embedded images)
        if settings.yolo_enabled:
            self.logger.info(f"   🎯 [Job: {job_id}] Layer 2: YOLO-guided extraction...")
            yolo_images = await self._extract_batch_images_with_yolo(
                pdf_path, image_dir, batch_pages, job_id, document_id
            )

            # Mark extraction layer
            for img in yolo_images:
                img['extraction_layer'] = 'yolo_crop'

            all_images.extend(yolo_images)

            self.logger.info(
                f"   ✅ [Job: {job_id}] Layer 2 complete: {len(yolo_images)} YOLO-cropped images"
            )
        else:
            self.logger.info(f"   ⚠️ [Job: {job_id}] Layer 2 skipped: YOLO disabled")

        # Layer 3: Full page render is already handled in PyMuPDF method
        # (it renders full page if no embedded images found)

        # Layer 4: Deduplicate across all layers
        self.logger.info(
            f"   🔍 [Job: {job_id}] Layer 4: Deduplicating {len(all_images)} images..."
        )
        unique_images = self._deduplicate_images(all_images, threshold=5, job_id=job_id)

        self.logger.info(
            f"   ✅ [Job: {job_id}] 4-layer extraction complete: "
            f"{len(unique_images)} unique images "
            f"(embedded: {len(embedded_images)}, yolo: {len(all_images) - len(embedded_images)}, "
            f"duplicates removed: {len(all_images) - len(unique_images)})"
        )

        return unique_images

    async def _extract_batch_images_with_yolo(
        self,
        pdf_path: str,
        image_dir: str,
        batch_pages: List[int],
        job_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract images using YOLO layout detection (Layer 2).

        ✅ INTEGRATED with job tracking and error logging

        Uses YOLO DocParser to detect IMAGE regions and crops them from rendered pages.
        This is more accurate than embedded extraction for vector graphics and complex layouts.

        Returns:
            List of extracted image data with YOLO metadata
        """
        from app.services.pdf.yolo_layout_detector import YoloLayoutDetector
        from app.config import get_settings
        import fitz
        import gc

        settings = get_settings()

        # Check if YOLO is enabled
        if not settings.yolo_enabled:
            self.logger.info(f"   ⚠️ [Job: {job_id}] YOLO layout detection disabled")
            return []

        extracted_images = []

        try:
            # Initialize YOLO detector
            yolo_detector = YoloLayoutDetector()

            # Process each page
            for page_idx in batch_pages:
                try:
                    # PDF page number (1-based)
                    pdf_page = page_idx + 1

                    self.logger.info(
                        f"   🎯 [Job: {job_id}] YOLO detecting layout on PDF page {pdf_page}..."
                    )

                    # Detect layout regions
                    layout_result = await yolo_detector.detect_layout_regions(
                        pdf_path=pdf_path,
                        page_num=page_idx,
                        dpi=150
                    )

                    # Get IMAGE regions only
                    image_regions = layout_result.get_regions_by_type("IMAGE")

                    if not image_regions:
                        self.logger.info(
                            f"   ℹ️ [Job: {job_id}] No IMAGE regions detected on PDF page {pdf_page}"
                        )
                        continue

                    self.logger.info(
                        f"   ✅ [Job: {job_id}] Found {len(image_regions)} IMAGE regions on PDF page {pdf_page}"
                    )

                    # Render full page for cropping
                    doc = fitz.open(pdf_path)
                    page = doc[page_idx]

                    # Render at high DPI for quality
                    zoom = 150 / 72  # 150 DPI
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)

                    # Convert to PIL Image
                    from PIL import Image
                    import io
                    img_data = pix.tobytes("png")
                    full_page_image = Image.open(io.BytesIO(img_data))

                    # Crop each IMAGE region
                    for region_idx, region in enumerate(image_regions):
                        try:
                            # Get bounding box coordinates
                            bbox = region.bbox

                            # Crop region from full page image
                            cropped_image = full_page_image.crop((
                                bbox.x,
                                bbox.y,
                                bbox.x2,
                                bbox.y2
                            ))

                            # Save cropped image
                            image_filename = f"page_{pdf_page}_yolo_region_{region_idx}.jpg"
                            image_path = os.path.join(image_dir, image_filename)

                            cropped_image.save(image_path, "JPEG", quality=95)

                            # Get image dimensions
                            width, height = cropped_image.size

                            # Create image metadata (using PDF page number)
                            image_info = {
                                'path': image_path,
                                'filename': image_filename,
                                'page_number': pdf_page,
                                'width': width,
                                'height': height,
                                'format': 'JPEG',
                                'detection_method': 'yolo_guided',
                                'extraction_layer': 'yolo_crop',
                                'yolo_confidence': region.confidence,
                                'yolo_region_type': region.type,
                                'yolo_reading_order': region.reading_order,
                                'bbox': {
                                    'x': bbox.x,
                                    'y': bbox.y,
                                    'width': bbox.width,
                                    'height': bbox.height
                                }
                            }

                            extracted_images.append(image_info)

                            self.logger.debug(
                                f"   ✅ [Job: {job_id}] Extracted YOLO region {region_idx} "
                                f"from PDF page {pdf_page} (confidence: {region.confidence:.2f})"
                            )

                        except Exception as e:
                            self.logger.error(
                                f"   ❌ [Job: {job_id}] Failed to crop YOLO region {region_idx} "
                                f"on PDF page {pdf_page}: {e}"
                            )
                            continue

                    # Cleanup
                    full_page_image.close()
                    doc.close()
                    gc.collect()

                except Exception as e:
                    self.logger.error(
                        f"   ❌ [Job: {job_id}] YOLO extraction failed for PDF page {pdf_page}: {e}"
                    )
                    continue

            self.logger.info(
                f"   ✅ [Job: {job_id}] YOLO extraction complete: {len(extracted_images)} images extracted"
            )

            # Pause YOLO endpoint after batch
            yolo_detector.pause_endpoint()

        except Exception as e:
            self.logger.error(f"   ❌ [Job: {job_id}] YOLO batch extraction failed: {e}")

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

                # PDF page number (1-based)
                pdf_page = page_idx + 1

                page = doc[page_idx]
                image_list = page.get_images(full=True)

                # Log page info
                if len(image_list) > 0:
                    self.logger.info(
                        f"   📄 [Job: {job_id}] PyMuPDF: PDF page {pdf_page} has {len(image_list)} embedded images"
                    )
                else:
                    self.logger.info(
                        f"   📄 [Job: {job_id}] PyMuPDF: PDF page {pdf_page} has NO embedded images"
                    )
                    self.logger.info(
                        f"      This could mean: text-only page, scanned page, or vector graphics"
                    )

                # ============================================================
                # LAYER 1: Extract embedded images (normal PDFs)
                # ============================================================
                if len(image_list) > 0:
                    for img_idx, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]

                            # Save image to disk
                            image_filename = f"page_{pdf_page}_image_{img_idx}.{image_ext}"
                            image_path = os.path.join(image_dir, image_filename)

                            with open(image_path, "wb") as img_file:
                                img_file.write(image_bytes)

                            # Populate image metadata with Layer 1 information (using PDF page number)
                            extracted_images.append({
                                'path': image_path,
                                'filename': image_filename,
                                'page_number': pdf_page,
                                'extraction_method': 'pymupdf_embedded',  # Layer 1: Embedded images
                                'layer': 1,
                                'captures_vector_graphics': True,  # Embedded images don't capture vector graphics
                                'format': image_ext,
                                'size_bytes': len(image_bytes),
                                'width': base_image.get('width'),
                                'height': base_image.get('height')
                            })

                            # Immediately free memory
                            del image_bytes, base_image

                            self.logger.info(
                                f"   ✅ [Job: {job_id}] Extracted image {img_idx + 1}/{len(image_list)}: {image_filename} "
                                f"({extracted_images[-1]['width']}x{extracted_images[-1]['height']}, "
                                f"{extracted_images[-1]['size_bytes']} bytes)"
                            )

                        except Exception as e:
                            self.logger.error(
                                f"   ❌ [Job: {job_id}] Failed to extract image {img_idx} "
                                f"from PDF page {pdf_page}: {e}"
                            )
                            continue

                # ============================================================
                # LAYER 2: Full Page Rendering (for vector graphics)
                # Only render if Layer 1 found 0 embedded images
                # ============================================================
                else:
                    try:
                        self.logger.info(
                            f"   📸 [Job: {job_id}] No embedded images on PDF page {pdf_page} - "
                            f"rendering full page to capture vector graphics"
                        )

                        # Render page to high-res image (2x zoom = 144 DPI)
                        zoom = 2.0
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat, alpha=False)

                        # Convert pixmap to PIL Image
                        from PIL import Image
                        import io
                        img_data = pix.tobytes('jpeg')
                        pil_image = Image.open(io.BytesIO(img_data))

                        # Save full page render
                        full_page_filename = f"page_{pdf_page}_full_render.jpg"
                        full_page_path = os.path.join(image_dir, full_page_filename)
                        pil_image.save(full_page_path, "JPEG", quality=85)

                        # Get file size
                        file_size = os.path.getsize(full_page_path)

                        # Add to extracted images with Layer 2 metadata (using PDF page number)
                        extracted_images.append({
                            'path': full_page_path,
                            'filename': full_page_filename,
                            'page_number': pdf_page,
                            'extraction_method': 'pymupdf_full_render',  # Layer 2: Full page render
                            'layer': 2,
                            'captures_vector_graphics': True,  # Full render captures vector graphics
                            'format': 'jpg',
                            'size_bytes': file_size,
                            'width': pil_image.width,
                            'height': pil_image.height
                        })

                        # Free memory immediately
                        pix = None
                        pil_image = None

                        self.logger.info(
                            f"   ✅ [Job: {job_id}] Full page render saved: {full_page_filename}"
                        )

                    except Exception as e:
                        self.logger.error(
                            f"   ❌ [Job: {job_id}] Failed to render full PDF page {pdf_page}: {e}"
                        )
                        continue


                # Free page memory
                page = None
                gc.collect()

            self.logger.info(
                f"   ✅ [Job: {job_id}] PyMuPDF extraction complete: {len(extracted_images)} images extracted"
            )

        finally:
            doc.close()
            gc.collect()

        # Layer 4: Deduplicate images using perceptual hashing
        extracted_images = self._deduplicate_images(extracted_images, threshold=5, job_id=job_id)

        return extracted_images

    def _deduplicate_images(
        self,
        extracted_images: List[Dict[str, Any]],
        threshold: int = 5,
        job_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Layer 4: Deduplicate images using perceptual hashing.

        Uses imagehash.phash() to compute perceptual hashes and removes duplicates
        based on Hamming distance threshold.

        Args:
            extracted_images: List of image dictionaries with 'path' key
            threshold: Hamming distance threshold (default: 5)
            job_id: Optional job ID for logging

        Returns:
            List of unique images with deduplication metadata
        """
        if not extracted_images:
            return extracted_images

        self.logger.info(
            f"   🔍 [Job: {job_id}] Layer 4: Starting perceptual hash deduplication "
            f"(threshold={threshold}, {len(extracted_images)} images)"
        )

        unique_images = []
        seen_hashes = {}  # hash -> image_info mapping
        duplicate_count = 0

        for img_info in extracted_images:
            image_path = img_info.get('path')

            if not image_path or not os.path.exists(image_path):
                self.logger.warning(f"   ⚠️ [Job: {job_id}] Image file not found: {image_path}")
                continue

            try:
                # Compute perceptual hash
                with Image.open(image_path) as img:
                    phash = imagehash.phash(img)

                # Check for duplicates
                is_duplicate = False
                duplicate_of = None

                for existing_hash, existing_info in seen_hashes.items():
                    hamming_distance = phash - existing_hash

                    if hamming_distance <= threshold:
                        # Found a duplicate
                        is_duplicate = True
                        duplicate_of = existing_info.get('filename')
                        duplicate_count += 1

                        self.logger.debug(
                            f"   🔄 [Job: {job_id}] Duplicate found: {img_info.get('filename')} "
                            f"(similar to {duplicate_of}, distance={hamming_distance})"
                        )
                        break

                if is_duplicate:
                    # Mark as duplicate but don't add to unique list
                    img_info['is_duplicate'] = True
                    img_info['duplicate_of'] = duplicate_of
                    img_info['perceptual_hash'] = str(phash)
                else:
                    # Add to unique images
                    img_info['is_duplicate'] = False
                    img_info['duplicate_of'] = None
                    img_info['perceptual_hash'] = str(phash)
                    unique_images.append(img_info)
                    seen_hashes[phash] = img_info

            except Exception as e:
                self.logger.error(
                    f"   ❌ [Job: {job_id}] Failed to compute hash for {image_path}: {e}"
                )
                # Include image anyway if hashing fails
                img_info['is_duplicate'] = False
                img_info['duplicate_of'] = None
                img_info['perceptual_hash'] = None
                unique_images.append(img_info)

        self.logger.info(
            f"   ✅ [Job: {job_id}] Layer 4 complete: {len(unique_images)} unique images "
            f"({duplicate_count} duplicates removed)"
        )

        return unique_images

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
                # ✅ FIX: Skip upload during extraction - keep files for classification
                # Images will be uploaded AFTER classification in stage_3_images.py
                # This prevents files from being deleted before classification can use them

                # Just process metadata without uploading
                processed_info = await self._process_extracted_image(
                    image_path,
                    document_id,
                    processing_options,
                    skip_upload=True  # NEW: Skip upload, keep local files
                )

                if processed_info:
                    # ✅ MERGE metadata: Combine extraction metadata with processing info
                    # Extraction info (confidence, bbox, product_name) is preserved
                    # ⚠️ CRITICAL: processed_info goes first, then img_info overwrites with correct values
                    # This ensures page_number from extraction is NOT overwritten by filename regex
                    merged_info = {**processed_info, **img_info}
                    batch_images.append(merged_info)

                # ✅ FIX: DO NOT delete local files - they're needed for classification
                # Files will be cleaned up by admin cron job after classification completes

                # Free memory after each image
                gc.collect()

            except Exception as e:
                self.logger.warning(f"Failed to process image {filename}: {e}")
                # Don't delete files on error - they might still be usable for classification
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
        processing_options: Dict[str, Any],
        skip_upload: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single extracted image with advanced capabilities.

        Args:
            image_path: Path to the image file
            document_id: Document ID
            processing_options: Processing configuration
            skip_upload: If True, skip upload to Supabase (keep local files for classification)

        Features:
        - Format conversion and optimization
        - Metadata extraction (EXIF, technical specs)
        - Quality assessment
        - Image enhancement options
        - Optional upload to Supabase Storage
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

                # ✅ FIX: Conditionally upload based on skip_upload parameter
                if skip_upload:
                    # Skip upload - keep local files for classification
                    self.logger.debug(f"⏭️  Skipping upload for {basic_info['filename']} (will upload after classification)")
                    upload_result = {
                        'success': False,
                        'skipped': True,
                        'public_url': None,
                        'storage_path': None,
                        'page_number': None
                    }
                else:
                    # Upload image to Supabase Storage
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
                    self.logger.info(f"   page_number: {upload_result.get('page_number')}")
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

            # ============================================================================
            # IMAGE VALIDATION - Resize large images to prevent 400 errors
            # ============================================================================
            MAX_DIMENSION = 1024
            if image.width > MAX_DIMENSION or image.height > MAX_DIMENSION:
                original_size = (image.width, image.height)
                image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
                self.logger.debug(f"📐 Resized image from {original_size} to {image.size} for SLIG classification")

            # ✅ Use singleton SLIG client from endpoint registry (prevents repeated warmups)
            if not hasattr(self, '_slig_client_for_ocr') or self._slig_client_for_ocr is None:
                from app.services.embeddings.endpoint_registry import endpoint_registry

                # Try to get client from registry first (pre-warmed)
                self._slig_client_for_ocr = endpoint_registry.get_slig_client()

                if self._slig_client_for_ocr:
                    self.logger.info("♻️ Using SLIG client from endpoint registry (singleton)")
                else:
                    self.logger.warning("⚠️ SLIG client not available from registry")
                    return {'should_process': True, 'reason': 'slig_not_available', 'confidence': 0.5}

            # ✅ Define candidate labels for zero-shot classification
            # Simplified labels for better classification accuracy
            candidate_labels = [
                "technical specification with text and measurements",
                "decorative image without technical content"
            ]

            # ✅ Use SLIG zero_shot mode for efficient classification
            # This is more efficient than similarity mode for binary classification
            import asyncio

            classification_result = await self._slig_client_for_ocr.zero_shot_classification(
                image=image,
                candidate_labels=candidate_labels
            )

            # Extract top prediction
            top_prediction = classification_result[0]  # Highest scoring label
            label = top_prediction['label']
            score = top_prediction['score']

            # Decision logic based on zero-shot classification
            # If top label is "technical specification" with high confidence, process with OCR
            if "technical" in label.lower() and score > 0.6:
                # Image likely contains technical/specification content
                should_process = True
                reason = f"technical_content (confidence: {score:.3f})"
                confidence = score
            else:
                # Image is likely decorative/historical
                should_process = False
                reason = f"decorative_content (confidence: {score:.3f})"
                confidence = score

            self.logger.debug(f"OCR classification for {os.path.basename(image_path)}: {label} ({score:.3f})")

            return {
                'should_process': should_process,
                'reason': reason,
                'confidence': confidence,
                'classification': label,
                'score': score
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


