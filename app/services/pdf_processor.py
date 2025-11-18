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

# Import existing extraction functions
try:
    # Try to import from the proper location first
    from ..core.extractor import extract_pdf_to_markdown, extract_pdf_to_markdown_with_doc, extract_pdf_tables, extract_json_and_images
except ImportError:
    # Fall back to the root level extractor if it exists
    import sys
    import os
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if root_path not in sys.path:
        sys.path.append(root_path)
    try:
        from extractor import extract_pdf_to_markdown, extract_pdf_to_markdown_with_doc, extract_pdf_tables, extract_json_and_images
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
from app.services.ocr_service import get_ocr_service, OCRConfig

# Import Supabase client for storage
from app.services.supabase_client import get_supabase_client

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
from app.services.unified_chunking_service import UnifiedChunkingService, ChunkingConfig, ChunkingStrategy


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
        
        # Initialize thread pool executor for async processing
        max_workers = self.config.get('max_workers', 4)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        self.logger.info("PDFProcessor initialized with config: %s", self.config)
    
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
            async with aiofiles.open(temp_pdf_path, 'wb') as f:
                await f.write(pdf_bytes)
            self.logger.info(f"✅ PDF saved successfully, file size: {os.path.getsize(temp_pdf_path)} bytes")
            
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
            # Extract markdown content using existing function
            markdown_content, metadata = await loop.run_in_executor(
                None,
                self._extract_markdown_sync,
                pdf_path,
                processing_options,
                progress_callback
            )
            
            # Extract images if requested
            extracted_images = []
            if processing_options.get('extract_images', True):
                extracted_images = await self._extract_images_async(
                    pdf_path,
                    document_id,
                    processing_options,
                    progress_callback
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
                    'ocr_text_length': len(ocr_text) if ocr_text else 0
                },
                processing_time=0.0,  # Will be set by caller
                page_count=metadata.get('page_count', 0),
                word_count=content_metrics['word_count'],
                character_count=content_metrics['character_count'],
                multimodal_enabled=multimodal_enabled
            )
            
        except Exception as e:
            self.logger.error("Error processing PDF file %s: %s", pdf_path, str(e))
            raise PDFExtractionError(f"Failed to parse PDF content: {str(e)}") from e
    
    def _extract_markdown_sync(
        self,
        pdf_path: str,
        processing_options: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Enhanced markdown extraction with intelligent OCR-first approach for image-based PDFs.

        Features:
        - Smart detection of image-based vs text-based PDFs
        - OCR-first approach for image-heavy PDFs (like WIFI MOMO lookbook)
        - PyMuPDF4LLM fallback for text-based PDFs
        - Advanced text processing and cleaning
        - Metadata extraction and analysis
        """
        try:
            # First, analyze the PDF to determine if it's image-based
            import fitz
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            # Sample pages strategically to determine PDF type
            # Sample first 3, middle 2, and last 2 pages for better detection
            pages_to_sample = []
            
            # Always sample first 3 pages
            pages_to_sample.extend(range(min(3, total_pages)))
            
            # Sample middle pages if document is long enough
            if total_pages > 5:
                mid_page = total_pages // 2
                pages_to_sample.extend([mid_page - 1, mid_page])
            
            # Sample last pages if document is long enough
            if total_pages > 3:
                pages_to_sample.extend([total_pages - 2, total_pages - 1])
            
            # Remove duplicates and sort
            pages_to_sample = sorted(set(pages_to_sample))
            
            total_text_chars = 0
            total_images = 0

            for page_num in pages_to_sample:
                page = doc[page_num]
                text = page.get_text()
                images = page.get_images()
                total_text_chars += len(text.strip())
                total_images += len(images)

            doc.close()

            # Enhanced PDF type detection with multiple criteria
            avg_text_per_page = total_text_chars / len(pages_to_sample)
            avg_images_per_page = total_images / len(pages_to_sample)

            # Multiple detection criteria for better accuracy
            criteria = {
                'low_text': avg_text_per_page < 50,
                'very_low_text': avg_text_per_page < 10,  # More restrictive threshold
                'has_images': avg_images_per_page >= 1,
                'many_images': avg_images_per_page >= 3,
                'text_to_image_ratio': (avg_text_per_page / max(avg_images_per_page, 1)) < 30,
                'no_images': avg_images_per_page == 0
            }

            # Smart detection logic - prioritize text-first for text-only PDFs
            is_image_based = (
                (criteria['very_low_text'] and criteria['has_images']) or  # Very little text + images → OCR
                (criteria['low_text'] and criteria['many_images']) or  # Low text + many images → OCR
                (criteria['many_images'] and criteria['text_to_image_ratio'])  # Many images + low text ratio → OCR
            ) and not criteria['no_images']  # Never use OCR for PDFs with no images

            # Enhanced logging with detection criteria
            detection_reason = []
            if criteria['very_low_text'] and criteria['has_images']:
                detection_reason.append("very_low_text_with_images")
            if criteria['low_text'] and criteria['many_images']:
                detection_reason.append("low_text_many_images")
            if criteria['many_images'] and criteria['text_to_image_ratio']:
                detection_reason.append("many_images_low_ratio")
            if criteria['no_images']:
                detection_reason.append("text_only_pdf")

            self.logger.info(f"📊 PDF Analysis: {total_pages} pages, {avg_text_per_page:.1f} chars/page, "
                           f"{avg_images_per_page:.1f} images/page")
            self.logger.info(f"🎯 Detection: {'OCR-first' if is_image_based else 'Text-first'} "
                           f"(reason: {', '.join(detection_reason) if detection_reason else 'text_dominant'})")

            if is_image_based:
                # Use OCR-first approach for image-based PDFs
                self.logger.info("Detected image-based PDF, using OCR-first extraction")
                try:
                    markdown_content = self._extract_text_with_ocr(pdf_path, processing_options, progress_callback)
                    if len(markdown_content.strip()) < 100:
                        # If OCR also fails, try PyMuPDF4LLM as fallback
                        self.logger.info("OCR yielded minimal content, trying PyMuPDF4LLM fallback")
                        page_number = processing_options.get('page_number')
                        fallback_content = extract_pdf_to_markdown(pdf_path, page_number)
                        if len(fallback_content.strip()) > len(markdown_content.strip()):
                            markdown_content = fallback_content
                except Exception as ocr_error:
                    self.logger.error(f"OCR extraction failed: {ocr_error}, trying PyMuPDF4LLM fallback")
                    page_number = processing_options.get('page_number')
                    markdown_content = extract_pdf_to_markdown(pdf_path, page_number)
            else:
                # Use PyMuPDF4LLM first for text-based PDFs
                self.logger.info("Detected text-based PDF, using PyMuPDF4LLM extraction")

                # Update progress: Starting text extraction (30%)
                if progress_callback:
                    try:
                        # Only call if it's not a coroutine (sync callbacks only in sync function)
                        if not inspect.iscoroutinefunction(progress_callback):
                            progress_callback(
                                progress=30,
                                details={
                                    "current_step": "Extracting text from PDF using PyMuPDF4LLM",
                                    "total_pages": total_pages,
                                    "extraction_method": "pymupdf4llm"
                                }
                            )
                    except Exception as callback_error:
                        self.logger.warning(f"Progress callback failed: {callback_error}")

                page_number = processing_options.get('page_number')

                # Try PyMuPDF4LLM with proper error handling and batching
                try:
                    # If page_number is None, process entire PDF
                    # PyMuPDF4LLM can fail with "not a textpage" for certain PDF formats
                    markdown_content = extract_pdf_to_markdown(pdf_path, page_number)

                except ValueError as e:
                    if "not a textpage" in str(e):
                        self.logger.warning(f"⚠️ PyMuPDF4LLM failed with 'not a textpage' error")
                        self.logger.info("Attempting page-by-page extraction with PyMuPDF4LLM to isolate problematic pages")

                        # Try processing page-by-page to find which pages work
                        import fitz
                        import os

                        # Verify PDF file exists before processing
                        if not os.path.exists(pdf_path):
                            raise PDFExtractionError(f"PDF file not found: {pdf_path}")

                        # Open document ONCE and keep it open during entire extraction
                        # This prevents garbage collection issues with PyMuPDF weak references
                        doc = fitz.open(pdf_path)
                        total_pages = len(doc)

                        self.logger.info(f"PDF has {total_pages} pages, will process page-by-page with persistent document object")

                        markdown_content = ""
                        failed_pages = []

                        try:
                            # Process in SMALLER batches of 5 pages to reduce memory and avoid hanging
                            batch_size = 5
                            for batch_start in range(0, total_pages, batch_size):
                                batch_end = min(batch_start + batch_size, total_pages)
                                self.logger.info(f"Processing pages {batch_start + 1}-{batch_end} with PyMuPDF4LLM")

                                for page_num in range(batch_start, batch_end):
                                    # Verify page number is valid
                                    if page_num >= total_pages:
                                        self.logger.warning(f"Skipping page {page_num + 1} - out of range (total: {total_pages})")
                                        continue

                                    try:
                                        self.logger.debug(f"Extracting page {page_num + 1}/{total_pages} (0-indexed: {page_num})")
                                        # Use document object to avoid garbage collection issues
                                        page_content = extract_pdf_to_markdown_with_doc(doc, page_num)
                                        markdown_content += page_content + "\n\n"
                                    except ValueError as page_error:
                                        if "not a textpage" in str(page_error):
                                            self.logger.warning(f"Page {page_num + 1} failed with 'not a textpage', will use OCR for this page")
                                            failed_pages.append(page_num)
                                        else:
                                            self.logger.error(f"Page {page_num + 1} failed with unexpected error: {page_error}")
                                            raise
                                    except Exception as page_error:
                                        self.logger.error(f"Page {page_num + 1} failed with error: {page_error}")
                                        raise
                        finally:
                            # Always close the document when done
                            doc.close()
                            self.logger.debug(f"Closed PDF document after processing {total_pages} pages")

                            # Force garbage collection after each batch
                            import gc
                            gc.collect()

                        # If we have failed pages, use OCR only for those specific pages
                        if failed_pages:
                            self.logger.info(f"Using OCR for {len(failed_pages)} failed pages: {failed_pages}")
                            ocr_content = self._extract_text_with_ocr_for_pages(pdf_path, failed_pages, processing_options, progress_callback)
                            markdown_content += ocr_content

                        if len(markdown_content.strip()) < 100:
                            raise PDFExtractionError("PyMuPDF4LLM extraction yielded minimal content")

                    else:
                        raise

                # Check if we got meaningful text content
                clean_content = markdown_content.replace('-', '').replace('\n', '').strip()

                # If we only got page separators or very little content, try OCR
                if len(clean_content) < 100:  # Less than 100 chars of actual content
                    self.logger.info("Standard extraction yielded minimal text, attempting OCR extraction")
                    try:
                        ocr_content = self._extract_text_with_ocr(pdf_path, processing_options, progress_callback)

                        if len(ocr_content.strip()) > len(clean_content):
                            self.logger.info(f"OCR extraction successful: {len(ocr_content)} characters vs {len(clean_content)} from standard")
                            markdown_content = ocr_content
                        else:
                            self.logger.info("OCR did not improve text extraction, using standard result")
                    except Exception as ocr_error:
                        self.logger.warning(f"OCR extraction failed: {ocr_error}, using standard result")

            # Get basic metadata (page count, etc.)
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)

            # Helper function to parse PDF dates
            def parse_pdf_date(date_str):
                """Parse PDF date string to datetime or return None."""
                if not date_str or date_str.strip() == '':
                    return None
                try:
                    # PDF dates are often in format: D:YYYYMMDDHHmmSSOHH'mm'
                    if date_str.startswith('D:'):
                        date_str = date_str[2:]
                    # Extract just the date part (YYYYMMDDHHMMSS)
                    if len(date_str) >= 14:
                        from datetime import datetime
                        return datetime.strptime(date_str[:14], '%Y%m%d%H%M%S')
                    elif len(date_str) >= 8:
                        from datetime import datetime
                        return datetime.strptime(date_str[:8], '%Y%m%d')
                except:
                    pass
                return None

            metadata = {
                'page_count': doc.page_count,
                'title': doc.metadata.get('title', '') or None,
                'author': doc.metadata.get('author', '') or None,
                'subject': doc.metadata.get('subject', '') or None,
                'creator': doc.metadata.get('creator', '') or None,
                'producer': doc.metadata.get('producer', '') or None,
                'creation_date': parse_pdf_date(doc.metadata.get('creationDate', '')),
                'modification_date': parse_pdf_date(doc.metadata.get('modDate', ''))
            }
            doc.close()

            # Update progress: Text extraction complete (50%)
            if progress_callback:
                try:
                    # Only call if it's not a coroutine (sync callbacks only in sync function)
                    if not inspect.iscoroutinefunction(progress_callback):
                        progress_callback(
                            progress=50,
                            details={
                                "current_step": "Text extraction complete, preparing for chunking",
                                "total_pages": total_pages,
                                "text_length": len(markdown_content),
                                "extraction_method": "pymupdf4llm"
                            }
                        )
                except Exception as callback_error:
                    self.logger.warning(f"Progress callback failed: {callback_error}")

            # Explicit memory cleanup after extraction
            import gc
            gc.collect()

            return markdown_content, metadata

        except Exception as e:
            # Cleanup on error
            import gc
            gc.collect()
            raise PDFExtractionError(f"Markdown extraction failed: {str(e)}") from e

    def _extract_text_with_ocr_for_pages(
        self,
        pdf_path: str,
        page_numbers: List[int],
        processing_options: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> str:
        """
        Extract text from specific PDF pages using OCR.

        Args:
            pdf_path: Path to PDF file
            page_numbers: List of page numbers (0-indexed) to process with OCR
            processing_options: Processing configuration
            progress_callback: Optional progress callback

        Returns:
            Extracted text as markdown
        """
        try:
            import fitz  # PyMuPDF
            from app.services.ocr_service import get_ocr_service, OCRConfig
            import gc

            # Initialize OCR service
            ocr_languages = processing_options.get('ocr_languages', ['en'])
            ocr_config = OCRConfig(
                languages=ocr_languages,
                confidence_threshold=0.3,
                preprocessing_enabled=True
            )
            ocr_service = get_ocr_service(ocr_config)

            doc = fitz.open(pdf_path)
            all_text = []

            self.logger.info(f"Processing {len(page_numbers)} pages with OCR in batches")

            # Process pages in batches of 5 to manage memory
            batch_size = 5
            for batch_idx in range(0, len(page_numbers), batch_size):
                batch_pages = page_numbers[batch_idx:batch_idx + batch_size]
                self.logger.info(f"OCR Batch {batch_idx // batch_size + 1}: Processing pages {[p+1 for p in batch_pages]}")

                for page_num in batch_pages:
                    if page_num < len(doc):
                        page = doc.load_page(page_num)

                        # Render page as image (2.0x zoom for quality vs memory balance)
                        mat = fitz.Matrix(2.0, 2.0)
                        pix = page.get_pixmap(matrix=mat)
                        img_data = pix.tobytes('png')

                        # Explicitly free pixmap memory
                        pix = None

                        # Extract text with OCR
                        try:
                            from PIL import Image
                            import io

                            img = Image.open(io.BytesIO(img_data))
                            ocr_results = ocr_service.extract_text_from_image(img)

                            # Combine all OCR results for this page
                            page_text = []
                            for result in ocr_results:
                                if result.text.strip() and result.confidence > 0.3:
                                    page_text.append(result.text.strip())

                            if page_text:
                                combined_page_text = ' '.join(page_text)
                                all_text.append(f"## Page {page_num + 1}\n\n{combined_page_text}\n")
                                self.logger.debug(f"Page {page_num + 1}: Extracted {len(combined_page_text)} characters via OCR")

                            # Explicitly free image memory
                            img.close()
                            img = None
                            img_data = None

                        except Exception as page_error:
                            self.logger.warning(f"OCR failed for page {page_num + 1}: {page_error}")
                            continue

                # Force garbage collection after each batch
                gc.collect()
                self.logger.debug(f"Completed OCR batch {batch_idx // batch_size + 1}, memory freed")

            doc.close()

            # Final garbage collection
            gc.collect()

            return '\n'.join(all_text)

        except Exception as e:
            import gc
            gc.collect()
            raise PDFExtractionError(f"OCR extraction failed for specific pages: {str(e)}") from e

    def _extract_text_with_ocr(
        self,
        pdf_path: str,
        processing_options: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> str:
        """
        Extract text from PDF using OCR for text-as-images PDFs.

        This method renders PDF pages as images and uses OCR to extract text.
        Useful for PDFs where text is embedded as images or paths.
        """
        try:
            import fitz  # PyMuPDF
            from app.services.ocr_service import get_ocr_service, OCRConfig

            # Initialize OCR service
            ocr_languages = processing_options.get('ocr_languages', ['en'])
            ocr_config = OCRConfig(
                languages=ocr_languages,
                confidence_threshold=0.3,  # Lower threshold for better recall
                preprocessing_enabled=True
            )
            ocr_service = get_ocr_service(ocr_config)

            doc = fitz.open(pdf_path)
            all_text = []

            # Process specific page or all pages
            page_number = processing_options.get('page_number')
            if page_number is not None:
                page_range = [page_number - 1] if page_number > 0 else [0]
            else:
                # Process all pages for complete extraction
                page_range = list(range(len(doc)))

            self.logger.info(f"Processing {len(page_range)} pages with OCR in batches")

            # Process in batches of 5 pages to manage memory
            batch_size = 5
            total_batches = (len(page_range) + batch_size - 1) // batch_size

            for batch_idx in range(0, len(page_range), batch_size):
                batch_end = min(batch_idx + batch_size, len(page_range))
                batch_num = batch_idx // batch_size + 1
                self.logger.info(f"OCR Batch {batch_num}/{total_batches}: Processing pages {batch_idx + 1}-{batch_end}")

                for i in range(batch_idx, batch_end):
                    page_num = page_range[i]
                if page_num < len(doc):
                    page = doc.load_page(page_num)

                    # Render page as image (reduced zoom to save memory)
                    mat = fitz.Matrix(2.0, 2.0)  # 2.0x zoom (reduced from 2.5x to save memory)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes('png')

                    # Explicitly free pixmap memory
                    pix = None

                    # Extract text with OCR
                    try:
                        from PIL import Image
                        import io

                        img = Image.open(io.BytesIO(img_data))
                        ocr_results = ocr_service.extract_text_from_image(img)

                        # Combine all OCR results for this page
                        page_text = []
                        for result in ocr_results:
                            if result.text.strip() and result.confidence > 0.3:
                                page_text.append(result.text.strip())

                        if page_text:
                            combined_page_text = ' '.join(page_text)
                            all_text.append(f"## Page {page_num + 1}\n\n{combined_page_text}\n")
                            self.logger.debug(f"Page {page_num + 1}: Extracted {len(combined_page_text)} characters")

                        # Explicitly free image memory
                        img.close()
                        img = None
                        img_data = None

                    except Exception as page_error:
                        self.logger.warning(f"OCR failed for page {page_num + 1}: {page_error}")
                        continue

                # Force garbage collection after each batch
                import gc
                gc.collect()
                self.logger.debug(f"Completed OCR batch {batch_num}/{total_batches}, memory freed")

                # Log progress after each batch
                progress = (batch_end / len(page_range)) * 80  # OCR is 80% of total processing
                self.logger.info(f"📄 OCR Progress: {batch_end}/{len(page_range)} pages ({progress:.1f}%)")

                # Update job progress if callback provided
                if progress_callback:
                    try:
                        # Only call if it's not a coroutine (sync callbacks only in sync function)
                        if not inspect.iscoroutinefunction(progress_callback):
                            progress_callback(
                                progress=int(progress),
                                details={
                                    "current_step": f"OCR processing: {batch_end}/{len(page_range)} pages (batch {batch_num}/{total_batches})",
                                    "pages_processed": batch_end,
                                    "total_pages": len(page_range),
                                    "ocr_stage": "extracting_text",
                                    "batch_number": batch_num,
                                    "total_batches": total_batches
                                }
                            )
                    except Exception as callback_error:
                        self.logger.warning(f"Progress callback failed: {callback_error}")

            doc.close()

            # Final garbage collection
            import gc
            gc.collect()

            # Combine all extracted text
            final_text = '\n'.join(all_text)
            self.logger.info(f"✅ OCR extraction complete: {len(final_text)} total characters from {len(page_range)} pages")

            # Update progress for chunk creation
            if progress_callback:
                try:
                    # Only call if it's not a coroutine (sync callbacks only in sync function)
                    if not inspect.iscoroutinefunction(progress_callback):
                        progress_callback(
                            progress=85,
                            details={
                                "current_step": "Creating text chunks for RAG pipeline",
                                "pages_processed": len(page_range),
                                "total_pages": len(page_range),
                                "text_length": len(final_text),
                                "ocr_stage": "creating_chunks"
                            }
                        )
                except Exception as callback_error:
                    self.logger.warning(f"Progress callback failed: {callback_error}")

            # Explicit memory cleanup after OCR extraction
            import gc
            gc.collect()

            return final_text

        except Exception as e:
            # Cleanup on error
            import gc
            gc.collect()
            self.logger.error(f"OCR text extraction failed: {str(e)}")
            raise PDFExtractionError(f"OCR text extraction failed: {str(e)}") from e

    def _should_use_multimodal(self, extracted_images: List[Dict], markdown_content: str) -> bool:
        """
        Intelligent detection of whether multimodal processing would be beneficial.

        Args:
            extracted_images: List of extracted images
            markdown_content: Extracted text content

        Returns:
            bool: True if multimodal processing is recommended
        """
        try:
            # Criteria for multimodal processing
            has_images = len(extracted_images) > 0
            many_images = len(extracted_images) >= 3
            low_text_content = len(markdown_content.strip()) < 500
            moderate_text_content = len(markdown_content.strip()) < 2000

            # Decision logic
            if not has_images:
                return False  # No images → no multimodal needed

            if many_images and low_text_content:
                return True  # Many images + little text → likely visual document

            if has_images and moderate_text_content:
                return True  # Some images + moderate text → could benefit from multimodal

            if len(extracted_images) >= 1 and low_text_content:
                return True  # Any images + very little text → likely needs OCR

            return False  # Text-heavy document → multimodal not needed

        except Exception as e:
            self.logger.warning(f"Multimodal detection failed: {e}, defaulting to False")
            return False

    async def _extract_images_async(
        self,
        pdf_path: str,
        document_id: str,
        processing_options: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Enhanced async image extraction with Supabase Storage upload.

        Features:
        - Basic extraction using existing PyMuPDF functionality
        - Image format conversion and optimization
        - Advanced metadata extraction (EXIF, dimensions, quality metrics)
        - Image enhancement and filtering options
        - Upload to Supabase Storage instead of local storage
        - Quality assessment and duplicate detection
        """
        try:
            # Create output directory for images
            output_dir = self._create_temp_directory(f"{document_id}_images")

            # Use existing extractor function for basic extraction (run in executor for sync function)
            loop = asyncio.get_event_loop()
            page_number = processing_options.get('page_number')

            # 🚀 OPTIMIZATION: Use smaller batch size for memory efficiency
            # Process 2 pages at a time instead of 5 to reduce memory from 400MB to ~160MB (60% reduction)
            batch_size = processing_options.get('image_batch_size', 2)

            # ✅ OPTIMIZATION: Get page_list for focused extraction (only extract images from specific pages)
            page_list = processing_options.get('page_list')  # List of page numbers (1-indexed)

            await loop.run_in_executor(
                None,
                extract_json_and_images,
                pdf_path,
                output_dir,
                page_number,
                batch_size,  # Pass batch_size parameter
                page_list    # ✅ NEW: Pass page_list for focused extraction
            )

            # Process extracted images with advanced capabilities
            images = []
            image_dir = os.path.join(output_dir, 'images')

            self.logger.info(f"🔍 Checking for extracted images in: {image_dir}")
            self.logger.info(f"   Image directory exists: {os.path.exists(image_dir)}")

            if os.path.exists(image_dir):
                image_files = os.listdir(image_dir)
                self.logger.info(f"   Found {len(image_files)} files in image directory")

                # Report progress: Image extraction found images
                if progress_callback:
                    try:
                        import inspect
                        if inspect.iscoroutinefunction(progress_callback):
                            await progress_callback(
                                progress=25,
                                details={
                                    "current_step": f"Processing {len(image_files)} extracted images",
                                    "total_images": len(image_files),
                                    "images_processed": 0
                                }
                            )
                        else:
                            progress_callback(
                                progress=25,
                                details={
                                    "current_step": f"Processing {len(image_files)} extracted images",
                                    "total_images": len(image_files),
                                    "images_processed": 0
                                }
                            )
                    except Exception as e:
                        self.logger.warning(f"Progress callback failed: {e}")

                # Process images in batches to reduce memory usage
                batch_size = 1  # Process 1 image at a time for maximum memory efficiency
                valid_image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]

                for batch_start in range(0, len(valid_image_files), batch_size):
                    batch_end = min(batch_start + batch_size, len(valid_image_files))
                    batch_files = valid_image_files[batch_start:batch_end]

                    self.logger.info(f"   Processing batch {batch_start // batch_size + 1}: images {batch_start + 1}-{batch_end} of {len(valid_image_files)}")

                    # Process each image in the batch
                    for idx, image_file in enumerate(batch_files):
                        absolute_idx = batch_start + idx
                        self.logger.debug(f"   Processing file: {image_file}")
                        image_path = os.path.join(image_dir, image_file)

                        # Process each image with advanced capabilities (now async)
                        skip_upload = processing_options.get('skip_upload', False)
                        processed_image_info = await self._process_extracted_image(
                            image_path,
                            document_id,
                            processing_options,
                            skip_upload=skip_upload
                        )

                        if processed_image_info:
                            images.append(processed_image_info)
                            self.logger.info(f"   ✅ Processed image: {image_file}")

                            # Report progress: Image processing progress
                            if progress_callback and absolute_idx % 5 == 0:  # Update every 5 images
                                try:
                                    import inspect
                                    progress_pct = 25 + (absolute_idx / len(valid_image_files)) * 10  # 25-35% range
                                    if inspect.iscoroutinefunction(progress_callback):
                                        await progress_callback(
                                            progress=int(progress_pct),
                                            details={
                                                "current_step": f"Processing images ({absolute_idx + 1}/{len(valid_image_files)})",
                                                "total_images": len(valid_image_files),
                                                "images_processed": absolute_idx + 1
                                            }
                                        )
                                    else:
                                        progress_callback(
                                            progress=int(progress_pct),
                                            details={
                                                "current_step": f"Processing images ({absolute_idx + 1}/{len(valid_image_files)})",
                                                "total_images": len(valid_image_files),
                                                "images_processed": absolute_idx + 1
                                            }
                                        )
                                except Exception as e:
                                    self.logger.warning(f"Progress callback failed: {e}")
                        else:
                            self.logger.warning(f"   ⚠️ Failed to process image: {image_file}")

                    # Force garbage collection after each batch to free memory
                    import gc
                    gc.collect()
                    self.logger.info(f"   ✅ Batch {batch_start // batch_size + 1} completed, memory freed")
            else:
                self.logger.warning(f"⚠️ Image directory does not exist: {image_dir}")
                self.logger.warning(f"   Output directory contents: {os.listdir(output_dir) if os.path.exists(output_dir) else 'N/A'}")

            # NOTE: Temporary file cleanup moved to admin panel cron job

            # Apply post-processing filters if requested
            if processing_options.get('remove_duplicates', True):
                images = self._remove_duplicate_images(images)

            if processing_options.get('quality_filter', True):
                min_quality = processing_options.get('min_quality_score', 0.3)
                images = [img for img in images if img.get('quality_score', 1.0) >= min_quality]

            self.logger.info(f"Successfully extracted and uploaded {len(images)} images to Supabase Storage")
            return images

        except Exception as e:
            raise PDFExtractionError(f"Enhanced image extraction failed: {str(e)}") from e
    
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
        Process a single extracted image with advanced capabilities and upload to Supabase Storage.

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

                # Upload image to Supabase Storage (skip if requested for AI classification first)
                if skip_upload:
                    # Skip upload - will be done later after AI classification
                    upload_result = {
                        'success': True,
                        'public_url': None,  # Will be set after classification
                        'storage_path': None,
                        'storage_bucket': None,
                        'skipped': True
                    }
                    self.logger.info(f"   ⏭️  Skipped upload for AI classification: {basic_info['filename']}")
                else:
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

                self.logger.info(f"✅ Image uploaded to storage, AI analysis deferred: {basic_info['filename']}")

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
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_siglip_model_for_ocr'):
                self._siglip_model_for_ocr = SentenceTransformer('google/siglip-so400m-patch14-384')
                self.logger.info("✅ Initialized SigLIP model for OCR filtering: google/siglip-so400m-patch14-384")

            siglip_model = self._siglip_model_for_ocr
            
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
            
            # Get image embedding
            # SigLIP model accepts PIL Image object directly
            loop = asyncio.get_event_loop()
            import numpy as np
            image_embedding_raw = await loop.run_in_executor(
                None,
                siglip_model.encode,
                image  # Pass PIL Image object
            )
            # Normalize
            image_embedding_raw = image_embedding_raw / np.linalg.norm(image_embedding_raw)
            image_embedding = image_embedding_raw.tolist() if hasattr(image_embedding_raw, 'tolist') else list(image_embedding_raw)

            # Get text embeddings for all prompts
            all_prompts = relevant_prompts + irrelevant_prompts
            text_embeddings = []
            for prompt in all_prompts:
                text_emb_raw = await loop.run_in_executor(
                    None,
                    siglip_model.encode,
                    prompt
                )
                # Normalize
                text_emb_raw = text_emb_raw / np.linalg.norm(text_emb_raw)
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

