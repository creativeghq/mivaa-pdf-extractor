"""
PDF Processing Service - Integration with existing PyMuPDF4LLM functionality

This service wraps the existing extractor.py functionality to work with the 
production FastAPI application structure, providing async interfaces and 
proper error handling while leveraging the proven PDF extraction code.
"""

import asyncio
import logging
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    from ..core.extractor import extract_pdf_to_markdown, extract_pdf_tables, extract_json_and_images
except ImportError:
    # Fall back to the root level extractor if it exists
    import sys
    import os
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if root_path not in sys.path:
        sys.path.append(root_path)
    try:
        from extractor import extract_pdf_to_markdown, extract_pdf_tables, extract_json_and_images
    except ImportError as e:
        # Log the error and provide a fallback
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to import extractor functions: {e}")
        # Define placeholder functions that will raise NotImplementedError
        def extract_pdf_to_markdown(*args, **kwargs):
            raise NotImplementedError("PDF extraction functions not available")
        def extract_pdf_tables(*args, **kwargs):
            raise NotImplementedError("PDF table extraction functions not available")
        def extract_json_and_images(*args, **kwargs):
            raise NotImplementedError("PDF image extraction functions not available")

# Import OCR service
from app.services.ocr_service import get_ocr_service, OCRConfig

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
        
        # Default processing options
        self.default_timeout = self.config.get('timeout_seconds', 300)
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
        processing_options: Optional[Dict[str, Any]] = None
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
            
            # Save PDF bytes to temporary file
            temp_pdf_path = os.path.join(temp_dir, f"{document_id}.pdf")
            async with aiofiles.open(temp_pdf_path, 'wb') as f:
                await f.write(pdf_bytes)
            
            # Process with timeout
            timeout = processing_options.get('timeout_seconds', self.default_timeout)
            
            try:
                result = await asyncio.wait_for(
                    self._process_pdf_file(temp_pdf_path, document_id, processing_options),
                    timeout=timeout
                )
                
                processing_time = time.time() - start_time
                result.processing_time = processing_time
                
                self.logger.info(
                    "PDF processing completed for document %s in %.2f seconds", 
                    document_id, processing_time
                )
                
                return result
                
            except asyncio.TimeoutError:
                raise PDFTimeoutError(f"PDF processing timed out after {timeout} seconds")
                
        except Exception as e:
            self.logger.error("PDF processing failed for document %s: %s", document_id, str(e))
            if isinstance(e, (PDFProcessingError, PDFTimeoutError)):
                raise
            raise PDFProcessingError(f"Unexpected error during PDF processing: {str(e)}") from e
            
        finally:
            # Cleanup temporary files
            if temp_dir:
                self._cleanup_temp_files(temp_dir)
    
    async def process_pdf_from_url(
        self, 
        pdf_url: str, 
        document_id: Optional[str] = None,
        processing_options: Optional[Dict[str, Any]] = None
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
        return await self.process_pdf_from_bytes(pdf_bytes, document_id, processing_options)
    
    async def _process_pdf_file(
        self,
        pdf_path: str,
        document_id: str,
        processing_options: Dict[str, Any]
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
                processing_options
            )
            
            # Extract images if requested
            extracted_images = []
            if processing_options.get('extract_images', True):
                extracted_images = await loop.run_in_executor(
                    None,
                    self._extract_images_sync,
                    pdf_path,
                    document_id,
                    processing_options
                )
            
            # Calculate content metrics
            content_metrics = self._calculate_content_metrics(markdown_content)
            
            # Initialize OCR results
            ocr_text = ""
            ocr_results = []
            multimodal_enabled = processing_options.get('enable_multimodal', False)
            
            # Process images with OCR if multimodal is enabled
            if multimodal_enabled and extracted_images:
                self.logger.info(f"Processing {len(extracted_images)} images with OCR")
                ocr_languages = processing_options.get('ocr_languages', ['en'])
                ocr_text, ocr_results = await self._process_images_with_ocr(
                    extracted_images, ocr_languages
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
        processing_options: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Enhanced markdown extraction with OCR fallback for text-as-images PDFs.

        Features:
        - Primary extraction using PyMuPDF4LLM
        - OCR fallback for PDFs with text-as-images (like WIFI MOMO lookbook)
        - Advanced text processing and cleaning
        - Metadata extraction and analysis
        """
        try:
            # Use existing extractor function
            page_number = processing_options.get('page_number')
            markdown_content = extract_pdf_to_markdown(pdf_path, page_number)

            # Check if we got meaningful text content
            clean_content = markdown_content.replace('-', '').replace('\n', '').strip()

            # If we only got page separators or very little content, try OCR
            if len(clean_content) < 100:  # Less than 100 chars of actual content
                self.logger.info("Standard extraction yielded minimal text, attempting OCR extraction")
                try:
                    ocr_content = self._extract_text_with_ocr(pdf_path, processing_options)

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
            
            return markdown_content, metadata
            
        except Exception as e:
            raise PDFExtractionError(f"Markdown extraction failed: {str(e)}") from e

    def _extract_text_with_ocr(
        self,
        pdf_path: str,
        processing_options: Dict[str, Any]
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
                preprocessing_enabled=True,
                fallback_to_tesseract=True
            )
            ocr_service = get_ocr_service(ocr_config)

            doc = fitz.open(pdf_path)
            all_text = []

            # Process specific page or all pages
            page_number = processing_options.get('page_number')
            if page_number is not None:
                page_range = [page_number - 1] if page_number > 0 else [0]
            else:
                page_range = range(min(20, len(doc)))  # Limit to first 20 pages for performance

            self.logger.info(f"Processing {len(page_range)} pages with OCR")

            for page_num in page_range:
                if page_num < len(doc):
                    page = doc.load_page(page_num)

                    # Render page as high-quality image
                    mat = fitz.Matrix(2.5, 2.5)  # 2.5x zoom for good OCR quality
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes('png')

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

                    except Exception as page_error:
                        self.logger.warning(f"OCR failed for page {page_num + 1}: {page_error}")
                        continue

            doc.close()

            # Combine all extracted text
            final_text = '\n'.join(all_text)
            self.logger.info(f"OCR extraction complete: {len(final_text)} total characters from {len(page_range)} pages")

            return final_text

        except Exception as e:
            self.logger.error(f"OCR text extraction failed: {str(e)}")
            raise PDFExtractionError(f"OCR text extraction failed: {str(e)}") from e

    def _extract_images_sync(
        self,
        pdf_path: str,
        document_id: str,
        processing_options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Enhanced image extraction with advanced processing capabilities.
        
        Features:
        - Basic extraction using existing PyMuPDF functionality
        - Image format conversion and optimization
        - Advanced metadata extraction (EXIF, dimensions, quality metrics)
        - Image enhancement and filtering options
        - Quality assessment and duplicate detection
        """
        try:
            # Create output directory for images
            output_dir = self._create_temp_directory(f"{document_id}_images")
            
            # Use existing extractor function for basic extraction
            page_number = processing_options.get('page_number')
            extract_json_and_images(pdf_path, output_dir, page_number)
            
            # Process extracted images with advanced capabilities
            images = []
            image_dir = os.path.join(output_dir, 'images')
            
            if os.path.exists(image_dir):
                for image_file in os.listdir(image_dir):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                        image_path = os.path.join(image_dir, image_file)
                        
                        # Process each image with advanced capabilities
                        processed_image_info = self._process_extracted_image(
                            image_path,
                            document_id,
                            processing_options
                        )
                        
                        if processed_image_info:
                            images.append(processed_image_info)
            
            # Apply post-processing filters if requested
            if processing_options.get('remove_duplicates', True):
                images = self._remove_duplicate_images(images)
            
            if processing_options.get('quality_filter', True):
                min_quality = processing_options.get('min_quality_score', 0.3)
                images = [img for img in images if img.get('quality_score', 1.0) >= min_quality]
            
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
    
    def _process_extracted_image(
        self,
        image_path: str,
        document_id: str,
        processing_options: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single extracted image with advanced capabilities.
        
        Features:
        - Format conversion and optimization
        - Metadata extraction (EXIF, technical specs)
        - Quality assessment
        - Image enhancement options
        - Duplicate detection preparation
        """
        try:
            # Load image with PIL for metadata and basic processing
            with Image.open(image_path) as pil_image:
                # Extract basic metadata
                basic_info = {
                    'filename': os.path.basename(image_path),
                    'path': image_path,
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

                # Combine all metadata
                return {
                    **basic_info,
                    'exif': exif_data,
                    'quality_score': quality_metrics['overall_score'],
                    'quality_metrics': quality_metrics,
                    'image_hash': image_hash,
                    'enhanced_path': enhanced_path,
                    'converted_path': converted_path,
                    'processing_timestamp': datetime.utcnow().isoformat()
                }
                    
        except Exception as e:
            self.logger.error("Error processing image %s: %s", image_path, str(e))
            return None
    
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
    
    async def _process_images_with_ocr(
        self,
        extracted_images: List[Dict[str, Any]],
        ocr_languages: List[str]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process extracted images with OCR using the OCR service.
        
        Args:
            extracted_images: List of image dictionaries with metadata
            ocr_languages: List of language codes for OCR processing
            
        Returns:
            Tuple of (combined_ocr_text, ocr_results_list)
        """
        try:
            ocr_service = get_ocr_service()
            ocr_config = OCRConfig(languages=ocr_languages)
            
            combined_ocr_text = ""
            ocr_results = []
            
            for image_data in extracted_images:
                image_path = image_data.get('path')
                if not image_path or not os.path.exists(image_path):
                    continue
                
                try:
                    # Process image with OCR
                    ocr_result = await ocr_service.process_image(image_path, ocr_config)
                    
                    if ocr_result and ocr_result.get('text'):
                        combined_ocr_text += ocr_result['text'] + "\n"
                        ocr_results.append({
                            'image_path': image_path,
                            'text': ocr_result['text'],
                            'confidence': ocr_result.get('confidence', 0.0),
                            'language': ocr_result.get('language', 'unknown'),
                            'processing_time': ocr_result.get('processing_time', 0.0)
                        })
                    
                except Exception as e:
                    self.logger.warning("OCR processing failed for image %s: %s", image_path, str(e))
                    ocr_results.append({
                        'image_path': image_path,
                        'text': '',
                        'confidence': 0.0,
                        'language': 'unknown',
                        'error': str(e)
                    })
            
            return combined_ocr_text.strip(), ocr_results
            
        except Exception as e:
            self.logger.error("Error in OCR processing: %s", str(e))
            return "", []
