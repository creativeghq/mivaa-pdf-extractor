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

# Import image processing libraries
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from PIL.ExifTags import TAGS
import imageio
from skimage import filters, morphology, measure
from scipy import ndimage

# Import existing extraction functions
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from extractor import extract_pdf_to_markdown, extract_pdf_tables, extract_json_and_images

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
    metadata: Dict[str, Any]
    processing_time: float
    page_count: int
    word_count: int
    character_count: int


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
            
            return PDFProcessingResult(
                document_id=document_id,
                markdown_content=markdown_content,
                extracted_images=extracted_images,
                metadata={
                    **metadata,
                    **content_metrics,
                    'processing_options': processing_options,
                    'timestamp': datetime.utcnow().isoformat()
                },
                processing_time=0.0,  # Will be set by caller
                page_count=metadata.get('page_count', 0),
                word_count=content_metrics['word_count'],
                character_count=content_metrics['character_count']
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
        Synchronous wrapper for existing markdown extraction function.
        """
        try:
            # Use existing extractor function
            page_number = processing_options.get('page_number')
            markdown_content = extract_pdf_to_markdown(pdf_path, page_number)
            
            # Get basic metadata (page count, etc.)
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            metadata = {
                'page_count': doc.page_count,
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', '')
            }
            doc.close()
            
            return markdown_content, metadata
            
        except Exception as e:
            raise PDFExtractionError(f"Markdown extraction failed: {str(e)}") from e
    
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
                
                # Load with OpenCV for advanced analysis
                cv_image = cv2.imread(image_path)
                if cv_image is not None:
                    # Calculate quality metrics
                    quality_metrics = self._calculate_image_quality(cv_image)
                    
                    # Calculate image hash for duplicate detection
                    image_hash = self._calculate_image_hash(cv_image)
                    
                    # Apply enhancements if requested
                    enhanced_path = None
                    if processing_options.get('enhance_images', False):
                        enhanced_path = self._enhance_image(
                            cv_image,
                            image_path,
                            processing_options
                        )
                    
                    # Convert format if requested
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
                else:
                    self.logger.warning("Could not load image with OpenCV: %s", image_path)
                    return {
                        **basic_info,
                        'exif': exif_data,
                        'quality_score': 0.5,  # Default for images we can't analyze
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


# Convenience function for backward compatibility
async def process_pdf_bytes(
    pdf_bytes: bytes,
    document_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> PDFProcessingResult:
    """
    Convenience function to process PDF bytes.
    
    Args:
        pdf_bytes: Raw PDF file bytes
        document_id: Optional document identifier
        config: Processing configuration
        
    Returns:
        PDFProcessingResult with extracted content
    """
    processor = PDFProcessor(config)
    return await processor.process_pdf_from_bytes(pdf_bytes, document_id)


async def process_pdf_url(
    pdf_url: str,
    document_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> PDFProcessingResult:
    """
    Convenience function to process PDF from URL.
    
    Args:
        pdf_url: URL to PDF file
        document_id: Optional document identifier
        config: Processing configuration
        
    Returns:
        PDFProcessingResult with extracted content
    """
    processor = PDFProcessor(config)
    return await processor.process_pdf_from_url(pdf_url, document_id)