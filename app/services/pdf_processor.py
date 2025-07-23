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
        Synchronous wrapper for existing image extraction function.
        """
        try:
            # Create output directory for images
            output_dir = self._create_temp_directory(f"{document_id}_images")
            
            # Use existing extractor function
            page_number = processing_options.get('page_number')
            extract_json_and_images(pdf_path, output_dir, page_number)
            
            # Collect image metadata
            images = []
            image_dir = os.path.join(output_dir, 'images')
            
            if os.path.exists(image_dir):
                for image_file in os.listdir(image_dir):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(image_dir, image_file)
                        image_info = {
                            'filename': image_file,
                            'path': image_path,
                            'size_bytes': os.path.getsize(image_path),
                            'format': image_file.split('.')[-1].upper()
                        }
                        images.append(image_info)
            
            return images
            
        except Exception as e:
            raise PDFExtractionError(f"Image extraction failed: {str(e)}") from e
    
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