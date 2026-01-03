"""
PDF to Images Converter

Fast conversion of PDF pages to images for vision-based AI analysis.
Uses PyMuPDF (fitz) for high-speed rendering.
"""

import logging
import base64
import io
from typing import List, Optional, Tuple
from PIL import Image
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFToImagesConverter:
    """Convert PDF pages to images for vision model analysis."""
    
    def __init__(self, dpi: int = 250, max_dimension: int = 1800):
        """
        Initialize converter.

        Args:
            dpi: Resolution for rendering (250 optimized for material detail extraction)
            max_dimension: Maximum width/height (Claude Vision limit: 2000px for multi-image, using 1800 to be safe)
        """
        self.dpi = dpi
        self.max_dimension = max_dimension
        self.zoom = dpi / 72  # PyMuPDF zoom factor (72 DPI is default)
    
    def convert_pdf_to_images(
        self,
        pdf_path: str,
        max_pages: Optional[int] = None,
        start_page: int = 0
    ) -> List[Tuple[int, str]]:
        """
        Convert PDF pages to base64-encoded images.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to convert (None = all pages)
            start_page: Starting page number (0-indexed)
            
        Returns:
            List of tuples: (page_number, base64_image)
        """
        logger.info(f"🖼️  Converting PDF to images: {pdf_path}")
        logger.info(f"   DPI: {self.dpi}, Max dimension: {self.max_dimension}")
        
        images = []
        
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            # Determine page range
            end_page = min(start_page + max_pages, total_pages) if max_pages else total_pages
            
            logger.info(f"   Converting pages {start_page + 1} to {end_page} (total: {total_pages})")
            
            # Convert each page
            for page_num in range(start_page, end_page):
                try:
                    # Get page
                    page = doc[page_num]
                    
                    # Render page to image
                    mat = fitz.Matrix(self.zoom, self.zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Resize if too large
                    if img.width > self.max_dimension or img.height > self.max_dimension:
                        img.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)
                    
                    # Convert to base64
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=85, optimize=True)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    images.append((page_num + 1, img_base64))  # 1-indexed page numbers
                    
                    if (page_num - start_page + 1) % 10 == 0:
                        logger.info(f"   ✅ Converted {page_num - start_page + 1}/{end_page - start_page} pages")
                    
                except Exception as e:
                    logger.error(f"   ❌ Failed to convert page {page_num + 1}: {e}")
                    continue
            
            doc.close()
            
            logger.info(f"✅ Converted {len(images)} pages to images")
            return images
            
        except Exception as e:
            logger.error(f"❌ PDF to images conversion failed: {e}")
            raise
    
    def convert_pdf_bytes_to_images(
        self,
        pdf_bytes: bytes,
        max_pages: Optional[int] = None,
        start_page: int = 0
    ) -> List[Tuple[int, str]]:
        """
        Convert PDF bytes to base64-encoded images.
        
        Args:
            pdf_bytes: PDF file bytes
            max_pages: Maximum number of pages to convert (None = all pages)
            start_page: Starting page number (0-indexed)
            
        Returns:
            List of tuples: (page_number, base64_image)
        """
        logger.info(f"🖼️  Converting PDF bytes to images ({len(pdf_bytes)} bytes)")
        
        images = []
        
        try:
            # Open PDF from bytes
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)
            
            # Determine page range
            end_page = min(start_page + max_pages, total_pages) if max_pages else total_pages
            
            logger.info(f"   Converting pages {start_page + 1} to {end_page} (total: {total_pages})")
            
            # Convert each page
            for page_num in range(start_page, end_page):
                try:
                    page = doc[page_num]
                    mat = fitz.Matrix(self.zoom, self.zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    if img.width > self.max_dimension or img.height > self.max_dimension:
                        img.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)
                    
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=85, optimize=True)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    images.append((page_num + 1, img_base64))
                    
                except Exception as e:
                    logger.error(f"   ❌ Failed to convert page {page_num + 1}: {e}")
                    continue
            
            doc.close()
            logger.info(f"✅ Converted {len(images)} pages to images")
            return images
            
        except Exception as e:
            logger.error(f"❌ PDF bytes to images conversion failed: {e}")
            raise


