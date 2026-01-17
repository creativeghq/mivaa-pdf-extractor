"""
PDF to Images Converter

Fast conversion of PDF pages to images for vision-based AI analysis.
Uses PyMuPDF (fitz) for high-speed rendering.

Also includes spread layout detection for catalogs with 2-page spreads.
"""

import logging
import base64
import io
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
from PIL import Image
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# =============================================================================
# SPREAD LAYOUT DETECTION
# =============================================================================

class PageLayoutType(Enum):
    """Type of page layout."""
    SINGLE = "single"                    # Single page (portrait)
    SPREAD = "spread"                    # Two pages side by side (landscape)
    SPREAD_FULL_IMAGE = "spread_full_image"  # Spread with full-width image (don't split)


@dataclass
class PageLayoutInfo:
    """Information about a PDF page's layout."""
    pdf_page_num: int           # 0-based PDF page index
    layout_type: PageLayoutType
    physical_pages: List[int]   # List of physical page numbers (1-based)
    width: float
    height: float
    aspect_ratio: float
    has_full_spread_image: bool = False


@dataclass
class PDFLayoutAnalysis:
    """Complete layout analysis of a PDF."""
    total_pdf_pages: int
    total_physical_pages: int
    has_spread_layout: bool
    pages: List[PageLayoutInfo]
    # Mapping from physical page to PDF page info
    # physical_page -> (pdf_page_idx, 'left'|'right'|'single'|'full')
    physical_to_pdf_map: Dict[int, Tuple[int, str]]


def detect_full_spread_image(page: fitz.Page, threshold: float = 0.75) -> bool:
    """
    Detect if a spread page contains a full-width image that shouldn't be split.

    Args:
        page: PyMuPDF page object
        threshold: Minimum width ratio for image to be considered full-spread (default 75%)

    Returns:
        True if page has an image spanning most of its width
    """
    page_width = page.rect.width
    page_height = page.rect.height

    # Get all images on the page
    image_list = page.get_images(full=True)

    for img_index, img in enumerate(image_list):
        try:
            # Get image bbox
            img_rects = page.get_image_rects(img[0])
            for rect in img_rects:
                img_width = rect.width
                img_height = rect.height

                # Check if image spans most of the page width
                width_ratio = img_width / page_width
                height_ratio = img_height / page_height

                # Full spread image: covers > threshold of width AND significant height
                if width_ratio >= threshold and height_ratio >= 0.5:
                    logger.debug(f"Full spread image detected: {width_ratio:.1%} width, {height_ratio:.1%} height")
                    return True
        except Exception as e:
            logger.debug(f"Error checking image {img_index}: {e}")
            continue

    # Also check for vector graphics that might be full-page illustrations
    try:
        drawings = page.get_drawings()
        if drawings:
            # Calculate bounding box of all drawings
            all_x0 = min(d["rect"].x0 for d in drawings)
            all_x1 = max(d["rect"].x1 for d in drawings)
            drawings_width = all_x1 - all_x0

            if drawings_width / page_width >= threshold:
                # Check if there's minimal text (suggesting it's an image page)
                text = page.get_text().strip()
                if len(text) < 200:  # Very little text
                    logger.debug(f"Full spread graphics detected: {drawings_width/page_width:.1%} width, {len(text)} chars text")
                    return True
    except Exception as e:
        logger.debug(f"Error checking drawings: {e}")

    return False


def analyze_pdf_layout(pdf_path: str) -> PDFLayoutAnalysis:
    """
    Analyze PDF layout to detect spreads and map physical page numbers.

    Handles:
    - Single page PDFs (portrait aspect ratio < 1.2)
    - Spread layout PDFs (landscape aspect ratio > 1.4, contains 2 physical pages per PDF page)
    - Full-spread images (don't split - image spans both physical pages)

    Args:
        pdf_path: Path to PDF file

    Returns:
        PDFLayoutAnalysis with complete layout information
    """
    doc = fitz.open(pdf_path)
    total_pdf_pages = len(doc)

    pages_info: List[PageLayoutInfo] = []
    physical_to_pdf_map: Dict[int, Tuple[int, str]] = {}

    current_physical_page = 1  # Start from 1 (cover is page 1)
    has_any_spread = False

    for pdf_page_idx in range(total_pdf_pages):
        page = doc[pdf_page_idx]
        width = page.rect.width
        height = page.rect.height
        aspect_ratio = width / height

        # Determine layout type based on aspect ratio
        # Portrait/square: aspect < 1.2 (single page)
        # Landscape: aspect > 1.4 (spread - two pages)
        # In-between (1.2-1.4): treat as single to be safe

        if aspect_ratio > 1.4:
            # This is a spread page (two physical pages)
            has_any_spread = True

            # Check if it's a full-spread image
            has_full_image = detect_full_spread_image(page)

            if has_full_image:
                # Full spread image - both physical pages show same content
                layout_type = PageLayoutType.SPREAD_FULL_IMAGE
                physical_pages = [current_physical_page, current_physical_page + 1]

                # Map both physical pages to this PDF page as 'full'
                physical_to_pdf_map[current_physical_page] = (pdf_page_idx, 'full')
                physical_to_pdf_map[current_physical_page + 1] = (pdf_page_idx, 'full')
            else:
                # Regular spread - left and right halves are separate pages
                layout_type = PageLayoutType.SPREAD
                physical_pages = [current_physical_page, current_physical_page + 1]

                # Map left physical page
                physical_to_pdf_map[current_physical_page] = (pdf_page_idx, 'left')
                # Map right physical page
                physical_to_pdf_map[current_physical_page + 1] = (pdf_page_idx, 'right')

            current_physical_page += 2

        else:
            # Single page
            layout_type = PageLayoutType.SINGLE
            physical_pages = [current_physical_page]
            has_full_image = False

            physical_to_pdf_map[current_physical_page] = (pdf_page_idx, 'single')
            current_physical_page += 1

        page_info = PageLayoutInfo(
            pdf_page_num=pdf_page_idx,
            layout_type=layout_type,
            physical_pages=physical_pages,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            has_full_spread_image=has_full_image
        )
        pages_info.append(page_info)

    doc.close()

    total_physical_pages = current_physical_page - 1  # 1-based: pages 1 to (current-1) = current-1 total pages

    analysis = PDFLayoutAnalysis(
        total_pdf_pages=total_pdf_pages,
        total_physical_pages=total_physical_pages,
        has_spread_layout=has_any_spread,
        pages=pages_info,
        physical_to_pdf_map=physical_to_pdf_map
    )

    logger.info(f"📐 PDF Layout: {total_pdf_pages} PDF pages -> {total_physical_pages} physical pages (spread layout: {has_any_spread})")

    return analysis


def get_physical_page_text(
    doc: fitz.Document,
    layout: PDFLayoutAnalysis,
    physical_page: int
) -> Tuple[str, Optional[fitz.Rect]]:
    """
    Get text content for a specific physical page.

    For spread pages, extracts only the relevant half (left or right).
    For full-spread images, returns minimal text with note about image.

    Args:
        doc: Open PyMuPDF document
        layout: PDF layout analysis
        physical_page: Physical page number (1-based)

    Returns:
        Tuple of (text_content, clip_rect or None)
    """
    if physical_page not in layout.physical_to_pdf_map:
        return "", None

    pdf_page_idx, position = layout.physical_to_pdf_map[physical_page]
    page = doc[pdf_page_idx]

    if position == 'single':
        # Full page text
        return page.get_text(), None

    elif position == 'full':
        # Full spread image - return any text present, note it's an image spread
        text = page.get_text().strip()
        return text, None

    elif position == 'left':
        # Left half of spread
        rect = page.rect
        mid_x = rect.width / 2
        clip = fitz.Rect(0, 0, mid_x, rect.height)
        text = page.get_text(clip=clip)
        return text, clip

    elif position == 'right':
        # Right half of spread
        rect = page.rect
        mid_x = rect.width / 2
        clip = fitz.Rect(mid_x, 0, rect.width, rect.height)
        text = page.get_text(clip=clip)
        return text, clip

    return "", None


def extract_text_with_physical_pages(pdf_path: str) -> Tuple[str, PDFLayoutAnalysis]:
    """
    Extract text from PDF with proper physical page markers.

    This handles spread layouts by splitting content into left/right halves
    and assigning correct physical page numbers.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (markdown_text with physical page markers, layout_analysis)
    """
    layout = analyze_pdf_layout(pdf_path)
    doc = fitz.open(pdf_path)

    text_parts = []

    for physical_page in range(1, layout.total_physical_pages + 1):
        content, _ = get_physical_page_text(doc, layout, physical_page)

        # Add physical page marker
        page_marker = f"\n\n--- # Page {physical_page} ---\n\n"
        text_parts.append(page_marker + content)

    doc.close()

    full_text = "\n".join(text_parts)

    logger.info(f"📄 Extracted text with {layout.total_physical_pages} physical page markers")

    return full_text, layout


def get_physical_page_count(pdf_path: str) -> int:
    """
    Get the total number of physical pages in a PDF (accounting for spreads).

    Args:
        pdf_path: Path to PDF file

    Returns:
        Total physical page count
    """
    layout = analyze_pdf_layout(pdf_path)
    return layout.total_physical_pages


# =============================================================================
# PDF TO IMAGES CONVERTER
# =============================================================================


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


