"""
PDF Page Numbering Service

Pre-processing service that adds PHYSICAL page numbers to each PDF page
BEFORE discovery begins. This ensures:
1. Every page has a visible, unambiguous page number
2. Claude and other AI models can reference exact pages
3. Debugging and verification is straightforward
4. Page numbers persist through all processing stages

SPREAD-AWARE: For PDFs with 2-page spreads, this service adds TWO page
numbers per PDF sheet (one on the left half, one on the right half),
matching the physical/human-readable page numbers used throughout the system.

Example: PDF page 12 (a spread) shows "Page 24" on the left and "Page 25" on the right.

Uses PyMuPDF (fitz) to overlay page numbers in a non-intrusive way.
"""

import asyncio
import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any, Callable, Tuple, List

import fitz  # PyMuPDF

from app.services.tracking.checkpoint_recovery_service import (
    checkpoint_recovery_service,
    ProcessingStage
)
from app.utils.pdf_to_images import (
    analyze_pdf_layout,
    PDFLayoutAnalysis,
    PageLayoutType
)

logger = logging.getLogger(__name__)


class PDFPageNumberingService:
    """
    Service to add physical page numbers to PDF pages.

    This runs as a pre-processing step before discovery to ensure
    all pages have clear, visible page numbers for AI processing.
    """

    # Configuration for page number appearance
    DEFAULT_CONFIG = {
        "font_name": "helv",           # Helvetica
        "font_size": 10,               # 10pt font
        "color": (0.4, 0.4, 0.4),      # Medium gray (RGB 0-1 scale)
        "position": "bottom_right",    # Position on page
        "margin_x": 30,                # Margin from edge (points)
        "margin_y": 20,                # Margin from edge (points)
        "prefix": "Page ",             # Text before number
        "background": False,           # Add white background behind text
        "background_color": (1, 1, 1), # White background
        "background_padding": 3,       # Padding around text
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the page numbering service.

        Args:
            config: Optional configuration overrides
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.logger = logging.getLogger(__name__)

    async def add_page_numbers(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        job_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        product_pages: Optional[Dict[str, list]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Add page numbers to all pages of a PDF.

        This is the main entry point for the service. It:
        1. Checks for existing checkpoint (skip if already done)
        2. Opens the PDF
        3. Adds page numbers with progress tracking
        4. Saves to output path
        5. Creates checkpoint

        Args:
            input_path: Path to input PDF
            output_path: Path for output PDF (default: input_path with _numbered suffix)
            job_id: Optional job ID for checkpoint/progress tracking
            progress_callback: Callback for progress updates: (current_page, total_pages, status_message)
            product_pages: Optional dict mapping product names to page lists (for debug annotations)

        Returns:
            Tuple of (output_path, stats_dict)
        """
        start_time = datetime.now()

        # Validate input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input PDF not found: {input_path}")

        # Determine output path
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_numbered{ext}"

        # Check for existing checkpoint
        if job_id:
            checkpoint = await checkpoint_recovery_service.get_last_checkpoint(job_id)
            if checkpoint and checkpoint.get("stage") == ProcessingStage.PDF_PAGES_NUMBERED.value:
                checkpoint_data = checkpoint.get("checkpoint_data", {})
                numbered_pdf_path = checkpoint_data.get("numbered_pdf_path")

                if numbered_pdf_path and os.path.exists(numbered_pdf_path):
                    self.logger.info(f"♻️ [CHECKPOINT] Reusing numbered PDF from checkpoint: {numbered_pdf_path}")
                    return numbered_pdf_path, checkpoint_data
                else:
                    self.logger.warning(f"⚠️ Checkpoint exists but file not found, re-processing")

        self.logger.info(f"📝 Starting page numbering: {input_path}")

        # Run the CPU-intensive work in executor to not block async loop
        loop = asyncio.get_event_loop()
        result_path, stats = await loop.run_in_executor(
            None,
            self._add_page_numbers_sync,
            input_path,
            output_path,
            progress_callback,
            product_pages
        )

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        stats["processing_time_seconds"] = processing_time
        stats["numbered_pdf_path"] = result_path

        # Create checkpoint
        if job_id:
            await checkpoint_recovery_service.create_checkpoint(
                job_id=job_id,
                stage=ProcessingStage.PDF_PAGES_NUMBERED,
                data={
                    "numbered_pdf_path": result_path,
                    "total_pages": stats["total_pages"],
                    "pages_numbered": stats["pages_numbered"],
                    "processing_time_seconds": processing_time,
                },
                metadata={
                    "service": "pdf_page_numbering",
                    "config": self.config,
                }
            )
            self.logger.info(f"💾 Created checkpoint for page numbering: {stats['pages_numbered']} pages")

        # Log final stats
        if stats.get("has_spread_layout"):
            self.logger.info(
                f"✅ Page numbering complete: {stats['physical_pages_numbered']} physical page numbers "
                f"added to {stats['pages_numbered']} PDF sheets in {processing_time:.2f}s"
            )
        else:
            self.logger.info(f"✅ Page numbering complete: {stats['pages_numbered']} pages in {processing_time:.2f}s")
        return result_path, stats

    def _add_page_numbers_sync(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        product_pages: Optional[Dict[str, list]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Synchronous implementation of SPREAD-AWARE page numbering.

        For spread PDFs (2-page spreads), this adds TWO page numbers per PDF sheet:
        - Left half: "Page X" (left physical page number)
        - Right half: "Page Y" (right physical page number)

        For single-page PDFs, adds one page number per page.

        This is called in an executor to not block the async loop.
        """
        stats = {
            "total_pdf_pages": 0,
            "total_physical_pages": 0,
            "total_pages": 0,  # Keep for backward compatibility
            "pages_numbered": 0,
            "physical_pages_numbered": 0,
            "pages_skipped": 0,
            "has_spread_layout": False,
            "errors": [],
        }

        try:
            # Build reverse mapping: physical page -> product (for optional annotations)
            page_to_product = {}
            if product_pages:
                for product_name, pages in product_pages.items():
                    for page in pages:
                        page_to_product[page] = product_name

            # First, analyze the PDF layout to detect spreads
            self.logger.info(f"📐 Analyzing PDF layout...")
            layout = analyze_pdf_layout(input_path)

            stats["total_pdf_pages"] = layout.total_pdf_pages
            stats["total_physical_pages"] = layout.total_physical_pages
            stats["total_pages"] = layout.total_physical_pages  # For backward compat, use physical
            stats["has_spread_layout"] = layout.has_spread_layout

            self.logger.info(
                f"📖 PDF Layout: {layout.total_pdf_pages} PDF sheets -> "
                f"{layout.total_physical_pages} physical pages "
                f"(spread layout: {layout.has_spread_layout})"
            )

            # Open the PDF for editing
            doc = fitz.open(input_path)

            # Process each PDF page using layout info
            for page_info in layout.pages:
                pdf_page_idx = page_info.pdf_page_num
                page = doc[pdf_page_idx]
                rect = page.rect

                try:
                    # Report progress
                    if progress_callback:
                        progress_callback(
                            pdf_page_idx + 1,
                            layout.total_pdf_pages,
                            f"Numbering PDF page {pdf_page_idx + 1}/{layout.total_pdf_pages} "
                            f"(physical pages {page_info.physical_pages})"
                        )

                    # Add page numbers based on layout type
                    if page_info.layout_type in (PageLayoutType.SPREAD, PageLayoutType.SPREAD_FULL_IMAGE):
                        # Spread page: add TWO page numbers (left and right)
                        left_phys = page_info.physical_pages[0]
                        right_phys = page_info.physical_pages[1]

                        # Add left page number (bottom-left of left half)
                        self._add_page_number_to_spread(
                            page, rect, left_phys, "left", page_to_product
                        )
                        stats["physical_pages_numbered"] += 1

                        # Add right page number (bottom-right of right half)
                        self._add_page_number_to_spread(
                            page, rect, right_phys, "right", page_to_product
                        )
                        stats["physical_pages_numbered"] += 1

                    else:
                        # Single page: add ONE page number
                        phys_page = page_info.physical_pages[0]
                        self._add_page_number_to_single(
                            page, rect, phys_page, page_to_product
                        )
                        stats["physical_pages_numbered"] += 1

                    stats["pages_numbered"] += 1

                    # Log progress every 20 PDF pages
                    if (pdf_page_idx + 1) % 20 == 0:
                        self.logger.info(
                            f"   📝 Processed {pdf_page_idx + 1}/{layout.total_pdf_pages} PDF pages"
                        )

                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to number PDF page {pdf_page_idx}: {e}")
                    stats["pages_skipped"] += 1
                    stats["errors"].append({
                        "pdf_page": pdf_page_idx,
                        "physical_pages": page_info.physical_pages,
                        "error": str(e)
                    })

            # Save the modified PDF
            doc.save(output_path)
            doc.close()

            self.logger.info(
                f"💾 Saved numbered PDF: {stats['physical_pages_numbered']} physical page numbers "
                f"added to {stats['pages_numbered']} PDF sheets"
            )

            return output_path, stats

        except Exception as e:
            self.logger.error(f"❌ Page numbering failed: {e}")
            stats["errors"].append({"global": str(e)})
            raise

    def _add_page_number_to_spread(
        self,
        page: fitz.Page,
        rect: fitz.Rect,
        physical_page: int,
        side: str,  # "left" or "right"
        page_to_product: Dict[int, str]
    ) -> None:
        """
        Add a page number to one half of a spread page.

        Args:
            page: PyMuPDF page object
            rect: Page rectangle
            physical_page: Physical page number to display
            side: "left" or "right" half
            page_to_product: Optional mapping for product labels
        """
        margin_x = self.config["margin_x"]
        margin_y = self.config["margin_y"]
        half_width = rect.width / 2

        text = f"{self.config['prefix']}{physical_page}"

        if side == "left":
            # Position in bottom-left of left half
            x = margin_x
            y = rect.height - margin_y
        else:
            # Position in bottom-right of right half
            x = rect.width - margin_x - 50  # Account for text width
            y = rect.height - margin_y

        # Add background if configured
        if self.config["background"]:
            self._add_text_background(page, x, y, text)

        # Insert page number
        page.insert_text(
            (x, y),
            text,
            fontsize=self.config["font_size"],
            fontname=self.config["font_name"],
            color=self.config["color"]
        )

        # Optionally add product label
        if physical_page in page_to_product:
            product_name = page_to_product[physical_page]
            label_x = margin_x if side == "left" else half_width + margin_x
            page.insert_text(
                (label_x, 20),
                f"[{product_name}]",
                fontsize=8,
                fontname=self.config["font_name"],
                color=(0, 0.5, 0)  # Green
            )

    def _add_page_number_to_single(
        self,
        page: fitz.Page,
        rect: fitz.Rect,
        physical_page: int,
        page_to_product: Dict[int, str]
    ) -> None:
        """
        Add a page number to a single (non-spread) page.

        Args:
            page: PyMuPDF page object
            rect: Page rectangle
            physical_page: Physical page number to display
            page_to_product: Optional mapping for product labels
        """
        # Calculate position based on config
        x, y = self._calculate_position(rect, physical_page)

        text = f"{self.config['prefix']}{physical_page}"

        # Add background if configured
        if self.config["background"]:
            self._add_text_background(page, x, y, text)

        # Insert page number
        page.insert_text(
            (x, y),
            text,
            fontsize=self.config["font_size"],
            fontname=self.config["font_name"],
            color=self.config["color"]
        )

        # Optionally add product label
        if physical_page in page_to_product:
            product_name = page_to_product[physical_page]
            page.insert_text(
                (10, 20),
                f"[{product_name}]",
                fontsize=8,
                fontname=self.config["font_name"],
                color=(0, 0.5, 0)  # Green
            )

    def _add_text_background(
        self,
        page: fitz.Page,
        x: float,
        y: float,
        text: str
    ) -> None:
        """Add a background rectangle behind text."""
        font = fitz.Font(fontname=self.config["font_name"])
        text_length = font.text_length(text, fontsize=self.config["font_size"])
        padding = self.config["background_padding"]

        bg_rect = fitz.Rect(
            x - padding,
            y - self.config["font_size"] - padding,
            x + text_length + padding,
            y + padding
        )
        page.draw_rect(bg_rect, color=None, fill=self.config["background_color"])

    def _calculate_position(self, rect: fitz.Rect, page_num: int) -> Tuple[float, float]:
        """
        Calculate the position for the page number based on config.

        Args:
            rect: Page rectangle
            page_num: Page number (for potential alternating positions)

        Returns:
            (x, y) coordinates for text insertion
        """
        position = self.config["position"]
        margin_x = self.config["margin_x"]
        margin_y = self.config["margin_y"]

        if position == "bottom_right":
            x = rect.width - margin_x - 30  # Account for text width
            y = rect.height - margin_y
        elif position == "bottom_left":
            x = margin_x
            y = rect.height - margin_y
        elif position == "bottom_center":
            x = rect.width / 2 - 15  # Approximate center
            y = rect.height - margin_y
        elif position == "top_right":
            x = rect.width - margin_x - 30
            y = margin_y + self.config["font_size"]
        elif position == "top_left":
            x = margin_x
            y = margin_y + self.config["font_size"]
        elif position == "top_center":
            x = rect.width / 2 - 15
            y = margin_y + self.config["font_size"]
        else:
            # Default to bottom right
            x = rect.width - margin_x - 30
            y = rect.height - margin_y

        return x, y


# Global service instance
pdf_page_numbering_service = PDFPageNumberingService()


async def preprocess_pdf_with_page_numbers(
    pdf_path: str,
    job_id: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    output_dir: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to preprocess a PDF by adding page numbers.

    This should be called BEFORE discovery begins.

    Args:
        pdf_path: Path to the input PDF
        job_id: Optional job ID for tracking
        progress_callback: Optional progress callback
        output_dir: Optional directory for output (default: same as input)

    Returns:
        Tuple of (numbered_pdf_path, stats)
    """
    # Determine output path
    if output_dir:
        filename = os.path.basename(pdf_path)
        base, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{base}_numbered{ext}")
    else:
        output_path = None  # Will use default (input_path_numbered.pdf)

    return await pdf_page_numbering_service.add_page_numbers(
        input_path=pdf_path,
        output_path=output_path,
        job_id=job_id,
        progress_callback=progress_callback
    )
