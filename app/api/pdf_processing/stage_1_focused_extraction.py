"""
Stage 1: Focused Extraction with PaddleOCR Layout Detection

This module handles page extraction and layout analysis for individual products
in the product-centric pipeline.

IMPORTANT: This module uses PHYSICAL PAGE NUMBERS (1-based) throughout.
Physical pages are what users see in catalogs. PDF sheet indices are only
used internally when accessing PyMuPDF - never exposed to other stages.

Features:
- Page validation (physical pages)
- Layout detection (TEXT, IMAGE, TABLE, TITLE, CAPTION regions)
- Layout region storage in database
- Caption-to-image linking
"""

import logging
from typing import Any, Optional, Dict, List
from app.schemas.page_types import PhysicalPageBound


async def extract_product_pages(
    file_content: bytes,
    product: Any,
    document_id: str,
    job_id: str,
    logger: logging.Logger,
    physical_page_upper_bound: Optional[PhysicalPageBound] = None,
    enable_layout_detection: bool = True,
    product_id: Optional[str] = None,
    catalog: Optional[Any] = None,  # Catalog with spread layout info
) -> Dict[str, Any]:
    """
    Extract pages and detect layout regions for a single product.

    IMPORTANT: Returns PHYSICAL PAGE NUMBERS (1-based) as the primary output.
    PDF sheet indices are only used internally for PyMuPDF access.

    This function:
    1. Validates PHYSICAL page numbers
    2. Handles spread layouts internally (for layout detection)
    3. Detects layout regions using the PaddleOCR structural pass (if enabled)
    4. Stores layout regions in database
    5. Links captions to images

    Args:
        file_content: PDF file bytes
        product: Single product object from catalog
        document_id: Document identifier
        job_id: Job identifier
        logger: Logger instance
        physical_page_upper_bound: Largest physical page number in the document.
            Used as the upper bound for validating product.page_range entries.
            For spread-layout PDFs (where 1 PDF sheet = 2 physical pages), this
            MUST be the physical page count (e.g. 140), NOT the PDF sheet count
            (e.g. 71). Pages above this bound are dropped silently.
            See bug 2026-05-01: passing PDF-sheet-count silently dropped 50%
            of products in a spread-layout catalog.
        enable_layout_detection: Enable layout detection (default: True)
        product_id: Database product ID (required for layout storage)
        catalog: Optional catalog with spread layout info (has_spread_layout, physical_to_pdf_map)

    Returns:
        Dict with:
        - physical_pages: List of physical page numbers (1-based) - PRIMARY OUTPUT
        - layout_regions: List of detected layout regions (if enabled)
        - layout_stats: Statistics on detected regions
        - has_spread_layout: Whether document uses spread layout
        - physical_to_pdf_map: Mapping for internal PDF access (passed through from catalog)
    """
    logger.info(f"📄 [STAGE 1] Extracting pages for product: {product.name}")
    logger.info(f"   Physical page range: {product.page_range}")
    logger.info(f"   Physical page upper bound: {physical_page_upper_bound}")
    logger.info(f"   Layout detection: {'ENABLED' if enable_layout_detection else 'DISABLED'}")

    # Check for spread layout
    has_spread_layout = catalog and getattr(catalog, 'has_spread_layout', False)
    physical_to_pdf_map = catalog.physical_to_pdf_map if catalog and hasattr(catalog, 'physical_to_pdf_map') else {}

    if has_spread_layout:
        logger.info(f"   📐 Spread layout detected")
        logger.info(f"   📐 Physical-to-PDF mapping available: {len(physical_to_pdf_map)} entries")

    # ========================================================================
    # STEP 1: Validate Physical Pages (NO CONVERSION - keep as physical pages)
    # ========================================================================
    physical_pages = []  # Physical page numbers (1-based) - this is our PRIMARY output
    pages_dropped_out_of_bounds: List[int] = []

    if product.page_range:
        for physical_page in product.page_range:
            # Validate physical page is within the document's PHYSICAL page bound.
            # Caller must pass catalog.total_pages (physical), NOT len(fitz_doc)
            # which is the PDF sheet count and can be half of the physical count
            # for spread-layout PDFs.
            if physical_page_upper_bound and physical_page > physical_page_upper_bound:
                pages_dropped_out_of_bounds.append(physical_page)
                continue
            if physical_page > 0:
                physical_pages.append(physical_page)

    # Out-of-bounds drops should never happen for a correctly-bounded call.
    # Promote to ERROR with full context so the next regression of this kind
    # fires loud immediately, not after 13 silent warnings (see bug 2026-05-01).
    if pages_dropped_out_of_bounds:
        requested = len(product.page_range or [])
        dropped = len(pages_dropped_out_of_bounds)
        logger.error(
            f"   ❌ [STAGE 1] {product.name}: dropped {dropped}/{requested} pages "
            f"as out-of-bounds (>{physical_page_upper_bound}). "
            f"Dropped pages: {pages_dropped_out_of_bounds}. "
            f"This usually means the caller passed PDF-sheet-count instead of "
            f"physical-page-count for `physical_page_upper_bound`. "
            f"For spread-layout PDFs, pass catalog.total_pages (physical), "
            f"not len(fitz_doc) (sheets)."
        )

    logger.info(f"   ✅ Validated {len(physical_pages)} physical pages: {physical_pages}")

    # ========================================================================
    # STEP 2: Layout Detection — REMOVED (dead code, 2026-06-14)
    # ========================================================================
    # The per-product layout-collection path here was dead: the only caller
    # (product_processor.extract_product_pages) passes enable_layout_detection=False.
    # Layout regions are owned by the PaddleOCR structural pass (Stage 1.5), and
    # IMAGE/FIGURE region crops now come from the spread-aware
    # region_crop_extractor invoked in Stage 3. `layout_regions` / `layout_stats`
    # stay empty here to preserve the return contract for consumers that still
    # read those keys (product_processor, stage_2 chunking).
    layout_regions: List[Any] = []
    layout_stats: Dict[str, Any] = {}

    # Return comprehensive result - PHYSICAL PAGES are the primary output
    return {
        'physical_pages': physical_pages,  # Physical page numbers (1-based) - PRIMARY OUTPUT
        'layout_regions': layout_regions,
        'layout_stats': layout_stats,
        # Pass through spread layout info for stages that need PDF access
        'has_spread_layout': has_spread_layout,
        'physical_to_pdf_map': physical_to_pdf_map
    }
