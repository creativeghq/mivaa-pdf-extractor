"""
Stage 1: Focused Extraction

This module handles page extraction for individual products in the product-centric pipeline.
"""

import logging
from typing import Set, Any, Optional
from app.utils.page_converter import PageConverter


async def extract_product_pages(
    file_content: bytes,
    product: Any,
    document_id: str,
    job_id: str,
    logger: logging.Logger,
    total_pages: Optional[int] = None,
    pages_per_sheet: int = 1
) -> Set[int]:
    """
    Extract pages for a single product (product-centric pipeline).

    This function identifies which pages belong to a specific product
    based on the product's page_range from discovery, mapping catalog
    pages to physical PDF pages (e.g., for 2-page spreads).

    Args:
        file_content: PDF file bytes (not used, kept for API consistency)
        product: Single product object from catalog
        document_id: Document identifier (not used, kept for API consistency)
        job_id: Job identifier (not used, kept for API consistency)
        logger: Logger instance
        total_pages: Optional total pages in PDF for validation
        pages_per_sheet: Number of catalog pages per PDF sheet (default: 1)

    Returns:
        Set of physical PDF page indices (0-based) for this product
    """
    logger.info(f"📄 Mapping pages for product: {product.name} (layout: {pages_per_sheet} pages/sheet)")
    logger.info(f"   Catalog page range: {product.page_range}")

    # ✅ USE PAGE CONVERTER: Type-safe conversion
    converter = PageConverter(pages_per_sheet=pages_per_sheet, total_pdf_pages=total_pages)

    # Validate and map page numbers
    product_pages = set()
    if product.page_range:
        for p in product.page_range:
            if p > 0:
                try:
                    # Convert catalog page to PDF page index using PageConverter
                    page = converter.from_catalog_page(p)
                    product_pages.add(page.array_index)
                except ValueError as e:
                    # Page is out of bounds
                    logger.warning(
                        f"   ⚠️ Skipping out-of-bounds page: Catalog {p} -> {e}"
                    )
                    continue

    logger.info(f"   ✅ Mapped to PDF indices: {sorted(product_pages)} ({len(product_pages)} pages)")

    return product_pages
