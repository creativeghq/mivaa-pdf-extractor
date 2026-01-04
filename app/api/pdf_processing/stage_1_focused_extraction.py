"""
Stage 1: Focused Extraction

This module handles page extraction for individual products in the product-centric pipeline.
"""

import logging
from typing import Set, Any, Optional


async def extract_product_pages(
    file_content: bytes,
    product: Any,
    document_id: str,
    job_id: str,
    logger: logging.Logger,
    total_pages: Optional[int] = None
) -> Set[int]:
    """
    Extract pages for a single product (product-centric pipeline).

    This function identifies which pages belong to a specific product
    based on the product's page_range from discovery.

    Args:
        file_content: PDF file bytes (not used, kept for API consistency)
        product: Single product object from catalog
        document_id: Document identifier (not used, kept for API consistency)
        job_id: Job identifier (not used, kept for API consistency)
        logger: Logger instance
        total_pages: Optional total pages in PDF for validation

    Returns:
        Set of page numbers for this product
    """
    logger.info(f"📄 Extracting pages for product: {product.name}")
    logger.info(f"   Page range: {product.page_range}")

    # Validate page numbers (in case AI hallucinated invalid pages)
    product_pages = set()
    if product.page_range:
        for p in product.page_range:
            if p > 0:
                if total_pages and p > total_pages:
                    logger.warning(f"   ⚠️ Skipping hallucinated page {p} (PDF has only {total_pages} pages)")
                    continue
                product_pages.add(p)

    logger.info(f"   ✅ Product pages: {sorted(product_pages)} ({len(product_pages)} pages)")

    return product_pages

