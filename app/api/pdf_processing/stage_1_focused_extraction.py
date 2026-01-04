"""
Stage 1: Focused Extraction

This module handles page extraction for individual products in the product-centric pipeline.
"""

import logging
from typing import Set, Any


async def extract_product_pages(
    file_content: bytes,
    product: Any,
    document_id: str,
    job_id: str,
    logger: logging.Logger
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

    Returns:
        Set of page numbers for this product
    """
    logger.info(f"📄 Extracting pages for product: {product.name}")
    logger.info(f"   Page range: {product.page_range}")

    # Validate page numbers (in case AI hallucinated invalid pages)
    # We don't have page_count here, so we'll trust the product's page_range
    # The actual PDF extraction will fail if pages don't exist
    product_pages = set(product.page_range)

    logger.info(f"   ✅ Product pages: {sorted(product_pages)} ({len(product_pages)} pages)")

    return product_pages

