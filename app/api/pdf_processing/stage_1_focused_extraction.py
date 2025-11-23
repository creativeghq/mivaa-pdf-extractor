"""
Stage 1: Focused Extraction

This module handles focused extraction logic - determining which pages to process
based on product discovery results.
"""

from typing import Dict, Any, Set
from app.schemas.jobs import ProcessingStage


async def process_stage_1_focused_extraction(
    catalog: Any,
    pdf_result: Any,
    focused_extraction: bool,
    tracker: Any,
    logger: Any
) -> Dict[str, Any]:
    """
    Stage 1: Focused Extraction
    
    Determines which pages to process based on focused extraction settings.
    
    Args:
        catalog: Product catalog from Stage 0
        pdf_result: PDF extraction result with page count
        focused_extraction: Whether to process only product pages
        tracker: Job progress tracker
        logger: Logger instance
        
    Returns:
        Dictionary containing:
        - product_pages: Set of page numbers to process
    """
    logger.info("🎯 [STAGE 1] Focused Extraction - Starting...")
    await tracker.update_stage(ProcessingStage.EXTRACTING_TEXT, stage_name="focused_extraction")
    
    product_pages = set()
    if focused_extraction:
        logger.info(f"   ENABLED - Processing ONLY pages with {len(catalog.products)} products")
        invalid_pages_found = []
        for product in catalog.products:
            # ✅ CRITICAL FIX: Validate page numbers against PDF page count before adding
            # Claude can hallucinate pages that don't exist (e.g., page 73 in a 71-page PDF)
            valid_pages = [p for p in product.page_range if 1 <= p <= pdf_result.page_count]
            invalid_pages = [p for p in product.page_range if p < 1 or p > pdf_result.page_count]
            
            if invalid_pages:
                invalid_pages_found.extend(invalid_pages)
                logger.warning(f"   ⚠️ Product '{product.name}' has invalid pages {invalid_pages} (PDF has {pdf_result.page_count} pages) - skipping these pages")
            
            product_pages.update(valid_pages)
        
        if invalid_pages_found:
            logger.warning(f"   ⚠️ Total invalid pages skipped: {sorted(set(invalid_pages_found))}")
        
        pages_to_skip = set(range(1, pdf_result.page_count + 1)) - product_pages
        for page_num in pages_to_skip:
            tracker.skip_page_processing(page_num, "Not a product page (focused extraction)")
        
        logger.info(f"   Product pages: {sorted(product_pages)}")
        logger.info(f"   Processing: {len(product_pages)} / {pdf_result.page_count} pages")
    else:
        logger.info(f"   DISABLED - Processing ALL {pdf_result.page_count} pages")
        product_pages = set(range(1, pdf_result.page_count + 1))
    
    await tracker._sync_to_database(stage="focused_extraction")
    
    return {
        "product_pages": product_pages
    }

