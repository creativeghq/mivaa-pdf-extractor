"""
Stage 1: Focused Extraction

This module handles focused extraction logic - determining which pages to process
based on product discovery results, then extracting ONLY those pages.
"""

import logging
from typing import Dict, Any, Set
import sentry_sdk

from app.schemas.jobs import ProcessingStage
from app.utils.timeout_guard import with_timeout, ProgressiveTimeoutStrategy


async def process_stage_1_focused_extraction(
    file_content: bytes,
    document_id: str,
    catalog: Any,
    page_count: int,
    file_size_mb: float,
    focused_extraction: bool,
    tracker: Any,
    checkpoint_recovery_service: Any,
    job_id: str,

) -> Dict[str, Any]:
    """
    Stage 1: Focused Extraction

    Determines which pages to process based on focused extraction settings,
    then extracts ONLY those pages from the PDF.

    Args:
        file_content: PDF file bytes
        document_id: Document identifier
        catalog: Product catalog from Stage 0
        page_count: Total page count from Stage 0
        file_size_mb: File size in MB from Stage 0
        focused_extraction: Whether to process only product pages
        tracker: Job progress tracker
        logger: Logger instance

    Returns:
        Dictionary containing:
        - product_pages: Set of page numbers to process
        - pdf_result: PDF extraction result with markdown content
    """
    from app.services.pdf_processor import PDFProcessor
    from app.services.supabase_client import get_supabase_client

    logger = logging.getLogger(__name__)
    logger.info("🎯 [STAGE 1] Focused Extraction - Starting...")
    await tracker.update_stage(ProcessingStage.EXTRACTING_TEXT, stage_name="focused_extraction")

    product_pages = set()
    if focused_extraction:
        logger.info(f"   ENABLED - Processing ONLY pages with {len(catalog.products)} products")
        invalid_pages_found = []
        for product in catalog.products:
            # ✅ CRITICAL FIX: Validate page numbers against PDF page count before adding
            # Claude can hallucinate pages that don't exist (e.g., page 73 in a 71-page PDF)
            valid_pages = [p for p in product.page_range if 1 <= p <= page_count]
            invalid_pages = [p for p in product.page_range if p < 1 or p > page_count]

            if invalid_pages:
                invalid_pages_found.extend(invalid_pages)
                logger.warning(f"   ⚠️ Product '{product.name}' has invalid pages {invalid_pages} (PDF has {page_count} pages) - skipping these pages")

            product_pages.update(valid_pages)

        if invalid_pages_found:
            logger.warning(f"   ⚠️ Total invalid pages skipped: {sorted(set(invalid_pages_found))}")

        pages_to_skip = set(range(1, page_count + 1)) - product_pages
        for page_num in pages_to_skip:
            tracker.skip_page_processing(page_num, "Not a product page (focused extraction)")

        logger.info(f"   Product pages: {sorted(product_pages)}")
        logger.info(f"   Processing: {len(product_pages)} / {page_count} pages")
    else:
        logger.info(f"   DISABLED - Processing ALL {page_count} pages")
        product_pages = set(range(1, page_count + 1))

    await tracker._sync_to_database(stage="focused_extraction")

    # ✅ ARCHITECTURE FIX: NOW extract PDF text for ONLY the product pages
    # This prevents OCR from being triggered on all pages
    logger.info(f"📄 [STAGE 1B] Extracting text from {len(product_pages)} product pages...")

    # Calculate progressive timeout for PDF extraction
    pdf_extraction_timeout = ProgressiveTimeoutStrategy.calculate_pdf_extraction_timeout(
        page_count=len(product_pages),  # Only product pages
        file_size_mb=file_size_mb
    )
    logger.info(f"📊 Focused extraction: {len(product_pages)} pages, {file_size_mb:.1f} MB → timeout: {pdf_extraction_timeout:.0f}s")

    pdf_processor = PDFProcessor()
    pdf_result = await with_timeout(
        pdf_processor.process_pdf_from_bytes(
            pdf_bytes=file_content,
            document_id=document_id,
            processing_options={
                'extract_images': False,
                'extract_tables': False,
                'page_list': list(product_pages),  # ✅ KEY FIX: Extract ONLY product pages
                'markdown_timeout': pdf_extraction_timeout  # Pass calculated timeout to PDFProcessor
            }
        ),
        timeout_seconds=pdf_extraction_timeout,
        operation_name="Focused PDF text extraction"
    )

    # ============================================================
    # MONITORING: Track Stage 1 metrics
    # ============================================================
    extracted_pages_count = len(product_pages)
    total_pages_count = page_count
    text_length = len(pdf_result.markdown_content) if pdf_result.markdown_content else 0
    extraction_rate = (extracted_pages_count / total_pages_count) if total_pages_count > 0 else 0

    logger.info(f"✅ [STAGE 1] Focused Extraction Complete:")
    logger.info(f"   Extracted Pages: {extracted_pages_count}/{total_pages_count} ({extraction_rate*100:.1f}%)")
    logger.info(f"   Text Length: {text_length:,} characters")
    logger.info(f"   Focused Extraction: {'ENABLED' if focused_extraction else 'DISABLED'}")

    # Update processed_documents with extracted content
    supabase = get_supabase_client()
    try:
        supabase.client.table('processed_documents').update({
            "content": pdf_result.markdown_content or "",
            "metadata": {
                "filename": document_id,  # Will be updated with actual filename later
                "file_size": len(file_content),
                "page_count": page_count,
                "extracted_pages": extracted_pages_count,
                "text_length": text_length,
                "extraction_rate": extraction_rate
            }
        }).eq('id', document_id).execute()
        logger.info(f"✅ Updated processed_documents with extracted content")
    except Exception as e:
        logger.warning(f"⚠️ Failed to update processed_documents content: {e}")
        sentry_sdk.capture_exception(e)

    # Stage 1 progress: 10% → 30% (fixed when complete)
    await tracker.update_stage(
        ProcessingStage.EXTRACTING_TEXT,
        stage_name="focused_extraction",
        progress_percentage=30
    )

    await tracker._sync_to_database(stage="focused_extraction")

    logger.info(f"📊 Progress updated: 30% (Stage 1 complete - {extracted_pages_count} pages extracted)")

    # Create PDF_EXTRACTED checkpoint
    from app.services.checkpoint_recovery_service import ProcessingStage as CheckpointStage
    await checkpoint_recovery_service.create_checkpoint(
        job_id=job_id,
        stage=CheckpointStage.PDF_EXTRACTED,
        data={
            "document_id": document_id,
            "product_pages": sorted(list(product_pages)),
            "total_pages": total_pages_count,
            "extracted_pages": extracted_pages_count
        },
        metadata={
            "focused_extraction": focused_extraction,
            "file_size_mb": file_size_mb,
            # NEW: Add comprehensive metrics to checkpoint
            "text_length": text_length,
            "extraction_rate": extraction_rate
        }
    )
    logger.info(f"✅ Created PDF_EXTRACTED checkpoint for job {job_id}")

    # ============================================================
    # RETURN DATA: Include comprehensive metrics
    # ============================================================
    return {
        "product_pages": product_pages,
        "pdf_result": pdf_result,
        # NEW: Return Stage 1 metrics
        "extracted_pages_count": extracted_pages_count,
        "total_pages_count": total_pages_count,
        "text_length": text_length,
        "extraction_rate": extraction_rate,
        "focused_extraction": focused_extraction
    }

