"""
Stage 3: Image Processing

This module handles image processing for individual products in the product-centric pipeline.

IMPORTANT: This module uses PHYSICAL PAGE NUMBERS (1-based) throughout.
Physical pages are what users see in catalogs. PDF sheet indices are only
used internally when accessing PyMuPDF - never exposed to other modules.
"""

import os
import logging
from typing import Dict, Any, Optional, List

# ============================================================================
# SINGLETON PDF PROCESSOR - Reuse across all products to prevent re-initialization
# ============================================================================
_pdf_processor_instance: Optional[Any] = None


def get_pdf_processor():
    """Get or create singleton PDFProcessor instance."""
    global _pdf_processor_instance
    if _pdf_processor_instance is None:
        from app.services.pdf.pdf_processor import PDFProcessor
        _pdf_processor_instance = PDFProcessor()
        logging.getLogger(__name__).info("♻️ Created singleton PDFProcessor for Stage 3")
    return _pdf_processor_instance


def clear_pdf_processor():
    """Clear the singleton PDFProcessor (call at job completion)."""
    global _pdf_processor_instance
    _pdf_processor_instance = None


def _get_pdf_index_for_physical_page(
    physical_page: int,
    has_spread_layout: bool,
    physical_to_pdf_map: Dict[int, tuple]
) -> tuple:
    """
    Convert a physical page number to PDF sheet index and position.

    Args:
        physical_page: Physical page number (1-based)
        has_spread_layout: Whether document uses spread layout
        physical_to_pdf_map: Mapping from physical page to (pdf_idx, position)

    Returns:
        Tuple of (pdf_idx, position) where pdf_idx is 0-based
        position is 'left', 'right', or 'single'
    """
    if has_spread_layout and physical_page in physical_to_pdf_map:
        return physical_to_pdf_map[physical_page]
    else:
        # Non-spread: simple 1-based to 0-based conversion
        return (physical_page - 1, 'single')


async def process_product_images(
    file_content: bytes,
    document_id: str,
    workspace_id: str,
    job_id: str,
    product: Any,
    physical_pages: List[int],  # ✅ NOW USING PHYSICAL PAGES (1-based)
    catalog: Any,
    config: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Process images for a single product (product-centric pipeline).

    IMPORTANT: Uses PHYSICAL PAGE NUMBERS (1-based) throughout.
    PDF sheet indices are only used internally for PyMuPDF access.

    Args:
        file_content: PDF file bytes
        document_id: Document identifier
        workspace_id: Workspace identifier
        job_id: Job identifier
        product: Single product object
        physical_pages: List of physical page numbers (1-based) for this product
        catalog: Full catalog (for spread layout info)
        config: Processing configuration
        logger: Logger instance

    Returns:
        Dictionary with images_processed, images_material, images_non_material counts
    """
    from app.services.images.image_processing_service import ImageProcessingService

    logger.info(f"🖼️  Processing images for product: {product.name}")
    logger.info(f"   Physical pages (1-based): {sorted(physical_pages)}")

    # Get spread layout info from catalog
    has_spread_layout = catalog and getattr(catalog, 'has_spread_layout', False)
    physical_to_pdf_map = catalog.physical_to_pdf_map if catalog and hasattr(catalog, 'physical_to_pdf_map') else {}

    if has_spread_layout:
        logger.info(f"   📐 Spread layout detected")

    # ✅ Use singleton PDFProcessor to prevent repeated initialization
    pdf_processor = get_pdf_processor()

    # ========================================================================
    # EXTRACT IMAGES FOR EACH PHYSICAL PAGE
    # ========================================================================
    # Group physical pages by their PDF sheet to avoid redundant extraction
    # For spreads: multiple physical pages share one PDF sheet
    pdf_sheets_to_extract: Dict[int, List[int]] = {}  # pdf_idx -> [physical_pages]

    for physical_page in sorted(physical_pages):
        pdf_idx, position = _get_pdf_index_for_physical_page(
            physical_page, has_spread_layout, physical_to_pdf_map
        )
        if pdf_idx not in pdf_sheets_to_extract:
            pdf_sheets_to_extract[pdf_idx] = []
        pdf_sheets_to_extract[pdf_idx].append(physical_page)

    logger.info(f"   📋 Physical pages grouped by PDF sheet:")
    for pdf_idx, pages in sorted(pdf_sheets_to_extract.items()):
        logger.info(f"      PDF sheet {pdf_idx} (1-based: {pdf_idx + 1}) -> physical pages {pages}")

    # Process each unique PDF sheet once
    extracted_images_list = []

    for pdf_idx, sheet_physical_pages in sorted(pdf_sheets_to_extract.items()):
        pdf_page_1based = pdf_idx + 1

        logger.info(f"   🔍 Extracting from PDF sheet {pdf_page_1based} for physical pages {sheet_physical_pages}")

        # Extract images for this PDF sheet
        processing_options = {
            'extract_images': True,
            'extract_text': False,
            'extract_tables': False,
            'page_list': [pdf_page_1based],  # 1-based for pdf_processor
            'extract_categories': ['products'],
            'job_id': job_id
        }

        page_result = await pdf_processor.process_pdf_from_bytes(
            pdf_bytes=file_content,
            document_id=document_id,
            processing_options=processing_options
        )

        extraction_count = len(page_result.extracted_images) if page_result.extracted_images else 0
        logger.info(f"   📸 PDF sheet {pdf_page_1based}: Extracted {extraction_count} images")

        if not page_result.extracted_images:
            logger.warning(f"   ⚠️ No images extracted from PDF sheet {pdf_page_1based}")
            continue

        # ========================================================================
        # ASSIGN IMAGES TO CORRECT PHYSICAL PAGES
        # ========================================================================
        if has_spread_layout and len(sheet_physical_pages) == 2:
            # Spread layout: determine which physical page each image belongs to
            import fitz

            # Get sheet width for side detection
            sheet_width = 0
            if catalog and hasattr(catalog, 'pdf_page_widths') and catalog.pdf_page_widths:
                sheet_width = catalog.pdf_page_widths.get(pdf_idx, 0)

            if sheet_width == 0:
                doc = fitz.open(stream=file_content, filetype="pdf")
                sheet = doc[pdf_idx]
                sheet_width = sheet.rect.width
                doc.close()

            mid_x = sheet_width / 2
            left_phys = min(sheet_physical_pages)
            right_phys = max(sheet_physical_pages)

            for img in page_result.extracted_images:
                bbox = img.get('bbox', [0, 0, 0, 0])
                img_width = bbox[2] if len(bbox) > 2 else 0
                center_x = bbox[0] + (img_width / 2) if len(bbox) > 0 else 0

                # Scene detection: wide image spanning both pages
                spans_middle = len(bbox) >= 3 and (bbox[0] < mid_x) and (bbox[0] + img_width > mid_x)
                is_scene = spans_middle and (img_width > sheet_width * 0.45)

                if is_scene:
                    # Scene spans both pages - assign to left page with scene flag
                    img['page_number'] = left_phys  # Physical page (1-based)
                    img['physical_side'] = 'spread'
                    img['is_scene'] = True
                    logger.info(f"      🎞️ Scene image detected spanning pages {left_phys}-{right_phys}")
                else:
                    # Normal image - assign to left or right physical page
                    is_left = center_x < mid_x
                    img['page_number'] = left_phys if is_left else right_phys  # Physical page (1-based)
                    img['physical_side'] = 'left' if is_left else 'right'

                extracted_images_list.append(img)
        else:
            # Single page or non-spread: all images go to the physical page
            target_physical_page = sheet_physical_pages[0]
            for img in page_result.extracted_images:
                img['page_number'] = target_physical_page  # Physical page (1-based)
                extracted_images_list.append(img)

    total_images = len(extracted_images_list)

    # ========================================================================
    # LOG EXTRACTION RESULTS
    # ========================================================================
    if extracted_images_list:
        images_by_page = {}
        for img in extracted_images_list:
            page_num = img.get('page_number', 'unknown')
            images_by_page[page_num] = images_by_page.get(page_num, 0) + 1

        logger.info(f"      Images per physical page: {dict(sorted(images_by_page.items()))}")

        extraction_methods = {}
        for img in extracted_images_list:
            method = img.get('extraction_method', 'unknown')
            extraction_methods[method] = extraction_methods.get(method, 0) + 1

        logger.info(f"      Extraction methods: {extraction_methods}")

    if total_images == 0:
        logger.warning(f"   ⚠️ NO IMAGES EXTRACTED!")
        logger.warning(f"      Physical pages requested: {sorted(physical_pages)}")
        logger.warning(f"      This could mean:")
        logger.warning(f"      1. Pages are text-only (no embedded images)")
        logger.warning(f"      2. Images were filtered out by size/quality thresholds")
        logger.warning(f"      3. YOLO endpoint returned errors")
        return {'images_processed': 0, 'images_material': 0, 'images_non_material': 0, 'clip_embeddings_generated': 0}

    logger.info(f"   ✅ Extracted {total_images} images from {len(physical_pages)} physical pages")

    # ========================================================================
    # VERIFY IMAGE FILES EXIST
    # ========================================================================
    logger.info(f"   Verifying extracted image files...")
    missing_files = []
    for i, img in enumerate(extracted_images_list[:5]):
        img_path = img.get('path', '')
        exists = os.path.exists(img_path) if img_path else False
        logger.debug(f"     Image {i+1}: {img.get('filename')} - exists: {exists}")
        if not exists:
            missing_files.append(img.get('filename'))

    if missing_files:
        logger.error(f"   ❌ WARNING: {len(missing_files)} image files are missing!")
        logger.error(f"      Missing files: {missing_files[:3]}")

    # ========================================================================
    # CLASSIFY IMAGES (Material vs Non-Material)
    # ========================================================================
    image_service = ImageProcessingService()

    logger.info(f"   🤖 Starting AI classification of {total_images} images...")

    try:
        material_images, non_material_images = await image_service.classify_images(
            extracted_images=extracted_images_list,
            confidence_threshold=0.6,
            primary_model="Qwen/Qwen3-VL-32B-Instruct",
            validation_model="claude-sonnet-4-20250514",
            batch_size=15
        )
    except Exception as e:
        logger.error(f"   ❌ Image classification failed: {type(e).__name__}: {str(e)}")
        raise

    non_material_count = len(non_material_images)

    logger.info(f"   📊 CLASSIFICATION RESULTS:")
    logger.info(f"      Material images: {len(material_images)}")
    logger.info(f"      Non-material images: {non_material_count}")

    # ========================================================================
    # UPLOAD MATERIAL IMAGES TO STORAGE
    # ========================================================================
    logger.info(f"   📤 UPLOAD STAGE: Uploading {len(material_images)} material images...")
    uploaded_images = await image_service.upload_images_to_storage(
        material_images=material_images,
        document_id=document_id
    )

    logger.info(f"      Successfully uploaded: {len(uploaded_images)}/{len(material_images)}")

    # ========================================================================
    # SAVE TO DATABASE AND GENERATE CLIP EMBEDDINGS
    # ========================================================================
    # Get material_category from config (passed from upload settings)
    material_category = config.get('material_category')
    logger.info(f"   💾 DATABASE STAGE: Saving metadata and generating CLIP embeddings...")
    logger.info(f"      Material category: {material_category or 'not specified'}")
    save_result = await image_service.save_images_and_generate_clips(
        material_images=uploaded_images,
        document_id=document_id,
        workspace_id=workspace_id,
        material_category=material_category
    )

    images_processed = save_result.get('images_saved', 0)
    clip_embeddings = save_result.get('clip_embeddings_generated', 0)
    failed_images = save_result.get('failed_images', [])

    logger.info(f"   📊 DATABASE RESULTS:")
    logger.info(f"      Images saved to DB: {images_processed}/{len(uploaded_images)}")
    logger.info(f"      CLIP embeddings generated: {clip_embeddings}/{images_processed}")

    if failed_images:
        logger.warning(f"      ⚠️ Failed to save {len(failed_images)} images")

    logger.info(f"   ✅ STAGE 3 COMPLETE for {product.name}")
    logger.info(f"      Total extracted: {total_images}")
    logger.info(f"      Material images: {len(material_images)}")
    logger.info(f"      Successfully processed: {images_processed}")
    logger.info(f"      CLIP embeddings: {clip_embeddings}")

    return {
        'images_processed': images_processed,
        'clip_embeddings_generated': clip_embeddings,
        'images_material': len(material_images),
        'images_non_material': non_material_count
    }
