"""
Stage 3: Image Processing

This module handles image processing for individual products in the product-centric pipeline.

IMPORTANT: This module uses PHYSICAL PAGE NUMBERS (1-based) throughout.
Physical pages are what users see in catalogs. PDF sheet indices are only
used internally when accessing PyMuPDF - never exposed to other modules.

It also exposes `process_catalog_wide_icons()`, a once-per-document pre-pass
that scans the PDF's SUPPLEMENTARY pages (pages not assigned to any product
during discovery — i.e. shared legend / iconography / regulation / care /
certification pages) for spec icon strips and routes them through the same
OCR + Claude icon extraction pipeline used for per-product icons. The
resulting `document_images` rows are stored with the document_id, so the
existing `_merge_icon_metadata_into_product` rollup in Stage 4 naturally
picks them up for every product in the catalog without any per-product
association logic.
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
    logger: logging.Logger,
    layout_regions: Optional[List[Any]] = None,  # ✅ NEW: YOLO layout regions with bbox data
    tracker: Optional[Any] = None,  # ✅ NEW: ProgressTracker for per-image progress events
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
        layout_regions: Optional YOLO layout regions with bbox data for accurate positioning
        tracker: Optional ProgressTracker — when provided, image-level progress
                 events ("Image 12/50 — v:12 c:12 t:12 s:11 m:12 u:12") are
                 pushed to background_jobs so the admin UI can display them.

    Returns:
        Dictionary with images_processed, images_material, images_non_material counts,
        clip_embeddings_generated, vector_stats (per-vector counts), failed_images.
    """
    from app.services.images.image_processing_service import ImageProcessingService

    logger.info(f"🖼️  Processing images for product: {product.name}")
    logger.info(f"   Physical pages (1-based): {sorted(physical_pages)}")

    # Get spread layout info from catalog
    has_spread_layout = catalog and getattr(catalog, 'has_spread_layout', False)
    physical_to_pdf_map = catalog.physical_to_pdf_map if catalog and hasattr(catalog, 'physical_to_pdf_map') else {}

    if has_spread_layout:
        logger.info(f"   📐 Spread layout detected")

    # ✅ NEW: Build YOLO region lookup by physical page for better bbox data
    yolo_regions_by_page: Dict[int, List[Any]] = {}
    if layout_regions:
        logger.info(f"   🎯 YOLO layout regions available: {len(layout_regions)} regions")
        for region in layout_regions:
            # Get physical page from region (set in product_processor.py)
            page_num = getattr(region.bbox, 'page', None) if hasattr(region, 'bbox') else None
            if page_num:
                if page_num not in yolo_regions_by_page:
                    yolo_regions_by_page[page_num] = []
                yolo_regions_by_page[page_num].append(region)
        logger.info(f"   📍 YOLO regions by page: {dict((k, len(v)) for k, v in yolo_regions_by_page.items())}")

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

            # Track images without valid bbox for fallback assignment
            images_without_bbox = []

            for img_idx, img in enumerate(page_result.extracted_images):
                bbox = img.get('bbox')

                # ✅ FIX: Properly handle None or invalid bbox
                has_valid_bbox = (
                    bbox is not None and
                    isinstance(bbox, (list, tuple)) and
                    len(bbox) >= 3 and
                    (bbox[2] > 0 or bbox[0] > 0)  # Has actual position/size data
                )

                if has_valid_bbox:
                    img_width = bbox[2] if len(bbox) > 2 else 0
                    center_x = bbox[0] + (img_width / 2)

                    # Scene detection: wide image spanning both pages
                    spans_middle = (bbox[0] < mid_x) and (bbox[0] + img_width > mid_x)
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
                        logger.debug(f"      📍 Image {img_idx} assigned to {'left' if is_left else 'right'} page (center_x={center_x:.1f}, mid_x={mid_x:.1f})")
                else:
                    # ✅ FIX: Track images without bbox for fallback assignment
                    images_without_bbox.append((img_idx, img))
                    continue  # Will assign after loop

                extracted_images_list.append(img)

            # ✅ FIX: Fallback for images without valid bbox
            # Try YOLO regions first, then distribute evenly
            if images_without_bbox:
                logger.warning(f"      ⚠️ {len(images_without_bbox)} images without valid PyMuPDF bbox")

                # ✅ NEW: Try to match images with YOLO regions by filename pattern
                yolo_left_regions = yolo_regions_by_page.get(left_phys, [])
                yolo_right_regions = yolo_regions_by_page.get(right_phys, [])

                if yolo_left_regions or yolo_right_regions:
                    logger.info(f"      🎯 Using YOLO regions: {len(yolo_left_regions)} left, {len(yolo_right_regions)} right")

                for fallback_idx, (img_idx, img) in enumerate(images_without_bbox):
                    filename = img.get('filename', '')
                    assigned = False

                    # ✅ Check if this is a YOLO region image (filename contains 'yolo_region')
                    if 'yolo_region' in filename:
                        # YOLO images already have physical page from YOLO detection
                        # Try to match by region index in filename
                        import re
                        region_match = re.search(r'yolo_region_(\d+)', filename)
                        if region_match:
                            region_idx = int(region_match.group(1))
                            # Check if this region was detected on left or right page
                            for region in yolo_left_regions:
                                if hasattr(region, 'bbox') and region.bbox:
                                    bbox = region.bbox
                                    center = bbox.x + (bbox.width / 2) if hasattr(bbox, 'x') else 0
                                    if center < mid_x:
                                        img['page_number'] = left_phys
                                        img['physical_side'] = 'left'
                                        img['yolo_assisted'] = True
                                        assigned = True
                                        logger.debug(f"      📍 YOLO image {filename} assigned to left page via YOLO bbox")
                                        break
                            if not assigned:
                                for region in yolo_right_regions:
                                    if hasattr(region, 'bbox') and region.bbox:
                                        img['page_number'] = right_phys
                                        img['physical_side'] = 'right'
                                        img['yolo_assisted'] = True
                                        assigned = True
                                        logger.debug(f"      📍 YOLO image {filename} assigned to right page via YOLO bbox")
                                        break

                    # ✅ Final fallback: alternate between left and right
                    if not assigned:
                        is_left = (fallback_idx % 2 == 0)
                        img['page_number'] = left_phys if is_left else right_phys
                        img['physical_side'] = 'left' if is_left else 'right'
                        img['bbox_fallback'] = True
                        logger.debug(f"      📍 Image {img_idx} ({filename}) fallback assigned to {'left' if is_left else 'right'} page")

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

    # 🔍 BBOX TRACE: Log bbox before classification
    logger.info(f"   🔍 [BBOX TRACE] Before classification - checking {len(extracted_images_list)} images:")
    for i, img in enumerate(extracted_images_list[:5]):  # Log first 5
        bbox = img.get('bbox')
        bbox_len = len(bbox) if isinstance(bbox, (list, tuple)) else 'N/A'
        logger.info(f"      Image {i}: {img.get('filename')}, bbox_len={bbox_len}, bbox={bbox[:4] if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else bbox}, id={id(img)}")

    logger.info(f"   🤖 Starting AI classification of {total_images} images...")

    try:
        material_images, non_material_images = await image_service.classify_images(
            extracted_images=extracted_images_list,
            confidence_threshold=0.6,
            primary_model="Qwen/Qwen3-VL-32B-Instruct",
            validation_model="claude-sonnet-4-7",
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
    # ICON-CANDIDATE SPLIT
    # ========================================================================
    # Re-route small grid-of-icons images (R-rating, PEI, slip, fire ratings,
    # packaging icons, …) to the icon extraction path so they get OCR + Claude
    # → spec metadata into products.metadata, NOT visual SLIG embeddings.
    # The DECORATIVE override re-routes images Qwen labelled as DECORATIVE if
    # they ALSO meet the icon size + grid rules (Qwen often misclassifies
    # spec icons as decoration).
    regular_material_images, icon_candidates, remaining_non_material = (
        image_service._split_material_and_icon_candidates(
            material_images=material_images,
            non_material_images=non_material_images,
        )
    )

    # The non_material list shrinks if any of its decorative entries got
    # promoted to the icon path (DECORATIVE override).
    non_material_count = len(remaining_non_material)

    # 🔍 BBOX TRACE: Log bbox after classification + icon split
    logger.info(
        f"   🔍 [BBOX TRACE] After split - checking {len(regular_material_images)} regular + "
        f"{len(icon_candidates)} icon candidates:"
    )
    for i, img in enumerate(regular_material_images[:5]):  # Log first 5
        bbox = img.get('bbox')
        bbox_len = len(bbox) if isinstance(bbox, (list, tuple)) else 'N/A'
        logger.info(f"      Image {i}: {img.get('filename')}, bbox_len={bbox_len}, bbox={bbox[:4] if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else bbox}, id={id(img)}")

    # ========================================================================
    # UPLOAD MATERIAL + ICON IMAGES TO STORAGE
    # ========================================================================
    # Both regular material and icon candidates need storage URLs — the icons
    # so the admin UI can display them, the regular ones for embeddings later.
    images_to_upload = regular_material_images + icon_candidates
    logger.info(
        f"   📤 UPLOAD STAGE: Uploading {len(images_to_upload)} images "
        f"({len(regular_material_images)} regular + {len(icon_candidates)} icons)..."
    )
    uploaded_all = await image_service.upload_images_to_storage(
        material_images=images_to_upload,
        document_id=document_id
    )
    logger.info(f"      Successfully uploaded: {len(uploaded_all)}/{len(images_to_upload)}")

    # The uploader returns the same dicts with `storage_url` populated. We
    # need to re-split into regular vs icons by id() so we know which ones
    # to send through which pipeline.
    icon_object_ids = {id(img) for img in icon_candidates}
    uploaded_regular = [img for img in uploaded_all if id(img) not in icon_object_ids]
    uploaded_icons = [img for img in uploaded_all if id(img) in icon_object_ids]

    # 🔍 BBOX TRACE: Log bbox after upload
    logger.info(f"   🔍 [BBOX TRACE] After upload - checking {len(uploaded_regular)} regular uploaded images:")
    for i, img in enumerate(uploaded_regular[:5]):  # Log first 5
        bbox = img.get('bbox')
        bbox_len = len(bbox) if isinstance(bbox, (list, tuple)) else 'N/A'
        logger.info(f"      Image {i}: {img.get('filename')}, bbox_len={bbox_len}, bbox={bbox[:4] if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else bbox}, id={id(img)}")

    # ========================================================================
    # SAVE TO DATABASE AND GENERATE CLIP EMBEDDINGS
    # ========================================================================
    # Get material_category from config (passed from upload settings)
    material_category = config.get('material_category')
    logger.info(f"   💾 DATABASE STAGE: Saving metadata and generating CLIP embeddings...")
    logger.info(f"      Material category: {material_category or 'not specified'}")
    if uploaded_icons:
        logger.info(f"      Icon candidates: {len(uploaded_icons)} → OCR + Claude path")
    save_result = await image_service.save_images_and_generate_clips(
        material_images=uploaded_regular,
        document_id=document_id,
        workspace_id=workspace_id,
        material_category=material_category,
        job_id=job_id,
        tracker=tracker,
        progress_label=f"Stage 3: Processing images for {product.name}",
        icon_candidates=uploaded_icons,
    )

    images_processed = save_result.get('images_saved', 0)
    clip_embeddings = save_result.get('clip_embeddings_generated', 0)
    failed_images = save_result.get('failed_images', [])
    vector_stats = save_result.get('vector_stats', {})

    logger.info(f"   📊 DATABASE RESULTS:")
    logger.info(f"      Images saved to DB: {images_processed}/{len(uploaded_all)}")
    logger.info(f"      CLIP embeddings generated: {clip_embeddings}/{images_processed}")
    if vector_stats:
        logger.info(
            f"      Vectors per type: visual={vector_stats.get('visual_slig', 0)}, "
            f"color={vector_stats.get('color_slig', 0)}, "
            f"texture={vector_stats.get('texture_slig', 0)}, "
            f"style={vector_stats.get('style_slig', 0)}, "
            f"material={vector_stats.get('material_slig', 0)}, "
            f"understanding={vector_stats.get('understanding', 0)}"
        )
        logger.info(
            f"      Vision analysis: qwen={vector_stats.get('vision_analysis_qwen', 0)}, "
            f"claude_fallback={vector_stats.get('vision_analysis_claude_fallback', 0)}, "
            f"failed={vector_stats.get('vision_analysis_failed', 0)}"
        )
        if vector_stats.get('icon_candidates_processed', 0) > 0:
            logger.info(
                f"      Icons: extracted={vector_stats.get('icon_metadata_extracted', 0)}/"
                f"{vector_stats.get('icon_candidates_processed', 0)} "
                f"(failed: {vector_stats.get('icon_extraction_failed', 0)})"
            )

    if failed_images:
        logger.warning(f"      ⚠️ Failed to save {len(failed_images)} images")

    logger.info(f"   ✅ STAGE 3 COMPLETE for {product.name}")
    logger.info(f"      Total extracted: {total_images}")
    logger.info(f"      Regular material images: {len(regular_material_images)}")
    logger.info(f"      Icon candidates: {len(icon_candidates)}")
    logger.info(f"      Successfully processed: {images_processed}")
    logger.info(f"      CLIP embeddings: {clip_embeddings}")

    return {
        'images_processed': images_processed,
        'clip_embeddings_generated': clip_embeddings,
        'images_material': len(regular_material_images),
        'images_icon_candidates': len(icon_candidates),
        'images_non_material': non_material_count,
        'vector_stats': vector_stats,
        'failed_images': failed_images,
    }


async def process_catalog_wide_icons(
    file_content: bytes,
    document_id: str,
    workspace_id: str,
    job_id: str,
    catalog: Any,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Catalog-wide icon extraction pre-pass.

    Scans the PDF's SUPPLEMENTARY pages — i.e. pages not assigned to any product
    during Stage 0 discovery — for spec icon strips and routes them through the
    same OCR + Claude icon extraction pipeline as per-product icons.

    This exists because ceramic catalogs commonly put shared legend/iconography
    pages (R9/R10/R11 slip ratings, PEI wear classes, fire ratings, shade
    variation V1-V4) at the start or end of the book, not on each product page.
    Without this pass, those icons are never OCR'd because per-product Stage 3
    only looks at the product's own pages.

    Per-product rollup is unchanged — the icon rows land in `document_images`
    with `document_id` set, and `_merge_icon_metadata_into_product` in Stage 4
    already walks all images for the document (not just a product's associated
    images), so every product in the catalog gets those catalog-wide spec
    defaults merged into `product.metadata` automatically.

    Args:
        file_content: PDF file bytes.
        document_id: Document ID for the run.
        workspace_id: Workspace ID.
        job_id: Job ID (logged, used for AI cost attribution).
        catalog: The ProductCatalog from Stage 0 discovery. Must have
                 `supplementary_pages` populated. If empty, this function
                 is a no-op.
        logger: Logger.

    Returns:
        Stats dict:
            {
                'supplementary_pages_scanned': int,
                'images_extracted': int,
                'icon_candidates_found': int,
                'icons_processed': int,
                'icon_metadata_extracted': int,  # icons that yielded >= 1 spec item
                'icon_extraction_failed': int,
            }
        All zeros when there are no supplementary pages or no icon candidates.
    """
    from app.services.images.image_processing_service import ImageProcessingService

    supplementary_pages_0based = list(getattr(catalog, 'supplementary_pages', None) or [])

    stats = {
        'supplementary_pages_scanned': 0,
        'images_extracted': 0,
        'icon_candidates_found': 0,
        'icons_processed': 0,
        'icon_metadata_extracted': 0,
        'icon_extraction_failed': 0,
    }

    if not supplementary_pages_0based:
        logger.info("🔖 [catalog-icons] No supplementary pages — skipping catalog-wide icon pass")
        return stats

    logger.info(
        f"🔖 [catalog-icons] Scanning {len(supplementary_pages_0based)} supplementary "
        f"PDF pages for shared legend/iconography content..."
    )
    stats['supplementary_pages_scanned'] = len(supplementary_pages_0based)

    image_service = ImageProcessingService(workspace_id=workspace_id)
    pdf_processor = get_pdf_processor()

    # Build a physical-page map for assignment. Supplementary pages are by
    # definition not in any product's page range, but we still want to store
    # a sensible page_number on each icon image row (for downstream display
    # and for the per-image icon audit trail). We use the 1-based physical
    # page number that corresponds to the PDF page index.
    has_spread_layout = bool(getattr(catalog, 'has_spread_layout', False))
    physical_to_pdf_map: Dict[int, tuple] = getattr(catalog, 'physical_to_pdf_map', None) or {}

    # Invert physical_to_pdf_map so we can go pdf_idx -> [physical pages].
    pdf_idx_to_physical: Dict[int, List[int]] = {}
    if has_spread_layout and physical_to_pdf_map:
        for phys, (pdf_idx, _pos) in physical_to_pdf_map.items():
            pdf_idx_to_physical.setdefault(pdf_idx, []).append(phys)

    def _physical_for(pdf_idx: int) -> int:
        """Best-effort physical page number for a given PDF sheet index."""
        if pdf_idx in pdf_idx_to_physical:
            return min(pdf_idx_to_physical[pdf_idx])
        return pdf_idx + 1  # 1-based physical = 0-based pdf idx + 1

    # Extract images from every supplementary PDF page.
    extracted_images_list: List[Dict[str, Any]] = []
    for pdf_idx in supplementary_pages_0based:
        pdf_page_1based = pdf_idx + 1
        processing_options = {
            'extract_images': True,
            'extract_text': False,
            'extract_tables': False,
            'page_list': [pdf_page_1based],
            'extract_categories': ['products'],
            'job_id': job_id,
        }
        try:
            page_result = await pdf_processor.process_pdf_from_bytes(
                pdf_bytes=file_content,
                document_id=document_id,
                processing_options=processing_options,
            )
        except Exception as extract_err:
            logger.warning(
                f"   ⚠️ [catalog-icons] Failed to extract PDF page {pdf_page_1based}: {extract_err}"
            )
            continue

        if not page_result or not page_result.extracted_images:
            continue

        phys_page = _physical_for(pdf_idx)
        for img in page_result.extracted_images:
            img['page_number'] = phys_page
            img['catalog_wide_icon_source'] = True
            extracted_images_list.append(img)

    stats['images_extracted'] = len(extracted_images_list)
    if not extracted_images_list:
        logger.info("🔖 [catalog-icons] No images extracted from supplementary pages")
        return stats

    logger.info(
        f"🔖 [catalog-icons] Extracted {len(extracted_images_list)} images from "
        f"{len(supplementary_pages_0based)} supplementary page(s); classifying..."
    )

    # Classify so the icon filter has `ai_classification` on every image.
    try:
        material_images, non_material_images = await image_service.classify_images(
            extracted_images=extracted_images_list,
        )
    except Exception as cls_err:
        logger.warning(
            f"   ⚠️ [catalog-icons] Classification failed — skipping catalog-wide icon pass: {cls_err}"
        )
        return stats

    # Split into regular / icon / non-material using the existing rules.
    _regular, icon_candidates, _non_material = (
        image_service._split_material_and_icon_candidates(
            material_images=material_images,
            non_material_images=non_material_images,
        )
    )

    stats['icon_candidates_found'] = len(icon_candidates)
    if not icon_candidates:
        logger.info(
            "🔖 [catalog-icons] No icon-shaped candidates on supplementary pages — "
            "nothing to OCR"
        )
        return stats

    logger.info(
        f"🔖 [catalog-icons] {len(icon_candidates)} icon candidates found on shared "
        f"pages — uploading and running OCR + Claude..."
    )

    # Upload icon images to storage so they have the same URL contract as
    # per-product icons (admin UI display, audit trail).
    try:
        uploaded_icons = await image_service.upload_images_to_storage(
            material_images=icon_candidates,
            document_id=document_id,
        )
    except Exception as upload_err:
        logger.warning(
            f"   ⚠️ [catalog-icons] Failed to upload icon candidates, continuing with local paths: {upload_err}"
        )
        uploaded_icons = icon_candidates

    # Tag each icon's dict so the DB row carries a catalog-wide flag for the
    # admin UI and for any future filter that wants to exclude catalog icons
    # from per-product counts.
    for icon in uploaded_icons:
        meta = icon.get('metadata') or {}
        if not isinstance(meta, dict):
            meta = {}
        meta['catalog_wide_icon'] = True
        icon['metadata'] = meta

    # Route each icon through the existing _process_icon_candidate path, which
    # handles: save_single_image → OCR + Claude → merge icon_metadata back onto
    # the document_images row.
    total_icons = len(uploaded_icons)
    for idx, icon_img in enumerate(uploaded_icons, start=1):
        try:
            (_saved, _embedded, err, per_image_stats) = await image_service._process_icon_candidate(
                img_data=icon_img,
                document_id=document_id,
                workspace_id=workspace_id,
                idx=idx,
                total=total_icons,
            )
        except Exception as icon_err:
            logger.warning(
                f"   ⚠️ [catalog-icons] Icon {idx}/{total_icons} raised: {icon_err}"
            )
            stats['icon_extraction_failed'] += 1
            continue

        stats['icons_processed'] += 1
        if err:
            stats['icon_extraction_failed'] += 1
        elif per_image_stats.get('icon_metadata_count', 0) > 0:
            stats['icon_metadata_extracted'] += 1

    logger.info(
        f"🔖 [catalog-icons] Done: scanned={stats['supplementary_pages_scanned']} pages, "
        f"extracted={stats['images_extracted']} images, "
        f"icons={stats['icons_processed']}/{stats['icon_candidates_found']}, "
        f"spec-rich={stats['icon_metadata_extracted']}, "
        f"failed={stats['icon_extraction_failed']}"
    )
    return stats
