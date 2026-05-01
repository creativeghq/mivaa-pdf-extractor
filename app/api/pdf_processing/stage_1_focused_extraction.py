"""
Stage 1: Focused Extraction with YOLO Layout Detection

This module handles page extraction and layout analysis for individual products
in the product-centric pipeline.

IMPORTANT: This module uses PHYSICAL PAGE NUMBERS (1-based) throughout.
Physical pages are what users see in catalogs. PDF sheet indices are only
used internally when accessing PyMuPDF - never exposed to other stages.

Features:
- Page validation (physical pages)
- YOLO layout detection (TEXT, IMAGE, TABLE, TITLE, CAPTION regions)
- Layout region storage in database
- Caption-to-image linking
"""

import logging
import tempfile
import os
import fitz
from typing import Set, Any, Optional, Dict, List
from app.utils.pdf_to_images import get_physical_page_text
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
    total_pages: Optional[int] = None,  # DEPRECATED — kept for back-compat with old callers
) -> Dict[str, Any]:
    """
    Extract pages and detect layout regions for a single product.

    IMPORTANT: Returns PHYSICAL PAGE NUMBERS (1-based) as the primary output.
    PDF sheet indices are only used internally for PyMuPDF access.

    This function:
    1. Validates PHYSICAL page numbers
    2. Handles spread layouts internally (for YOLO detection)
    3. Detects layout regions using YOLO DocParser (if enabled)
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
        enable_layout_detection: Enable YOLO layout detection (default: True)
        product_id: Database product ID (required for layout storage)
        catalog: Optional catalog with spread layout info (has_spread_layout, physical_to_pdf_map)
        total_pages: DEPRECATED alias for `physical_page_upper_bound`. Kept so
            existing callers don't break, but new code should use
            `physical_page_upper_bound` to make the contract explicit.

    Returns:
        Dict with:
        - physical_pages: List of physical page numbers (1-based) - PRIMARY OUTPUT
        - layout_regions: List of detected layout regions (if enabled)
        - layout_stats: Statistics on detected regions
        - has_spread_layout: Whether document uses spread layout
        - physical_to_pdf_map: Mapping for internal PDF access (passed through from catalog)
    """
    # Resolve the bound — prefer the new explicit name, fall back to legacy.
    if physical_page_upper_bound is None:
        physical_page_upper_bound = total_pages

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
    # STEP 2: YOLO Layout Detection (if enabled)
    # ========================================================================
    layout_regions = []
    layout_stats = {}

    if enable_layout_detection and physical_pages:
        logger.info(f"   🎯 Running YOLO layout detection on {len(physical_pages)} physical pages...")

        # Cache lookup: pdf_worker.execute_pdf_extraction_job persists merged
        # YOLO+text regions to `document_layout_analysis` per page during
        # markdown extraction. Reuse those rather than re-running YOLO when
        # they exist - saves an HF call per product page and guarantees the
        # same regions feed both stage_1 and the layout-aware chunker.
        cached_regions = await _load_cached_layout_regions(
            document_id=document_id,
            physical_pages=physical_pages,
            has_spread_layout=has_spread_layout,
            physical_to_pdf_map=physical_to_pdf_map,
            logger=logger,
        )

        try:
            from app.services.pdf.yolo_layout_detector import YoloLayoutDetector
            from app.config import get_settings

            settings = get_settings()

            # Check if YOLO is enabled
            if not settings.yolo_enabled:
                logger.info("   ⚠️ YOLO disabled in settings, skipping layout detection")
            else:
                # Initialize YOLO detector
                yolo_config = settings.get_yolo_config()
                detector = YoloLayoutDetector(config=yolo_config)

                # Save PDF to temp file for YOLO processing
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(file_content)
                    tmp_pdf_path = tmp_file.name

                try:
                    # Detect layout regions for each product page
                    all_regions = []
                    for physical_page in sorted(physical_pages):
                        # Convert physical page to PDF page index and position
                        if has_spread_layout and physical_page in physical_to_pdf_map:
                            pdf_page_idx, position = physical_to_pdf_map[physical_page]
                        else:
                            pdf_page_idx = physical_page - 1
                            position = 'single'

                        # Check cache before invoking YOLO - if pdf_worker
                        # already persisted regions for this page, reuse them
                        # so we get identical regions to what the chunker has
                        # already seen and we save an HF call per page.
                        cached_for_page = cached_regions.get(physical_page) if cached_regions else None
                        if cached_for_page:
                            logger.info(
                                f"      ♻️ Using {len(cached_for_page)} cached regions for "
                                f"physical page {physical_page} (no YOLO re-run)"
                            )

                            class _CachedResult:
                                regions = cached_for_page
                                detection_time_ms = 0
                            result = _CachedResult()
                        else:
                            logger.info(
                                f"      Detecting regions on physical page {physical_page} "
                                f"(PDF sheet {pdf_page_idx}, position {position})..."
                            )
                            # YOLO always detects on the full sheet
                            result = await detector.detect_layout_regions(tmp_pdf_path, pdf_page_idx)

                        if result and result.regions:
                            # Filter and clip regions if it's a spread
                            if position in ['left', 'right']:
                                # Use catalog's pre-analyzed sheet width (avoids redundant PDF opening)
                                sheet_width = 0
                                if catalog and hasattr(catalog, 'pdf_page_widths') and catalog.pdf_page_widths:
                                    sheet_width = catalog.pdf_page_widths.get(pdf_page_idx, 0)
                                
                                # Fallback if catalog missing width (unlikely)
                                if sheet_width == 0:
                                    doc = fitz.open(tmp_pdf_path)
                                    sheet = doc[pdf_page_idx]
                                    sheet_width = sheet.rect.width
                                    doc.close()
                                    logger.debug(f"      ⚠️ Sheet width for page {pdf_page_idx} missing from catalog, opened PDF")
                                
                                mid_x = sheet_width / 2

                                filtered_regions = []
                                for region in result.regions:
                                    # Check spatial position relative to center
                                    region_mid_x = region.bbox.x + (region.bbox.width / 2)
                                    is_left = region_mid_x < mid_x
                                    
                                    # SCENE DETECTION: Spans across the middle
                                    spans_middle = (region.bbox.x < mid_x) and (region.bbox.x + region.bbox.width > mid_x)
                                    is_scene = spans_middle and (region.bbox.width > sheet.rect.width * 0.4)
                                    
                                    if is_scene:
                                        # Keep scene regions on both physical pages (or just the first one)
                                        # For now, we assign it to the physical_page currently being processed
                                        region.bbox.page = physical_page
                                        region.is_scene = True # Flag for downstream
                                        filtered_regions.append(region)
                                        logger.info(f"      🎞️ Scene detected: {region.label} bridging spread")
                                    elif (position == 'left' and is_left) or (position == 'right' and not is_left):
                                        # Normal side-specific region
                                        region.bbox.page = physical_page
                                        filtered_regions.append(region)
                                
                                logger.info(f"      ✅ Found {len(filtered_regions)} regions on {position} side")
                                all_regions.extend(filtered_regions)
                            else:
                                # Normal single page or full spread image
                                for region in result.regions:
                                    region.bbox.page = physical_page
                                all_regions.extend(result.regions)
                                logger.info(
                                    f"      ✅ Found {len(result.regions)} regions "
                                    f"({result.detection_time_ms}ms)"
                                )

                    layout_regions = all_regions

                    # Calculate statistics
                    layout_stats = _calculate_layout_stats(layout_regions)
                    logger.info(f"   ✅ Layout detection complete: {layout_stats}")

                    # Store layout regions in database (if product_id provided)
                    if product_id and layout_regions:
                        await _store_layout_regions(
                            product_id=product_id,
                            regions=layout_regions,
                            logger=logger
                        )

                    # ========================================================================
                    # STEP 3: Table Extraction (if TABLE regions detected)
                    # ========================================================================
                    table_regions = [r for r in layout_regions if r.type == 'TABLE']
                    tables_extracted = 0

                    if table_regions and product_id:
                        logger.info(f"   📊 Extracting {len(table_regions)} tables...")
                        tables_extracted = await _extract_and_store_tables(
                            pdf_path=tmp_pdf_path,
                            product_id=product_id,
                            table_regions=table_regions,
                            logger=logger
                        )
                        logger.info(f"   ✅ Extracted {tables_extracted} tables")

                finally:
                    # Cleanup temp file
                    if os.path.exists(tmp_pdf_path):
                        os.unlink(tmp_pdf_path)

        except Exception as e:
            logger.error(f"   ❌ Layout detection failed: {e}")
            logger.info("   Continuing without layout detection...")

    # Return comprehensive result - PHYSICAL PAGES are the primary output
    return {
        'physical_pages': physical_pages,  # Physical page numbers (1-based) - PRIMARY OUTPUT
        'layout_regions': layout_regions,
        'layout_stats': layout_stats,
        # Pass through spread layout info for stages that need PDF access
        'has_spread_layout': has_spread_layout,
        'physical_to_pdf_map': physical_to_pdf_map
    }


def _calculate_layout_stats(regions: List[Any]) -> Dict[str, Any]:
    """
    Calculate statistics on detected layout regions.

    Args:
        regions: List of LayoutRegion objects

    Returns:
        Dict with region counts by type
    """
    stats = {
        'total_regions': len(regions),
        'by_type': {}
    }

    for region in regions:
        region_type = region.type
        stats['by_type'][region_type] = stats['by_type'].get(region_type, 0) + 1

    return stats


async def _store_layout_regions(
    product_id: str,
    regions: List[Any],
    logger: logging.Logger
) -> None:
    """
    Store layout regions in database.

    Args:
        product_id: Database product ID
        regions: List of LayoutRegion objects
        logger: Logger instance
    """
    try:
        from app.services.core.supabase_client import get_supabase_client

        supabase = get_supabase_client()

        # Prepare region data for insertion
        region_data = []
        for region in regions:
            # Normalize pixel coords to 0-1 using image_size from YOLO metadata
            img_size = region.metadata.get('image_size') if region.metadata else None
            if isinstance(img_size, (tuple, list)) and len(img_size) == 2:
                img_w = float(img_size[0]) or 1.0
                img_h = float(img_size[1]) or 1.0
            else:
                img_w = img_h = 1.0  # already normalized or unknown

            def _clamp01(v: float) -> float:
                return min(1.0, max(0.0, v))

            bbox_x = _clamp01(region.bbox.x / img_w) if img_w > 1.0 else _clamp01(region.bbox.x)
            bbox_y = _clamp01(region.bbox.y / img_h) if img_h > 1.0 else _clamp01(region.bbox.y)
            bbox_w = _clamp01(region.bbox.width / img_w) if img_w > 1.0 else _clamp01(region.bbox.width)
            bbox_h = _clamp01(region.bbox.height / img_h) if img_h > 1.0 else _clamp01(region.bbox.height)

            region_data.append({
                'product_id': product_id,
                'page_number': region.bbox.page,
                'region_type': region.type,
                'bbox_x': bbox_x,
                'bbox_y': bbox_y,
                'bbox_width': bbox_w,
                'bbox_height': bbox_h,
                'confidence': region.confidence,
                'reading_order': region.reading_order,
                'text_content': getattr(region, 'text_content', None),
                'metadata': {
                    'yolo_model': 'yolo-docparser',
                    'extraction_method': 'yolo_guided'
                }
            })

        # Batch insert
        if region_data:
            result = supabase.client.table('product_layout_regions').insert(region_data).execute()
            logger.info(f"   💾 Stored {len(region_data)} layout regions in database")

            # Update product layout stats
            await _update_product_layout_stats(product_id, logger)

    except Exception as e:
        logger.error(f"   ❌ Failed to store layout regions: {e}")


async def _update_product_layout_stats(product_id: str, logger: logging.Logger) -> None:
    """
    Update product layout statistics using database function.

    Args:
        product_id: Database product ID
        logger: Logger instance
    """
    try:
        from app.services.core.supabase_client import get_supabase_client

        supabase = get_supabase_client()

        # Call database function to update stats
        supabase.client.rpc('update_product_layout_stats', {'p_product_id': product_id}).execute()
        logger.info(f"   ✅ Updated product layout statistics")

    except Exception as e:
        logger.error(f"   ⚠️ Failed to update layout stats: {e}")


async def _extract_and_store_tables(
    pdf_path: str,
    product_id: str,
    table_regions: List[Any],
    logger: logging.Logger
) -> int:
    """
    Extract tables from TABLE regions and store in database.

    Args:
        pdf_path: Path to PDF file
        product_id: Database product ID
        table_regions: List of TABLE LayoutRegion objects from YOLO
        logger: Logger instance

    Returns:
        Number of tables successfully extracted and stored
    """
    try:
        from app.services.pdf.table_extraction import TableExtractor
        from app.services.core.supabase_client import get_supabase_client

        # Initialize table extractor
        extractor = TableExtractor()
        supabase = get_supabase_client()

        # Group table regions by page
        tables_by_page = {}
        for region in table_regions:
            page_num = region.page_number
            if page_num not in tables_by_page:
                tables_by_page[page_num] = []
            tables_by_page[page_num].append(region)

        # Extract tables page by page
        all_tables = []
        for page_num, regions in tables_by_page.items():
            logger.info(f"      Extracting {len(regions)} tables from page {page_num}...")

            # Extract tables using pdfplumber
            tables = extractor.extract_tables_from_page(
                pdf_path=pdf_path,
                page_number=page_num,
                table_regions=regions
            )

            all_tables.extend(tables)

        # Store tables in database
        if all_tables:
            stored_count = await extractor.store_tables_in_database(
                product_id=product_id,
                tables=all_tables,
                supabase_client=supabase
            )
            return stored_count

        return 0

    except Exception as e:
        logger.error(f"   ❌ Table extraction failed: {e}")
        return 0


async def _load_cached_layout_regions(
    document_id: str,
    physical_pages: List[int],
    has_spread_layout: bool,
    physical_to_pdf_map: Dict[int, Any],
    logger: logging.Logger,
) -> Dict[int, List[Any]]:
    """Load YOLO+text merged regions from `document_layout_analysis` cache.

    Returns a dict keyed by physical page number (1-based). Each value is
    a list of `LayoutRegion` objects rebuilt from the cached jsonb so the
    rest of stage_1 can treat them identically to fresh YOLO output.

    The cache is written by `pdf_processor._persist_document_layout`
    during PDF markdown extraction. We look it up here to avoid running
    YOLO twice for the same page.
    """
    from app.models.layout_models import BoundingBox, LayoutRegion

    if not document_id or not physical_pages:
        return {}

    # Cache stores PDF sheet page numbers, not physical pages, so we need
    # to translate. For non-spread documents, physical == pdf_idx + 1.
    page_lookup_map: Dict[int, int] = {}
    for pp in physical_pages:
        if has_spread_layout and pp in physical_to_pdf_map:
            pdf_idx, _position = physical_to_pdf_map[pp]
            page_lookup_map[pdf_idx] = pp
        else:
            page_lookup_map[pp - 1] = pp

    try:
        from app.services.core.supabase_client import get_supabase_client
        supabase = get_supabase_client()
        # `pdf_worker` writes 1-based page_number; layout regions use the
        # same 1-based scheme. Look up by PDF-sheet-1-based.
        response = supabase.client.table('document_layout_analysis').select(
            'page_number, layout_elements, processing_version'
        ).eq('document_id', document_id).execute()
    except Exception as exc:
        logger.debug(f"   ↺ Layout cache lookup skipped: {exc}")
        return {}

    rows = response.data or []
    cached: Dict[int, List[Any]] = {}
    for row in rows:
        if row.get('processing_version') != 'yolo+chandra-v2':
            # Only honour cache rows produced by the new pipeline; older
            # rows are markdown-analysis stubs from product_creation_service.
            continue
        page_1_based_in_cache = int(row['page_number'])
        # Cache uses 1-based sheet numbering - translate to physical page.
        pdf_idx = page_1_based_in_cache - 1
        physical_page = page_lookup_map.get(pdf_idx)
        if physical_page is None:
            continue
        elements = row.get('layout_elements') or []
        if not isinstance(elements, list):
            continue

        regions: List[LayoutRegion] = []
        for elem in elements:
            try:
                region_type = elem.get('region_type', 'TEXT')
                # UNCLASSIFIED is an internal merge-service marker for
                # orphan text fragments; downstream YOLO consumers only
                # know the canonical 5 types. Skip orphans here.
                if region_type not in ("TEXT", "IMAGE", "TABLE", "TITLE", "CAPTION"):
                    continue
                bbox_dict = elem.get('bbox') or {}
                width = float(bbox_dict.get('width', 1)) or 1.0
                height = float(bbox_dict.get('height', 1)) or 1.0
                regions.append(LayoutRegion(
                    type=region_type,
                    bbox=BoundingBox(
                        x=float(bbox_dict.get('x', 0)),
                        y=float(bbox_dict.get('y', 0)),
                        width=width,
                        height=height,
                        page=int(bbox_dict.get('page', physical_page)),
                    ),
                    confidence=float(elem.get('confidence') or 0.85),
                    text_content=elem.get('text_content'),
                    reading_order=elem.get('reading_order'),
                    metadata=elem.get('metadata') or {},
                ))
            except Exception as build_err:
                logger.debug(f"   ↺ Skipped malformed cached region: {build_err}")

        if regions:
            cached[physical_page] = regions

    if cached:
        logger.info(
            f"   ♻️ Loaded cached layout for {len(cached)} pages from "
            f"document_layout_analysis (skipping YOLO re-run)"
        )
    return cached
