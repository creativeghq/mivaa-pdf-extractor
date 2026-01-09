"""
Stage 1: Focused Extraction with YOLO Layout Detection

This module handles page extraction and layout analysis for individual products
in the product-centric pipeline.

Features:
- Page mapping (catalog pages → PDF pages)
- YOLO layout detection (TEXT, IMAGE, TABLE, TITLE, CAPTION regions)
- Layout region storage in database
- Caption-to-image linking
"""

import logging
import tempfile
import os
from typing import Set, Any, Optional, Dict, List
from app.utils.page_converter import PageConverter


async def extract_product_pages(
    file_content: bytes,
    product: Any,
    document_id: str,
    job_id: str,
    logger: logging.Logger,
    total_pages: Optional[int] = None,
    pages_per_sheet: int = 1,
    enable_layout_detection: bool = True,
    product_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract pages and detect layout regions for a single product.

    This function:
    1. Maps catalog pages to physical PDF pages
    2. Detects layout regions using YOLO DocParser (if enabled)
    3. Stores layout regions in database
    4. Links captions to images

    Args:
        file_content: PDF file bytes
        product: Single product object from catalog
        document_id: Document identifier
        job_id: Job identifier
        logger: Logger instance
        total_pages: Optional total pages in PDF for validation
        pages_per_sheet: Number of catalog pages per PDF sheet (default: 1)
        enable_layout_detection: Enable YOLO layout detection (default: True)
        product_id: Database product ID (required for layout storage)

    Returns:
        Dict with:
        - product_pages: Set of physical PDF page indices (0-based)
        - layout_regions: List of detected layout regions (if enabled)
        - layout_stats: Statistics on detected regions
    """
    logger.info(f"📄 [STAGE 1] Extracting pages for product: {product.name}")
    logger.info(f"   Catalog page range: {product.page_range}")
    logger.info(f"   Layout detection: {'ENABLED' if enable_layout_detection else 'DISABLED'}")

    # ========================================================================
    # STEP 1: Map Catalog Pages to PDF Pages
    # ========================================================================
    converter = PageConverter(pages_per_sheet=pages_per_sheet, total_pdf_pages=total_pages)

    product_pages = set()
    if product.page_range:
        for p in product.page_range:
            if p > 0:
                try:
                    page = converter.from_catalog_page(p)
                    product_pages.add(page.array_index)
                except ValueError as e:
                    logger.warning(f"   ⚠️ Skipping out-of-bounds page: Catalog {p} -> {e}")
                    continue

    logger.info(f"   ✅ Mapped to PDF indices: {sorted(product_pages)} ({len(product_pages)} pages)")

    # ========================================================================
    # STEP 2: YOLO Layout Detection (if enabled)
    # ========================================================================
    layout_regions = []
    layout_stats = {}

    if enable_layout_detection and product_pages:
        logger.info(f"   🎯 Running YOLO layout detection on {len(product_pages)} pages...")

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
                    for page_idx in sorted(product_pages):
                        # YOLO uses 0-based page numbers
                        logger.info(f"      Detecting regions on page {page_idx}...")
                        result = await detector.detect_layout_regions(tmp_pdf_path, page_idx)

                        if result and result.regions:
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

    # Return comprehensive result
    return {
        'product_pages': product_pages,
        'layout_regions': layout_regions,
        'layout_stats': layout_stats
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
            region_data.append({
                'product_id': product_id,
                'page_number': region.bbox.page,
                'region_type': region.type,
                'bbox_x': region.bbox.x,
                'bbox_y': region.bbox.y,
                'bbox_width': region.bbox.width,
                'bbox_height': region.bbox.height,
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
