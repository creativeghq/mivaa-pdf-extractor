"""
Stage 2: Text Chunking

This module handles text chunking for individual products in the product-centric pipeline.
Includes metadata-first chunking, context enrichment, and type classification.

IMPORTANT: This module uses PHYSICAL PAGE NUMBERS (1-based) throughout.
Physical pages are what users see in catalogs.
"""

import logging
import fitz
from typing import Dict, Any, Set, List, Optional
from app.utils.pdf_to_images import analyze_pdf_layout, get_physical_page_text


async def process_product_chunking(
    file_content: bytes,
    document_id: str,
    workspace_id: str,
    job_id: str,
    product: Any,
    physical_pages: List[int],
    catalog: Any,
    config: Dict[str, Any],
    supabase: Any,
    logger: logging.Logger,
    product_id: Optional[str] = None,
    temp_pdf_path: Optional[str] = None,
    layout_regions: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Create text chunks for a single product (product-centric pipeline).

    Uses PHYSICAL PAGE NUMBERS (1-based) throughout.

    Pipeline:
    1. Metadata-First: Exclude product metadata pages from chunking
    2. Extract per-page text for each chunkable page
    3. Chunk + embed via RAGService.index_pdf_content (with layout regions if available)
    4. Classify chunk types + attach structured metadata

    Returns:
        Dictionary with chunks_created count and processing stats.
    """
    from app.services.search.rag_service import RAGService
    from app.services.chunking.chunk_type_classification_service import ChunkTypeClassificationService

    logger.info(f"📝 Creating chunks for product: {product.name}")
    logger.info(f"   Physical pages (1-based): {sorted(physical_pages)}")

    # We're already scoped to one product here, so every page passed in IS
    # this product's content. The legacy "metadata-first" filter was designed
    # for document-wide chunking and would silently exclude the product's own
    # pages when invoked per-product.
    chunkable_pages = set(physical_pages)

    # ========================================================================
    # STEP 1: Load Layout Regions (for layout-aware chunking)
    # ========================================================================
    layout_regions_by_page = {}

    # Priority 1 (preferred): Stage 1.5 document-level cache.
    # Stage 1.5 runs YOLO + bbox-text merge once per page upfront and
    # persists merged regions (with `text_content` populated) to
    # `document_layout_analysis`. Reading from that cache gives the
    # chunker text-aware regions without any per-product YOLO/Chandra
    # work. See app/api/pdf_processing/stage_1_layout_precompute.py.
    if config.get('enable_layout_aware_chunking', True):
        try:
            from app.api.pdf_processing.stage_1_layout_precompute import (
                get_layout_from_document_cache,
            )
            cached = await get_layout_from_document_cache(
                document_id=document_id,
                physical_pages=physical_pages,
                supabase=supabase,
                logger=logger,
            )
            if cached:
                layout_regions_by_page = cached
        except Exception as cache_err:
            logger.debug(f"   document layout cache read failed (non-fatal): {cache_err}")

    # Priority 2: Caller-provided in-memory regions (legacy paths that
    # ran YOLO themselves). Only used if Priority 1 produced nothing.
    if not layout_regions_by_page and layout_regions:
        logger.info(f"   📐 Using {len(layout_regions)} caller-provided layout regions")
        for region in layout_regions:
            page_num = region.bbox.page
            if page_num not in layout_regions_by_page:
                layout_regions_by_page[page_num] = []
            # Convert to dict if it's a Pydantic model
            region_dict = region.dict() if hasattr(region, 'dict') else region
            layout_regions_by_page[page_num].append(region_dict)

    # Priority 3: Per-product DB regions (older code path that wrote
    # `product_layout_regions` rows during Stage 4 of a prior job).
    if not layout_regions_by_page and product_id and config.get('enable_layout_aware_chunking', True):
        logger.info(f"   📐 Loading layout regions from product_layout_regions cache...")
        layout_regions_by_page = await get_layout_regions(
            product_id=product_id,
            supabase=supabase,
            logger=logger
        )

    # ========================================================================
    # STEP 2: Create Chunks (existing logic)
    # ========================================================================
    rag_service = RAGService(config={
        'chunk_size': config.get('chunk_size', 1000),
        'chunk_overlap': config.get('chunk_overlap', 200)
    })

    # Extract per-page text for chunkable pages as a page_chunks list
    # [{metadata:{page: 0-indexed}, text: ...}] so chunk_pages() can preserve
    # page_number per chunk and match layout regions correctly.
    page_chunks_data: List[Dict[str, Any]] = []
    chunkable_pages_sorted = sorted(chunkable_pages)
    logger.info(f"   📄 Extracting text from {len(chunkable_pages_sorted)} chunkable physical pages: {chunkable_pages_sorted}")
    try:
        import tempfile
        import os

        used_temp_path = temp_pdf_path
        created_temp = False

        if not used_temp_path or not os.path.exists(used_temp_path):
            logger.info("      ⚠️ No temp_pdf_path provided, creating temporary copy for extraction...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                used_temp_path = tmp_file.name
                created_temp = True
        else:
            logger.info(f"      ♻️ Reusing existing temp PDF: {used_temp_path}")

        try:
            doc = fitz.open(used_temp_path)
            if catalog and hasattr(catalog, 'has_spread_layout'):
                layout_analysis = catalog
                logger.info("      ♻️ Reusing layout analysis from catalog")
            else:
                logger.info("      📐 No layout in catalog, performing fresh layout analysis...")
                layout_analysis = analyze_pdf_layout(used_temp_path)

            total_chars = 0
            for phys_page in chunkable_pages_sorted:
                page_text, _ = get_physical_page_text(doc, layout_analysis, phys_page)
                if not page_text or not page_text.strip():
                    continue
                # chunk_pages() expects 0-indexed page metadata; it converts back
                # to 1-based when stamping chunk metadata.
                page_chunks_data.append({
                    "metadata": {"page": phys_page - 1},
                    "text": page_text,
                })
                total_chars += len(page_text)

            doc.close()
            logger.info(f"   ✅ Extracted {total_chars} characters across {len(page_chunks_data)} non-empty pages")
        finally:
            if created_temp and used_temp_path and os.path.exists(used_temp_path):
                os.unlink(used_temp_path)
    except Exception as e:
        logger.error(f"   ❌ Failed to extract product text: {e}")
        page_chunks_data = []

    if not page_chunks_data:
        # Promoted from WARNING to ERROR (2026-05-01): silent zero-chunk
        # outcomes are how products end up with no search index entries
        # without anyone noticing for weeks. If this fires, something is
        # actually wrong upstream — usually a bad physical_page_upper_bound,
        # a missing/corrupt PDF, or a YOLO/Chandra failure that nuked the
        # text extraction. Don't whisper it; shout.
        try:
            requested_pages = list(getattr(product, 'page_range', None) or [])
        except Exception:
            requested_pages = []
        logger.error(
            f"   ❌ [STAGE 2] {product.name}: 0 chunks created — page_text "
            f"extraction returned empty for ALL {len(requested_pages)} requested "
            f"physical pages: {requested_pages}. Likely upstream causes: "
            f"(a) physical_page_upper_bound was too small and Stage 1 dropped "
            f"every page as out-of-bounds; (b) the document's text layer is "
            f"empty (scanned PDF without OCR fallback); (c) Stage 1.5 cache "
            f"miss forced an extraction that timed out. Check Stage 1's "
            f"validated-physical-pages log for this product."
        )
        return {
            'chunks_created': 0,
            'chunk_ids': [],
            'embeddings_generated': 0,
            'pages_chunked': 0,
            'pages_excluded': 0,
        }

    # Create chunks with product-specific metadata
    chunk_result = await rag_service.index_pdf_content(
        pdf_content=file_content,
        document_id=document_id,
        metadata={
            'filename': f"{product.name}_chunks",
            'title': product.name,
            'page_count': len(chunkable_pages),
            'product_pages': sorted(chunkable_pages),
            'chunk_size': config.get('chunk_size', 1000),
            'chunk_overlap': config.get('chunk_overlap', 200),
            'workspace_id': workspace_id,
            'job_id': job_id,
            'product_id': product_id,
            'product_name': product.name
        },
        catalog=catalog,
        page_chunks=page_chunks_data,
        layout_regions_by_page=layout_regions_by_page
    )

    chunks_created = chunk_result.get('chunks_created', 0)
    chunk_ids = chunk_result.get('chunk_ids', [])
    logger.info(f"   ✅ Created {chunks_created} chunks for {product.name} (already stored in DB)")

    # STEP 3: Classify chunks + attach structured metadata
    # Fetches stored chunks, runs pattern-based classification (fast, no API),
    # falls back to Qwen for ambiguous cases, then bulk-updates the metadata.
    if chunk_ids and config.get('enable_chunk_classification', True):
        try:
            classification_service = ChunkTypeClassificationService()
            await _classify_and_update_chunks(
                chunk_ids=chunk_ids,
                classification_service=classification_service,
                supabase=supabase,
                logger=logger,
            )
        except Exception as e:
            # Classification is non-fatal — chunks remain usable without it.
            logger.warning(f"   ⚠️ Chunk classification failed for {product.name}: {e}")

    return {
        'chunks_created': chunks_created,
        'chunk_ids': chunk_ids,
        'embeddings_generated': chunk_result.get('embeddings_generated', 0),
        'pages_chunked': len(chunkable_pages),
        'pages_excluded': 0,
    }


async def _classify_and_update_chunks(
    chunk_ids: List[str],
    classification_service: Any,
    supabase: Any,
    logger: logging.Logger,
) -> None:
    """Classify stored chunks and merge results into their metadata.

    Runs in batches, merges `chunk_type` and structured metadata into each
    chunk's existing metadata JSON, and writes back. Never modifies `content`.
    """
    if not chunk_ids:
        return

    # Fetch the chunks we just stored so we can classify their content.
    result = supabase.client.table('document_chunks') \
        .select('id, content, metadata') \
        .in_('id', chunk_ids) \
        .execute()
    rows = result.data or []
    if not rows:
        logger.debug("   Classification: no chunks fetched — nothing to classify")
        return

    logger.info(f"   🏷️ Classifying {len(rows)} chunks...")
    classifications = await classification_service.classify_chunks_batch(
        [{'id': r['id'], 'content': r['content']} for r in rows]
    )

    updates_made = 0
    for row, cls in zip(rows, classifications):
        try:
            existing_meta = row.get('metadata') or {}
            existing_meta['chunk_type'] = cls.chunk_type.value
            existing_meta['classification_confidence'] = cls.confidence
            if cls.metadata:
                existing_meta['structured_metadata'] = cls.metadata

            supabase.client.table('document_chunks') \
                .update({'metadata': existing_meta}) \
                .eq('id', row['id']) \
                .execute()
            updates_made += 1
        except Exception as e:
            logger.debug(f"      Failed to update classification for chunk {row['id']}: {e}")

    logger.info(f"   ✅ Classified {updates_made}/{len(rows)} chunks")


async def get_layout_regions(
    product_id: str,
    supabase: Any,
    logger: logging.Logger
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Fetch layout regions from database for a product.

    Args:
        product_id: Database product ID
        supabase: Supabase client
        logger: Logger instance

    Returns:
        Dict mapping page_number -> list of regions sorted by reading_order
        Example: {1: [region1, region2], 2: [region3, region4]}
    """
    try:
        # Fetch all layout regions for this product
        result = supabase.client.table('product_layout_regions')\
            .select('*')\
            .eq('product_id', product_id)\
            .order('page_number')\
            .order('reading_order')\
            .execute()

        if not result.data:
            logger.info(f"   No layout regions found for product {product_id}")
            return {}

        # Group regions by page number
        regions_by_page = {}
        for region in result.data:
            page_num = region['page_number']
            if page_num not in regions_by_page:
                regions_by_page[page_num] = []
            regions_by_page[page_num].append(region)

        logger.info(f"   Loaded {len(result.data)} layout regions across {len(regions_by_page)} pages")
        return regions_by_page

    except Exception as e:
        logger.error(f"   ❌ Failed to fetch layout regions: {e}")
        return {}

