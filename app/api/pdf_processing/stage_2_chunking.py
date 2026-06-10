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
    chunking_strategy = "no_layout_regions"  # default — plain text chunking

    # Priority 1 (preferred): Stage 1.5 document-level cache.
    # Stage 1.5 runs YOLO + bbox-text merge once per page upfront and
    # persists merged regions (with `text_content` populated) to
    # `document_layout_analysis`. Reading from that cache gives the
    # chunker text-aware regions without any per-product YOLO/Chandra
    # work. See app/api/pdf_processing/stage_1_layout_precompute.py.
    if config.get('enable_layout_aware_chunking', True):
        try:
            from app.api.pdf_processing.stage_1_layout_precompute import (
                get_layout_from_document_cache_with_status,
            )
            # Use the _with_status variant so we can distinguish "no row"
            # from "row exists but cache_status=ocr_failed/page_failed"
            # (which Stage 1.5's resume-skip will retry on the next job run).
            # The plain get_layout_from_document_cache silently drops failed
            # rows — meaning the chunker fell back to text-based chunking
            # without anyone knowing layout was actually broken for the page.
            cached_with_status = await get_layout_from_document_cache_with_status(
                document_id=document_id,
                physical_pages=physical_pages,
                supabase=supabase,
                logger=logger,
            )
            cached = {p: v['regions'] for p, v in cached_with_status.items() if v['regions']}
            # Surface failed-page telemetry: count rows with failure cache_status
            # so they appear in pipeline_strategy_metrics (operators can see
            # how often Stage 1.5 broke on this document, and the chunker's
            # fallback rate).
            _failed_statuses = {'ocr_failed', 'page_failed'}
            _failed_pages = [
                p for p, v in cached_with_status.items()
                if v.get('cache_status') in _failed_statuses
            ]
            if _failed_pages:
                logger.warning(
                    f"   ⚠️ Stage 1.5 cache has {len(_failed_pages)} page(s) marked "
                    f"failed ({', '.join(map(str, _failed_pages))}); chunker falling "
                    f"back to text-based for these pages until Stage 1.5 retries"
                )
                # Emit a stage_history event so the admin UI shows the
                # fallback explicitly. Without this, "Stage 1.5 incomplete,
                # Stage 2 falling back to text" was log-only — operators had
                # no visibility from /full-status.
                try:
                    from datetime import datetime as _dt
                    supabase.client.rpc(
                        'append_stage_history',
                        {
                            'p_job_id': job_id,
                            'p_event': {
                                'stage': 'stage_2_chunking',
                                'status': 'fallback_text_based',
                                'data': {
                                    'product_id': product_id,
                                    'product_name': getattr(product, 'name', None),
                                    'stage_1_5_failed_pages': _failed_pages,
                                    'reason': 'stage_1_5_failed_pages — fell back to text-based chunking',
                                },
                                'occurred_at': _dt.utcnow().isoformat(),
                            },
                        }
                    ).execute()
                except Exception as _hist_err:
                    logger.debug(f"   append_stage_history (fallback) failed: {_hist_err}")
            if cached:
                layout_regions_by_page = cached
                chunking_strategy = "stage_1_5_cache"
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
        if layout_regions_by_page:
            chunking_strategy = "caller_provided_regions"

    # Priority 3: Per-product DB regions (older code path that wrote
    # `product_layout_regions` rows during Stage 4 of a prior job).
    if not layout_regions_by_page and product_id and config.get('enable_layout_aware_chunking', True):
        logger.info(f"   📐 Loading layout regions from product_layout_regions cache...")
        layout_regions_by_page = await get_layout_regions(
            product_id=product_id,
            supabase=supabase,
            logger=logger
        )
        if layout_regions_by_page:
            chunking_strategy = "product_layout_regions_cache"

    # Telemetry: record which chunking strategy actually fired for this product.
    # `pipeline_strategy_metrics` is the per-stage distribution log the 2026-05-01
    # audit added — best-effort write, never blocks chunking on a metrics row.
    try:
        # Compute failed-page count from the cache_status pass above so the
        # metric captures Stage 1.5 health for this product.
        _failed_n = 0
        try:
            _failed_n = len(_failed_pages)  # defined when cache read ran
        except NameError:
            _failed_n = 0
        supabase.client.table('pipeline_strategy_metrics').insert({
            'job_id': job_id,
            'document_id': document_id,
            'product_id': product_id,
            'page_number': None,  # product-level metric, not page-level
            'metric_kind': 'chunking_strategy',
            'metric_value': chunking_strategy,
            'notes': {
                'pages_with_regions': len(layout_regions_by_page),
                'total_pages': len(physical_pages),
                'stage_1_5_failed_pages': _failed_n,
                'product_name': getattr(product, 'name', None),
            },
        }).execute()
    except Exception as metrics_err:
        logger.debug(f"   pipeline_strategy_metrics insert failed (non-fatal): {metrics_err}")

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
            failed_pages: list = []
            for phys_page in chunkable_pages_sorted:
                # Isolate per-page failures — previously a single corrupt/
                # malformed page threw to the outer except, which zeroed
                # page_chunks_data and silently lost the product's ENTIRE
                # text even though every other page extracted fine.
                try:
                    page_text, _ = get_physical_page_text(doc, layout_analysis, phys_page)
                except Exception as page_err:
                    failed_pages.append(phys_page)
                    logger.warning(
                        f"   ⚠️ Text extraction failed for physical page {phys_page} "
                        f"— skipping page, continuing with the rest: {page_err}"
                    )
                    continue
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
            if failed_pages:
                logger.error(
                    f"   ❌ {len(failed_pages)} page(s) failed text extraction and were "
                    f"skipped: {failed_pages}"
                )
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
            # Mark every chunk as failed so re-classification jobs can pick
            # them up. Without this, the failure is invisible — chunks stay
            # at `chunk_type_status='pending'` indistinguishable from
            # not-yet-classified.
            logger.warning(f"   ⚠️ Chunk classification failed for {product.name}: {e}")
            try:
                supabase.client.table('document_chunks') \
                    .update({'chunk_type_status': 'failed'}) \
                    .in_('id', chunk_ids) \
                    .execute()
            except Exception as _mark_err:
                logger.debug(f"      failed to mark chunks as failed: {_mark_err}")

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

            # Write the AI verdict to BOTH the canonical column and the metadata
            # jsonb. Previously we only wrote metadata, leaving `chunk_type` column
            # stuck at the default 'unclassified' even when the classifier produced
            # a real verdict — admin UI / search filters / RPC indexes that key off
            # the column never saw it. Audit incident: job acff9ebb 2026-05-03,
            # 16/16 chunks had column='unclassified' while metadata.chunk_type was
            # correct on every row.
            supabase.client.table('document_chunks') \
                .update({
                    'metadata': existing_meta,
                    'chunk_type': cls.chunk_type.value,
                    'chunk_type_confidence': cls.confidence,
                    'chunk_type_metadata': cls.metadata or None,
                    # 2026-05-23: flip status to 'classified' so operators can
                    # distinguish "Sonnet returned 'unclassified' as the verdict"
                    # from "Sonnet crashed mid-batch and the column is at default".
                    # The failure branch below stamps 'failed'.
                    'chunk_type_status': 'classified',
                }) \
                .eq('id', row['id']) \
                .execute()
            updates_made += 1
        except Exception as e:
            logger.debug(f"      Failed to update classification for chunk {row['id']}: {e}")
            # Mark this chunk's classification as failed so it's visible to
            # backfill / re-classification jobs (not just buried in logs).
            try:
                supabase.client.table('document_chunks') \
                    .update({'chunk_type_status': 'failed'}) \
                    .eq('id', row['id']) \
                    .execute()
            except Exception:
                pass

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
        # Empty dict and DB-query-failure used to be indistinguishable to
        # the caller (both fell through to text-based chunking); the only
        # signal was an ERROR log that often got missed. Capture to Sentry
        # so operators can react to layout-aware chunking degradation.
        logger.error(
            f"   ❌ Failed to fetch layout regions for product {product_id} "
            f"(falling through to text-based chunking): {e}"
        )
        try:
            import sentry_sdk
            sentry_sdk.capture_exception(e)
        except Exception:
            pass
        return {}

