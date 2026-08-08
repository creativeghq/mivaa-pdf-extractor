"""
Single Product Processor

Handles processing of a single product through all stages in the product-centric pipeline.
Includes checkpoint creation for recovery and visibility.
"""

import gc
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.schemas.product_progress import (
    ProductStage,
    ProductProcessingResult
)
from app.services.tracking.product_progress_tracker import ProductProgressTracker
from app.services.tracking.checkpoint_recovery_service import ProcessingStage as CheckpointStage
from app.utils.memory_monitor import MemoryPressureMonitor


logger = logging.getLogger(__name__)


async def process_single_product(
    product: Any,  # Product from catalog
    product_index: int,
    total_products: int,
    file_content: bytes,
    document_id: str,
    workspace_id: str,
    job_id: str,
    catalog: Any,
    tracker: Any,  # Main job tracker
    product_tracker: ProductProgressTracker,
    checkpoint_recovery_service: Any,
    supabase: Any,
    config: Dict[str, Any],
    logger_instance: logging.Logger,
    physical_page_upper_bound: Optional[int] = None,
    temp_pdf_path: Optional[str] = None,
) -> ProductProcessingResult:
    """
    Process a single product through all stages.

    Args:
        product: Product object from catalog
        product_index: 1-based index of product
        total_products: Total number of products
        file_content: PDF file bytes
        document_id: Document identifier
        workspace_id: Workspace identifier
        job_id: Job identifier
        catalog: Full product catalog
        tracker: Main job progress tracker
        product_tracker: Product-specific progress tracker
        checkpoint_recovery_service: Checkpoint service
        supabase: Supabase client
        config: Processing configuration
        logger_instance: Logger instance

    Returns:
        ProductProcessingResult with success/failure and metrics
    """
    start_time = datetime.utcnow()
    memory_monitor = MemoryPressureMonitor()
    product_id = f"product_{product_index}_{product.name.replace(' ', '_')}"

    logger_instance.info(f"\n{'='*80}")
    logger_instance.info(f"🏭 PRODUCT {product_index}/{total_products}: {product.name}")
    logger_instance.info(f"{'='*80}")

    # Initialize product tracking
    await product_tracker.initialize_product(
        product_id=product_id,
        product_name=product.name,
        product_index=product_index,
        metadata={
            "page_range": product.page_range,
            "confidence": product.confidence
        }
    )

    # Use direct attributes instead of nested metrics object
    result = ProductProcessingResult(
        product_id=product_id,
        product_name=product.name,
        product_index=product_index,
        success=False
    )

    # Track current stage for accurate error reporting
    current_stage = ProductStage.EXTRACTION

    # Fix E: per-product resume — pick up from the LAST checkpoint we
    # actually reached, instead of re-processing every stage from scratch.
    # On the previous run, products that reached chunks_created or
    # images_extracted were producing duplicates because we re-ran Stage 1+2
    # on resume; this short-circuits each stage individually.
    prior_stages: set = set()
    prior_db_id: Optional[str] = None
    skip_extraction = False
    skip_chunking = False
    skip_images = False
    skip_creation = False
    # What a completed-stage checkpoint claimed it wrote, so the DB row counts below
    # can be judged complete-or-partial rather than merely non-zero. None = no
    # checkpoint recorded a total, which is itself evidence the stage never finished.
    expected_chunks: Optional[int] = None
    expected_images: Optional[int] = None
    partial_image_resume: Optional[Dict[str, Any]] = None
    try:
        prior_status = await product_tracker.get_product_status(product_id)
        prior_db_id = (prior_status.metadata or {}).get('product_db_id') if prior_status else None
        if prior_status and prior_status.stages_completed:
            prior_stages.update(prior_status.stages_completed)
        # Also peek at job stage_history for per-product checkpoint events
        # tied to THIS product_index (catches resumes where product_tracker
        # state was wiped on the previous restart).
        # Audit fix #11: previously a transient Supabase 503 here would silently
        # swallow the exception → prior_stages stays empty → all stages re-run
        # → duplicate chunks/images. Now we log loudly so operator sees it,
        # but still don't raise (resume-from-DB-state below is a safety net).
        try:
            sb_resp = supabase.client.table('background_jobs') \
                .select('stage_history') \
                .eq('id', job_id).single().execute()
            for entry in (sb_resp.data or {}).get('stage_history', []) or []:
                ed = entry.get('data') or {}
                if str(ed.get('product_index')) == str(product_index):
                    # 'completed_empty' means the stage ran and produced NOTHING. It is
                    # not a reason to skip the stage forever — it is the reason to run
                    # it again. Adding it to prior_stages is what made a product that
                    # chunked to 0 once permanently un-chunkable on every later resume.
                    if entry.get('status') == 'completed_empty':
                        logger_instance.info(
                            f"↻ [RESUME] Product {product_index} stage "
                            f"'{entry.get('stage')}' previously completed EMPTY — will re-run"
                        )
                        continue
                    prior_stages.add(entry.get('stage'))
                    # Capture what the completed stage SAID it wrote. This is the
                    # expectation the DB state below is checked against — without it
                    # "some rows exist" is the only available signal, and that cannot
                    # tell a finished product from one the worker abandoned partway.
                    if ed.get('chunks_created') is not None:
                        expected_chunks = max(expected_chunks or 0, int(ed['chunks_created'] or 0))
                    if ed.get('images_extracted') is not None:
                        expected_images = max(expected_images or 0, int(ed['images_extracted'] or 0))
        except Exception as ckpt_err:
            logger_instance.error(
                f"⚠️ Resume checkpoint read FAILED for product {product_index} — "
                f"DB state will be sole source of truth: {ckpt_err}"
            )

        # Verify with database state — checkpoints alone aren't authoritative
        # because a stage might have been MID-INSERT when the worker died.
        # If chunks/images for this product already exist in DB, treat the
        # corresponding stage as done.
        # `> 0` is not "done". A worker that died after writing 2 of 30 chunks (or 1
        # of 40 images) leaves a non-zero count, which used to set the skip flag and
        # short-circuit the stage on EVERY future resume — the product stayed
        # permanently under-processed while the job completed green. The count is
        # compared against what the completed-stage checkpoint said it wrote.
        try:
            if prior_db_id:
                existing_chunks = supabase.client.table('document_chunks') \
                    .select('id', count='exact') \
                    .eq('document_id', document_id) \
                    .eq('product_id', prior_db_id) \
                    .execute()
                chunk_count = existing_chunks.count or 0
                if chunk_count > 0 and expected_chunks is not None and chunk_count >= expected_chunks:
                    prior_stages.add('chunks_created')
                elif chunk_count > 0:
                    # Partial (or unverifiable — no checkpoint recorded a total).
                    # Chunks are safe to rebuild: the embedding lives inline on the
                    # row, so deleting the partial set removes its vectors with it,
                    # and there is no unique constraint to make a re-run idempotent.
                    # Leaving them and re-running would DUPLICATE every chunk.
                    logger_instance.warning(
                        f"⚠️ [RESUME] Product {product_index} '{product.name}' has {chunk_count} "
                        f"chunk(s) but the stage never completed"
                        f"{f' (checkpoint expected {expected_chunks})' if expected_chunks is not None else ' (no checkpoint)'}"
                        f" — deleting the partial set and re-chunking"
                    )
                    supabase.client.table('document_chunks') \
                        .delete() \
                        .eq('document_id', document_id) \
                        .eq('product_id', prior_db_id) \
                        .execute()

                existing_imgs = supabase.client.table('document_images') \
                    .select('id', count='exact') \
                    .eq('document_id', document_id) \
                    .eq('product_id', prior_db_id) \
                    .execute()
                image_count = existing_imgs.count or 0
                if image_count > 0 and expected_images is not None and image_count >= expected_images:
                    prior_stages.add('images_extracted')
                elif image_count > 0:
                    # Images are NOT rebuilt the same way. Each row has vectors in the
                    # VECS collections that a plain row delete would orphan, and a
                    # re-run would re-bill Claude vision for every image. So the skip
                    # stands — but it is recorded rather than assumed, so an operator
                    # or repair pass can find products that resumed on a partial image
                    # set instead of them silently reading as complete.
                    prior_stages.add('images_extracted')
                    partial_image_resume = {
                        'found': image_count,
                        'expected': expected_images,
                        'product_index': product_index,
                    }
                    logger_instance.warning(
                        f"⚠️ [RESUME] Product {product_index} '{product.name}' resuming on "
                        f"{image_count} image(s) with no completed images checkpoint"
                        f"{f' (expected {expected_images})' if expected_images is not None else ''} — "
                        f"reusing them (delete would orphan VECS vectors and re-bill vision); "
                        f"flagged as resume_incomplete on the product row"
                    )
        except Exception as db_check_err:
            logger_instance.debug(f"DB state check failed for {product.name}: {db_check_err}")

        # Whole-product skip if everything's done.
        if 'relationships_created' in prior_stages or 'completed' in prior_stages:
            logger_instance.info(
                f"♻️  [RESUME] Product {product_index}/{total_products} '{product.name}' "
                f"already fully processed — skipping all stages"
            )
            result.success = True
            result.product_db_id = prior_db_id
            await product_tracker.mark_product_complete(product_id, result)
            return result

        # Per-stage skip flags. Each stage will check these and short-circuit.
        # We DON'T skip Stage 1 extraction because the in-memory page list +
        # layout regions are needed by Stage 3; just re-extracting pages is
        # cheap (no AI calls) so it's fine to redo.
        if 'chunks_created' in prior_stages:
            skip_chunking = True
            logger_instance.info(
                f"♻️  [RESUME] Product {product_index} '{product.name}' chunks already in DB — "
                f"will reuse instead of re-creating"
            )
        if 'images_extracted' in prior_stages:
            skip_images = True
            logger_instance.info(
                f"♻️  [RESUME] Product {product_index} '{product.name}' images already in DB — "
                f"will reuse instead of re-creating"
            )
        if 'products_created' in prior_stages:
            skip_creation = True

    except Exception as resume_check_err:
        logger_instance.debug(f"Per-product resume check failed (continuing): {resume_check_err}")

    # Persist the partial-resume marker so a product that resumed on an incomplete
    # image set is DISCOVERABLE, not merely mentioned in a log line that scrolls away.
    # Same convention as embedding_metadata.status='failed' (S3-8) and
    # vision_analysis_failed: query for it with
    #     metadata->'resume_incomplete' is not null
    if partial_image_resume and prior_db_id:
        try:
            _existing = supabase.client.table('products') \
                .select('metadata').eq('id', prior_db_id).single().execute()
            _meta = (_existing.data or {}).get('metadata') or {}
            supabase.client.table('products').update({
                'metadata': {
                    **_meta,
                    'resume_incomplete': {
                        **partial_image_resume,
                        'stage': 'images_extracted',
                        'detected_at': datetime.utcnow().isoformat(),
                    },
                }
            }).eq('id', prior_db_id).execute()
        except Exception as _mark_err:
            logger_instance.warning(
                f"⚠️ could not stamp resume_incomplete on product {prior_db_id}: {_mark_err}"
            )

    try:
        # ========================================================================
        # STAGE 1: Extract Product Pages + Layout Detection
        # ========================================================================
        current_stage = ProductStage.EXTRACTION
        await product_tracker.update_product_stage(product_id, ProductStage.EXTRACTION)
        logger_instance.info(f"📄 [STAGE 1/{product_index}] Extracting pages for {product.name}...")

        # UPDATE PROGRESS: Update tracker at start of each stage
        if tracker:
            tracker.current_step = f"Stage 1: Extracting pages for {product.name}"
            await tracker.update_heartbeat()

        from app.api.pdf_processing.stage_1_focused_extraction import extract_product_pages

        # extract_product_pages returns a dict with layout detection results.
        # Pass catalog for spread layout info (physical page -> PDF page mapping)
        extraction_result = await extract_product_pages(
            file_content=file_content,
            product=product,
            document_id=document_id,
            job_id=job_id,
            logger=logger_instance,
            physical_page_upper_bound=physical_page_upper_bound,
            enable_layout_detection=False,  # Disable for now - will run after product creation
            product_id=None,  # Will be set after product creation
            catalog=catalog  # NEW: Pass catalog for spread layout handling
        )

        # Extract results - NOW USING PHYSICAL PAGES as primary
        physical_pages = extraction_result['physical_pages']  # Physical page numbers (1-based)
        layout_regions = extraction_result.get('layout_regions', [])
        layout_stats = extraction_result.get('layout_stats', {})
        # Get spread layout info for internal PDF access
        has_spread_layout = extraction_result.get('has_spread_layout', False)
        physical_to_pdf_map = extraction_result.get('physical_to_pdf_map', {})

        await product_tracker.mark_stage_complete(
            product_id,
            ProductStage.EXTRACTION,
            {
                "pages_extracted": len(physical_pages),
                "layout_regions_detected": len(layout_regions),
                "layout_stats": layout_stats
            }
        )
        pages_extracted = len(physical_pages)
        logger_instance.info(
            f"✅ Extracted {pages_extracted} physical pages for {product.name}: {physical_pages}"
        )

        # ✅ CHECKPOINT: PDF_EXTRACTED - Stage 1 complete
        await checkpoint_recovery_service.create_checkpoint(
            job_id=job_id,
            stage=CheckpointStage.PDF_EXTRACTED,
            data={
                "document_id": document_id,
                "product_name": product.name,
                "product_index": product_index,
                "pages_extracted": pages_extracted,
                "physical_pages": physical_pages
            },
            metadata={
                "layout_regions_detected": len(layout_regions),
                "layout_stats": layout_stats,
                "has_spread_layout": has_spread_layout
            }
        )
        logger_instance.info(f"   📌 Created PDF_EXTRACTED checkpoint for {product.name}")

        # Layout regions come from the doc-level PaddleOCR structural pass (Stage 1),
        # read from document_layout_analysis by the chunker. No per-product pass.
        layout_regions = []


        # ========================================================================
        # Get product_db_id early (set in Stage 0 discovery) for chunk linking
        # ========================================================================
        product_status = await product_tracker.get_product_status(product_id)
        product_db_id = product_status.metadata.get('product_db_id') if product_status else None
        if not product_db_id:
            logger_instance.warning(f"⚠️ product_db_id not found for {product.name} - chunks won't be linked")

        # ========================================================================
        # STAGE 2: Create Text Chunks
        # ========================================================================
        current_stage = ProductStage.CHUNKING
        await product_tracker.update_product_stage(product_id, ProductStage.CHUNKING)
        logger_instance.info(f"📝 [STAGE 2/{product_index}] Creating chunks for {product.name}...")

        # ✅ UPDATE PROGRESS: Update tracker at start of chunking stage
        if tracker:
            tracker.current_step = f"Stage 2: Creating chunks for {product.name}"
            await tracker.update_heartbeat()

        from app.api.pdf_processing.stage_2_chunking import process_product_chunking

        if skip_chunking and product_db_id:
            # Resume optimization: chunks already exist in DB for this
            # product (from a prior run). Read counts and skip the call.
            try:
                existing = supabase.client.table('document_chunks') \
                    .select('id, text_embedding', count='exact') \
                    .eq('document_id', document_id) \
                    .eq('product_id', product_db_id) \
                    .execute()
                chunks_created = existing.count or 0
                # Count rows that already have an embedding stored
                embeddings_generated = sum(
                    1 for r in (existing.data or []) if r.get('text_embedding') is not None
                )
                logger_instance.info(
                    f"♻️  [RESUME Stage 2] Reusing {chunks_created} existing chunks "
                    f"({embeddings_generated} embeddings) for {product.name} — Voyage call skipped"
                )
                chunk_result = {
                    'chunks_created': chunks_created,
                    'embeddings_generated': embeddings_generated,
                    'skipped': True,
                }
            except Exception as reuse_err:
                logger_instance.warning(
                    f"⚠️ Could not reuse existing chunks for {product.name}: {reuse_err} — "
                    f"falling through to fresh chunking"
                )
                skip_chunking = False

        if not skip_chunking:
            chunk_result = await process_product_chunking(
                file_content=file_content,
                document_id=document_id,
                workspace_id=workspace_id,
                job_id=job_id,
                product=product,
                physical_pages=physical_pages,
                catalog=catalog,
                config=config,
                supabase=supabase,
                logger=logger_instance,
                product_id=product_db_id,
                temp_pdf_path=temp_pdf_path,
                layout_regions=layout_regions
            )

            chunks_created = chunk_result.get('chunks_created', 0)
            embeddings_generated = chunk_result.get('embeddings_generated', 0)
            # 'failed' means the extractor broke, not that the product has no text.
            # Checkpointing that as a completed stage is what made the outcome
            # permanent: the resume path would skip chunking for this product on
            # every subsequent run. Raise so the product is recorded as failed and
            # stays eligible for a retry.
            if chunk_result.get('chunking_status') == 'failed':
                raise RuntimeError(
                    f"chunking failed for {product.name}: "
                    f"{chunk_result.get('error') or 'text extraction error'}"
                )
            logger_instance.info(f"✅ Created {chunks_created} chunks for {product.name} ({embeddings_generated} text embeddings)")
        await product_tracker.mark_stage_complete(
            product_id,
            ProductStage.CHUNKING,
            {"chunks_created": chunks_created, "text_embeddings_generated": embeddings_generated, "layout_aware": True}
        )
        result.chunks_created = chunks_created

        # Update tracker with text embeddings count
        if tracker:
            await tracker.update_database_stats(
                chunks_created=chunks_created,
                text_embeddings=embeddings_generated,
                sync_to_db=True
            )
            # Log the actual tracker values to verify sync
            logger_instance.info(f"   📊 Updated tracker: {chunks_created} chunks, {embeddings_generated} text embeddings")
            logger_instance.info(f"   📊 Tracker totals: chunks={tracker.chunks_created}, text_embeddings={tracker.text_embeddings_generated}")
        logger_instance.info(f"✅ Created {chunks_created} chunks for {product.name}")

        # ✅ CHECKPOINT: CHUNKS_CREATED - Stage 2 complete
        await checkpoint_recovery_service.create_checkpoint(
            job_id=job_id,
            stage=CheckpointStage.CHUNKS_CREATED,
            data={
                "document_id": document_id,
                "product_name": product.name,
                "product_index": product_index,
                "chunks_created": chunks_created
            },
            metadata={
                "text_embeddings_generated": embeddings_generated,
                "layout_aware": True,
                "product_db_id": product_db_id
            }
        )
        logger_instance.info(f"   📌 Created CHUNKS_CREATED checkpoint for {product.name}")

        # ✅ CHECKPOINT: TEXT_EMBEDDINGS_GENERATED - Text embeddings complete
        if embeddings_generated > 0:
            await checkpoint_recovery_service.create_checkpoint(
                job_id=job_id,
                stage=CheckpointStage.TEXT_EMBEDDINGS_GENERATED,
                data={
                    "document_id": document_id,
                    "product_name": product.name,
                    "product_index": product_index,
                    "text_embeddings_generated": embeddings_generated
                },
                metadata={
                    "chunks_created": chunks_created,
                    "product_db_id": product_db_id
                }
            )
            logger_instance.info(f"   📌 Created TEXT_EMBEDDINGS_GENERATED checkpoint for {product.name}")

        # ========================================================================
        # STAGE 2.5: Persist VLM-recognized tables → product_tables
        # ========================================================================
        # Stage 1 already recognized every TABLE region to markdown/HTML and
        # cached it on the layout element (metadata.html). Parsing it into
        # product_tables here is what lets TableMetadataExtractor (Stage 5, via
        # entity linking) read packaging + performance specs. Before this
        # wire-up product_tables was never written by any stage — #248 removed
        # the per-product writer assuming Stage 2 consumed metadata.html, and it
        # never did — so that extractor mined an always-empty table on every
        # product and available_sizes/thickness fell through to the Stage 4.6
        # regex safety net.
        #
        # Deliberately NOT gated on skip_chunking: a job resuming from a run
        # that predates this stage has chunks but no tables and would otherwise
        # never get them. persist_tables_from_layout_cache is idempotent per
        # product (it clears this product's rows before inserting).
        tables_stored = 0
        if product_db_id:
            try:
                from app.services.pdf.table_extraction import TableExtractor

                tables_stored = await TableExtractor().persist_tables_from_layout_cache(
                    document_id=document_id,
                    product_id=product_db_id,
                    physical_pages=physical_pages,
                    supabase=supabase,
                    logger=logger_instance,
                )
            except Exception as table_err:
                logger_instance.warning(
                    f"   ⚠️ Table persistence failed for {product.name} (non-fatal): {table_err}"
                )

        # ========================================================================
        # STAGE 3: Process Images
        # ========================================================================
        current_stage = ProductStage.IMAGES
        await product_tracker.update_product_stage(product_id, ProductStage.IMAGES)
        logger_instance.info(f"🖼️  [STAGE 3/{product_index}] Processing images for {product.name}...")

        # ✅ UPDATE PROGRESS: Update tracker at start of image processing stage
        if tracker:
            tracker.current_step = f"Stage 3: Processing images for {product.name}"
            await tracker.update_heartbeat()

        from app.api.pdf_processing.stage_3_images import process_product_images

        # Safe defaults — handles the edge case where skip_images=True but
        # product_db_id is missing (Stage 0 didn't write the DB row): neither
        # the resume block nor the fresh-process block runs, and the
        # downstream `if images_failed_count:` would UnboundLocalError into
        # the bare except.
        images_processed = 0
        clip_embeddings = 0
        vector_stats: Dict[str, Any] = {}
        failed_images_list: List[Any] = []
        images_failed_count = 0
        image_result: Dict[str, Any] = {
            'images_processed': 0,
            'clip_embeddings_generated': 0,
            'vector_stats': {},
            'failed_images': [],
            'images_material': 0,
            'images_icon_candidates': 0,
            'images_non_material': 0,
            'skipped': True,
        }

        if skip_images and product_db_id:
            try:
                existing_imgs = supabase.client.table('document_images') \
                    .select('id, has_slig_embedding', count='exact') \
                    .eq('document_id', document_id) \
                    .eq('product_id', product_db_id) \
                    .execute()
                images_processed = existing_imgs.count or 0
                clip_embeddings = sum(
                    1 for r in (existing_imgs.data or []) if r.get('has_slig_embedding')
                )
                logger_instance.info(
                    f"♻️  [RESUME Stage 3] Reusing {images_processed} existing images "
                    f"({clip_embeddings} with SLIG embeddings) for {product.name} — "
                    f"Claude vision + SLIG + Voyage calls skipped"
                )
                image_result = {
                    'images_processed': images_processed,
                    'clip_embeddings_generated': clip_embeddings,
                    'vector_stats': {},
                    'failed_images': [],
                    'images_material': images_processed,
                    'images_icon_candidates': 0,
                    'images_non_material': 0,
                    'skipped': True,
                }
                vector_stats = {}
                failed_images_list = []
                images_failed_count = 0
            except Exception as reuse_err:
                logger_instance.warning(
                    f"⚠️ Could not reuse existing images for {product.name}: {reuse_err} — "
                    f"falling through to fresh image processing"
                )
                skip_images = False

        if not skip_images:
            image_result = await process_product_images(
                file_content=file_content,
                document_id=document_id,
                workspace_id=workspace_id,
                job_id=job_id,
                product=product,
                physical_pages=physical_pages,  # 1-based physical pages
                catalog=catalog,
                config=config,
                logger=logger_instance,
                layout_regions=layout_regions,  # layout regions for bbox data
                tracker=tracker,  # Per-image progress events visible in admin UI
                product_db_id=product_db_id,  # plumb FK for ai_usage_logs cost attribution
            )

            images_processed = image_result.get('images_processed', 0)
            clip_embeddings = image_result.get('clip_embeddings_generated', 0)
            vector_stats = image_result.get('vector_stats', {}) or {}
            failed_images_list = image_result.get('failed_images', []) or []
            images_failed_count = len(failed_images_list)
        if images_failed_count:
            logger_instance.warning(
                f"   ⚠️ {images_failed_count} image(s) failed to save for {product.name} — "
                f"surfaced to job status under image_save_failed"
            )
        await product_tracker.mark_stage_complete(
            product_id,
            ProductStage.IMAGES,
            {
                "images_processed": images_processed,
                "images_material": image_result.get('images_material', 0),
                "images_icon_candidates": image_result.get('images_icon_candidates', 0),
                "images_non_material": image_result.get('images_non_material', 0),
                "clip_embeddings_generated": clip_embeddings,
                # Failure surface — was previously only logged, not tracked.
                "image_save_failed": images_failed_count,
                # Per-vector breakdown — visible in admin UI
                "visual_slig_count": vector_stats.get('visual_slig', 0),
                "color_slig_count": vector_stats.get('color_slig', 0),
                "texture_slig_count": vector_stats.get('texture_slig', 0),
                "style_slig_count": vector_stats.get('style_slig', 0),
                "material_slig_count": vector_stats.get('material_slig', 0),
                "understanding_count": vector_stats.get('understanding', 0),
                "vision_analysis_claude": vector_stats.get('vision_analysis_claude', 0),
                "vision_analysis_claude_fallback": vector_stats.get('vision_analysis_claude_fallback', 0),
                "vision_analysis_failed": vector_stats.get('vision_analysis_failed', 0),
                # Icon extraction stats — visible in admin UI
                "icon_candidates_processed": vector_stats.get('icon_candidates_processed', 0),
                "icon_metadata_extracted": vector_stats.get('icon_metadata_extracted', 0),
                "icon_extraction_failed": vector_stats.get('icon_extraction_failed', 0),
            }
        )
        result.images_processed = images_processed
        result.clip_embeddings_generated = clip_embeddings
        logger_instance.info(f"✅ Processed {images_processed} images for {product.name}")
        logger_instance.info(f"✅ Generated {clip_embeddings} SLIG embeddings for {product.name}")

        # Update tracker with SLIG embeddings count.
        # ProgressTracker.update_database_stats() expects `images_extracted`, not `images_stored`.
        if tracker:
            await tracker.update_database_stats(
                images_extracted=images_processed,
                clip_embeddings=clip_embeddings,
                image_embeddings=clip_embeddings,
                sync_to_db=True
            )
            # Log the actual tracker values to verify sync
            logger_instance.info(f"   📊 Updated tracker: {images_processed} images, {clip_embeddings} SLIG embeddings")
            logger_instance.info(f"   📊 Tracker totals: images_extracted={tracker.images_extracted}, clip_embeddings={tracker.clip_embeddings_generated}, image_embeddings={tracker.image_embeddings_generated}")

        # ✅ CHECKPOINT: IMAGES_EXTRACTED - Stage 3 complete
        await checkpoint_recovery_service.create_checkpoint(
            job_id=job_id,
            stage=CheckpointStage.IMAGES_EXTRACTED,
            data={
                "document_id": document_id,
                "product_name": product.name,
                "product_index": product_index,
                "images_processed": images_processed
            },
            metadata={
                "images_material": image_result.get('images_material', 0),
                "images_non_material": image_result.get('images_non_material', 0),
                "product_db_id": product_db_id
            }
        )
        logger_instance.info(f"   📌 Created IMAGES_EXTRACTED checkpoint for {product.name}")

        # ✅ CHECKPOINT: IMAGE_EMBEDDINGS_GENERATED - SLIG embeddings complete
        if clip_embeddings > 0:
            await checkpoint_recovery_service.create_checkpoint(
                job_id=job_id,
                stage=CheckpointStage.IMAGE_EMBEDDINGS_GENERATED,
                data={
                    "document_id": document_id,
                    "product_name": product.name,
                    "product_index": product_index,
                    "clip_embeddings_generated": clip_embeddings
                },
                metadata={
                    "images_processed": images_processed,
                    "product_db_id": product_db_id
                }
            )
            logger_instance.info(f"   📌 Created IMAGE_EMBEDDINGS_GENERATED checkpoint for {product.name}")

        # ========================================================================
        # STAGE 4: Update Product with Extracted Metadata
        # ========================================================================
        # NOTE: Product was already created in Stage 0 (discovery)
        # Here we just update it with extracted metadata from processing
        current_stage = ProductStage.CREATION
        await product_tracker.update_product_stage(product_id, ProductStage.CREATION)
        logger_instance.info(f"🏭 [STAGE 4/{product_index}] Updating product with extracted metadata...")

        # ✅ UPDATE PROGRESS: Update tracker at start of product update stage
        if tracker:
            tracker.current_step = f"Stage 4: Updating product metadata for {product.name}"
            await tracker.update_heartbeat()

        # product_db_id was already retrieved early (before Stage 2) for chunk linking

        if not product_db_id:
            raise Exception(f"Product DB ID not found for {product.name} - product should have been created in Stage 0")

        # Update product with extracted metadata from Stage 1.
        # MERGE with existing metadata to preserve discovery metadata.
        extracted_metadata = extraction_result.get('metadata', {})

        # Pull chunk-level structured_metadata aggregated across all of this
        # product's chunks. Sonnet 4.6 chunk classification writes per-chunk
        # `metadata.structured_metadata.{dimensions, colors, materials,
        # keyFeatures, productName, studioName}` but those values were never
        # rolled up onto the product's metadata before — leaving
        # `product.metadata.dimensions=[]` even when chunks clearly captured
        # the size. Audit incident: job acff9ebb 2026-05-03, FOLD chunks
        # carried `structured_metadata.dimensions='15x38'` but
        # `products.metadata.dimensions` was an empty list.
        chunk_aggregated: Dict[str, Any] = {}
        try:
            chunk_resp = supabase.client.table('document_chunks') \
                .select('metadata') \
                .eq('product_id', product_db_id) \
                .execute()
            agg_dimensions: set = set()
            agg_colors: set = set()
            agg_materials: set = set()
            agg_features: set = set()
            studio_name: Optional[str] = None
            for ch in (chunk_resp.data or []):
                sm = ((ch.get('metadata') or {}).get('structured_metadata') or {})
                # dimensions: usually a single string like "15x38" but allow lists.
                sm_dim = sm.get('dimensions')
                if isinstance(sm_dim, str) and sm_dim.strip():
                    agg_dimensions.add(sm_dim.strip())
                elif isinstance(sm_dim, list):
                    agg_dimensions.update(d for d in sm_dim if isinstance(d, str) and d.strip())
                for k_src, target_set in (
                    ('colors', agg_colors),
                    ('materials', agg_materials),
                    ('keyFeatures', agg_features),
                ):
                    val = sm.get(k_src)
                    if isinstance(val, list):
                        target_set.update(v for v in val if isinstance(v, str) and v.strip())
                    elif isinstance(val, str) and val.strip():
                        target_set.add(val.strip())
                if not studio_name:
                    sn = sm.get('studioName')
                    if isinstance(sn, str) and sn.strip():
                        studio_name = sn.strip()
            if agg_dimensions:
                chunk_aggregated['dimensions'] = sorted(agg_dimensions)
            if agg_colors:
                chunk_aggregated['available_colors'] = sorted(agg_colors)
            if agg_materials:
                chunk_aggregated.setdefault('material_properties', {})
                chunk_aggregated['material_properties']['materials_mentioned'] = sorted(agg_materials)
            if agg_features:
                chunk_aggregated['key_features'] = sorted(agg_features)
            if studio_name and not extracted_metadata.get('studio_name'):
                chunk_aggregated['studio_name'] = studio_name
            if chunk_aggregated:
                logger_instance.info(
                    f"   📦 Chunk-aggregated structured_metadata: "
                    f"dimensions={len(agg_dimensions)}, colors={len(agg_colors)}, "
                    f"materials={len(agg_materials)}, features={len(agg_features)}"
                )
        except Exception as agg_err:
            logger_instance.warning(
                f"   ⚠️ Failed to aggregate chunk structured_metadata for {product.name}: {agg_err}"
            )

        if extracted_metadata or chunk_aggregated:
            try:
                # First fetch existing metadata to merge (preserves discovery metadata)
                existing_product = supabase.client.table('products')\
                    .select('metadata')\
                    .eq('id', product_db_id)\
                    .single()\
                    .execute()

                existing_metadata = existing_product.data.get('metadata', {}) if existing_product.data else {}
                if existing_metadata is None:
                    existing_metadata = {}

                # Deep merge: extracted_metadata takes priority but preserves existing fields.
                # Chunk-aggregated values come in last so they fill empty / missing keys
                # without overriding values the per-product extractor already produced.
                merged_metadata = {**existing_metadata}
                for source in (extracted_metadata, chunk_aggregated):
                    for key, value in source.items():
                        if value is None or value == "" or value == [] or value == {}:
                            continue
                        existing_val = merged_metadata.get(key)
                        existing_empty = existing_val in (None, "", [], {})
                        if existing_empty:
                            merged_metadata[key] = value
                        elif isinstance(existing_val, dict) and isinstance(value, dict):
                            merged_metadata[key] = {**existing_val, **value}
                        elif isinstance(existing_val, list) and isinstance(value, list):
                            try:
                                existing_set = set(existing_val) if all(isinstance(x, (str, int, float)) for x in existing_val) else None
                                new_set = set(value) if all(isinstance(x, (str, int, float)) for x in value) else None
                                if existing_set is not None and new_set is not None:
                                    merged_metadata[key] = sorted(list(existing_set | new_set))
                                else:
                                    merged_metadata[key] = existing_val + [v for v in value if v not in existing_val]
                            except TypeError:
                                merged_metadata[key] = existing_val + [v for v in value if v not in existing_val]
                        elif source is extracted_metadata:
                            # extracted_metadata wins over existing
                            merged_metadata[key] = value
                        # else: chunk-aggregated value loses to non-empty extracted/existing

                supabase.client.table('products')\
                    .update({'metadata': merged_metadata})\
                    .eq('id', product_db_id)\
                    .execute()
                logger_instance.info(f"✅ Merged and updated product metadata in DB: {product_db_id} ({len(merged_metadata)} fields)")
            except Exception as e:
                # SPN-8: make the impact explicit rather than success-masking. The
                # product row already exists (created upstream) and keeps its
                # pre-merge metadata — this is a LOST enrichment merge, not a total
                # product failure. Logged at ERROR with the product id so operators
                # can re-run enrichment for exactly this product if needed.
                logger_instance.error(
                    f"❌ [SPN-8] Product metadata MERGE dropped for {product_db_id} "
                    f"(product retains pre-merge metadata; enrichment lost, not a "
                    f"product failure): {e}"
                )

        # 4b. Layout-region storage + per-product table extraction — REMOVED
        # (SPN-4, 2026-07-04). `layout_regions` has been ALWAYS [] here since the
        # 2026-06-14 cutover (stage_1_focused_extraction returns layout_regions=[]),
        # so the old `if layout_regions and product_db_id:` block never executed:
        # no product_layout_regions rows were written and no per-product tables were
        # extracted. Layout is owned by the PaddleOCR Stage 1 pass. Deleted the dead
        # block wholesale (issue #248).
        #
        # #248 assumed TABLE content (preserved as metadata.html in
        # document_layout_analysis) was "consumed by Stage 2". It was not — nothing
        # read that field, so product_tables stayed empty from 2026-07-04 until the
        # Stage 2.5 wire-up above now parses it.

        await product_tracker.mark_stage_complete(
            product_id,
            ProductStage.CREATION,
            {"product_db_id": product_db_id}
        )
        result.product_db_id = product_db_id
        logger_instance.info(f"✅ Product updated in DB: {product_db_id}")

        # ✅ CHECKPOINT: PRODUCTS_CREATED - Stage 4 complete
        await checkpoint_recovery_service.create_checkpoint(
            job_id=job_id,
            stage=CheckpointStage.PRODUCTS_CREATED,
            data={
                "document_id": document_id,
                "product_name": product.name,
                "product_index": product_index,
                "product_db_id": product_db_id
            },
            metadata={
                "layout_regions_stored": len(layout_regions) if layout_regions else 0,
                "tables_extracted": tables_stored
            }
        )
        logger_instance.info(f"   📌 Created PRODUCTS_CREATED checkpoint for {product.name}")

        # ========================================================================
        # STAGE 4.5: Auto-create KB documents from extracted metadata
        # ========================================================================
        logger_instance.info(f"📚 [STAGE 4.5/{product_index}] Creating knowledge base documents...")

        try:
            from app.services.knowledge.auto_kb_document_service import AutoKBDocumentService

            kb_service = AutoKBDocumentService()
            kb_result = await kb_service.create_kb_documents_from_metadata(
                product_id=product_db_id,
                product_name=product.name,
                workspace_id=workspace_id,
                metadata=extraction_result.get('metadata', {})
            )

            kb_docs_created = kb_result.get('documents_created', 0)
            if kb_docs_created > 0:
                logger_instance.info(f"   ✅ Created {kb_docs_created} KB documents")
            else:
                logger_instance.info(f"   ℹ️ No KB documents created (no eligible metadata)")
        except Exception as e:
            logger_instance.warning(f"   ⚠️ KB creation failed: {e}")

        # ========================================================================
        # STAGE 5: Create Relationships (Link chunks/images to product)
        # ========================================================================
        current_stage = ProductStage.RELATIONSHIPS
        await product_tracker.update_product_stage(product_id, ProductStage.RELATIONSHIPS)
        logger_instance.info(f"🔗 [STAGE 5/{product_index}] Creating relationships...")

        from app.services.discovery.entity_linking_service import EntityLinkingService

        entity_linking_service = EntityLinkingService(supabase)
        linking_result = await entity_linking_service.link_product_entities(
            product_id=product_db_id,
            product_name=product.name,
            document_id=document_id,
            physical_pages=set(physical_pages),  # 1-based physical pages
            logger=logger_instance
        )

        # Per-product chunk-image linking — was previously only run as a
        # document-level pass at end-of-job, which meant a partial-success
        # run (some products fail, others succeed) lost ALL chunk-image
        # links. Audit incident: job acff9ebb 2026-05-03, FOLD completed
        # cleanly but the cancelled job never reached the finalize block →
        # 0 chunk_image_relationships for an otherwise-successful product.
        # The end-of-document pass remains as a safety net.
        try:
            chunk_image_links = await entity_linking_service.link_images_to_chunks(
                document_id=document_id,
                product_db_id=product_db_id,
            )
            logger_instance.info(
                f"   🔗 Per-product chunk-image links: {chunk_image_links} for {product.name}"
            )
        except Exception as link_err:
            logger_instance.warning(
                f"   ⚠️ Per-product chunk-image linking failed for {product.name} "
                f"(safety-net pass at end-of-document will retry): {link_err}"
            )
            chunk_image_links = 0

        relationships_created = linking_result.get('relationships_created', 0) + chunk_image_links
        await product_tracker.mark_stage_complete(
            product_id,
            ProductStage.RELATIONSHIPS,
            {"relationships_created": relationships_created}
        )
        result.relationships_created = relationships_created
        logger_instance.info(f"✅ Created {relationships_created} relationships")

        # Update tracker with relationships count
        if tracker:
            await tracker.update_database_stats(
                relations_created=relationships_created,
                sync_to_db=True
            )
            logger_instance.info(f"   📊 Updated tracker: {relationships_created} relationships (total relations={tracker.relations_created})")

        # ✅ CHECKPOINT: RELATIONSHIPS_CREATED - Stage 5 complete
        await checkpoint_recovery_service.create_checkpoint(
            job_id=job_id,
            stage=CheckpointStage.RELATIONSHIPS_CREATED,
            data={
                "document_id": document_id,
                "product_name": product.name,
                "product_index": product_index,
                "relationships_created": relationships_created
            },
            metadata={
                "product_db_id": product_db_id,
                "chunks_linked": linking_result.get('chunks_linked', 0),
                "images_linked": linking_result.get('images_linked', 0)
            }
        )
        logger_instance.info(f"   📌 Created RELATIONSHIPS_CREATED checkpoint for {product.name}")

        # ========================================================================
        # SUCCESS: Mark product as complete
        # ========================================================================
        end_time = datetime.utcnow()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        result.processing_time_ms = processing_time_ms
        result.success = True

        # Cleanup memory and track freed amount
        memory_before = memory_monitor.get_memory_stats().used_mb
        await cleanup_product_memory(logger_instance)
        memory_after = memory_monitor.get_memory_stats().used_mb
        result.memory_freed_mb = max(0, memory_before - memory_after)

        await product_tracker.mark_product_complete(product_id, result)

        logger_instance.info(f"\n{'='*80}")
        logger_instance.info(f"✅ PRODUCT {product_index}/{total_products} COMPLETE: {product.name}")
        logger_instance.info(f"   Chunks: {result.chunks_created}, Images: {result.images_processed}, Relationships: {result.relationships_created}")
        logger_instance.info(f"   Time: {processing_time_ms/1000:.1f}s, Memory freed: {result.memory_freed_mb:.1f} MB")
        logger_instance.info(f"{'='*80}\n")

        return result

    except Exception as e:
        # ========================================================================
        # ERROR: Mark product as failed
        # ========================================================================
        logger_instance.error(f"❌ Product {product_index}/{total_products} FAILED: {product.name}")
        logger_instance.error(f"   Error: {str(e)}")
        logger_instance.error(f"   Failed at stage: {current_stage.value}")

        import traceback
        logger_instance.error(f"   Traceback: {traceback.format_exc()}")

        # Use the tracked current_stage (set before each stage execution)
        # This is accurate because we update it right before each stage starts
        result.error = str(e)
        result.error_stage = current_stage

        await product_tracker.mark_product_failed(
            product_id=product_id,
            error_message=str(e),
            error_stage=current_stage
        )

        # Clear any in-flight slow-op marker so the auto-recovery cron sees
        # a clean state for the next product (or for the next job). Pass THIS
        # product's exact per-product key (SPN-9) so we don't pop a sibling's
        # still-active marker under parallel product processing.
        if tracker is not None:
            try:
                await tracker.clear_slow_operation(operation=f"stage_3_images:{product.name}")
            except Exception:
                pass

        # Still cleanup memory even on failure
        await cleanup_product_memory(logger_instance)

        return result


async def cleanup_product_memory(logger_instance: logging.Logger) -> None:
    """
    Smart memory cleanup after processing a product.

    PRESERVES (needed for next products):
    - file_content (bytes) - Original PDF file
    - catalog - Product discovery results
    - temp_pdf_path - Temporary PDF file on disk
    - tracker - Main job tracker
    - product_tracker - Product progress tracker
    - supabase - Database client
    - config - Processing configuration

    CLEANS UP (product-specific data):
    - physical_pages (List[int]) - Physical page numbers (1-based) for this product
    - chunks - Text chunks for this product
    - images - Image data for this product
    - embedding vectors - Temporary embeddings
    - AI model caches - Temporary model outputs

    Args:
        logger_instance: Logger for tracking cleanup
    """
    logger_instance.debug("🧹 Starting smart product memory cleanup...")

    # Import memory_monitor from the module (fix NameError)
    from app.utils.memory_monitor import memory_monitor

    # Get memory before cleanup
    mem_before = memory_monitor.get_memory_stats()
    logger_instance.debug(f"   💾 Memory before: {mem_before.used_mb:.1f} MB ({mem_before.percent_used:.1f}%)")

    # Force garbage collection (generation 0 - recent objects)
    collected = gc.collect(0)
    logger_instance.debug(f"   Collected {collected} gen-0 objects (product-specific data)")

    # Additional cleanup for generation 1 (medium-lived objects)
    collected_gen1 = gc.collect(1)
    logger_instance.debug(f"   Collected {collected_gen1} gen-1 objects")

    # Generation-2 sweep — model caches, HF endpoint clients, numpy buffers,
    # SLIG embeddings cached at module level all live here. Without this
    # sweep, RSS climbs by ~100-200MB per product (Bug — job b7d70de1
    # hit RSS=2.8GB on product 1 alone).
    collected_gen2 = gc.collect(2)
    logger_instance.debug(f"   Collected {collected_gen2} gen-2 objects (model caches, long-lived buffers)")

    # Drop the cross-product PDFProcessor singleton so the next product's
    # Stage 3 starts fresh. Without this, fitz docs + image buffers
    # accumulate in module-level `_pdf_processor_instance` from
    # stage_3_images.py.
    try:
        from app.api.pdf_processing.stage_3_images import clear_pdf_processor
        clear_pdf_processor()
    except Exception as cleanup_err:
        logger_instance.debug(f"   PDFProcessor singleton clear skipped: {cleanup_err}")

    # Get memory after cleanup
    mem_after = memory_monitor.get_memory_stats()
    mem_freed = mem_before.used_mb - mem_after.used_mb

    logger_instance.debug(f"   💾 Memory after: {mem_after.used_mb:.1f} MB ({mem_after.percent_used:.1f}%)")
    logger_instance.debug(f"   ✅ Freed: {mem_freed:.1f} MB")
    logger_instance.debug("✅ Product memory cleanup complete")


def update_product_progress(
    product_index: int,
    total_products: int,
    stage: ProductStage,
    tracker: Any
) -> None:
    """
    Update overall job progress based on product progress.

    Calculates progress as:
    - Each product contributes (100 / total_products)%
    - Each stage within a product contributes (100 / total_products / 5)%

    Args:
        product_index: Current product index (1-based)
        total_products: Total number of products
        stage: Current processing stage
        tracker: Main job tracker
    """
    # Map stages to progress within a product (0-100%)
    stage_progress = {
        ProductStage.EXTRACTION: 20,
        ProductStage.CHUNKING: 40,
        ProductStage.IMAGES: 60,
        ProductStage.CREATION: 80,
        ProductStage.RELATIONSHIPS: 100,
        ProductStage.COMPLETED: 100
    }

    # Calculate progress
    completed_products = product_index - 1
    current_product_progress = stage_progress.get(stage, 0)

    # Overall progress = (completed products * 100 + current product progress) / total products
    overall_progress = (completed_products * 100 + current_product_progress) / total_products

    # Clamp to 0-100
    overall_progress = max(0, min(100, int(overall_progress)))

    logger.debug(f"Progress: Product {product_index}/{total_products}, Stage {stage.value} → {overall_progress}%")

