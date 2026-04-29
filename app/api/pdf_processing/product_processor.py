"""
Single Product Processor

Handles processing of a single product through all stages in the product-centric pipeline.
Includes checkpoint creation for recovery and visibility.
"""

import gc
import logging
from typing import Dict, Any, Optional, Set
from datetime import datetime

from app.schemas.product_progress import (
    ProductStage,
    ProductStatus,
    ProductProcessingResult,
    ProductMetrics
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
    total_pages: Optional[int] = None,
    temp_pdf_path: Optional[str] = None
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
    try:
        prior_status = await product_tracker.get_product_status(product_id)
        prior_db_id = (prior_status.metadata or {}).get('product_db_id') if prior_status else None
        if prior_status and prior_status.stages_completed:
            prior_stages.update(prior_status.stages_completed)
        # Also peek at job stage_history for per-product checkpoint events
        # tied to THIS product_index (catches resumes where product_tracker
        # state was wiped on the previous restart).
        try:
            sb_resp = supabase.client.table('background_jobs') \
                .select('stage_history') \
                .eq('id', job_id).single().execute()
            for entry in (sb_resp.data or {}).get('stage_history', []) or []:
                ed = entry.get('data') or {}
                if str(ed.get('product_index')) == str(product_index):
                    prior_stages.add(entry.get('stage'))
        except Exception:
            pass

        # Verify with database state — checkpoints alone aren't authoritative
        # because a stage might have been MID-INSERT when the worker died.
        # If chunks/images for this product already exist in DB, treat the
        # corresponding stage as done.
        try:
            if prior_db_id:
                existing_chunks = supabase.client.table('document_chunks') \
                    .select('id', count='exact') \
                    .eq('document_id', document_id) \
                    .eq('product_id', prior_db_id) \
                    .execute()
                if (existing_chunks.count or 0) > 0:
                    prior_stages.add('chunks_created')
                existing_imgs = supabase.client.table('document_images') \
                    .select('id', count='exact') \
                    .eq('document_id', document_id) \
                    .eq('product_id', prior_db_id) \
                    .execute()
                if (existing_imgs.count or 0) > 0:
                    prior_stages.add('images_extracted')
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

    try:
        # ========================================================================
        # STAGE 1: Extract Product Pages + YOLO Layout Detection
        # ========================================================================
        current_stage = ProductStage.EXTRACTION
        await product_tracker.update_product_stage(product_id, ProductStage.EXTRACTION)
        logger_instance.info(f"📄 [STAGE 1/{product_index}] Extracting pages for {product.name}...")

        # UPDATE PROGRESS: Update tracker at start of each stage
        if tracker:
            tracker.current_step = f"Stage 1: Extracting pages for {product.name}"
            await tracker.update_heartbeat()

        from app.api.pdf_processing.stage_1_focused_extraction import extract_product_pages

        # ✅ NEW: extract_product_pages now returns a dict with layout detection results
        # Pass catalog for spread layout info (physical page -> PDF page mapping)
        extraction_result = await extract_product_pages(
            file_content=file_content,
            product=product,
            document_id=document_id,
            job_id=job_id,
            logger=logger_instance,
            total_pages=total_pages,
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

        # ========================================================================
        # STAGE 1.5: YOLO Layout Detection
        # ========================================================================
        logger_instance.info(f"🎯 [STAGE 1.5/{product_index}] Running YOLO layout detection...")
        layout_regions = []

        try:
            from app.services.pdf.yolo_layout_detector import YoloLayoutDetector
            from app.config import get_settings
            import os
            import tempfile

            settings = get_settings()

            if settings.yolo_enabled and physical_pages:
                # Initialize YOLO detector
                yolo_config = settings.get_yolo_config()
                detector = YoloLayoutDetector(config=yolo_config)

                # P2-6: STRICT temp PDF reuse. The orchestrator always
                # provides temp_pdf_path; if it's missing or gone, that's
                # a bug — fail loudly instead of silently re-creating.
                used_temp_path = temp_pdf_path
                created_temp = False

                if not used_temp_path or not os.path.exists(used_temp_path):
                    raise RuntimeError(
                        f"YOLO stage: temp PDF not available "
                        f"(path={used_temp_path!r}). Orchestrator must keep "
                        f"the temp PDF alive for the duration of the job."
                    )
                logger_instance.info(f"      ♻️ Reusing existing temp PDF: {used_temp_path}")

                try:
                    # Detect layout regions for each physical page IN PARALLEL.
                    # YOLO endpoint has yolo_concurrency=48 cap (4 replicas).
                    # The previous sequential loop wasted ~80% of YOLO capacity
                    # — 6 pages × ~3s sequential = ~18s; parallel with replica
                    # fan-out → ~3-5s total per product. Bound to
                    # min(len(pages), yolo_concurrency / max_concurrent_products)
                    # so 4 products × 12 pages don't all stampede at once.
                    import asyncio as _asyncio_yolo
                    from app.config import get_settings as _get_settings_for_yolo
                    _yolo_settings = _get_settings_for_yolo()
                    _per_product_yolo_cap = max(
                        2,
                        _yolo_settings.yolo_concurrency // max(1, _yolo_settings.max_concurrent_products)
                    )
                    yolo_sem = _asyncio_yolo.Semaphore(_per_product_yolo_cap)

                    async def _detect_one_page(physical_page: int):
                        if has_spread_layout and physical_page in physical_to_pdf_map:
                            pdf_idx, _position = physical_to_pdf_map[physical_page]
                        else:
                            pdf_idx = physical_page - 1
                        async with yolo_sem:
                            try:
                                yolo_result = await detector.detect_layout_regions(used_temp_path, pdf_idx)
                            except Exception as page_err:
                                logger_instance.warning(
                                    f"      ⚠️ YOLO failed for page {physical_page}: {page_err}"
                                )
                                return None
                            if yolo_result and yolo_result.regions:
                                for region in yolo_result.regions:
                                    region.bbox.page = physical_page
                                return yolo_result.regions
                            return None

                    sorted_pages = sorted(physical_pages)
                    logger_instance.info(
                        f"      Detecting regions on {len(sorted_pages)} pages "
                        f"(parallel, cap={_per_product_yolo_cap})..."
                    )
                    page_results = await _asyncio_yolo.gather(
                        *(_detect_one_page(p) for p in sorted_pages),
                        return_exceptions=False,
                    )
                    for regions in page_results:
                        if regions:
                            layout_regions.extend(regions)
                    logger_instance.info(
                        f"      ✅ YOLO complete: {len(layout_regions)} regions across {len(sorted_pages)} pages"
                    )
                finally:
                    # Only delete if we created it locally in this stage
                    if created_temp and used_temp_path and os.path.exists(used_temp_path):
                        os.unlink(used_temp_path)
            else:
                logger_instance.info("   ⚠️ YOLO disabled or no pages to process")

        except Exception as e:
            logger_instance.error(f"   ❌ YOLO layout detection failed: {e}")
            logger_instance.info("   Continuing without layout detection...")

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
            logger_instance.info(f"✅ Created {chunks_created} chunks for {product.name} ({embeddings_generated} text embeddings)")
        await product_tracker.mark_stage_complete(
            product_id,
            ProductStage.CHUNKING,
            {"chunks_created": chunks_created, "text_embeddings_generated": embeddings_generated, "layout_aware": True}
        )
        result.chunks_created = chunks_created

        # ✅ FIX: Update tracker with text embeddings count
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
                    f"Qwen + SLIG + Voyage calls skipped"
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
                physical_pages=physical_pages,  # ✅ FIXED: Now using physical pages (1-based)
                catalog=catalog,
                config=config,
                logger=logger_instance,
                layout_regions=layout_regions,  # ✅ NEW: Pass YOLO layout regions for bbox data
                tracker=tracker,  # ✅ NEW: Per-image progress events visible in admin UI
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
                "vision_analysis_qwen": vector_stats.get('vision_analysis_qwen', 0),
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
        logger_instance.info(f"✅ Generated {clip_embeddings} CLIP embeddings for {product.name}")

        # ✅ FIX: Update tracker with CLIP embeddings count
        # 2026-04-10: kwarg renamed images_stored → images_extracted to match
        # ProgressTracker.update_database_stats() signature (the tracker uses
        # `images_extracted` everywhere; `images_stored` was a stale name from
        # an older schema). Was producing
        # `update_database_stats() got an unexpected keyword argument 'images_stored'`
        # → Stage 4 product persistence crash → checkpoint validation reports
        # 0 products / 0 chunks / 0 images.
        if tracker:
            await tracker.update_database_stats(
                images_extracted=images_processed,
                clip_embeddings=clip_embeddings,
                image_embeddings=clip_embeddings,
                sync_to_db=True
            )
            # Log the actual tracker values to verify sync
            logger_instance.info(f"   📊 Updated tracker: {images_processed} images, {clip_embeddings} CLIP embeddings")
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

        # ✅ CHECKPOINT: IMAGE_EMBEDDINGS_GENERATED - CLIP embeddings complete
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

        # Update product with extracted metadata from Stage 1
        # FIXED: MERGE with existing metadata instead of REPLACING
        extracted_metadata = extraction_result.get('metadata', {})
        if extracted_metadata:
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

                # Deep merge: extracted_metadata takes priority but preserves existing fields
                merged_metadata = {**existing_metadata}
                for key, value in extracted_metadata.items():
                    if value is not None:  # Only update if new value is not None
                        if key in merged_metadata and isinstance(merged_metadata[key], dict) and isinstance(value, dict):
                            # Merge nested dicts
                            merged_metadata[key] = {**merged_metadata[key], **value}
                        elif key in merged_metadata and isinstance(merged_metadata[key], list) and isinstance(value, list):
                            # Merge lists (deduplicate)
                            existing_set = set(merged_metadata[key]) if all(isinstance(x, (str, int, float)) for x in merged_metadata[key]) else merged_metadata[key]
                            new_set = set(value) if all(isinstance(x, (str, int, float)) for x in value) else value
                            if isinstance(existing_set, set) and isinstance(new_set, set):
                                merged_metadata[key] = sorted(list(existing_set | new_set))
                            else:
                                merged_metadata[key] = merged_metadata[key] + [v for v in value if v not in merged_metadata[key]]
                        else:
                            merged_metadata[key] = value

                supabase.client.table('products')\
                    .update({'metadata': merged_metadata})\
                    .eq('id', product_db_id)\
                    .execute()
                logger_instance.info(f"✅ Merged and updated product metadata in DB: {product_db_id} ({len(merged_metadata)} fields)")
            except Exception as e:
                logger_instance.error(f"❌ Failed to update product metadata: {e}")

        # 4b. Store layout regions and extract tables
        if layout_regions and product_db_id:
            try:
                from app.services.pdf.table_extraction import TableExtractor
                from app.services.core.supabase_client import get_supabase_client
                supabase_client = get_supabase_client()

                # Store layout regions
                region_data = []
                table_regions = []
                dropped_invalid_bbox = 0
                for region in layout_regions:
                    if region.type == 'TABLE':
                        table_regions.append(region)

                    # Normalise YOLO bbox to satisfy product_layout_regions
                    # CHECK constraints (width > 0, height > 0). YOLO occasionally
                    # emits degenerate regions — we saw one with negative height on
                    # page 24 of harmony-signature-book-24-25.pdf that crashed the
                    # whole insert batch. Clamp negatives to abs(); drop anything
                    # that remains zero-size.
                    raw_x = float(region.bbox.x)
                    raw_y = float(region.bbox.y)
                    raw_w = float(region.bbox.width)
                    raw_h = float(region.bbox.height)
                    w = abs(raw_w)
                    h = abs(raw_h)
                    if w <= 0 or h <= 0:
                        dropped_invalid_bbox += 1
                        logger_instance.warning(
                            f"   ⚠️ Dropping degenerate YOLO region on page {region.bbox.page}: "
                            f"x={raw_x}, y={raw_y}, w={raw_w}, h={raw_h}"
                        )
                        continue
                    if raw_w < 0 or raw_h < 0:
                        logger_instance.warning(
                            f"   ⚠️ Clamped inverted YOLO bbox on page {region.bbox.page}: "
                            f"(w={raw_w}, h={raw_h}) → (w={w}, h={h})"
                        )

                    region_data.append({
                        'product_id': product_db_id,
                        'page_number': region.bbox.page,
                        'region_type': region.type,
                        'bbox_x': raw_x,
                        'bbox_y': raw_y,
                        'bbox_width': w,
                        'bbox_height': h,
                        'confidence': region.confidence,
                        'reading_order': region.reading_order,
                        'text_content': getattr(region, 'text_content', None),
                        'metadata': {'yolo_model': 'yolo-docparser'}
                    })

                if region_data:
                    supabase_client.client.table('product_layout_regions').insert(region_data).execute()
                    logger_instance.info(
                        f"   💾 Stored {len(region_data)} layout regions"
                        + (f" ({dropped_invalid_bbox} degenerate dropped)" if dropped_invalid_bbox else "")
                    )
                elif dropped_invalid_bbox:
                    logger_instance.warning(
                        f"   ⚠️ All {dropped_invalid_bbox} YOLO regions were degenerate — nothing stored"
                    )

                # Extract tables if TABLE regions found
                if table_regions:
                    logger_instance.info(f"   📊 Extracting {len(table_regions)} tables...")
                    extractor = TableExtractor()
                    
                    # Group by page
                    tables_by_page = {}
                    for region in table_regions:
                        p_num = region.bbox.page
                        if p_num not in tables_by_page:
                            tables_by_page[p_num] = []
                        tables_by_page[p_num].append(region)

                    # Extract tables — P2-6: strict reuse of orchestrator temp PDF
                    all_tables = []
                    tab_temp_path = temp_pdf_path
                    if not tab_temp_path or not os.path.exists(tab_temp_path):
                        raise RuntimeError(
                            f"Table extraction stage: temp PDF not available "
                            f"(path={tab_temp_path!r})."
                        )

                    for page_num, regions in tables_by_page.items():
                        tables = extractor.extract_tables_from_page(
                            pdf_path=tab_temp_path,
                            page_number=page_num,
                            table_regions=regions
                        )
                        all_tables.extend(tables)

                    if all_tables:
                        stored_count = await extractor.store_tables_in_database(
                            product_id=product_db_id,
                            tables=all_tables,
                            supabase_client=supabase_client
                        )
                        logger_instance.info(f"   ✅ Stored {stored_count} tables")
            except Exception as e:
                logger_instance.error(f"❌ Failed to store layout/tables: {e}")

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
                "tables_extracted": len(table_regions) if 'table_regions' in locals() else 0
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
            physical_pages=set(physical_pages),  # ✅ FIXED: Using physical_pages (1-based)
            logger=logger_instance
        )

        relationships_created = linking_result.get('relationships_created', 0)
        await product_tracker.mark_stage_complete(
            product_id,
            ProductStage.RELATIONSHIPS,
            {"relationships_created": relationships_created}
        )
        result.relationships_created = relationships_created
        logger_instance.info(f"✅ Created {relationships_created} relationships")

        # ✅ FIX: Update tracker with relationships count
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

