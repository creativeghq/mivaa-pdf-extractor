"""
Single Product Processor

Handles processing of a single product through all stages in the product-centric pipeline.
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
    pdf_result: Any,
    tracker: Any,  # Main job tracker
    product_tracker: ProductProgressTracker,
    checkpoint_recovery_service: Any,
    supabase: Any,
    config: Dict[str, Any],
    logger_instance: logging.Logger,
    total_pages: Optional[int] = None,
    temp_pdf_path: Optional[str] = None  # ✅ NEW: Reuse existing temp path
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
        pdf_result: PDF extraction result
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

        # Extract results
        product_pages = extraction_result['product_pages']
        layout_regions = extraction_result.get('layout_regions', [])
        layout_stats = extraction_result.get('layout_stats', {})

        await product_tracker.mark_stage_complete(
            product_id,
            ProductStage.EXTRACTION,
            {
                "pages_extracted": len(product_pages),
                "layout_regions_detected": len(layout_regions),
                "layout_stats": layout_stats
            }
        )
        pages_extracted = len(product_pages)
        logger_instance.info(
            f"✅ Extracted {pages_extracted} pages for {product.name}"
        )

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

            if settings.yolo_enabled and product_pages:
                # Initialize YOLO detector
                yolo_config = settings.get_yolo_config()
                detector = YoloLayoutDetector(config=yolo_config)

                # ✅ FIX: Reuse existing temp PDF
                used_temp_path = temp_pdf_path
                created_temp = False

                if not used_temp_path or not os.path.exists(used_temp_path):
                    logger_instance.info("      ⚠️ No temp_pdf_path provided, creating temporary copy...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(file_content)
                        used_temp_path = tmp_file.name
                        created_temp = True
                else:
                    logger_instance.info(f"      ♻️ Reusing existing temp PDF: {used_temp_path}")

                try:
                    # Detect layout regions for each product page
                    for page_idx in sorted(product_pages):
                        logger_instance.info(f"      Detecting regions on page {page_idx}...")
                        result_yolo = await detector.detect_layout_regions(used_temp_path, page_idx)

                        if result_yolo and result_yolo.regions:
                            layout_regions.extend(result_yolo.regions)
                            logger_instance.info(f"      ✅ Found {len(result_yolo.regions)} regions")
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

        chunk_result = await process_product_chunking(
            file_content=file_content,
            document_id=document_id,
            workspace_id=workspace_id,
            job_id=job_id,
            product=product,
            product_pages=product_pages,
            catalog=catalog,
            pdf_result=pdf_result,
            config=config,
            supabase=supabase,
            logger=logger_instance,
            product_id=None,
            temp_pdf_path=temp_pdf_path,  # ✅ NEW: Pass temp path
            layout_regions=layout_regions  # ✅ NEW: Pass layout regions directly
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

        image_result = await process_product_images(
            file_content=file_content,
            document_id=document_id,
            workspace_id=workspace_id,
            job_id=job_id,
            product=product,
            product_pages=product_pages,
            catalog=catalog,
            config=config,
            logger=logger_instance
        )

        images_processed = image_result.get('images_processed', 0)
        clip_embeddings = image_result.get('clip_embeddings_generated', 0)
        await product_tracker.mark_stage_complete(
            product_id,
            ProductStage.IMAGES,
            {
                "images_processed": images_processed,
                "images_material": image_result.get('images_material', 0),
                "images_non_material": image_result.get('images_non_material', 0),
                "clip_embeddings_generated": clip_embeddings
            }
        )
        result.images_processed = images_processed
        result.clip_embeddings_generated = clip_embeddings
        logger_instance.info(f"✅ Processed {images_processed} images for {product.name}")
        logger_instance.info(f"✅ Generated {clip_embeddings} CLIP embeddings for {product.name}")

        # ✅ FIX: Update tracker with CLIP embeddings count
        if tracker:
            await tracker.update_database_stats(
                images_stored=images_processed,
                clip_embeddings=clip_embeddings,
                image_embeddings=clip_embeddings,
                sync_to_db=True
            )
            # Log the actual tracker values to verify sync
            logger_instance.info(f"   📊 Updated tracker: {images_processed} images, {clip_embeddings} CLIP embeddings")
            logger_instance.info(f"   📊 Tracker totals: images_stored={tracker.images_stored}, clip_embeddings={tracker.clip_embeddings_generated}, image_embeddings={tracker.image_embeddings_generated}")

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

        # Get product_db_id from product_progress metadata (set in Stage 0)
        product_status = await product_tracker.get_product_status(product_id)
        product_db_id = product_status.metadata.get('product_db_id') if product_status else None

        if not product_db_id:
            raise Exception(f"Product DB ID not found for {product.name} - product should have been created in Stage 0")

        # Update product with extracted metadata from Stage 1
        extracted_metadata = extraction_result.get('metadata', {})
        if extracted_metadata:
            try:
                supabase.client.table('products')\
                    .update({'metadata': extracted_metadata})\
                    .eq('id', product_db_id)\
                    .execute()
                logger_instance.info(f"✅ Updated product metadata in DB: {product_db_id}")
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
                for region in layout_regions:
                    if region.type == 'TABLE':
                        table_regions.append(region)
                    
                    region_data.append({
                        'product_id': product_db_id,
                        'page_number': region.bbox.page,
                        'region_type': region.type,
                        'bbox_x': region.bbox.x,
                        'bbox_y': region.bbox.y,
                        'bbox_width': region.bbox.width,
                        'bbox_height': region.bbox.height,
                        'confidence': region.confidence,
                        'reading_order': region.reading_order,
                        'text_content': getattr(region, 'text_content', None),
                        'metadata': {'yolo_model': 'yolo-docparser'}
                    })

                supabase_client.client.table('product_layout_regions').insert(region_data).execute()
                logger_instance.info(f"   💾 Stored {len(region_data)} layout regions")

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

                    # Extract tables
                    all_tables = []
                    # Use existing temp path if available
                    tab_temp_path = temp_pdf_path
                    tab_created_temp = False
                    
                    if not tab_temp_path or not os.path.exists(tab_temp_path):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(file_content)
                            tab_temp_path = tmp_file.name
                            tab_created_temp = True

                    try:
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
                    finally:
                        if tab_created_temp and tab_temp_path and os.path.exists(tab_temp_path):
                            os.unlink(tab_temp_path)
            except Exception as e:
                logger_instance.error(f"❌ Failed to store layout/tables: {e}")

        await product_tracker.mark_stage_complete(
            product_id,
            ProductStage.CREATION,
            {"product_db_id": product_db_id}
        )
        result.product_db_id = product_db_id
        logger_instance.info(f"✅ Product updated in DB: {product_db_id}")

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
            product_pages=product_pages,
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
    - product_pages (Set[int]) - Page numbers for this product
    - pdf_result - Extracted text/images for this product
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

