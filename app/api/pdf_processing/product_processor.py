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
    total_pages: Optional[int] = None
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
        # STAGE 1: Extract Product Pages
        # ========================================================================
        current_stage = ProductStage.EXTRACTION
        await product_tracker.update_product_stage(product_id, ProductStage.EXTRACTION)
        logger_instance.info(f"📄 [STAGE 1/{product_index}] Extracting pages for {product.name}...")

        from app.api.pdf_processing.stage_1_focused_extraction import extract_product_pages

        product_pages = await extract_product_pages(
            file_content=file_content,
            product=product,
            document_id=document_id,
            job_id=job_id,
            logger=logger_instance,
            total_pages=total_pages,
            pages_per_sheet=getattr(catalog, 'pages_per_sheet', 1)
        )

        await product_tracker.mark_stage_complete(
            product_id,
            ProductStage.EXTRACTION,
            {"pages_extracted": len(product_pages)}
        )
        pages_extracted = len(product_pages)
        logger_instance.info(f"✅ Extracted {pages_extracted} pages for {product.name}")

        # ========================================================================
        # STAGE 2: Create Text Chunks
        # ========================================================================
        current_stage = ProductStage.CHUNKING
        await product_tracker.update_product_stage(product_id, ProductStage.CHUNKING)
        logger_instance.info(f"📝 [STAGE 2/{product_index}] Creating chunks for {product.name}...")

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
            logger=logger_instance
        )

        chunks_created = chunk_result.get('chunks_created', 0)
        await product_tracker.mark_stage_complete(
            product_id,
            ProductStage.CHUNKING,
            {"chunks_created": chunks_created}
        )
        result.chunks_created = chunks_created
        logger_instance.info(f"✅ Created {chunks_created} chunks for {product.name}")

        # ========================================================================
        # STAGE 3: Process Images
        # ========================================================================
        current_stage = ProductStage.IMAGES
        await product_tracker.update_product_stage(product_id, ProductStage.IMAGES)
        logger_instance.info(f"🖼️  [STAGE 3/{product_index}] Processing images for {product.name}...")

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
        await product_tracker.mark_stage_complete(
            product_id,
            ProductStage.IMAGES,
            {
                "images_processed": images_processed,
                "images_material": image_result.get('images_material', 0),
                "images_non_material": image_result.get('images_non_material', 0)
            }
        )
        result.images_processed = images_processed
        logger_instance.info(f"✅ Processed {images_processed} images for {product.name}")

        # ========================================================================
        # STAGE 4: Create Product in Database
        # ========================================================================
        current_stage = ProductStage.CREATION
        await product_tracker.update_product_stage(product_id, ProductStage.CREATION)
        logger_instance.info(f"🏭 [STAGE 4/{product_index}] Creating product in database...")

        from app.api.pdf_processing.stage_4_products import create_single_product

        product_creation_result = await create_single_product(
            product=product,
            document_id=document_id,
            workspace_id=workspace_id,
            job_id=job_id,
            catalog=catalog,
            supabase=supabase,
            logger=logger_instance
        )

        product_db_id = product_creation_result.get('product_id')
        await product_tracker.mark_stage_complete(
            product_id,
            ProductStage.CREATION,
            {"product_db_id": product_db_id}
        )
        result.product_db_id = product_db_id
        logger_instance.info(f"✅ Created product in DB: {product_db_id}")

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

