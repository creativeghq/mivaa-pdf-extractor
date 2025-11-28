"""
import logging
Stage 5: Quality Enhancement

This module handles async quality enhancement using Claude validation.
"""

from typing import Dict, Any
from datetime import datetime
from app.schemas.jobs import ProcessingStage
from app.services.checkpoint_recovery_service import ProcessingStage as CheckpointStage
from app.services.droplet_scaler import droplet_scaler

logger = logging.getLogger(__name__)
from app.utils.circuit_breaker import CircuitBreakerError


async def process_stage_5_quality(
    document_id: str,
    job_id: str,
    workspace_id: str,
    catalog: Any,
    product_pages: set,
    products_created: int,
    images_processed: int,
    focused_extraction: bool,
    quality_validation_model: str,
    start_time: datetime,
    tracker: Any,
    checkpoint_recovery_service: Any,
    component_manager: Any,
    loaded_components: list,
    claude_breaker: Any,

) -> Dict[str, Any]:
    """
    Stage 5: Quality Enhancement
    
    Performs async quality enhancement using Claude validation for low-scoring images.
    
    Args:
        document_id: Unique document identifier
        job_id: Job identifier for tracking
        workspace_id: Workspace identifier
        catalog: Product catalog from Stage 0
        product_pages: Set of processed page numbers
        products_created: Number of products created
        images_processed: Number of images processed
        focused_extraction: Whether focused extraction was enabled
        start_time: Processing start time
        tracker: Job progress tracker
        checkpoint_recovery_service: Checkpoint service
        component_manager: Component manager for lazy loading
        loaded_components: List of loaded components
        claude_breaker: Circuit breaker for Claude API
        logger: Logger instance
        
    Returns:
        Dictionary containing:
        - result: Final processing result
        - validation_results: Claude validation results
    """
    from app.services.claude_validation_service import ClaudeValidationService
    
    logger.info("⚡ [STAGE 5] Quality Enhancement - Starting (Async)...")
    await tracker.update_stage(ProcessingStage.COMPLETED, stage_name="quality_enhancement")
    
    # Process Claude validation queue (for low-scoring images) with circuit breaker
    claude_service = ClaudeValidationService()
    try:
        validation_results = await claude_breaker.call(
            claude_service.process_validation_queue,
            document_id=document_id
        )
        logger.info(f"   Claude validation: {validation_results.get('validated', 0)} images validated")
        logger.info(f"   Average quality improvement: {validation_results.get('avg_improvement', 0):.2f}")
    except CircuitBreakerError as cb_error:
        logger.warning(f"⚠️ Claude validation skipped (circuit breaker open): {cb_error}")
        validation_results = {'validated': 0, 'avg_improvement': 0}
    
    await tracker._sync_to_database(stage="quality_enhancement")
    
    # NOTE: Cleanup moved to admin panel cron job
    
    # Mark job as complete
    result = {
        "document_id": document_id,
        "products_discovered": len(catalog.products),
        "products_created": products_created,
        "product_names": [p.name for p in catalog.products],
        "chunks_created": tracker.chunks_created,
        "images_processed": images_processed,
        "claude_validations": validation_results.get('validated', 0),
        "focused_extraction": focused_extraction,
        "pages_processed": len(product_pages),
        "pages_skipped": len([p for p in range(1, tracker.total_pages + 1) if p not in product_pages]),
        "confidence_score": catalog.confidence_score
    }
    
    await tracker.complete_job(result=result)
    
    # Create COMPLETED checkpoint
    await checkpoint_recovery_service.create_checkpoint(
        job_id=job_id,
        stage=CheckpointStage.COMPLETED,
        data={
            "document_id": document_id,
            "products_created": products_created,
            "chunks_created": tracker.chunks_created,
            "images_processed": images_processed
        },
        metadata={
            "processing_time": (datetime.utcnow() - start_time).total_seconds(),
            "confidence_score": catalog.confidence_score,
            "focused_extraction": focused_extraction,
            "pages_processed": len(product_pages)
        }
    )
    logger.info(f"✅ Created COMPLETED checkpoint for job {job_id}")
    
    logger.info("=" * 80)
    logger.info(f"✅ [PRODUCT DISCOVERY PIPELINE] COMPLETED")
    logger.info(f"   Products: {products_created}")
    logger.info(f"   Chunks: {tracker.chunks_created}")
    logger.info(f"   Images: {images_processed}")
    logger.info("=" * 80)
    
    # LAZY LOADING: Unload all loaded components after successful completion
    logger.info("🧹 Unloading all loaded components...")
    for component_name in loaded_components:
        try:
            await component_manager.unload(component_name)
            logger.info(f"✅ Unloaded {component_name}")
        except Exception as unload_error:
            logger.warning(f"⚠️ Failed to unload {component_name}: {unload_error}")
    
    # Force garbage collection
    import gc
    gc.collect()
    logger.info("✅ All components unloaded, memory freed")

    # AUTO-SCALE: Scale down droplet after PDF processing completes
    logger.info("🔄 Scaling down droplet to save costs...")
    try:
        scale_down_success = await droplet_scaler.scale_down_after_processing(force=False)
        if scale_down_success:
            logger.info("✅ Droplet scaled down successfully - saving money! 💰")
        else:
            logger.info("ℹ️ Droplet not scaled down (may have active jobs or already at small size)")
    except Exception as e:
        logger.warning(f"⚠️ Failed to scale down droplet: {e}")
        # Don't fail the job if scaling fails

    # EVENT-BASED CLEANUP: Release temp PDF file and cleanup
    from app.utils.resource_manager import get_resource_manager
    resource_manager = get_resource_manager()

    # Release the temp PDF file
    await resource_manager.release_resource(f"temp_pdf_{document_id}", job_id)

    # Cleanup all ready resources
    cleaned_count = await resource_manager.cleanup_ready_resources()
    logger.info(f"✅ Cleaned up {cleaned_count} temporary files")

    return {
        "result": result,
        "validation_results": validation_results
    }

