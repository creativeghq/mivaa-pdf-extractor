"""
Product Progress Tracker Service

Tracks processing progress for individual products in the product-centric pipeline.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.schemas.product_progress import (
    ProductProgress,
    ProductStatus,
    ProductStage,
    ProductMetrics,
    ProductProcessingResult,
    JobProductSummary
)
from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class ProductProgressTracker:
    """
    Tracks processing progress for individual products.
    
    Provides methods to:
    - Initialize product tracking
    - Update product stage
    - Mark product complete/failed
    - Get product status
    - Get job summary
    """
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.supabase = get_supabase_client()
        self.table = "product_processing_status"
        logger.info(f"ProductProgressTracker initialized for job {job_id}")
    
    async def initialize_product(
        self,
        product_id: str,
        product_name: str,
        product_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProductProgress:
        """
        Initialize tracking for a product.
        
        Args:
            product_id: Unique product identifier
            product_name: Product name
            product_index: 1-based index in processing order
            metadata: Additional metadata
            
        Returns:
            ProductProgress object
        """
        try:
            product_progress = ProductProgress(
                product_id=product_id,
                product_name=product_name,
                product_index=product_index,
                status=ProductStatus.PENDING,
                metadata=metadata or {}
            )
            
            # Insert into database
            data = {
                "job_id": self.job_id,
                "product_id": product_id,
                "product_name": product_name,
                "product_index": product_index,
                "status": ProductStatus.PENDING.value,
                "stages_completed": [],
                "metrics": {},
                "created_at": datetime.utcnow().isoformat()
            }
            
            self.supabase.client.table(self.table).insert(data).execute()
            
            logger.info(f"✅ Initialized product tracking: {product_name} (index {product_index})")
            return product_progress
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize product {product_name}: {e}")
            raise
    
    async def update_product_stage(
        self,
        product_id: str,
        stage: ProductStage,
        status: ProductStatus = ProductStatus.PROCESSING
    ) -> None:
        """
        Update the current processing stage for a product.
        
        Args:
            product_id: Product identifier
            stage: Current processing stage
            status: Current status (default: PROCESSING)
        """
        try:
            update_data = {
                "status": status.value,
                "current_stage": stage.value,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Set started_at if this is the first stage
            if stage == ProductStage.EXTRACTION:
                update_data["started_at"] = datetime.utcnow().isoformat()
            
            self.supabase.client.table(self.table)\
                .update(update_data)\
                .eq("job_id", self.job_id)\
                .eq("product_id", product_id)\
                .execute()
            
            logger.debug(f"Updated product {product_id} to stage {stage.value}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update product stage: {e}")
            raise
    
    async def mark_stage_complete(
        self,
        product_id: str,
        stage: ProductStage,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark a stage as complete for a product.
        
        Args:
            product_id: Product identifier
            stage: Completed stage
            metrics: Stage-specific metrics to update
        """
        try:
            # Get current product status
            result = self.supabase.client.table(self.table)\
                .select("stages_completed, metrics")\
                .eq("job_id", self.job_id)\
                .eq("product_id", product_id)\
                .single()\
                .execute()
            
            if not result.data:
                logger.warning(f"Product {product_id} not found in tracking")
                return
            
            # Update stages_completed
            stages_completed = result.data.get("stages_completed", [])
            if stage.value not in stages_completed:
                stages_completed.append(stage.value)
            
            # Merge metrics
            current_metrics = result.data.get("metrics", {})
            if metrics:
                current_metrics.update(metrics)
            
            # Update database
            update_data = {
                "stages_completed": stages_completed,
                "metrics": current_metrics,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            self.supabase.client.table(self.table)\
                .update(update_data)\
                .eq("job_id", self.job_id)\
                .eq("product_id", product_id)\
                .execute()
            
            logger.debug(f"✅ Marked stage {stage.value} complete for product {product_id}")

        except Exception as e:
            logger.error(f"❌ Failed to mark stage complete: {e}")
            raise

    async def mark_product_complete(
        self,
        product_id: str,
        result: ProductProcessingResult
    ) -> None:
        """
        Mark a product as successfully completed.

        Args:
            product_id: Product identifier
            result: Processing result with metrics
        """
        try:
            update_data = {
                "status": ProductStatus.COMPLETED.value,
                "current_stage": ProductStage.COMPLETED.value,
                "completed_at": datetime.utcnow().isoformat(),
                "metrics": {
                    "chunks_created": result.chunks_created,
                    "images_processed": result.images_processed,
                    "relationships_created": result.relationships_created,
                    "processing_time_ms": result.processing_time_ms,
                    "memory_freed_mb": result.memory_freed_mb,
                    "product_db_id": result.product_db_id
                },
                "updated_at": datetime.utcnow().isoformat()
            }

            self.supabase.client.table(self.table)\
                .update(update_data)\
                .eq("job_id", self.job_id)\
                .eq("product_id", product_id)\
                .execute()

            logger.info(f"✅ Product {product_id} marked as COMPLETED")

        except Exception as e:
            logger.error(f"❌ Failed to mark product complete: {e}")
            raise

    async def mark_product_failed(
        self,
        product_id: str,
        error_message: str,
        error_stage: ProductStage
    ) -> None:
        """
        Mark a product as failed.

        Args:
            product_id: Product identifier
            error_message: Error description
            error_stage: Stage where error occurred
        """
        try:
            update_data = {
                "status": ProductStatus.FAILED.value,
                "error_message": error_message,
                "error_stage": error_stage.value,
                "error_timestamp": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }

            self.supabase.client.table(self.table)\
                .update(update_data)\
                .eq("job_id", self.job_id)\
                .eq("product_id", product_id)\
                .execute()

            logger.error(f"❌ Product {product_id} marked as FAILED at stage {error_stage.value}: {error_message}")

        except Exception as e:
            logger.error(f"❌ Failed to mark product as failed: {e}")
            raise

    async def get_product_status(self, product_id: str) -> Optional[ProductProgress]:
        """
        Get current status of a product.

        Args:
            product_id: Product identifier

        Returns:
            ProductProgress object or None if not found
        """
        try:
            result = self.supabase.client.table(self.table)\
                .select("*")\
                .eq("job_id", self.job_id)\
                .eq("product_id", product_id)\
                .single()\
                .execute()

            if not result.data:
                return None

            data = result.data
            return ProductProgress(
                product_id=data["product_id"],
                product_name=data["product_name"],
                product_index=data["product_index"],
                status=ProductStatus(data["status"]),
                current_stage=ProductStage(data["current_stage"]) if data.get("current_stage") else None,
                stages_completed=[ProductStage(s) for s in data.get("stages_completed", [])],
                error_message=data.get("error_message"),
                error_stage=ProductStage(data["error_stage"]) if data.get("error_stage") else None,
                error_timestamp=data.get("error_timestamp"),
                metrics=ProductMetrics(**data.get("metrics", {})),
                started_at=data.get("started_at"),
                completed_at=data.get("completed_at")
            )

        except Exception as e:
            logger.error(f"❌ Failed to get product status: {e}")
            return None

    async def get_job_summary(self) -> JobProductSummary:
        """
        Get summary of all products in the job.

        Returns:
            JobProductSummary with statistics and product list
        """
        try:
            # Get all products for this job
            result = self.supabase.client.table(self.table)\
                .select("*")\
                .eq("job_id", self.job_id)\
                .order("product_index")\
                .execute()

            products_data = result.data or []

            # Convert to ProductProgress objects
            products = []
            for data in products_data:
                products.append(ProductProgress(
                    product_id=data["product_id"],
                    product_name=data["product_name"],
                    product_index=data["product_index"],
                    status=ProductStatus(data["status"]),
                    current_stage=ProductStage(data["current_stage"]) if data.get("current_stage") else None,
                    stages_completed=[ProductStage(s) for s in data.get("stages_completed", [])],
                    error_message=data.get("error_message"),
                    error_stage=ProductStage(data["error_stage"]) if data.get("error_stage") else None,
                    error_timestamp=data.get("error_timestamp"),
                    metrics=ProductMetrics(**data.get("metrics", {})),
                    started_at=data.get("started_at"),
                    completed_at=data.get("completed_at")
                ))

            # Calculate statistics
            total = len(products)
            completed = sum(1 for p in products if p.status == ProductStatus.COMPLETED)
            failed = sum(1 for p in products if p.status == ProductStatus.FAILED)
            pending = sum(1 for p in products if p.status == ProductStatus.PENDING)
            processing = sum(1 for p in products if p.status == ProductStatus.PROCESSING)

            completion_pct = (completed / total * 100) if total > 0 else 0.0

            failed_ids = [p.product_id for p in products if p.status == ProductStatus.FAILED]

            return JobProductSummary(
                job_id=self.job_id,
                total_products=total,
                completed_products=completed,
                failed_products=failed,
                pending_products=pending,
                processing_products=processing,
                completion_percentage=round(completion_pct, 2),
                products=products,
                failed_product_ids=failed_ids
            )

        except Exception as e:
            logger.error(f"❌ Failed to get job summary: {e}")
            raise

    async def update_metrics(
        self,
        product_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Update metrics for a product.

        Args:
            product_id: Product identifier
            metrics: Metrics to update
        """
        try:
            # Get current metrics
            result = self.supabase.client.table(self.table)\
                .select("metrics")\
                .eq("job_id", self.job_id)\
                .eq("product_id", product_id)\
                .single()\
                .execute()

            if not result.data:
                logger.warning(f"Product {product_id} not found")
                return

            # Merge metrics
            current_metrics = result.data.get("metrics", {})
            current_metrics.update(metrics)

            # Update database
            self.supabase.client.table(self.table)\
                .update({
                    "metrics": current_metrics,
                    "updated_at": datetime.utcnow().isoformat()
                })\
                .eq("job_id", self.job_id)\
                .eq("product_id", product_id)\
                .execute()

            logger.debug(f"Updated metrics for product {product_id}")

        except Exception as e:
            logger.error(f"❌ Failed to update metrics: {e}")
            raise

