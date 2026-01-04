"""
Product Progress Tracking Schemas

Data models for tracking individual product processing in the product-centric pipeline.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


class ProductStage(str, Enum):
    """Processing stages for a single product"""
    PENDING = "pending"
    EXTRACTION = "extraction"  # Extract product pages
    CHUNKING = "chunking"  # Create text chunks
    IMAGES = "images"  # Process images
    CREATION = "creation"  # Create product in DB
    RELATIONSHIPS = "relationships"  # Link chunks/images
    COMPLETED = "completed"
    FAILED = "failed"


class ProductStatus(str, Enum):
    """Overall status of product processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProductMetrics(BaseModel):
    """Metrics for a processed product"""
    chunks_created: int = Field(default=0, description="Number of text chunks created")
    images_processed: int = Field(default=0, description="Number of images processed")
    images_material: int = Field(default=0, description="Number of material images")
    images_non_material: int = Field(default=0, description="Number of non-material images")
    relationships_created: int = Field(default=0, description="Number of relationships created")
    clip_embeddings_generated: int = Field(default=0, description="Number of CLIP embeddings")
    pages_extracted: int = Field(default=0, description="Number of pages extracted")
    processing_time_ms: Optional[int] = Field(default=None, description="Total processing time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "chunks_created": 15,
                "images_processed": 8,
                "images_material": 6,
                "images_non_material": 2,
                "relationships_created": 23,
                "clip_embeddings_generated": 30,
                "pages_extracted": 3,
                "processing_time_ms": 45000
            }
        }


class ProductProgress(BaseModel):
    """Progress tracking for a single product"""
    product_id: str = Field(..., description="Unique product identifier")
    product_name: str = Field(..., description="Product name")
    product_index: int = Field(..., description="1-based index in processing order")
    
    # Status
    status: ProductStatus = Field(default=ProductStatus.PENDING, description="Current status")
    current_stage: Optional[ProductStage] = Field(default=None, description="Current processing stage")
    stages_completed: List[ProductStage] = Field(default_factory=list, description="Completed stages")
    
    # Error tracking
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    error_stage: Optional[ProductStage] = Field(default=None, description="Stage where error occurred")
    error_timestamp: Optional[datetime] = Field(default=None, description="When error occurred")
    
    # Metrics
    metrics: ProductMetrics = Field(default_factory=ProductMetrics, description="Processing metrics")
    
    # Timestamps
    started_at: Optional[datetime] = Field(default=None, description="When processing started")
    completed_at: Optional[datetime] = Field(default=None, description="When processing completed")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('product_index')
    def validate_product_index(cls, v):
        if v < 1:
            raise ValueError('product_index must be >= 1')
        return v
    
    @property
    def is_complete(self) -> bool:
        """Check if product processing is complete"""
        return self.status == ProductStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Check if product processing failed"""
        return self.status == ProductStatus.FAILED
    
    @property
    def is_processing(self) -> bool:
        """Check if product is currently being processed"""
        return self.status == ProductStatus.PROCESSING
    
    @property
    def progress_percentage(self) -> int:
        """Calculate progress percentage based on completed stages"""
        total_stages = 5  # extraction, chunking, images, creation, relationships
        completed = len(self.stages_completed)
        return min(100, int((completed / total_stages) * 100))
    
    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "prod_abc123",
                "product_name": "Acme Widget Pro",
                "product_index": 1,
                "status": "processing",
                "current_stage": "images",
                "stages_completed": ["extraction", "chunking"],
                "metrics": {
                    "chunks_created": 15,
                    "images_processed": 3,
                    "pages_extracted": 2
                }
            }
        }


class ProductProcessingResult(BaseModel):
    """Result of processing a single product"""
    product_id: str
    product_name: str
    product_index: int
    success: bool
    
    # Results
    product_db_id: Optional[str] = Field(default=None, description="Database ID of created product")
    chunks_created: int = Field(default=0)
    images_processed: int = Field(default=0)
    relationships_created: int = Field(default=0)
    
    # Error info
    error: Optional[str] = Field(default=None)
    error_stage: Optional[ProductStage] = Field(default=None)
    
    # Timing
    processing_time_ms: Optional[int] = Field(default=None)
    
    # Memory stats
    memory_freed_mb: Optional[float] = Field(default=None, description="Memory freed after cleanup")
    
    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "prod_abc123",
                "product_name": "Acme Widget Pro",
                "product_index": 1,
                "success": True,
                "product_db_id": "uuid-here",
                "chunks_created": 15,
                "images_processed": 8,
                "relationships_created": 23,
                "processing_time_ms": 45000,
                "memory_freed_mb": 125.5
            }
        }


class JobProductSummary(BaseModel):
    """Summary of all products in a job"""
    job_id: str
    total_products: int
    completed_products: int
    failed_products: int
    pending_products: int
    processing_products: int
    completion_percentage: float
    
    # Lists
    products: List[ProductProgress] = Field(default_factory=list)
    failed_product_ids: List[str] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_xyz789",
                "total_products": 100,
                "completed_products": 75,
                "failed_products": 2,
                "pending_products": 20,
                "processing_products": 3,
                "completion_percentage": 75.0
            }
        }

