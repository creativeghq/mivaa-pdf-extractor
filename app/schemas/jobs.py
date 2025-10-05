"""
Job processing and async task Pydantic schemas.

This module contains schemas for background job management,
async processing status, and task queuing.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

try:
    # Try Pydantic v2 first
    from pydantic import BaseModel, Field, field_validator, model_validator
except ImportError:
    # Fall back to Pydantic v1
    from pydantic import BaseModel, Field, validator as field_validator, root_validator as model_validator

from .common import BaseResponse, ProcessingStatus, PaginationParams


class JobStatus(str, Enum):
    """Extended job status enumeration."""
    
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    RETRYING = "retrying"


class JobPriority(str, Enum):
    """Job priority levels."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class JobType(str, Enum):
    """Types of background jobs."""
    
    DOCUMENT_PROCESSING = "document_processing"
    IMAGE_ANALYSIS = "image_analysis"
    BATCH_PROCESSING = "batch_processing"
    EMBEDDING_GENERATION = "embedding_generation"
    SEARCH_INDEXING = "search_indexing"
    DATA_EXPORT = "data_export"
    CLEANUP = "cleanup"


class JobCreateRequest(BaseModel):
    """Request model for creating a background job."""
    
    job_type: JobType = Field(..., description="Type of job to create")
    priority: JobPriority = Field(JobPriority.NORMAL, description="Job priority level")
    
    # Job parameters
    parameters: Dict[str, Any] = Field(..., description="Job-specific parameters")
    
    # Scheduling options
    schedule_at: Optional[datetime] = Field(None, description="Schedule job for future execution")
    retry_count: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    timeout_seconds: Optional[int] = Field(None, description="Job timeout in seconds")
    
    # Metadata
    description: Optional[str] = Field(None, description="Human-readable job description")
    tags: List[str] = Field(default_factory=list, description="Job tags for organization")
    
    class Config:
        schema_extra = {
            "example": {
                "job_type": "document_processing",
                "priority": "normal",
                "parameters": {
                    "document_url": "https://example.com/document.pdf",
                    "extract_images": True,
                    "generate_embeddings": True
                },
                "retry_count": 3,
                "timeout_seconds": 600,
                "description": "Process research paper with full analysis",
                "tags": ["research", "batch"]
            }
        }


class JobProgress(BaseModel):
    """Job progress information."""
    
    current_step: str = Field(..., description="Current processing step")
    total_steps: int = Field(..., description="Total number of steps")
    completed_steps: int = Field(..., description="Number of completed steps")
    progress_percentage: float = Field(..., ge=0.0, le=100.0, description="Progress percentage")
    
    # Detailed progress info
    step_details: Optional[Dict[str, Any]] = Field(None, description="Step-specific details")
    estimated_remaining_seconds: Optional[int] = Field(None, description="Estimated time remaining")
    
    @model_validator(mode='after')
    def calculate_progress(self):
        completed = getattr(self, 'completed_steps', 0)
        total = getattr(self, 'total_steps', 1)
        self.progress_percentage = (completed / total) * 100 if total > 0 else 0
        return self


class JobResult(BaseModel):
    """Job execution result."""
    
    success: bool = Field(..., description="Whether job completed successfully")
    result_data: Optional[Dict[str, Any]] = Field(None, description="Job result data")
    
    # Output information
    output_files: List[str] = Field(default_factory=list, description="Generated output files")
    artifacts: List[str] = Field(default_factory=list, description="Generated artifacts")
    
    # Performance metrics
    processing_time_seconds: float = Field(..., description="Total processing time")
    memory_usage_mb: Optional[float] = Field(None, description="Peak memory usage")
    cpu_time_seconds: Optional[float] = Field(None, description="CPU time used")
    
    # Error information (if failed)
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")


class JobResponse(BaseResponse):
    """Response model for job operations."""
    
    job_id: str = Field(..., description="Unique job identifier")
    job_type: JobType = Field(..., description="Type of job")
    status: JobStatus = Field(..., description="Current job status")
    priority: JobPriority = Field(..., description="Job priority")
    
    # Timing information
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    
    # Progress tracking
    progress: Optional[JobProgress] = Field(None, description="Job progress information")
    
    # Job configuration
    parameters: Dict[str, Any] = Field(..., description="Job parameters")
    retry_count: int = Field(..., description="Maximum retry attempts")
    current_retry: int = Field(0, description="Current retry attempt")
    
    # Results (if completed)
    result: Optional[JobResult] = Field(None, description="Job result if completed")
    
    # Metadata
    description: Optional[str] = Field(None, description="Job description")
    tags: List[str] = Field(default_factory=list, description="Job tags")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "job_id": "job_123e4567-e89b-12d3-a456-426614174000",
                "job_type": "document_processing",
                "status": "running",
                "priority": "normal",
                "created_at": "2024-07-26T18:00:00Z",
                "started_at": "2024-07-26T18:00:05Z",
                "progress": {
                    "current_step": "Extracting text",
                    "total_steps": 5,
                    "completed_steps": 2,
                    "progress_percentage": 40.0,
                    "estimated_remaining_seconds": 120
                },
                "parameters": {
                    "document_url": "https://example.com/doc.pdf",
                    "extract_images": True
                },
                "retry_count": 3,
                "current_retry": 0,
                "description": "Process research paper",
                "tags": ["research"],
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class JobListRequest(BaseModel):
    """Request model for listing jobs with filters."""
    
    # Pagination
    pagination: PaginationParams = Field(default_factory=PaginationParams)
    
    # Filters
    job_types: Optional[List[JobType]] = Field(None, description="Filter by job types")
    statuses: Optional[List[JobStatus]] = Field(None, description="Filter by job statuses")
    priorities: Optional[List[JobPriority]] = Field(None, description="Filter by priorities")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    
    # Date filters
    created_after: Optional[datetime] = Field(None, description="Filter jobs created after this date")
    created_before: Optional[datetime] = Field(None, description="Filter jobs created before this date")
    
    # Search
    search_query: Optional[str] = Field(None, description="Search in job descriptions")
    
    class Config:
        schema_extra = {
            "example": {
                "pagination": {
                    "page": 1,
                    "page_size": 20,
                    "sort_by": "created_at",
                    "sort_order": "desc"
                },
                "job_types": ["document_processing", "image_analysis"],
                "statuses": ["running", "completed"],
                "priorities": ["normal", "high"],
                "tags": ["research"],
                "created_after": "2024-07-26T00:00:00Z"
            }
        }


class JobListItem(BaseModel):
    """Simplified job information for list responses."""
    
    job_id: str = Field(..., description="Job identifier")
    job_type: JobType = Field(..., description="Job type")
    status: JobStatus = Field(..., description="Job status")
    priority: JobPriority = Field(..., description="Job priority")
    
    # Timing
    created_at: datetime = Field(..., description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    # Progress
    progress_percentage: float = Field(0.0, description="Progress percentage")
    current_step: Optional[str] = Field(None, description="Current processing step")
    
    # Basic info
    description: Optional[str] = Field(None, description="Job description")
    tags: List[str] = Field(default_factory=list, description="Job tags")
    
    # Results summary
    success: Optional[bool] = Field(None, description="Success status if completed")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class JobListResponse(BaseResponse):
    """Response model for job listing."""
    
    jobs: List[JobListItem] = Field(..., description="List of jobs")
    total_count: int = Field(..., description="Total number of jobs")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    
    # Summary statistics
    status_counts: Dict[str, int] = Field(default_factory=dict, description="Count by status")
    type_counts: Dict[str, int] = Field(default_factory=dict, description="Count by type")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "jobs": [
                    {
                        "job_id": "job_123",
                        "job_type": "document_processing",
                        "status": "completed",
                        "priority": "normal",
                        "created_at": "2024-07-26T18:00:00Z",
                        "completed_at": "2024-07-26T18:05:00Z",
                        "progress_percentage": 100.0,
                        "description": "Process research paper",
                        "success": True
                    }
                ],
                "total_count": 1,
                "page": 1,
                "page_size": 20,
                "status_counts": {
                    "completed": 1,
                    "running": 0,
                    "failed": 0
                },
                "type_counts": {
                    "document_processing": 1
                },
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class JobUpdateRequest(BaseModel):
    """Request model for updating job properties."""
    
    priority: Optional[JobPriority] = Field(None, description="Update job priority")
    description: Optional[str] = Field(None, description="Update job description")
    tags: Optional[List[str]] = Field(None, description="Update job tags")
    
    class Config:
        schema_extra = {
            "example": {
                "priority": "high",
                "description": "Updated: High priority research paper processing",
                "tags": ["research", "urgent", "ai"]
            }
        }


class JobActionRequest(BaseModel):
    """Request model for job actions (pause, resume, cancel)."""
    
    action: str = Field(..., pattern="^(pause|resume|cancel|retry)$", description="Action to perform")
    reason: Optional[str] = Field(None, description="Reason for the action")
    
    class Config:
        schema_extra = {
            "example": {
                "action": "pause",
                "reason": "System maintenance scheduled"
            }
        }


class JobActionResponse(BaseResponse):
    """Response model for job actions."""
    
    job_id: str = Field(..., description="Job identifier")
    action: str = Field(..., description="Action performed")
    previous_status: JobStatus = Field(..., description="Previous job status")
    new_status: JobStatus = Field(..., description="New job status")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Job paused successfully",
                "job_id": "job_123e4567-e89b-12d3-a456-426614174000",
                "action": "pause",
                "previous_status": "running",
                "new_status": "paused",
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class BatchJobRequest(BaseModel):
    """Request model for creating multiple jobs."""
    
    jobs: List[JobCreateRequest] = Field(..., min_items=1, max_items=100, description="List of jobs to create")
    batch_priority: Optional[JobPriority] = Field(None, description="Override priority for all jobs")
    batch_tags: List[str] = Field(default_factory=list, description="Additional tags for all jobs")
    
    class Config:
        schema_extra = {
            "example": {
                "jobs": [
                    {
                        "job_type": "document_processing",
                        "parameters": {"document_url": "https://example.com/doc1.pdf"},
                        "description": "Process document 1"
                    },
                    {
                        "job_type": "document_processing", 
                        "parameters": {"document_url": "https://example.com/doc2.pdf"},
                        "description": "Process document 2"
                    }
                ],
                "batch_priority": "high",
                "batch_tags": ["batch_2024_07_26"]
            }
        }


class BatchJobResponse(BaseResponse):
    """Response model for batch job creation."""
    
    batch_id: str = Field(..., description="Batch identifier")
    created_jobs: List[str] = Field(..., description="List of created job IDs")
    total_jobs: int = Field(..., description="Total number of jobs created")
    
    # Batch summary
    estimated_total_time_seconds: Optional[int] = Field(None, description="Estimated total processing time")
    batch_priority: JobPriority = Field(..., description="Batch priority level")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Batch jobs created successfully",
                "batch_id": "batch_789e0123-e89b-12d3-a456-426614174000",
                "created_jobs": [
                    "job_123e4567-e89b-12d3-a456-426614174000",
                    "job_456e7890-e89b-12d3-a456-426614174001"
                ],
                "total_jobs": 2,
                "estimated_total_time_seconds": 600,
                "batch_priority": "high",
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class JobStatsResponse(BaseResponse):
    """Response model for job statistics."""
    
    # Current statistics
    total_jobs: int = Field(..., description="Total number of jobs")
    active_jobs: int = Field(..., description="Currently active jobs")
    queued_jobs: int = Field(..., description="Jobs waiting in queue")
    
    # Status breakdown
    status_breakdown: Dict[str, int] = Field(..., description="Jobs by status")
    type_breakdown: Dict[str, int] = Field(..., description="Jobs by type")
    priority_breakdown: Dict[str, int] = Field(..., description="Jobs by priority")
    
    # Performance metrics
    average_processing_time_seconds: float = Field(..., description="Average job processing time")
    success_rate_percentage: float = Field(..., description="Job success rate")
    
    # System health
    queue_health: str = Field(..., description="Queue health status")
    worker_count: int = Field(..., description="Number of active workers")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "total_jobs": 150,
                "active_jobs": 5,
                "queued_jobs": 2,
                "status_breakdown": {
                    "completed": 140,
                    "running": 5,
                    "queued": 2,
                    "failed": 3
                },
                "type_breakdown": {
                    "document_processing": 120,
                    "image_analysis": 20,
                    "batch_processing": 10
                },
                "priority_breakdown": {
                    "normal": 130,
                    "high": 15,
                    "urgent": 5
                },
                "average_processing_time_seconds": 245.7,
                "success_rate_percentage": 98.0,
                "queue_health": "healthy",
                "worker_count": 4,
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }