"""
Common Pydantic schemas used across all API endpoints.

This module contains base response models, pagination, error handling,
and other shared data structures.

NOTE: For new endpoints, use UnifiedApiResponse from unified_response.py
instead of BaseResponse. BaseResponse is kept for backward compatibility.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

# Import unified response for convenience
from .unified_response import (
    UnifiedApiResponse,
    ErrorDetail,
    ResponseMetadata,
    ErrorCode
)

# Generic type for paginated responses
T = TypeVar('T')


class BaseResponse(BaseModel):
    """Base response model with common fields."""
    
    success: bool = Field(True, description="Indicates if the request was successful")
    message: Optional[str] = Field(None, description="Optional message or description")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseResponse):
    """Error response model with detailed error information."""
    
    success: bool = Field(False, description="Always false for error responses")
    error_code: str = Field(..., description="Machine-readable error code")
    error_type: str = Field(..., description="Type of error (validation, processing, etc.)")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    trace_id: Optional[str] = Field(None, description="Request trace ID for debugging")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "message": "PDF processing failed",
                "error_code": "PDF_PROCESSING_ERROR",
                "error_type": "ProcessingError",
                "details": {
                    "file_size": "50MB",
                    "max_allowed": "25MB"
                },
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class SuccessResponse(BaseResponse):
    """Success response model with optional data payload."""

    success: bool = Field(True, description="Always true for success responses")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data payload")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {
                    "result": "example_value",
                    "count": 42
                },
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""
    
    page: int = Field(1, ge=1, description="Page number (1-based)")
    page_size: int = Field(20, ge=1, le=100, description="Number of items per page")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: Optional[str] = Field("desc", pattern="^(asc|desc)$", description="Sort order")
    
    @field_validator('sort_order')
    @classmethod
    def validate_sort_order(cls, v):
        if v and v not in ['asc', 'desc']:
            raise ValueError('sort_order must be either "asc" or "desc"')
        return v


class PaginationResponse(BaseModel, Generic[T]):
    """Generic pagination response wrapper."""
    
    items: List[T] = Field(..., description="List of items for current page")
    total_count: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")
    
    @model_validator(mode='after')
    def calculate_pagination_fields(self):
        # Calculate total_pages
        total_count = getattr(self, 'total_count', 0)
        page_size = getattr(self, 'page_size', 1)
        self.total_pages = max(1, (total_count + page_size - 1) // page_size)

        # Calculate has_next
        page = getattr(self, 'page', 1)
        self.has_next = page < self.total_pages

        # Calculate has_previous
        self.has_previous = page > 1

        return self


class ProcessingStatus(str, Enum):
    """Status enumeration for processing jobs."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HealthStatus(str, Enum):
    """Health check status enumeration."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseResponse):
    """Health check response model."""
    
    status: HealthStatus = Field(..., description="Overall health status")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")
    services: Dict[str, Dict[str, Any]] = Field(..., description="Status of individual services")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "status": "healthy",
                "version": "1.0.0",
                "uptime_seconds": 3600.5,
                "services": {
                    "pdf_processor": {
                        "status": "healthy",
                        "response_time_ms": 45
                    },
                    "supabase": {
                        "status": "healthy",
                        "connection_pool": "8/10"
                    },
                    "rag": {
                        "status": "healthy",
                        "service_type": "Direct Vector DB"
                    }
                },
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class HealthCheckResponse(BaseModel):
    """Document service health check response model."""

    status: str = Field(..., description="Service health status")
    timestamp: str = Field(..., description="Health check timestamp")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    dependencies: Optional[Dict[str, bool]] = Field(None, description="Status of service dependencies")
    error: Optional[str] = Field(None, description="Error message if unhealthy")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-07-26T18:00:00Z",
                "service": "Document Processing API",
                "version": "1.0.0",
                "dependencies": {
                    "pdf_processor": True,
                    "temp_directory": True,
                    "job_storage": True
                },
                "error": None
            }
        }


class FileUploadInfo(BaseModel):
    """Information about uploaded file."""

    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME content type")
    size_bytes: int = Field(..., description="File size in bytes")
    checksum: Optional[str] = Field(None, description="File checksum (MD5/SHA256)")
    storage_url: Optional[str] = Field(None, description="Supabase Storage public URL")
    storage_path: Optional[str] = Field(None, description="Supabase Storage file path")


class URLInfo(BaseModel):
    """Information about URL-based processing."""
    
    url: str = Field(..., description="Source URL")
    content_type: Optional[str] = Field(None, description="Detected content type")
    size_bytes: Optional[int] = Field(None, description="Content size in bytes")
    last_modified: Optional[datetime] = Field(None, description="Last modified timestamp")


class ProcessingOptions(BaseModel):
    """Configuration options for document processing."""
    
    extract_images: bool = Field(True, description="Whether to extract images")
    extract_tables: bool = Field(True, description="Whether to extract tables")
    page_number: Optional[int] = Field(None, description="Process specific page only")
    timeout_seconds: Optional[int] = Field(300, description="Processing timeout")
    quality: Optional[str] = Field("standard", pattern="^(fast|standard|high)$", description="Processing quality")
    language: Optional[str] = Field("auto", description="Document language hint")
    chunk_size: Optional[int] = Field(1000, description="Text chunk size for processing")
    overlap: Optional[int] = Field(200, description="Overlap between text chunks")
    
    @field_validator('timeout_seconds')
    @classmethod
    def validate_timeout(cls, v):
        if v is not None and (v < 10 or v > 3600):
            raise ValueError('timeout_seconds must be between 10 and 3600')
        return v


class MetricsSummary(BaseModel):
    """Summary metrics for content analysis."""
    
    word_count: int = Field(..., description="Total word count")
    character_count: int = Field(..., description="Total character count")
    page_count: int = Field(..., description="Number of pages")
    image_count: int = Field(0, description="Number of extracted images")
    table_count: int = Field(0, description="Number of extracted tables")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
