"""
Common Pydantic schemas used across all API endpoints.

This module contains base response models, pagination, error handling,
and other shared data structures.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from uuid import UUID

try:
    # Try Pydantic v2 first
    from pydantic import BaseModel, Field, field_validator as validator
except ImportError:
    # Fall back to Pydantic v1
    from pydantic import BaseModel, Field, validator


# Generic type for paginated responses
T = TypeVar('T')


class BaseResponse(BaseModel):
    """Base response model with common fields."""
    
    success: bool = Field(True, description="Indicates if the request was successful")
    message: Optional[str] = Field(None, description="Optional message or description")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
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
        schema_extra = {
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


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""
    
    page: int = Field(1, ge=1, description="Page number (1-based)")
    page_size: int = Field(20, ge=1, le=100, description="Number of items per page")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: Optional[str] = Field("desc", pattern="^(asc|desc)$", description="Sort order")
    
    @validator('sort_order')
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
    
    @validator('total_pages', always=True)
    def calculate_total_pages(cls, v, values):
        total_count = values.get('total_count', 0)
        page_size = values.get('page_size', 1)
        return max(1, (total_count + page_size - 1) // page_size)
    
    @validator('has_next', always=True)
    def calculate_has_next(cls, v, values):
        page = values.get('page', 1)
        total_pages = values.get('total_pages', 1)
        return page < total_pages
    
    @validator('has_previous', always=True)
    def calculate_has_previous(cls, v, values):
        page = values.get('page', 1)
        return page > 1


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
        schema_extra = {
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
                    "llamaindex": {
                        "status": "healthy",
                        "index_size": 1024
                    }
                },
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class FileUploadInfo(BaseModel):
    """Information about uploaded file."""
    
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME content type")
    size_bytes: int = Field(..., description="File size in bytes")
    checksum: Optional[str] = Field(None, description="File checksum (MD5/SHA256)")


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
    
    @validator('timeout_seconds')
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