"""
PDF Processing Data Models

This module defines Pydantic models for PDF processing requests, responses,
and configuration options, providing type safety and validation for the API.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ImageFormat(str, Enum):
    """Supported image formats for extraction."""
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    WEBP = "webp"


class ProcessingOptions(BaseModel):
    """Configuration options for PDF processing."""
    
    # Page selection
    page_number: Optional[int] = Field(
        None, 
        description="Specific page to process (None for all pages)",
        ge=1
    )
    pages: Optional[List[int]] = Field(
        None, 
        description="List of specific pages to process (None for all)"
    )
    max_pages: Optional[int] = Field(
        None, 
        description="Maximum number of pages to process",
        ge=1
    )
    
    # Image extraction options
    extract_images: bool = Field(
        True, 
        description="Whether to extract images from PDF"
    )
    image_format: ImageFormat = Field(
        ImageFormat.PNG, 
        description="Format for extracted images"
    )
    image_dpi: int = Field(
        150, 
        description="DPI for image extraction",
        ge=72,
        le=600
    )
    min_image_size: int = Field(
        100, 
        description="Minimum image size in pixels",
        ge=10
    )
    max_image_size: int = Field(
        2048, 
        description="Maximum image size in pixels",
        ge=100
    )
    
    # Content processing options
    extract_tables: bool = Field(
        True, 
        description="Whether to extract table content"
    )
    extract_headers: bool = Field(
        True, 
        description="Whether to extract header information"
    )
    extract_footers: bool = Field(
        False, 
        description="Whether to extract footer information"
    )
    
    # Processing limits and timeouts
    timeout_seconds: int = Field(
        300, 
        description="Processing timeout in seconds",
        ge=10,
        le=1800
    )
    download_timeout: int = Field(
        30,
        description="Download timeout for URLs in seconds",
        ge=5,
        le=300
    )
    
    @validator('pages')
    def validate_pages(cls, v):
        """Validate page numbers are positive."""
        if v is not None:
            for page in v:
                if page < 1:
                    raise ValueError("Page numbers must be positive")
        return v
    
    @validator('max_image_size')
    def validate_image_size_range(cls, v, values):
        """Ensure max_image_size is greater than min_image_size."""
        min_size = values.get('min_image_size', 100)
        if v <= min_size:
            raise ValueError("max_image_size must be greater than min_image_size")
        return v


class PDFProcessingRequest(BaseModel):
    """Request model for PDF processing."""
    
    document_id: Optional[str] = Field(
        None, 
        description="Optional document identifier for tracking"
    )
    pdf_url: Optional[str] = Field(
        None, 
        description="URL to PDF file (alternative to file upload)"
    )
    processing_options: Optional[ProcessingOptions] = Field(
        default_factory=ProcessingOptions,
        description="Processing configuration options"
    )
    
    @validator('pdf_url')
    def validate_pdf_url(cls, v):
        """Basic URL validation for PDF URLs."""
        if v is not None:
            if not v.startswith(('http://', 'https://')):
                raise ValueError("PDF URL must start with http:// or https://")
            if not (v.lower().endswith('.pdf') or 'pdf' in v.lower()):
                raise ValueError("URL should point to a PDF file")
        return v


class ImageInfo(BaseModel):
    """Information about an extracted image."""
    
    filename: str = Field(description="Image filename")
    path: str = Field(description="Path to the image file")
    size_bytes: int = Field(description="Image file size in bytes", ge=0)
    format: str = Field(description="Image format (PNG, JPEG, etc.)")
    width: Optional[int] = Field(None, description="Image width in pixels", ge=1)
    height: Optional[int] = Field(None, description="Image height in pixels", ge=1)
    page_number: Optional[int] = Field(None, description="Source page number", ge=1)


class PDFMetadata(BaseModel):
    """PDF document metadata."""
    
    page_count: int = Field(description="Total number of pages", ge=0)
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    subject: Optional[str] = Field(None, description="Document subject")
    creator: Optional[str] = Field(None, description="Document creator")
    producer: Optional[str] = Field(None, description="PDF producer")
    creation_date: Optional[str] = Field(None, description="Creation date")
    modification_date: Optional[str] = Field(None, description="Last modification date")
    word_count: int = Field(description="Estimated word count", ge=0)
    character_count: int = Field(description="Character count", ge=0)
    line_count: int = Field(description="Line count", ge=0)


class PDFProcessingResponse(BaseModel):
    """Response model for PDF processing."""
    
    document_id: str = Field(description="Document identifier")
    status: str = Field(description="Processing status")
    markdown_content: str = Field(description="Extracted markdown content")
    extracted_images: List[ImageInfo] = Field(
        default_factory=list,
        description="List of extracted images with metadata"
    )
    metadata: PDFMetadata = Field(description="PDF document metadata")
    processing_time: float = Field(description="Processing time in seconds", ge=0)
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response creation timestamp"
    )
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProcessingStatus(str, Enum):
    """Processing status values."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class PDFProcessingJob(BaseModel):
    """Model for tracking PDF processing jobs."""
    
    job_id: str = Field(description="Unique job identifier")
    document_id: Optional[str] = Field(None, description="Document identifier")
    status: ProcessingStatus = Field(description="Current job status")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Job creation timestamp"
    )
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    result: Optional[PDFProcessingResponse] = Field(None, description="Processing result")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(description="Service health status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    version: str = Field(description="Service version")
    service: str = Field(description="Service name")
    dependencies: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of service dependencies"
    )
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str = Field(description="Error type or category")
    detail: str = Field(description="Detailed error message")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Convenience type aliases
ProcessingOptionsDict = Dict[str, Any]
MetadataDict = Dict[str, Any]