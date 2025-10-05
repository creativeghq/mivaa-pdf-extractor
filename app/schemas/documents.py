"""
Document processing Pydantic schemas.

This module contains all schemas related to document processing,
including requests, responses, and document metadata.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

try:
    # Try Pydantic v2 first
    from pydantic import BaseModel, Field, HttpUrl, field_validator
except ImportError:
    # Fall back to Pydantic v1
    from pydantic import BaseModel, Field, HttpUrl, validator as field_validator

from .common import (
    BaseResponse,
    FileUploadInfo,
    URLInfo,
    ProcessingOptions,
    ProcessingStatus,
    MetricsSummary
)


class DocumentProcessRequest(BaseModel):
    """Request model for document processing."""
    
    # Source specification (either file upload or URL)
    source_url: Optional[HttpUrl] = Field(None, description="URL to PDF document")
    
    # Processing options
    options: ProcessingOptions = Field(default_factory=ProcessingOptions, description="Processing configuration")
    
    # Metadata
    document_name: Optional[str] = Field(None, description="Custom document name")
    tags: List[str] = Field(default_factory=list, description="Document tags for organization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('source_url')
    @classmethod
    def validate_source(cls, v):
        # Note: File upload validation will be handled at the endpoint level
        # since FastAPI handles file uploads separately from Pydantic models
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "source_url": "https://example.com/document.pdf",
                "options": {
                    "extract_images": True,
                    "extract_tables": True,
                    "quality": "standard",
                    "timeout_seconds": 300
                },
                "document_name": "Research Paper 2024",
                "tags": ["research", "ai", "2024"],
                "metadata": {
                    "author": "John Doe",
                    "category": "academic"
                }
            }
        }


class DocumentChunk(BaseModel):
    """Represents a chunk of document content."""
    
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Text content of the chunk")
    page_number: int = Field(..., description="Source page number")
    chunk_index: int = Field(..., description="Index within the page")
    start_char: int = Field(..., description="Start character position")
    end_char: int = Field(..., description="End character position")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding (if generated)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk-specific metadata")


class ImageInfo(BaseModel):
    """Information about extracted images."""
    
    image_id: str = Field(..., description="Unique image identifier")
    filename: str = Field(..., description="Image filename")
    page_number: int = Field(..., description="Source page number")
    format: str = Field(..., description="Image format (PNG, JPEG, etc.)")
    size_bytes: int = Field(..., description="Image file size")
    dimensions: Dict[str, int] = Field(..., description="Image dimensions (width, height)")
    description: Optional[str] = Field(None, description="AI-generated image description")
    url: Optional[str] = Field(None, description="URL to access the image")


class TableInfo(BaseModel):
    """Information about extracted tables."""
    
    table_id: str = Field(..., description="Unique table identifier")
    page_number: int = Field(..., description="Source page number")
    rows: int = Field(..., description="Number of rows")
    columns: int = Field(..., description="Number of columns")
    csv_data: str = Field(..., description="Table data in CSV format")
    description: Optional[str] = Field(None, description="AI-generated table description")


class DocumentMetadata(BaseModel):
    """Comprehensive document metadata."""
    
    # Basic document info
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    subject: Optional[str] = Field(None, description="Document subject")
    creator: Optional[str] = Field(None, description="Document creator application")
    producer: Optional[str] = Field(None, description="PDF producer")
    creation_date: Optional[datetime] = Field(None, description="Document creation date")
    modification_date: Optional[datetime] = Field(None, description="Last modification date")
    
    # Processing metadata
    language: Optional[str] = Field(None, description="Detected document language")
    confidence_score: Optional[float] = Field(None, description="Processing confidence score")
    
    # Custom metadata
    tags: List[str] = Field(default_factory=list, description="User-defined tags")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")


class DocumentContent(BaseModel):
    """Complete document content structure."""
    
    markdown_content: str = Field(..., description="Full document content in Markdown format")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="Document chunks for RAG")
    images: List[ImageInfo] = Field(default_factory=list, description="Extracted images")
    tables: List[TableInfo] = Field(default_factory=list, description="Extracted tables")
    
    # Content analysis
    summary: Optional[str] = Field(None, description="AI-generated document summary")
    key_topics: List[str] = Field(default_factory=list, description="Identified key topics")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Named entities")


class DocumentProcessResponse(BaseResponse):
    """Response model for document processing."""
    
    document_id: str = Field(..., description="Unique document identifier")
    status: ProcessingStatus = Field(..., description="Processing status")
    
    # Source information
    source_info: Union[FileUploadInfo, URLInfo] = Field(..., description="Source document information")
    
    # Processing results
    content: Optional[DocumentContent] = Field(None, description="Processed document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    metrics: MetricsSummary = Field(..., description="Processing metrics")
    
    # Storage information
    storage_path: Optional[str] = Field(None, description="Path where document is stored")
    embeddings_generated: bool = Field(False, description="Whether embeddings were generated")
    
    # Error information (if status is failed)
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error details if processing failed")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "document_id": "doc_123e4567-e89b-12d3-a456-426614174000",
                "status": "completed",
                "source_info": {
                    "filename": "research_paper.pdf",
                    "content_type": "application/pdf",
                    "size_bytes": 2048576
                },
                "content": {
                    "markdown_content": "# Research Paper Title\n\nAbstract content...",
                    "chunks": [],
                    "images": [],
                    "tables": []
                },
                "metadata": {
                    "title": "Research Paper Title",
                    "author": "John Doe",
                    "page_count": 15,
                    "tags": ["research", "ai"]
                },
                "metrics": {
                    "word_count": 5000,
                    "character_count": 30000,
                    "page_count": 15,
                    "processing_time_seconds": 45.2
                },
                "embeddings_generated": True,
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class DocumentListItem(BaseModel):
    """Simplified document information for list responses."""
    
    document_id: str = Field(..., description="Unique document identifier")
    document_name: str = Field(..., description="Document name")
    status: ProcessingStatus = Field(..., description="Processing status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    # Basic metadata
    page_count: int = Field(..., description="Number of pages")
    word_count: int = Field(..., description="Word count")
    file_size: int = Field(..., description="File size in bytes")
    
    # Tags and categorization
    tags: List[str] = Field(default_factory=list, description="Document tags")
    
    # Processing info
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    has_embeddings: bool = Field(False, description="Whether embeddings are available")


class DocumentListResponse(BaseResponse):
    """Response model for document listing."""
    
    documents: List[DocumentListItem] = Field(..., description="List of documents")
    total_count: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "documents": [
                    {
                        "document_id": "doc_123",
                        "document_name": "Research Paper 2024",
                        "status": "completed",
                        "created_at": "2024-07-26T18:00:00Z",
                        "updated_at": "2024-07-26T18:05:00Z",
                        "page_count": 15,
                        "word_count": 5000,
                        "file_size": 2048576,
                        "tags": ["research", "ai"],
                        "processing_time": 45.2,
                        "has_embeddings": True
                    }
                ],
                "total_count": 1,
                "page": 1,
                "page_size": 20,
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class DocumentUpdateRequest(BaseModel):
    """Request model for updating document metadata."""
    
    document_name: Optional[str] = Field(None, description="Updated document name")
    tags: Optional[List[str]] = Field(None, description="Updated tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata fields")
    
    class Config:
        schema_extra = {
            "example": {
                "document_name": "Updated Research Paper Title",
                "tags": ["research", "ai", "2024", "updated"],
                "metadata": {
                    "category": "academic",
                    "priority": "high"
                }
            }
        }


class DocumentDeleteResponse(BaseResponse):
    """Response model for document deletion."""
    
    document_id: str = Field(..., description="ID of deleted document")
    deleted_files: List[str] = Field(default_factory=list, description="List of deleted file paths")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Document deleted successfully",
                "document_id": "doc_123e4567-e89b-12d3-a456-426614174000",
                "deleted_files": [
                    "/storage/documents/doc_123.pdf",
                    "/storage/images/doc_123_img1.png"
                ],
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }