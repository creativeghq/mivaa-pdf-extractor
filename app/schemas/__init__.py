"""Pydantic schemas for API request/response validation."""

from .common import (
    BaseResponse,
    ErrorResponse,
    PaginationParams,
    PaginationResponse,
    HealthResponse,
)
from .documents import (
    DocumentProcessRequest,
    DocumentProcessResponse,
    DocumentMetadata,
    DocumentContent,
    DocumentChunk,
    DocumentListResponse,
)
from .search import (
    SearchRequest,
    SearchResponse,
    QueryRequest,
    QueryResponse,
    SimilaritySearchRequest,
)
from .images import (
    ImageAnalysisRequest,
    ImageAnalysisResponse,
    ImageMetadata,
)
from .jobs import (
    JobStatus,
    JobResponse,
    JobListResponse,
)

__all__ = [
    # Common schemas
    "BaseResponse",
    "ErrorResponse",
    "PaginationParams",
    "PaginationResponse",
    "HealthResponse",

    # Document schemas
    "DocumentProcessRequest",
    "DocumentProcessResponse",
    "DocumentMetadata",
    "DocumentContent",
    "DocumentChunk",
    "DocumentListResponse",

    # Search schemas
    "SearchRequest",
    "SearchResponse",
    "QueryRequest",
    "QueryResponse",
    "SimilaritySearchRequest",

    # Image schemas
    "ImageAnalysisRequest",
    "ImageAnalysisResponse",
    "ImageMetadata",

    # Job schemas
    "JobStatus",
    "JobResponse",
    "JobListResponse",
]
