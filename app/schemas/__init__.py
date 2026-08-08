"""
Pydantic schemas for API request/response validation.

This module contains all the data models used for API validation,
serialization, and documentation.

Imports are EXPLICIT rather than `from .common import *`. The five star imports this
replaced meant ruff could not tell whether the 22 names in `__all__` were actually
bound (F403/F405 x27) — and neither could a reader. Worse, the failure mode was
silent: a name listed in `__all__` but absent from its submodule would simply not be
exported, and `from app.schemas import ThatName` would fail at the call site rather
than here. Every name below was verified against its defining module before the
conversion; an explicit import now fails loudly at startup if one goes missing.
"""

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
