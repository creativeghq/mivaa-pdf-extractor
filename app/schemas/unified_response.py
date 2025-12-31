"""
Unified API Response Schema

This module provides a standardized response format for all API endpoints,
ensuring consistency between frontend and backend communication.

The unified response format matches the frontend StandardMivaaResponse interface
defined in src/config/mivaaStandardization.ts
"""

from typing import Any, Dict, Generic, Optional, TypeVar
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


# Generic type for response data
T = TypeVar('T')


class ErrorDetail(BaseModel):
    """Detailed error information."""

    code: str = Field(..., description="Machine-readable error code (e.g., 'VALIDATION_ERROR', 'SERVICE_UNAVAILABLE')")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error context and debugging information")
    retryable: bool = Field(False, description="Whether the request can be retried")

    class Config:
        json_schema_extra = {
            "example": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid input parameters",
                "details": {"field": "workspace_id", "issue": "required"},
                "retryable": False
            }
        }


class ResponseMetadata(BaseModel):
    """Response metadata for tracking and debugging."""

    request_id: Optional[str] = Field(None, description="Unique request identifier for tracing")
    processing_time: float = Field(..., description="Request processing time in milliseconds")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Response timestamp in ISO 8601 format"
    )
    version: str = Field(default="1.0.0", description="API version")
    endpoint: str = Field(..., description="API endpoint path")

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_abc123xyz",
                "processing_time": 245.5,
                "timestamp": "2024-01-15T10:30:00.000Z",
                "version": "1.0.0",
                "endpoint": "/api/search/semantic"
            }
        }


class UnifiedApiResponse(BaseModel, Generic[T]):
    """
    Unified API response format for all endpoints.

    This format ensures consistency across all API responses and matches
    the frontend StandardMivaaResponse interface.

    Type Parameters:
        T: Type of the response data payload

    Examples:
        Success response:
        ```python
        return UnifiedApiResponse(
            success=True,
            data={"results": [...]},
            metadata=ResponseMetadata(
                processing_time=123.4,
                endpoint="/api/search"
            )
        )
        ```

        Error response:
        ```python
        return UnifiedApiResponse(
            success=False,
            error=ErrorDetail(
                code="NOT_FOUND",
                message="Resource not found",
                retryable=False
            ),
            metadata=ResponseMetadata(
                processing_time=45.2,
                endpoint="/api/resource/123"
            )
        )
        ```
    """

    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[T] = Field(None, description="Response data payload (present on success)")
    error: Optional[ErrorDetail] = Field(None, description="Error details (present on failure)")
    metadata: ResponseMetadata = Field(..., description="Response metadata for tracking and debugging")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {"id": "123", "name": "Example"},
                "error": None,
                "metadata": {
                    "request_id": "req_abc123",
                    "processing_time": 150.5,
                    "timestamp": "2024-01-15T10:30:00.000Z",
                    "version": "1.0.0",
                    "endpoint": "/api/example"
                }
            }
        }

    @classmethod
    def success_response(
        cls,
        data: T,
        endpoint: str,
        processing_time: float,
        request_id: Optional[str] = None,
        version: str = "1.0.0"
    ) -> "UnifiedApiResponse[T]":
        """
        Create a successful response.

        Args:
            data: Response data payload
            endpoint: API endpoint path
            processing_time: Processing time in milliseconds
            request_id: Optional request ID for tracing
            version: API version

        Returns:
            UnifiedApiResponse with success=True
        """
        return cls(
            success=True,
            data=data,
            error=None,
            metadata=ResponseMetadata(
                request_id=request_id,
                processing_time=processing_time,
                endpoint=endpoint,
                version=version
            )
        )


    @classmethod
    def error_response(
        cls,
        code: str,
        message: str,
        endpoint: str,
        processing_time: float,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
        request_id: Optional[str] = None,
        version: str = "1.0.0"
    ) -> "UnifiedApiResponse":
        """
        Create an error response.

        Args:
            code: Machine-readable error code
            message: Human-readable error message
            endpoint: API endpoint path
            processing_time: Processing time in milliseconds
            details: Additional error context
            retryable: Whether the request can be retried
            request_id: Optional request ID for tracing
            version: API version

        Returns:
            UnifiedApiResponse with success=False
        """
        return cls(
            success=False,
            data=None,
            error=ErrorDetail(
                code=code,
                message=message,
                details=details,
                retryable=retryable
            ),
            metadata=ResponseMetadata(
                request_id=request_id,
                processing_time=processing_time,
                endpoint=endpoint,
                version=version
            )
        )


# =============================================================================
# Common Error Codes
# =============================================================================

class ErrorCode(str, Enum):
    """Standard error codes for consistent error handling."""

    # Client Errors (4xx)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # Server Errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    DATABASE_ERROR = "DATABASE_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    TIMEOUT = "TIMEOUT"

    # Processing Errors
    PDF_PROCESSING_ERROR = "PDF_PROCESSING_ERROR"
    EMBEDDING_GENERATION_ERROR = "EMBEDDING_GENERATION_ERROR"
    AI_SERVICE_ERROR = "AI_SERVICE_ERROR"
    VECTOR_SEARCH_ERROR = "VECTOR_SEARCH_ERROR"

    # Business Logic Errors
    INVALID_WORKSPACE = "INVALID_WORKSPACE"
    DUPLICATE_RESOURCE = "DUPLICATE_RESOURCE"
    RESOURCE_LOCKED = "RESOURCE_LOCKED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"

