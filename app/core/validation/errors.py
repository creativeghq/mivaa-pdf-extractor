"""
Validation error handling framework for the PDF2Markdown microservice.

This module provides custom exception classes, error handlers, and utilities
for managing validation errors in a consistent and user-friendly manner.
"""

import traceback
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

from pydantic import ValidationError as PydanticValidationError
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

from app.schemas.common import ErrorResponse


class ValidationErrorType(str, Enum):
    """Enumeration of validation error types."""
    
    SCHEMA_VALIDATION = "schema_validation"
    SECURITY_VIOLATION = "security_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    REQUEST_TOO_LARGE = "request_too_large"
    INVALID_CONTENT_TYPE = "invalid_content_type"
    JSON_STRUCTURE_INVALID = "json_structure_invalid"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"
    CONFIGURATION_ERROR = "configuration_error"


class ValidationSeverity(str, Enum):
    """Severity levels for validation errors."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationError(Exception):
    """
    Base validation error class.
    
    This is the base class for all validation-related errors in the system.
    It provides structured error information including error type, severity,
    and detailed context.
    """
    
    def __init__(
        self,
        message: str,
        error_type: ValidationErrorType = ValidationErrorType.SCHEMA_VALIDATION,
        severity: ValidationSeverity = ValidationSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        field_errors: Optional[List[Dict[str, Any]]] = None,
        error_code: Optional[str] = None,
        status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize validation error.
        
        Args:
            message: Human-readable error message
            error_type: Type of validation error
            severity: Severity level of the error
            details: Additional error details
            field_errors: Field-specific validation errors
            error_code: Custom error code
            status_code: HTTP status code
            context: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.severity = severity
        self.details = details or {}
        self.field_errors = field_errors or []
        self.error_code = error_code or f"VALIDATION_{error_type.value.upper()}"
        self.status_code = status_code
        self.context = context or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self, include_details: bool = True) -> Dict[str, Any]:
        """
        Convert error to dictionary representation.
        
        Args:
            include_details: Whether to include detailed error information
            
        Returns:
            Dictionary representation of the error
        """
        error_dict = {
            "message": self.message,
            "error_type": self.error_type.value,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
        }
        
        if include_details:
            if self.details:
                error_dict["details"] = self.details
            if self.field_errors:
                error_dict["field_errors"] = self.field_errors
            if self.context:
                error_dict["context"] = self.context
        
        return error_dict
    
    def to_response(self, include_details: bool = True) -> ErrorResponse:
        """
        Convert error to ErrorResponse schema.
        
        Args:
            include_details: Whether to include detailed error information
            
        Returns:
            ErrorResponse instance
        """
        return ErrorResponse(
            success=False,
            message=self.message,
            error_code=self.error_code,
            timestamp=self.timestamp,
            details=self.to_dict(include_details) if include_details else None
        )


class SecurityValidationError(ValidationError):
    """Security-related validation error."""
    
    def __init__(
        self,
        message: str,
        violation_type: str,
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize security validation error.
        
        Args:
            message: Error message
            violation_type: Type of security violation
            details: Additional details
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(
            message=message,
            error_type=ValidationErrorType.SECURITY_VIOLATION,
            severity=ValidationSeverity.HIGH,
            details={**(details or {}), "violation_type": violation_type},
            status_code=status.HTTP_403_FORBIDDEN,
            **kwargs
        )
        self.violation_type = violation_type


class RateLimitError(ValidationError):
    """Rate limiting error."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        window: Optional[int] = None,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize rate limit error.
        
        Args:
            message: Error message
            limit: Rate limit threshold
            window: Rate limit window in seconds
            retry_after: Seconds to wait before retrying
            **kwargs: Additional arguments passed to parent
        """
        details = {}
        if limit is not None:
            details["limit"] = limit
        if window is not None:
            details["window"] = window
        if retry_after is not None:
            details["retry_after"] = retry_after
        
        super().__init__(
            message=message,
            error_type=ValidationErrorType.RATE_LIMIT_EXCEEDED,
            severity=ValidationSeverity.MEDIUM,
            details=details,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            **kwargs
        )
        self.retry_after = retry_after


class RequestSizeError(ValidationError):
    """Request size validation error."""
    
    def __init__(
        self,
        message: str = "Request size exceeds limit",
        actual_size: Optional[int] = None,
        max_size: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize request size error.
        
        Args:
            message: Error message
            actual_size: Actual request size in bytes
            max_size: Maximum allowed size in bytes
            **kwargs: Additional arguments passed to parent
        """
        details = {}
        if actual_size is not None:
            details["actual_size"] = actual_size
        if max_size is not None:
            details["max_size"] = max_size
        
        super().__init__(
            message=message,
            error_type=ValidationErrorType.REQUEST_TOO_LARGE,
            severity=ValidationSeverity.MEDIUM,
            details=details,
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            **kwargs
        )


class ContentTypeError(ValidationError):
    """Content type validation error."""
    
    def __init__(
        self,
        message: str = "Invalid content type",
        actual_type: Optional[str] = None,
        allowed_types: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize content type error.
        
        Args:
            message: Error message
            actual_type: Actual content type received
            allowed_types: List of allowed content types
            **kwargs: Additional arguments passed to parent
        """
        details = {}
        if actual_type is not None:
            details["actual_type"] = actual_type
        if allowed_types is not None:
            details["allowed_types"] = allowed_types
        
        super().__init__(
            message=message,
            error_type=ValidationErrorType.INVALID_CONTENT_TYPE,
            severity=ValidationSeverity.MEDIUM,
            details=details,
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            **kwargs
        )


class JSONStructureError(ValidationError):
    """JSON structure validation error."""
    
    def __init__(
        self,
        message: str = "Invalid JSON structure",
        violation_reason: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize JSON structure error.
        
        Args:
            message: Error message
            violation_reason: Reason for structure violation
            **kwargs: Additional arguments passed to parent
        """
        details = {}
        if violation_reason is not None:
            details["violation_reason"] = violation_reason
        
        super().__init__(
            message=message,
            error_type=ValidationErrorType.JSON_STRUCTURE_INVALID,
            severity=ValidationSeverity.MEDIUM,
            details=details,
            **kwargs
        )


class ValidationTimeoutError(ValidationError):
    """Validation timeout error."""
    
    def __init__(
        self,
        message: str = "Validation timeout",
        timeout_duration: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize validation timeout error.
        
        Args:
            message: Error message
            timeout_duration: Timeout duration in seconds
            **kwargs: Additional arguments passed to parent
        """
        details = {}
        if timeout_duration is not None:
            details["timeout_duration"] = timeout_duration
        
        super().__init__(
            message=message,
            error_type=ValidationErrorType.TIMEOUT,
            severity=ValidationSeverity.HIGH,
            details=details,
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            **kwargs
        )


class ValidationConfigurationError(ValidationError):
    """Validation configuration error."""
    
    def __init__(
        self,
        message: str = "Validation configuration error",
        config_issue: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize validation configuration error.
        
        Args:
            message: Error message
            config_issue: Description of configuration issue
            **kwargs: Additional arguments passed to parent
        """
        details = {}
        if config_issue is not None:
            details["config_issue"] = config_issue
        
        super().__init__(
            message=message,
            error_type=ValidationErrorType.CONFIGURATION_ERROR,
            severity=ValidationSeverity.CRITICAL,
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            **kwargs
        )


class ValidationErrorHandler:
    """
    Centralized validation error handler.
    
    This class provides utilities for handling, formatting, and responding
    to validation errors in a consistent manner across the application.
    """
    
    def __init__(self, include_details: bool = False, max_message_length: int = 500):
        """
        Initialize error handler.
        
        Args:
            include_details: Whether to include detailed error information
            max_message_length: Maximum length of error messages
        """
        self.include_details = include_details
        self.max_message_length = max_message_length
    
    def handle_pydantic_error(
        self,
        error: PydanticValidationError,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationError:
        """
        Convert Pydantic validation error to ValidationError.
        
        Args:
            error: Pydantic validation error
            context: Additional context information
            
        Returns:
            ValidationError instance
        """
        field_errors = []
        
        for pydantic_error in error.errors():
            field_error = {
                "field": ".".join(str(loc) for loc in pydantic_error["loc"]),
                "message": pydantic_error["msg"],
                "type": pydantic_error["type"],
            }
            
            if self.include_details and "ctx" in pydantic_error:
                field_error["context"] = pydantic_error["ctx"]
            
            field_errors.append(field_error)
        
        # Create summary message
        if len(field_errors) == 1:
            message = f"Validation failed for field '{field_errors[0]['field']}': {field_errors[0]['message']}"
        else:
            message = f"Validation failed for {len(field_errors)} fields"
        
        return ValidationError(
            message=self._truncate_message(message),
            error_type=ValidationErrorType.SCHEMA_VALIDATION,
            field_errors=field_errors,
            context=context or {}
        )
    
    def handle_http_exception(
        self,
        error: HTTPException,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationError:
        """
        Convert HTTPException to ValidationError.
        
        Args:
            error: HTTP exception
            context: Additional context information
            
        Returns:
            ValidationError instance
        """
        # Map HTTP status codes to validation error types
        error_type_mapping = {
            status.HTTP_400_BAD_REQUEST: ValidationErrorType.SCHEMA_VALIDATION,
            status.HTTP_403_FORBIDDEN: ValidationErrorType.SECURITY_VIOLATION,
            status.HTTP_408_REQUEST_TIMEOUT: ValidationErrorType.TIMEOUT,
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: ValidationErrorType.REQUEST_TOO_LARGE,
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: ValidationErrorType.INVALID_CONTENT_TYPE,
            status.HTTP_422_UNPROCESSABLE_ENTITY: ValidationErrorType.SCHEMA_VALIDATION,
            status.HTTP_429_TOO_MANY_REQUESTS: ValidationErrorType.RATE_LIMIT_EXCEEDED,
        }
        
        error_type = error_type_mapping.get(
            error.status_code,
            ValidationErrorType.INTERNAL_ERROR
        )
        
        return ValidationError(
            message=self._truncate_message(str(error.detail)),
            error_type=error_type,
            status_code=error.status_code,
            context=context or {}
        )
    
    def handle_generic_exception(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationError:
        """
        Convert generic exception to ValidationError.
        
        Args:
            error: Generic exception
            context: Additional context information
            
        Returns:
            ValidationError instance
        """
        details = {
            "exception_type": type(error).__name__,
        }
        
        if self.include_details:
            details["traceback"] = traceback.format_exc()
        
        return ValidationError(
            message=self._truncate_message(f"Internal validation error: {str(error)}"),
            error_type=ValidationErrorType.INTERNAL_ERROR,
            severity=ValidationSeverity.CRITICAL,
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            context=context or {}
        )
    
    def create_error_response(
        self,
        error: Union[ValidationError, Exception],
        context: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """
        Create JSON error response from validation error or exception.
        
        Args:
            error: Validation error or exception
            context: Additional context information
            
        Returns:
            JSON response with error information
        """
        if isinstance(error, ValidationError):
            validation_error = error
        elif isinstance(error, PydanticValidationError):
            validation_error = self.handle_pydantic_error(error, context)
        elif isinstance(error, HTTPException):
            validation_error = self.handle_http_exception(error, context)
        else:
            validation_error = self.handle_generic_exception(error, context)
        
        error_response = validation_error.to_response(self.include_details)
        
        return JSONResponse(
            status_code=validation_error.status_code,
            content=error_response.model_dump(exclude_none=True)
        )
    
    def _truncate_message(self, message: str) -> str:
        """
        Truncate error message to maximum length.
        
        Args:
            message: Original message
            
        Returns:
            Truncated message
        """
        if len(message) <= self.max_message_length:
            return message
        
        return message[:self.max_message_length - 3] + "..."
    
    def log_error(
        self,
        error: ValidationError,
        logger,
        include_context: bool = True
    ):
        """
        Log validation error with appropriate level.
        
        Args:
            error: Validation error to log
            logger: Logger instance
            include_context: Whether to include context in log
        """
        log_data = {
            "error_type": error.error_type.value,
            "error_code": error.error_code,
            "message": error.message,
            "severity": error.severity.value,
        }
        
        if include_context and error.context:
            log_data["context"] = error.context
        
        if error.field_errors:
            log_data["field_count"] = len(error.field_errors)
        
        # Log with appropriate level based on severity
        if error.severity == ValidationSeverity.CRITICAL:
            logger.critical("Validation error: %s", log_data)
        elif error.severity == ValidationSeverity.HIGH:
            logger.error("Validation error: %s", log_data)
        elif error.severity == ValidationSeverity.MEDIUM:
            logger.warning("Validation error: %s", log_data)
        else:
            logger.info("Validation error: %s", log_data)


def create_error_handler(
    include_details: bool = False,
    max_message_length: int = 500
) -> ValidationErrorHandler:
    """
    Factory function to create validation error handler.
    
    Args:
        include_details: Whether to include detailed error information
        max_message_length: Maximum length of error messages
        
    Returns:
        ValidationErrorHandler instance
    """
    return ValidationErrorHandler(include_details, max_message_length)


# Utility functions for common error scenarios

def create_schema_validation_error(
    field_name: str,
    message: str,
    value: Any = None,
    context: Optional[Dict[str, Any]] = None
) -> ValidationError:
    """Create a schema validation error for a specific field."""
    field_errors = [{
        "field": field_name,
        "message": message,
        "type": "value_error",
    }]
    
    if value is not None:
        field_errors[0]["value"] = str(value)
    
    return ValidationError(
        message=f"Validation failed for field '{field_name}': {message}",
        error_type=ValidationErrorType.SCHEMA_VALIDATION,
        field_errors=field_errors,
        context=context or {}
    )


def create_security_error(
    violation_type: str,
    message: str,
    context: Optional[Dict[str, Any]] = None
) -> SecurityValidationError:
    """Create a security validation error."""
    return SecurityValidationError(
        message=message,
        violation_type=violation_type,
        context=context or {}
    )


def create_rate_limit_error(
    limit: int,
    window: int,
    retry_after: Optional[int] = None
) -> RateLimitError:
    """Create a rate limit error."""
    return RateLimitError(
        message=f"Rate limit of {limit} requests per {window} seconds exceeded",
        limit=limit,
        window=window,
        retry_after=retry_after
    )


# Export all error classes and utilities
__all__ = [
    "ValidationErrorType",
    "ValidationSeverity",
    "ValidationError",
    "SecurityValidationError",
    "RateLimitError",
    "RequestSizeError",
    "ContentTypeError",
    "JSONStructureError",
    "ValidationTimeoutError",
    "ValidationConfigurationError",
    "ValidationErrorHandler",
    "create_error_handler",
    "create_schema_validation_error",
    "create_security_error",
    "create_rate_limit_error",
]