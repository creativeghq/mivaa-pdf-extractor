"""
Custom Exception Classes for PDF Processing

This module defines custom exception classes for the PDF2Markdown microservice,
providing structured error handling with appropriate HTTP status codes and
detailed error messages for different failure scenarios.
"""

from typing import Optional, Dict, Any


class MaterialKaiIntegrationError(Exception):
    """Custom exception for Material Kai integration errors."""
    pass


class ServiceError(Exception):
    """Base exception for service-related errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "SERVICE_ERROR"
        self.details = details or {}


class ExternalServiceError(ServiceError):
    """Exception for external service integration errors."""

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)
        self.service_name = service_name
        self.error_code = error_code or "EXTERNAL_SERVICE_ERROR"

class PDFProcessingError(Exception):
    """
    Base exception class for all PDF processing related errors.
    
    This serves as the parent class for all custom exceptions in the PDF processing
    pipeline, providing a common interface for error handling.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the PDF processing error.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "PDF_PROCESSING_ERROR"
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


class PDFValidationError(PDFProcessingError):
    """
    Exception raised when PDF file validation fails.
    
    This includes scenarios like:
    - Invalid file format
    - Corrupted PDF files
    - Unsupported PDF versions
    - Password-protected PDFs
    """
    
    def __init__(
        self, 
        message: str = "PDF file validation failed", 
        file_path: Optional[str] = None,
        validation_details: Optional[Dict[str, Any]] = None
    ):
        details = {"file_path": file_path}
        if validation_details:
            details.update(validation_details)
        
        super().__init__(
            message=message,
            error_code="PDF_VALIDATION_ERROR",
            details=details
        )


class PDFExtractionError(PDFProcessingError):
    """
    Exception raised when PDF content extraction fails.
    
    This covers failures in:
    - Text extraction
    - Image extraction
    - Table extraction
    - Metadata extraction
    """
    
    def __init__(
        self, 
        message: str = "PDF content extraction failed",
        extraction_type: Optional[str] = None,
        page_number: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        details = {
            "extraction_type": extraction_type,
            "page_number": page_number
        }
        
        if original_error:
            details["original_error"] = str(original_error)
            details["original_error_type"] = type(original_error).__name__
        
        super().__init__(
            message=message,
            error_code="PDF_EXTRACTION_ERROR",
            details=details
        )


class PDFDownloadError(PDFProcessingError):
    """
    Exception raised when PDF download from URL fails.
    
    This includes:
    - Network connectivity issues
    - Invalid URLs
    - HTTP errors (404, 403, etc.)
    - Timeout errors
    - File size limitations
    """
    
    def __init__(
        self, 
        message: str = "PDF download failed",
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        timeout: Optional[bool] = False
    ):
        details = {
            "url": url,
            "status_code": status_code,
            "timeout": timeout
        }
        
        super().__init__(
            message=message,
            error_code="PDF_DOWNLOAD_ERROR",
            details=details
        )


class PDFSizeError(PDFProcessingError):
    """
    Exception raised when PDF file size exceeds limits.
    
    This handles:
    - Files too large for processing
    - Memory limitations
    - Storage constraints
    """
    
    def __init__(
        self, 
        message: str = "PDF file size exceeds limits",
        file_size: Optional[int] = None,
        max_size: Optional[int] = None
    ):
        details = {
            "file_size_bytes": file_size,
            "max_size_bytes": max_size
        }
        
        if file_size and max_size:
            details["size_ratio"] = file_size / max_size
        
        super().__init__(
            message=message,
            error_code="PDF_SIZE_ERROR",
            details=details
        )


class PDFTimeoutError(PDFProcessingError):
    """
    Exception raised when PDF processing times out.
    
    This covers:
    - Processing timeout
    - Download timeout
    - Network timeout
    """
    
    def __init__(
        self, 
        message: str = "PDF processing timed out",
        timeout_seconds: Optional[int] = None,
        operation: Optional[str] = None
    ):
        details = {
            "timeout_seconds": timeout_seconds,
            "operation": operation
        }
        
        super().__init__(
            message=message,
            error_code="PDF_TIMEOUT_ERROR",
            details=details
        )


class PDFConfigurationError(PDFProcessingError):
    """
    Exception raised when there are configuration issues.
    
    This includes:
    - Missing required settings
    - Invalid configuration values
    - Environment setup issues
    """
    
    def __init__(
        self, 
        message: str = "PDF processing configuration error",
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None
    ):
        details = {
            "config_key": config_key,
            "config_value": str(config_value) if config_value is not None else None
        }
        
        super().__init__(
            message=message,
            error_code="PDF_CONFIGURATION_ERROR",
            details=details
        )


class PDFStorageError(PDFProcessingError):
    """
    Exception raised when file storage operations fail.
    
    This covers:
    - Temporary file creation failures
    - File system permissions
    - Disk space issues
    - File cleanup failures
    """
    
    def __init__(
        self, 
        message: str = "PDF storage operation failed",
        operation: Optional[str] = None,
        file_path: Optional[str] = None
    ):
        details = {
            "operation": operation,
            "file_path": file_path
        }
        
        super().__init__(
            message=message,
            error_code="PDF_STORAGE_ERROR",
            details=details
        )


class PDFFormatError(PDFProcessingError):
    """
    Exception raised when PDF format is unsupported or invalid.
    
    This includes:
    - Unsupported PDF versions
    - Malformed PDF structure
    - Encrypted/protected PDFs
    - Non-PDF files with PDF extension
    """
    
    def __init__(
        self, 
        message: str = "PDF format is unsupported or invalid",
        pdf_version: Optional[str] = None,
        is_encrypted: Optional[bool] = None
    ):
        details = {
            "pdf_version": pdf_version,
            "is_encrypted": is_encrypted
        }
        
        super().__init__(
            message=message,
            error_code="PDF_FORMAT_ERROR",
            details=details
        )


# Convenience mapping for HTTP status codes
EXCEPTION_STATUS_CODES = {
    PDFValidationError: 400,  # Bad Request
    PDFExtractionError: 422,  # Unprocessable Entity
    PDFDownloadError: 502,    # Bad Gateway
    PDFSizeError: 413,        # Payload Too Large
    PDFTimeoutError: 504,     # Gateway Timeout
    PDFConfigurationError: 500,  # Internal Server Error
    PDFStorageError: 500,     # Internal Server Error
    PDFFormatError: 415,      # Unsupported Media Type
    PDFProcessingError: 500,  # Internal Server Error (fallback)
}


def get_http_status_code(exception: Exception) -> int:
    """
    Get the appropriate HTTP status code for an exception.
    
    Args:
        exception: The exception instance
        
    Returns:
        HTTP status code (defaults to 500 for unknown exceptions)
    """
    for exc_type, status_code in EXCEPTION_STATUS_CODES.items():
        if isinstance(exception, exc_type):
            return status_code
    
    # Default to 500 for unknown exceptions
    return 500


def create_error_response(exception: Exception) -> Dict[str, Any]:
    """
    Create a standardized error response dictionary from an exception.

    Args:
        exception: The exception instance

    Returns:
        Dictionary containing error information suitable for API responses
    """
    if isinstance(exception, PDFProcessingError):
        return exception.to_dict()

    # Handle non-custom exceptions
    return {
        "error": "UNKNOWN_ERROR",
        "message": str(exception),
        "details": {
            "exception_type": type(exception).__name__
        }
    }


# =============================================================================
# ERROR HANDLING UTILITIES
# =============================================================================

import functools
import logging
from typing import Callable, TypeVar, Union

T = TypeVar('T')


def handle_extraction_errors(
    operation_name: str,
    default_return: T = None,
    reraise: bool = False,
    log_level: int = logging.ERROR
) -> Callable:
    """
    Decorator for consistent error handling in PDF extraction methods.

    Provides standardized logging, exception conversion, and recovery behavior
    for extraction operations.

    Args:
        operation_name: Name of the operation for logging (e.g., "image extraction")
        default_return: Default value to return on error (if not reraising)
        reraise: If True, reraises the exception after logging; if False, returns default
        log_level: Logging level for error messages (default: ERROR)

    Usage:
        @handle_extraction_errors("image extraction", default_return=[], reraise=False)
        async def extract_images(self, pdf_path: str) -> List[Dict]:
            ...

        @handle_extraction_errors("PDF processing", reraise=True)
        async def process_pdf(self, pdf_bytes: bytes) -> PDFProcessingResult:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            logger = logging.getLogger(func.__module__)
            try:
                return await func(*args, **kwargs)
            except PDFProcessingError:
                # Already a custom exception, just log and handle
                logger.log(log_level, f"{operation_name} failed with PDF processing error", exc_info=True)
                if reraise:
                    raise
                return default_return
            except Exception as e:
                # Convert to appropriate custom exception
                logger.log(log_level, f"{operation_name} failed: {str(e)}", exc_info=True)
                if reraise:
                    raise PDFExtractionError(
                        message=f"{operation_name} failed: {str(e)}",
                        extraction_type=operation_name,
                        original_error=e
                    )
                return default_return

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            logger = logging.getLogger(func.__module__)
            try:
                return func(*args, **kwargs)
            except PDFProcessingError:
                logger.log(log_level, f"{operation_name} failed with PDF processing error", exc_info=True)
                if reraise:
                    raise
                return default_return
            except Exception as e:
                logger.log(log_level, f"{operation_name} failed: {str(e)}", exc_info=True)
                if reraise:
                    raise PDFExtractionError(
                        message=f"{operation_name} failed: {str(e)}",
                        extraction_type=operation_name,
                        original_error=e
                    )
                return default_return

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class ErrorRecoveryContext:
    """
    Context manager for recoverable operations with automatic cleanup.

    Provides structured error handling with resource cleanup for operations
    that should attempt recovery before failing.

    Usage:
        async with ErrorRecoveryContext("image processing", logger) as ctx:
            result = await process_image(path)
            ctx.set_result(result)
        # If exception occurred, ctx.result will be None but cleanup runs
    """

    def __init__(
        self,
        operation_name: str,
        logger: logging.Logger,
        cleanup_func: Optional[Callable] = None,
        suppress_errors: bool = True
    ):
        self.operation_name = operation_name
        self.logger = logger
        self.cleanup_func = cleanup_func
        self.suppress_errors = suppress_errors
        self.result = None
        self.error = None
        self.success = False

    def set_result(self, result: Any) -> None:
        """Set the successful result."""
        self.result = result
        self.success = True

    async def __aenter__(self) -> 'ErrorRecoveryContext':
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self.error = exc_val
            self.logger.warning(
                f"{self.operation_name} encountered error: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb)
            )

        # Run cleanup if provided
        if self.cleanup_func:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(self.cleanup_func):
                    await self.cleanup_func()
                else:
                    self.cleanup_func()
            except Exception as cleanup_error:
                self.logger.warning(f"Cleanup failed for {self.operation_name}: {cleanup_error}")

        # Return True to suppress exception if configured
        return self.suppress_errors and exc_type is not None

    def __enter__(self) -> 'ErrorRecoveryContext':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self.error = exc_val
            self.logger.warning(
                f"{self.operation_name} encountered error: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb)
            )

        # Run cleanup if provided
        if self.cleanup_func:
            try:
                self.cleanup_func()
            except Exception as cleanup_error:
                self.logger.warning(f"Cleanup failed for {self.operation_name}: {cleanup_error}")

        return self.suppress_errors and exc_type is not None
