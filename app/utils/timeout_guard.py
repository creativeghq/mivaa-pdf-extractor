"""
Timeout Guard Utility

Provides timeout protection for async operations to prevent indefinite hangs.
All critical async operations should be wrapped with these guards.
"""

import asyncio
import logging
from typing import TypeVar, Callable, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TimeoutError(Exception):
    """Raised when an operation times out"""
    pass


async def with_timeout(
    coro,
    timeout_seconds: float,
    operation_name: str = "operation",
    raise_on_timeout: bool = True,
    default_value: Any = None
) -> Any:
    """
    Execute an async operation with a timeout guard.
    
    Args:
        coro: Coroutine to execute
        timeout_seconds: Maximum time to wait in seconds
        operation_name: Name of operation for logging
        raise_on_timeout: If True, raise TimeoutError; if False, return default_value
        default_value: Value to return if timeout occurs and raise_on_timeout=False
    
    Returns:
        Result of the coroutine or default_value on timeout
    
    Raises:
        TimeoutError: If operation times out and raise_on_timeout=True
    
    Example:
        result = await with_timeout(
            some_async_function(),
            timeout_seconds=30,
            operation_name="PDF extraction"
        )
    """
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_seconds)
        return result
    
    except asyncio.TimeoutError:
        error_msg = f"⏱️ {operation_name} timed out after {timeout_seconds}s"
        logger.error(error_msg)
        
        if raise_on_timeout:
            raise TimeoutError(error_msg)
        else:
            logger.warning(f"   Returning default value: {default_value}")
            return default_value


def timeout_guard(timeout_seconds: float, operation_name: Optional[str] = None):
    """
    Decorator to add timeout protection to async functions.
    
    Args:
        timeout_seconds: Maximum time to wait in seconds
        operation_name: Optional name for logging (defaults to function name)
    
    Example:
        @timeout_guard(timeout_seconds=60, operation_name="CLIP embedding")
        async def generate_clip_embedding(image_data):
            # ... long-running operation ...
            return embedding
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            return await with_timeout(
                func(*args, **kwargs),
                timeout_seconds=timeout_seconds,
                operation_name=op_name,
                raise_on_timeout=True
            )
        return wrapper
    return decorator


# Predefined timeout constants for common operations
class TimeoutConstants:
    """Standard timeout values for different operations"""
    
    # PDF Processing
    PDF_EXTRACTION_PER_PAGE = 10  # 10s per page
    PDF_FULL_EXTRACTION = 300  # 5min for full PDF
    PYMUPDF4LLM_EXTRACTION = 180  # 3min for PyMuPDF4LLM
    
    # AI Model Calls
    CLAUDE_API_CALL = 120  # 2min for Claude API
    LLAMA_VISION_CALL = 90  # 1.5min for Llama Vision
    CLIP_EMBEDDING = 30  # 30s for CLIP embedding
    GPT_API_CALL = 60  # 1min for GPT API
    
    # Database Operations
    DATABASE_QUERY = 30  # 30s for database query
    DATABASE_INSERT = 15  # 15s for database insert
    DATABASE_BATCH_INSERT = 60  # 1min for batch insert
    
    # Product Discovery
    PRODUCT_DISCOVERY_STAGE_0A = 60  # 1min for index scan
    PRODUCT_DISCOVERY_STAGE_0B = 300  # 5min for metadata extraction
    
    # Image Processing
    IMAGE_ANALYSIS = 45  # 45s per image
    IMAGE_UPLOAD = 20  # 20s per image upload
    
    # Chunking
    CHUNKING_OPERATION = 180  # 3min for chunking
    
    # Overall Pipeline
    FULL_PIPELINE = 1800  # 30min for full pipeline (safety net)


# Convenience functions for common operations
async def with_pdf_timeout(coro, operation_name: str = "PDF operation"):
    """Execute PDF operation with standard timeout"""
    return await with_timeout(
        coro,
        timeout_seconds=TimeoutConstants.PDF_FULL_EXTRACTION,
        operation_name=operation_name
    )


async def with_ai_timeout(coro, operation_name: str = "AI operation"):
    """Execute AI operation with standard timeout"""
    return await with_timeout(
        coro,
        timeout_seconds=TimeoutConstants.CLAUDE_API_CALL,
        operation_name=operation_name
    )


async def with_db_timeout(coro, operation_name: str = "Database operation"):
    """Execute database operation with standard timeout"""
    return await with_timeout(
        coro,
        timeout_seconds=TimeoutConstants.DATABASE_QUERY,
        operation_name=operation_name
    )

