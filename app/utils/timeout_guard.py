"""
Timeout Guard Utility

Provides timeout protection for async operations to prevent indefinite hangs.
All critical async operations should be wrapped with these guards.
"""

import asyncio
import logging
import psutil
from typing import TypeVar, Callable, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)

def get_memory_multiplier() -> float:
    """
    Calculate timeout multiplier based on current memory usage.
    
    Returns:
        float: Multiplier for timeout values (1.0 to 3.0)
        - < 60% memory: 1.0x (normal)
        - 60-80% memory: 1.5x (moderate pressure)
        - 80-90% memory: 2.0x (high pressure)
        - > 90% memory: 3.0x (critical pressure)
    """
    try:
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        
        if mem_percent < 60:
            return 1.0
        elif mem_percent < 80:
            multiplier = 1.5
            logger.debug(f"⚠️ Moderate memory pressure ({mem_percent:.1f}%) - increasing timeouts by {multiplier}x")
            return multiplier
        elif mem_percent < 90:
            multiplier = 2.0
            logger.warning(f"🔴 High memory pressure ({mem_percent:.1f}%) - increasing timeouts by {multiplier}x")
            return multiplier
        else:
            multiplier = 3.0
            logger.error(f"🔴🔴 CRITICAL memory pressure ({mem_percent:.1f}%) - increasing timeouts by {multiplier}x")
            return multiplier
    except Exception as e:
        logger.warning(f"Failed to get memory stats: {e}, using default multiplier 1.0")
        return 1.0


def get_memory_aware_timeout(base_timeout: float, operation_name: str = "") -> float:
    """
    Get memory-aware timeout value.
    
    Args:
        base_timeout: Base timeout in seconds
        operation_name: Optional operation name for logging
    
    Returns:
        Adjusted timeout based on current memory pressure
    """
    multiplier = get_memory_multiplier()
    adjusted_timeout = base_timeout * multiplier
    
    if multiplier > 1.0:
        logger.info(f"⏱️ {operation_name} timeout adjusted: {base_timeout}s → {adjusted_timeout:.0f}s (memory pressure)")
    
    return adjusted_timeout


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
    # Apply memory-aware timeout adjustment
    timeout_seconds = get_memory_aware_timeout(timeout_seconds, operation_name)
    
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


class ProgressiveTimeoutStrategy:
    """
    Progressive timeout strategy that adjusts timeouts based on document characteristics.

    Timeouts scale with:
    - Document size (page count, file size)
    - Processing stage (later stages get more time)
    - Complexity (number of images, products)
    """

    @staticmethod
    def calculate_pdf_extraction_timeout(page_count: int, file_size_mb: float) -> float:
        """
        Calculate timeout for PDF extraction based on document size.

        Base: 20s per page (increased from 10s due to memory optimizations)
        Scaling: +5s per page for large PDFs (>50 pages)
        Max: 60min (increased from 30min)
        """
        base_timeout = page_count * 20  # 20s per page (memory-optimized processing is slower)

        # Add extra time for large PDFs
        if page_count > 50:
            extra_time = (page_count - 50) * 5  # Increased from 2s to 5s
            base_timeout += extra_time

        # Add time based on file size (2s per MB, increased from 1s)
        base_timeout += file_size_mb * 2

        # Cap at 60 minutes (increased from 30min to accommodate slower memory-optimized processing)
        return min(base_timeout, 3600)

    @staticmethod
    def calculate_product_discovery_timeout(page_count: int, categories: list) -> float:
        """
        Calculate timeout for product discovery based on pages and categories.

        Base: 60s for index scan
        Scaling: +30s per 10 pages for metadata extraction
        Categories: +30s per category
        Max: 10min
        """
        base_timeout = 60  # 1min for index scan

        # Add time for metadata extraction (30s per 10 pages)
        metadata_time = (page_count / 10) * 30
        base_timeout += metadata_time

        # Add time per category
        category_time = len(categories) * 30
        base_timeout += category_time

        # Cap at 10 minutes
        return min(base_timeout, 600)

    @staticmethod
    def calculate_chunking_timeout(page_count: int, chunk_size: int = 512) -> float:
        """
        Calculate timeout for chunking based on document size.

        Base: 30s
        Scaling: +10s per 10 pages
        Max: 5min
        """
        base_timeout = 30

        # Add time per 10 pages
        page_time = (page_count / 10) * 10
        base_timeout += page_time

        # Cap at 5 minutes
        return min(base_timeout, 300)

    @staticmethod
    def calculate_image_processing_timeout(image_count: int, concurrent_limit: int = 5) -> float:
        """
        Calculate timeout for image processing based on image count.

        Base: 45s per image (CLIP + Llama Vision)
        Parallel: Divide by concurrency limit
        Buffer: +20% safety margin
        Max: 30min
        """
        # Time per image (45s for CLIP + Llama)
        total_time = image_count * 45

        # Account for parallel processing
        parallel_time = total_time / concurrent_limit

        # Add 20% safety margin
        timeout = parallel_time * 1.2

        # Cap at 30 minutes
        return min(timeout, 1800)

    @staticmethod
    def calculate_stage_timeout(
        stage: str,
        page_count: int = 0,
        image_count: int = 0,
        file_size_mb: float = 0,
        categories: list = None
    ) -> float:
        """
        Calculate timeout for a specific processing stage.

        Args:
            stage: Processing stage name
            page_count: Number of pages in document
            image_count: Number of images to process
            file_size_mb: File size in MB
            categories: List of extraction categories

        Returns:
            Timeout in seconds for the stage
        """
        categories = categories or ['products']

        stage_calculators = {
            'pdf_extraction': lambda: ProgressiveTimeoutStrategy.calculate_pdf_extraction_timeout(
                page_count, file_size_mb
            ),
            'product_discovery': lambda: ProgressiveTimeoutStrategy.calculate_product_discovery_timeout(
                page_count, categories
            ),
            'chunking': lambda: ProgressiveTimeoutStrategy.calculate_chunking_timeout(page_count),
            'image_processing': lambda: ProgressiveTimeoutStrategy.calculate_image_processing_timeout(image_count),
        }

        calculator = stage_calculators.get(stage)
        if calculator:
            timeout = calculator()
            return timeout

        # Default fallback
        return TimeoutConstants.FULL_PIPELINE

