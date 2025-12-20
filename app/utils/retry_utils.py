"""
Retry Utilities with Exponential Backoff

Provides decorators and utilities for retrying operations with:
- Exponential backoff
- Jitter to prevent thundering herd
- Configurable retry conditions
- Detailed logging
"""

import asyncio
import logging
import random
import time
from functools import wraps
from typing import Callable, Type, Tuple, Optional

logger = logging.getLogger(__name__)


def exponential_backoff_with_jitter(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
) -> float:
    """
    Calculate exponential backoff delay with optional jitter.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter
    
    Returns:
        Delay in seconds
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    
    if jitter:
        # Add random jitter (±25%)
        delay = delay * (0.75 + random.random() * 0.5)
    
    return delay


def retry_async(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Decorator for retrying async functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exceptions to retry on
        on_retry: Optional callback function called on each retry
    
    Example:
        @retry_async(max_attempts=3, base_delay=1.0, exceptions=(TimeoutError, ConnectionError))
        async def fetch_data():
            return await api.get_data()
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # Last attempt failed
                        logger.error(
                            f"❌ {func.__name__} failed after {max_attempts} attempts: {e}",
                            exc_info=True
                        )
                        raise
                    
                    # Calculate backoff delay
                    delay = exponential_backoff_with_jitter(
                        attempt,
                        base_delay=base_delay,
                        max_delay=max_delay
                    )
                    
                    logger.warning(
                        f"⚠️ {func.__name__} attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        try:
                            await on_retry(attempt, e, delay)
                        except Exception as callback_error:
                            logger.error(f"Retry callback error: {callback_error}")
                    
                    # Wait before retry
                    await asyncio.sleep(delay)
            
            # Should never reach here, but just in case
            raise last_exception
        
        return wrapper
    return decorator


def retry_sync(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for retrying synchronous functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exceptions to retry on
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"❌ {func.__name__} failed after {max_attempts} attempts: {e}",
                            exc_info=True
                        )
                        raise
                    
                    delay = exponential_backoff_with_jitter(
                        attempt,
                        base_delay=base_delay,
                        max_delay=max_delay
                    )
                    
                    logger.warning(
                        f"⚠️ {func.__name__} attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator

