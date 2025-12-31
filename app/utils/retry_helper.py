"""
Retry Helper Utility

Provides decorators and utilities for retrying operations with exponential backoff.
Specifically designed to handle DNS resolution failures and network errors.

Author: Material Kai Vision Platform
Created: 2025-11-26
"""

import asyncio
import logging
import functools
from typing import Callable, TypeVar, Any, Tuple, Type
from datetime import datetime

logger = logging.getLogger(__name__)

# Type variable for generic function return type
T = TypeVar('T')


def async_retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    max_delay: float = 10.0,
    retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_retries: bool = True
):
    """
    Decorator for async functions to retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
        max_delay: Maximum delay between retries in seconds (default: 10.0)
        retry_on_exceptions: Tuple of exception types to retry on (default: all exceptions)
        log_retries: Whether to log retry attempts (default: True)
    
    Returns:
        Decorated async function with retry logic
    
    Example:
        @async_retry_with_backoff(max_retries=3, initial_delay=1.0)
        async def fetch_data():
            # This will retry up to 3 times with exponential backoff
            return await api_call()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    # Attempt the function call
                    result = await func(*args, **kwargs)
                    
                    # Log success if this was a retry
                    if attempt > 0 and log_retries:
                        logger.info(
                            f"✅ {func.__name__} succeeded on attempt {attempt + 1}/{max_retries + 1}"
                        )
                    
                    return result
                    
                except retry_on_exceptions as e:
                    last_exception = e
                    
                    # Check if we should retry
                    if attempt < max_retries:
                        if log_retries:
                            logger.warning(
                                f"⚠️ {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}"
                            )
                            logger.warning(f"   Retrying in {delay:.1f} seconds...")
                        
                        # Wait before retrying
                        await asyncio.sleep(delay)
                        
                        # Calculate next delay with exponential backoff
                        delay = min(delay * backoff_multiplier, max_delay)
                    else:
                        # Max retries reached
                        if log_retries:
                            logger.error(
                                f"❌ {func.__name__} failed after {max_retries + 1} attempts: {str(e)}"
                            )
                        raise last_exception
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator


def is_dns_error(exception: Exception) -> bool:
    """
    Check if an exception is a DNS resolution error.
    
    Args:
        exception: Exception to check
    
    Returns:
        True if exception is DNS-related, False otherwise
    """
    error_str = str(exception).lower()
    dns_indicators = [
        'temporary failure in name resolution',
        'errno -3',
        '[errno -3]',
        'name resolution',
        'dns',
        'getaddrinfo failed'
    ]
    return any(indicator in error_str for indicator in dns_indicators)


def is_connection_error(exception: Exception) -> bool:
    """
    Check if an exception is a connection error.
    
    Args:
        exception: Exception to check
    
    Returns:
        True if exception is connection-related, False otherwise
    """
    error_str = str(exception).lower()
    connection_indicators = [
        'connection',
        'timeout',
        'timed out',
        'connect',
        'refused',
        'reset',
        'broken pipe'
    ]
    return any(indicator in error_str for indicator in connection_indicators)


def should_retry_exception(exception: Exception) -> bool:
    """
    Determine if an exception should trigger a retry.
    
    Args:
        exception: Exception to evaluate
    
    Returns:
        True if exception should be retried, False otherwise
    """
    # Check for DNS errors
    if is_dns_error(exception):
        return True
    
    # Check for connection errors
    if is_connection_error(exception):
        return True
    
    # Check exception type
    import httpx
    if isinstance(exception, (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx.NetworkError
    )):
        return True
    
    return False


# Specialized decorator for database operations
async_retry_db_operation = async_retry_with_backoff(
    max_retries=3,
    initial_delay=1.0,
    backoff_multiplier=2.0,
    max_delay=10.0,
    retry_on_exceptions=(Exception,),  # Retry on all exceptions, but check with should_retry_exception
    log_retries=True
)


# Specialized decorator for API calls
async_retry_api_call = async_retry_with_backoff(
    max_retries=5,
    initial_delay=0.5,
    backoff_multiplier=2.0,
    max_delay=30.0,
    retry_on_exceptions=(Exception,),
    log_retries=True
)


class RetryStats:
    """Track retry statistics for monitoring."""
    
    def __init__(self):
        self.total_retries = 0
        self.successful_retries = 0
        self.failed_retries = 0
        self.dns_errors = 0
        self.connection_errors = 0
        self.last_error_time = None
        self.last_error_message = None
    
    def record_retry(self, exception: Exception, success: bool):
        """Record a retry attempt."""
        self.total_retries += 1
        
        if success:
            self.successful_retries += 1
        else:
            self.failed_retries += 1
            self.last_error_time = datetime.utcnow()
            self.last_error_message = str(exception)
        
        if is_dns_error(exception):
            self.dns_errors += 1
        elif is_connection_error(exception):
            self.connection_errors += 1
    
    def get_stats(self) -> dict:
        """Get retry statistics."""
        return {
            'total_retries': self.total_retries,
            'successful_retries': self.successful_retries,
            'failed_retries': self.failed_retries,
            'dns_errors': self.dns_errors,
            'connection_errors': self.connection_errors,
            'success_rate': (
                self.successful_retries / self.total_retries 
                if self.total_retries > 0 else 0.0
            ),
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
            'last_error_message': self.last_error_message
        }


# Global retry stats instance
retry_stats = RetryStats()

