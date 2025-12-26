"""
Circuit Breaker Pattern for AI API Calls

Prevents cascading failures by failing fast when AI APIs are down or slow.
Automatically recovers when service is healthy again.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is down, fail fast without calling API
- HALF_OPEN: Testing if service recovered, allow limited requests

Example:
    breaker = CircuitBreaker(
        failure_threshold=5,  # Open after 5 failures
        timeout_seconds=60,   # Stay open for 60s
        recovery_timeout=30   # Test recovery after 30s
    )
    
    result = await breaker.call(
        some_ai_api_function,
        arg1, arg2,
        kwarg1=value1
    )
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Callable, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Calls rejected while OPEN
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is OPEN"""
    pass


class CircuitBreaker:
    """
    Circuit breaker for AI API calls.
    
    Prevents cascading failures by failing fast when service is unhealthy.
    """
    
    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        recovery_timeout: int = 30,
        half_open_max_calls: int = 3
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name for logging
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: How long to stay OPEN before trying HALF_OPEN
            recovery_timeout: How long to wait in HALF_OPEN before closing
            half_open_max_calls: Max calls to allow in HALF_OPEN state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
        self.last_failure_time: Optional[float] = None
        self.state_changed_time = time.time()
        
        self.stats = CircuitBreakerStats()
        
        logger.info(f"🔌 Circuit breaker '{name}' initialized (threshold={failure_threshold}, timeout={timeout_seconds}s)")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
        
        Returns:
            Result of func
        
        Raises:
            CircuitBreakerError: If circuit is OPEN
            Exception: Original exception from func
        """
        self.stats.total_calls += 1
        
        # Check if circuit should transition states
        await self._check_state_transition()
        
        # If OPEN, fail fast
        if self.state == CircuitState.OPEN:
            self.stats.rejected_calls += 1
            error_msg = f"🔌 Circuit breaker '{self.name}' is OPEN - failing fast"
            logger.warning(error_msg)
            raise CircuitBreakerError(error_msg)
        
        # If HALF_OPEN, limit concurrent calls
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                self.stats.rejected_calls += 1
                error_msg = f"🔌 Circuit breaker '{self.name}' is HALF_OPEN - max calls reached"
                logger.warning(error_msg)
                raise CircuitBreakerError(error_msg)
            
            self.half_open_calls += 1
        
        # Execute function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        
        except Exception as e:
            await self._on_failure(e)
            raise
    
    async def _on_success(self):
        """Handle successful call"""
        self.stats.successful_calls += 1
        self.stats.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            # Successful call in HALF_OPEN - close circuit
            logger.info(f"✅ Circuit breaker '{self.name}' recovered - closing circuit")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
            self.state_changed_time = time.time()
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    async def _on_failure(self, error: Exception):
        """Handle failed call"""
        self.stats.failed_calls += 1
        self.stats.last_failure_time = time.time()
        self.last_failure_time = time.time()
        self.failure_count += 1
        
        logger.warning(f"❌ Circuit breaker '{self.name}' failure {self.failure_count}/{self.failure_threshold}: {error}")
        
        if self.failure_count >= self.failure_threshold:
            # Open circuit
            logger.error(f"🔌 Circuit breaker '{self.name}' OPENED after {self.failure_count} failures")
            self.state = CircuitState.OPEN
            self.state_changed_time = time.time()
    
    async def _check_state_transition(self):
        """Check if circuit should transition to different state"""
        now = time.time()
        time_in_state = now - self.state_changed_time
        
        if self.state == CircuitState.OPEN and time_in_state >= self.timeout_seconds:
            # Try recovery
            logger.info(f"🔌 Circuit breaker '{self.name}' transitioning to HALF_OPEN (testing recovery)")
            self.state = CircuitState.HALF_OPEN
            self.half_open_calls = 0
            self.state_changed_time = now
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "stats": {
                "total_calls": self.stats.total_calls,
                "successful_calls": self.stats.successful_calls,
                "failed_calls": self.stats.failed_calls,
                "rejected_calls": self.stats.rejected_calls,
                "success_rate": (
                    self.stats.successful_calls / self.stats.total_calls * 100
                    if self.stats.total_calls > 0 else 0
                )
            }
        }


# Global circuit breakers for different AI services
# Llama: More lenient due to transient JSON formatting issues (not actual API failures)
claude_breaker = CircuitBreaker(name="Claude API", failure_threshold=3, timeout_seconds=120)
llama_breaker = CircuitBreaker(name="Llama Vision", failure_threshold=8, timeout_seconds=45, recovery_timeout=30, half_open_max_calls=3)
clip_breaker = CircuitBreaker(name="CLIP Embeddings", failure_threshold=5, timeout_seconds=60)
gpt_breaker = CircuitBreaker(name="GPT API", failure_threshold=3, timeout_seconds=120)

