"""
Retry Helper Utility

Provides decorators and utilities for retrying operations with exponential backoff.
Specifically designed to handle DNS resolution failures and network errors.

Author: Material Kai Vision Platform
Created: 2025-11-26
"""

import asyncio
import logging
import re
import functools
from typing import Callable, TypeVar, Tuple, Type
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


#: A gateway error arrives as a whole HTML page. Cloudflare's 522 for a Supabase origin
#: timeout is ~6 KB of markup, and PostgREST hands it back inside an APIError's `details`.
_HTML_PAGE_RE = re.compile(r"<!DOCTYPE.*|<html[^a-zA-Z].*", re.IGNORECASE | re.DOTALL)


def describe_exception(exc: BaseException, limit: int = 300) -> str:
    """One line naming an exception, safe to put in a log message.

    Interpolated raw, a Cloudflare 522 became the Sentry ISSUE TITLE (MIVAA-5K3: six
    kilobytes of markup where the error should be) and a 6 KB `system_logs` row for every
    retry decision taken during the outage. The diagnosis is in the first line; the page
    is not diagnosis.

    Also prepends the exception TYPE and, where there is one, the PostgREST/Postgres error
    code — neither of which a bare `{e}` carried, and the code is the thing that says
    whether a repeat could ever have helped.
    """
    text = _HTML_PAGE_RE.sub("<html error page omitted>", str(exc))
    text = " ".join(text.split())
    if len(text) > limit:
        text = f"{text[:limit]}… (+{len(text) - limit} more chars)"

    label = type(exc).__name__
    code = _error_code(exc) if isinstance(exc, Exception) else ""
    if code:
        label = f"{label}[{code}]"
    return f"{label}: {text}"


def is_dns_error(exception: Exception) -> bool:
    """
    Check if an exception is a DNS resolution error.
    
    Args:
        exception: Exception to check
    
    Returns:
        True if exception is DNS-related, False otherwise
    """
    error_str = str(exception).lower()
    # No bare 'dns': three letters that match any message happening to contain them, which
    # is exactly the accident this classifier now exists to stop making.
    dns_indicators = [
        'temporary failure in name resolution',
        'errno -3',
        '[errno -3]',
        'name resolution',
        'dns lookup',
        'dns resolution',
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
    # No bare 'connect' / 'reset' / 'refused': a CHECK violation quotes the failing ROW
    # back, and business text mentioning a connection or a password reset is not a network
    # fault. Every token here names a transport failure on its own.
    connection_indicators = [
        'server disconnected',
        'connection refused',
        'connection reset',
        'connection aborted',
        'connection timed out',
        'connection error',
        'read timeout',
        'write timeout',
        'pool timeout',
        'timed out',
        'broken pipe'
    ]
    return any(indicator in error_str for indicator in connection_indicators)


#: SQLSTATE classes Postgres uses for a definitive REJECTION of this request.
#: Repeating one changes nothing: the row still violates the constraint, the role still
#: lacks the grant, the column still does not exist.
_PERMANENT_SQLSTATE_CLASSES = frozenset({
    "22",  # data exception — bad cast, value out of range
    "23",  # integrity constraint violation — CHECK, FK, NOT NULL, unique
    "28",  # invalid authorization specification
    "42",  # syntax error or access rule violation — includes 42501 (RLS denial)
    "44",  # WITH CHECK OPTION violation
})

#: SQLSTATE classes that ARE worth another attempt.
_TRANSIENT_SQLSTATE_CLASSES = frozenset({
    "08",  # connection exception
    "53",  # insufficient resources — 53300 too many connections
    "57",  # operator intervention — 57014 canceled, 57P01 admin shutdown
    "58",  # system error
})

#: Individually transient, in classes that are otherwise permanent.
_TRANSIENT_SQLSTATES = frozenset({"40001", "40P01", "55P03"})

#: Gateway / rate-limit statuses. Supabase sits behind Cloudflare, which answers 522 when
#: the origin times out; PostgREST surfaces that as an APIError whose `code` is the HTTP
#: status rather than a SQLSTATE.
_TRANSIENT_HTTP_STATUS = frozenset({408, 425, 429, 502, 503, 504, 522, 524})

_SQLSTATE_CHARS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _error_code(exception: Exception) -> str:
    """The structured error code an exception carries, or '' if it carries none.

    postgrest's APIError exposes `.code`; some paths only have the dict in `args[0]`.
    """
    code = getattr(exception, "code", None)
    if code is None:
        args = getattr(exception, "args", ()) or ()
        if args and isinstance(args[0], dict):
            code = args[0].get("code")
    if code is None:
        return ""
    return str(code).strip()


def classify_error_code(code: str):
    """True = retry, False = never retry, None = the code says nothing either way.

    This is the half that must run BEFORE any text matching. The text of a rejection is
    not ours to control: a CHECK violation quotes the FAILING ROW back at you, so a row
    that merely happens to mention a connection or a timeout used to be classified as a
    transient network fault. That is how a permanent 23514 on
    `agent_run_logs_level_check` was reported as "PostgREST transient failure"
    (MIVAA-5JV) — the words came from the data, not from the fault.
    """
    if not code:
        return None

    # PostgREST's own codes (PGRST116 no rows, PGRST301 JWT expired) are all decisions
    # about this exact request. None of them changes with a second attempt.
    if code.upper().startswith("PGRST"):
        return False

    if code in _TRANSIENT_SQLSTATES:
        return True

    is_sqlstate = len(code) == 5 and all(c in _SQLSTATE_CHARS for c in code.upper())
    if is_sqlstate:
        cls = code[:2].upper()
        if cls in _TRANSIENT_SQLSTATE_CLASSES:
            return True
        if cls in _PERMANENT_SQLSTATE_CLASSES:
            return False
        return None

    if code.isdigit():
        return int(code) in _TRANSIENT_HTTP_STATUS

    return None


def should_retry_exception(exception: Exception) -> bool:
    """
    Determine if an exception should trigger a retry.

    Order matters. A structured error code is a statement by the server about THIS
    request and settles the question; only when there is no code do we fall back to
    reading the message, which is guesswork over text we did not write.

    Args:
        exception: Exception to evaluate

    Returns:
        True if exception should be retried, False otherwise
    """
    verdict = classify_error_code(_error_code(exception))
    if verdict is not None:
        return verdict

    # Transport-level failures, by type. `RemoteProtocolError` is the one that actually
    # matters here — "Server disconnected without sending a response" after an idle gap —
    # and it is NOT an httpx.NetworkError, so it used to be caught only by the accident of
    # "disconnected" containing the substring "connect".
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is always present at runtime
        httpx = None
    if httpx is not None and isinstance(exception, (
        httpx.TimeoutException,
        httpx.NetworkError,
        httpx.RemoteProtocolError,
        httpx.ProxyError,
    )):
        return True

    # Text heuristics last, and only for an exception that told us nothing structured.
    if is_dns_error(exception):
        return True
    if is_connection_error(exception):
        return True

    return False


async def execute_db_with_retry(
    query_factory: Callable[[], T],
    *,
    label: str = "db_query",
    max_retries: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 8.0,
) -> T:
    """
    Run a synchronous Supabase/PostgREST query with retry on transient
    connection errors.

    PostgREST keeps a pooled keep-alive HTTP connection; after a long idle
    period (e.g. between background-cron ticks) the server closes it, so the
    next query raises httpx "Server disconnected" / ConnectError. Re-issuing
    the request transparently establishes a fresh connection, so a short
    backoff retry is all that's needed.

    `query_factory` MUST build AND execute the query (a zero-arg callable that
    returns the PostgREST response). It is re-invoked from scratch on every
    attempt so each retry issues a brand-new request rather than replaying a
    consumed builder. Non-retryable exceptions (per `should_retry_exception`)
    are raised immediately without burning attempts.
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):  # +1 for the initial attempt
        try:
            result = query_factory()
            if attempt > 0:
                logger.info(
                    f"✅ {label} succeeded on attempt {attempt + 1}/{max_retries + 1}"
                )
            return result
        except Exception as e:
            last_exception = e
            if attempt < max_retries and should_retry_exception(e):
                logger.warning(
                    f"⚠️ {label} transient connection failure "
                    f"(attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, max_delay)
                continue
            raise

    # Unreachable, but keeps type-checkers happy.
    raise last_exception


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

