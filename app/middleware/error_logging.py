"""
Comprehensive Error Logging Middleware

Provides centralized error logging with:
- Structured logging with correlation IDs
- Request/response context capture
- Error categorization and severity levels
- Integration with monitoring systems
- Performance metrics tracking
"""

import logging
import time
import traceback
import uuid
from datetime import datetime
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.schemas.unified_response import ErrorCode

logger = logging.getLogger(__name__)


class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive error logging and tracking.
    
    Features:
    - Automatic correlation ID generation
    - Request/response context capture
    - Error categorization by severity
    - Performance metrics tracking
    - Structured logging format
    """
    
    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = False,
        log_response_body: bool = False,
        max_body_length: int = 1000
    ):
        """
        Initialize error logging middleware.
        
        Args:
            app: ASGI application
            log_request_body: Whether to log request bodies
            log_response_body: Whether to log response bodies
            max_body_length: Maximum body length to log (truncate if longer)
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_length = max_body_length
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with comprehensive error logging."""
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        # Capture request start time
        start_time = time.time()

        # Build request context
        request_context = self._build_request_context(request, correlation_id)

        # Skip logging for system_logs table requests to avoid recursive logging
        is_system_logs_request = '/rest/v1/system_logs' in str(request.url.path)

        try:
            # Log request (skip for system_logs to avoid recursion)
            if not is_system_logs_request:
                logger.info(
                    f"[{correlation_id}] Incoming request: {request.method} {request.url.path}",
                    extra={"correlation_id": correlation_id, "request": request_context}
                )
            
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Log response (skip for system_logs to avoid recursion)
            if not is_system_logs_request:
                response_context = self._build_response_context(response, processing_time)

                if response.status_code >= 400:
                    logger.warning(
                        f"[{correlation_id}] Request failed: {request.method} {request.url.path} "
                        f"- Status: {response.status_code} - Time: {processing_time:.2f}ms",
                        extra={
                            "correlation_id": correlation_id,
                            "request": request_context,
                            "response": response_context
                        }
                    )
                else:
                    logger.info(
                        f"[{correlation_id}] Request completed: {request.method} {request.url.path} "
                        f"- Status: {response.status_code} - Time: {processing_time:.2f}ms",
                        extra={
                            "correlation_id": correlation_id,
                            "request": request_context,
                            "response": response_context
                        }
                    )
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Build error context
            error_context = self._build_error_context(e, request_context, processing_time)
            
            # Log error with full context
            logger.error(
                f"[{correlation_id}] Unhandled exception: {type(e).__name__}: {str(e)} "
                f"- Endpoint: {request.method} {request.url.path} - Time: {processing_time:.2f}ms",
                extra={
                    "correlation_id": correlation_id,
                    "error": error_context
                },
                exc_info=True
            )
            
            # Return structured error response
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": {
                        "code": ErrorCode.INTERNAL_ERROR,
                        "message": "An internal server error occurred",
                        "details": {
                            "correlation_id": correlation_id,
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        "retryable": True
                    },
                    "metadata": {
                        "correlation_id": correlation_id,
                        "processing_time": processing_time,
                        "timestamp": datetime.utcnow().isoformat(),
                        "endpoint": str(request.url.path)
                    }
                },
                headers={"X-Correlation-ID": correlation_id}
            )
    
    def _build_request_context(self, request: Request, correlation_id: str) -> Dict[str, Any]:
        """Build structured request context for logging."""
        return {
            "correlation_id": correlation_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_host": request.client.host if request.client else None,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _build_response_context(self, response: Response, processing_time: float) -> Dict[str, Any]:
        """Build structured response context for logging."""
        return {
            "status_code": response.status_code,
            "processing_time_ms": round(processing_time, 2),
            "headers": dict(response.headers)
        }

    def _build_error_context(
        self,
        error: Exception,
        request_context: Dict[str, Any],
        processing_time: float
    ) -> Dict[str, Any]:
        """Build structured error context for logging."""
        return {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_traceback": traceback.format_exc(),
            "request": request_context,
            "processing_time_ms": round(processing_time, 2),
            "timestamp": datetime.utcnow().isoformat()
        }


def get_correlation_id(request: Request) -> Optional[str]:
    """
    Get correlation ID from request state.

    Args:
        request: FastAPI request object

    Returns:
        Correlation ID if available, None otherwise
    """
    return getattr(request.state, "correlation_id", None)


