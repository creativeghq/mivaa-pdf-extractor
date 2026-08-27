"""
Logs API Routes

Endpoints for fetching and managing system logs from the database.

`POST /frontend` is deliberately the one route here that does NOT require auth: the
browser logger (`src/services/logger.service.ts`) sends errors with `keepalive` and
no Authorization header, and it swallows every failure — so requiring a token would
switch frontend error reporting off silently, which is worse than the endpoint being
open.

What it may not do is TRUST the caller (audit #24 M11-4). It accepted a body-supplied
`user_id`, an unconstrained `level` and unbounded fields, so any caller could write a
forged CRITICAL entry attributed to somebody else, at any size, into a table whose
only defence is a TTL. Attribution now comes from a bearer token when one is present
and is absent otherwise; it is never taken from the body.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from app.services.core.supabase_client import get_supabase_client
from app.schemas.api_responses import StatusResponse, LogStatsResponse
from app.dependencies import require_admin
from app.utils.postgrest_filters import escape_like

logger = logging.getLogger(__name__)

#: Bounds on the one unauthenticated write in this repository. Sizes are generous
#: enough for a real stack trace and small enough that the route is not a write
#: amplifier into a TTL'd table.
MAX_MESSAGE_CHARS = 4_000
MAX_LOGGER_NAME_CHARS = 200
MAX_URL_CHARS = 2_000
MAX_USER_AGENT_CHARS = 500
MAX_CONTEXT_BYTES = 16_000


router = APIRouter(prefix="/api/admin/logs", tags=["Admin", "Logs"])


class LogEntry(BaseModel):
    """Log entry model."""
    id: str
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    context: Optional[dict] = None
    job_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    created_at: datetime


class FrontendLogRequest(BaseModel):
    """Request model for frontend log submission.

    There is deliberately no `user_id` field. Attribution is derived from the bearer
    token when the caller presents one — a body-supplied id is a claim, not a fact,
    and this route is open.
    """
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    message: str = Field(..., max_length=MAX_MESSAGE_CHARS)
    logger_name: str = Field("frontend", max_length=MAX_LOGGER_NAME_CHARS)
    context: Optional[dict] = None
    url: Optional[str] = Field(None, max_length=MAX_URL_CHARS)
    user_agent: Optional[str] = Field(None, max_length=MAX_USER_AGENT_CHARS)

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, v: Any) -> Any:
        """Accept the frontend's casing without widening what is accepted."""
        return v.upper() if isinstance(v, str) else v

    @field_validator("context")
    @classmethod
    def _bound_context(cls, v: Optional[dict]) -> Optional[dict]:
        """`context` is a free-form dict, so its SIZE is the only thing that can be
        bounded — and it has to be, or the field is the amplifier the other caps
        removed."""
        if v is None:
            return v
        import json
        try:
            encoded = json.dumps(v, default=str)
        except (TypeError, ValueError):
            raise ValueError("context is not JSON-serialisable")
        if len(encoded.encode("utf-8")) > MAX_CONTEXT_BYTES:
            raise ValueError(f"context exceeds {MAX_CONTEXT_BYTES} bytes")
        return v


async def _attributed_user_id(request: Request) -> Optional[str]:
    """The caller's user id IF they presented a valid token, else None.

    Best-effort by design: this route stays open (see the module docstring), so an
    anonymous frontend error is a legitimate outcome and must record no user rather
    than an asserted one.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    try:
        from app.dependencies import _get_jwt_middleware
        claims = await _get_jwt_middleware()._validate_token(auth.split(" ", 1)[1])
    except Exception:
        return None
    if not claims:
        return None
    uid = claims.get("sub") or claims.get("user_id")
    return str(uid) if uid else None


class LogsResponse(BaseModel):
    """Response model for logs endpoint."""
    logs: List[LogEntry]
    total: int
    page: int
    page_size: int
    has_more: bool


@router.post("/frontend", responses={200: {"model": StatusResponse}})
async def log_frontend_error(log_request: FrontendLogRequest, request: Request):
    """
    Log a frontend error to the database.

    This endpoint allows the frontend to send errors to the same logging
    system as the backend, enabling unified error tracking.

    The logs are tagged with source='frontend' to distinguish them from backend logs.
    """
    try:
        supabase = get_supabase_client()
        user_id = await _attributed_user_id(request)

        # Prepare log entry
        log_entry: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "level": log_request.level.upper(),
            "logger_name": log_request.logger_name,
            "message": log_request.message,
            "context": {
                **(log_request.context or {}),
                "source": "frontend",  # Tag as frontend
                "url": log_request.url,
                "user_agent": log_request.user_agent or request.headers.get("user-agent"),
                "ip_address": request.client.host if request.client else None,
            },
            # From the token or nothing. Never from the body (M11-4).
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat()
        }

        # Insert into database
        supabase.client.table('system_logs').insert(log_entry).execute()

        return {
            "success": True,
            "log_id": log_entry["id"],
            "message": "Frontend log recorded successfully"
        }

    except Exception as e:
        # Don't fail the frontend if logging fails. `logger`, not `print`: a print
        # reaches no sink anyone can query, so the failure would be invisible.
        logger.warning("Failed to record frontend log: %s", e)
        return {
            "success": False,
            "error": str(e)
        }


@router.get("", response_model=LogsResponse, dependencies=[Depends(require_admin)])
async def get_logs(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=1000, description="Number of logs per page"),
    level: Optional[str] = Query(None, description="Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    logger_name: Optional[str] = Query(None, description="Filter by logger name"),
    job_id: Optional[str] = Query(None, description="Filter by job ID"),
    search: Optional[str] = Query(None, description="Search in message"),
    source: Optional[str] = Query(None, description="Filter by source (frontend or backend)"),
    hours: Optional[int] = Query(24, description="Number of hours to look back (default: 24)")
):
    """
    Get system logs from the database.
    
    Supports filtering by:
    - Log level
    - Logger name
    - Job ID
    - Search term in message
    - Time range (hours)
    
    Returns paginated results.
    """
    try:
        supabase = get_supabase_client()
        
        # Build query
        query = supabase.client.table('system_logs').select('*', count='exact')
        
        # Apply time filter
        if hours:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            query = query.gte('timestamp', cutoff_time.isoformat())
        
        # Apply filters
        if level:
            query = query.eq('level', level.upper())

        if logger_name:
            query = query.eq('logger_name', logger_name)

        if job_id:
            query = query.eq('job_id', job_id)

        if search:
            query = query.ilike('message', f'%{escape_like(search)}%')

        if source:
            # Filter by source (frontend or backend) using context->source
            query = query.contains('context', {'source': source})
        
        # Get total count
        count_response = query.execute()
        total = count_response.count if hasattr(count_response, 'count') else 0
        
        # Apply pagination and ordering
        offset = (page - 1) * page_size
        query = query.order('timestamp', desc=True).range(offset, offset + page_size - 1)
        
        # Execute query
        response = query.execute()
        logs = response.data
        
        # Calculate has_more
        has_more = (offset + len(logs)) < total
        
        return LogsResponse(
            logs=logs,
            total=total,
            page=page,
            page_size=page_size,
            has_more=has_more
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {str(e)}")


@router.delete("", responses={200: {"model": StatusResponse}}, dependencies=[Depends(require_admin)])
async def clear_logs(
    hours: Optional[int] = Query(None, description="Clear logs older than N hours (if not specified, clears all)")
):
    """
    Clear system logs.
    
    If hours is specified, only clears logs older than that many hours.
    Otherwise, clears all logs.
    """
    try:
        supabase = get_supabase_client()
        
        if hours:
            # Delete logs older than specified hours
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            response = supabase.client.table('system_logs').delete().lt('timestamp', cutoff_time.isoformat()).execute()
        else:
            # Delete all logs
            response = supabase.client.table('system_logs').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
        
        deleted_count = len(response.data) if response.data else 0
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Cleared {deleted_count} log entries"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear logs: {str(e)}")


@router.get("/stats", responses={200: {"model": LogStatsResponse}}, dependencies=[Depends(require_admin)])
async def get_log_stats(
    hours: int = Query(24, description="Number of hours to analyze")
):
    """
    Get statistics about logs.
    
    Returns:
    - Total logs
    - Breakdown by level
    - Top loggers
    - Recent errors
    """
    try:
        supabase = get_supabase_client()
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Get all logs in time range
        response = supabase.client.table('system_logs').select('*').gte('timestamp', cutoff_time.isoformat()).execute()
        logs = response.data
        
        # Calculate stats
        total = len(logs)
        by_level = {}
        by_logger = {}
        
        for log in logs:
            level = log.get('level', 'UNKNOWN')
            logger_name = log.get('logger_name', 'unknown')
            
            by_level[level] = by_level.get(level, 0) + 1
            by_logger[logger_name] = by_logger.get(logger_name, 0) + 1
        
        # Get top loggers
        top_loggers = sorted(by_logger.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_logs": total,
            "time_range_hours": hours,
            "by_level": by_level,
            "top_loggers": [{"logger": name, "count": count} for name, count in top_loggers]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get log stats: {str(e)}")


