"""
Logs API Routes

Endpoints for fetching and managing system logs from the database.
"""

from fastapi import APIRouter, HTTPException, Query, Request
from typing import Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel
import uuid

from app.database.supabase_client import get_supabase_client


router = APIRouter(prefix="/admin/logs", tags=["admin", "logs"])


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
    """Request model for frontend log submission."""
    level: str  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    message: str
    logger_name: str = "frontend"
    context: Optional[dict] = None
    user_id: Optional[str] = None
    url: Optional[str] = None
    user_agent: Optional[str] = None


class LogsResponse(BaseModel):
    """Response model for logs endpoint."""
    logs: List[LogEntry]
    total: int
    page: int
    page_size: int
    has_more: bool


@router.post("/frontend")
async def log_frontend_error(log_request: FrontendLogRequest, request: Request):
    """
    Log a frontend error to the database.

    This endpoint allows the frontend to send errors to the same logging
    system as the backend, enabling unified error tracking.

    The logs are tagged with source='frontend' to distinguish them from backend logs.
    """
    try:
        supabase = get_supabase_client()

        # Prepare log entry
        log_entry = {
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
            "user_id": log_request.user_id,
            "created_at": datetime.utcnow().isoformat()
        }

        # Insert into database
        response = supabase.table('system_logs').insert(log_entry).execute()

        return {
            "success": True,
            "log_id": log_entry["id"],
            "message": "Frontend log recorded successfully"
        }

    except Exception as e:
        # Don't fail the frontend if logging fails
        print(f"Failed to log frontend error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


@router.get("", response_model=LogsResponse)
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
        query = supabase.table('system_logs').select('*', count='exact')
        
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
            query = query.ilike('message', f'%{search}%')

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


@router.delete("")
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
            response = supabase.table('system_logs').delete().lt('timestamp', cutoff_time.isoformat()).execute()
        else:
            # Delete all logs
            response = supabase.table('system_logs').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
        
        deleted_count = len(response.data) if response.data else 0
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Cleared {deleted_count} log entries"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear logs: {str(e)}")


@router.get("/stats")
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
        response = supabase.table('system_logs').select('*').gte('timestamp', cutoff_time.isoformat()).execute()
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

