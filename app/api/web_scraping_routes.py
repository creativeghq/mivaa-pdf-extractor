"""
Web Scraping API Routes

Endpoints for processing Firecrawl scraping sessions:
- POST /api/scraping/process-session - Process a scraping session and create products
- GET /api/scraping/session/{session_id}/status - Get session processing status
- POST /api/scraping/session/{session_id}/retry - Retry failed session processing
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Path
from pydantic import BaseModel, Field
from datetime import datetime

from app.services.integrations.web_scraping_service import WebScrapingService
from app.services.core.supabase_client import get_supabase_client
from app.services.core.async_queue_service import AsyncQueueService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scraping", tags=["web-scraping"])


# ============================================================================
# Request/Response Models
# ============================================================================

class ProcessSessionRequest(BaseModel):
    """Request model for processing a scraping session"""
    session_id: str = Field(..., description="Scraping session ID")
    workspace_id: str = Field(..., description="Workspace ID")
    categories: Optional[list[str]] = Field(default=None, description="Categories to discover (default: ['products'])")
    model: Optional[str] = Field(default="claude", description="AI model to use (claude, gpt)")


class SessionStatusResponse(BaseModel):
    """Scraping session status response"""
    session_id: str
    status: str  # pending, processing, completed, failed
    source_url: str
    progress_percentage: Optional[int]
    total_pages: Optional[int]
    completed_pages: Optional[int]
    failed_pages: Optional[int]
    materials_processed: Optional[int]
    products_created: Optional[int]
    error_message: Optional[str]
    created_at: str
    updated_at: str


class ProcessSessionResponse(BaseModel):
    """Process session response"""
    success: bool
    session_id: str
    message: str
    job_id: Optional[str] = None
    products_created: Optional[int] = None


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/process-session")
async def process_scraping_session(
    request: ProcessSessionRequest,
    background_tasks: BackgroundTasks
) -> ProcessSessionResponse:
    """
    Process a Firecrawl scraping session and create products.

    This endpoint is called by the Edge Function (scrape-session-manager) when
    scraping completes. It processes the session in the background using
    ProductDiscoveryService.discover_products_from_text().

    Args:
        request: Process session request
        background_tasks: FastAPI background tasks

    Returns:
        Processing status with job ID
    """
    try:
        logger.info(f"🚀 Starting scraping session processing: {request.session_id}")

        # Initialize service
        scraping_service = WebScrapingService(model=request.model)

        # Validate session exists
        supabase = get_supabase_client()
        session_response = supabase.client.table("scraping_sessions").select("*").eq("id", request.session_id).single().execute()

        if not session_response.data:
            raise HTTPException(status_code=404, detail=f"Scraping session not found: {request.session_id}")

        session = session_response.data

        # Update session status to processing
        supabase.client.table("scraping_sessions").update({
            "status": "processing",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", request.session_id).execute()

        # Process in background
        async def process_session_background():
            try:
                result = await scraping_service.process_scraping_session(
                    session_id=request.session_id,
                    workspace_id=request.workspace_id,
                    categories=request.categories or ["products"]
                )
                logger.info(f"✅ Session processing complete: {result}")
            except Exception as e:
                logger.error(f"❌ Session processing failed: {e}")
                raise

        background_tasks.add_task(process_session_background)

        return ProcessSessionResponse(
            success=True,
            session_id=request.session_id,
            message="Session processing started in background",
            job_id=None  # TODO: Integrate with AsyncQueueService for job tracking
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start session processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start session processing: {str(e)}")


@router.get("/session/{session_id}/status")
async def get_session_status(
    session_id: str = Path(..., description="Scraping session ID")
) -> SessionStatusResponse:
    """
    Get scraping session processing status.

    Args:
        session_id: Scraping session ID

    Returns:
        Session status with progress information
    """
    try:
        supabase = get_supabase_client()

        # Fetch session
        response = supabase.client.table("scraping_sessions").select("*").eq("id", session_id).single().execute()

        if not response.data:
            raise HTTPException(status_code=404, detail=f"Scraping session not found: {session_id}")

        session = response.data

        return SessionStatusResponse(
            session_id=session["id"],
            status=session["status"],
            source_url=session["source_url"],
            progress_percentage=session.get("progress_percentage"),
            total_pages=session.get("total_pages"),
            completed_pages=session.get("completed_pages"),
            failed_pages=session.get("failed_pages"),
            materials_processed=session.get("materials_processed"),
            products_created=session.get("materials_processed"),  # Same as materials_processed for now
            error_message=session.get("scraping_config", {}).get("error") if isinstance(session.get("scraping_config"), dict) else None,
            created_at=session["created_at"],
            updated_at=session["updated_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session status: {str(e)}")


@router.post("/session/{session_id}/retry")
async def retry_session_processing(
    session_id: str = Path(..., description="Scraping session ID"),
    background_tasks: BackgroundTasks = None
) -> ProcessSessionResponse:
    """
    Retry processing a failed scraping session.

    Args:
        session_id: Scraping session ID
        background_tasks: FastAPI background tasks

    Returns:
        Processing status
    """
    try:
        logger.info(f"🔄 Retrying scraping session: {session_id}")

        supabase = get_supabase_client()

        # Fetch session
        response = supabase.client.table("scraping_sessions").select("*").eq("id", session_id).single().execute()

        if not response.data:
            raise HTTPException(status_code=404, detail=f"Scraping session not found: {session_id}")

        session = response.data

        # Validate session can be retried
        if session["status"] not in ["failed", "completed"]:
            raise HTTPException(
                status_code=400,
                detail=f"Session status is '{session['status']}', can only retry 'failed' or 'completed' sessions"
            )

        # Reset session status
        supabase.client.table("scraping_sessions").update({
            "status": "processing",
            "materials_processed": 0,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", session_id).execute()

        # Initialize service
        scraping_service = WebScrapingService(model="claude")

        # Process in background
        async def retry_session_background():
            try:
                result = await scraping_service.process_scraping_session(
                    session_id=session_id,
                    workspace_id=session.get("workspace_id") or "default",  # TODO: Get from session
                    categories=["products"]
                )
                logger.info(f"✅ Session retry complete: {result}")
            except Exception as e:
                logger.error(f"❌ Session retry failed: {e}")
                raise

        if background_tasks:
            background_tasks.add_task(retry_session_background)

        return ProcessSessionResponse(
            success=True,
            session_id=session_id,
            message="Session retry started in background"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retry session: {str(e)}")


