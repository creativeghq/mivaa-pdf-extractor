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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scraping", tags=["Web Scraping"])


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

@router.post("/process-session", response_model=ProcessSessionResponse)
async def process_scraping_session(
    request: ProcessSessionRequest,
    background_tasks: BackgroundTasks
) -> ProcessSessionResponse:
    """
    Process a Firecrawl scraping session and create products.

    Called by the Edge Function (scrape-session-manager) when scraping completes.
    Creates a background job for tracking, then processes the session via
    WebScrapingService which runs the full pipeline:
      1. Product discovery (Claude/GPT from markdown)
      2. Product creation with text embeddings (Voyage AI, inline)
      3. Image download → classification (Qwen/Claude) → 5 SLIG embeddings
      4. Phase 2: Qwen3-VL analysis → understanding embeddings (1024D)
    """
    try:
        logger.info(f"🚀 Starting scraping session processing: {request.session_id}")

        supabase = get_supabase_client()

        # Validate session exists
        session_response = supabase.client.table("scraping_sessions").select("*").eq("id", request.session_id).single().execute()
        if not session_response.data:
            raise HTTPException(status_code=404, detail=f"Scraping session not found: {request.session_id}")

        session = session_response.data

        # Initialize service
        scraping_service = WebScrapingService(model=request.model)

        # Create background job for tracking
        job_id = await scraping_service.create_background_job(
            session_id=request.session_id,
            workspace_id=request.workspace_id,
            categories=request.categories or ["products"]
        )

        # Link background job to session so edge function can track progress
        supabase.client.table("scraping_sessions").update({
            "status": "processing",
            "background_job_id": job_id,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", request.session_id).execute()

        logger.info(f"✅ Created background job {job_id} for session {request.session_id}")

        # Process in background
        async def process_session_background():
            try:
                result = await scraping_service.process_scraping_session(
                    session_id=request.session_id,
                    workspace_id=request.workspace_id,
                    categories=request.categories or ["products"],
                    job_id=job_id
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
            job_id=job_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start session processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start session processing: {str(e)}")


@router.get("/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(
    session_id: str = Path(..., description="Scraping session ID")
) -> SessionStatusResponse:
    """
    Get scraping session processing status.
    """
    try:
        supabase = get_supabase_client()

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
            products_created=session.get("materials_processed"),
            error_message=session.get("scraping_config", {}).get("error") if isinstance(session.get("scraping_config"), dict) else None,
            created_at=session["created_at"],
            updated_at=session["updated_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session status: {str(e)}")


@router.post("/session/{session_id}/retry", response_model=ProcessSessionResponse)
async def retry_session_processing(
    session_id: str = Path(..., description="Scraping session ID"),
    background_tasks: BackgroundTasks = None
) -> ProcessSessionResponse:
    """
    Retry processing a failed scraping session.
    """
    try:
        logger.info(f"🔄 Retrying scraping session: {session_id}")

        supabase = get_supabase_client()

        response = supabase.client.table("scraping_sessions").select("*").eq("id", session_id).single().execute()
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Scraping session not found: {session_id}")

        session = response.data

        if session["status"] not in ["failed", "completed"]:
            raise HTTPException(
                status_code=400,
                detail=f"Session status is '{session['status']}', can only retry 'failed' or 'completed' sessions"
            )

        workspace_id = session.get("workspace_id") or "default"
        scraping_service = WebScrapingService(model="claude")

        # Create a fresh background job for the retry
        job_id = await scraping_service.create_background_job(
            session_id=session_id,
            workspace_id=workspace_id,
            categories=["products"]
        )

        supabase.client.table("scraping_sessions").update({
            "status": "processing",
            "materials_processed": 0,
            "background_job_id": job_id,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", session_id).execute()

        async def retry_session_background():
            try:
                result = await scraping_service.process_scraping_session(
                    session_id=session_id,
                    workspace_id=workspace_id,
                    categories=["products"],
                    job_id=job_id
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
            message="Session retry started in background",
            job_id=job_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retry session: {str(e)}")
