"""
Data Import API Routes

Endpoints for processing XML/web scraping import jobs:
- POST /api/import/process - Start processing an import job
- GET /api/import/jobs/{job_id} - Get import job status
- GET /api/import/history - Get import history
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from datetime import datetime

from app.services.data_import_service import DataImportService
from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/import", tags=["data-import"])


# ============================================================================
# Request/Response Models
# ============================================================================

class ProcessImportRequest(BaseModel):
    """Request model for processing an import job"""
    job_id: str = Field(..., description="Import job ID")
    workspace_id: str = Field(..., description="Workspace ID")


class ImportJobStatus(BaseModel):
    """Import job status response"""
    job_id: str
    status: str
    import_type: str
    source_name: Optional[str]
    total_products: int
    processed_products: int
    failed_products: int
    progress_percentage: int
    current_stage: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    estimated_time_remaining: Optional[int]  # seconds


class ImportHistoryItem(BaseModel):
    """Import history item"""
    job_id: str
    import_type: str
    source_name: Optional[str]
    status: str
    total_products: int
    processed_products: int
    failed_products: int
    created_at: str
    completed_at: Optional[str]


class ImportHistoryResponse(BaseModel):
    """Import history response"""
    imports: list[ImportHistoryItem]
    total_count: int
    page: int
    page_size: int


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/process")
async def process_import_job(
    request: ProcessImportRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Start processing an import job.
    
    This endpoint is called by the Edge Function after creating the import job.
    It processes the job in the background using batch processing, image downloads,
    and product normalization.
    
    Args:
        request: Process import request
        background_tasks: FastAPI background tasks
        
    Returns:
        Processing status
    """
    try:
        logger.info(f"🚀 Starting import job processing: {request.job_id}")
        
        # Initialize service
        import_service = DataImportService()
        
        # Add processing task to background
        background_tasks.add_task(
            import_service.process_import_job,
            job_id=request.job_id,
            workspace_id=request.workspace_id
        )
        
        return {
            "success": True,
            "message": "Import job processing started",
            "job_id": request.job_id
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to start import job processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_import_job_status(job_id: str) -> ImportJobStatus:
    """
    Get import job status and progress.
    
    Args:
        job_id: Import job ID
        
    Returns:
        Job status with progress information
    """
    try:
        supabase_wrapper = get_supabase_client()
        supabase = supabase_wrapper.client
        
        # Get job from database
        response = supabase.table('data_import_jobs').select('*').eq('id', job_id).single().execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job = response.data
        
        # Calculate progress percentage
        total = job.get('total_products', 0)
        processed = job.get('processed_products', 0)
        progress = int((processed / total) * 100) if total > 0 else 0
        
        # Estimate time remaining (rough estimate based on average processing time)
        estimated_time = None
        if job.get('status') == 'processing' and total > 0:
            # Assume 2 seconds per product on average
            remaining_products = total - processed
            estimated_time = remaining_products * 2
        
        # Get current stage from metadata
        metadata = job.get('metadata', {})
        current_stage = metadata.get('current_stage')
        
        return ImportJobStatus(
            job_id=job['id'],
            status=job['status'],
            import_type=job['import_type'],
            source_name=job.get('source_name'),
            total_products=total,
            processed_products=processed,
            failed_products=job.get('failed_products', 0),
            progress_percentage=progress,
            current_stage=current_stage,
            started_at=job.get('started_at'),
            completed_at=job.get('completed_at'),
            error_message=job.get('error_message'),
            estimated_time_remaining=estimated_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get job status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_import_history(
    workspace_id: str = Query(..., description="Workspace ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    import_type: Optional[str] = Query(None, description="Filter by import type")
) -> ImportHistoryResponse:
    """
    Get import history for a workspace.
    
    Args:
        workspace_id: Workspace ID
        page: Page number (1-indexed)
        page_size: Items per page
        status: Optional status filter
        import_type: Optional import type filter
        
    Returns:
        Paginated import history
    """
    try:
        supabase_wrapper = get_supabase_client()
        supabase = supabase_wrapper.client
        
        # Build query
        query = supabase.table('data_import_jobs').select('*', count='exact').eq('workspace_id', workspace_id)
        
        # Apply filters
        if status:
            query = query.eq('status', status)
        if import_type:
            query = query.eq('import_type', import_type)
        
        # Apply pagination
        offset = (page - 1) * page_size
        query = query.order('created_at', desc=True).range(offset, offset + page_size - 1)
        
        # Execute query
        response = query.execute()
        
        # Build response
        imports = []
        for job in response.data:
            imports.append(ImportHistoryItem(
                job_id=job['id'],
                import_type=job['import_type'],
                source_name=job.get('source_name'),
                status=job['status'],
                total_products=job.get('total_products', 0),
                processed_products=job.get('processed_products', 0),
                failed_products=job.get('failed_products', 0),
                created_at=job['created_at'],
                completed_at=job.get('completed_at')
            ))
        
        return ImportHistoryResponse(
            imports=imports,
            total_count=response.count or 0,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get import history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def import_health_check() -> Dict[str, Any]:
    """
    Health check endpoint for the data import API.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "service": "data-import-api",
        "version": "1.0.0",
        "features": {
            "xml_import": True,
            "web_scraping": False,  # Phase 4
            "batch_processing": True,
            "concurrent_image_downloads": True,
            "checkpoint_recovery": True,
            "real_time_progress": True
        },
        "endpoints": {
            "process": "/api/import/process",
            "job_status": "/api/import/jobs/{job_id}",
            "history": "/api/import/history",
            "health": "/api/import/health"
        }
    }

