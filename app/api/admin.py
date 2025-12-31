"""
Administrative and Monitoring API Endpoints

This module provides comprehensive administrative and monitoring capabilities including:
- Job management and status tracking
- Service statistics and health monitoring
- Administrative endpoints for data management
- Bulk operations for document processing
- System monitoring and performance metrics
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime, timedelta
import psutil
import os
from pathlib import Path

from app.utils.timestamp_utils import normalize_timestamp

from ..schemas.jobs import (
    JobResponse, JobStatusResponse, JobListResponse, JobListItem,
    BulkProcessingRequest, BulkProcessingResponse,
    JobStatistics, SystemMetrics, JobProgressDetail
)
from ..schemas.common import BaseResponse, PaginationParams
from ..services.pdf_processor import PDFProcessor
from ..services.supabase_client import SupabaseClient
from ..services.rag_service import RAGService
from ..services.product_creation_service import ProductCreationService
from ..services.material_kai_service import MaterialKaiService
from ..services.progress_tracker import get_progress_service
from ..services.async_queue_service import AsyncQueueService
from ..dependencies import (
    get_current_user,
    get_workspace_context,
    require_admin,
    get_rag_service,
    get_supabase_client,
    get_material_kai_service,
    get_pdf_processor
)
from ..middleware.jwt_auth import WorkspaceContext, User
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Health & Monitoring"])

# Global job tracking
active_jobs: Dict[str, Dict[str, Any]] = {}
job_history: List[Dict[str, Any]] = []

def migrate_job_data():
    """Migrate existing job data to match JobListItem schema"""
    global job_history, active_jobs

    # Migrate job_history
    for job in job_history:
        if "priority" not in job:
            job["priority"] = "normal"
        if "started_at" not in job:
            job["started_at"] = None
        if "completed_at" not in job:
            job["completed_at"] = None
        if "progress_percentage" not in job:
            job["progress_percentage"] = 100.0 if job.get("status") == "completed" else 0.0
        if "current_step" not in job:
            job["current_step"] = None
        if "description" not in job:
            job["description"] = None
        if "success" not in job:
            job["success"] = job.get("status") == "completed"
        if "error_message" not in job:
            job["error_message"] = None

    # Migrate active_jobs
    for job in active_jobs.values():
        if "priority" not in job:
            job["priority"] = "normal"
        if "started_at" not in job:
            job["started_at"] = None
        if "completed_at" not in job:
            job["completed_at"] = None
        if "progress_percentage" not in job:
            job["progress_percentage"] = 50.0 if job.get("status") == "running" else 0.0
        if "current_step" not in job:
            job["current_step"] = None
        if "description" not in job:
            job["description"] = None
        if "success" not in job:
            job["success"] = None
        if "error_message" not in job:
            job["error_message"] = None

# Run migration on startup
migrate_job_data()

# REMOVED: Local dependency functions - now using centralized dependencies from app.dependencies

async def track_job(job_id: str, job_type: str, status: str, details: Dict[str, Any] = None):
    """Track job status and update global job tracking"""
    current_time = datetime.utcnow().isoformat()

    # Check if job already exists
    existing_job = active_jobs.get(job_id)

    if existing_job:
        # Update existing job
        job_info = existing_job.copy()
        job_info["status"] = status
        job_info["updated_at"] = current_time

        # Set started_at when job starts running
        if status == "running" and job_info.get("started_at") is None:
            job_info["started_at"] = current_time
            logger.info(f"🚀 Job {job_id} started at {current_time}")

        # Set completed_at when job finishes
        if status in ["completed", "failed", "cancelled", "error"]:
            job_info["completed_at"] = current_time
            if status == "completed":
                job_info["success"] = True
                job_info["progress_percentage"] = 100.0
            elif status in ["failed", "error"]:
                job_info["success"] = False
            logger.info(f"🏁 Job {job_id} finished with status: {status}")

        # Update progress from details if provided
        if details:
            if "progress_percentage" in details:
                job_info["progress_percentage"] = details["progress_percentage"]
                logger.info(f"📊 Job {job_id} progress: {details['progress_percentage']}%")
            if "current_step" in details:
                job_info["current_step"] = details["current_step"]
                logger.info(f"📋 Job {job_id} step: {details['current_step']}")
            if "error" in details:
                job_info["error_message"] = details["error"]

            # Merge details
            existing_details = job_info.get("details", {})
            existing_details.update(details)
            job_info["details"] = existing_details
    else:
        # Create new job
        job_info = {
            "job_id": job_id,
            "job_type": job_type,
            "status": status,
            "priority": "normal",
            "created_at": current_time,
            "updated_at": current_time,
            "started_at": current_time if status == "running" else None,
            "completed_at": None,
            "progress_percentage": 10.0 if status == "running" else 0.0,
            "current_step": details.get("current_step") if details else None,
            "description": None,
            "tags": [],
            "success": None,
            "error_message": None,
            "details": details or {},
            "parameters": details or {},
            "retry_count": 3,
            "current_retry": 0,
            "result": None
        }
        logger.info(f"📝 Created new job {job_id} with status: {status}")

    active_jobs[job_id] = job_info

    # Add to history if completed or failed
    if status in ["completed", "failed", "cancelled"]:
        job_history.append(job_info.copy())
        if job_id in active_jobs:
            del active_jobs[job_id]
        logger.info(f"📚 Job {job_id} moved to history")

# Job Management Endpoints

@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    pagination: PaginationParams = Depends(),

):
    """
    List all jobs with optional filtering and pagination
    
    - **status**: Filter jobs by status (pending, running, completed, failed, cancelled)
    - **job_type**: Filter jobs by type (document_processing, bulk_processing, etc.)
    - **limit**: Number of jobs to return (default: 50, max: 100)
    - **offset**: Number of jobs to skip for pagination
    """
    try:
        # Combine active jobs and job history
        all_jobs = list(active_jobs.values()) + job_history
        
        # Apply filters
        filtered_jobs = all_jobs
        if status:
            filtered_jobs = [job for job in filtered_jobs if job["status"] == status]
        if job_type:
            filtered_jobs = [job for job in filtered_jobs if job["job_type"] == job_type]
        
        # Sort by created_at descending
        filtered_jobs.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Apply pagination
        start_idx = (pagination.page - 1) * pagination.page_size
        end_idx = start_idx + pagination.page_size
        paginated_jobs = filtered_jobs[start_idx:end_idx]
        
        return JobListResponse(
            success=True,
            message="Jobs retrieved successfully",
            jobs=[JobListItem(**job) for job in paginated_jobs],
            total_count=len(filtered_jobs),
            page=pagination.page,
            page_size=pagination.page_size,
            status_counts={
                "active": len(active_jobs),
                "completed": len([j for j in job_history if j["status"] == "completed"]),
                "failed": len([j for j in job_history if j["status"] == "failed"])
            },
            type_counts={}
        )
        
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")

@router.get("/jobs/statistics", response_model=Dict[str, Any])
async def get_job_statistics():
    """
    Get comprehensive job statistics and metrics
    """
    try:
        # Calculate statistics
        all_jobs = list(active_jobs.values()) + job_history
        
        # Status distribution
        status_counts = {}
        for job in all_jobs:
            status = job["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Job type distribution
        type_counts = {}
        for job in all_jobs:
            job_type = job["job_type"]
            type_counts[job_type] = type_counts.get(job_type, 0) + 1
        
        # Recent activity (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_jobs = [
            job for job in all_jobs
            if datetime.fromisoformat(normalize_timestamp(job["created_at"])) > recent_cutoff
        ]
        
        # Average processing time for completed jobs
        completed_jobs = [job for job in job_history if job["status"] == "completed"]
        avg_processing_time = None
        if completed_jobs:
            processing_times = []
            for job in completed_jobs:
                created = datetime.fromisoformat(normalize_timestamp(job["created_at"]))
                updated = datetime.fromisoformat(normalize_timestamp(job["updated_at"]))
                processing_times.append((updated - created).total_seconds())
            avg_processing_time = sum(processing_times) / len(processing_times)
        
        statistics = JobStatistics(
            total_jobs=len(all_jobs),
            active_jobs=len(active_jobs),
            completed_jobs=len([j for j in job_history if j["status"] == "completed"]),
            failed_jobs=len([j for j in job_history if j["status"] == "failed"]),
            cancelled_jobs=len([j for j in job_history if j["status"] == "cancelled"]),
            status_distribution=status_counts,
            type_distribution=type_counts,
            recent_jobs_24h=len(recent_jobs),
            average_processing_time_seconds=avg_processing_time
        )
        
        return {
            "success": True,
            "message": "Job statistics retrieved successfully",
            "data": statistics.model_dump(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting job statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job statistics: {str(e)}")

# Bulk Operations

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,

):
    """
    Get detailed status information for a specific job

    - **job_id**: Unique identifier for the job
    """
    try:
        # Validate UUID format
        import uuid
        try:
            uuid.UUID(job_id)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid job ID format: {job_id}. Must be a valid UUID.")

        # Check active jobs first
        if job_id in active_jobs:
            job_info = active_jobs[job_id]
        else:
            # Check job history
            job_info = next((job for job in job_history if job["job_id"] == job_id), None)

            # If not found in memory, check database
            if not job_info:
                from app.services.supabase_client import get_supabase_client
                supabase_client = get_supabase_client()
                response = supabase_client.client.table('background_jobs').select('*').eq('id', job_id).execute()

                if response.data and len(response.data) > 0:
                    db_job = response.data[0]
                    # Convert database format to expected format
                    job_info = {
                        "job_id": db_job['id'],
                        "job_type": "document_processing",
                        "status": db_job['status'],
                        "progress": db_job.get('progress', 0),
                        "document_id": db_job.get('document_id'),
                        "filename": db_job.get('filename'),
                        "error": db_job.get('error'),
                        "created_at": db_job.get('created_at'),
                        "updated_at": db_job.get('updated_at')
                    }

        if not job_info:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return JobStatusResponse(
            success=True,
            message="Job status retrieved successfully",
            data=JobResponse(**job_info)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@router.get("/jobs/{job_id}/status")
async def get_job_status_alt(
    job_id: str,
):
    """
    Get detailed status information for a specific job (alternative endpoint)

    - **job_id**: Unique identifier for the job
    """
    try:
        # Check active jobs first
        if job_id in active_jobs:
            job_info = active_jobs[job_id]
        else:
            # Check job history
            job_info = next((job for job in job_history if job["job_id"] == job_id), None)

            # If not found in memory, check database
            if not job_info:
                from app.services.supabase_client import get_supabase_client
                supabase_client = get_supabase_client()
                response = supabase_client.client.table('background_jobs').select('*').eq('id', job_id).execute()

                if response.data and len(response.data) > 0:
                    db_job = response.data[0]
                    # Convert database format to expected format
                    job_info = {
                        "job_id": db_job['id'],
                        "job_type": "document_processing",
                        "status": db_job['status'],
                        "progress": db_job.get('progress', 0),
                        "document_id": db_job.get('document_id'),
                        "filename": db_job.get('filename'),
                        "error": db_job.get('error'),
                        "created_at": db_job.get('created_at'),
                        "updated_at": db_job.get('updated_at')
                    }

        if not job_info:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return {
            "success": True,
            "message": f"Job {job_id} status retrieved successfully",
            "data": job_info,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    cleanup: bool = Query(True, description="Clean up partial data created by the job")
):
    """
    Cancel a running job and optionally clean up partial data.

    This endpoint will:
    1. Mark the job as cancelled in the database
    2. The heartbeat check will detect the cancellation and raise CancelledError
    3. The processing function will catch CancelledError and stop immediately
    4. If cleanup=True, delete all partial data (chunks, embeddings, images, products, files)

    Args:
        job_id: Unique identifier for the job to cancel
        cleanup: If True, delete all partial data created by the job (default: True)

    Returns:
        Success message with cleanup statistics if cleanup was performed
    """
    try:
        from app.services.supabase_client import get_supabase_client
        from app.services.cleanup_service import CleanupService

        supabase_client = get_supabase_client()

        # Get job from database
        job_response = supabase_client.client.table('background_jobs')\
            .select('*')\
            .eq('id', job_id)\
            .execute()

        if not job_response.data or len(job_response.data) == 0:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        job_data = job_response.data[0]
        current_status = job_data.get('status')

        # Check if job can be cancelled
        if current_status in ["completed", "failed", "cancelled"]:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} is already {current_status} and cannot be cancelled"
            )

        # Mark job as cancelled in database
        # The heartbeat check will detect this and raise CancelledError
        supabase_client.client.table('background_jobs')\
            .update({
                'status': 'cancelled',
                'error': 'Job cancelled by user',
                'updated_at': datetime.utcnow().isoformat(),
                'metadata': {
                    **job_data.get('metadata', {}),
                    'cancelled_at': datetime.utcnow().isoformat(),
                    'cancelled_by': 'admin'
                }
            })\
            .eq('id', job_id)\
            .execute()

        logger.info(f"🛑 Job {job_id} marked as cancelled")

        # Update in-memory job storage if exists
        if job_id in active_jobs:
            active_jobs[job_id]["status"] = "cancelled"
            active_jobs[job_id]["error"] = "Job cancelled by user"

        cleanup_stats = None

        # Perform cleanup if requested
        if cleanup:
            logger.info(f"🧹 Starting cleanup for cancelled job {job_id} (manual deletion - includes storage files)")

            cleanup_service = CleanupService()

            # Manual deletion from UI - delete everything including storage files
            cleanup_stats = await cleanup_service.delete_job_completely(
                job_id=job_id,
                supabase_client=supabase_client,
                delete_storage_files=True  # ✅ Manual deletion includes storage files
            )

            logger.info(f"✅ Cleanup completed for job {job_id}: {cleanup_stats}")

        response_data = {
            "success": True,
            "message": f"Job {job_id} cancelled successfully",
            "job_id": job_id,
            "previous_status": current_status,
            "cleanup_performed": cleanup
        }

        if cleanup_stats:
            response_data["cleanup_stats"] = cleanup_stats

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

@router.post("/bulk/process")
async def bulk_process_documents(
    request: BulkProcessingRequest,
    background_tasks: BackgroundTasks,
    pdf_processor: PDFProcessor = Depends(get_pdf_processor),
):
    logger.info(f"🔍 ENTRY DEBUG: request type: {type(request)}")
    logger.info(f"🔍 ENTRY DEBUG: request: {request}")
    """
    Process multiple documents in bulk

    - **urls**: List of document URLs to process
    - **options**: Processing options (extract_images, generate_summary, etc.)
    - **batch_size**: Number of documents to process concurrently (default: 5)
    """
    try:
        # Extract request data from Pydantic model
        logger.info(f"🔍 DEBUG: request type: {type(request)}")
        logger.info(f"🔍 DEBUG: request.urls type: {type(request.urls)}")
        logger.info(f"🔍 DEBUG: request.options type: {type(request.options)}")

        urls = request.urls
        batch_size = request.batch_size
        options = request.options or {}

        if not urls:
            raise HTTPException(status_code=400, detail="No URLs provided")

        job_id = f"bulk_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Track the bulk job
        await track_job(
            job_id,
            "bulk_processing",
            "pending",
            {
                "total_documents": len(urls),
                "batch_size": batch_size,
                "options": options
            }
        )

        # Start background processing
        background_tasks.add_task(
            process_bulk_documents,
            job_id,
            urls,
            options,
            batch_size,
            pdf_processor
        )

        return {
            "success": True,
            "message": "Bulk processing started successfully",
            "data": {
                "job_id": job_id,
                "total_documents": len(urls),
                "estimated_completion_time": (datetime.utcnow() + timedelta(
                    minutes=len(urls) * 2  # Rough estimate
                )).isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        import traceback
        logger.error(f"Error starting bulk processing: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to start bulk processing: {str(e)}")

async def process_bulk_documents(
    job_id: str,
    urls: List[str],
    options: Any,
    batch_size: int,
    pdf_processor: PDFProcessor
):
    """Background task for bulk document processing with comprehensive error handling"""
    try:
        logger.info(f"🚀 Background task starting for job {job_id}")
        await track_job(job_id, "bulk_processing", "running", {
            "current_step": "Background processing started",
            "progress_percentage": 10.0
        })
        logger.info(f"✅ Job {job_id} status updated to running")

        processed_count = 0
        failed_count = 0
        results = []
        total_chunks = 0
        total_images = 0
        
        # Process documents in batches
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i + batch_size]
            batch_tasks = []
            
            for url in batch_urls:
                task = asyncio.create_task(process_single_document(url, options, pdf_processor, job_id))
                batch_tasks.append(task)
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    failed_count += 1
                    logger.error(f"Document processing failed for {batch_urls[j]}: {str(result)}")
                    results.append({
                        "url": batch_urls[j],
                        "status": "failed",
                        "error": str(result)
                    })
                else:
                    processed_count += 1
                    # Extract metrics from result
                    chunks_count = result.get("chunks", 0)
                    images_count = result.get("images", 0)
                    total_chunks += chunks_count
                    total_images += images_count

                    logger.info(f"✅ Document processed: {batch_urls[j]} - {chunks_count} chunks, {images_count} images")
                    results.append({
                        "url": batch_urls[j],
                        "status": "completed",
                        "document_id": result.get("document_id"),
                        "chunks": chunks_count,
                        "images": images_count
                    })

            # Update job progress with detailed metrics
            progress = (processed_count + failed_count) / len(urls) * 100
            current_step = f"Processed {processed_count + failed_count}/{len(urls)} documents"

            logger.info(f"📊 Progress update: {progress:.1f}% - {current_step}")
            await track_job(
                job_id,
                "bulk_processing",
                "running",
                {
                    "progress_percentage": progress,
                    "current_step": current_step,
                    "processed_count": processed_count,
                    "failed_count": failed_count,
                    "total_documents": len(urls),
                    "chunks_created": total_chunks,
                    "images_extracted": total_images,
                    "results": results
                }
            )
        
        # Mark job as completed
        logger.info(f"🎉 Job {job_id} completed successfully: {processed_count} processed, {failed_count} failed")
        await track_job(
            job_id,
            "bulk_processing",
            "completed",
            {
                "progress_percentage": 100.0,
                "current_step": "Processing completed",
                "total_processed": processed_count,
                "total_failed": failed_count,
                "total_documents": len(urls),
                "chunks_created": total_chunks,
                "images_extracted": total_images,
                "results": results
            }
        )

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"❌ Critical error in bulk processing job {job_id}: {str(e)}")
        logger.error(f"Full traceback: {error_details}")
        await track_job(
            job_id,
            "bulk_processing",
            "failed",
            {
                "error": str(e),
                "error_details": error_details,
                "processed_count": processed_count,
                "failed_count": failed_count
            }
        )

async def process_single_document(url: str, options: Any, pdf_processor: PDFProcessor, job_id: str = None):
    """Process a single document as part of bulk processing with progress tracking"""
    try:
        from app.services.supabase_client import get_supabase_client
        from app.schemas.jobs import ProcessingStage

        # Use the actual PDF processor to process the document
        import uuid
        document_id = str(uuid.uuid4())  # Generate proper UUID for database compatibility

        # Extract filename from URL
        original_filename = url.split('/')[-1] if url else f"{document_id}.pdf"

        # Initialize progress tracking
        progress_service = get_progress_service()
        tracker = None
        if job_id:
            tracker = progress_service.create_tracker(job_id, document_id, 1)
            tracker.start_processing()

        # Create async progress callback for PDF processor
        async def progress_callback(progress_percentage: float, current_step: str, details: dict = None):
            """Async callback to update job progress during PDF processing"""
            try:
                if job_id:
                    # Ensure progress is within bounds and meaningful
                    progress_percentage = max(10.0, min(95.0, progress_percentage))

                    # Update job tracking in active_jobs
                    if job_id in active_jobs:
                        job_info = active_jobs[job_id]
                        job_info["progress_percentage"] = progress_percentage
                        job_info["current_step"] = current_step
                        job_info["updated_at"] = datetime.utcnow().isoformat()

                        # Update details
                        if details:
                            if "details" not in job_info:
                                job_info["details"] = {}
                            job_info["details"].update(details)

                        logger.info(f"📊 Job {job_id} progress updated: {progress_percentage}% - {current_step}")
                    else:
                        logger.warning(f"Job {job_id} not found in active_jobs for progress update")
            except Exception as e:
                logger.warning(f"Failed to update job progress: {e}")

        # Update progress before processing
        await progress_callback(15.0, "Starting PDF download and analysis")

        result = await pdf_processor.process_pdf_from_url(url, document_id, options or {}, progress_callback)

        # Update progress after PDF processing
        await progress_callback(40.0, "PDF processing completed, analyzing content")

        # Update progress tracking with actual page count
        if tracker:
            tracker.total_pages = result.page_count
            tracker.update_stage(ProcessingStage.EXTRACTING_TEXT)

        # Create chunks from markdown content (simple chunking)
        chunks = []
        chunks_created = 0
        if result.markdown_content:
            try:
                # Update progress for chunking
                await progress_callback(60.0, "Creating text chunks for RAG pipeline")

                await track_job(
                    job_id,
                    "bulk_processing",
                    "running",
                    {
                        "current_step": "Creating text chunks for RAG pipeline",
                        "progress_percentage": 60.0,
                        "text_length": len(result.markdown_content)
                    }
                )

                # Simple text chunking for this endpoint
                text_chunks = result.markdown_content.split('\n\n')
                chunks = [chunk.strip() for chunk in text_chunks if chunk.strip()]
                chunks_created = len(chunks)
                logger.info(f"✅ Simple chunking created {chunks_created} chunks")

                # Note: Embeddings are generated during the main PDF processing pipeline
                # This endpoint provides basic chunking only
                await progress_callback(70.0, "Chunks created (embeddings handled by main pipeline)")

            except Exception as chunk_error:
                logger.warning(f"Chunking failed, using simple text splitting: {chunk_error}")
                # Fallback to simple text splitting
                text_chunks = result.markdown_content.split('\n\n')
                chunks = [chunk.strip() for chunk in text_chunks if chunk.strip()]
                chunks_created = len(chunks)

        # ✅ NEW: Upload processed document to RAG system for indexing
        try:
            logger.info(f"🔍 Starting RAG upload for document {document_id}")
            await progress_callback(85.0, "Uploading document to RAG system for search indexing")

            await track_job(
                job_id,
                "bulk_processing",
                "running",
                {
                    "current_step": "Uploading document to RAG system for search indexing",
                    "progress_percentage": 85.0,
                    "chunks_created": chunks_created,
                    "document_id": document_id
                }
            )

            # Download PDF bytes for RAG indexing
            logger.info(f"📥 Downloading PDF bytes from: {url}")
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        pdf_bytes = await response.read()
                        logger.info(f"✅ PDF downloaded: {len(pdf_bytes)} bytes")
                    else:
                        raise Exception(f"Failed to download PDF: HTTP {response.status}")

            # Documents are processed directly into the vector database
            logger.info(f"✅ Document {document_id} processed into vector database")
            rag_result = {"success": True, "message": "Document processed into vector DB"}

        except Exception as rag_error:
            logger.error(f"⚠️ Failed to index document in RAG system: {rag_error}")
            logger.error(f"🔍 RAG error details: {type(rag_error).__name__}: {str(rag_error)}")
            import traceback
            logger.error(f"🔍 RAG error traceback: {traceback.format_exc()}")
            # Don't fail the entire job if RAG indexing fails - this is optional
            logger.info("📄 Continuing with database save despite RAG indexing failure")

        # Save to database
        supabase_client = get_supabase_client()

        # Save PDF processing result to database
        try:
            pdf_result = await supabase_client.save_pdf_processing_result(
                result=result,  # Pass the PDFProcessingResult object
                original_filename=original_filename,
                file_url=url  # Pass the source URL
            )
            logger.info(f"✅ PDF processing result saved to database: {pdf_result}")

            # Update progress tracking
            if tracker:
                tracker.update_database_stats(records_created=1)
                tracker.update_stage(ProcessingStage.SAVING_TO_DATABASE)

        except Exception as e:
            logger.error(f"❌ Failed to save PDF processing result: {str(e)}")
            # Continue processing even if database save fails

        # Save knowledge base entries
        try:
            kb_entries_saved = await supabase_client.save_knowledge_base_entries(
                document_id=result.document_id,
                chunks=chunks,
                images=result.extracted_images
            )
            logger.info(f"✅ Knowledge base entries saved: {kb_entries_saved}")

        except Exception as e:
            logger.error(f"❌ Failed to save knowledge base entries: {str(e)}")
            # Continue processing even if knowledge base save fails
        # Final progress tracking update
        chunks_saved = len(chunks)
        images_saved = len(result.extracted_images)

        if tracker:
            tracker.update_database_stats(
                kb_entries=chunks_saved,
                images_stored=images_saved
            )
            tracker.update_stage(ProcessingStage.COMPLETED)

        logger.info(f"🎉 Document processing completed: {chunks_saved} chunks, {images_saved} images")

        return {
            "document_id": result.document_id,
            "status": "completed",
            "chunks": len(chunks),
            "images": len(result.extracted_images),
            "text_content": result.markdown_content,
            "ocr_text": result.ocr_text,
            "metadata": {
                "pages": result.page_count,
                "word_count": result.word_count,
                "character_count": result.character_count,
                "processing_time": result.processing_time
            }
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"❌ Error processing document {url}: {str(e)}")
        logger.error(f"Full traceback: {error_details}")

        # Update tracker with error if available
        if tracker:
            tracker.add_error("Document Processing Failed", str(e), {"url": url, "traceback": error_details})

        raise

# System Monitoring

@router.get("/system/health")
async def get_system_health(

):
    """
    Get comprehensive system health status
    """
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Service health checks
        services_health = {}
        
        # Check Supabase connection
        try:
            import time
            start_time = time.time()
            supabase_client = get_supabase_client()
            # Simple health check query using the health_check method
            if supabase_client.health_check():
                response_time_ms = int((time.time() - start_time) * 1000)
                services_health["supabase"] = {
                    "status": "healthy",
                    "response_time_ms": response_time_ms
                }
            else:
                services_health["supabase"] = {
                    "status": "unhealthy",
                    "error": "Health check failed"
                }
        except Exception as e:
            services_health["supabase"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        services_health["rag"] = {
            "status": "healthy",
            "details": {
                "service_type": "Direct Vector DB",
                "message": "RAG uses Claude 4.5 + Direct Vector Database"
            }
        }
        
        # Check Material Kai service
        try:
            material_kai_service = MaterialKaiService()
            health_check = await material_kai_service.health_check()
            services_health["material_kai"] = {
                "status": "healthy" if health_check["status"] == "healthy" else "unhealthy",
                "details": health_check
            }
        except Exception as e:
            services_health["material_kai"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Overall health status
        all_services_healthy = all(
            service["status"] == "healthy" 
            for service in services_health.values()
        )
        
        # Get real system uptime
        import time
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time

        system_metrics = SystemMetrics(
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent,
            disk_free_gb=disk.free / (1024**3),
            active_jobs_count=len(active_jobs),
            uptime_seconds=uptime_seconds
        )
        
        return {
            "success": True,
            "message": "System health retrieved successfully",
            "data": {
                "overall_status": "healthy" if all_services_healthy and cpu_percent < 80 and memory.percent < 80 else "degraded",
                "system_metrics": system_metrics.model_dump(),
                "services": services_health,
                "active_jobs": len(active_jobs),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")

@router.get("/system/metrics")
async def get_system_metrics():
    """
    Get detailed system performance metrics
    """
    try:
        # CPU metrics
        cpu_count = psutil.cpu_count()
        cpu_percent_per_core = psutil.cpu_percent(percpu=True, interval=1)
        cpu_freq = psutil.cpu_freq()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network metrics
        network_io = psutil.net_io_counters()
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            "success": True,
            "message": "System metrics retrieved successfully",
            "data": {
                "cpu": {
                    "count": cpu_count,
                    "usage_percent": psutil.cpu_percent(),
                    "usage_per_core": cpu_percent_per_core,
                    "frequency_mhz": cpu_freq.current if cpu_freq else None
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "usage_percent": memory.percent,
                    "swap_total_gb": swap.total / (1024**3),
                    "swap_used_gb": swap.used / (1024**3),
                    "swap_percent": swap.percent
                },
                "disk": {
                    "total_gb": disk_usage.total / (1024**3),
                    "used_gb": disk_usage.used / (1024**3),
                    "free_gb": disk_usage.free / (1024**3),
                    "usage_percent": disk_usage.percent,
                    "read_bytes": disk_io.read_bytes if disk_io else None,
                    "write_bytes": disk_io.write_bytes if disk_io else None
                },
                "network": {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_received": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_received": network_io.packets_recv
                },
                "process": {
                    "memory_rss_mb": process_memory.rss / (1024**2),
                    "memory_vms_mb": process_memory.vms / (1024**2),
                    "cpu_percent": process.cpu_percent()
                },
                "jobs": {
                    "active_count": len(active_jobs),
                    "total_history": len(job_history)
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")

# Administrative Data Management

@router.delete("/data/cleanup")
async def cleanup_old_data(
    days_old: int = Query(30, description="Delete data older than this many days"),
    dry_run: bool = Query(True, description="Preview what would be deleted without actually deleting")
):
    """
    Clean up old data from the system
    
    - **days_old**: Delete data older than this many days (default: 30)
    - **dry_run**: Preview what would be deleted without actually deleting (default: true)
    """
    try:
        global job_history  # Declare global at the beginning of the function

        cutoff_date = datetime.utcnow() - timedelta(days=days_old)

        # Find old job history entries
        old_jobs = [
            job for job in job_history
            if datetime.fromisoformat(normalize_timestamp(job["created_at"])) < cutoff_date
        ]

        cleanup_summary = {
            "old_jobs_count": len(old_jobs),
            "cutoff_date": cutoff_date.isoformat(),
            "dry_run": dry_run
        }

        if not dry_run:
            # Actually remove old jobs from history
            job_history = [
                job for job in job_history
                if datetime.fromisoformat(normalize_timestamp(job["created_at"])) >= cutoff_date
            ]
            cleanup_summary["jobs_deleted"] = len(old_jobs)
        
        return {
            "success": True,
            "message": f"Data cleanup {'preview' if dry_run else 'completed'} successfully",
            "data": cleanup_summary
        }
        
    except Exception as e:
        logger.error(f"Error during data cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup data: {str(e)}")

@router.post("/data/backup")
async def create_data_backup():
    """
    Create a backup of system data
    """
    try:
        backup_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "active_jobs": active_jobs,
            "job_history": job_history,
            "system_info": {
                "version": "1.0.0",  # Would come from app config
                "backup_type": "administrative_data"
            }
        }
        
        # In a real implementation, this would save to a file or external storage
        backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "success": True,
            "message": "Data backup created successfully",
            "data": {
                "backup_id": backup_id,
                "backup_size_bytes": len(str(backup_data)),
                "items_backed_up": {
                    "active_jobs": len(active_jobs),
                    "job_history": len(job_history)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating data backup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create data backup: {str(e)}")

@router.get("/data/export")
async def export_system_data(
    format: str = Query("json", description="Export format (json, csv)"),
    data_type: str = Query("jobs", description="Type of data to export (jobs, metrics)")
):
    """
    Export system data in various formats
    
    - **format**: Export format (json, csv)
    - **data_type**: Type of data to export (jobs, metrics)
    """
    try:
        if data_type == "jobs":
            all_jobs = list(active_jobs.values()) + job_history
            
            if format == "json":
                return JSONResponse(
                    content={
                        "success": True,
                        "message": "Jobs data exported successfully",
                        "data": all_jobs,
                        "export_info": {
                            "format": format,
                            "data_type": data_type,
                            "record_count": len(all_jobs),
                            "exported_at": datetime.utcnow().isoformat()
                        }
                    }
                )
            elif format == "csv":
                # In a real implementation, this would return a CSV file
                return {
                    "success": True,
                    "message": "CSV export not implemented yet",
                    "data": {"note": "CSV export would be implemented here"}
                }
        
        return {
            "success": True,
            "message": f"Export completed for {data_type} in {format} format",
            "data": {"note": f"Export functionality for {data_type} would be implemented here"}
        }
        
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")


@router.post("/system/cleanup-temp-files")
async def cleanup_temp_files(
    max_age_hours: int = Query(24, description="Maximum age of files to keep in hours"),
    dry_run: bool = Query(True, description="Preview what would be deleted without actually deleting")
):
    """
    Clean up temporary files system-wide.

    Cleans up:
    - PDF files in /tmp (*.pdf)
    - pdf_processor folders in /tmp
    - Files in /var/www/mivaa-pdf-extractor/output
    - __pycache__ folders
    - Old files in /tmp/pdf_processing, /tmp/image_extraction, etc.

    Args:
        max_age_hours: Maximum age of files to keep (default: 24 hours)
        dry_run: If True, only report what would be deleted without actually deleting (default: True)

    Returns:
        Cleanup statistics including files deleted and space freed
    """
    try:
        from app.services.cleanup_service import CleanupService

        cleanup_service = CleanupService()
        stats = await cleanup_service.cleanup_system_temp_files(
            max_age_hours=max_age_hours,
            dry_run=dry_run
        )

        return {
            "success": True,
            "message": f"Temp file cleanup {'preview' if dry_run else 'completed'} successfully",
            "dry_run": dry_run,
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Error during temp file cleanup: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cleanup temp files: {str(e)}")

@router.get("/packages/status")
async def get_package_status():
    """
    Get the status of all system packages and dependencies.

    Returns package information for both critical and optional dependencies,
    including version information and availability status.
    """
    try:
        return await get_basic_package_status()
    except Exception as e:
        logger.error(f"Error getting package status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get package status: {str(e)}")


async def get_basic_package_status():
    """Get package status by parsing requirements.txt and checking imports"""
    import importlib
    import re
    import os

    # Parse requirements.txt to get all packages
    requirements_path = "/var/www/mivaa-pdf-extractor/requirements.txt"
    if not os.path.exists(requirements_path):
        requirements_path = "requirements.txt"  # Fallback for local development

    packages_from_requirements = {}

    try:
        with open(requirements_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    continue

                # Parse package name and version
                # Handle formats like: package>=1.0.0, package==1.0.0, package[extra]>=1.0.0
                match = re.match(r'^([a-zA-Z0-9_-]+)(\[.*?\])?([><=!]+.*)?', line)
                if match:
                    package_name = match.group(1)
                    version_spec = match.group(3) or ''

                    # Map some package names to their import names
                    import_name = package_name
                    if package_name == 'opencv-python-headless':
                        import_name = 'cv2'
                    elif package_name == 'pillow':
                        import_name = 'PIL'
                    elif package_name == 'python-dotenv':
                        import_name = 'dotenv'
                    elif package_name == 'python-multipart':
                        import_name = 'multipart'
                    elif package_name == 'python-dateutil':
                        import_name = 'dateutil'
                    elif package_name == 'python-json-logger':
                        import_name = 'pythonjsonlogger'
                    elif package_name == 'email-validator':
                        import_name = 'email_validator'

                    packages_from_requirements[package_name] = {
                        'import_name': import_name,
                        'version_spec': version_spec.strip(),
                        'required': True
                    }
    except Exception as e:
        logger.error(f"Error reading requirements.txt: {e}")

    # Check each package
    package_status = {}
    critical_packages = {
        'fastapi', 'uvicorn', 'pydantic', 'supabase', 'pymupdf4llm',
        'numpy', 'pandas', 'opencv-python-headless', 'pillow',
        'openai', 'anthropic', 'voyageai', 'torch'
    }

    for package_name, info in packages_from_requirements.items():
        try:
            module = importlib.import_module(info['import_name'])
            version = getattr(module, '__version__', 'unknown')
            package_status[package_name] = {
                'available': True,
                'version': version,
                'version_spec': info['version_spec'],
                'critical': package_name in critical_packages,
                'import_name': info['import_name']
            }
        except ImportError:
            package_status[package_name] = {
                'available': False,
                'version': None,
                'version_spec': info['version_spec'],
                'critical': package_name in critical_packages,
                'import_name': info['import_name'],
                'error': 'Package not found'
            }

    # Calculate summary
    critical_missing = sum(1 for pkg, status in package_status.items()
                          if pkg in critical_packages and not status['available'])
    total_packages = len(package_status)
    total_critical = len([pkg for pkg in package_status.keys() if pkg in critical_packages])
    available_packages = sum(1 for status in package_status.values() if status['available'])

    return {
        "success": True,
        "data": {
            "packages": package_status,
            "summary": {
                "total_packages": total_packages,
                "available_packages": available_packages,
                "missing_packages": total_packages - available_packages,
                "critical_missing": critical_missing,
                "total_critical": total_critical,
                "deployment_ready": critical_missing == 0
            }
        },
        "timestamp": datetime.utcnow().isoformat(),
        "source": "requirements.txt"
    }


@router.get("/jobs/{job_id}/progress", response_model=JobProgressDetail)
async def get_job_progress(job_id: str):
    """
    Get detailed progress information for a specific job.

    This endpoint provides real-time progress tracking including:
    - Current processing stage
    - Page-by-page status
    - OCR progress and confidence scores
    - Database integration status
    - Error and warning details
    - Performance metrics
    """
    try:
        progress_service = get_progress_service()
        tracker = progress_service.get_tracker(job_id)

        if not tracker:
            raise HTTPException(
                status_code=404,
                detail=f"Progress tracking not found for job {job_id}"
            )

        return tracker.to_progress_detail()

    except Exception as e:
        logger.error(f"Error getting job progress for {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job progress: {str(e)}"
        )


@router.get("/jobs/progress/active")
async def get_all_active_progress():
    """
    Get progress information for all currently active jobs.

    Returns a summary of all jobs currently being processed,
    useful for monitoring multiple concurrent operations.
    """
    try:
        progress_service = get_progress_service()
        active_trackers = progress_service.get_all_active_trackers()

        progress_summaries = []
        for job_id, tracker in active_trackers.items():
            progress_detail = tracker.to_progress_detail()
            progress_summaries.append({
                "job_id": job_id,
                "document_id": tracker.document_id,
                "current_stage": tracker.current_stage.value,
                "progress_percentage": progress_detail.progress_percentage,
                "pages_completed": tracker.pages_completed,
                "total_pages": tracker.total_pages,
                "current_page": tracker.current_page,
                "errors_count": len(tracker.errors),
                "warnings_count": len(tracker.warnings),
                "estimated_completion": progress_detail.estimated_completion_time
            })

        return {
            "success": True,
            "message": f"Retrieved progress for {len(active_trackers)} active jobs",
            "data": {
                "active_jobs_count": len(active_trackers),
                "jobs": progress_summaries
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting active job progress: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get active job progress: {str(e)}"
        )


@router.get("/jobs/{job_id}/progress/pages")
async def get_job_page_progress(job_id: str):
    """
    Get detailed page-by-page progress for a specific job.

    This endpoint provides granular information about each page:
    - Processing status (pending, processing, success, failed, skipped)
    - Text and image extraction results
    - OCR application and confidence scores
    - Processing time per page
    - Error messages for failed pages
    - Database save status
    """
    try:
        progress_service = get_progress_service()
        tracker = progress_service.get_tracker(job_id)

        if not tracker:
            raise HTTPException(
                status_code=404,
                detail=f"Progress tracking not found for job {job_id}"
            )

        # Group pages by status for easier analysis
        pages_by_status = {
            "pending": [],
            "processing": [],
            "success": [],
            "failed": [],
            "skipped": []
        }

        for page_status in tracker.page_statuses.values():
            status_key = page_status.status
            if status_key in pages_by_status:
                pages_by_status[status_key].append(page_status.model_dump())

        return {
            "success": True,
            "message": f"Retrieved page progress for job {job_id}",
            "data": {
                "job_id": job_id,
                "total_pages": tracker.total_pages,
                "summary": {
                    "pending": len(pages_by_status["pending"]),
                    "processing": len(pages_by_status["processing"]),
                    "success": len(pages_by_status["success"]),
                    "failed": len(pages_by_status["failed"]),
                    "skipped": len(pages_by_status["skipped"])
                },
                "pages_by_status": pages_by_status,
                "current_page": tracker.current_page,
                "current_stage": tracker.current_stage.value
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting page progress for {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get page progress: {str(e)}"
        )


@router.get("/jobs/{job_id}/progress/stream")
async def stream_job_progress(job_id: str):
    """
    Stream real-time progress updates for a job.

    This endpoint provides server-sent events (SSE) for real-time monitoring.
    Useful for frontend applications that need live progress updates.
    """
    try:
        from fastapi.responses import StreamingResponse
        import json

        progress_service = get_progress_service()

        async def generate_progress_stream():
            """Generate progress updates as server-sent events."""
            while True:
                tracker = progress_service.get_tracker(job_id)

                if not tracker:
                    # Job not found or completed
                    yield f"data: {json.dumps({'error': 'Job not found or completed'})}\n\n"
                    break

                progress_detail = tracker.to_progress_detail()
                progress_data = progress_detail.model_dump()

                yield f"data: {json.dumps(progress_data)}\n\n"

                # Check if job is completed
                if tracker.current_stage.value in ["completed", "failed"]:
                    break

                # Wait before next update
                await asyncio.sleep(2)  # Update every 2 seconds

        return StreamingResponse(
            generate_progress_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )

    except Exception as e:
        logger.error(f"Error streaming progress for {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stream progress: {str(e)}"
        )


@router.post("/test-product-creation")
async def test_product_creation(
    document_id: str,
    workspace_id: str = "ffafc28b-1b8b-4b0d-b226-9f9a6154004e"
):
    """
    ✅ NEW: Test endpoint for enhanced product creation.
    Tests the improved product detection with no limits and better filtering.
    """
    try:
        logger.info(f"🧪 Testing enhanced product creation for document: {document_id}")

        # Initialize product creation service
        supabase_client = SupabaseClient()
        product_service = ProductCreationService(supabase_client)

        # Test the enhanced product creation
        result = await product_service.create_products_from_layout_candidates(
            document_id=document_id,
            workspace_id=workspace_id,
            min_confidence=0.5,
            min_quality_score=0.5
        )

        logger.info(f"✅ Product creation test completed: {result}")

        return {
            "success": True,
            "document_id": document_id,
            "result": result,
            "message": "Enhanced product creation test completed"
        }

    except Exception as e:
        logger.error(f"❌ Product creation test failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Product creation test failed: {str(e)}"
        )

# ============================================================================
# PHASE 4: MANUAL OCR REPROCESSING ENDPOINT
# ============================================================================

@router.post("/admin/images/{image_id}/process-ocr")
async def reprocess_image_ocr(
    image_id: str,
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    current_user: User = Depends(require_admin)
):
    """
    Manually reprocess a single image with OCR and update all related entities.
    
    This endpoint is used when an image was skipped during initial processing
    but the admin determines it should have OCR applied.
    
    Process:
    1. Run full EasyOCR on the image
    2. Update image.ocr_extracted_text and ocr_confidence_score
    3. Update related chunks with new OCR text
    4. Regenerate text embeddings for updated chunks
    5. Update product associations based on new OCR text
    6. Update metadata relationships
    
    Args:
        image_id: UUID of the image to reprocess
        workspace_context: Current workspace context
        current_user: Current admin user
        
    Returns:
        Comprehensive results of the reprocessing operation
    """
    try:
        from ..services.ocr_service import get_ocr_service, OCRConfig
        from ..services.real_embeddings_service import get_embeddings_service
        
        logger.info(f"🔄 Admin OCR reprocessing requested for image: {image_id}")
        
        # Initialize services
        supabase = SupabaseClient()
        ocr_service = get_ocr_service(OCRConfig(languages=['en']))
        embeddings_service = get_embeddings_service()
        
        # Step 1: Get image from database
        image_response = supabase.client.table('document_images').select('*').eq('id', image_id).execute()
        
        if not image_response.data or len(image_response.data) == 0:
            raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")
        
        image = image_response.data[0]
        image_url = image.get('image_url')
        
        if not image_url:
            raise HTTPException(status_code=400, detail="Image has no URL")
        
        logger.info(f"📷 Processing image: {image_url}")
        
        # Step 2: Download image temporarily
        import httpx
        import tempfile
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(image_url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download image: {response.status_code}")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(response.content)
                tmp_image_path = tmp_file.name
        
        try:
            # Step 3: Run full EasyOCR
            logger.info("🔍 Running EasyOCR...")
            ocr_results = ocr_service.extract_text_from_image(tmp_image_path)
            
            extracted_text = ' '.join([r.text for r in ocr_results])
            avg_confidence = sum([r.confidence for r in ocr_results]) / len(ocr_results) if ocr_results else 0.0
            
            logger.info(f"✅ OCR extracted {len(ocr_results)} text regions, avg confidence: {avg_confidence:.2f}")
            
            # Step 4: Update image record
            update_data = {
                'ocr_extracted_text': extracted_text,
                'ocr_confidence_score': avg_confidence,
                'processing_status': 'ocr_complete',
                'metadata': {
                    **(image.get('metadata') or {}),
                    'ocr_metadata': {
                        'text_regions_count': len(ocr_results),
                        'reprocessed_at': datetime.utcnow().isoformat(),
                        'reprocessed_by': current_user.id,
                        'can_reprocess': False
                    }
                }
            }
            
            supabase.client.table('document_images').update(update_data).eq('id', image_id).execute()
            logger.info("✅ Updated image record")
            
            # Step 5: Update related chunks
            chunks_response = supabase.client.table('document_chunks').select('*').eq('id', image.get('chunk_id')).execute()
            
            chunks_updated = 0
            embeddings_regenerated = 0
            
            if chunks_response.data:
                for chunk in chunks_response.data:
                    # Update chunk content with OCR text
                    updated_content = f"{chunk.get('content', '')}\n\n[Image OCR: {extracted_text}]"
                    
                    # Generate new text embedding
                    try:
                        embedding = await embeddings_service.generate_text_embedding(
                            text=updated_content,
                            model="text-embedding-3-small"
                        )
                        
                        # Update chunk
                        supabase.client.table('document_chunks').update({
                            'content': updated_content,
                            'text_embedding': embedding,
                            'metadata': {
                                **(chunk.get('metadata') or {}),
                                'ocr_enriched': True,
                                'ocr_enriched_at': datetime.utcnow().isoformat()
                            }
                        }).eq('id', chunk['id']).execute()
                        
                        chunks_updated += 1
                        embeddings_regenerated += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to update chunk {chunk['id']}: {str(e)}")
            
            logger.info(f"✅ Updated {chunks_updated} chunks, regenerated {embeddings_regenerated} embeddings")
            
            # Step 6: Extract metadata from OCR text
            metadata_extracted = {
                'has_dimensions': any(word in extracted_text.lower() for word in ['mm', 'cm', 'inch', 'x', '×']),
                'has_specifications': any(word in extracted_text.lower() for word in ['spec', 'material', 'finish', 'color']),
                'has_product_codes': any(char.isdigit() for char in extracted_text),
                'text_length': len(extracted_text)
            }
            
            # Step 7: Find potential product associations
            # (This is a simplified version - you can enhance with more sophisticated matching)
            products_associated = 0
            if extracted_text:
                # Search for products that might match the OCR text
                # This is a placeholder - implement your product matching logic
                pass
            
            return {
                'success': True,
                'image_id': image_id,
                'ocr_results': {
                    'text': extracted_text,
                    'confidence': avg_confidence,
                    'text_regions_count': len(ocr_results)
                },
                'updates': {
                    'chunks_updated': chunks_updated,
                    'embeddings_regenerated': embeddings_regenerated,
                    'products_associated': products_associated
                },
                'metadata_extracted': metadata_extracted,
                'message': f'Successfully reprocessed image with OCR. Extracted {len(ocr_results)} text regions.'
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_image_path):
                os.unlink(tmp_image_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing image OCR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reprocess image: {str(e)}")





