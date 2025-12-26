"""
Job Routes - Job management and monitoring
"""

import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status, BackgroundTasks
from fastapi.responses import JSONResponse
import httpx
import uuid

from app.services.supabase_client import get_supabase_client
from app.services.checkpoint_recovery_service import checkpoint_recovery_service, ProcessingStage
from app.services.stuck_job_analyzer import stuck_job_analyzer
from app.services.ai_model_tracker import AIModelTracker
from .shared import job_storage, run_async_in_background

logger = logging.getLogger(__name__)
router = APIRouter()

# Reserved keywords that should not be treated as job IDs
RESERVED_KEYWORDS = {'health', 'status', 'metrics', 'list', 'all'}


def validate_job_id(job_id: str) -> None:
    """
    Validate that job_id is a valid UUID and not a reserved keyword.

    Args:
        job_id: The job ID to validate

    Raises:
        HTTPException: If job_id is invalid or a reserved keyword
    """
    # CRITICAL FIX: Better error messages for reserved keywords (fixes MIVAA-4W)
    # Health checks were hitting /api/documents/job/health instead of /health
    if job_id.lower() in RESERVED_KEYWORDS:
        # Provide helpful redirect information
        endpoint_map = {
            'health': '/health or /api/health',
            'status': '/api/jobs/list',
            'metrics': '/health/metrics',
            'list': '/api/jobs/list',
            'all': '/api/jobs/list'
        }
        suggested_endpoint = endpoint_map.get(job_id.lower(), 'the appropriate endpoint')

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid job ID: '{job_id}' is a reserved keyword. "
                   f"Did you mean to use {suggested_endpoint}? "
                   f"Job IDs must be valid UUIDs (e.g., '550e8400-e29b-41d4-a716-446655440000')."
        )

    # Validate UUID format
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid job ID format: '{job_id}'. Must be a valid UUID (e.g., '550e8400-e29b-41d4-a716-446655440000')."
        )


@router.get("/documents/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of an async document processing job with checkpoint information.

    ALWAYS queries the database first as the source of truth, then optionally merges
    with in-memory data for additional real-time details.

    Returns:
        - Job status and progress (from database)
        - Latest checkpoint information
        - Detailed metadata including AI usage, chunks, images, products
        - In-memory state comparison (if available)
    """
    # Validate job_id format
    validate_job_id(job_id)

    # ALWAYS check database FIRST - this is the source of truth
    try:
        supabase_client = get_supabase_client()
        logger.info(f"?? [DB QUERY] Checking database for job {job_id}")
        response = supabase_client.client.table('background_jobs').select('*').eq('id', job_id).execute()
        logger.info(f"?? [DB QUERY] Database response: data={response.data}, count={len(response.data) if response.data else 0}")

        if response.data and len(response.data) > 0:
            job = response.data[0]
            logger.info(f"? [DB QUERY] Found job in database: {job['id']}, status={job['status']}, progress={job.get('progress', 0)}%")

            # Build response from DATABASE data (source of truth)
            job_response = {
                "job_id": job['id'],
                "status": job['status'],
                "document_id": job.get('document_id'),
                "progress": job.get('progress', 0),
                "error": job.get('error'),
                "metadata": job.get('metadata', {}),
                "created_at": job.get('created_at'),
                "updated_at": job.get('updated_at'),
                "source": "database"  # Indicate this came from DB
            }

            # Optionally merge with in-memory data for comparison/debugging
            if job_id in job_storage:
                memory_data = job_storage[job_id]
                logger.info(f"?? [COMPARISON] In-memory status: {memory_data.get('status')}, progress: {memory_data.get('progress', 0)}%")

                # Add comparison data
                job_response["memory_state"] = {
                    "status": memory_data.get('status'),
                    "progress": memory_data.get('progress', 0),
                    "matches_db": (
                        memory_data.get('status') == job['status'] and
                        memory_data.get('progress', 0) == job.get('progress', 0)
                    )
                }

                # Log discrepancies
                if not job_response["memory_state"]["matches_db"]:
                    logger.warning(
                        f"?? [MISMATCH] DB vs Memory mismatch for job {job_id}: "
                        f"DB({job['status']}, {job.get('progress', 0)}%) vs "
                        f"Memory({memory_data.get('status')}, {memory_data.get('progress', 0)}%)"
                    )

            # Add checkpoint information
            try:
                last_checkpoint = await checkpoint_recovery_service.get_last_checkpoint(job_id)
                if last_checkpoint:
                    job_response["last_checkpoint"] = {
                        "stage": last_checkpoint.get('stage'),
                        "created_at": last_checkpoint.get('created_at'),
                        "data": last_checkpoint.get('checkpoint_data', {})
                    }
            except Exception as e:
                logger.error(f"Failed to get checkpoint for job {job_id}: {e}")

            return JSONResponse(content=job_response)
        else:
            logger.warning(f"?? [DB QUERY] Job {job_id} not found in database")

            # Check if it exists in memory (shouldn't happen in normal flow)
            if job_id in job_storage:
                logger.error(
                    f"?? [CRITICAL] Job {job_id} exists in memory but NOT in database! "
                    f"This indicates a database sync failure."
                )
                # Create serializable copy of job_storage (exclude ai_tracker)
                memory_state = {k: v for k, v in job_storage[job_id].items() if k != 'ai_tracker'}
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Database sync failure",
                        "detail": "Job exists in memory but not in database",
                        "job_id": job_id,
                        "memory_state": memory_state
                    }
                )

    except Exception as e:
        logger.error(f"? [DB ERROR] Error checking database for job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query failed: {str(e)}"
        )

    # Job not found in database or memory
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Job {job_id} not found in database"
    )

@router.get("/jobs/{job_id}/checkpoints")
async def get_job_checkpoints(job_id: str):
    """
    Get all checkpoints for a job.

    Returns:
        - List of all checkpoints with stage, data, and metadata
        - Checkpoint count
        - Processing timeline
    """
    # Validate job_id format
    validate_job_id(job_id)

    try:
        checkpoints = await checkpoint_recovery_service.get_all_checkpoints(job_id)

        return JSONResponse(content={
            "job_id": job_id,
            "checkpoints": checkpoints,
            "count": len(checkpoints),
            "stages_completed": [cp.get('stage') for cp in checkpoints]
        })
    except Exception as e:
        logger.error(f"Failed to get checkpoints for job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve checkpoints: {str(e)}"
        )

@router.post("/jobs/{job_id}/restart")
async def restart_job_from_checkpoint(job_id: str, background_tasks: BackgroundTasks):
    """
    Manually restart a job from its last checkpoint.

    Note: Imports background processing functions inside the function to avoid circular imports.

    This endpoint allows manual recovery of stuck or failed jobs.
    The job will resume from the last successful checkpoint.
    """
    # Validate job_id format
    validate_job_id(job_id)

    try:
        # Get last checkpoint
        last_checkpoint = await checkpoint_recovery_service.get_last_checkpoint(job_id)

        if not last_checkpoint:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No checkpoint found for job {job_id}"
            )

        # Verify checkpoint data exists
        resume_stage_str = last_checkpoint.get('stage')
        resume_stage = ProcessingStage(resume_stage_str)
        can_resume = await checkpoint_recovery_service.verify_checkpoint_data(job_id, resume_stage)

        if not can_resume:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Checkpoint data verification failed for stage {resume_stage}"
            )

        # Get job details from database
        supabase_client = get_supabase_client()
        job_result = supabase_client.client.table('background_jobs').select('*').eq('id', job_id).execute()

        if not job_result.data or len(job_result.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found in database"
            )

        job_data = job_result.data[0]
        document_id = job_data['document_id']

        # Mark job for restart
        supabase_client.client.table('background_jobs').update({
            "status": "processing",  # ? Set to processing immediately
            "error": None,  # Clear previous error
            "interrupted_at": None,  # Clear interrupted timestamp
            "started_at": datetime.utcnow().isoformat(),
            "metadata": {
                **job_data.get('metadata', {}),
                "restart_from_stage": resume_stage.value,
                "restart_reason": "manual_restart",
                "restart_at": datetime.utcnow().isoformat()
            }
        }).eq('id', job_id).execute()

        logger.info(f"? Job {job_id} marked for restart from {resume_stage}")

        # ? CRITICAL FIX: Restart the job by calling the LlamaIndex service directly
        # The process_document_background function doesn't support resume_from_checkpoint
        # Instead, we need to trigger the processing through the service layer

        # Get the file content from storage
        try:
            # Get document details
            doc_result = supabase_client.client.table('documents').select('*').eq('id', document_id).execute()
            if not doc_result.data or len(doc_result.data) == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document {document_id} not found"
                )

            doc_data = doc_result.data[0]
            file_path = doc_data.get('file_path')
            filename = doc_data.get('filename', 'document.pdf')
            metadata = doc_data.get('metadata', {})

            # CRITICAL FIX: If file_path is a local temp file, use file_url from metadata instead
            if file_path and file_path.startswith('/tmp/'):
                file_url = metadata.get('file_url')
                if file_url:
                    logger.info(f"?? file_path is local temp file ({file_path}), using file_url from metadata: {file_url}")
                    file_path = file_url
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Document {document_id} has local temp file_path but no file_url in metadata"
                    )

            if not file_path:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Document {document_id} has no file_path"
                )

            # Download file from storage or URL
            logger.info(f"?? Downloading file from: {file_path}")

            # Check if file_path is a full URL (starts with http:// or https://)
            if file_path.startswith('http://') or file_path.startswith('https://'):
                # Download from URL with extended timeout for large PDFs
                import httpx
                async with httpx.AsyncClient(timeout=60.0) as client:  # 60 second timeout for large files
                    response = await client.get(file_path)
                    response.raise_for_status()
                    file_response = response.content
                    logger.info(f"? Downloaded file from URL: {len(file_response)} bytes")
            else:
                # Download from Supabase storage
                bucket_name = file_path.split('/')[0] if '/' in file_path else 'pdf-documents'
                storage_path = '/'.join(file_path.split('/')[1:]) if '/' in file_path else file_path
                file_response = supabase_client.client.storage.from_(bucket_name).download(storage_path)
            if not file_response:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"File not found in storage: {file_path}"
                )

            file_content = file_response
            logger.info(f"? Downloaded file: {len(file_content)} bytes")

            # Initialize job in job_storage (CRITICAL: required by process_document_background)
            job_storage[job_id] = {
                "job_id": job_id,
                "document_id": document_id,
                "status": "processing",
                "progress": job_data.get('progress', 0),
                "metadata": job_data.get('metadata', {})
            }
            logger.info(f"? Job {job_id} added to job_storage for resume")

            # Determine which processing function to use based on job_type
            job_type = job_data.get('job_type', 'document_upload')

            if job_type == 'product_discovery_upload':
                # Use product discovery pipeline for resume
                logger.info(f"?? Resuming product discovery job {job_id}")

                # Extract parameters from job metadata
                job_metadata = job_data.get('metadata', {})
                discovery_model = job_metadata.get('discovery_model', 'claude-sonnet-4.5')
                categories = job_metadata.get('categories', ['products'])
                enable_prompt_enhancement = job_metadata.get('prompt_enhancement_enabled', False)
                agent_prompt = job_metadata.get('agent_prompt')

                # Determine focused extraction based on categories
                use_focused_extraction = 'all' not in categories

                logger.info(f"   Resume parameters: discovery_model={discovery_model}, categories={categories}, focused={use_focused_extraction}")

                # Import background processing function (avoid circular import)
                from app.api.rag_routes import process_document_with_discovery

                background_tasks.add_task(
                    run_async_in_background(process_document_with_discovery),
                    job_id=job_id,
                    document_id=document_id,
                    file_content=file_content,
                    filename=filename,
                    workspace_id=doc_data.get('workspace_id', 'ffafc28b-1b8b-4b0d-b226-9f9a6154004e'),
                    title=doc_data.get('title'),
                    description=doc_data.get('description'),
                    document_tags=doc_data.get('tags', []),
                    discovery_model=discovery_model,
                    focused_extraction=use_focused_extraction,
                    extract_categories=categories,
                    chunk_size=1000,
                    chunk_overlap=200,
                    agent_prompt=agent_prompt,
                    enable_prompt_enhancement=enable_prompt_enhancement
                )
            else:
                # Use standard processing for resume
                logger.info(f"?? Resuming standard document job {job_id}")

                # Import background processing function (avoid circular import)
                from app.api.rag_routes import process_document_background

                background_tasks.add_task(
                    run_async_in_background(process_document_background),
                    job_id=job_id,
                    document_id=document_id,
                    file_content=file_content,
                    filename=filename,
                    title=doc_data.get('title'),
                    description=doc_data.get('description'),
                    document_tags=doc_data.get('tags', []),
                    chunk_size=1000,
                    chunk_overlap=200,
                    llamaindex_service=None  # Will be retrieved from app state
                )

            logger.info(f"? Background task triggered for job {job_id} (type: {job_type})")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to download file for restart: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to download file for restart: {str(e)}"
            )

        return JSONResponse(content={
            "success": True,
            "message": f"Job restarted from checkpoint: {resume_stage}",
            "job_id": job_id,
            "restart_stage": resume_stage.value,
            "checkpoint_data": last_checkpoint.get('checkpoint_data', {})
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restart job: {str(e)}"
        )

@router.post("/documents/job/{job_id}/resume")
async def resume_job(job_id: str, background_tasks: BackgroundTasks):
    """
    Resume a job from its last checkpoint (alias for restart).

    This endpoint is the same as /jobs/{job_id}/restart but with a more intuitive name.
    """
    # Validate job_id format
    validate_job_id(job_id)

    return await restart_job_from_checkpoint(job_id, background_tasks)

@router.get("/documents/jobs")
async def list_jobs(
    limit: int = 10,
    offset: int = 0,
    status_filter: Optional[str] = None,
    sort: str = "created_at:desc"
):
    """
    List all background jobs with optional filtering and sorting.

    Args:
        limit: Maximum number of jobs to return (default: 10)
        offset: Number of jobs to skip (default: 0)
        status_filter: Filter by status (pending, processing, completed, failed, interrupted)
        sort: Sort order (created_at:desc, created_at:asc, progress:desc, progress:asc)

    Returns:
        List of jobs with status, progress, and metadata
    """
    try:
        supabase_client = get_supabase_client()

        # Build query
        query = supabase_client.client.table('background_jobs').select('*')

        # Apply status filter
        if status_filter:
            query = query.eq('status', status_filter)

        # Apply sorting
        if ':' in sort:
            field, direction = sort.split(':')
            ascending = direction.lower() == 'asc'
            query = query.order(field, desc=not ascending)
        else:
            query = query.order('created_at', desc=True)

        # Apply pagination
        query = query.range(offset, offset + limit - 1)

        # Execute query
        result = query.execute()

        jobs = result.data if result.data else []

        return JSONResponse(content={
            "jobs": jobs,
            "count": len(jobs),
            "limit": limit,
            "offset": offset
        })

    except Exception as e:
        logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}"
        )

@router.delete("/documents/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and all its associated data.

    This endpoint:
    1. Removes job from in-memory job_storage
    2. Deletes job record from database
    3. Cleans up any temporary files associated with the job

    Args:
        job_id: The unique identifier of the job to delete

    Returns:
        Success message with deleted job_id

    Raises:
        HTTPException: If job not found or deletion fails
    """
    # Validate job_id format
    validate_job_id(job_id)

    try:
        logger.info(f"??? DELETE /documents/jobs/{job_id} - Deleting job")

        # Remove from in-memory storage if exists
        if job_id in job_storage:
            del job_storage[job_id]
            logger.info(f"   ? Removed job {job_id} from job_storage")

        # Delete from database
        supabase_client = get_supabase_client()
        result = supabase_client.client.table('background_jobs').delete().eq('id', job_id).execute()

        if not result.data:
            logger.warning(f"   ?? Job {job_id} not found in database")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )

        logger.info(f"   ? Deleted job {job_id} from database")

        # NOTE: Temporary file cleanup moved to admin panel cron job

        return {
            "success": True,
            "message": f"Job {job_id} deleted successfully",
            "job_id": job_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete job: {str(e)}"
        )

@router.get("/job/{job_id}/ai-tracking")
async def get_job_ai_tracking(job_id: str):
    """
    Get detailed AI model tracking information for a job.

    Returns comprehensive metrics on:
    - Which AI models were used (LLAMA, Anthropic, CLIP, OpenAI)
    - Confidence scores and results
    - Token usage and processing time
    - Success/failure rates
    - Per-stage breakdown
    """
    # Validate job_id format
    validate_job_id(job_id)

    try:
        if job_id not in job_storage:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )

        job_info = job_storage[job_id]
        ai_tracker = job_info.get("ai_tracker")

        if not ai_tracker:
            return {
                "job_id": job_id,
                "message": "No AI tracking data available for this job",
                "status": job_info.get("status", "unknown")
            }

        # Get comprehensive summary
        summary = ai_tracker.get_job_summary()

        return {
            "job_id": job_id,
            "status": job_info.get("status", "processing"),
            "progress": job_info.get("progress", 0),
            "ai_tracking": summary,
            "metadata": job_info.get("metadata", {})
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get AI tracking for job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get AI tracking: {str(e)}"
        )

@router.get("/job/{job_id}/ai-tracking/stage/{stage}")
async def get_job_ai_tracking_by_stage(job_id: str, stage: str):
    """
    Get AI model tracking information for a specific processing stage.

    Args:
        job_id: Job identifier
        stage: Processing stage (classification, boundary_detection, embedding, etc.)

    Returns:
        Detailed metrics for the specified stage
    """
    # Validate job_id format
    validate_job_id(job_id)

    try:
        if job_id not in job_storage:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )

        job_info = job_storage[job_id]
        ai_tracker = job_info.get("ai_tracker")

        if not ai_tracker:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No AI tracking data available for this job"
            )

        stage_details = ai_tracker.get_stage_details(stage)

        if not stage_details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No tracking data for stage: {stage}"
            )

        return stage_details

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get AI tracking for job {job_id} stage {stage}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get AI tracking: {str(e)}"
        )

@router.get("/job/{job_id}/ai-tracking/model/{model_name}")
async def get_job_ai_tracking_by_model(job_id: str, model_name: str):
    """
    Get AI model tracking information for a specific AI model.

    Args:
        job_id: Job identifier
        model_name: AI model name (LLAMA, Anthropic, CLIP, OpenAI)

    Returns:
        Statistics for the specified AI model
    """
    # Validate job_id format
    validate_job_id(job_id)

    try:
        if job_id not in job_storage:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )

        job_info = job_storage[job_id]
        ai_tracker = job_info.get("ai_tracker")

        if not ai_tracker:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No AI tracking data available for this job"
            )

        model_stats = ai_tracker.get_model_stats(model_name)

        if not model_stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No tracking data for model: {model_name}"
            )

        return model_stats

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get AI tracking for job {job_id} model {model_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get AI tracking: {str(e)}"
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return AdvancedQueryResponse(
            original_query=request.query,
            optimized_query=results.get('optimized_query', request.query),
            query_type=request.query_type,
            results=results.get('results', []),
            total_results=results.get('total_results', 0),
            expansion_terms=results.get('expansion_terms', []),
            processing_time=processing_time,
            confidence_score=results.get('confidence_score', 0.0)
        )
        
    except ValueError as e:
        logger.error(f"Invalid query parameters: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid query parameters: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Advanced query search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advanced query search failed: {str(e)}"
        )

@router.get("/admin/stuck-jobs/analyze/{job_id}")
async def analyze_stuck_job(job_id: str):
    """
    Analyze a stuck job to determine root cause and get recommendations.

    Returns detailed analysis including:
    - Root cause identification
    - Bottleneck stage
    - Stage-by-stage timing analysis
    - Recovery options
    - Optimization recommendations
    """
    # Validate job_id format
    validate_job_id(job_id)

    try:
        analysis = await stuck_job_analyzer.analyze_stuck_job(job_id)
        return JSONResponse(content=analysis)
    except Exception as e:
        logger.error(f"Failed to analyze stuck job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze stuck job: {str(e)}"
        )

@router.get("/admin/stuck-jobs/statistics")
async def get_stuck_job_statistics():
    """
    Get overall statistics about stuck jobs.

    Returns:
    - Total stuck jobs
    - Stage breakdown (which stages jobs get stuck at)
    - Most common stuck stage
    - Historical patterns
    """
    try:
        stats = await stuck_job_analyzer.get_stuck_job_statistics()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Failed to get stuck job statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stuck job statistics: {str(e)}"
        )
