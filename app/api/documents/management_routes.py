"""
Document Management API Routes

This module handles all document CRUD operations including:
- Job status tracking and management
- Job checkpoints and recovery
- Document content retrieval
- AI tracking and metrics
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, status
from fastapi.responses import JSONResponse

try:
    from pydantic import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field

from app.services.core.supabase_client import get_supabase_client
from app.services.tracking.checkpoint_recovery_service import CheckpointRecoveryService
from app.schemas.jobs import ProcessingStage
from app.config import get_settings

# Import orchestration functions from centralized module
# CONSOLIDATED: All jobs now use process_document_with_discovery (removed legacy process_document_background)
from app.orchestration import (
    run_async_in_background,
    process_document_with_discovery,
)

# Import for file download
import httpx
import tempfile

from app.utils.resource_manager import get_resource_manager

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(tags=["documents"])

# Initialize services
checkpoint_recovery_service = CheckpointRecoveryService()

# In-memory job storage (imported from rag_routes for compatibility)
# This will be replaced with proper service layer in future refactoring
from app.api.rag_routes import job_storage


# ============================================================================
# Job Status and Management Endpoints
# ============================================================================

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
    # ALWAYS check database FIRST - this is the source of truth
    try:
        supabase_client = get_supabase_client()
        logger.info(f"🔍 [DB QUERY] Checking database for job {job_id}")
        response = supabase_client.client.table('background_jobs').select('*').eq('id', job_id).execute()
        logger.info(f"🔍 [DB QUERY] Database response: data={response.data}, count={len(response.data) if response.data else 0}")

        if response.data and len(response.data) > 0:
            job = response.data[0]
            logger.info(f"✅ [DB QUERY] Found job in database: {job['id']}, status={job['status']}, progress={job.get('progress', 0)}%")

            # Build response from DATABASE data (source of truth)
            # UPDATED: Now includes ALL database columns for complete job information
            job_response = {
                # Core identifiers
                "job_id": job['id'],
                "document_id": job.get('document_id'),
                "filename": job.get('filename'),
                "job_type": job.get('job_type', 'pdf_processing'),
                "workspace_id": job.get('workspace_id'),

                # Status and progress
                "status": job['status'],
                "progress": job.get('progress', 0),
                "error": job.get('error'),

                # Timestamps
                "created_at": job.get('created_at'),
                "updated_at": job.get('updated_at'),
                "started_at": job.get('started_at'),
                "completed_at": job.get('completed_at'),
                "failed_at": job.get('failed_at'),
                "interrupted_at": job.get('interrupted_at'),

                # Recovery and monitoring
                "last_heartbeat": job.get('last_heartbeat'),
                "recovery_attempts": job.get('recovery_attempts', 0),
                "last_recovery_at": job.get('last_recovery_at'),

                # Relationships
                "parent_job_id": job.get('parent_job_id'),

                # Data
                "metadata": job.get('metadata', {}),
                "last_checkpoint": job.get('last_checkpoint'),

                # Debug info
                "source": "database"  # Indicate this came from DB
            }

            # Optionally merge with in-memory data for comparison/debugging
            if job_id in job_storage:
                memory_data = job_storage[job_id]
                logger.info(f"📊 [COMPARISON] In-memory status: {memory_data.get('status')}, progress: {memory_data.get('progress', 0)}%")

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
                        f"⚠️ [MISMATCH] DB vs Memory mismatch for job {job_id}: "
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
            logger.warning(f"⚠️ [DB QUERY] Job {job_id} not found in database")

            # Check if it exists in memory (shouldn't happen in normal flow)
            if job_id in job_storage:
                logger.error(
                    f"🚨 [CRITICAL] Job {job_id} exists in memory but NOT in database! "
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
        logger.error(f"❌ [DB ERROR] Error checking database for job {job_id}: {e}", exc_info=True)
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

    This endpoint allows manual recovery of stuck or failed jobs.
    The job will resume from the last successful checkpoint.
    """
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
            "status": "processing",  # ✅ Set to processing immediately
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

        logger.info(f"✅ Job {job_id} marked for restart from {resume_stage}")

        # ✅ CRITICAL FIX: Restart the job by re-triggering the processing pipeline
        # The process_document_with_discovery function supports checkpoint recovery
        # Documents are processed directly into the vector database

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
                    logger.info(f"⚠️ file_path is local temp file ({file_path}), using file_url from metadata: {file_url}")
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
            logger.info(f"📥 Downloading file from: {file_path}")

            # Check if file_path is a full URL (starts with http:// or https://)
            if file_path.startswith('http://') or file_path.startswith('https://'):
                # Download from URL with extended timeout for large PDFs
                async with httpx.AsyncClient(timeout=60.0) as client:  # 60 second timeout for large files
                    response = await client.get(file_path)
                    response.raise_for_status()
                    file_response = response.content
                    logger.info(f"✅ Downloaded file from URL: {len(file_response)} bytes")
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
            logger.info(f"✅ Downloaded file: {len(file_content)} bytes")

            # STREAMING REFACTOR: Save to temp file for processing
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_file.write(file_content)
            temp_file.close()
            # Update file_path to point to local temp file
            file_path = temp_file.name
            logger.info(f"✅ Saved to temp file for processing: {file_path}")

            # Register temp file with ResourceManager for cleanup tracking
            resource_manager = get_resource_manager()
            await resource_manager.register_resource(
                resource_id=f"temp_pdf_{document_id}",
                resource_type="temp_file",
                path=file_path,
                job_id=job_id,
                metadata={"filename": filename, "source": "management_routes_resume"}
            )
            logger.info(f"✅ Registered temp PDF with ResourceManager: {file_path}")

            # Free memory
            file_content = None

            # Initialize job in job_storage (CRITICAL: required by process_document_with_discovery)
            job_storage[job_id] = {
                "job_id": job_id,
                "document_id": document_id,
                "status": "processing",
                "progress": job_data.get('progress', 0),
                "metadata": job_data.get('metadata', {})
            }
            logger.info(f"✅ Job {job_id} added to job_storage for resume")

            # CONSOLIDATED: All jobs now use process_document_with_discovery
            # This pipeline handles checkpoint recovery and continues from where it left off
            job_type = job_data.get('job_type', 'document_upload')
            logger.info(f"🔄 Resuming job {job_id} (type: {job_type}) using unified discovery pipeline")

            # Extract parameters from job metadata (works for both legacy and discovery jobs)
            job_metadata = job_data.get('metadata', {})
            discovery_model = job_metadata.get('discovery_model', 'claude-sonnet-4.5')
            categories = job_metadata.get('categories', ['products'])
            enable_prompt_enhancement = job_metadata.get('prompt_enhancement_enabled', False)
            agent_prompt = job_metadata.get('agent_prompt')
            test_single_product = job_metadata.get('test_single_product', False)

            # Determine focused extraction based on categories
            use_focused_extraction = 'all' not in categories

            logger.info(f"   Resume parameters: discovery_model={discovery_model}, categories={categories}, focused={use_focused_extraction}")

            background_tasks.add_task(
                run_async_in_background(process_document_with_discovery),
                job_id=job_id,
                document_id=document_id,
                file_path=file_path,
                filename=filename,
                workspace_id=doc_data.get('workspace_id') or get_settings().default_workspace_id,
                title=doc_data.get('title'),
                description=doc_data.get('description'),
                document_tags=doc_data.get('tags', []),
                discovery_model=discovery_model,
                focused_extraction=use_focused_extraction,
                extract_categories=categories,
                chunk_size=1000,
                chunk_overlap=200,
                agent_prompt=agent_prompt,
                enable_prompt_enhancement=enable_prompt_enhancement,
                test_single_product=test_single_product
            )

            logger.info(f"✅ Background task triggered for job {job_id} using unified pipeline")

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
    Delete a job and ALL its associated data.

    This endpoint performs complete cleanup including:
    1. Job record from background_jobs table
    2. Document record (if exists)
    3. All chunks from document_chunks
    4. All embeddings from vecs collections
    5. All images from document_images
    6. All products
    7. Files from storage buckets
    8. Checkpoints
    9. Temporary files
    10. In-memory job_storage

    Args:
        job_id: The unique identifier of the job to delete

    Returns:
        Success message with deletion statistics

    Raises:
        HTTPException: If job not found or deletion fails
    """
    try:
        logger.info(f"🗑️ DELETE /documents/jobs/{job_id} - Starting complete deletion")

        # Remove from in-memory storage if exists
        if job_id in job_storage:
            del job_storage[job_id]
            logger.info(f"   ✅ Removed job {job_id} from job_storage")

        # Get services
        supabase_client = get_supabase_client()
        from app.services.core.vecs_service import get_vecs_service
        vecs_service = get_vecs_service()

        # Import cleanup service
        from app.services.utilities.cleanup_service import CleanupService
        cleanup_service = CleanupService()

        # Perform complete deletion (manual deletion from UI - includes storage files)
        stats = await cleanup_service.delete_job_completely(
            job_id=job_id,
            supabase_client=supabase_client,
            vecs_service=vecs_service,
            delete_storage_files=True  # ✅ Manual deletion includes storage files
        )

        # Check if job was actually deleted
        if not stats['job_deleted']:
            logger.warning(f"   ⚠️ Job {job_id} not found or deletion failed")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found or deletion failed"
            )

        logger.info(f"   ✅ Complete deletion finished for job {job_id}")
        logger.info(f"   📊 Stats: {stats}")

        return {
            "success": True,
            "message": f"Job {job_id} and all associated data deleted successfully",
            "job_id": job_id,
            "stats": stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete job: {str(e)}"
        )


# ============================================================================
# Document Content Endpoints
# ============================================================================

@router.get("/documents/documents/{document_id}/content")
async def get_document_content(
    document_id: str,
    include_chunks: bool = Query(True, description="Include document chunks"),
    include_images: bool = Query(True, description="Include document images"),
    include_products: bool = Query(False, description="Include products created from document")
):
    """
    Get complete document content with all AI analysis results.

    Returns comprehensive document data including:
    - Document metadata
    - All chunks with embeddings
    - All images with AI analysis (CLIP, Qwen, Claude)
    - All products created from the document
    - Complete AI model usage statistics
    """
    try:
        logger.info(f"📊 Fetching complete content for document {document_id}")
        supabase_client = get_supabase_client()

        # Get document metadata
        doc_response = supabase_client.client.table('documents').select('*').eq('id', document_id).execute()
        if not doc_response.data or len(doc_response.data) == 0:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

        document = doc_response.data[0]
        result = {
            "id": document['id'],
            "created_at": document['created_at'],
            "metadata": document.get('metadata', {}),
            "chunks": [],
            "images": [],
            "products": [],
            "statistics": {}
        }

        # Get chunks with embeddings
        if include_chunks:
            logger.info(f"📄 Fetching chunks for document {document_id}")
            chunks_response = supabase_client.client.table('document_chunks').select('*').eq('document_id', document_id).execute()
            chunks = chunks_response.data or []

            # Embeddings are stored directly in document_chunks.text_embedding
            for chunk in chunks:
                text_embedding = chunk.get('text_embedding')
                dimension = chunk.get('embedding_dimension', 1024)
                chunk['embeddings'] = [{'embedding': text_embedding, 'type': f'text_{dimension}'}] if text_embedding else []

            result['chunks'] = chunks
            logger.info(f"✅ Fetched {len(chunks)} chunks")

        # Get images with AI analysis
        if include_images:
            logger.info(f"🖼️ Fetching images for document {document_id}")
            images_response = supabase_client.client.table('document_images').select('*').eq('document_id', document_id).execute()
            result['images'] = images_response.data or []
            logger.info(f"✅ Fetched {len(result['images'])} images")

        # Get products
        if include_products:
            logger.info(f"🏭 Fetching products for document {document_id}")
            products_response = supabase_client.client.table('products').select('*').eq('source_document_id', document_id).execute()
            result['products'] = products_response.data or []
            logger.info(f"✅ Fetched {len(result['products'])} products")

        # Calculate statistics
        chunks_count = len(result['chunks'])
        images_count = len(result['images'])
        products_count = len(result['products'])

        # Count embeddings
        text_embeddings = sum(1 for chunk in result['chunks'] if chunk.get('embeddings'))
        visual_embeddings = sum(1 for img in result['images'] if img.get('visual_clip_embedding_512'))
        vision_analysis = sum(1 for img in result['images'] if img.get('vision_analysis'))
        claude_validation = sum(1 for img in result['images'] if img.get('claude_validation'))
        # Understanding embeddings are stored in VECS (not document_images columns)
        # Images with vision_analysis have understanding embeddings generated
        understanding_embeddings = vision_analysis
        # Specialized embeddings (color, texture, style, material) are stored in VECS collections
        # Visual embedding count serves as proxy for SLIG specialized embeddings

        result['statistics'] = {
            "chunks_count": chunks_count,
            "images_count": images_count,
            "products_count": products_count,
            "ai_usage": {
                "openai_calls": text_embeddings,
                "vision_calls": vision_analysis,
                "claude_calls": claude_validation,
                "visual_embeddings": visual_embeddings
            },
            "embeddings_generated": {
                "text": text_embeddings,
                "visual": visual_embeddings,
                "understanding": understanding_embeddings,
                "specialized_slig": visual_embeddings,
                "total": text_embeddings + visual_embeddings + understanding_embeddings
            },
            "completion_rates": {
                "text_embeddings": f"{(text_embeddings / chunks_count * 100) if chunks_count > 0 else 0:.1f}%",
                "image_analysis": f"{(visual_embeddings / images_count * 100) if images_count > 0 else 0:.1f}%"
            }
        }

        logger.info(f"✅ Document content fetched successfully: {chunks_count} chunks, {images_count} images, {products_count} products")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching document content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching document content: {str(e)}")


# ============================================================================
# AI Tracking Endpoints
# ============================================================================

@router.get("/job/{job_id}/ai-tracking")
async def get_job_ai_tracking(job_id: str):
    """
    Get detailed AI model tracking information for a job.

    Returns comprehensive metrics on:
    - Which AI models were used (QWEN, Anthropic, CLIP, OpenAI)
    - Confidence scores and results
    - Token usage and processing time
    - Success/failure rates
    - Per-stage breakdown
    """
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
        model_name: AI model name (QWEN, Anthropic, CLIP, OpenAI)

    Returns:
        Statistics for the specified AI model
    """
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

