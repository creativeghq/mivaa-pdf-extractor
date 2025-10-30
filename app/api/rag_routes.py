"""
RAG (Retrieval-Augmented Generation) API Routes

This module provides comprehensive FastAPI endpoints for RAG functionality including
document embedding, querying, chat interface, and document management.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Query, status, BackgroundTasks
from fastapi.responses import JSONResponse
import asyncio
try:
    # Try Pydantic v2 first
    from pydantic import BaseModel, Field, field_validator as validator
except ImportError:
    # Fall back to Pydantic v1
    from pydantic import BaseModel, Field, validator

from app.config import get_settings
from app.services.llamaindex_service import LlamaIndexService
from app.services.real_embeddings_service import RealEmbeddingsService
from app.services.advanced_search_service import QueryType, SearchOperator
from app.services.product_creation_service import ProductCreationService
from app.services.job_recovery_service import JobRecoveryService
from app.services.checkpoint_recovery_service import checkpoint_recovery_service, ProcessingStage
from app.services.supabase_client import get_supabase_client
from app.services.ai_model_tracker import AIModelTracker
from app.services.focused_product_extractor import get_focused_product_extractor
from app.utils.logging import PDFProcessingLogger

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/rag", tags=["RAG"])

# Job storage for async processing (in-memory cache)
job_storage: Dict[str, Dict[str, Any]] = {}

# Job recovery service (initialized on startup)
job_recovery_service: Optional[JobRecoveryService] = None


async def initialize_job_recovery():
    """
    Initialize job recovery service and mark any interrupted jobs.
    This should be called on application startup.
    """
    global job_recovery_service

    try:
        logger.info("🔄 Initializing job recovery service...")

        supabase_client = get_supabase_client()
        job_recovery_service = JobRecoveryService(supabase_client)

        # Mark all processing jobs as interrupted (they were interrupted by restart)
        interrupted_count = await job_recovery_service.mark_all_processing_as_interrupted(
            reason="Service restart detected"
        )

        if interrupted_count > 0:
            logger.warning(f"🛑 Marked {interrupted_count} jobs as interrupted due to service restart")

        # Get statistics
        stats = await job_recovery_service.get_job_statistics()
        logger.info(f"📊 Job statistics: {stats}")

        # Cleanup old jobs (older than 7 days)
        cleaned = await job_recovery_service.cleanup_old_jobs(days=7)
        if cleaned > 0:
            logger.info(f"🧹 Cleaned up {cleaned} old jobs")

        logger.info("✅ Job recovery service initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize job recovery service: {e}", exc_info=True)
        # Don't fail startup if job recovery fails
        job_recovery_service = None

# Pydantic models for request/response validation
class DocumentUploadRequest(BaseModel):
    """Request model for document upload and processing."""
    title: Optional[str] = Field(None, description="Document title")
    description: Optional[str] = Field(None, description="Document description")
    tags: Optional[List[str]] = Field(default_factory=list, description="Document tags")
    chunk_size: Optional[int] = Field(1000, ge=100, le=4000, description="Chunk size for processing")
    chunk_overlap: Optional[int] = Field(200, ge=0, le=1000, description="Chunk overlap")
    enable_embedding: bool = Field(True, description="Enable automatic embedding generation")

class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    status: str = Field(..., description="Processing status")
    chunks_created: int = Field(..., description="Number of chunks created")
    embeddings_generated: bool = Field(..., description="Whether embeddings were generated")
    processing_time: float = Field(..., description="Processing time in seconds")
    message: str = Field(..., description="Status message")

class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., min_length=1, max_length=2000, description="Query text")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of top results to retrieve")
    similarity_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")
    include_metadata: bool = Field(True, description="Include document metadata in response")
    enable_reranking: bool = Field(True, description="Enable result reranking")
    document_ids: Optional[List[str]] = Field(None, description="Filter by specific document IDs")

class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents and chunks")
    confidence_score: float = Field(..., description="Confidence score for the answer")
    processing_time: float = Field(..., description="Query processing time in seconds")
    retrieved_chunks: int = Field(..., description="Number of chunks retrieved")

class ChatRequest(BaseModel):
    """Request model for conversational RAG."""
    message: str = Field(..., min_length=1, max_length=2000, description="Chat message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")
    include_history: bool = Field(True, description="Include conversation history in context")
    document_ids: Optional[List[str]] = Field(None, description="Filter by specific document IDs")

class ChatResponse(BaseModel):
    """Response model for conversational RAG."""
    message: str = Field(..., description="Original message")
    response: str = Field(..., description="AI response")
    conversation_id: str = Field(..., description="Conversation ID")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents used")
    processing_time: float = Field(..., description="Response generation time")

class SearchRequest(BaseModel):
    """Request model for semantic search."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    search_type: str = Field("semantic", pattern="^(semantic|hybrid|keyword)$", description="Search type")
    top_k: Optional[int] = Field(10, ge=1, le=50, description="Number of results to return")
    similarity_threshold: Optional[float] = Field(0.6, ge=0.0, le=1.0, description="Similarity threshold")
    document_ids: Optional[List[str]] = Field(None, description="Filter by document IDs")
    include_content: bool = Field(True, description="Include chunk content in results")

class SearchResponse(BaseModel):
    """Response model for semantic search."""
    query: str = Field(..., description="Original search query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_type: str = Field(..., description="Type of search performed")
    processing_time: float = Field(..., description="Search processing time")

class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    documents: List[Dict[str, Any]] = Field(..., description="List of documents")
    total_count: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")

class HealthCheckResponse(BaseModel):
    """Response model for RAG health check."""
    status: str = Field(..., description="Health status")
    services: Dict[str, Dict[str, Any]] = Field(..., description="Service health details")
    timestamp: str = Field(..., description="Health check timestamp")

# Advanced Search Models for Phase 7 Features
class MMRSearchRequest(BaseModel):
    """Request model for MMR (Maximal Marginal Relevance) search."""
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: Optional[int] = Field(10, ge=1, le=50, description="Number of initial results to retrieve")
    diversity_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="MMR diversity threshold")
    lambda_param: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="MMR lambda parameter for relevance vs diversity balance")
    document_ids: Optional[List[str]] = Field(None, description="Filter by specific document IDs")
    include_metadata: bool = Field(True, description="Include document metadata in response")

class MMRSearchResponse(BaseModel):
    """Response model for MMR search."""
    query: str = Field(..., description="Original search query")
    results: List[Dict[str, Any]] = Field(..., description="MMR search results with diversity scores")
    total_results: int = Field(..., description="Total number of results")
    diversity_score: float = Field(..., description="Overall diversity score of results")
    processing_time: float = Field(..., description="Search processing time in seconds")

class AdvancedQueryRequest(BaseModel):
    """Request model for advanced query with optimization."""
    query: str = Field(..., min_length=1, max_length=2000, description="Query text")
    query_type: str = Field("semantic", pattern="^(factual|analytical|conversational|boolean|fuzzy|semantic)$", description="Type of query")
    top_k: Optional[int] = Field(10, ge=1, le=50, description="Number of results to retrieve")
    enable_expansion: bool = Field(True, description="Enable query expansion")
    enable_rewriting: bool = Field(True, description="Enable query rewriting")
    similarity_threshold: Optional[float] = Field(0.6, ge=0.0, le=1.0, description="Similarity threshold")
    document_ids: Optional[List[str]] = Field(None, description="Filter by specific document IDs")
    metadata_filters: Optional[Dict[str, Any]] = Field(None, description="Metadata-based filters")
    search_operator: str = Field("AND", pattern="^(AND|OR|NOT)$", description="Search operator for multiple terms")

class AdvancedQueryResponse(BaseModel):
    """Response model for advanced query."""
    original_query: str = Field(..., description="Original query text")
    optimized_query: str = Field(..., description="Optimized/expanded query")
    query_type: str = Field(..., description="Type of query processed")
    results: List[Dict[str, Any]] = Field(..., description="Search results with relevance scores")
    total_results: int = Field(..., description="Total number of results")
    expansion_terms: List[str] = Field(..., description="Terms added during query expansion")
    processing_time: float = Field(..., description="Query processing time in seconds")
    confidence_score: float = Field(..., description="Overall confidence score")

# Dependency functions
async def get_llamaindex_service() -> LlamaIndexService:
    """Get LlamaIndex service instance."""
    from app.main import app
    if not hasattr(app.state, 'llamaindex_service') or app.state.llamaindex_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LlamaIndex service is not available"
        )
    return app.state.llamaindex_service

async def get_embedding_service() -> RealEmbeddingsService:
    """Get embedding service instance."""
    try:
        supabase_client = get_supabase_client()
        return RealEmbeddingsService(supabase_client=supabase_client)
    except Exception as e:
        logger.error(f"Failed to initialize embedding service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service is not available"
        )

# API Endpoints

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # JSON string of tags
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    enable_embedding: bool = Form(True),
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    Upload and process a document for RAG functionality.
    
    This endpoint accepts document uploads, processes them into chunks,
    generates embeddings, and stores them in the vector database.
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith(('application/pdf', 'text/', 'application/msword')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file type. Please upload PDF, text, or Word documents."
            )
        
        # Parse tags if provided
        document_tags = []
        if tags:
            try:
                import json
                document_tags = json.loads(tags)
            except json.JSONDecodeError:
                document_tags = [tag.strip() for tag in tags.split(',')]
        
        # Generate document ID
        document_id = str(uuid4())
        
        # Read file content
        file_content = await file.read()
        
        # Process document through LlamaIndex service
        processing_result = await llamaindex_service.index_document_content(
            file_content=file_content,
            document_id=document_id,
            file_path=file.filename,
            metadata={
                "filename": file.filename,
                "title": title or file.filename,
                "description": description,
                "tags": document_tags,
                "source": "rag_upload"
            },
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Check if processing actually succeeded
        result_status = processing_result.get('status', 'completed')
        chunks_created = processing_result.get('statistics', {}).get('total_chunks', 0)

        # If status is error, raise an exception with details
        if result_status == 'error':
            error_message = processing_result.get('error', 'Unknown error during document processing')
            logger.error(f"Document processing failed for {document_id}: {error_message}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document processing failed: {error_message}"
            )

        return DocumentUploadResponse(
            document_id=document_id,
            title=title or file.filename,
            status=result_status,
            chunks_created=chunks_created,
            embeddings_generated=chunks_created > 0,  # Only true if chunks were actually created
            processing_time=processing_time,
            message=f"Document processed successfully: {chunks_created} chunks created"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )

@router.post("/documents/upload-async")
async def upload_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    enable_embedding: bool = Form(True),
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    Upload and process a document asynchronously for RAG functionality.

    Returns immediately with a job_id that can be used to check processing status.
    Use GET /documents/job/{job_id} to check the status and get results.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith(('application/pdf', 'text/', 'application/msword')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file type. Please upload PDF, text, or Word documents."
            )

        # Generate IDs
        job_id = str(uuid4())
        document_id = str(uuid4())

        # Parse tags if provided
        document_tags = []
        if tags:
            try:
                import json
                document_tags = json.loads(tags)
            except json.JSONDecodeError:
                document_tags = [tag.strip() for tag in tags.split(',')]

        # Read file content
        file_content = await file.read()

        # Create placeholder document record FIRST (to satisfy foreign key constraint)
        supabase_client = get_supabase_client()
        try:
            supabase_client.client.table('documents').insert({
                'id': document_id,
                'filename': file.filename,
                'content_type': file.content_type,
                'processing_status': 'processing',
                'metadata': {
                    'title': title or file.filename,
                    'description': description,
                    'catalog': catalog
                }
            }).execute()
            logger.info(f"✅ Created placeholder document record: {document_id}")
        except Exception as e:
            logger.error(f"Failed to create placeholder document: {e}")
            # Continue anyway - the background task will create it

        # Initialize job storage (in-memory)
        job_storage[job_id] = {
            "status": "pending",
            "document_id": document_id,
            "filename": file.filename,
            "created_at": datetime.utcnow().isoformat(),
            "progress": 0
        }

        # Persist job to database for recovery (now document exists)
        if job_recovery_service:
            await job_recovery_service.persist_job(
                job_id=job_id,
                document_id=document_id,
                filename=file.filename,
                status="pending",
                metadata={
                    "title": title,
                    "description": description,
                    "tags": document_tags,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                },
                progress=0
            )

        # Start background processing
        background_tasks.add_task(
            process_document_background,
            job_id=job_id,
            document_id=document_id,
            file_content=file_content,
            filename=file.filename,
            title=title,
            description=description,
            document_tags=document_tags,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            llamaindex_service=llamaindex_service
        )

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id": job_id,
                "document_id": document_id,
                "status": "pending",
                "message": "Document upload accepted. Processing in background.",
                "status_url": f"/api/rag/documents/job/{job_id}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document upload failed: {str(e)}"
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

        # ✅ CRITICAL FIX: Restart the job by calling the LlamaIndex service directly
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
            file_url = doc_data.get('file_url')
            filename = doc_data.get('filename', 'document.pdf')

            if not file_url:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Document {document_id} has no file_url"
                )

            # Download file from storage
            logger.info(f"📥 Downloading file from storage: {file_url}")
            bucket_name = file_url.split('/')[0] if '/' in file_url else 'pdf-documents'
            file_path = '/'.join(file_url.split('/')[1:]) if '/' in file_url else file_url

            file_response = supabase_client.client.storage.from_(bucket_name).download(file_path)
            if not file_response:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"File not found in storage: {file_url}"
                )

            file_content = file_response
            logger.info(f"✅ Downloaded file: {len(file_content)} bytes")

            # Trigger background processing with resume support
            background_tasks.add_task(
                process_document_background,
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

            logger.info(f"✅ Background task triggered for job {job_id}")

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
        query = supabase_client.table('background_jobs').select('*')

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

        return JSONResponse(content=jobs)

    except Exception as e:
        logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}"
        )


# REMOVED: Duplicate endpoint - use /api/admin/jobs/{job_id}/status instead
# This endpoint was conflicting with admin.py and never being reached due to router registration order


async def create_products_background(
    document_id: str,
    workspace_id: str,
    job_id: str
):
    """
    Background task to create products from chunks with checkpoint support.
    Runs separately to avoid blocking main PDF processing.
    Creates a sub-job for tracking.
    """
    supabase_client = get_supabase_client()
    sub_job_id = f"{job_id}_products"

    try:
        logger.info(f"🏭 Starting background product creation for document {document_id}")

        # Create sub-job in database
        try:
            supabase_client.client.table('background_jobs').insert({
                "id": sub_job_id,
                "parent_job_id": job_id,
                "job_type": "product_creation",
                "document_id": document_id,
                "status": "processing",
                "progress": 0,
                "metadata": {
                    "workspace_id": workspace_id,
                    "started_at": datetime.utcnow().isoformat()
                },
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).execute()
            logger.info(f"✅ Created sub-job {sub_job_id} for product creation")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create sub-job: {e}")

        product_service = ProductCreationService(supabase_client)

        # Create PRODUCTS_DETECTED checkpoint before detection
        await checkpoint_recovery_service.create_checkpoint(
            job_id=job_id,
            stage=ProcessingStage.PRODUCTS_DETECTED,
            data={
                "document_id": document_id,
                "workspace_id": workspace_id,
                "detection_started": True
            },
            metadata={
                "current_step": "Detecting product candidates",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        logger.info(f"✅ Created PRODUCTS_DETECTED checkpoint for job {job_id}")

        # Use layout-based product detection
        product_result = await product_service.create_products_from_layout_candidates(
            document_id=document_id,
            workspace_id=workspace_id,
            min_confidence=0.5,
            min_quality_score=0.5
        )

        products_created = product_result.get('products_created', 0)
        logger.info(f"✅ Background product creation completed: {products_created} products created")

        # Update sub-job status to completed
        try:
            supabase_client.client.table('background_jobs').update({
                "status": "completed",
                "progress": 100,
                "metadata": {
                    "workspace_id": workspace_id,
                    "products_created": products_created,
                    "candidates_detected": product_result.get('candidates_detected', 0),
                    "validation_passed": product_result.get('validation_passed', 0),
                    "completed_at": datetime.utcnow().isoformat()
                },
                "updated_at": datetime.utcnow().isoformat()
            }).eq('id', sub_job_id).execute()
            logger.info(f"✅ Marked sub-job {sub_job_id} as completed")
        except Exception as e:
            logger.warning(f"⚠️ Failed to update sub-job: {e}")

        # Create PRODUCTS_CREATED checkpoint after successful creation
        await checkpoint_recovery_service.create_checkpoint(
            job_id=job_id,
            stage=ProcessingStage.PRODUCTS_CREATED,
            data={
                "document_id": document_id,
                "products_created": products_created,
                "product_ids": product_result.get('product_ids', [])
            },
            metadata={
                "current_step": "Products created successfully",
                "candidates_detected": product_result.get('candidates_detected', 0),
                "validation_passed": product_result.get('validation_passed', 0),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        logger.info(f"✅ Created PRODUCTS_CREATED checkpoint for job {job_id}")

        # Update job metadata with product count (in-memory)
        if job_id in job_storage:
            job_storage[job_id]["progress"] = 100  # Now fully complete
            if "result" in job_storage[job_id]:
                job_storage[job_id]["result"]["products_created"] = products_created
                job_storage[job_id]["result"]["message"] = f"Document processed successfully: {products_created} products created"

        # ✅ FIX: Persist products_created to database background_jobs table
        try:
            job_recovery_service = JobRecoveryService(supabase_client)

            # Get current job from database
            current_job = await job_recovery_service.get_job(job_id)
            if current_job:
                # Update metadata with products_created
                updated_metadata = current_job.get('metadata', {})
                updated_metadata['products_created'] = products_created

                # Persist updated metadata
                await job_recovery_service.persist_job(
                    job_id=job_id,
                    document_id=document_id,
                    filename=current_job.get('filename', 'unknown'),
                    status='completed',
                    progress=100,
                    metadata=updated_metadata
                )
                logger.info(f"✅ Persisted products_created={products_created} to database for job {job_id}")
        except Exception as persist_error:
            logger.error(f"❌ Failed to persist products_created to database: {persist_error}")

    except Exception as e:
        logger.error(f"❌ Background product creation failed: {e}", exc_info=True)

        # Mark sub-job as failed
        try:
            supabase_client.client.table('background_jobs').update({
                "status": "failed",
                "error": str(e),
                "metadata": {
                    "workspace_id": workspace_id,
                    "error_message": str(e),
                    "failed_at": datetime.utcnow().isoformat()
                },
                "updated_at": datetime.utcnow().isoformat()
            }).eq('id', sub_job_id).execute()
            logger.info(f"✅ Marked sub-job {sub_job_id} as failed")
        except Exception as sub_error:
            logger.warning(f"⚠️ Failed to update sub-job: {sub_error}")

        # Create failed checkpoint
        try:
            await checkpoint_recovery_service.create_checkpoint(
                job_id=job_id,
                stage=ProcessingStage.PRODUCTS_DETECTED,
                data={
                    "document_id": document_id,
                    "error": str(e),
                    "failed": True
                },
                metadata={
                    "current_step": "Product creation failed",
                    "error_message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        except:
            pass  # Don't fail if checkpoint creation fails


async def process_images_background(
    document_id: str,
    job_id: str,
    images_count: int
):
    """
    Background task to process image AI analysis with checkpoint support.
    Runs separately to avoid blocking main PDF processing.
    Creates a sub-job for tracking.
    """
    supabase_client = get_supabase_client()
    sub_job_id = f"{job_id}_images"

    try:
        logger.info(f"🖼️ Starting background image AI analysis for document {document_id}")

        # Create sub-job in database
        try:
            supabase_client.client.table('background_jobs').insert({
                "id": sub_job_id,
                "parent_job_id": job_id,
                "job_type": "image_analysis",
                "document_id": document_id,
                "status": "processing",
                "progress": 0,
                "metadata": {
                    "images_count": images_count,
                    "started_at": datetime.utcnow().isoformat()
                },
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).execute()
            logger.info(f"✅ Created sub-job {sub_job_id} for image analysis")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create sub-job: {e}")

        # Run background image processing
        from app.services.background_image_processor import start_background_image_processing
        result = await start_background_image_processing(
            document_id=document_id,
            supabase_client=supabase_client
        )

        images_processed = result.get('total_processed', 0)
        images_failed = result.get('total_failed', 0)
        logger.info(f"✅ Background image analysis completed: {images_processed} processed, {images_failed} failed")

        # Update sub-job status to completed
        try:
            supabase_client.client.table('background_jobs').update({
                "status": "completed",
                "progress": 100,
                "metadata": {
                    "images_count": images_count,
                    "images_processed": images_processed,
                    "images_failed": images_failed,
                    "batches_processed": result.get('batches_processed', 0),
                    "completed_at": datetime.utcnow().isoformat()
                },
                "updated_at": datetime.utcnow().isoformat()
            }).eq('id', sub_job_id).execute()
            logger.info(f"✅ Marked sub-job {sub_job_id} as completed")
        except Exception as e:
            logger.warning(f"⚠️ Failed to update sub-job: {e}")

        # Create IMAGE_ANALYSIS_COMPLETED checkpoint
        try:
            await checkpoint_recovery_service.create_checkpoint(
                job_id=job_id,
                stage=ProcessingStage.COMPLETED,  # Use COMPLETED since this is the final stage
                data={
                    "document_id": document_id,
                    "images_processed": images_processed,
                    "images_failed": images_failed
                },
                metadata={
                    "current_step": "Image AI analysis completed",
                    "images_count": images_count,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            logger.info(f"✅ Created IMAGE_ANALYSIS_COMPLETED checkpoint for job {job_id}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create checkpoint: {e}")

    except Exception as e:
        logger.error(f"❌ Background image analysis failed: {e}", exc_info=True)

        # Mark sub-job as failed
        try:
            supabase_client.client.table('background_jobs').update({
                "status": "failed",
                "error": str(e),
                "metadata": {
                    "images_count": images_count,
                    "error_message": str(e),
                    "failed_at": datetime.utcnow().isoformat()
                },
                "updated_at": datetime.utcnow().isoformat()
            }).eq('id', sub_job_id).execute()
            logger.info(f"✅ Marked sub-job {sub_job_id} as failed")
        except Exception as sub_error:
            logger.warning(f"⚠️ Failed to update sub-job: {sub_error}")


async def process_document_background(
    job_id: str,
    document_id: str,
    file_content: bytes,
    filename: str,
    title: Optional[str],
    description: Optional[str],
    document_tags: List[str],
    chunk_size: int,
    chunk_overlap: int,
    llamaindex_service: Optional[LlamaIndexService] = None
):
    """
    Background task to process document with checkpoint recovery support.
    """
    start_time = datetime.utcnow()

    logger.info("=" * 80)
    logger.info(f"🚀 [BACKGROUND TASK] STARTING PROCESS_DOCUMENT_BACKGROUND")
    logger.info("=" * 80)
    logger.info(f"📋 Job ID: {job_id}")
    logger.info(f"📄 Document ID: {document_id}")
    logger.info(f"📝 Filename: {filename}")
    logger.info(f"📦 File size: {len(file_content)} bytes")
    logger.info(f"⏰ Started at: {start_time.isoformat()}")
    logger.info(f"🔧 Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    logger.info(f"📚 Title: {title}")
    logger.info(f"📝 Description: {description}")
    logger.info(f"🏷️  Tags: {document_tags}")
    logger.info("=" * 80)

    # Get LlamaIndex service from app state if not provided
    logger.info("🔧 [STEP 1] Getting LlamaIndex service...")
    if llamaindex_service is None:
        try:
            from app.main import app
            if hasattr(app.state, 'llamaindex_service'):
                llamaindex_service = app.state.llamaindex_service
                logger.info("✅ [STEP 1] Retrieved LlamaIndex service from app state")
            else:
                logger.error("❌ [STEP 1] LlamaIndex service not available in app state")
                job_storage[job_id]["status"] = "failed"
                job_storage[job_id]["error"] = "LlamaIndex service not available"
                return
        except Exception as e:
            logger.error(f"❌ [STEP 1] Failed to get LlamaIndex service: {e}")
            job_storage[job_id]["status"] = "failed"
            job_storage[job_id]["error"] = str(e)
            return
    else:
        logger.info("✅ [STEP 1] LlamaIndex service provided as parameter")

    # Check for existing checkpoint to resume from
    logger.info("🔍 [STEP 2] Checking for existing checkpoints...")
    last_checkpoint = None
    resume_from_stage = None
    try:
        last_checkpoint = await checkpoint_recovery_service.get_last_checkpoint(job_id)
        if last_checkpoint:
            resume_from_stage_str = last_checkpoint.get('stage')
            resume_from_stage = ProcessingStage(resume_from_stage_str)
            logger.info(f"🔄 [STEP 2] RESUMING FROM CHECKPOINT: {resume_from_stage.value}")
            logger.info(f"   📅 Checkpoint created at: {last_checkpoint.get('created_at')}")
            logger.info(f"   📦 Checkpoint data keys: {list(last_checkpoint.get('checkpoint_data', {}).keys())}")

            # Verify checkpoint data exists in database
            logger.info(f"🔍 [STEP 2] Verifying checkpoint data for stage {resume_from_stage.value}...")
            can_resume = await checkpoint_recovery_service.verify_checkpoint_data(job_id, resume_from_stage)
            if not can_resume:
                logger.warning(f"⚠️ [STEP 2] Checkpoint data verification failed, starting from scratch")
                resume_from_stage = None
                last_checkpoint = None
            else:
                logger.info(f"✅ [STEP 2] Checkpoint data verified successfully")
        else:
            logger.info("📝 [STEP 2] No checkpoint found, starting fresh processing")
    except Exception as e:
        logger.error(f"❌ [STEP 2] Failed to check for checkpoint: {e}", exc_info=True)
        resume_from_stage = None

    try:
        # Create placeholder document record FIRST (required for foreign key constraint)
        supabase_client = get_supabase_client()

        # Skip document creation if resuming from checkpoint
        if not resume_from_stage:
            try:
                supabase_client.client.table('documents').insert({
                    "id": document_id,
                    "workspace_id": "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
                    "filename": filename,
                    "content_type": "application/pdf",
                    "file_size": len(file_content),
                    "processing_status": "processing",
                    "metadata": {
                        "title": title or filename,
                        "description": description,
                        "tags": document_tags,
                        "source": "rag_upload_async"
                    },
                    "created_at": start_time.isoformat(),
                    "updated_at": start_time.isoformat()
                }).execute()
                logger.info(f"✅ Created placeholder document record: {document_id}")

                # Also create processed_documents record (required for ai_analysis_queue foreign key)
                try:
                    supabase_client.client.table('processed_documents').insert({
                        "id": document_id,  # Use same ID as documents table
                        "workspace_id": "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
                        "pdf_document_id": document_id,
                        "content": "",  # Will be populated during processing
                        "processing_status": "processing",
                        "processing_started_at": start_time.isoformat(),
                        "metadata": {},
                        "created_at": start_time.isoformat(),
                        "updated_at": start_time.isoformat()
                    }).execute()
                    logger.info(f"✅ Created processed_documents record: {document_id}")
                except Exception as proc_doc_error:
                    logger.error(f"Failed to create processed_documents record: {proc_doc_error}")
                    # Continue anyway

            except Exception as doc_error:
                logger.error(f"Failed to create document record: {doc_error}")
                # Continue anyway - the document might already exist

            # Create INITIALIZED checkpoint
            await checkpoint_recovery_service.create_checkpoint(
                job_id=job_id,
                stage=ProcessingStage.INITIALIZED,
                data={
                    "document_id": document_id,
                    "filename": filename,
                    "file_size": len(file_content)
                },
                metadata={
                    "title": title or filename,
                    "description": description,
                    "tags": document_tags
                }
            )
            logger.info(f"✅ Created INITIALIZED checkpoint for job {job_id}")

        # Update status (in-memory)
        job_storage[job_id]["status"] = "processing"
        job_storage[job_id]["started_at"] = start_time.isoformat()
        job_storage[job_id]["document_id"] = document_id
        job_storage[job_id]["progress"] = 10

        # Persist status change to database (now document exists)
        if job_recovery_service:
            await job_recovery_service.persist_job(
                job_id=job_id,
                document_id=document_id,
                filename=filename,
                status="processing",
                progress=10
            )

        # Initialize AI model tracker for this job
        ai_tracker = AIModelTracker(job_id)
        job_storage[job_id]["ai_tracker"] = ai_tracker

        # Define progress callback to update job progress with detailed metadata
        async def update_progress(progress: int, details: dict = None):
            """Update job progress in memory and database with detailed stats and AI tracking"""
            job_storage[job_id]["progress"] = progress

            # Build detailed metadata
            detailed_metadata = {
                "document_id": document_id,  # ✅ FIX 1: Add document_id
                "filename": filename,
                "workspace_id": "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
                "title": title or filename,
                "description": description,
                "tags": document_tags,
                "source": "rag_upload_async"
            }

            # ✅ FIX 2: Add detailed progress stats
            if details:
                detailed_metadata.update({
                    "current_page": details.get("current_page"),
                    "total_pages": details.get("total_pages"),
                    "chunks_created": details.get("chunks_created", 0),
                    "images_extracted": details.get("images_extracted", 0),
                    "products_created": details.get("products_created", 0),
                    "ai_usage": {
                        "llama_calls": details.get("llama_calls", 0),
                        "claude_calls": details.get("claude_calls", 0),
                        "openai_calls": details.get("openai_calls", 0),
                        "clip_embeddings": details.get("clip_embeddings", 0)
                    },
                    "embeddings_generated": {
                        "text": details.get("text_embeddings", 0),
                        "visual": details.get("visual_embeddings", 0),
                        "color": details.get("color_embeddings", 0),
                        "texture": details.get("texture_embeddings", 0),
                        "application": details.get("application_embeddings", 0)
                    },
                    "current_step": details.get("current_step", "Processing")
                })

                # ✅ Log AI model calls if provided
                if details.get("ai_model_call"):
                    ai_call = details["ai_model_call"]
                    ai_tracker.log_model_call(
                        model_name=ai_call.get("model_name", "Unknown"),
                        stage=ai_call.get("stage", "unknown"),
                        task=ai_call.get("task", "unknown"),
                        latency_ms=ai_call.get("latency_ms", 0),
                        confidence_score=ai_call.get("confidence_score"),
                        result_summary=ai_call.get("result_summary"),
                        items_processed=ai_call.get("items_processed", 0),
                        input_tokens=ai_call.get("input_tokens"),
                        output_tokens=ai_call.get("output_tokens"),
                        success=ai_call.get("success", True),
                        error=ai_call.get("error")
                    )

            # Add AI tracker summary to metadata
            detailed_metadata["ai_tracking"] = ai_tracker.format_for_metadata()

            # ✅ FIX: Store metadata in job_storage so it's returned by get_job_status
            job_storage[job_id]["metadata"] = detailed_metadata

            if job_recovery_service:
                await job_recovery_service.persist_job(
                    job_id=job_id,
                    document_id=document_id,
                    filename=filename,
                    status="processing",
                    progress=progress,
                    metadata=detailed_metadata
                )
            logger.info(f"📊 Job {job_id} progress: {progress}% - {detailed_metadata.get('current_step', 'Processing')}")



        # Process document through LlamaIndex service with progress tracking and granular checkpoints
        # Skip if resuming from a later checkpoint
        if not resume_from_stage or resume_from_stage == ProcessingStage.INITIALIZED:
            # Enhanced progress callback that creates checkpoints at each stage
            async def enhanced_progress_callback(progress: int, details: dict = None):
                """Enhanced progress callback with checkpoint creation"""
                await update_progress(progress, details)

                # Create checkpoints at specific progress milestones
                current_step = details.get('current_step', '') if details else ''

                # PDF_EXTRACTED checkpoint (after PDF extraction - 20%)
                if progress == 20 and 'Extracting text and images' in current_step:
                    await checkpoint_recovery_service.create_checkpoint(
                        job_id=job_id,
                        stage=ProcessingStage.PDF_EXTRACTED,
                        data={
                            "document_id": document_id,
                            "total_pages": details.get('total_pages', 0) if details else 0
                        },
                        metadata={
                            "current_step": current_step,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    logger.info(f"✅ Created PDF_EXTRACTED checkpoint for job {job_id}")

                # CHUNKS_CREATED checkpoint (after chunking - 40%)
                elif progress == 40 and 'Creating semantic chunks' in current_step:
                    await checkpoint_recovery_service.create_checkpoint(
                        job_id=job_id,
                        stage=ProcessingStage.CHUNKS_CREATED,
                        data={
                            "document_id": document_id,
                            "total_pages": details.get('total_pages', 0) if details else 0,
                            "images_extracted": details.get('images_extracted', 0) if details else 0
                        },
                        metadata={
                            "current_step": current_step,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    logger.info(f"✅ Created CHUNKS_CREATED checkpoint for job {job_id}")

                # TEXT_EMBEDDINGS_GENERATED checkpoint (after embeddings - 60%)
                elif progress == 60 and 'Generating embeddings' in current_step:
                    await checkpoint_recovery_service.create_checkpoint(
                        job_id=job_id,
                        stage=ProcessingStage.TEXT_EMBEDDINGS_GENERATED,
                        data={
                            "document_id": document_id,
                            "chunks_created": details.get('chunks_created', 0) if details else 0,
                            "total_pages": details.get('total_pages', 0) if details else 0,
                            "images_extracted": details.get('images_extracted', 0) if details else 0
                        },
                        metadata={
                            "current_step": current_step,
                            "text_embeddings": details.get('text_embeddings', 0) if details else 0,
                            "openai_calls": details.get('openai_calls', 0) if details else 0,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    logger.info(f"✅ Created TEXT_EMBEDDINGS_GENERATED checkpoint for job {job_id}")

                # IMAGES_EXTRACTED checkpoint (after image processing - 80%)
                elif progress == 80 and 'Processing images' in current_step:
                    await checkpoint_recovery_service.create_checkpoint(
                        job_id=job_id,
                        stage=ProcessingStage.IMAGES_EXTRACTED,
                        data={
                            "document_id": document_id,
                            "chunks_created": details.get('chunks_created', 0) if details else 0,
                            "images_extracted": details.get('images_extracted', 0) if details else 0
                        },
                        metadata={
                            "current_step": current_step,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    logger.info(f"✅ Created IMAGES_EXTRACTED checkpoint for job {job_id}")

            logger.info("=" * 80)
            logger.info("🚀 [STEP 4] CALLING LLAMAINDEX SERVICE index_document_content")
            logger.info("=" * 80)
            logger.info(f"📄 File content size: {len(file_content)} bytes")
            logger.info(f"📝 Document ID: {document_id}")
            logger.info(f"📂 Filename: {filename}")
            logger.info(f"🔧 Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
            logger.info("=" * 80)

            processing_result = await llamaindex_service.index_document_content(
                file_content=file_content,
                document_id=document_id,
                file_path=filename,
                metadata={
                    "workspace_id": "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",  # Default workspace UUID
                    "filename": filename,
                    "title": title or filename,
                    "description": description,
                    "tags": document_tags,
                    "source": "rag_upload_async"
                },
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                progress_callback=enhanced_progress_callback
            )

            logger.info("=" * 80)
            logger.info("✅ [STEP 4] LLAMAINDEX SERVICE COMPLETED")
            logger.info("=" * 80)

            # Create IMAGE_EMBEDDINGS_GENERATED checkpoint after CLIP embeddings
            chunks_created = processing_result.get('statistics', {}).get('total_chunks', 0)
            images_extracted = processing_result.get('statistics', {}).get('images_extracted', 0)
            clip_embeddings = processing_result.get('statistics', {}).get('clip_embeddings_generated', 0)

            if clip_embeddings > 0:
                await checkpoint_recovery_service.create_checkpoint(
                    job_id=job_id,
                    stage=ProcessingStage.IMAGE_EMBEDDINGS_GENERATED,
                    data={
                        "document_id": document_id,
                        "chunks_created": chunks_created,
                        "images_extracted": images_extracted,
                        "clip_embeddings": clip_embeddings
                    },
                    metadata={
                        "current_step": "CLIP embeddings generated",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                logger.info(f"✅ Created IMAGE_EMBEDDINGS_GENERATED checkpoint for job {job_id}")

            # Log the processing result status for debugging
            result_status = processing_result.get('status', 'unknown')
            logger.info(f"📊 Processing result status: {result_status}")
            logger.info(f"📊 Processing result keys: {list(processing_result.keys())}")
            logger.info(f"📊 Statistics: {processing_result.get('statistics', {})}")

            if result_status == 'success':
                # Create COMPLETED checkpoint with all processing data
                await checkpoint_recovery_service.create_checkpoint(
                    job_id=job_id,
                    stage=ProcessingStage.COMPLETED,
                    data={
                        "document_id": document_id,
                        "chunks_created": chunks_created,
                        "images_extracted": images_extracted,
                        "embeddings_generated": processing_result.get('statistics', {}).get('database_embeddings_stored', 0),
                        "clip_embeddings": clip_embeddings
                    },
                    metadata={
                        "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                        "statistics": processing_result.get('statistics', {})
                    }
                )
                logger.info(f"✅ Created COMPLETED checkpoint for job {job_id}")
            else:
                # Log why we're not creating COMPLETED checkpoint
                logger.error(f"❌ Processing did NOT return success status!")
                logger.error(f"   Status: {result_status}")
                logger.error(f"   Full result: {processing_result}")
                # Raise an exception so it gets caught and logged properly
                raise RuntimeError(f"Document processing returned status '{result_status}' instead of 'success'")
        else:
            # Resuming from checkpoint - retrieve data from last checkpoint
            logger.info(f"⏭️ Skipping document processing - resuming from {resume_from_stage}")
            checkpoint_data = last_checkpoint.get('checkpoint_data', {})
            processing_result = {
                'status': 'success',
                'statistics': {
                    'total_chunks': checkpoint_data.get('chunks_created', 0),
                    'images_extracted': checkpoint_data.get('images_extracted', 0),
                    'database_embeddings_stored': checkpoint_data.get('embeddings_generated', 0),
                    'clip_embeddings_generated': checkpoint_data.get('clip_embeddings', 0)
                }
            }

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Check if processing succeeded
        result_status = processing_result.get('status', 'completed')
        chunks_created = processing_result.get('statistics', {}).get('total_chunks', 0)

        if result_status == 'error':
            error_message = processing_result.get('error', 'Unknown error during document processing')
            job_storage[job_id]["status"] = "failed"
            job_storage[job_id]["error"] = error_message
            job_storage[job_id]["progress"] = 100
        else:
            # ✅ FIX 3: Run product creation in background to prevent timeout
            # Mark main job as 90% complete, product creation runs separately
            products_created = 0

            if chunks_created > 0:
                logger.info(f"🏭 Scheduling background product creation for document {document_id}")
                # Start product creation in background (don't await)
                # Note: asyncio is already imported at the top of the file
                asyncio.create_task(create_products_background(
                    document_id=document_id,
                    workspace_id="ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
                    job_id=job_id
                ))
                logger.info("✅ Product creation scheduled in background")
            else:
                logger.info("⚠️ No chunks created, skipping product creation")

            # ✅ FIX 4: Start background image processing with sub-job tracking
            images_extracted = processing_result.get('statistics', {}).get('total_images', 0)
            if images_extracted > 0:
                logger.info(f"🖼️ Scheduling background image AI analysis for {images_extracted} images")
                asyncio.create_task(process_images_background(
                    document_id=document_id,
                    job_id=job_id,
                    images_count=images_extracted
                ))
                logger.info("✅ Image AI analysis scheduled in background")
            else:
                logger.info("⚠️ No images extracted, skipping background image analysis")

            job_storage[job_id]["status"] = "completed"
            job_storage[job_id]["progress"] = 90  # Main processing complete, products/images running in background
            job_storage[job_id]["completed_at"] = datetime.utcnow().isoformat()
            job_storage[job_id]["result"] = {
                "document_id": document_id,
                "title": title or filename,
                "status": result_status,
                "chunks_created": chunks_created,
                "embeddings_generated": chunks_created > 0,
                "products_created": products_created,
                "processing_time": processing_time,
                "message": f"Document processed successfully: {chunks_created} chunks created, {products_created} products created"
            }

            # Persist completion to database
            if job_recovery_service:
                await job_recovery_service.persist_job(
                    job_id=job_id,
                    document_id=document_id,
                    filename=filename,
                    status="completed",
                    progress=100,
                    metadata={
                        "chunks_created": chunks_created,
                        "products_created": products_created,
                        "processing_time": processing_time
                    }
                )

    except asyncio.CancelledError:
        # Job was cancelled (likely due to service shutdown)
        logger.error(f"🛑 JOB INTERRUPTED: {job_id}")
        logger.error(f"   Document ID: {document_id}")
        logger.error(f"   Filename: {filename}")
        logger.error(f"   Reason: Service shutdown or task cancellation")
        logger.error(f"   Duration before interruption: {(datetime.utcnow() - start_time).total_seconds():.2f}s")

        job_storage[job_id]["status"] = "interrupted"
        job_storage[job_id]["error"] = "Job interrupted due to service shutdown"
        job_storage[job_id]["progress"] = job_storage[job_id].get("progress", 0)
        job_storage[job_id]["interrupted_at"] = datetime.utcnow().isoformat()

        # Persist interruption to database
        if job_recovery_service:
            await job_recovery_service.mark_job_interrupted(
                job_id=job_id,
                reason="Service shutdown or task cancellation"
            )

        # Re-raise to allow proper cleanup
        raise

    except Exception as e:
        logger.error(f"❌ BACKGROUND JOB FAILED: {job_id}")
        logger.error(f"   Document ID: {document_id}")
        logger.error(f"   Error: {e}", exc_info=True)
        logger.error(f"   Duration before failure: {(datetime.utcnow() - start_time).total_seconds():.2f}s")

        job_storage[job_id]["status"] = "failed"
        job_storage[job_id]["error"] = str(e)
        job_storage[job_id]["progress"] = 100
        job_storage[job_id]["failed_at"] = datetime.utcnow().isoformat()

        # Persist failure to database
        if job_recovery_service:
            await job_recovery_service.persist_job(
                job_id=job_id,
                document_id=document_id,
                filename=filename,
                status="failed",
                progress=100,
                error=str(e)
            )

    finally:
        end_time = datetime.utcnow()
        total_duration = (end_time - start_time).total_seconds()
        final_status = job_storage[job_id].get("status", "unknown")

        logger.info(f"📋 BACKGROUND JOB FINISHED: {job_id}")
        logger.info(f"   Final status: {final_status}")
        logger.info(f"   Total duration: {total_duration:.2f}s")
        logger.info(f"   Ended at: {end_time.isoformat()}")


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    Query documents using RAG (Retrieval-Augmented Generation).
    
    This endpoint performs semantic search over the document collection
    and generates contextual answers using the retrieved information.
    """
    start_time = datetime.utcnow()
    
    try:
        # Perform RAG query using advanced_rag_query
        result = await llamaindex_service.advanced_rag_query(
            query=request.query,
            max_results=request.top_k,
            similarity_threshold=request.similarity_threshold,
            enable_reranking=request.enable_reranking,
            query_type="factual"
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return QueryResponse(
            query=request.query,
            answer=result.get('answer', ''),
            sources=result.get('sources', []),
            confidence_score=result.get('confidence_score', 0.0),
            processing_time=processing_time,
            retrieved_chunks=len(result.get('sources', []))
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )

@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(
    request: ChatRequest,
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    Conversational interface for document Q&A.
    
    This endpoint maintains conversation context and provides
    contextual responses based on the document collection.
    """
    start_time = datetime.utcnow()
    
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid4())
        
        # Process chat message using advanced_rag_query
        result = await llamaindex_service.advanced_rag_query(
            query=request.message,
            max_results=request.top_k,
            query_type="conversational"
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return ChatResponse(
            message=request.message,
            response=result.get('response', ''),
            conversation_id=conversation_id,
            sources=result.get('sources', []),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Chat processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}"
        )

@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    Semantic search across document collection.
    
    This endpoint provides various search capabilities including
    semantic, hybrid, and keyword search.
    """
    start_time = datetime.utcnow()
    
    try:
        # Perform search using semantic_search_with_mmr
        results = await llamaindex_service.semantic_search_with_mmr(
            query=request.query,
            k=request.top_k,
            lambda_mult=0.5  # Default MMR parameter
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return SearchResponse(
            query=request.query,
            results=results.get('results', []),
            total_results=results.get('total_results', 0),
            search_type=request.search_type,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Search processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search processing failed: {str(e)}"
        )

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
    - All images with AI analysis (CLIP, Llama, Claude)
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

            # Get embeddings for each chunk
            for chunk in chunks:
                embeddings_response = supabase_client.client.table('embeddings').select('*').eq('chunk_id', chunk['id']).execute()
                chunk['embeddings'] = embeddings_response.data or []

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
        clip_embeddings = sum(1 for img in result['images'] if img.get('visual_clip_embedding_512'))
        llama_analysis = sum(1 for img in result['images'] if img.get('llama_analysis'))
        claude_validation = sum(1 for img in result['images'] if img.get('claude_validation'))
        color_embeddings = sum(1 for img in result['images'] if img.get('color_embedding_256'))
        texture_embeddings = sum(1 for img in result['images'] if img.get('texture_embedding_256'))
        application_embeddings = sum(1 for img in result['images'] if img.get('application_embedding_512'))

        result['statistics'] = {
            "chunks_count": chunks_count,
            "images_count": images_count,
            "products_count": products_count,
            "ai_usage": {
                "openai_calls": text_embeddings,
                "llama_calls": llama_analysis,
                "claude_calls": claude_validation,
                "clip_embeddings": clip_embeddings
            },
            "embeddings_generated": {
                "text": text_embeddings,
                "visual": clip_embeddings,
                "color": color_embeddings,
                "texture": texture_embeddings,
                "application": application_embeddings,
                "total": text_embeddings + clip_embeddings + color_embeddings + texture_embeddings + application_embeddings
            },
            "completion_rates": {
                "text_embeddings": f"{(text_embeddings / chunks_count * 100) if chunks_count > 0 else 0:.1f}%",
                "image_analysis": f"{(clip_embeddings / images_count * 100) if images_count > 0 else 0:.1f}%"
            }
        }

        logger.info(f"✅ Document content fetched successfully: {chunks_count} chunks, {images_count} images, {products_count} products")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching document content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching document content: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    search: Optional[str] = Query(None, description="Search term for filtering"),
    tags: Optional[str] = Query(None, description="Comma-separated tags for filtering"),
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    List and filter documents in the collection.

    This endpoint provides paginated access to the document collection
    with optional filtering by search terms and tags.
    """
    try:
        # Parse tags filter
        tag_filter = []
        if tags:
            tag_filter = [tag.strip() for tag in tags.split(',')]
        
        # Get documents using list_indexed_documents
        result = await llamaindex_service.list_indexed_documents()
        
        return DocumentListResponse(
            documents=result.get('documents', []),
            total_count=result.get('total_count', 0),
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document listing failed: {str(e)}"
        )

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    Delete a document and its associated embeddings.
    
    This endpoint removes a document from the collection and
    cleans up all associated data including embeddings and chunks.
    """
    try:
        # Delete document
        result = await llamaindex_service.delete_document(document_id)
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Document deleted successfully",
                "document_id": document_id,
                "chunks_deleted": result.get('chunks_deleted', 0)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document deletion failed: {str(e)}"
        )

@router.get("/health", response_model=HealthCheckResponse)
async def rag_health_check(
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    Health check for RAG services.
    
    This endpoint checks the health of all RAG-related services
    including LlamaIndex, embedding service, and vector store.
    """
    try:
        # Check LlamaIndex service health
        llamaindex_health = await llamaindex_service.health_check()

        # Try to check embedding service health (optional)
        embedding_health = {"status": "unknown", "message": "Embedding service not available"}
        try:
            embedding_service = await get_embedding_service()
            embedding_health = await embedding_service.health_check()
        except Exception as e:
            logger.warning(f"Embedding service health check failed: {e}")
            embedding_health = {"status": "error", "error": str(e)}

        # Determine overall status
        overall_status = "healthy"
        if llamaindex_health.get("status") != "healthy":
            overall_status = "degraded"

        return HealthCheckResponse(
            status=overall_status,
            services={
                "llamaindex": llamaindex_health,
                "embedding": embedding_health
            },
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"RAG health check failed: {e}", exc_info=True)
        return HealthCheckResponse(
            status="unhealthy",
            services={
                "llamaindex": {"status": "error", "error": str(e)},
                "embedding": {"status": "unknown"}
            },
            timestamp=datetime.utcnow().isoformat()
        )

@router.get("/stats")
async def get_rag_statistics(
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    Get RAG system statistics.
    
    This endpoint provides statistics about the RAG system including
    document counts, embedding statistics, and performance metrics.
    """
    try:
        # Get available statistics from the service
        memory_stats = llamaindex_service.get_memory_stats()
        health_check = await llamaindex_service.health_check()

        # Combine statistics
        stats = {
            "memory": memory_stats,
            "health": health_check,
            "indices_count": len(llamaindex_service.indices),
            "storage_dir": llamaindex_service.storage_dir,
            "embedding_model": llamaindex_service.embedding_model,
            "llm_model": llamaindex_service.llm_model
        }

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "statistics": stats,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Statistics retrieval failed: {str(e)}"
        )

# Advanced Search Endpoints for Phase 7 Features

@router.post("/search/mmr", response_model=MMRSearchResponse)
async def mmr_search(
    request: MMRSearchRequest,
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    Perform MMR (Maximal Marginal Relevance) search for diverse results.
    
    This endpoint implements MMR search to provide diverse, non-redundant results
    by balancing relevance and diversity using the lambda parameter.
    """
    try:
        start_time = datetime.utcnow()
        
        # Call the MMR search method from LlamaIndex service
        results = await llamaindex_service.semantic_search_with_mmr(
            query=request.query,
            top_k=request.top_k,
            diversity_threshold=request.diversity_threshold,
            lambda_param=request.lambda_param,
            document_ids=request.document_ids,
            include_metadata=request.include_metadata
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return MMRSearchResponse(
            query=request.query,
            results=results.get('results', []),
            total_results=results.get('total_results', 0),
            diversity_score=results.get('diversity_score', 0.0),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"MMR search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MMR search failed: {str(e)}"
        )

@router.post("/search/advanced", response_model=AdvancedQueryResponse)
async def advanced_query_search(
    request: AdvancedQueryRequest,
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    Perform advanced query search with optimization and expansion.
    
    This endpoint provides advanced query processing including query expansion,
    rewriting, and optimization based on query type and search parameters.
    """
    try:
        start_time = datetime.utcnow()
        
        # Convert string enums to proper enum types
        query_type = QueryType(request.query_type.upper())
        search_operator = SearchOperator(request.search_operator.upper())
        
        # Call the advanced query method from LlamaIndex service
        results = await llamaindex_service.advanced_query_with_optimization(
            query=request.query,
            query_type=query_type,
            top_k=request.top_k,
            enable_expansion=request.enable_expansion,
            enable_rewriting=request.enable_rewriting,
            similarity_threshold=request.similarity_threshold,
            document_ids=request.document_ids,
            metadata_filters=request.metadata_filters,
            search_operator=search_operator
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        return AdvancedQueryResponse(
            query=request.query,
            results=results.get('results', []),
            total_results=results.get('total_results', 0),
            processing_time=processing_time,
            query_type=request.query_type,
            optimizations_applied=results.get('optimizations_applied', [])
        )

    except Exception as e:
        logger.error(f"Advanced query search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advanced query search failed: {str(e)}"
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
        model_name: AI model name (LLAMA, Anthropic, CLIP, OpenAI)

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


@router.post("/documents/upload-focused")
async def upload_focused_product_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    product_name: str = Form(...),
    designer: Optional[str] = Form(None),
    search_terms: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """
    Upload PDF and process ONLY pages containing a specific product.

    This endpoint:
    1. Scans the PDF to find pages containing the product
    2. Extracts only those pages to a new PDF
    3. Processes only the focused PDF (not the entire document)
    4. Returns detailed results for the specific product

    Args:
        file: PDF file to upload
        product_name: Product name to search for (e.g., "NOVA")
        designer: Optional designer name (e.g., "SG NY")
        search_terms: Optional comma-separated additional search terms
        title: Optional document title
        description: Optional description
        tags: Optional comma-separated tags

    Returns:
        Job ID and status URL for monitoring
    """
    import tempfile
    import os

    try:
        # Generate IDs
        job_id = str(uuid4())
        document_id = str(uuid4())

        logger.info(f"🎯 FOCUSED PRODUCT EXTRACTION: {product_name}")
        logger.info(f"   Job ID: {job_id}")
        logger.info(f"   Document ID: {document_id}")
        logger.info(f"   Designer: {designer}")

        # Read PDF content
        file_content = await file.read()

        # Save to temporary file for page scanning
        temp_dir = tempfile.mkdtemp()
        original_pdf_path = os.path.join(temp_dir, "original.pdf")
        focused_pdf_path = os.path.join(temp_dir, "focused.pdf")

        with open(original_pdf_path, 'wb') as f:
            f.write(file_content)

        # Find product pages
        extractor = get_focused_product_extractor()
        search_terms_list = search_terms.split(',') if search_terms else None

        page_numbers = extractor.find_product_pages(
            original_pdf_path,
            product_name,
            designer,
            search_terms_list
        )

        if not page_numbers:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product '{product_name}' not found in PDF"
            )

        # Extract product metadata
        product_metadata = extractor.get_product_metadata_from_pages(
            original_pdf_path,
            page_numbers,
            product_name
        )

        # Extract focused pages to new PDF
        extractor.extract_product_pages_to_pdf(
            original_pdf_path,
            focused_pdf_path,
            page_numbers
        )

        # Read focused PDF
        with open(focused_pdf_path, 'rb') as f:
            focused_pdf_content = f.read()

        # Clean up temp files
        os.remove(original_pdf_path)
        os.remove(focused_pdf_path)
        os.rmdir(temp_dir)

        logger.info(f"✅ Created focused PDF with {len(page_numbers)} pages")
        logger.info(f"   Pages: {[p+1 for p in page_numbers]}")

        # Parse tags and add focused extraction metadata
        document_tags = tags.split(',') if tags else []
        document_tags.extend([
            f"focused:{product_name}",
            f"designer:{designer}" if designer else None,
            f"pages:{len(page_numbers)}"
        ])
        document_tags = [t for t in document_tags if t]  # Remove None values

        # Initialize job in job_storage (required by process_document_background)
        job_storage[job_id] = {
            "job_id": job_id,
            "document_id": document_id,
            "status": "pending",
            "progress": 0,
            "metadata": {
                "focused_extraction": True,
                "product_name": product_name,
                "designer": designer,
                "pages_found": len(page_numbers),
                "page_numbers": [p+1 for p in page_numbers],
                "product_metadata": product_metadata
            }
        }

        # Process the focused PDF using existing pipeline
        # This will now only process the product-specific pages
        background_tasks.add_task(
            process_document_background,
            job_id,
            document_id,
            focused_pdf_content,
            file.filename,
            title or f"{product_name} - Focused Extraction",
            description or f"Focused extraction of {product_name} product. Pages: {[p+1 for p in page_numbers]}. Metadata: {product_metadata}",
            document_tags,
            1000,  # chunk_size
            200    # chunk_overlap
        )

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id": job_id,
                "document_id": document_id,
                "status": "pending",
                "message": f"Focused extraction started for product: {product_name}",
                "status_url": f"/api/rag/documents/job/{job_id}",
                "pages_found": len(page_numbers),
                "page_numbers": [p+1 for p in page_numbers],
                "product_metadata": product_metadata
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Focused product extraction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Focused product extraction failed: {str(e)}"
        )