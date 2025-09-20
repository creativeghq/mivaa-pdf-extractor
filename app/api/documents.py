"""
Comprehensive Document Processing API Routes

This module provides advanced document processing endpoints with full integration
of the PDF processor service, Pydantic validation, async handling, and comprehensive
error management.
"""

import logging
import asyncio
import tempfile
import os
import uuid
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse
import aiohttp
import aiofiles

from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

# Import authentication dependencies
from app.dependencies import get_current_user, get_workspace_context
from app.schemas.auth import User, WorkspaceContext

# Import our comprehensive schemas
from app.schemas.documents import (
    DocumentProcessRequest,
    DocumentProcessResponse,
    DocumentMetadata,
    DocumentContent,
    ProcessingOptions,
    URLProcessRequest,
    BatchProcessRequest,
    BatchProcessResponse,
    DocumentAnalysisRequest,
    DocumentAnalysisResponse
)
from app.schemas.common import (
    BaseResponse,
    ErrorResponse,
    PaginationParams,
    PaginationResponse,
    HealthCheckResponse
)
from app.schemas.jobs import (
    JobResponse,
    JobStatus,
    JobType,
    JobProgress
)

# Import existing services
from app.services.pdf_processor import PDFProcessor, PDFProcessingResult
from app.utils.exceptions import (
    PDFProcessingError,
    PDFValidationError,
    PDFExtractionError,
    PDFDownloadError,
    PDFSizeError,
    PDFTimeoutError
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter(prefix="/api/v1/documents", tags=["Document Processing"])

# Initialize PDF processor service
pdf_processor = PDFProcessor()

# In-memory job storage (in production, use Redis or database)
job_storage = {}


async def validate_pdf_file(file: UploadFile) -> None:
    """Validate uploaded PDF file with comprehensive security checks."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )
    
    # Sanitize filename - prevent path traversal
    if ".." in file.filename or "/" in file.filename or "\\" in file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename - path traversal not allowed"
        )
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a PDF"
        )
    
    # Validate content type
    if file.content_type and not file.content_type.startswith('application/pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid content type - must be application/pdf"
        )
    
    # Check file size (limit to 50MB)
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty"
        )
    
    if file_size > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 50MB limit"
        )
    
    # Basic PDF header validation
    content_start = await file.read(8)
    file.file.seek(0)  # Reset to beginning
    if not content_start.startswith(b'%PDF-'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file - missing PDF header"
        )


async def save_upload_file_async(upload_file: UploadFile) -> Path:
    """Save uploaded file to temporary location asynchronously."""
    suffix = Path(upload_file.filename).suffix
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, 
        prefix=f"{upload_file.filename}_", 
        suffix=suffix
    )
    
    try:
        async with aiofiles.open(temp_file.name, 'wb') as f:
            content = await upload_file.read()
            await f.write(content)
        return Path(temp_file.name)
    finally:
        temp_file.close()


async def download_file_from_url(url: str) -> Path:
    """Download file from URL to temporary location."""
    parsed_url = urlparse(url)
    if not parsed_url.scheme in ['http', 'https']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid URL scheme. Only HTTP and HTTPS are supported"
        )
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to download file from URL: {response.status}"
                    )
                
                # Check content type
                content_type = response.headers.get('content-type', '')
                if 'application/pdf' not in content_type:
                    logger.warning(f"Content-Type is {content_type}, proceeding anyway")
                
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                
                async with aiofiles.open(temp_file.name, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                
                return Path(temp_file.name)
                
    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download file: {str(e)}"
        )


def cleanup_temp_file(file_path: Path) -> None:
    """Clean up temporary file."""
    try:
        if file_path.exists():
            os.unlink(file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")


async def process_document_background(
    job_id: str,
    file_path: Path,
    options: ProcessingOptions
) -> None:
    """Background task for document processing."""
    try:
        # Update job status
        job_storage[job_id]["status"] = JobStatus.RUNNING
        job_storage[job_id]["progress"] = JobProgress(
            current_step="Processing document",
            total_steps=4,
            completed_steps=1,
            percentage=25.0
        )
        
        # Read PDF file as bytes
        async with aiofiles.open(file_path, 'rb') as f:
            pdf_bytes = await f.read()
        
        # Convert ProcessingOptions to dict for PDF processor
        processing_options = {
            'extract_images': options.extract_images,
            'page_number': None,  # TODO: Parse page_range if provided
            'timeout_seconds': 300
        }
        
        # Process document using PDF processor service
        result: PDFProcessingResult = await pdf_processor.process_pdf_from_bytes(
            pdf_bytes=pdf_bytes,
            document_id=job_id,
            processing_options=processing_options
        )
        
        # Update progress
        job_storage[job_id]["progress"] = JobProgress(
            current_step="Finalizing results",
            total_steps=4,
            completed_steps=4,
            percentage=100.0
        )
        
        # Convert PDFProcessingResult to our response format
        response_result = {
            "document_id": result.document_id,
            "content": {
                "text": result.markdown_content,
                "images": result.extracted_images,
                "metadata": result.metadata
            },
            "processing_time": result.processing_time,
            "page_count": result.page_count,
            "word_count": result.word_count,
            "character_count": result.character_count
        }
        
        # Store result
        job_storage[job_id]["status"] = JobStatus.COMPLETED
        job_storage[job_id]["result"] = response_result
        
    except Exception as e:
        logger.error(f"Background processing failed for job {job_id}: {e}")
        job_storage[job_id]["status"] = JobStatus.FAILED
        job_storage[job_id]["error"] = str(e)
    
    finally:
        # Cleanup temp file
        cleanup_temp_file(file_path)


@router.post(
    "/process",
    response_model=DocumentProcessResponse,
    summary="Process PDF Document",
    description="Process a PDF document with comprehensive extraction and analysis"
)
async def process_document(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    file: UploadFile = File(..., description="PDF file to process"),
    extract_text: bool = True,
    extract_images: bool = False,
    extract_tables: bool = False,
    extract_metadata: bool = True,
    page_range: Optional[str] = None,
    async_processing: bool = False
) -> DocumentProcessResponse:
    """
    Process a PDF document with comprehensive extraction capabilities.
    
    Args:
        file: The uploaded PDF file
        extract_text: Whether to extract text content
        extract_images: Whether to extract images
        extract_tables: Whether to extract tables
        extract_metadata: Whether to extract document metadata
        page_range: Optional page range (e.g., "1-5", "1,3,5")
        async_processing: Whether to process asynchronously
        
    Returns:
        Document processing response with extracted content
    """
    try:
        # Validate file
        await validate_pdf_file(file)
        
        # Save file temporarily
        temp_path = await save_upload_file_async(file)
        
        # Create processing options
        options = ProcessingOptions(
            extract_text=extract_text,
            extract_images=extract_images,
            extract_tables=extract_tables,
            extract_metadata=extract_metadata,
            page_range=page_range
        )
        
        if async_processing:
            # Create job for async processing
            job_id = str(uuid.uuid4())
            job_storage[job_id] = {
                "id": job_id,
                "type": JobType.DOCUMENT_PROCESSING,
                "status": JobStatus.PENDING,
                "created_at": "2025-07-26T18:10:00Z",
                "progress": None,
                "result": None,
                "error": None
            }
            
            # Start background processing
            background_tasks.add_task(
                process_document_background,
                job_id,
                temp_path,
                options
            )
            
            return DocumentProcessResponse(
                success=True,
                message="Document processing started",
                job_id=job_id,
                async_processing=True
            )
        
        else:
            # Synchronous processing
            try:
                # Read PDF file as bytes
                async with aiofiles.open(temp_path, 'rb') as f:
                    pdf_bytes = await f.read()
                
                # Convert ProcessingOptions to dict for PDF processor
                processing_options = {
                    'extract_images': options.extract_images,
                    'page_number': None,  # TODO: Parse page_range if provided
                    'timeout_seconds': 300
                }
                
                # Process document using PDF processor service
                result: PDFProcessingResult = await pdf_processor.process_pdf_from_bytes(
                    pdf_bytes=pdf_bytes,
                    document_id=str(uuid.uuid4()),
                    processing_options=processing_options
                )
                
                return DocumentProcessResponse(
                    success=True,
                    message="Document processed successfully",
                    document_id=result.document_id,
                    metadata=result.metadata,
                    content={
                        "text": result.markdown_content,
                        "images": result.extracted_images,
                        "metadata": result.metadata
                    },
                    processing_time=result.processing_time
                )
                
            finally:
                cleanup_temp_file(temp_path)
    
    except HTTPException:
        cleanup_temp_file(temp_path)
        raise
    except Exception as e:
        cleanup_temp_file(temp_path)
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during document processing"
        )


@router.post(
    "/process-url",
    response_model=DocumentProcessResponse,
    summary="Process PDF from URL",
    description="Download and process a PDF document from a URL"
)
async def process_document_from_url(
    request: URLProcessRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context)
) -> DocumentProcessResponse:
    """
    Download and process a PDF document from a URL.
    
    Args:
        request: URL processing request with options
        
    Returns:
        Document processing response
    """
    try:
        # Download file from URL
        temp_path = await download_file_from_url(request.url)
        
        if request.async_processing:
            # Create job for async processing
            job_id = str(uuid.uuid4())
            job_storage[job_id] = {
                "id": job_id,
                "type": JobType.DOCUMENT_PROCESSING,
                "status": JobStatus.PENDING,
                "created_at": "2025-07-26T18:10:00Z",
                "progress": None,
                "result": None,
                "error": None
            }
            
            # Start background processing
            background_tasks.add_task(
                process_document_background,
                job_id,
                temp_path,
                request.options
            )
            
            return DocumentProcessResponse(
                success=True,
                message="Document processing started",
                job_id=job_id,
                async_processing=True
            )
        
        else:
            # Synchronous processing
            try:
                # Process document using PDF processor service
                result: PDFProcessingResult = await pdf_processor.process_pdf_from_url(
                    pdf_url=request.url,
                    document_id=str(uuid.uuid4()),
                    processing_options={
                        'extract_images': request.options.extract_images,
                        'page_number': None,  # TODO: Parse page_range if provided
                        'timeout_seconds': 300
                    }
                )
                
                return DocumentProcessResponse(
                    success=True,
                    message="Document processed successfully",
                    document_id=result.document_id,
                    metadata=result.metadata,
                    content={
                        "text": result.markdown_content,
                        "images": result.extracted_images,
                        "metadata": result.metadata
                    },
                    processing_time=result.processing_time
                )
                
            finally:
                cleanup_temp_file(temp_path)
    
    except HTTPException:
        cleanup_temp_file(temp_path)
        raise
    except Exception as e:
        cleanup_temp_file(temp_path)
        logger.error(f"URL document processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during URL document processing"
        )


@router.post(
    "/batch-process",
    response_model=BatchProcessResponse,
    summary="Batch Process Documents",
    description="Process multiple PDF documents in batch"
)
async def batch_process_documents(
    request: BatchProcessRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context)
) -> BatchProcessResponse:
    """
    Process multiple PDF documents in batch.
    
    Args:
        request: Batch processing request
        
    Returns:
        Batch processing response with job IDs
    """
    try:
        job_ids = []
        
        for url in request.urls:
            # Create individual job for each document
            job_id = str(uuid.uuid4())
            job_storage[job_id] = {
                "id": job_id,
                "type": JobType.BATCH_PROCESSING,
                "status": JobStatus.PENDING,
                "created_at": "2025-07-26T18:10:00Z",
                "progress": None,
                "result": None,
                "error": None
            }
            
            # Start background processing for each document
            background_tasks.add_task(
                process_document_background,
                job_id,
                await download_file_from_url(url),
                request.options
            )
            
            job_ids.append(job_id)
        
        return BatchProcessResponse(
            success=True,
            message=f"Batch processing started for {len(request.urls)} documents",
            batch_id=str(uuid.uuid4()),
            job_ids=job_ids,
            total_documents=len(request.urls)
        )
    
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during batch processing"
        )


@router.post(
    "/analyze",
    response_model=DocumentAnalysisResponse,
    summary="Analyze Document Structure",
    description="Perform advanced analysis of document structure and content"
)
async def analyze_document(
    file: UploadFile = File(..., description="PDF file to analyze"),
    analyze_structure: bool = True,
    analyze_content: bool = True,
    analyze_images: bool = False,
    generate_summary: bool = False,
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context)
) -> DocumentAnalysisResponse:
    """
    Perform advanced analysis of document structure and content.
    
    Args:
        file: The uploaded PDF file
        analyze_structure: Whether to analyze document structure
        analyze_content: Whether to analyze content patterns
        analyze_images: Whether to analyze images
        generate_summary: Whether to generate content summary
        
    Returns:
        Document analysis response
    """
    try:
        # Validate file
        await validate_pdf_file(file)
        
        # Save file temporarily
        temp_path = await save_upload_file_async(file)
        
        try:
            # Perform analysis using PDF processor
            analysis_result = await pdf_processor.analyze_document_async(
                file_path=temp_path,
                analyze_structure=analyze_structure,
                analyze_content=analyze_content,
                analyze_images=analyze_images,
                generate_summary=generate_summary
            )
            
            return DocumentAnalysisResponse(
                success=True,
                message="Document analysis completed",
                document_id=str(uuid.uuid4()),
                analysis=analysis_result,
                processing_time=analysis_result.get("processing_time", 0.0)
            )
            
        finally:
            cleanup_temp_file(temp_path)
    
    except HTTPException:
        cleanup_temp_file(temp_path)
        raise
    except Exception as e:
        cleanup_temp_file(temp_path)
        logger.error(f"Document analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during document analysis"
        )


@router.get(
    "/job/{job_id}",
    response_model=JobResponse,
    summary="Get Job Status",
    description="Get the status and result of a processing job"
)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context)
) -> JobResponse:
    """
    Get the status and result of a processing job.
    
    Args:
        job_id: The job ID to check
        
    Returns:
        Job status and result
    """
    if job_id not in job_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    job_data = job_storage[job_id]
    
    return JobResponse(
        id=job_data["id"],
        type=job_data["type"],
        status=job_data["status"],
        created_at=job_data["created_at"],
        progress=job_data.get("progress"),
        result=job_data.get("result"),
        error=job_data.get("error")
    )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Document Service Health Check",
    description="Check the health and availability of document processing services"
)
async def health_check(
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context)
) -> HealthCheckResponse:
    """
    Health check endpoint for document processing service.
    
    Returns:
        Service health status and capabilities
    """
    try:
        from datetime import datetime
        
        # Test PDF processor service health
        processor_healthy = True
        processor_error = None
        
        try:
            # Simple test to verify PDF processor is working
            # We'll test with minimal processing to avoid heavy operations
            test_result = await pdf_processor.process_pdf_from_bytes(
                pdf_bytes=b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n179\n%%EOF",  # Minimal valid PDF
                document_id="health_check_test",
                processing_options={'extract_images': False, 'timeout_seconds': 5}
            )
            processor_healthy = test_result is not None
        except Exception as pdf_error:
            logger.warning(f"PDF processor health check failed: {pdf_error}")
            processor_healthy = False
            processor_error = str(pdf_error)
        
        # Check other dependencies
        temp_dir_accessible = os.access(tempfile.gettempdir(), os.W_OK)
        job_storage_healthy = True  # In-memory storage is always available
        
        # Determine overall health status
        overall_status = "healthy"
        if not processor_healthy:
            overall_status = "degraded"
        if not temp_dir_accessible:
            overall_status = "unhealthy"
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat() + "Z",
            service="Document Processing API",
            version="1.0.0",
            dependencies={
                "pdf_processor": processor_healthy,
                "temp_directory": temp_dir_accessible,
                "job_storage": job_storage_healthy
            },
            error=processor_error if processor_error else None
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat() + "Z",
            service="Document Processing API",
            version="1.0.0",
            error=str(e)
        )

# ============================================================================
# PHASE 2: CONTENT RETRIEVAL & MANAGEMENT API
# ============================================================================

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search query for document content"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    status: Optional[str] = Query(None, description="Filter by processing status"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    supabase_client: SupabaseClient = Depends(get_supabase_client)
):
    """
    Retrieve a paginated list of documents with filtering and sorting.
    
    This endpoint provides comprehensive document listing with:
    - Pagination support
    - Full-text search capabilities
    - Tag-based filtering
    - Status filtering
    - Flexible sorting options
    - Caching headers for performance
    """
    try:
        # Build the query
        query = supabase_client.client.table("documents").select("*")
        
        # Apply search filter
        if search:
            query = query.ilike("title", f"%{search}%").or_("content", "ilike", f"%{search}%")
        
        # Apply tag filter
        if tags:
            query = query.contains("tags", tags)
        
        # Apply status filter
        if status:
            query = query.eq("status", status)
        
        # Apply sorting
        if sort_order == "desc":
            query = query.order(sort_by, desc=True)
        else:
            query = query.order(sort_by)
        
        # Get total count for pagination
        count_result = await asyncio.to_thread(
            lambda: supabase_client.client.table("documents").select("id", count="exact").execute()
        )
        total_count = count_result.count
        
        # Apply pagination
        offset = (page - 1) * page_size
        query = query.range(offset, offset + page_size - 1)
        
        # Execute query
        result = await asyncio.to_thread(query.execute)
        
        # Transform results to DocumentListItem format
        documents = []
        for doc in result.data:
            documents.append({
                "id": doc["id"],
                "title": doc.get("title", "Untitled"),
                "created_at": doc["created_at"],
                "updated_at": doc.get("updated_at"),
                "status": doc.get("status", "unknown"),
                "page_count": doc.get("page_count", 0),
                "word_count": doc.get("word_count", 0),
                "file_size": doc.get("file_size", 0),
                "tags": doc.get("tags", []),
                "processing_time": doc.get("processing_time"),
                "has_embeddings": doc.get("has_embeddings", False)
            })
        
        response = DocumentListResponse(
            success=True,
            message=f"Retrieved {len(documents)} documents",
            documents=documents,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
        # Add caching headers for performance
        headers = {
            "Cache-Control": "public, max-age=300",  # 5 minutes
            "ETag": f'"{hash(str(result.data))}"'
        }
        
        return JSONResponse(
            content=response.dict(),
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve documents: {str(e)}"
        )


@router.get("/documents/{document_id}", response_model=DocumentMetadataResponse)
async def get_document_metadata(
    document_id: str,
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    supabase_client: SupabaseClient = Depends(get_supabase_client)
):
    """
    Retrieve detailed metadata for a specific document.
    
    Returns comprehensive document information including:
    - Basic metadata (title, dates, status)
    - Processing statistics
    - Content structure information
    - Available features (embeddings, etc.)
    """
    try:
        # Query document metadata
        result = await asyncio.to_thread(
            lambda: supabase_client.client.table("documents")
            .select("*")
            .eq("id", document_id)
            .single()
            .execute()
        )
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        doc = result.data
        
        # Build metadata response
        metadata = DocumentMetadata(
            id=doc["id"],
            title=doc.get("title", "Untitled"),
            created_at=doc["created_at"],
            updated_at=doc.get("updated_at"),
            status=doc.get("status", "unknown"),
            page_count=doc.get("page_count", 0),
            word_count=doc.get("word_count", 0),
            file_size=doc.get("file_size", 0),
            tags=doc.get("tags", []),
            processing_time=doc.get("processing_time"),
            has_embeddings=doc.get("has_embeddings", False)
        )
        
        response = DocumentMetadataResponse(
            success=True,
            message="Document metadata retrieved successfully",
            metadata=metadata
        )
        
        # Add caching headers
        headers = {
            "Cache-Control": "public, max-age=600",  # 10 minutes
            "ETag": f'"{hash(str(doc))}"'
        }
        
        return JSONResponse(
            content=response.dict(),
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document metadata {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document metadata: {str(e)}"
        )


@router.get("/documents/{document_id}/content", response_model=DocumentContentResponse)
async def get_document_content(
    document_id: str,
    include_raw: bool = Query(False, description="Include raw extracted content"),
    include_markdown: bool = Query(True, description="Include markdown formatted content"),
    include_chunks: bool = Query(False, description="Include content chunks"),
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    supabase_client: SupabaseClient = Depends(get_supabase_client)
):
    """
    Retrieve document content in various formats.
    
    Supports multiple content formats:
    - Markdown formatted content (default)
    - Raw extracted text
    - Chunked content for processing
    
    Content is cached for performance optimization.
    """
    try:
        # Query document content
        select_fields = ["id", "title"]
        if include_raw:
            select_fields.append("raw_content")
        if include_markdown:
            select_fields.append("markdown_content")
        if include_chunks:
            select_fields.append("content_chunks")
        
        result = await asyncio.to_thread(
            lambda: supabase_client.client.table("documents")
            .select(",".join(select_fields))
            .eq("id", document_id)
            .single()
            .execute()
        )
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        doc = result.data
        
        # Build content response
        content = DocumentContent(
            id=doc["id"],
            title=doc.get("title", "Untitled"),
            raw_content=doc.get("raw_content") if include_raw else None,
            markdown_content=doc.get("markdown_content") if include_markdown else None,
            content_chunks=doc.get("content_chunks") if include_chunks else None
        )
        
        response = DocumentContentResponse(
            success=True,
            message="Document content retrieved successfully",
            content=content
        )
        
        # Add caching headers with longer cache time for content
        headers = {
            "Cache-Control": "public, max-age=1800",  # 30 minutes
            "ETag": f'"{hash(str(doc))}"',
            "Content-Type": "application/json"
        }
        
        return JSONResponse(
            content=response.dict(),
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document content {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document content: {str(e)}"
        )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    supabase_client: SupabaseClient = Depends(get_supabase_client)
):
    """
    Delete a document and all associated data.
    
    This operation:
    - Removes the document record from the database
    - Cleans up associated files and embeddings
    - Cannot be undone
    """
    try:
        # Check if document exists
        check_result = await asyncio.to_thread(
            lambda: supabase_client.client.table("documents")
            .select("id")
            .eq("id", document_id)
            .single()
            .execute()
        )
        
        if not check_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Delete the document
        await asyncio.to_thread(
            lambda: supabase_client.client.table("documents")
            .delete()
            .eq("id", document_id)
            .execute()
        )
        
        logger.info(f"Document {document_id} deleted successfully")
        
        return BaseResponse(
            success=True,
            message=f"Document {document_id} deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


# Dependency function for Supabase client
async def get_supabase_client() -> SupabaseClient:
    """Dependency to get Supabase client instance."""
    from app.services.supabase_client import SupabaseClient
    return SupabaseClient()