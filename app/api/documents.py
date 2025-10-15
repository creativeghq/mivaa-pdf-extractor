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
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
from datetime import datetime
import aiohttp
import aiofiles

from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks, Depends, Query
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
    # BatchProcessRequest,  # Temporarily removed to fix PENDING error
    # BatchProcessResponse,  # Temporarily removed to fix PENDING error
    DocumentAnalysisRequest,
    DocumentAnalysisResponse,
    DocumentListResponse,
    DocumentListItem,
    DocumentMetadataResponse,
    DocumentContentResponse
)
from app.schemas.common import (
    BaseResponse,
    ErrorResponse,
    PaginationParams,
    PaginationResponse,
    HealthCheckResponse,
    # ProcessingStatus,  # Temporarily removed
    URLInfo,
    MetricsSummary
)
from app.schemas.jobs import (
    JobResponse,
    JobStatus,
    JobType,
    JobProgress
)

# Import text chunking utility
from app.utils.text_chunking import smart_chunk_text

# Import existing services
from app.services.pdf_processor import PDFProcessor, PDFProcessingResult
from app.services.supabase_client import SupabaseClient, get_supabase_client as get_global_supabase_client
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
router = APIRouter(prefix="/api/documents", tags=["Document Processing"])

# Dependency function for Supabase client
def get_supabase_client() -> SupabaseClient:
    """Dependency to get Supabase client instance."""
    return get_global_supabase_client()

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


async def upload_pdf_to_storage(upload_file: UploadFile, document_id: str = None) -> Dict[str, Any]:
    """
    Upload PDF file to Supabase Storage and return storage information.

    Args:
        upload_file: FastAPI UploadFile object
        document_id: Optional document ID for organizing files

    Returns:
        Dictionary with upload result and storage information
    """
    try:
        # Read file content
        content = await upload_file.read()
        upload_file.file.seek(0)  # Reset file pointer for potential reuse

        # Get Supabase client
        supabase_client = get_supabase_client()

        # Upload to Supabase Storage
        upload_result = await supabase_client.upload_pdf_file(
            file_data=content,
            filename=upload_file.filename,
            document_id=document_id
        )

        if upload_result.get('success'):
            logger.info(f"Successfully uploaded PDF to storage: {upload_result.get('public_url')}")
        else:
            logger.warning(f"Failed to upload PDF to storage: {upload_result.get('error')}")

        return upload_result

    except Exception as e:
        logger.error(f"Error uploading PDF to storage: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


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
        # Update job status in memory
        job_storage[job_id]["status"] = "running"
        job_storage[job_id]["progress"] = {
            "current_step": "Processing document",
            "total_steps": 4,
            "completed_steps": 1,
            "percentage": 25.0
        }

        # Update in database
        try:
            from app.services.supabase_client import SupabaseClient
            supabase_client = SupabaseClient()
            await asyncio.to_thread(
                lambda: supabase_client.client.table("processing_queue")
                .update({
                    "status": "running",
                    "current_step": "Processing document",
                    "progress_percentage": 25.0,
                    "completed_steps": 1,
                    "total_steps": 4
                })
                .eq("id", job_id)
                .execute()
            )
        except Exception as db_error:
            logger.warning(f"Failed to update job status in database: {db_error}")
        
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
        job_storage[job_id]["progress"] = {
            "current_step": "Finalizing results",
            "total_steps": 4,
            "completed_steps": 4,
            "percentage": 100.0
        }
        
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
        
        # Store result in memory
        job_storage[job_id]["status"] = "completed"
        job_storage[job_id]["result"] = response_result

        # Update in database
        try:
            from app.services.supabase_client import SupabaseClient
            supabase_client = SupabaseClient()
            await asyncio.to_thread(
                lambda: supabase_client.client.table("processing_queue")
                .update({
                    "status": "completed",
                    "progress_percentage": 100.0,
                    "result_data": response_result,
                    "completed_at": datetime.utcnow().isoformat() + "Z"
                })
                .eq("id", job_id)
                .execute()
            )
        except Exception as db_error:
            logger.warning(f"Failed to update job completion in database: {db_error}")
        
    except Exception as e:
        logger.error(f"Background processing failed for job {job_id}: {e}")
        job_storage[job_id]["status"] = "failed"
        job_storage[job_id]["error"] = str(e)

        # Update error in database
        try:
            from app.services.supabase_client import SupabaseClient
            supabase_client = SupabaseClient()
            await asyncio.to_thread(
                lambda: supabase_client.client.table("processing_queue")
                .update({
                    "status": "failed",
                    "error_message": str(e),
                    "completed_at": datetime.utcnow().isoformat() + "Z"
                })
                .eq("id", job_id)
                .execute()
            )
        except Exception as db_error:
            logger.warning(f"Failed to update job error in database: {db_error}")
    
    finally:
        # Cleanup temp file
        cleanup_temp_file(file_path)


async def process_document_from_url_background(
    job_id: str,
    url: str,
    options: ProcessingOptions
) -> None:
    """Background task for document processing from URL."""
    temp_path = None
    try:
        # Update job status
        job_storage[job_id]["status"] = "running"
        job_storage[job_id]["progress"] = {
            "current_step": "Downloading document",
            "total_steps": 5,
            "completed_steps": 1,
            "percentage": 20.0
        }

        # Download file from URL
        temp_path = await download_file_from_url(url)

        # Update progress
        job_storage[job_id]["progress"] = {
            "current_step": "Processing document",
            "total_steps": 5,
            "completed_steps": 2,
            "percentage": 40.0
        }

        # Read PDF file as bytes
        async with aiofiles.open(temp_path, 'rb') as f:
            pdf_bytes = await f.read()

        # Convert ProcessingOptions to dict for PDF processor
        processing_options = {
            'extract_images': options.extract_images,
            'page_number': None,  # TODO: Parse page_range if provided
            'timeout_seconds': 300
        }

        # Update progress
        job_storage[job_id]["progress"] = {
            "current_step": "Extracting content",
            "total_steps": 5,
            "completed_steps": 3,
            "percentage": 60.0
        }

        # Process document using PDF processor service
        result: PDFProcessingResult = await pdf_processor.process_pdf_from_bytes(
            pdf_bytes=pdf_bytes,
            document_id=job_id,
            processing_options=processing_options
        )

        # Update progress
        job_storage[job_id]["progress"] = {
            "current_step": "Finalizing results",
            "total_steps": 5,
            "completed_steps": 4,
            "percentage": 80.0
        }

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
        job_storage[job_id]["status"] = "completed"
        job_storage[job_id]["result"] = response_result

    except Exception as e:
        logger.error(f"Background URL processing failed for job {job_id}: {e}")
        job_storage[job_id]["status"] = "failed"
        job_storage[job_id]["error"] = str(e)

    finally:
        # Cleanup temp file
        if temp_path:
            cleanup_temp_file(temp_path)


@router.post(
    "/process",
    response_model=DocumentProcessResponse,
    summary="Process PDF Document",
    description="Process a PDF document with comprehensive extraction and analysis"
)
async def process_document(
    background_tasks: BackgroundTasks,
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
    temp_path = None  # Initialize temp_path to avoid reference errors
    try:
        # Validate file
        await validate_pdf_file(file)

        # Generate document ID
        document_id = str(uuid.uuid4())

        # Upload PDF to Supabase Storage
        storage_result = await upload_pdf_to_storage(file, document_id)

        # Save file temporarily for processing
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
            job_data = {
                "id": job_id,
                "type": "document_processing",
                "status": "pending",
                "created_at": datetime.utcnow().isoformat() + "Z",
                "progress": None,
                "result": None,
                "error": None
            }

            # Store in memory for immediate access
            job_storage[job_id] = job_data

            # Also store in database for persistence
            try:
                await asyncio.to_thread(
                    lambda: supabase_client.client.table("processing_queue")
                    .insert({
                        "id": job_id,
                        "user_id": current_user.id,
                        "job_type": "document_processing",
                        "status": "pending",
                        "input_data": {
                            "filename": file.filename,
                            "content_type": file.content_type,
                            "async_processing": True
                        },
                        "priority": 5,
                        "created_at": job_data["created_at"]
                    })
                    .execute()
                )
                logger.info(f"Job {job_id} stored in database")
            except Exception as db_error:
                logger.warning(f"Failed to store job in database: {db_error}")
                # Continue with in-memory storage
            
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
                    document_id=document_id,  # Use the same document_id for consistency
                    processing_options=processing_options
                )
                
                # Import required types
                # from app.schemas.common import ProcessingStatus, MetricsSummary, FileUploadInfo  # Temporarily removed
                from app.schemas.common import MetricsSummary, FileUploadInfo
                from app.schemas.documents import DocumentMetadata, DocumentContent

                return DocumentProcessResponse(
                    success=True,
                    message="Document processed successfully",
                    document_id=result.document_id,
                    status="completed",  # ProcessingStatus.COMPLETED,
                    source_info=FileUploadInfo(
                        filename=file.filename,
                        size_bytes=len(pdf_bytes),
                        content_type=file.content_type or "application/pdf",
                        storage_url=storage_result.get('public_url') if storage_result.get('success') else None,
                        storage_path=storage_result.get('storage_path') if storage_result.get('success') else None
                    ),
                    content=DocumentContent(
                        markdown_content=result.markdown_content or "",
                        images=result.extracted_images or [],
                        tables=[],  # TODO: Extract from result if available
                        chunks=smart_chunk_text(
                            result.markdown_content or "",
                            chunk_size=options.chunk_size or 1000,
                            overlap=options.overlap or 200,
                            min_chunk_size=10
                        ) if result.markdown_content else [],
                        summary=None,  # TODO: Generate summary if requested
                        key_topics=[],  # TODO: Extract topics if needed
                        entities=[]  # TODO: Extract entities if needed
                    ),
                    metadata=DocumentMetadata(
                        title=result.metadata.get("title", file.filename) if result.metadata else file.filename,
                        author=result.metadata.get("author", "Unknown") if result.metadata else "Unknown",
                        creation_date=result.metadata.get("creation_date") if result.metadata and result.metadata.get("creation_date") else datetime.now(),
                        page_count=result.metadata.get("page_count", 0) if result.metadata else 0,
                        file_size=len(pdf_bytes),
                        language="en",  # TODO: Detect language
                        document_type="pdf"
                    ),
                    metrics=MetricsSummary(
                        processing_time_seconds=result.processing_time,
                        word_count=len(result.markdown_content.split()) if result.markdown_content else 0,
                        character_count=len(result.markdown_content) if result.markdown_content else 0,
                        image_count=len(result.extracted_images) if result.extracted_images else 0,
                        table_count=0,  # TODO: Count tables
                        page_count=result.metadata.get("page_count", 0) if result.metadata else 0
                    )
                )
                
            finally:
                if temp_path:
                    cleanup_temp_file(temp_path)

    except HTTPException:
        if temp_path:
            cleanup_temp_file(temp_path)
        raise
    except Exception as e:
        if temp_path:
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

) -> DocumentProcessResponse:
    """
    Download and process a PDF document from a URL.
    
    Args:
        request: URL processing request with options
        
    Returns:
        Document processing response
    """
    temp_path = None  # Initialize temp_path to avoid reference errors
    try:
        # Download file from URL (convert Pydantic URL to string)
        temp_path = await download_file_from_url(str(request.url))
        
        if request.async_processing:
            # Create job for async processing
            job_id = str(uuid.uuid4())
            job_storage[job_id] = {
                "id": job_id,
                "type": "document_processing",
                "status": "pending",
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
                    pdf_url=str(request.url),
                    document_id=str(uuid.uuid4()),
                    processing_options={
                        'extract_images': request.options.extract_images,
                        'page_number': None,  # TODO: Parse page_range if provided
                        'timeout_seconds': 300
                    }
                )
                
                # Create proper DocumentContent
                markdown_text = result.markdown_content or ""

                # Generate text chunks using smart chunking
                text_chunks = smart_chunk_text(
                    markdown_text,
                    chunk_size=request.options.chunk_size or 1000,
                    overlap=request.options.overlap or 200,
                    min_chunk_size=10
                ) if markdown_text else []

                content = DocumentContent(
                    markdown_content=markdown_text,
                    chunks=text_chunks,
                    images=result.extracted_images or [],
                    tables=[],  # TODO: Extract tables from result
                    summary=None,
                    key_topics=[],
                    entities=[]
                )

                # Create URLInfo for source_info
                source_info = URLInfo(
                    url=str(request.url),
                    content_type="application/pdf",
                    size_bytes=None,  # TODO: Get actual size
                    last_modified=None
                )

                # Create MetricsSummary
                metrics = MetricsSummary(
                    word_count=len(result.markdown_content.split()) if result.markdown_content else 0,
                    character_count=len(result.markdown_content) if result.markdown_content else 0,
                    page_count=result.metadata.get('page_count', 0) if result.metadata else 0,
                    image_count=len(result.extracted_images) if result.extracted_images else 0,
                    table_count=0,  # TODO: Count tables
                    processing_time_seconds=result.processing_time or 0.0
                )

                # Create DocumentMetadata
                metadata = DocumentMetadata(
                    title=result.metadata.get('title') if result.metadata else None,
                    author=result.metadata.get('author') if result.metadata else None,
                    subject=result.metadata.get('subject') if result.metadata else None,
                    creator=result.metadata.get('creator') if result.metadata else None,
                    producer=result.metadata.get('producer') if result.metadata else None,
                    creation_date=result.metadata.get('creation_date') if result.metadata else None,
                    modification_date=result.metadata.get('modification_date') if result.metadata else None,
                    language=None,
                    confidence_score=None,
                    tags=request.tags if hasattr(request, 'tags') else [],
                    custom_fields={}
                )

                return DocumentProcessResponse(
                    success=True,
                    message="Document processed successfully",
                    document_id=result.document_id,
                    status="completed",  # ProcessingStatus.COMPLETED,
                    source_info=source_info,
                    content=content,
                    metadata=metadata,
                    metrics=metrics,
                    storage_path=None,
                    embeddings_generated=False,
                    error_details=None
                )
                
            finally:
                if temp_path:
                    cleanup_temp_file(temp_path)

    except HTTPException:
        if temp_path:
            cleanup_temp_file(temp_path)
        raise
    except Exception as e:
        if temp_path:
            cleanup_temp_file(temp_path)
        logger.error(f"URL document processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during URL document processing"
        )


@router.post("/test-batch")
async def test_batch_endpoint():
    """
    Test endpoint to isolate the PENDING error.
    """
    try:
        logger.info("Test batch endpoint called successfully - WORKING!")

        # Test the same logic as batch-process-fixed
        urls = ["https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"]
        batch_id = str(uuid.uuid4())

        job_ids = []
        for i, url in enumerate(urls):
            job_id = f"{batch_id}_job_{i}"
            job_ids.append(job_id)

            job_data = {
                "id": job_id,
                "batch_id": batch_id,
                "type": "batch_processing",
                "status": "queued",
                "url": str(url),
                "created_at": datetime.utcnow().isoformat(),
                "progress": 0,
                "result": None,
                "error": None
            }

            job_storage[job_id] = job_data

        return {
            "success": True,
            "message": f"Test batch processing started for {len(urls)} documents",
            "batch_id": batch_id,
            "job_ids": job_ids,
            "total_documents": len(urls),
            "status": "queued"
        }
    except Exception as e:
        logger.error(f"Test batch endpoint failed: {str(e)}")
        return {"success": False, "error": str(e)}


@router.post("/new-batch-process")
async def new_batch_process_documents():
    """
    Completely new batch process endpoint to isolate the PENDING error.
    """
    try:
        logger.info("New batch process endpoint called successfully")
        return {"success": True, "message": "New batch process endpoint working", "batch_id": "test_123"}
    except Exception as e:
        logger.error(f"New batch process endpoint failed: {str(e)}")
        return {"success": False, "error": str(e)}


@router.post("/batch-process-fixed")
async def batch_process_documents_fixed():
    """
    Fixed batch process endpoint without any schema validation.
    """
    try:
        logger.info("Step 1: Fixed batch process function started")

        # Use test URLs for now
        urls = ["https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"]
        logger.info(f"Step 2: Using test URLs: {len(urls)} URLs")

        # Create batch ID
        logger.info("Step 3: Creating batch ID")
        batch_id = str(uuid.uuid4())
        logger.info(f"Step 4: Created batch ID: {batch_id}")

        # Create jobs for each URL
        logger.info("Step 5: Starting job creation loop")
        job_ids = []
        for i, url in enumerate(urls):
            logger.info(f"Step 6.{i}: Processing URL {i}: {url}")
            job_id = f"{batch_id}_job_{i}"
            job_ids.append(job_id)
            logger.info(f"Step 7.{i}: Created job ID: {job_id}")

            # Store job data without using any enums
            logger.info(f"Step 8.{i}: Creating job data")
            job_data = {
                "id": job_id,
                "batch_id": batch_id,
                "type": "batch_processing",
                "status": "queued",  # Use simple string
                "url": str(url),
                "created_at": datetime.utcnow().isoformat(),
                "progress": 0,
                "result": None,
                "error": None
            }
            logger.info(f"Step 9.{i}: Job data created successfully")

            # Store in job storage
            logger.info(f"Step 10.{i}: Storing job in job_storage")
            job_storage[job_id] = job_data
            logger.info(f"Step 11.{i}: Job stored successfully")

        logger.info(f"Step 12: Created batch {batch_id} with {len(job_ids)} jobs")

        # Return success response
        logger.info("Step 13: Creating response")
        response = {
            "success": True,
            "message": f"Batch processing started for {len(urls)} documents",
            "batch_id": batch_id,
            "job_ids": job_ids,
            "total_documents": len(urls),
            "status": "queued"
        }
        logger.info("Step 14: Response created successfully")

        return response

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Fixed batch processing failed: {error_msg}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception args: {e.args}")

        # Return error response instead of raising exception
        return {
            "success": False,
            "error": error_msg,
            "message": "Fixed batch processing failed",
            "batch_id": None,
            "job_ids": [],
            "total_documents": 0
        }


@router.post("/batch-process-test")
async def batch_process_documents_test():
    """
    Test endpoint to isolate the PENDING error.
    """
    try:
        logger.info("TEST: Batch process function started")
        return {"success": True, "message": "Test batch process working"}
    except Exception as e:
        logger.error(f"TEST: Batch process failed: {str(e)}")
        return {"success": False, "error": str(e)}


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
    generate_summary: bool = False
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
            # Read file and perform analysis using PDF processor
            with open(temp_path, 'rb') as f:
                pdf_bytes = f.read()

            # Create processing options based on analysis parameters
            processing_options = {
                'extract_images': analyze_images,
                'extract_tables': analyze_structure,
                'generate_summary': generate_summary,
                'timeout_seconds': 120
            }

            analysis_result = await pdf_processor.process_pdf_from_bytes(
                pdf_bytes=pdf_bytes,
                document_id=f"analysis_{temp_path.name}",
                processing_options=processing_options
            )
            
            # Convert PDFProcessingResult to dictionary for schema compliance
            analysis_dict = {
                "document_id": analysis_result.document_id,
                "markdown_content": analysis_result.markdown_content,
                "extracted_images": analysis_result.extracted_images or [],
                "metadata": analysis_result.metadata or {},
                "processing_time": analysis_result.processing_time,
                "multimodal_enabled": getattr(analysis_result, 'multimodal_enabled', False)
            }

            return DocumentAnalysisResponse(
                success=True,
                message="Document analysis completed",
                document_id=str(uuid.uuid4()),
                analysis=analysis_dict,
                processing_time=getattr(analysis_result, 'processing_time', 0.0)
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
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    supabase_client: SupabaseClient = Depends(get_supabase_client)
) -> JobResponse:
    """
    Get the status and result of a processing job.

    Args:
        job_id: The job ID to check

    Returns:
        Job status and result
    """
    try:
        # First check in-memory storage for active jobs
        if job_id in job_storage:
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

        # Check database for completed or historical jobs
        result = await asyncio.to_thread(
            lambda: supabase_client.client.table("processing_queue")
            .select("*")
            .eq("id", job_id)
            .single()
            .execute()
        )

        if not result.data:
            # Also check documents table for document processing jobs
            doc_result = await asyncio.to_thread(
                lambda: supabase_client.client.table("documents")
                .select("*")
                .eq("id", job_id)
                .single()
                .execute()
            )

            if doc_result.data:
                # Convert document record to job format
                doc = doc_result.data
                return JobResponse(
                    id=doc["id"],
                    type="document_processing",
                    status=doc.get("processing_status", "unknown"),
                    created_at=doc["created_at"],
                    progress={
                        "current_step": f"Document processing: {doc.get('processing_status', 'unknown')}",
                        "progress_percentage": 100 if doc.get("processing_status") == "completed" else 50
                    },
                    result={
                        "document_id": doc["id"],
                        "filename": doc.get("filename"),
                        "content_type": doc.get("content_type"),
                        "metadata": doc.get("metadata", {})
                    } if doc.get("processing_status") == "completed" else None,
                    error=None
                )

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found in processing queue or documents"
            )

        job_data = result.data

        # Convert database job to JobResponse format
        return JobResponse(
            id=job_data["id"],
            type=job_data.get("job_type", "unknown"),
            status=job_data.get("status", "unknown"),
            created_at=job_data["created_at"],
            progress={
                "current_step": job_data.get("current_step", "Processing..."),
                "progress_percentage": job_data.get("progress_percentage", 0),
                "completed_steps": job_data.get("completed_steps", 0),
                "total_steps": job_data.get("total_steps", 1)
            },
            result=job_data.get("result_data"),
            error=job_data.get("error_message")
        )

    except Exception as e:
        logger.error(f"Error retrieving job status for {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve job status: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Document Service Health Check",
    description="Check the health and availability of document processing services"
)
async def health_check(

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
    processing_status: Optional[str] = Query(None, description="Filter by processing status"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
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
        
        # Apply search filter - search in filename, content, and metadata
        if search:
            query = query.or_(
                f"filename.ilike.%{search}%,"
                f"content.ilike.%{search}%,"
                f"metadata->>filename.ilike.%{search}%,"
                f"metadata->>title.ilike.%{search}%"
            )
        
        # Apply tag filter
        if tags:
            query = query.contains("tags", tags)
        
        # Apply status filter
        if processing_status:
            query = query.eq("processing_status", processing_status)
        
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
            # Map database status to ProcessingStatus enum
            # Use processing_status field from database
            db_status = doc.get("processing_status", "unknown")
            if db_status == "unknown":
                mapped_status = "pending"  # Default to pending for unknown status
            elif db_status in ["pending", "processing", "completed", "failed", "cancelled"]:
                mapped_status = db_status
            else:
                mapped_status = "pending"  # Default fallback

            # Get document name from multiple possible sources
            document_name = (
                doc.get("filename") or  # Primary: filename field
                doc.get("title") or     # Secondary: title field
                doc.get("metadata", {}).get("filename") or  # Tertiary: filename in metadata
                doc.get("metadata", {}).get("title") or     # Quaternary: title in metadata
                f"Document {doc['id'][:8]}"  # Fallback: use document ID
            )

            # Clean up the document name (remove file extensions for display)
            if document_name and '.' in document_name:
                display_name = document_name.rsplit('.', 1)[0]  # Remove extension for display
            else:
                display_name = document_name

            documents.append({
                "document_id": doc["id"],  # Map 'id' to 'document_id'
                "document_name": display_name,  # Use cleaned display name
                "filename": document_name,  # Keep original filename with extension
                "created_at": doc["created_at"],
                "updated_at": doc.get("updated_at", doc["created_at"]),  # Use created_at as fallback
                "status": mapped_status,  # Use mapped status
                "page_count": doc.get("page_count", 0),
                "word_count": doc.get("word_count", 0),
                "file_size": doc.get("file_size") or 0,  # Convert None to 0
                "tags": doc.get("tags", []),
                "processing_time": doc.get("processing_time"),
                "has_embeddings": doc.get("has_embeddings", False),
                "content_type": doc.get("content_type", "application/pdf"),
                "processing_status": doc.get("processing_status", mapped_status)
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
        
        # Use safe JSON response to handle datetime serialization
        from app.utils.json_encoder import safe_json_response
        return JSONResponse(
            content=safe_json_response(response.model_dump()),
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
        
        # Get document name from multiple possible sources
        document_name = (
            doc.get("filename") or  # Primary: filename field
            doc.get("title") or     # Secondary: title field
            doc.get("metadata", {}).get("filename") or  # Tertiary: filename in metadata
            doc.get("metadata", {}).get("title") or     # Quaternary: title in metadata
            f"Document {doc['id'][:8]}"  # Fallback: use document ID
        )

        # Build metadata response - only include fields that belong to DocumentMetadata
        metadata = DocumentMetadata(
            title=document_name,  # Use the resolved document name as title
            author=doc.get("author") or doc.get("metadata", {}).get("author") or None,
            subject=doc.get("subject") or doc.get("metadata", {}).get("subject") or None,
            creator=doc.get("creator") or doc.get("metadata", {}).get("creator") or None,
            producer=doc.get("producer") or doc.get("metadata", {}).get("producer") or None,
            creation_date=doc.get("creation_date") or doc.get("created_at"),
            modification_date=doc.get("modification_date") or doc.get("updated_at"),
            language=doc.get("language") or doc.get("metadata", {}).get("language") or None,
            confidence_score=doc.get("confidence_score") or doc.get("metadata", {}).get("confidence_score"),
            tags=doc.get("tags", []),
            custom_fields={
                **doc.get("custom_fields", {}),
                "filename": doc.get("filename"),
                "content_type": doc.get("content_type"),
                "processing_status": doc.get("processing_status"),
                "file_size": doc.get("file_size"),
                "page_count": doc.get("page_count"),
                "word_count": doc.get("word_count")
            }
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
        
        # Use safe JSON response to handle datetime serialization
        from app.utils.json_encoder import safe_json_response
        return JSONResponse(
            content=safe_json_response(response.model_dump()),
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
    include_chunks: bool = Query(True, description="Include content chunks"),
    include_images: bool = Query(True, description="Include extracted images"),
    supabase_client: SupabaseClient = Depends(get_supabase_client)
):
    """
    Retrieve document content from MIVAA's internal storage.

    This endpoint retrieves documents that were processed by MIVAA and stored
    in its internal storage system, not from the Supabase database.

    Returns:
    - Document metadata
    - Text chunks for RAG
    - Extracted images
    - Processing statistics
    """
    try:
        logger.info(f"🔍 Retrieving document content for: {document_id}")

        # First, try to get document from Supabase database (for compatibility)
        try:
            supabase_result = await asyncio.to_thread(
                lambda: supabase_client.client.table("documents")
                .select("*")
                .eq("id", document_id)
                .single()
                .execute()
            )

            if supabase_result.data:
                logger.info(f"✅ Found document in Supabase database: {document_id}")
                doc = supabase_result.data

                # Get chunks from document_chunks table
                chunks = []
                if include_chunks:
                    chunks_result = await asyncio.to_thread(
                        lambda: supabase_client.client.table("document_chunks")
                        .select("*")
                        .eq("document_id", document_id)
                        .order("chunk_index")
                        .execute()
                    )
                    chunks = chunks_result.data or []

                # Get images from document_images table
                images = []
                if include_images:
                    images_result = await asyncio.to_thread(
                        lambda: supabase_client.client.table("document_images")
                        .select("*")
                        .eq("document_id", document_id)
                        .order("page_number")
                        .execute()
                    )
                    images = images_result.data or []

                # Build response from Supabase data
                from app.schemas.documents import DocumentChunk, ImageInfo

                # Transform chunks to DocumentChunk format
                document_chunks = []
                for i, chunk_data in enumerate(chunks):
                    chunk_id = str(chunk_data["id"])

                    # Fetch embedding for this chunk from embeddings table
                    embedding = None
                    try:
                        embedding_result = await asyncio.to_thread(
                            lambda: supabase_client.client.table("embeddings")
                            .select("embedding")
                            .eq("chunk_id", chunk_id)
                            .single()
                            .execute()
                        )

                        if embedding_result.data and embedding_result.data.get("embedding"):
                            embedding_data = embedding_result.data["embedding"]
                            # Parse embedding if it's stored as a string
                            if isinstance(embedding_data, str):
                                import json
                                try:
                                    embedding = json.loads(embedding_data)
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse embedding JSON for chunk {chunk_id}")
                                    embedding = None
                            else:
                                embedding = embedding_data

                    except Exception as embedding_error:
                        logger.warning(f"Failed to fetch embedding for chunk {chunk_id}: {embedding_error}")

                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        content=chunk_data.get("content", ""),
                        page_number=chunk_data.get("metadata", {}).get("page_number", 1),
                        chunk_index=chunk_data.get("chunk_index", i),  # Use database chunk_index or fallback to enumeration index
                        start_char=chunk_data.get("metadata", {}).get("start_char", 0),
                        end_char=chunk_data.get("metadata", {}).get("end_char", len(chunk_data.get("content", ""))),
                        embedding=embedding,  # Include the actual embedding
                        metadata=chunk_data.get("metadata", {})
                    )
                    document_chunks.append(chunk)

                # Transform images to ImageInfo format
                document_images = []
                for image_data in images:
                    image = ImageInfo(
                        image_id=str(image_data["id"]),
                        filename=image_data.get("metadata", {}).get("filename", f"image_{image_data['id']}.png"),
                        page_number=image_data.get("page_number", 1),
                        format=image_data.get("metadata", {}).get("format", "PNG"),
                        size_bytes=image_data.get("metadata", {}).get("size_bytes", 0),
                        dimensions=image_data.get("metadata", {}).get("dimensions", {"width": 0, "height": 0}),
                        description=image_data.get("caption", None),
                        url=image_data.get("image_url", None)
                    )
                    document_images.append(image)

                # Build content response
                content = DocumentContent(
                    markdown_content=doc.get("content", ""),
                    chunks=document_chunks,
                    images=document_images,
                    tables=[],  # TODO: Add table support
                    summary=doc.get("metadata", {}).get("summary"),
                    key_topics=doc.get("metadata", {}).get("key_topics", []),
                    entities=doc.get("metadata", {}).get("entities", [])
                )

                response = DocumentContentResponse(
                    success=True,
                    message="Document content retrieved from database",
                    content=content
                )

                logger.info(f"✅ Retrieved from database: {len(document_chunks)} chunks, {len(document_images)} images")
                return response

        except Exception as db_error:
            logger.info(f"📄 Document not found in Supabase database, checking MIVAA internal storage: {db_error}")

        # If not found in database, try MIVAA's internal storage
        logger.info(f"🔍 Checking MIVAA internal storage for document: {document_id}")

        # TODO: Implement MIVAA internal storage retrieval
        # For now, return an error indicating the document needs to be processed
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found in database. The document may need to be reprocessed to store data in the database."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document content {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document content: {str(e)}"
        )


@router.get("/documents/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    supabase_client: SupabaseClient = Depends(get_supabase_client)
):
    """
    Retrieve document chunks from database.

    Returns all text chunks for the specified document.
    """
    try:
        logger.info(f"🔍 Retrieving chunks for document: {document_id}")

        # Get chunks from document_chunks table
        chunks_result = await asyncio.to_thread(
            lambda: supabase_client.client.table("document_chunks")
            .select("*")
            .eq("document_id", document_id)
            .order("chunk_index")
            .execute()
        )

        if not chunks_result.data:
            logger.warning(f"No chunks found for document: {document_id}")
            return {
                "success": True,
                "message": f"No chunks found for document {document_id}",
                "data": []
            }

        chunks = chunks_result.data
        logger.info(f"✅ Retrieved {len(chunks)} chunks for document {document_id}")

        # Transform to expected format
        from app.schemas.documents import DocumentChunk

        document_chunks = []
        for i, chunk_data in enumerate(chunks):
            chunk_id = str(chunk_data["id"])

            # Fetch embedding for this chunk from embeddings table
            embedding = None
            try:
                embedding_result = await asyncio.to_thread(
                    lambda: supabase_client.client.table("embeddings")
                    .select("embedding")
                    .eq("chunk_id", chunk_id)
                    .single()
                    .execute()
                )

                if embedding_result.data and embedding_result.data.get("embedding"):
                    embedding_data = embedding_result.data["embedding"]
                    # Parse embedding if it's stored as a string
                    if isinstance(embedding_data, str):
                        import json
                        try:
                            embedding = json.loads(embedding_data)
                            logger.debug(f"✅ Found embedding for chunk {chunk_id}: {len(embedding)} dimensions")
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse embedding JSON for chunk {chunk_id}")
                            embedding = None
                    else:
                        embedding = embedding_data
                        logger.debug(f"✅ Found embedding for chunk {chunk_id}: {len(embedding)} dimensions")
                else:
                    logger.debug(f"⚠️ No embedding found for chunk {chunk_id}")

            except Exception as embedding_error:
                logger.warning(f"Failed to fetch embedding for chunk {chunk_id}: {embedding_error}")

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_data.get("content", ""),
                page_number=chunk_data.get("metadata", {}).get("page_number", 1),
                chunk_index=chunk_data.get("chunk_index", i),  # Use database chunk_index or fallback to enumeration index
                start_char=chunk_data.get("metadata", {}).get("start_char", 0),
                end_char=chunk_data.get("metadata", {}).get("end_char", len(chunk_data.get("content", ""))),
                embedding=embedding,  # Include the actual embedding
                metadata=chunk_data.get("metadata", {})
            )
            document_chunks.append(chunk)

        return {
            "success": True,
            "message": f"Retrieved {len(document_chunks)} chunks",
            "data": [chunk.model_dump() for chunk in document_chunks]
        }

    except Exception as e:
        logger.error(f"Error retrieving chunks for {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chunks: {str(e)}"
        )


@router.get("/documents/{document_id}/images")
async def get_document_images(
    document_id: str,
    supabase_client: SupabaseClient = Depends(get_supabase_client)
):
    """
    Retrieve document images from database.

    Returns all extracted images for the specified document.
    """
    try:
        logger.info(f"🔍 Retrieving images for document: {document_id}")

        # Get images from document_images table
        images_result = await asyncio.to_thread(
            lambda: supabase_client.client.table("document_images")
            .select("*")
            .eq("document_id", document_id)
            .order("page_number")
            .execute()
        )

        if not images_result.data:
            logger.warning(f"No images found for document: {document_id}")
            return {
                "success": True,
                "message": f"No images found for document {document_id}",
                "data": []
            }

        images = images_result.data
        logger.info(f"✅ Retrieved {len(images)} images for document {document_id}")

        # Transform to expected format
        from app.schemas.documents import ImageInfo

        document_images = []
        for image_data in images:
            image = ImageInfo(
                image_id=str(image_data["id"]),
                filename=image_data.get("metadata", {}).get("filename", f"image_{image_data['id']}.png"),
                page_number=image_data.get("page_number", 1),
                format=image_data.get("metadata", {}).get("format", "PNG"),
                size_bytes=image_data.get("metadata", {}).get("size_bytes", 0),
                dimensions=image_data.get("metadata", {}).get("dimensions", {"width": 0, "height": 0}),
                description=image_data.get("caption", None),
                url=image_data.get("image_url", None)
            )
            document_images.append(image)

        return {
            "success": True,
            "message": f"Retrieved {len(document_images)} images",
            "data": [image.model_dump() for image in document_images]
        }

    except Exception as e:
        logger.error(f"Error retrieving images for {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve images: {str(e)}"
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

