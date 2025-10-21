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
from app.services.embedding_service import EmbeddingService
from app.services.advanced_search_service import QueryType, SearchOperator
from app.services.product_creation_service import ProductCreationService
from app.services.supabase_client import get_supabase_client
from app.utils.logging import PDFProcessingLogger

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/rag", tags=["RAG"])

# Job storage for async processing
job_storage: Dict[str, Dict[str, Any]] = {}

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

async def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance."""
    try:
        settings = get_settings()
        llamaindex_config = settings.get_llamaindex_config()

        # Create embedding config from llamaindex config
        from app.schemas.embedding import EmbeddingConfig
        embedding_config = EmbeddingConfig(
            model_name=llamaindex_config.get("embedding_model", "text-embedding-3-small"),
            api_key=settings.openai_api_key,
            max_tokens=8191,
            batch_size=100,
            rate_limit_rpm=3000,
            rate_limit_tpm=1000000,
            cache_ttl=3600,
            enable_cache=True
        )
        return EmbeddingService(embedding_config)
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

        # Initialize job storage
        job_storage[job_id] = {
            "status": "pending",
            "document_id": document_id,
            "filename": file.filename,
            "created_at": datetime.utcnow().isoformat(),
            "progress": 0
        }

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
    Get the status of an async document processing job.
    """
    if job_id not in job_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    job_data = job_storage[job_id]
    return JSONResponse(content=job_data)


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
    llamaindex_service: LlamaIndexService
):
    """
    Background task to process document.
    """
    start_time = datetime.utcnow()

    logger.info(f"📋 BACKGROUND JOB STARTED: {job_id}")
    logger.info(f"   Document ID: {document_id}")
    logger.info(f"   Filename: {filename}")
    logger.info(f"   Started at: {start_time.isoformat()}")

    try:
        # Update status
        job_storage[job_id]["status"] = "processing"
        job_storage[job_id]["started_at"] = start_time.isoformat()
        job_storage[job_id]["document_id"] = document_id
        job_storage[job_id]["progress"] = 10

        # Process document through LlamaIndex service
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
            chunk_overlap=chunk_overlap
        )

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
            # ✅ AUTO-CREATE PRODUCTS FROM CHUNKS
            products_created = 0
            try:
                if chunks_created > 0:
                    logger.info(f"🏭 Starting automatic product creation for document {document_id}")
                    supabase_client = get_supabase_client()
                    product_service = ProductCreationService(supabase_client)

                    # Create products from chunks (limit to 10 products by default)
                    product_result = await product_service.create_products_from_chunks(
                        document_id=document_id,
                        workspace_id="ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
                        max_products=10,  # Limit to 10 products per document
                        min_chunk_length=100  # Only create products from chunks with at least 100 chars
                    )

                    products_created = product_result.get('products_created', 0)
                    logger.info(f"✅ Created {products_created} products from {chunks_created} chunks")
                else:
                    logger.info("⚠️ No chunks created, skipping product creation")
            except Exception as product_error:
                logger.error(f"❌ Product creation failed: {product_error}", exc_info=True)
                # Don't fail the entire job if product creation fails

            job_storage[job_id]["status"] = "completed"
            job_storage[job_id]["progress"] = 100
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