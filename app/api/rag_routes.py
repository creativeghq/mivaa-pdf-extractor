"""
RAG (Retrieval-Augmented Generation) API Routes

This module provides comprehensive FastAPI endpoints for RAG functionality including
document embedding, querying, chat interface, and document management.
"""

import logging
import os
import shutil
import tempfile
import gc
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Query, status, BackgroundTasks, Request
from fastapi.responses import JSONResponse
import asyncio
import aiohttp
import httpx
import sentry_sdk
from pydantic import BaseModel, Field, field_validator

from app.config import get_settings
from app.services.search.rag_service import RAGService
from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
from app.services.products.product_creation_service import ProductCreationService
from app.services.tracking.job_recovery_service import JobRecoveryService
from app.services.tracking.checkpoint_recovery_service import checkpoint_recovery_service, ProcessingStage
from app.services.core.supabase_client import get_supabase_client, SupabaseClient
from app.services.core.ai_model_tracker import AIModelTracker
from app.services.discovery.focused_product_extractor import get_focused_product_extractor
from app.services.products.product_relationship_service import ProductRelationshipService
from app.services.search.search_prompt_service import SearchPromptService
from app.services.tracking.stuck_job_analyzer import stuck_job_analyzer
from app.services.embeddings.vecs_service import get_vecs_service
from app.services.core.ai_client_service import get_ai_client_service
from app.utils.logging import PDFProcessingLogger
from app.utils.timeout_guard import with_timeout, TimeoutConstants, TimeoutError, ProgressiveTimeoutStrategy
from app.utils.circuit_breaker import claude_breaker, vision_breaker, clip_breaker, CircuitBreakerError
from app.utils.memory_monitor import memory_monitor
from app.utils.resource_manager import get_resource_manager

logger = logging.getLogger(__name__)

# ============================================================================
# Background Task Helper for Async Functions
# ============================================================================

def run_async_in_background(async_func):
    """
    Wrapper to run async functions in FastAPI BackgroundTasks.

    FastAPI's BackgroundTasks.add_task() expects synchronous functions.
    When an async function is passed, it doesn't execute properly because
    there's no event loop in the background thread.

    This wrapper creates a new event loop specifically for the background task,
    allowing async functions to run correctly in background threads.

    Usage:
        background_tasks.add_task(
            run_async_in_background(process_document_with_discovery),
            job_id=job_id,
            document_id=document_id,
            ...
        )

    Args:
        async_func: The async function to wrap

    Returns:
        A synchronous wrapper function that can be used with BackgroundTasks
    """
    def wrapper(*args, **kwargs):
        logger.info(f"🚀 Background task wrapper started for {async_func.__name__}")
        # Create a new event loop for this background task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info(f"▶️  Executing async function {async_func.__name__} in background")
            # Run the async function to completion
            loop.run_until_complete(async_func(*args, **kwargs))
            logger.info(f"✅ Background task {async_func.__name__} completed successfully")
        except Exception as e:
            logger.error(f"❌ Background task {async_func.__name__} failed: {str(e)}", exc_info=True)
            raise
        finally:
            # Clean up the event loop
            loop.close()
            logger.info(f"🔚 Background task wrapper finished for {async_func.__name__}")
    return wrapper

# Initialize router
router = APIRouter(prefix="/api/rag", tags=["RAG"])

# Job storage for async processing (in-memory cache)
job_storage: Dict[str, Dict[str, Any]] = {}
# Tracks insertion time for TTL eviction (job_id -> monotonic timestamp)
_job_storage_inserted_at: Dict[str, float] = {}

# Job recovery service (initialized on startup)
job_recovery_service: Optional[JobRecoveryService] = None

_JOB_STORAGE_TTL_SECONDS = 2 * 60 * 60  # 2 hours


def _evict_expired_job_storage() -> None:
    """Remove job_storage entries older than TTL. Call on each write to avoid unbounded growth."""
    import time
    now = time.monotonic()
    expired = [jid for jid, ts in _job_storage_inserted_at.items() if now - ts > _JOB_STORAGE_TTL_SECONDS]
    for jid in expired:
        job_storage.pop(jid, None)
        _job_storage_inserted_at.pop(jid, None)
    if expired:
        logger.info(f"🧹 Evicted {len(expired)} expired job_storage entries (TTL={_JOB_STORAGE_TTL_SECONDS}s)")


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

        # NOTE: Job cleanup moved to admin panel cron job

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

    # NEW: Consolidated upload parameters
    categories: Optional[str] = Field(
        "all",
        description="Categories to extract: 'products', 'certificates', 'logos', 'specifications', 'all', 'extract_only'. Comma-separated."
    )
    file_url: Optional[str] = Field(
        None,
        description="URL to download PDF from (alternative to file upload)"
    )
    discovery_model: Optional[str] = Field(
        "claude-vision",
        description="AI model for discovery: 'claude-vision' (Claude Sonnet 4.5 Vision - RECOMMENDED, 10x faster), 'claude-haiku-vision' (faster/cheaper), 'gpt-vision' (GPT-4o Vision), 'claude' (text-only, legacy), 'gpt' (text-only, legacy), 'haiku' (text-only, legacy)"
    )
    enable_prompt_enhancement: bool = Field(
        True,
        description="Enable AI prompt enhancement with admin customizations"
    )
    agent_prompt: Optional[str] = Field(
        None,
        description="Natural language instruction (e.g., 'extract products', 'search for NOVA')"
    )
    workspace_id: Optional[str] = Field(
        "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
        description="Workspace ID"
    )

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
    workspace_id: str = Field(..., description="Workspace ID for scoped search and related products")
    include_related_products: bool = Field(True, description="Include related products in results")
    related_products_limit: int = Field(3, ge=1, le=10, description="Max related products per result")
    use_search_prompts: bool = Field(True, description="Apply admin-configured search prompts")
    custom_formatting_prompt: Optional[str] = Field(None, description="Custom formatting prompt (overrides default)")

    # New fields for Issue #54 - Multi-Strategy Search
    material_filters: Optional[Dict[str, Any]] = Field(None, description="Material property filters for material search strategy")
    image_url: Optional[str] = Field(None, description="Image URL for image similarity search strategy")
    image_base64: Optional[str] = Field(None, description="Base64-encoded image for image similarity search strategy")

class SearchResponse(BaseModel):
    """Response model for semantic search."""
    query: str = Field(..., description="Original search query")
    enhanced_query: Optional[str] = Field(None, description="Enhanced query (if prompts applied)")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_type: str = Field(..., description="Type of search performed")
    processing_time: float = Field(..., description="Search processing time")
    search_metadata: Optional[Dict[str, Any]] = Field(None, description="Search metadata (prompts applied, etc.)")

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
async def get_rag_service() -> RAGService:
    """Get RAG service instance."""
    try:
        return RAGService()
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not available"
        )

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

# ============================================================================
# CONSOLIDATED UPLOAD ENDPOINT - Replaces all upload endpoints
# ============================================================================

@router.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None, description="PDF file to upload (required unless file_url is provided)"),

    # Basic metadata
    title: Optional[str] = Form(None, description="Document title"),
    description: Optional[str] = Form(None, description="Document description"),
    tags: Optional[str] = Form(None, description="Comma-separated tags"),

    # 🧪 TEST MODE: Process only first product for testing
    test_single_product: bool = Form(
        False,
        description="TEST MODE: Process only the first product (for testing/debugging)"
    ),

    # NEW: Category-based extraction
    categories: str = Form(
        "all",
        description="Categories to extract: 'products', 'certificates', 'logos', 'specifications', 'all', 'extract_only'. Comma-separated."
    ),

    # Material category (tiles, wood, heating, etc.)
    material_category: Optional[str] = Form(
        None,
        description="Material category: 'tiles', 'wood', 'decor', 'furniture', 'general_materials', 'paint_wall_decor', 'heating', 'sanitary', 'kitchen', 'lighting', etc."
    ),

    # NEW: URL-based upload
    file_url: Optional[str] = Form(
        None,
        description="URL to download PDF from (alternative to file upload)"
    ),

    # Discovery settings
    discovery_model: str = Form(
        "claude-vision",
        description="AI model for discovery: 'claude-vision' (Claude Sonnet 4.5 Vision - RECOMMENDED, 10x faster), 'claude-haiku-vision' (faster/cheaper), 'gpt-vision' (GPT-4o Vision), 'claude' (text-only, legacy), 'gpt' (text-only, legacy), 'haiku' (text-only, legacy)"
    ),

    # Processing settings
    chunk_size: int = Form(1000, ge=100, le=4000, description="Chunk size for text processing"),
    chunk_overlap: int = Form(200, ge=0, le=1000, description="Chunk overlap"),

    # Prompt enhancement
    enable_prompt_enhancement: bool = Form(
        True,
        description="Enable AI prompt enhancement with admin customizations"
    ),

    # Agent prompt - Optional natural language instruction
    agent_prompt: Optional[str] = Form(
        None,
        description="Natural language instruction (e.g., 'extract products', 'search for NOVA')"
    ),

    # Workspace
    workspace_id: str = Form(
        "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
        description="Workspace ID"
    )
):
    """
    **🎯 CONSOLIDATED UPLOAD ENDPOINT - Single Entry Point for All Upload Scenarios**

    This endpoint replaces:
    - `/api/documents/process` (simple extraction)
    - `/api/documents/process-url` (URL processing)
    - `/api/documents/upload` (old unified upload)

    ## 🎨 Category-Based Extraction

    Control what gets extracted:
    - `categories="products"` - Extract only products
    - `categories="certificates"` - Extract only certificates
    - `categories="logos"` - Extract only logos
    - `categories="specifications"` - Extract only specifications
    - `categories="products,certificates"` - Extract multiple categories
    - `categories="all"` - Extract everything (default - comprehensive deep analysis)
    - `categories="extract_only"` - Just extract text/images, no categorization

    **Processing:** All uploads use deep processing mode with:
    - Complete AI analysis with all models
    - Image embeddings (CLIP)
    - Advanced product enrichment
    - Quality validation
    - Full RAG pipeline

    ## 🌐 URL Processing

    Upload from URL instead of file:
    - Set `file_url="https://example.com/catalog.pdf"`
    - Leave `file` parameter empty
    - System downloads and processes automatically

    ## 🤖 AI Model Selection

    Choose discovery model:
    - `discovery_model="claude"` - Claude Sonnet 4.5 (best quality, default)
    - `discovery_model="gpt"` - GPT-5 (fast, good quality)
    - `discovery_model="haiku"` - Claude Haiku 4.5 (fastest, lower cost)

    ## 💬 Agent Prompts

    Use natural language instructions:
    - `agent_prompt="extract all products"` - Enhanced with product extraction details
    - `agent_prompt="search for NOVA"` - Enhanced with search context
    - `agent_prompt="find certificates"` - Enhanced with certificate extraction details

    ## 📊 Processing Pipeline

    **Stage 0: Discovery (0-15%)**
    - AI analyzes entire PDF
    - Identifies all content by category
    - Maps images to entities
    - Extracts metadata

    **Stage 1: Extraction (15-30%)**
    - Extracts content for specified categories
    - Filters pages based on discovery results

    **Stage 2: Chunking (30-50%)**
    - Creates semantic chunks
    - Tags chunks with categories
    - Generates text embeddings

    **Stage 3: Image Processing (50-70%)**
    - Processes images for specified categories
    - AI analysis (Vision models)
    - Generates image embeddings (CLIP)

    **Stage 4: Entity Creation (70-90%)**
    - Creates products, certificates, logos, specifications
    - Links chunks and images
    - Attaches metadata

    **Stage 5: Quality Enhancement (90-100%)**
    - Async quality validation
    - Advanced embeddings
    - Entity enrichment

    ## 📝 Examples

    ### Product Extraction
    ```bash
    curl -X POST "/api/rag/documents/upload" \\
      -F "file=@catalog.pdf" \\
      -F "categories=products"
    ```

    ### Multiple Categories
    ```bash
    curl -X POST "/api/rag/documents/upload" \\
      -F "file=@catalog.pdf" \\
      -F "categories=products,certificates,logos"
    ```

    ### URL Processing
    ```bash
    curl -X POST "/api/rag/documents/upload" \\
      -F "file_url=https://example.com/catalog.pdf" \\
      -F "categories=all"
    ```

    ### Agent-Driven Extraction
    ```bash
    curl -X POST "/api/rag/documents/upload" \\
      -F "file=@catalog.pdf" \\
      -F "agent_prompt=search for NOVA product" \\
      -F "categories=products"
    ```

    ## ✅ Response Example

    ```json
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "document_id": "660e8400-e29b-41d4-a716-446655440001",
      "status": "pending",
      "message": "Document upload successful. Processing started.",
      "status_url": "/api/rag/documents/job/550e8400-e29b-41d4-a716-446655440000",
      "categories": ["products", "certificates"],
      "estimated_time": "2-5 minutes"
    }
    ```

    ## 📊 Monitoring Progress

    Poll the status URL to track processing:
    ```bash
    curl -X GET "/api/rag/documents/job/{job_id}"
    ```

    Response includes:
    - Current stage and progress percentage
    - Checkpoint information
    - AI model usage statistics
    - Extracted entities count (chunks, images, products)
    - Error details if failed

    ## ⚠️ Error Codes

    - **400 Bad Request**: Invalid parameters (missing file/URL, invalid mode, unsupported file type)
    - **401 Unauthorized**: Missing or invalid authentication
    - **413 Payload Too Large**: File exceeds size limit (100MB)
    - **415 Unsupported Media Type**: Non-PDF file uploaded
    - **500 Internal Server Error**: Processing initialization failed
    - **503 Service Unavailable**: Background job queue full

    ## 📏 Limits

    - **Max file size**: 100MB
    - **Max concurrent jobs**: 5 per workspace
    - **Supported formats**: PDF only
    - **URL download timeout**: 60 seconds

    ## 🔄 Migration from Old Endpoints

    **Old:** `POST /api/documents/process`
    **New:** `POST /api/rag/documents/upload`

    **Old:** `POST /api/documents/process-url`
    **New:** `POST /api/rag/documents/upload` with `file_url` parameter

    **Old:** `POST /api/documents/upload`
    **New:** `POST /api/rag/documents/upload` (same endpoint, enhanced parameters)
    """

    try:
        # Validate input: either file or file_url must be provided
        if not file and not file_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'file' or 'file_url' must be provided"
            )

        if file and file_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide either 'file' or 'file_url', not both"
            )

        # Parse and validate categories
        category_list = [cat.strip() for cat in categories.split(',')]
        valid_categories = ['products', 'certificates', 'logos', 'specifications', 'all', 'extract_only']
        for cat in category_list:
            if cat not in valid_categories:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid category '{cat}'. Valid categories: {', '.join(valid_categories)}"
                )

        # Expand 'all' to all categories
        # When 'all' is specified, we want FULL extraction, not focused extraction
        use_focused_extraction = True
        if 'all' in category_list:
            category_list = ['products', 'certificates', 'logos', 'specifications']
            use_focused_extraction = False  # Process ALL pages, not just category pages

        # Handle file upload or URL download
        file_content = None
        filename = None

        if file:
            # Validate file
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only PDF files are supported"
                )
            filename = file.filename

            # STREAMING UPLOAD: Save directly to disk without loading into RAM
            # Create temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            file_path = temp_file.name
            
            try:
                 # Stream file content to temp file
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Get file size
                file_size = os.path.getsize(file_path)
                logger.info(f"✅ Streamed upload to temp file: {file_path} ({file_size} bytes)")
                
                # We do NOT load file_content into memory here
                file_content = None 
                
            except Exception as e:
                # Cleanup on failure
                if os.path.exists(file_path):
                    os.unlink(file_path)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save uploaded file: {str(e)}"
                )

        elif file_url:
            # Download from URL
            logger.info(f"📥 Downloading PDF from URL: {file_url}")

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(file_url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                        if response.status != 200:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Failed to download PDF from URL: HTTP {response.status}"
                            )

                        file_content = await response.read()

                        # Extract filename from URL or use default
                        from urllib.parse import urlparse
                        parsed_url = urlparse(file_url)
                        filename = parsed_url.path.split('/')[-1] or "downloaded.pdf"

                        if not filename.lower().endswith('.pdf'):
                            filename += '.pdf'

                        logger.info(f"✅ Downloaded {len(file_content)} bytes from URL")

            except aiohttp.ClientError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to download PDF from URL: {str(e)}"
                )

        # Generate IDs
        from uuid import uuid4
        job_id = str(uuid4())
        document_id = str(uuid4())

        logger.info(f"📤 CONSOLIDATED UPLOAD")
        logger.info(f"   Job ID: {job_id}")
        logger.info(f"   Document ID: {document_id}")
        logger.info(f"   Filename: {filename}")
        logger.info(f"   Categories: {category_list}")
        logger.info(f"   Discovery Model: {discovery_model}")
        logger.info(f"   Source: {'URL' if file_url else 'Upload'}")
        logger.info(f"   🧪 TEST MODE: {test_single_product} (type: {type(test_single_product).__name__})")
        if agent_prompt:
            logger.info(f"   Agent Prompt: {agent_prompt}")

        # Parse tags
        document_tags = []
        if tags:
            document_tags = [tag.strip() for tag in tags.split(',')]

        # File is already saved to file_path above for 'file' case
        # For URL case, we need to save it
        if file_url and file_content:
             temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
             temp_file.write(file_content)
             temp_file.close()
             file_path = temp_file.name
             file_size = len(file_content)
             # Free memory
             file_content = None
        else:
             # For file upload case, get file size from file_path
             file_size = os.path.getsize(file_path)

        # Register temp file with ResourceManager for cleanup tracking
        resource_manager = get_resource_manager()
        await resource_manager.register_resource(
            resource_id=f"temp_pdf_{document_id}",
            resource_type="temp_file",
            path=file_path,
            job_id=job_id,
            metadata={"filename": filename, "source": "rag_routes_upload"}
        )
        logger.info(f"✅ Registered temp PDF with ResourceManager: {file_path}")

        # Get Supabase client
        supabase_client = get_supabase_client()

        # Create document record
        try:
            from datetime import datetime

            # Validate workspace exists before creating document
            workspace_check = supabase_client.client.table('workspaces')\
                .select('id')\
                .eq('id', workspace_id)\
                .execute()

            if not workspace_check.data or len(workspace_check.data) == 0:
                logger.error(f"❌ Workspace {workspace_id} does not exist")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Workspace {workspace_id} does not exist. Please create the workspace first."
                )

            supabase_client.client.table('documents').insert({
                "id": document_id,
                "workspace_id": workspace_id,
                "filename": filename,
                "content_type": "application/pdf",
                "file_size": file_size,
                "file_path": file_path,
                "processing_status": "processing",
                "metadata": {
                    "title": title or filename,
                    "description": description or f"Document with {', '.join(category_list)} extraction",
                    "tags": document_tags,
                    "source": "consolidated_upload",
                    "categories": category_list,
                    "material_category": material_category,
                    "discovery_model": discovery_model,
                    "prompt_enhancement_enabled": enable_prompt_enhancement,
                    "agent_prompt": agent_prompt,
                    "file_url": file_url
                },
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).execute()
            logger.info(f"✅ Created document record {document_id}")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Failed to create document record: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create document record: {str(e)}"
            )

        # Create processed_documents record (required for job_progress foreign key)
        # Use upsert to handle cases where record already exists
        try:
            supabase_client.client.table('processed_documents').upsert({
                "id": document_id,  # Use same ID as documents table
                "workspace_id": workspace_id,
                "pdf_document_id": document_id,
                "content": "",  # Will be populated during processing
                "processing_status": "processing",
                "processing_started_at": datetime.utcnow().isoformat(),
                "metadata": {
                    "categories": category_list
                },
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).execute()
            logger.info(f"✅ Created/updated processed_documents record: {document_id}")
        except Exception as proc_doc_error:
            logger.error(f"❌ Failed to create processed_documents record: {proc_doc_error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create processed_documents record: {str(proc_doc_error)}"
            )

        # Create background job record
        try:
            supabase_client.client.table('background_jobs').insert({
                "id": job_id,
                "filename": filename,
                "job_type": "product_discovery_upload",  # CRITICAL: Must match resume logic check
                "status": "processing",
                "progress": 0,
                "document_id": document_id,
                "workspace_id": workspace_id,
                "metadata": {
                    "filename": filename,
                    "categories": category_list,
                    "material_category": material_category,
                    "discovery_model": discovery_model,
                    "prompt_enhancement_enabled": enable_prompt_enhancement,
                    "agent_prompt": agent_prompt,
                    "file_url": file_url,
                    "test_single_product": test_single_product  # 🧪 TEST MODE flag
                },
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).execute()
            logger.info(f"✅ Created background job record {job_id}")
        except Exception as e:
            logger.error(f"❌ Failed to create background job record: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create background job record: {str(e)}"
            )

        # Start background processing with deep mode
        # Use the existing process_document_with_discovery function
        # Use FastAPI BackgroundTasks to run in thread pool (prevents blocking event loop)
        # This ensures the API remains responsive during long-running processing
        background_tasks.add_task(
            run_async_in_background(process_document_with_discovery),
            job_id=job_id,
            document_id=document_id,
            file_path=file_path,  # PASS PATH, NOT CONTENT
            filename=filename,
            title=title,
            description=description,
            document_tags=document_tags,
            discovery_model=discovery_model,
            focused_extraction=use_focused_extraction,  # Use focused extraction based on categories
            extract_categories=category_list,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            workspace_id=workspace_id,
            agent_prompt=agent_prompt,
            enable_prompt_enhancement=enable_prompt_enhancement,
            test_single_product=test_single_product  # 🧪 TEST MODE flag
        )
        logger.info(f"✅ Background processing task started for job {job_id}")

        return {
            "job_id": job_id,
            "document_id": document_id,
            "status": "processing",
            "message": f"Document upload started with deep processing and {', '.join(category_list)} extraction",
            "status_url": f"/api/rag/documents/job/{job_id}",
            "categories": category_list,
            "discovery_model": discovery_model,
            "prompt_enhancement_enabled": enable_prompt_enhancement,
            "source": "url" if file_url else "upload"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Consolidated upload failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
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

            # Register with resource manager for lifecycle control
            resource_manager = get_resource_manager()
            await resource_manager.register_resource(
                resource_id=f"temp_pdf_{document_id}",
                resource_type="file",
                path=file_path,
                job_id=job_id,
                metadata={"document_id": document_id}
            )
            logger.info(f"✅ Registered temp PDF with ResourceManager: {file_path}")
            
            # Free memory
            file_content = None

            # Initialize job in job_storage (CRITICAL: required by process_document_with_discovery)
            import time as _time
            _evict_expired_job_storage()
            job_storage[job_id] = {
                "job_id": job_id,
                "document_id": document_id,
                "status": "processing",
                "progress": job_data.get('progress', 0),
                "metadata": job_data.get('metadata', {})
            }
            _job_storage_inserted_at[job_id] = _time.monotonic()
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

            logger.info(f"   Resume parameters: discovery_model={discovery_model}, categories={categories}, focused={use_focused_extraction}, test_mode={test_single_product}")

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

            logger.info(f"✅ Background task triggered for job {job_id} (type: {job_type})")

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


@router.get("/chunks")
async def get_chunks(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of chunks to return"),
    offset: int = Query(0, ge=0, description="Number of chunks to skip"),
    include_embeddings: bool = Query(True, description="Include embeddings in response")
):
    """
    Get chunks for a document with embeddings.

    Args:
        document_id: Document ID to filter chunks
        limit: Maximum number of chunks to return
        offset: Pagination offset
        include_embeddings: Whether to include embeddings (default: True)

    Returns:
        List of chunks with metadata and embeddings
    """
    try:
        supabase_client = get_supabase_client()

        if not document_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="document_id is required"
            )

        # Query chunks
        query = supabase_client.client.table('document_chunks').select('*').eq('document_id', document_id)
        query = query.range(offset, offset + limit - 1)
        result = query.execute()

        chunks = result.data if result.data else []

        # Add embeddings to each chunk if requested
        # Note: embeddings are stored directly in document_chunks.text_embedding
        if include_embeddings and chunks:
            for chunk in chunks:
                # Embeddings are stored directly in the chunk record
                text_embedding = chunk.get('text_embedding')
                chunk['embedding'] = text_embedding
                chunk['embeddings'] = [{'embedding': text_embedding, 'type': 'text'}] if text_embedding else []
                chunk['has_embedding'] = text_embedding is not None

        return JSONResponse(content={
            "document_id": document_id,
            "chunks": chunks,
            "count": len(chunks),
            "limit": limit,
            "offset": offset
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chunks for document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chunks: {str(e)}"
        )


@router.get("/images")
async def get_images(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of images to return"),
    offset: int = Query(0, ge=0, description="Number of images to skip")
):
    """
    Get images for a document.

    Args:
        document_id: Document ID to filter images
        limit: Maximum number of images to return
        offset: Pagination offset

    Returns:
        List of images with metadata
    """
    try:
        supabase_client = get_supabase_client()

        if not document_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="document_id is required"
            )

        # Query images
        query = supabase_client.client.table('document_images').select('*').eq('document_id', document_id)
        query = query.range(offset, offset + limit - 1)
        result = query.execute()

        images = result.data if result.data else []

        return JSONResponse(content={
            "document_id": document_id,
            "images": images,
            "count": len(images),
            "limit": limit,
            "offset": offset
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get images for document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve images: {str(e)}"
        )


@router.get("/products")
async def get_products(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of products to return"),
    offset: int = Query(0, ge=0, description="Number of products to skip"),
    include_tables: bool = Query(True, description="Include tables in product response")
):
    """
    Get products for a document.

    Args:
        document_id: Document ID to filter products
        limit: Maximum number of products to return
        offset: Pagination offset
        include_tables: Whether to include tables in the response

    Returns:
        List of products with metadata, optionally including tables
    """
    try:
        supabase_client = get_supabase_client()

        if not document_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="document_id is required"
            )

        # Query products
        query = supabase_client.client.table('products').select('*').eq('source_document_id', document_id)
        query = query.range(offset, offset + limit - 1)
        result = query.execute()

        products = result.data if result.data else []

        # Optionally fetch tables for each product
        if include_tables and products:
            product_ids = [p['id'] for p in products]
            tables_response = supabase_client.client.table('product_tables')\
                .select('*')\
                .in_('product_id', product_ids)\
                .execute()

            # Group tables by product_id
            tables_by_product = {}
            if tables_response.data:
                for table in tables_response.data:
                    product_id = table.get('product_id')
                    if product_id not in tables_by_product:
                        tables_by_product[product_id] = []
                    tables_by_product[product_id].append(table)

            # Add tables to each product
            for product in products:
                product['tables'] = tables_by_product.get(product['id'], [])

        return JSONResponse(content={
            "document_id": document_id,
            "products": products,
            "count": len(products),
            "limit": limit,
            "offset": offset
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get products for document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve products: {str(e)}"
        )


@router.get("/embeddings")
async def get_embeddings(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    embedding_type: Optional[str] = Query(None, description="Filter by embedding type (text, visual, multimodal, color, texture, application)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of embeddings to return"),
    offset: int = Query(0, ge=0, description="Number of embeddings to skip")
):
    """
    Get embeddings for a document.

    Args:
        document_id: Document ID to filter embeddings
        embedding_type: Type of embedding (text, visual, clip, color, texture, application, multimodal)
        limit: Maximum number of embeddings to return
        offset: Pagination offset

    Returns:
        List of embeddings with metadata from document_images and document_chunks
    """
    try:
        supabase_client = get_supabase_client()

        if not document_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="document_id is required"
            )

        embeddings = []
        embedding_stats = {}

        # Query image embeddings from document_images
        image_query = supabase_client.client.table('document_images').select(
            'id, image_url, visual_clip_embedding_512, color_embedding_256, texture_embedding_256, application_embedding_512, multimodal_fusion_embedding_2688'
        ).eq('document_id', document_id).range(offset, offset + limit - 1)

        image_result = image_query.execute()

        if image_result.data:
            for img in image_result.data:
                # Add CLIP embedding
                if img.get('visual_clip_embedding_512'):
                    if not embedding_type or embedding_type in ['visual', 'clip']:
                        embeddings.append({
                            'id': f"{img['id']}_clip",
                            'entity_id': img['id'],
                            'entity_type': 'image',
                            'embedding_type': 'visual_clip_512',
                            'dimension': 512,
                            'model': 'clip-vit-base-patch32',
                            'embedding': img['visual_clip_embedding_512']
                        })
                        embedding_stats['visual_clip_512'] = embedding_stats.get('visual_clip_512', 0) + 1

                # Add color embedding
                if img.get('color_embedding_256'):
                    if not embedding_type or embedding_type == 'color':
                        embeddings.append({
                            'id': f"{img['id']}_color",
                            'entity_id': img['id'],
                            'entity_type': 'image',
                            'embedding_type': 'color_256',
                            'dimension': 256,
                            'model': 'clip-vit-base-patch32',
                            'embedding': img['color_embedding_256']
                        })
                        embedding_stats['color_256'] = embedding_stats.get('color_256', 0) + 1

                # Add texture embedding
                if img.get('texture_embedding_256'):
                    if not embedding_type or embedding_type == 'texture':
                        embeddings.append({
                            'id': f"{img['id']}_texture",
                            'entity_id': img['id'],
                            'entity_type': 'image',
                            'embedding_type': 'texture_256',
                            'dimension': 256,
                            'model': 'clip-vit-base-patch32',
                            'embedding': img['texture_embedding_256']
                        })
                        embedding_stats['texture_256'] = embedding_stats.get('texture_256', 0) + 1

                # Add application embedding
                if img.get('application_embedding_512'):
                    if not embedding_type or embedding_type == 'application':
                        embeddings.append({
                            'id': f"{img['id']}_application",
                            'entity_id': img['id'],
                            'entity_type': 'image',
                            'embedding_type': 'application_512',
                            'dimension': 512,
                            'model': 'clip-vit-base-patch32',
                            'embedding': img['application_embedding_512']
                        })
                        embedding_stats['application_512'] = embedding_stats.get('application_512', 0) + 1

                # Add multimodal fusion embedding
                if img.get('multimodal_fusion_embedding_2688'):
                    if not embedding_type or embedding_type == 'multimodal':
                        embeddings.append({
                            'id': f"{img['id']}_multimodal",
                            'entity_id': img['id'],
                            'entity_type': 'image',
                            'embedding_type': 'multimodal_fusion_2688',
                            'dimension': 2688,
                            'model': 'siglip-so400m-14-384',
                            'embedding': img['multimodal_fusion_embedding_2688']
                        })
                        embedding_stats['multimodal_fusion_2688'] = embedding_stats.get('multimodal_fusion_2688', 0) + 1

        # Query text embeddings from document_chunks
        if not embedding_type or embedding_type == 'text':
            chunk_query = supabase_client.client.table('document_chunks').select(
                'id, content, text_embedding, embedding_dimension'
            ).eq('document_id', document_id).not_('text_embedding', 'is', None).range(offset, offset + limit - 1)

            chunk_result = chunk_query.execute()

            if chunk_result.data:
                for chunk in chunk_result.data:
                    dimension = chunk.get('embedding_dimension', 1024)  # Default to 1024 if not specified
                    embeddings.append({
                        'id': f"{chunk['id']}_text",
                        'entity_id': chunk['id'],
                        'entity_type': 'chunk',
                        'embedding_type': f'text_{dimension}',
                        'dimension': dimension,
                        'model': 'voyage-3',
                        'embedding': chunk['text_embedding']
                    })
                    embedding_stats[f'text_{dimension}'] = embedding_stats.get(f'text_{dimension}', 0) + 1

        return JSONResponse(content={
            "document_id": document_id,
            "embeddings": embeddings,
            "count": len(embeddings),
            "limit": limit,
            "offset": offset,
            "statistics": {
                "total_embeddings": len(embeddings),
                "by_type": embedding_stats
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get embeddings for document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve embeddings: {str(e)}"
        )


@router.get("/relevancies")
async def get_relevancies(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of relevancies to return"),
    offset: int = Query(0, ge=0, description="Number of relevancies to skip"),
    min_score: float = Query(0.0, ge=0.0, le=1.0, description="Minimum relevance score")
):
    """
    Get chunk-image relevancy relationships for a document.

    Args:
        document_id: Document ID to filter relevancies
        limit: Maximum number of relevancies to return
        offset: Pagination offset
        min_score: Minimum relevance score threshold

    Returns:
        List of chunk-image relationships with relevance scores
    """
    try:
        supabase_client = get_supabase_client()

        if not document_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="document_id is required"
            )

        # Query chunk-image relationships through chunks
        query = supabase_client.client.table('chunk_image_relationships').select(
            '*, document_chunks!inner(document_id, content), document_images(image_url, caption)'
        ).eq('document_chunks.document_id', document_id)

        if min_score > 0:
            query = query.gte('relevance_score', min_score)

        query = query.order('relevance_score', desc=True)
        query = query.range(offset, offset + limit - 1)
        result = query.execute()

        relevancies = result.data if result.data else []

        # Calculate statistics
        relationship_types = {}
        for rel in relevancies:
            rel_type = rel.get('relationship_type', 'unknown')
            if rel_type not in relationship_types:
                relationship_types[rel_type] = 0
            relationship_types[rel_type] += 1

        return JSONResponse(content={
            "document_id": document_id,
            "relevancies": relevancies,
            "count": len(relevancies),
            "limit": limit,
            "offset": offset,
            "statistics": {
                "total_relevancies": len(relevancies),
                "by_relationship_type": relationship_types,
                "min_score_filter": min_score
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get relevancies for document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve relevancies: {str(e)}"
        )


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
    sub_job_id = str(uuid4())  # ✅ FIX: Generate proper UUID instead of appending "_products"

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
            current_job = await job_recovery_service.get_job_status(job_id)  # ✅ FIX: Use correct method name get_job_status
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
        except Exception:
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
    sub_job_id = str(uuid4())  # ✅ FIX: Generate proper UUID instead of appending "_images"

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
        from app.services.images.background_image_processor import start_background_image_processing
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


async def process_document_with_discovery(
    job_id: str,
    document_id: str,
    file_path: str,  # CHANGED: file_path instead of file_content
    filename: str,
    title: Optional[str],
    description: Optional[str],
    document_tags: List[str],
    discovery_model: str,
    focused_extraction: bool,
    extract_categories: List[str],
    chunk_size: int,
    chunk_overlap: int,
    workspace_id: str = None,
    agent_prompt: Optional[str] = None,
    enable_prompt_enhancement: bool = True,
    test_single_product: bool = False  # 🧪 TEST MODE: Process only first product
):
    """
    Background task to process document with intelligent product discovery.

    PRODUCT-CENTRIC ARCHITECTURE (v2):
    Stage 0: Product Discovery (0-15%) - Analyze PDF with Claude/GPT, discover all products

    Then FOR EACH PRODUCT (15-85%):
      Stage 1: Extract product pages
      Stage 2: Create text chunks for product
      Stage 3: Process product images
      Stage 4: Create product in database
      Stage 5: Create relationships

    Stage 6: Quality Enhancement (85-100%) - Final validation

    Benefits:
    - Lower memory usage (process one product at a time)
    - Better progress tracking (per-product granularity)
    - Easier error recovery (failed products don't block others)
    - Clearer checkpointing (per-product state)

    Args:
        focused_extraction: If True (default), only process pages/images from extract_categories.
                          If False, process entire PDF.
        extract_categories: List of categories to extract (e.g., ['products'], ['certificates', 'logos']).
                          Categories: 'products', 'certificates', 'logos', 'specifications', 'all'
    """
    # Use default workspace ID from config if not provided
    workspace_id = workspace_id or get_settings().default_workspace_id
    start_time = datetime.utcnow()

    # Initialize lazy loading for this job
    from app.services.utilities.lazy_loader import get_component_manager
    component_manager = get_component_manager()

    # Track which components are loaded for cleanup
    loaded_components = []

    logger.info("=" * 80)
    logger.info(f"🔍 [PRODUCT DISCOVERY] STARTING")
    logger.info("=" * 80)
    logger.info(f"📋 Job ID: {job_id}")
    logger.info(f"📄 Document ID: {document_id}")
    logger.info(f"🤖 Discovery Model: {discovery_model.upper()}")
    logger.info(f"🎯 Focused Extraction: {'ENABLED' if focused_extraction else 'DISABLED (Full PDF)'}")
    logger.info(f"📦 Extract Categories: {', '.join(extract_categories).upper()}")

        # Send job start event to Sentry
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("job_id", job_id)
        scope.set_tag("document_id", document_id)
        scope.set_tag("discovery_model", discovery_model)
        scope.set_tag("focused_extraction", focused_extraction)
        scope.set_context("job_config", {
            "extract_categories": extract_categories,
            "chunk_size": chunk_size
        })

        scope.set_context("job_details", {
            "filename": filename,
            "discovery_model": discovery_model,
            "focused_extraction": focused_extraction,
            "extract_categories": extract_categories,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "workspace_id": workspace_id,
            "started_at": start_time.isoformat()
        })
        sentry_sdk.capture_message(
            f"🚀 PDF Processing Started: {filename} (Job: {job_id})",
            level="info"
        )

    # Initialize job_storage for this job if not already present
    # This is critical - the job may not exist in memory if this is a new upload
    if job_id not in job_storage:
        import time as _time
        _evict_expired_job_storage()
        job_storage[job_id] = {
            "job_id": job_id,
            "document_id": document_id,
            "status": "processing",
            "progress": 0,
            "metadata": {
                "filename": filename,
                "test_single_product": test_single_product
            }
        }
        _job_storage_inserted_at[job_id] = _time.monotonic()
        logger.info(f"✅ Initialized job_storage for job {job_id}")

    # Read file content ONLY when needed
    logger.info(f"📖 [BACKGROUND TASK] Reading file from disk: {file_path}")
    if not os.path.exists(file_path):
         logger.error(f"❌ [BACKGROUND TASK] File not found at {file_path}")
         raise FileNotFoundError(f"File not found at {file_path}")

    logger.info(f"🔧 [BACKGROUND TASK] Opening file for reading...")
    with open(file_path, 'rb') as f:
        file_content = f.read()

    file_size = len(file_content)
    logger.info(f"✅ [BACKGROUND TASK] File read successfully: {file_size} bytes ({file_size / (1024*1024):.1f} MB)")
    logger.info("=" * 80)

    # ============================================================================
    # PRE-PROCESSING: ADD PAGE NUMBERS TO PDF
    # ============================================================================
    logger.info("=" * 80)
    logger.info("📝 PRE-PROCESSING: ADDING PAGE NUMBERS TO PDF")
    logger.info("=" * 80)

    from app.services.preprocessing import preprocess_pdf_with_page_numbers

    # Update progress to 5% (page numbering started)
    job_storage[job_id]["progress"] = 5
    job_storage[job_id]["metadata"] = {
        **job_storage[job_id].get("metadata", {}),
        "current_step": "Adding page numbers to PDF"
    }

    def page_numbering_progress(current: int, total: int, message: str):
        """Progress callback for page numbering."""
        # Update job metadata with page numbering progress
        pct = 5 + int((current / max(total, 1)) * 5)  # 5-10% range
        job_storage[job_id]["progress"] = pct
        job_storage[job_id]["metadata"]["current_step"] = message
        if current % 50 == 0 or current == total:
            logger.info(f"   📝 {message}")

    try:
        numbered_pdf_path, numbering_stats = await preprocess_pdf_with_page_numbers(
            pdf_path=file_path,
            job_id=job_id,
            progress_callback=page_numbering_progress
        )
        logger.info(f"✅ Page numbering complete: {numbering_stats['pages_numbered']} pages")
        logger.info(f"   📄 Numbered PDF: {numbered_pdf_path}")

        # Use the numbered PDF for all subsequent processing
        file_path = numbered_pdf_path

        # Re-read file content from numbered PDF
        with open(file_path, 'rb') as f:
            file_content = f.read()
        logger.info(f"   📖 Re-read numbered PDF: {len(file_content)} bytes")

    except Exception as e:
        logger.warning(f"⚠️ Page numbering failed, continuing with original PDF: {e}")
        # Continue with original file_path and file_content

    # Update progress to 10% (page numbering complete)
    job_storage[job_id]["progress"] = 10
    job_storage[job_id]["metadata"]["current_step"] = "Page numbering complete"
    logger.info("=" * 80)

    # Get AI model configuration
    settings = get_settings()
    image_analysis_model = settings.image_analysis_model  # ✅ FIXED: Direct property access

    # ✅ FIX: Define missing model variables from AI config
    from app.models.ai_config import DEFAULT_AI_CONFIG
    product_creation_model = DEFAULT_AI_CONFIG.discovery_model  # Use discovery model for product creation
    quality_validation_model = DEFAULT_AI_CONFIG.classification_validation_model  # Claude for quality validation


    try:
        # ============================================================================
        # WARMUP ALL HUGGINGFACE ENDPOINTS IN PARALLEL
        # ============================================================================
        logger.info("=" * 80)
        logger.info("🔥 WARMING UP ALL HUGGINGFACE ENDPOINTS (PARALLEL)")
        logger.info("=" * 80)

        # Create WARMUP_STARTED checkpoint
        warmup_endpoints_to_start = ["qwen", "slig", "yolo", "chandra"]
        await checkpoint_recovery_service.create_checkpoint(
            job_id=job_id,
            stage=ProcessingStage.WARMUP_STARTED,
            data={
                "endpoints_to_warmup": warmup_endpoints_to_start,
                "total_endpoints": len(warmup_endpoints_to_start)
            },
            metadata={
                "current_step": "Starting HuggingFace endpoint warm-up",
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        # Update progress to 12% (warm-up started)
        job_storage[job_id]["progress"] = 12
        job_storage[job_id]["metadata"] = {
            **job_storage[job_id].get("metadata", {}),
            "current_step": "Warming up HuggingFace endpoints",
            "warmup_status": "started",
            "endpoints_to_warmup": warmup_endpoints_to_start
        }
        if job_recovery_service:
            await job_recovery_service.persist_job(
                job_id=job_id,
                document_id=document_id,
                filename=filename,
                status="processing",
                progress=12,
                metadata=job_storage[job_id]["metadata"]
            )
        logger.info(f"📊 Job {job_id} progress: 12% - Starting endpoint warm-up")

        settings = get_settings()
        endpoint_managers = {}
        warmup_results = {"success": [], "failed": [], "skipped": []}

        async def warmup_qwen():
            try:
                from app.services.embeddings.qwen_endpoint_manager import QwenEndpointManager
                qwen_config = settings.get_qwen_config()
                if qwen_config.get("enabled", False):
                    manager = QwenEndpointManager(
                        endpoint_url=qwen_config["endpoint_url"],
                        endpoint_token=qwen_config.get("endpoint_token", qwen_config.get("hf_token", "")),
                        endpoint_name=qwen_config.get("endpoint_name", "mh-qwen332binstruct"),
                        namespace=qwen_config.get("namespace", "basiliskan"),
                        warmup_timeout=qwen_config.get("warmup_timeout", 360),  # FIXED: Use config value, default 360s
                        enabled=True
                    )
                    endpoint_managers['qwen'] = manager

                    # Check status before resuming (use thread pool for blocking call)
                    def check_qwen_status():
                        endpoint = manager._get_endpoint()
                        if endpoint:
                            endpoint.fetch()
                            return endpoint.status
                        return None
                    
                    status = await asyncio.to_thread(check_qwen_status)
                    if status == "running":
                        logger.info("   ✅ Qwen endpoint already running, skipping warmup")
                        manager.warmup_completed = True
                        warmup_results["skipped"].append("qwen")
                        return

                    # Run blocking resume in thread pool
                    resumed = await asyncio.to_thread(manager.resume_if_needed)
                    if resumed:
                        logger.info(f"   ⏳ Warming up Qwen ({manager.warmup_timeout}s)...")
                        await asyncio.sleep(1)  # Minimal async yield
                        manager.warmup_completed = True
                        warmup_results["success"].append("qwen")
                        logger.info("   ✅ Qwen warmup complete")
                else:
                    warmup_results["skipped"].append("qwen")
            except Exception as e:
                warmup_results["failed"].append({"endpoint": "qwen", "error": str(e)})
                logger.warning(f"⚠️ Failed to warmup Qwen endpoint: {e}")

        async def warmup_slig():
            try:
                from app.services.embeddings.slig_endpoint_manager import SLIGEndpointManager
                slig_config = settings.get_slig_config()
                if slig_config.get("enabled", False):
                    manager = SLIGEndpointManager(
                        endpoint_url=slig_config["endpoint_url"],
                        hf_token=slig_config["hf_token"],
                        endpoint_name=slig_config.get("endpoint_name", "mh-slig"),
                        namespace=slig_config.get("namespace", "basiliskan"),
                        enabled=True
                    )
                    endpoint_managers['slig'] = manager

                    # Check if already running (use thread pool for blocking call)
                    def check_slig_status():
                        endpoint = manager._get_endpoint()
                        if endpoint:
                            endpoint.fetch()
                            return endpoint.status
                        return None
                    
                    status = await asyncio.to_thread(check_slig_status)
                    if status == "running":
                        logger.info("   ✅ SLIG endpoint already running")
                        warmup_results["skipped"].append("slig")
                        return

                    # Run blocking resume and warmup in thread pool
                    def resume_and_warmup_slig():
                        if manager.resume_if_needed():
                            return manager.warmup()
                        return False
                    
                    success = await asyncio.to_thread(resume_and_warmup_slig)
                    if success:
                        warmup_results["success"].append("slig")
                        logger.info("   ✅ SLIG warmup complete")
                else:
                    warmup_results["skipped"].append("slig")
            except Exception as e:
                warmup_results["failed"].append({"endpoint": "slig", "error": str(e)})
                logger.warning(f"⚠️ Failed to warmup SLIG endpoint: {e}")

        async def warmup_yolo():
            try:
                from app.services.pdf.yolo_endpoint_manager import YoloEndpointManager
                yolo_config = settings.get_yolo_config()
                if yolo_config.get("enabled", False):
                    manager = YoloEndpointManager(
                        endpoint_url=yolo_config["endpoint_url"],
                        hf_token=yolo_config.get("hf_token", ""),
                        endpoint_name=yolo_config.get("endpoint_name"),
                        namespace=yolo_config.get("namespace"),
                        enabled=True
                    )
                    endpoint_managers['yolo'] = manager

                    # Check if already running (use thread pool for blocking call)
                    def check_yolo_status():
                        endpoint = manager._get_endpoint()
                        if endpoint:
                            endpoint.fetch()
                            return endpoint.status
                        return None
                    
                    status = await asyncio.to_thread(check_yolo_status)
                    if status == "running":
                        logger.info("   ✅ YOLO endpoint already running")
                        warmup_results["skipped"].append("yolo")
                        return

                    # Run blocking resume and warmup in thread pool
                    def resume_and_warmup_yolo():
                        if manager.resume_if_needed():
                            return manager.warmup()
                        return False
                    
                    success = await asyncio.to_thread(resume_and_warmup_yolo)
                    if success:
                        warmup_results["success"].append("yolo")
                        logger.info("   ✅ YOLO warmup complete")
                else:
                    warmup_results["skipped"].append("yolo")
            except Exception as e:
                warmup_results["failed"].append({"endpoint": "yolo", "error": str(e)})
                logger.warning(f"⚠️ Failed to warmup YOLO endpoint: {e}")

        async def warmup_chandra():
            try:
                from app.services.pdf.chandra_endpoint_manager import ChandraEndpointManager
                chandra_config = settings.get_chandra_config()
                if chandra_config.get("enabled", False):
                    manager = ChandraEndpointManager(
                        endpoint_url=chandra_config["endpoint_url"],
                        hf_token=chandra_config.get("hf_token", ""),
                        endpoint_name=chandra_config.get("endpoint_name"),
                        namespace=chandra_config.get("namespace"),
                        enabled=True
                    )
                    endpoint_managers['chandra'] = manager

                    # Check if already running (use thread pool for blocking call)
                    def check_chandra_status():
                        endpoint = manager._get_endpoint()
                        if endpoint:
                            endpoint.fetch()
                            return endpoint.status
                        return None
                    
                    status = await asyncio.to_thread(check_chandra_status)
                    if status == "running":
                        logger.info("   ✅ Chandra endpoint already running")
                        warmup_results["skipped"].append("chandra")
                        return

                    # Run blocking resume in thread pool
                    resumed = await asyncio.to_thread(manager.resume_if_needed)
                    if resumed:
                        logger.info(f"   ⏳ Warming up Chandra (60s)...")
                        await asyncio.sleep(60)
                        warmup_results["success"].append("chandra")
                        logger.info("   ✅ Chandra warmup complete")
                else:
                    warmup_results["skipped"].append("chandra")
            except Exception as e:
                warmup_results["failed"].append({"endpoint": "chandra", "error": str(e)})
                logger.warning(f"⚠️ Failed to warmup Chandra endpoint: {e}")

        # Execute all warmups in parallel
        await asyncio.gather(
            warmup_qwen(),
            warmup_slig(),
            warmup_yolo(),
            warmup_chandra()
        )

        logger.info("=" * 80)
        logger.info(f"✅ WARMUP PHASE COMPLETE - {len(endpoint_managers)} endpoints resumed")
        logger.info(f"   Success: {warmup_results['success']}")
        logger.info(f"   Skipped (already running): {warmup_results['skipped']}")
        if warmup_results['failed']:
            logger.warning(f"   Failed: {warmup_results['failed']}")

            # Check if SLIG (critical for CLIP embeddings) failed
            failed_endpoints = [f.get('endpoint') if isinstance(f, dict) else f for f in warmup_results['failed']]
            if 'slig' in failed_endpoints:
                logger.error("❌ CRITICAL: SLIG endpoint warmup failed - cannot generate CLIP embeddings")
                logger.error("   Stopping processing to prevent incomplete data")

                # Update job status to failed
                job_storage[job_id]["status"] = "failed"
                job_storage[job_id]["error"] = "SLIG endpoint warmup failed - CLIP embeddings unavailable"
                if job_recovery_service:
                    await job_recovery_service.persist_job(
                        job_id=job_id,
                        document_id=document_id,
                        filename=filename,
                        status="failed",
                        progress=job_storage[job_id].get("progress", 0),
                        metadata=job_storage[job_id].get("metadata", {}),
                        error="SLIG endpoint warmup failed - CLIP embeddings unavailable"
                    )
                raise HTTPException(
                    status_code=503,
                    detail="SLIG endpoint warmup failed. Please retry later or check HuggingFace endpoint status."
                )
        logger.info("=" * 80)

        # Create WARMUP_COMPLETE checkpoint with results
        await checkpoint_recovery_service.create_checkpoint(
            job_id=job_id,
            stage=ProcessingStage.WARMUP_COMPLETE,
            data={
                "endpoints_warmed_up": warmup_results["success"],
                "endpoints_skipped": warmup_results["skipped"],
                "endpoints_failed": warmup_results["failed"],
                "total_ready": len(endpoint_managers),
                "endpoint_names": list(endpoint_managers.keys())
            },
            metadata={
                "current_step": "HuggingFace endpoint warm-up complete",
                "timestamp": datetime.utcnow().isoformat(),
                "warmup_summary": {
                    "success_count": len(warmup_results["success"]),
                    "skipped_count": len(warmup_results["skipped"]),
                    "failed_count": len(warmup_results["failed"])
                }
            }
        )

        # Update progress to 18% (warm-up complete, ready for processing)
        job_storage[job_id]["progress"] = 18
        job_storage[job_id]["metadata"] = {
            **job_storage[job_id].get("metadata", {}),
            "current_step": "Endpoints ready - starting PDF processing",
            "warmup_status": "complete",
            "endpoints_ready": list(endpoint_managers.keys()),
            "warmup_results": warmup_results
        }
        if job_recovery_service:
            await job_recovery_service.persist_job(
                job_id=job_id,
                document_id=document_id,
                filename=filename,
                status="processing",
                progress=18,
                metadata=job_storage[job_id]["metadata"]
            )
        logger.info(f"📊 Job {job_id} progress: 18% - Warm-up complete, {len(endpoint_managers)} endpoints ready")

        # ============================================================================
        # REGISTER WARMED-UP MANAGERS WITH SINGLETON REGISTRY
        # ============================================================================
        # This ensures all processing stages reuse the same warmed-up endpoint managers
        # instead of creating new ones (which would trigger repeated warmups)
        from app.services.embeddings.endpoint_registry import endpoint_registry
        endpoint_registry.register_endpoint_managers(endpoint_managers)
        logger.info(f"📌 Registered {len(endpoint_managers)} endpoint managers with singleton registry")

        # ============================================================================
        # HEALTH VALIDATION - Verify endpoints actually respond before processing
        # ============================================================================
        # Unlike warmup (which just waits), health checks verify actual inference works
        from app.services.embeddings.endpoint_health_checker import EndpointHealthChecker

        logger.info("=" * 80)
        logger.info("🔍 VALIDATING ENDPOINT HEALTH (inference-based)")
        logger.info("=" * 80)

        health_checker = EndpointHealthChecker(
            max_health_check_attempts=30,  # 30 attempts × 6 seconds = 180s max
            health_check_interval_seconds=6,
            health_check_timeout_seconds=30
        )

        # Build config for health checks based on what was warmed up
        endpoints_config = {}
        if 'qwen' in endpoint_managers:
            qwen_config = settings.get_qwen_config()
            endpoints_config['qwen'] = {
                'url': qwen_config['endpoint_url'],
                'token': qwen_config.get('endpoint_token', qwen_config.get('hf_token', ''))
            }
        if 'slig' in endpoint_managers:
            slig_config = settings.get_slig_config()
            endpoints_config['slig'] = {
                'url': slig_config['endpoint_url'],
                'token': slig_config['hf_token']
            }
        if 'yolo' in endpoint_managers:
            yolo_config = settings.get_yolo_config()
            endpoints_config['yolo'] = {
                'url': yolo_config['endpoint_url'],
                'token': yolo_config.get('hf_token', '')
            }
        if 'chandra' in endpoint_managers:
            chandra_config = settings.get_chandra_config()
            endpoints_config['chandra'] = {
                'url': chandra_config['endpoint_url'],
                'token': chandra_config.get('hf_token', '')
            }

        # SLIG is required for CLIP embeddings
        # Qwen is only required if NOT using Claude Vision for discovery
        # YOLO and Chandra are optional (layout detection enhancement)
        required_endpoints = []
        if 'qwen' in endpoints_config and discovery_model != 'claude-vision':
            # Only require Qwen when using it for discovery (not Claude Vision)
            required_endpoints.append('qwen')
        if 'slig' in endpoints_config:
            required_endpoints.append('slig')

        all_healthy, health_results = await health_checker.check_all_endpoints(
            endpoints_config=endpoints_config,
            required_endpoints=required_endpoints
        )

        # Store health results in registry
        endpoint_registry.set_health_validated(all_healthy, health_results)

        # ============================================================================
        # BLOCKING GATE - Stop processing if required endpoints are unhealthy
        # ============================================================================
        if not all_healthy:
            failed_endpoints = [
                name for name, result in health_results.items()
                if result.status.value != 'healthy' and name in required_endpoints
            ]

            if failed_endpoints:
                error_msg = f"Required endpoints failed health check: {failed_endpoints}"
                logger.error(f"❌ {error_msg}")
                logger.error("   Pipeline cannot proceed without healthy endpoints")

                # Update job status to failed (job_storage is a dict)
                job_storage[job_id]["status"] = "failed"
                job_storage[job_id]["error"] = error_msg
                job_storage[job_id]["failed_at"] = datetime.utcnow().isoformat()
                # Also persist to database
                if job_recovery_service:
                    await job_recovery_service.persist_job(
                        job_id=job_id,
                        document_id=document_id,
                        filename=filename,
                        status="failed",
                        progress=0,
                        error=error_msg
                    )
                raise RuntimeError(error_msg)
            else:
                # Only optional endpoints failed - log warning but proceed
                logger.warning("⚠️ Some optional endpoints unhealthy, proceeding with degraded functionality")

        logger.info("=" * 80)
        logger.info("✅ ALL REQUIRED ENDPOINTS HEALTHY - Ready to process")
        logger.info("=" * 80)

        # Mark processing as started (prevents auto-pause)
        endpoint_registry.start_processing(job_id)

        # ============================================================================
        # INITIALIZE PROGRESS TRACKING
        # ============================================================================
        logger.info(f"🔧 [BACKGROUND TASK] Initializing progress tracking components...")
        # Initialize Progress Tracker
        from app.services.tracking.progress_tracker import ProgressTracker
        # ProcessingStage already imported from checkpoint_recovery_service at module level (line 36)
        # checkpoint_recovery_service already imported at module level (line 36)
        from app.services.tracking.job_progress_monitor import JobProgressMonitor

        logger.info(f"🔧 [BACKGROUND TASK] Creating ProgressTracker...")
        tracker = ProgressTracker(
            job_id=job_id,
            document_id=document_id,
            total_pages=0,  # Will update after PDF extraction
            job_storage=job_storage
        )
        logger.info(f"🔧 [BACKGROUND TASK] Starting processing...")
        await tracker.start_processing()
        logger.info(f"✅ [BACKGROUND TASK] Processing started")

        logger.info(f"🔧 [BACKGROUND TASK] Starting heartbeat (30s interval)...")
        # 🫀 Start heartbeat monitoring (30s interval, 2min crash detection)
        await tracker.start_heartbeat(interval_seconds=30)
        logger.info(f"✅ [BACKGROUND TASK] Heartbeat started")

        logger.info(f"🔧 [BACKGROUND TASK] Starting progress monitor...")
        # 📊 Start detailed progress monitoring (reports every 60s to logs + Sentry)
        progress_monitor = JobProgressMonitor(job_id=job_id, document_id=document_id, total_stages=9)
        await progress_monitor.start()
        logger.info(f"✅ [BACKGROUND TASK] Progress monitoring started for job {job_id}")

        logger.info(f"🔧 [BACKGROUND TASK] Creating INITIALIZED checkpoint...")
        # Create INITIALIZED checkpoint
        await checkpoint_recovery_service.create_checkpoint(
            job_id=job_id,
            stage=ProcessingStage.INITIALIZED,
            data={
                "document_id": document_id,
                "filename": filename,
                "file_size": len(file_content),
                "pdf_path": file_path  # ✅ FIX: Add pdf_path for vision-guided extraction
            },
            metadata={
                "title": title or filename,
                "description": description,
                "tags": document_tags,
                "discovery_model": discovery_model,
                "focused_extraction": focused_extraction
            }
        )
        logger.info(f"✅ [BACKGROUND TASK] INITIALIZED checkpoint created for job {job_id}")
        logger.info(f"   📄 PDF path saved: {file_path}")

        # ============================================================================
        # STAGE 0: PRODUCT DISCOVERY (MODULAR)
        # ============================================================================
        logger.info(f"🚀 [BACKGROUND TASK] ========================================")
        logger.info(f"🚀 [BACKGROUND TASK] STARTING STAGE 0: PRODUCT DISCOVERY")
        logger.info(f"🚀 [BACKGROUND TASK] ========================================")
        progress_monitor.update_stage("product_discovery", {"discovery_model": discovery_model})
        from app.api.pdf_processing.stage_0_discovery import process_stage_0_discovery

        logger.info(f"🔧 [BACKGROUND TASK] Calling process_stage_0_discovery...")
        stage_0_result = await process_stage_0_discovery(
            file_content=file_content,
            document_id=document_id,
            workspace_id=workspace_id,
            job_id=job_id,
            filename=filename,
            title=title,
            description=description,
            extract_categories=extract_categories,
            discovery_model=discovery_model,
            agent_prompt=agent_prompt,
            enable_prompt_enhancement=enable_prompt_enhancement,
            tracker=tracker,
            checkpoint_recovery_service=checkpoint_recovery_service,
            logger=logger,
            temp_pdf_path=file_path,  # Use existing temp path
            test_single_product=test_single_product  # 🧪 TEST MODE flag
        )

        catalog = stage_0_result["catalog"]
        page_count = stage_0_result["page_count"]
        file_size_mb = stage_0_result["file_size_mb"]
        temp_pdf_path = stage_0_result["temp_pdf_path"]

        # ============================================================================
        # PRODUCT-CENTRIC PIPELINE: Process each product individually
        # ============================================================================
        logger.info(f"\n{'='*80}")
        logger.info(f"🏭 PRODUCT-CENTRIC PIPELINE: Processing {len(catalog.products)} products")
        logger.info(f"{'='*80}\n")

        # Initialize product progress tracker
        from app.services.tracking.product_progress_tracker import ProductProgressTracker
        from app.api.pdf_processing.product_processor import process_single_product
        from app.api.pdf_processing.parallel_product_processor import (
            process_products_parallel,
            ParallelProcessingConfig
        )

        product_tracker = ProductProgressTracker(job_id=job_id)
        supabase = get_supabase_client()

        # ========================================================================
        # SHARED RESOURCES (preserved across all products)
        # ========================================================================
        # These objects are reused for ALL products to minimize memory:
        # - file_content: Original PDF bytes (read once, used for all products)
        # - catalog: Product discovery results (contains all products)
        # - temp_pdf_path: Temporary PDF file on disk (from stage 0)
        # - tracker: Main job progress tracker
        # - product_tracker: Per-product progress tracker
        # - supabase: Database client (connection pooling)
        # - processing_config: Configuration settings
        # ========================================================================

        # Fetch material_category from document or job metadata
        # This is used for proper image categorization (tiles, heatpump, wood, etc.)
        material_category = None
        try:
            # Try to get from background_jobs metadata first
            job_result = supabase.client.table('background_jobs').select('metadata').eq('id', job_id).execute()
            if job_result.data and len(job_result.data) > 0:
                job_metadata = job_result.data[0].get('metadata', {})
                material_category = job_metadata.get('material_category')
                if material_category:
                    logger.info(f"📦 Using material_category from job metadata: {material_category}")

            # Fallback: try to get from documents metadata
            if not material_category:
                doc_result = supabase.client.table('documents').select('metadata').eq('id', document_id).execute()
                if doc_result.data and len(doc_result.data) > 0:
                    doc_metadata = doc_result.data[0].get('metadata', {})
                    material_category = doc_metadata.get('material_category')
                    if material_category:
                        logger.info(f"📦 Using material_category from document metadata: {material_category}")
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch material_category: {e}")

        # Processing configuration
        processing_config = {
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'image_analysis_model': image_analysis_model,
            'discovery_model': discovery_model,
            'focused_extraction': focused_extraction,
            'extract_categories': extract_categories,
            'material_category': material_category  # For image categorization
        }

        # Track overall metrics
        total_products = len(catalog.products)

        # 🧪 TEST MODE: Process only first product if test_single_product=True
        if test_single_product:
            logger.warning("=" * 80)
            logger.warning("🧪 TEST MODE ENABLED: Processing ONLY the first product")
            logger.warning("   This is for testing/debugging purposes only")
            logger.warning("   Set test_single_product=False to process all products")
            logger.warning("=" * 80)
            products_to_process = catalog.products[:1]  # Only first product
        else:
            products_to_process = catalog.products  # All products

        # ========================================================================
        # PARALLEL PRODUCT PROCESSING
        # ========================================================================
        # Uses controlled concurrency to process 2-3 products simultaneously:
        # - Semaphore limits concurrent processing to prevent resource exhaustion
        # - Each product still goes through all stages (extraction, chunking, images, etc.)
        # - Shared resources (file_content, catalog, tracker) are safely accessed
        # - Falls back to sequential processing for small catalogs (<=2 products)
        # ========================================================================

        # Configure parallel processing (can be adjusted based on server resources)
        parallel_config = ParallelProcessingConfig(
            max_concurrent=2,  # Process 2 products concurrently
            enable_parallel=len(products_to_process) > 2  # Only parallelize if >2 products
        )

        logger.info(f"🚀 Processing mode: {'PARALLEL' if parallel_config.enable_parallel else 'SEQUENTIAL'}")
        if parallel_config.enable_parallel:
            logger.info(f"   Max concurrent products: {parallel_config.max_concurrent}")

        # Process all products (parallel or sequential based on config)
        parallel_result = await process_products_parallel(
            products=products_to_process,
            file_content=file_content,  # SHARED: Reused for all products
            document_id=document_id,
            workspace_id=workspace_id,
            job_id=job_id,
            catalog=catalog,  # SHARED: Contains all products
            tracker=tracker,  # SHARED: Main job tracker
            product_tracker=product_tracker,  # SHARED: Product progress tracker
            checkpoint_recovery_service=checkpoint_recovery_service,
            supabase=supabase,  # SHARED: Database client
            config=processing_config,  # SHARED: Configuration
            logger_instance=logger,
            total_pages=page_count,
            temp_pdf_path=file_path,
            parallel_config=parallel_config
        )

        # Extract metrics from parallel result
        products_completed = parallel_result.products_completed
        products_failed = parallel_result.products_failed
        total_chunks_created = parallel_result.total_chunks_created
        total_images_processed = parallel_result.total_images_processed
        total_relationships_created = parallel_result.total_relationships_created
        total_clip_embeddings = parallel_result.total_clip_embeddings

        # Summary
        logger.info(f"\n{'='*80}")
        logger.info(f"🏭 PRODUCT-CENTRIC PIPELINE COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"✅ Products completed: {products_completed}/{total_products}")
        logger.info(f"❌ Products failed: {products_failed}/{total_products}")
        logger.info(f"📝 Total chunks created: {total_chunks_created}")
        logger.info(f"🖼️  Total images processed: {total_images_processed}")
        logger.info(f"🎨 Total CLIP embeddings: {total_clip_embeddings}")
        logger.info(f"🔗 Total relationships created: {total_relationships_created}")
        logger.info(f"⏱️  Processing time: {parallel_result.processing_time_seconds:.1f}s")
        logger.info(f"{'='*80}\n")

        # Update tracker with final counts
        tracker.chunks_created = total_chunks_created
        products_created = products_completed
        images_saved_count = total_images_processed
        linking_results = {"relationships_created": total_relationships_created}

        # In product-centric pipeline, each product has its own page_range
        # We need to collect all physical pages that were processed across all products
        # Physical pages are 1-based page numbers that users see in catalogs
        all_physical_pages = set()
        for product in catalog.products:
            if hasattr(product, 'page_range') and product.page_range:
                all_physical_pages.update(product.page_range)

        logger.info(f"📄 Aggregated {len(all_physical_pages)} unique physical pages from {len(catalog.products)} products")

        # ============================================================================
        # STAGE 4.5: PROPAGATE COMMON FIELDS ACROSS PRODUCTS
        # ============================================================================
        # Shares factory, manufacturing, material_category, available_sizes, and nested
        # material_properties fields (thickness, body_type, composition) across all
        # products from the same catalog PDF.
        progress_monitor.update_stage("field_propagation", {
            "products_created": products_created,
            "description": "Propagating shared fields across catalog siblings"
        })
        await tracker.update_progress(70, {
            "current_step": "Propagating common fields across catalog siblings"
        })

        from app.api.pdf_processing.stage_4_products import (
            propagate_common_fields_to_products,
            extract_dimensions_from_document_chunks,
        )

        propagation_result = await propagate_common_fields_to_products(
            document_id=document_id,
            supabase=supabase,
            logger=logger,
            material_category_override=material_category  # From upload settings
        )
        logger.info(
            f"🔄 Field propagation: "
            f"{propagation_result.get('products_updated', 0)}/{propagation_result.get('products_checked', 0)} "
            f"products updated — fields: {propagation_result.get('fields_propagated', [])}"
        )

        await tracker.update_progress(72, {
            "current_step": (
                f"Field propagation done — "
                f"{propagation_result.get('products_updated', 0)} products updated"
            )
        })

        # ============================================================================
        # STAGE 4.6: EXTRACT DIMENSIONS FROM DOCUMENT TEXT CHUNKS
        # ============================================================================
        # For products still missing sizes / thickness after sibling propagation,
        # scan the already-extracted text chunks for dimension patterns (regex, no AI call).
        progress_monitor.update_stage("dimension_extraction", {
            "products_checked": propagation_result.get('products_checked', 0),
            "description": "Scanning text chunks for tile sizes and thickness"
        })
        await tracker.update_progress(74, {
            "current_step": "Extracting dimensions from catalog text chunks"
        })

        dimension_result = await extract_dimensions_from_document_chunks(
            document_id=document_id,
            supabase=supabase,
            logger=logger,
        )
        logger.info(
            f"📐 Dimension extraction from text: "
            f"{dimension_result.get('products_updated', 0)}/{dimension_result.get('products_checked', 0)} "
            f"products updated — values: {dimension_result.get('dimensions_found', [])}"
        )

        await tracker.update_progress(76, {
            "current_step": (
                f"Dimension extraction done — "
                f"{dimension_result.get('products_updated', 0)} products updated"
            )
        })
        await tracker._sync_to_database(stage="dimension_extraction")

        # ── Factory enrichment trigger (async, non-blocking) ─────────────────
        try:
            from app.api.pdf_processing.stage_4_products import _trigger_factory_enrichment
            all_product_ids = [p['id'] for p in (
                supabase.client.table('products')
                .select('id')
                .eq('source_document_id', document_id)
                .execute()
            ).data or []]
            if all_product_ids:
                import asyncio
                _fe_task = asyncio.create_task(_trigger_factory_enrichment(
                    workspace_id=workspace_id,
                    product_ids=all_product_ids,
                    scope_column='source_document_id',
                    scope_value=document_id,
                    logger=logger,
                ))
                _fe_task.add_done_callback(lambda t: logger.error(
                    f"❌ Factory enrichment task failed: {t.exception()}",
                    exc_info=t.exception(),
                ) if not t.cancelled() and t.exception() else None)
        except Exception as _fe:
            logger.warning(f"⚠️ Factory enrichment trigger failed (non-blocking): {_fe}")

        # ============================================================================
        # STAGE 5: QUALITY ENHANCEMENT (MODULAR)
        # ============================================================================
        progress_monitor.update_stage("quality_enhancement", {"products_created": products_created})
        from app.api.pdf_processing.stage_5_quality import process_stage_5_quality
        from app.utils.circuit_breaker import claude_breaker

        stage_5_result = await process_stage_5_quality(
            document_id=document_id,
            job_id=job_id,
            workspace_id=workspace_id,
            catalog=catalog,
            physical_pages=list(all_physical_pages),  # ✅ FIXED: Now using physical pages
            products_created=products_created,
            images_processed=images_saved_count,
            focused_extraction=focused_extraction,
            quality_validation_model=quality_validation_model,
            start_time=start_time,
            tracker=tracker,
            checkpoint_recovery_service=checkpoint_recovery_service,
            component_manager=component_manager,
            loaded_components=loaded_components,
            claude_breaker=claude_breaker,
            logger=logger
        )

        # Stage 5 handles all SUCCESS cleanup (component unloading, resource cleanup, job completion)
        logger.info("✅ [MODULAR PIPELINE] All stages completed successfully")

        # Calculate total processing time
        end_time = datetime.utcnow()
        total_duration = (end_time - start_time).total_seconds()

        # Send success event to Sentry with comprehensive metrics
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("job_id", job_id)
            scope.set_tag("document_id", document_id)
            scope.set_tag("filename", filename)
            scope.set_tag("discovery_model", discovery_model)
            scope.set_tag("status", "completed")
            scope.set_tag("duration_minutes", round(total_duration / 60, 2))

            # Get final metrics from tracker if available
            final_metrics = {}
            if 'tracker' in locals():
                try:
                    final_metrics = {
                        "total_duration_seconds": total_duration,
                        "total_duration_minutes": round(total_duration / 60, 2),
                        "stages_completed": len(progress_monitor.stage_history) if 'progress_monitor' in locals() else 0,
                        "completed_at": end_time.isoformat()
                    }
                except Exception:
                    pass

            scope.set_context("completion_metrics", final_metrics)

            sentry_sdk.capture_message(
                f"✅ PDF Processing Completed: {filename} (Job: {job_id}) in {total_duration/60:.1f} minutes",
                level="info"
            )

        # Stop progress monitoring
        progress_monitor.update_stage("completed", {"success": True})
        await progress_monitor.stop()
        logger.info("✅ Stopped progress monitoring")

        # Mark processing complete and clear endpoint registry
        from app.services.embeddings.endpoint_registry import endpoint_registry
        endpoint_registry.end_processing(job_id)
        logger.info(f"🔓 Processing ended for job {job_id}")
        endpoint_registry.clear_all()
        logger.info("🗑️ Cleared endpoint registry")

        # Clear singleton PDFProcessor
        from app.api.pdf_processing.stage_3_images import clear_pdf_processor
        clear_pdf_processor()
        logger.info("🗑️ Cleared singleton PDFProcessor")

    except Exception as e:
        logger.error(f"❌ [PRODUCT DISCOVERY PIPELINE] FAILED: {e}", exc_info=True)

        # Send detailed error to Sentry
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("job_id", job_id)
            scope.set_tag("document_id", document_id)
            scope.set_tag("filename", filename)
            scope.set_tag("discovery_model", discovery_model)
            scope.set_tag("error_type", type(e).__name__)

            # Add context about where the error occurred
            current_stage = "unknown"
            if 'progress_monitor' in locals():
                current_stage = progress_monitor.current_stage
                scope.set_tag("failed_stage", current_stage)
                scope.set_context("stage_history", {
                    "stages_completed": len(progress_monitor.stage_history),
                    "current_stage": current_stage,
                    "stage_history": progress_monitor.stage_history[-5:]
                })

            scope.set_context("error_details", {
                "error_message": str(e),
                "error_type": type(e).__name__,
                "job_id": job_id,
                "document_id": document_id,
                "filename": filename,
                "failed_at": datetime.utcnow().isoformat()
            })

            # Capture the exception with full context
            sentry_sdk.capture_exception(e)

            # Also send a message for easier filtering
            sentry_sdk.capture_message(
                f"❌ PDF Processing Failed: {filename} at stage {current_stage} - {type(e).__name__}: {str(e)}",
                level="error"
            )

        # Stop progress monitoring on error
        if 'progress_monitor' in locals():
            progress_monitor.update_stage("failed", {"error": str(e)})
            await progress_monitor.stop()
            logger.info("✅ Stopped progress monitoring (error)")

        # Mark job as failed using tracker
        if 'tracker' in locals():
            await tracker.fail_job(error=e)

        # Rollback products created during discovery if processing failed
        # This prevents orphan products in the database
        try:
            from app.services.utilities.cleanup_service import CleanupService
            cleanup_service = CleanupService()
            rollback_stats = await cleanup_service.rollback_discovered_products(
                document_id=document_id,
                product_db_ids=None,  # Delete all products for this document
                supabase_client=None  # Will get from singleton
            )
            logger.info(f"🔄 Product rollback completed: {rollback_stats}")
        except Exception as rollback_error:
            logger.error(f"⚠️ Product rollback failed: {rollback_error}")
            sentry_sdk.capture_exception(rollback_error)

    finally:
        # ============================================================================
        # CONSOLIDATED CLEANUP (Failure or Success)
        # ============================================================================
        logger.info("🧹 [CLEANUP] Starting comprehensive pipeline cleanup...")

        # 0. End processing lock (allow auto-pause, even on error)
        try:
            from app.services.embeddings.endpoint_registry import endpoint_registry
            if endpoint_registry.is_processing():
                endpoint_registry.end_processing(job_id)
                logger.info(f"🔓 [CLEANUP] Processing lock released for job {job_id}")
        except Exception as lock_error:
            logger.warning(f"⚠️ Failed to release processing lock: {lock_error}")

        # 1. Stop progress monitoring (if still running)
        if 'progress_monitor' in locals():
            try:
                await progress_monitor.stop()
            except Exception:
                pass

        # 2. Unload lazy components
        if 'component_manager' in locals() and 'loaded_components' in locals():
            for component_name in loaded_components:
                try:
                    await component_manager.unload(component_name)
                    logger.info(f"   ✅ Unloaded {component_name}")
                except Exception as unload_error:
                    logger.warning(f"   ⚠️ Failed to unload {component_name}: {unload_error}")

        # 3. Release resources & delete temp files
        try:
            from app.utils.resource_manager import get_resource_manager
            resource_manager = get_resource_manager()
            
            # Release the main temp PDF
            await resource_manager.release_resource(f"temp_pdf_{document_id}", job_id)
            
            # Cleanup all ready resources (this also handles os.unlink for us if registered)
            cleaned_count = await resource_manager.cleanup_ready_resources()
            logger.info(f"   ✅ Cleaned up {cleaned_count} temporary resources")
        except Exception as cleanup_error:
            logger.warning(f"   ⚠️ Resource cleanup failed: {cleanup_error}")

        # 4. Scale HuggingFace endpoints to zero (Stop billing) - SINGLE LOCATION for all endpoint scaling
        # NOTE: Using scale_to_zero() instead of force_pause() allows auto-resume on next request
        logger.info("📉 [CLEANUP] Scaling AI endpoints to zero...")
        endpoints_to_scale = ['qwen', 'slig', 'yolo', 'chandra']
        scaled_count = 0

        # Only process if endpoint_managers exists (may not if job failed early)
        if 'endpoint_managers' not in locals():
            endpoint_managers = {}

        for name in endpoints_to_scale:
            if name in endpoint_managers:
                try:
                    if endpoint_managers[name].scale_to_zero():
                        scaled_count += 1
                        logger.info(f"   ✅ {name.upper()} endpoint scaled to zero")
                except Exception as scale_error:
                    logger.warning(f"   ⚠️ Failed to scale {name} endpoint to zero: {scale_error}")

        # 5. Clear endpoint registry (cleanup singleton state)
        try:
            endpoint_registry.clear_all()
            logger.info("   ✅ Endpoint registry cleared")
        except Exception as registry_error:
            logger.warning(f"   ⚠️ Failed to clear endpoint registry: {registry_error}")

        # 6. Final Garbage Collection
        gc.collect()
        logger.info(f"✨ [CLEANUP] Finished. Endpoints scaled to zero: {scaled_count}")


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    **🤖 CONSOLIDATED QUERY ENDPOINT - Text-Based RAG Query**

    This endpoint replaces:
    - `/api/documents/{id}/query` → Use with `document_ids` filter
    - `/api/documents/{id}/summarize` → Use with summarization prompt

    ## 🎯 Query Capabilities

    ### Text Query (Implemented) ✅
    - Pure text-based RAG with advanced retrieval
    - Semantic search with reranking
    - Best for: Factual questions, information retrieval, summarization

    ## 📝 Examples

    ### Text Query (Default)
    ```bash
    curl -X POST "/api/rag/query" \\
      -H "Content-Type: application/json" \\
      -d '{
        "query": "What are the dimensions of the NOVA product?",
        "top_k": 5
      }'
    ```

    ### Document-Specific Query
    ```bash
    curl -X POST "/api/rag/query" \\
      -H "Content-Type: application/json" \\
      -d '{
        "query": "Summarize this document",
        "document_ids": ["doc-123"],
        "top_k": 20
      }'
    ```

    ## 🔄 Migration from Old Endpoints

    **Old:** `POST /api/documents/{id}/query`
    **New:** `POST /api/rag/query` with `document_ids` filter

    **Old:** `POST /api/documents/{id}/summarize`
    **New:** `POST /api/rag/query` with summarization prompt
    """
    start_time = datetime.utcnow()

    try:
        # Advanced RAG query using Claude 4.5
        result = await rag_service.advanced_rag_query(
            query=request.query,
            workspace_id=request.workspace_id,
            document_ids=getattr(request, 'document_ids', None),
            max_results=request.top_k,
            similarity_threshold=request.similarity_threshold,
            enable_reranking=request.enable_reranking,
            query_type="factual"
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        return QueryResponse(
            query=request.query,
            answer=result.get('response', ''),
            sources=result.get('sources', []),
            confidence_score=result.get('confidence_score', 0.0),
            processing_time=processing_time,
            retrieved_chunks=len(result.get('sources', []))
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )

@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service)
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

        # Build conversation context from history
        conversation_context = None
        if hasattr(request, 'conversation_history') and request.conversation_history:
            conversation_context = request.conversation_history

        # Process chat message using advanced_rag_query with Claude 4.5
        result = await rag_service.advanced_rag_query(
            query=request.message,
            workspace_id=request.workspace_id,
            max_results=request.top_k,
            query_type="conversational",
            conversation_context=conversation_context
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


async def _enhance_search_results(
    results: List[Dict[str, Any]],
    workspace_id: str,
    include_related_products: bool = True,
    related_products_limit: int = 3
) -> List[Dict[str, Any]]:
    """
    Enhance search results with related products and images.

    Args:
        results: Raw search results
        workspace_id: Workspace ID for scoped queries
        include_related_products: Whether to include related products
        related_products_limit: Max related products per result

    Returns:
        Enhanced results with related products and images
    """
    try:
        supabase_client = get_supabase_client()
        product_rel_service = ProductRelationshipService(supabase_client=supabase_client.client)

        enhanced = []

        for result in results:
            # Get product ID from result
            product_id = result.get('id')
            if not product_id:
                enhanced.append(result)
                continue

            # Fetch related images
            try:
                # ✅ UPDATED: Use image_product_associations instead of product_image_relationships
                images_response = supabase_client.table('image_product_associations').select(
                    'id, image_id, reasoning, overall_score, document_images(id, image_url, caption)'
                ).eq('product_id', product_id).order('overall_score', desc=True).limit(10).execute()

                related_images = []
                for img_rel in images_response.data or []:
                    if img_rel.get('document_images'):
                        related_images.append({
                            'id': img_rel['document_images']['id'],
                            'url': img_rel['document_images']['image_url'],
                            'relationship_type': img_rel.get('reasoning', 'related'),  # reasoning replaces relationship_type
                            'relevance_score': img_rel.get('overall_score', 0.0),  # overall_score replaces relevance_score
                            'caption': img_rel['document_images'].get('caption')
                        })

                result['related_images'] = related_images
            except Exception as e:
                logger.warning(f"Failed to fetch related images for product {product_id}: {e}")
                result['related_images'] = []

            # Fetch related products
            if include_related_products:
                try:
                    related_products = await product_rel_service.find_related_products(
                        product_id=product_id,
                        workspace_id=workspace_id,
                        limit=related_products_limit
                    )
                    result['related_products'] = related_products
                except Exception as e:
                    logger.warning(f"Failed to fetch related products for product {product_id}: {e}")
                    result['related_products'] = []
            else:
                result['related_products'] = []

            # Ensure all metadata is included
            if 'metadata' not in result:
                result['metadata'] = {}

            enhanced.append(result)

        return enhanced

    except Exception as e:
        logger.error(f"Error enhancing search results: {e}", exc_info=True)
        # Return original results if enhancement fails
        return results


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    strategy: Optional[str] = Query(
        "multi_vector",
        description="Search strategy: 'multi_vector' (default and only supported strategy)"
    ),
    enable_query_understanding: bool = Query(
        True,  # ✅ ENABLED BY DEFAULT - Makes platform smarter with minimal cost ($0.0001/query)
        description="🧠 AI query parsing to automatically extract filters from natural language (e.g., 'waterproof ceramic tiles for outdoor patio, matte finish' → auto-extracts material_type, properties, finish, etc.). Set to false to disable."
    ),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    **🔍 SEARCH ENDPOINT - Multi-Vector Search with AI Query Understanding**

    ## 🎯 Supported Search Strategies

    ### Multi-Vector Search (`strategy="multi_vector"`) - ⭐ DEFAULT & RECOMMENDED ✅
    - 🎯 **ENHANCED**: 7-vector fusion search + JSONB metadata filtering
    - **Embeddings Combined:**
      - Text (15%) - Voyage AI 1024D semantic understanding
      - Visual (15%) - SLIG 768D visual similarity
      - Understanding (20%) - Voyage AI 1024D from Qwen3-VL vision analysis
      - Color (12.5%) - SLIG 1152D color palette matching
      - Texture (12.5%) - SLIG 1152D texture pattern matching
      - Style (12.5%) - SLIG 1152D design style matching
      - Material (12.5%) - SLIG 1152D material type matching
    - **+ JSONB Metadata Filtering**: Supports `material_filters` for property-based filtering
    - **+ Query Understanding**: ✅ **ENABLED BY DEFAULT** - Auto-extracts filters from natural language
    - **Performance**: Fast (~250-350ms with query understanding, ~200-300ms without)
    - **Best For:** ALL queries - comprehensive, accurate, fast
    - **Example:** "waterproof ceramic tiles for outdoor patio, matte finish"

    ### Material Property Search (`strategy="material"`) ✅
    - JSONB-based filtering with AND/OR logic
    - Requires `material_filters` in request body
    - Best for: Filtering by specific material properties
    - Uses direct database queries (no LLM required)

    ### Image Similarity Search (`strategy="image"`) ✅
    - Visual similarity using CLIP embeddings
    - Requires `image_url` or `image_base64` in request body
    - Best for: Finding visually similar products
    - Uses VECS vector database with HNSW indexing



    ## 📝 Examples

    ### Multi-Vector Search (⭐ DEFAULT - Recommended for all queries)
    ```bash
    curl -X POST "/api/rag/search" \\
      -H "Content-Type: application/json" \\
      -d '{"query": "modern minimalist furniture", "workspace_id": "xxx", "top_k": 10}'
    ```

    ### Multi-Vector with Natural Language Filters
    ```bash
    curl -X POST "/api/rag/search" \\
      -H "Content-Type: application/json" \\
      -d '{"query": "waterproof ceramic tiles for outdoor patio, matte finish", "workspace_id": "xxx", "top_k": 10}'
    # AI automatically extracts: material_type=ceramic, properties=waterproof, application=outdoor, finish=matte
    ```

    ### Material Property Search
    ```bash
    curl -X POST "/api/rag/search?strategy=material" \\
      -H "Content-Type: application/json" \\
      -d '{"workspace_id": "xxx", "material_filters": {"material_type": "fabric", "color": ["red", "blue"]}, "top_k": 10}'
    ```

    ### Image Similarity Search
    ```bash
    curl -X POST "/api/rag/search?strategy=image" \\
      -H "Content-Type: application/json" \\
      -d '{"workspace_id": "xxx", "image_url": "https://example.com/image.jpg", "top_k": 10}'
    ```

    ## 📊 Response Example
    ```json
    {
      "query": "modern oak furniture",
      "enhanced_query": "modern oak furniture",
      "results": [
        {
          "id": "product_uuid_1",
          "name": "Modern Oak Dining Table",
          "description": "Contemporary oak furniture...",
          "score": 0.92,
          "final_score": 0.85,
          "strategy_count": 4,
          "strategies": ["semantic", "vector", "multi_vector", "hybrid"]
        }
      ],
      "total_results": 10,
      "search_type": "all",
      "processing_time": 0.223,
      "search_metadata": {
        "strategies_executed": 4,
        "strategies_successful": 4,
        "strategies_failed": 0,
        "strategy_breakdown": {
          "semantic": {"count": 3, "success": true},
          "vector": {"count": 2, "success": true},
          "multi_vector": {"count": 4, "success": true},
          "hybrid": {"count": 5, "success": true}
        },
        "parallel_execution": true,
        "parallel_processing_time": 0.017
      }
    }
    ```

    ## ⚡ Performance Characteristics

    | Strategy | Typical Time | Max Time | Notes |
    |----------|-------------|----------|-------|
    | semantic | 100-150ms | 300ms | Indexed, MMR diversity |
    | vector | 50-100ms | 200ms | Fastest, pure similarity |
    | multi_vector | 200-300ms | 500ms | 3 embeddings, sequential scan for 2048-dim |
    | hybrid | 120-180ms | 350ms | Semantic + full-text search |
    | material | 30-50ms | 100ms | JSONB indexed |
    | image | 100-150ms | 300ms | CLIP indexed |
    | **all (parallel)** | **200-300ms** | **500ms** | **3-4x faster than sequential** |

    ## 🔄 Migration from Old Endpoints

    **Old:** `POST /api/search/semantic`
    **New:** `POST /api/rag/search?strategy=semantic`

    **Old:** `POST /api/search/similarity`
    **New:** `POST /api/rag/search?strategy=vector`

    **Old:** `POST /api/unified-search`
    **New:** `POST /api/rag/search` (same functionality, clearer naming)

    ## ⚠️ Error Codes

    - **400 Bad Request**: Invalid parameters (missing query, invalid strategy, etc.)
    - **401 Unauthorized**: Missing or invalid authentication
    - **404 Not Found**: Workspace not found
    - **500 Internal Server Error**: Search processing failed
    - **503 Service Unavailable**: RAG service not available

    ## 🎯 Rate Limits

    - **60 requests/minute** per user
    - **1000 requests/hour** per workspace
    - Parallel execution (`strategy="all"`) counts as 1 request
    """
    start_time = datetime.utcnow()

    try:
        # Validate strategy
        valid_strategies = ['multi_vector', 'material', 'image']
        if strategy not in valid_strategies:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid strategy '{strategy}'. Valid strategies: {', '.join(valid_strategies)}"
            )

        # Initialize services
        supabase_client = get_supabase_client()
        search_prompt_service = SearchPromptService(supabase_client=supabase_client.client)
        product_rel_service = ProductRelationshipService(supabase_client=supabase_client.client)

        # Apply enhancement prompt to query if enabled
        query_to_use = request.query
        enhanced_query = None
        prompts_applied = []

        if request.use_search_prompts:
            enhancement_result = await search_prompt_service.enhance_query(
                query=request.query,
                workspace_id=request.workspace_id,
                custom_prompt=request.custom_formatting_prompt
            )
            if enhancement_result.get('enhancement_applied'):
                query_to_use = enhancement_result['enhanced_query']
                enhanced_query = query_to_use
                prompts_applied.extend(enhancement_result.get('prompts_applied', []))

        # 🧠 STEP 1: Query Understanding (if enabled)
        # Parse natural language query to extract structured filters + dynamic weight profile
        parsed_filters = {}
        dynamic_weights = None
        weight_profile = "balanced"
        if enable_query_understanding:
            try:
                import asyncio
                from app.services.search.unified_search_service import UnifiedSearchService

                # Create temporary service instance for query parsing
                unified_service = UnifiedSearchService()
                visual_query, parsed_filters, weight_profile, dynamic_weights = await asyncio.wait_for(
                    unified_service._parse_query_with_ai(query_to_use),
                    timeout=8  # 8s timeout — don't let query understanding block search
                )

                # Update query to use visual query (core concept for embedding)
                query_to_use = visual_query

                # Merge parsed filters with existing material_filters (user filters take precedence)
                existing_filters = getattr(request, 'material_filters', {})
                if existing_filters:
                    # User-provided filters override AI-parsed filters
                    merged_filters = {**parsed_filters, **existing_filters}
                else:
                    merged_filters = parsed_filters

                # Update request with merged filters
                if merged_filters:
                    request.material_filters = merged_filters

                logger.info(f"🧠 Query understanding: '{request.query}' → visual_query='{visual_query}', profile='{weight_profile}', filters={parsed_filters}")

            except asyncio.TimeoutError:
                logger.warning(f"Query understanding timed out after 8s, continuing with original query")
            except Exception as e:
                logger.error(f"Query understanding failed: {e}, continuing with original query")
                # Continue with original query if parsing fails

        # 🔍 STEP 2: Route to appropriate search method based on strategy
        # All strategies now use the parsed query + extracted filters + dynamic weights
        if strategy == "multi_vector":
            # 🎯 Enhanced multi-vector search with dynamic weight profiles
            # Map 7-vector profile to RAG service's 9-source format
            rag_weights = None
            if dynamic_weights:
                text_w = dynamic_weights.get("text", 0.15)
                rag_weights = {
                    "visual": dynamic_weights.get("visual", 0.15),
                    "chunk": text_w * 0.40,
                    "understanding": dynamic_weights.get("understanding", 0.20),
                    "product": text_w * 0.35,
                    "keyword": text_w * 0.25,
                    "color": dynamic_weights.get("color", 0.125),
                    "texture": dynamic_weights.get("texture", 0.125),
                    "style": dynamic_weights.get("style", 0.125),
                    "material": dynamic_weights.get("material", 0.125),
                }
            material_filters = getattr(request, 'material_filters', None)
            results = await rag_service.multi_vector_search(
                query=query_to_use,
                workspace_id=request.workspace_id,
                top_k=request.top_k,
                material_filters=material_filters,
                search_config={"weights": rag_weights} if rag_weights else None,
                image_base64=getattr(request, 'image_base64', None),
            )

        elif strategy == "material":
            # Material property search using JSONB filtering
            # Requires material_filters in request
            material_filters = getattr(request, 'material_filters', {})
            if not material_filters:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="material_filters required for material property search"
                )
            results = await rag_service.material_property_search(
                workspace_id=request.workspace_id,
                material_filters=material_filters,
                top_k=request.top_k
            )

        elif strategy == "image":
            # Image similarity search using visual embeddings (SigLIP/CLIP)
            # Requires image_url or image_base64 in request
            image_url = getattr(request, 'image_url', None)
            image_base64 = getattr(request, 'image_base64', None)
            if not image_url and not image_base64:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="image_url or image_base64 required for image similarity search"
                )
            results = await rag_service.image_similarity_search(
                workspace_id=request.workspace_id,
                image_url=image_url,
                image_base64=image_base64,
                top_k=request.top_k
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid strategy '{strategy}'. Valid strategies: multi_vector, material, image"
            )

        # Get raw results
        raw_results = results.get('results', [])

        # Apply formatting, filtering, enrichment prompts if enabled
        processed_results = raw_results
        if request.use_search_prompts:
            processed_results = await search_prompt_service.format_results(
                raw_results, request.workspace_id, request.custom_formatting_prompt
            )
            processed_results = await search_prompt_service.filter_results(
                processed_results, request.workspace_id
            )
            processed_results = await search_prompt_service.enrich_results(
                processed_results, request.workspace_id
            )

        # Enhance results with related products and images
        enhanced_results = await _enhance_search_results(
            processed_results,
            request.workspace_id,
            request.include_related_products,
            request.related_products_limit
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Build search metadata
        search_metadata = {
            'prompts_applied': prompts_applied,
            'prompts_enabled': request.use_search_prompts,
            'related_products_included': request.include_related_products,
            'weight_profile': weight_profile,
            'dynamic_weights': dynamic_weights,
        }

        # Add parallel execution metadata for 'all' strategy
        if strategy == "all":
            search_metadata.update({
                'strategies_executed': results.get('strategies_executed', 0),
                'strategies_successful': results.get('strategies_successful', 0),
                'strategies_failed': results.get('strategies_failed', 0),
                'strategy_breakdown': results.get('strategy_breakdown', {}),
                'parallel_execution': True,
                'parallel_processing_time': results.get('processing_time', 0)
            })

        return SearchResponse(
            query=request.query,
            enhanced_query=enhanced_query,
            results=enhanced_results,
            total_results=results.get('total_results', 0),
            search_type=strategy,
            processing_time=processing_time,
            search_metadata=search_metadata
        )

    except HTTPException:
        raise
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
        clip_embeddings = sum(1 for img in result['images'] if img.get('visual_clip_embedding_512'))
        vision_analysis = sum(1 for img in result['images'] if img.get('vision_analysis'))
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
                "vision_calls": vision_analysis,
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




@router.get("/health", response_model=HealthCheckResponse)
async def rag_health_check(
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Health check for RAG services.

    This endpoint checks the health of all RAG-related services
    including embedding service and vector store.
    """
    try:
        # Check RAG service health
        rag_health = await rag_service.health_check()

        # Determine overall status
        overall_status = "healthy"
        if rag_health.get("status") != "healthy":
            overall_status = "degraded"

        return HealthCheckResponse(
            status=overall_status,
            services={
                "rag_service": rag_health,
                "service_type": "Direct Vector DB"
            },
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"RAG health check failed: {e}", exc_info=True)
        return HealthCheckResponse(
            status="unhealthy",
            services={
                "rag_service": {"status": "error", "error": str(e)}
            },
            timestamp=datetime.utcnow().isoformat()
        )

@router.get("/stats")
async def get_rag_statistics(
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Get RAG system statistics.

    This endpoint provides statistics about the RAG system including
    document counts, embedding statistics, and performance metrics.
    """
    try:
        # Get health check from RAG service
        health_check = await rag_service.health_check()

        # Combine statistics
        stats = {
            "health": health_check,
            "service_type": "RAG Service",
            "search_capabilities": [
                "multi_vector_search",
                "material_property_search",
                "image_similarity_search",
                "query_document",
                "advanced_rag_query"
            ],
            "ai_models": {
                "embeddings": "SigLIP-SO400M-14-384 / CLIP",
                "rag_synthesis": "Claude 4.5 Sonnet",
                "vision": "Qwen3-VL-32B-Instruct"
            }
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

@router.get("/workspace-stats")
async def get_workspace_statistics(
    workspace_id: str,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Get comprehensive workspace statistics including VECS embedding counts.

    Returns counts for:
    - Products
    - Chunks
    - Images
    - Text embeddings (from embeddings table)
    - Image embeddings (from VECS)
    - Total embeddings (text + image)
    """
    try:
        from app.services.embeddings.vecs_service import get_vecs_service

        # Query Supabase tables for counts
        products_response = supabase.client.table('products').select('id', count='exact').eq('workspace_id', workspace_id).execute()
        chunks_response = supabase.client.table('document_chunks').select('id', count='exact').eq('workspace_id', workspace_id).execute()
        images_response = supabase.client.table('document_images').select('id', count='exact').eq('workspace_id', workspace_id).execute()

        # Count text embeddings from document_chunks table (chunks with text_embedding not null)
        # Note: embeddings are stored directly in document_chunks.text_embedding, not in a separate table
        try:
            text_embeddings_response = supabase.client.table('document_chunks').select('id', count='exact').eq('workspace_id', workspace_id).not_.is_('text_embedding', 'null').execute()
            text_embeddings_count = text_embeddings_response.count or 0
        except Exception as text_emb_error:
            logger.warning(f"⚠️ Text embeddings count failed: {text_emb_error}")
            text_embeddings_count = 0

        # Get VECS image embeddings count using SQL function (bypasses VECS connection issues)
        # FIX: Cast workspace_id to UUID to avoid function overloading ambiguity
        try:
            # Use the UUID version of the function by casting the parameter
            vecs_count_result = supabase.client.rpc('count_vecs_embeddings', {'p_workspace_id': str(workspace_id)}).execute()
            image_embeddings_count = vecs_count_result.data if isinstance(vecs_count_result.data, int) else 0
        except Exception as vecs_error:
            logger.warning(f"⚠️ VECS count failed: {vecs_error}")
            image_embeddings_count = 0

        # Calculate totals
        products_count = products_response.count or 0
        chunks_count = chunks_response.count or 0
        images_count = images_response.count or 0
        total_embeddings = text_embeddings_count + image_embeddings_count

        stats = {
            "workspace_id": workspace_id,
            "products": products_count,
            "chunks": chunks_count,
            "images": images_count,
            "embeddings": {
                "text": text_embeddings_count,
                "images": image_embeddings_count,
                "total": total_embeddings
            }
        }

        logger.info(f"✅ Workspace stats: {stats}")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "statistics": stats,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Workspace statistics retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workspace statistics retrieval failed: {str(e)}"
        )




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


# ============================================================================
# RAG Knowledge Base Search (No PDF Upload Required)
# ============================================================================

class KnowledgeBaseSearchRequest(BaseModel):
    """Request model for knowledge base search"""
    query: str = Field(..., description="Search query")
    workspace_id: str = Field(..., description="Workspace ID to search within")
    search_types: List[str] = Field(
        default=["products", "entities", "chunks"],
        description="Types to search: products, entities, chunks, images, kb_docs"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Filter by categories: product, certificate, logo, specification, general"
    )
    entity_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by entity types: certificate, logo, specification"
    )
    top_k: int = Field(default=10, description="Number of results to return per type")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity score")
    caller: str = Field(
        default="agent",
        description="Caller context: 'admin' (all levels), 'agent' (agent+public), 'public' (public only)"
    )


class KnowledgeBaseSearchResponse(BaseModel):
    """Response model for knowledge base search"""
    query: str
    total_results: int
    products: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []
    images: List[Dict[str, Any]] = []
    processing_time: float
    search_metadata: Dict[str, Any]


@router.post("/search/knowledge-base", response_model=KnowledgeBaseSearchResponse)
async def search_knowledge_base(
    request: KnowledgeBaseSearchRequest,
    rag_service: RAGService = Depends(get_rag_service),
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    🔍 Search existing knowledge base without uploading a PDF.

    Uses the same **7-vector fusion search** as the main search endpoint, combining:
    - Text (15%) - Voyage AI 1024D semantic understanding
    - Visual (15%) - SLIG 768D visual similarity
    - Understanding (20%) - Voyage AI 1024D from Qwen3-VL analysis
    - Color (12.5%) - SLIG 1152D color palette matching
    - Texture (12.5%) - SLIG 1152D texture pattern matching
    - Style (12.5%) - SLIG 1152D design style matching
    - Material (12.5%) - SLIG 1152D material type matching

    Performs unified semantic search across:
    - **Products** (with all metadata, embeddings, and material properties)
    - **Document entities** (certificates, logos, specifications)
    - **Chunks** (text content from PDFs with category tags)
    - **Images** (visual content with CLIP embeddings)

    Supports:
    - Category filtering (product, certificate, logo, specification, general)
    - Entity type filtering (certificate, logo, specification)
    - Material property filtering via metadata

    Example queries:
    - "waterproof ceramic tiles with matte finish"
    - "ISO 9001 certificates"
    - "company logos"
    - "installation specifications"
    """
    try:
        start_time = datetime.utcnow()
        logger.info(f"🔍 Knowledge base search: '{request.query}' in workspace {request.workspace_id}")

        # Initialize services
        supabase = get_supabase_client()

        results = {
            "products": [],
            "entities": [],
            "chunks": [],
            "images": []
        }

        # 🎯 Search products using multi-vector search (same as main search endpoint)
        if "products" in request.search_types:
            logger.info("   🎯 Searching products with multi-vector search...")
            try:
                # Build material filters from categories if provided
                material_filters = {}
                if request.categories:
                    material_filters['categories'] = request.categories

                # Use the same multi_vector_search method as the main search endpoint
                product_results = await rag_service.multi_vector_search(
                    query=request.query,
                    workspace_id=request.workspace_id,
                    top_k=request.top_k,
                    material_filters=material_filters if material_filters else None,
                    similarity_threshold=request.similarity_threshold
                )

                # Extract products from results
                if product_results and product_results.get('results'):
                    for result in product_results['results']:
                        results["products"].append({
                            "id": result.get('id'),
                            "name": result.get('product_name') or result.get('name'),
                            "description": result.get('description'),
                            "metadata": result.get('metadata', {}),
                            "relevance_score": result.get('weighted_score', 0.0),
                            "type": "product",
                            "embeddings": {
                                "text": bool(result.get('text_embedding_1024')),
                                "visual": bool(result.get('visual_clip_embedding_512')),
                                "understanding": bool(result.get('vision_analysis')),
                            }
                        })

                logger.info(f"   ✅ Found {len(results['products'])} products")

            except Exception as e:
                logger.warning(f"Product search failed: {e}")

        # Search entities using embeddings
        if "entities" in request.search_types:
            logger.info("   Searching entities...")
            try:
                # Search entity embeddings using VECS
                entity_search_results = await vecs_service.search_similar(
                    collection_name="embeddings",
                    query_embedding=query_embedding,
                    limit=request.top_k * 2,  # Get more for filtering
                    filters={"entity_type": "entity"}
                )

                # Fetch full entity details
                for result in entity_search_results:
                    entity_id = result.get('id')
                    similarity = result.get('similarity', 0.0)

                    if similarity < request.similarity_threshold:
                        continue

                    # Fetch entity from database
                    entity_response = supabase.client.table('document_entities').select('*').eq(
                        'id', entity_id
                    ).eq('workspace_id', request.workspace_id).execute()

                    if entity_response.data and len(entity_response.data) > 0:
                        entity = entity_response.data[0]

                        # Apply entity type filter
                        if request.entity_types and entity.get('entity_type') not in request.entity_types:
                            continue

                        results["entities"].append({
                            "id": entity['id'],
                            "entity_type": entity.get('entity_type'),
                            "name": entity.get('name'),
                            "description": entity.get('description'),
                            "metadata": entity.get('metadata', {}),
                            "relevance_score": similarity,
                            "type": "entity"
                        })

                # Sort and limit
                results["entities"] = sorted(
                    results["entities"],
                    key=lambda x: x['relevance_score'],
                    reverse=True
                )[:request.top_k]

            except Exception as e:
                logger.warning(f"Entity search failed: {e}")

        # Search chunks using embeddings
        if "chunks" in request.search_types:
            logger.info("   Searching chunks...")
            try:
                # Build category filter if provided
                chunk_filters = {}
                if request.categories:
                    chunk_filters["category"] = request.categories

                # Search chunk embeddings
                chunks_response = supabase.client.table('document_chunks').select('*').eq(
                    'workspace_id', request.workspace_id
                ).limit(request.top_k * 3).execute()

                if chunks_response.data:
                    for chunk in chunks_response.data:
                        # Apply category filter
                        if request.categories and chunk.get('category') not in request.categories:
                            continue

                        # Simple text matching
                        chunk_text = chunk.get('content', '').lower()
                        query_lower = request.query.lower()

                        score = 0.0
                        query_words = query_lower.split()
                        for word in query_words:
                            if word in chunk_text:
                                score += 0.15

                        if score >= request.similarity_threshold:
                            results["chunks"].append({
                                "id": chunk['id'],
                                "content": chunk.get('content', '')[:500],  # Truncate for response
                                "category": chunk.get('category'),
                                "metadata": chunk.get('metadata', {}),
                                "relevance_score": min(score, 1.0),
                                "type": "chunk"
                            })

                    # Sort and limit
                    results["chunks"] = sorted(
                        results["chunks"],
                        key=lambda x: x['relevance_score'],
                        reverse=True
                    )[:request.top_k]

            except Exception as e:
                logger.warning(f"Chunk search failed: {e}")

        # Search KB docs (kb_docs table, filtered by category access_level + trigger_keyword)
        if "kb_docs" in request.search_types:
            logger.info("   📚 Searching KB docs...")
            try:
                caller = request.caller or "agent"
                query_lower = request.query.lower()

                if caller == "admin":
                    # Admin sees everything — no keyword restriction
                    allowed_access_levels = ["admin", "agent", "public"]
                    accessible_category_ids = None  # no post-filter needed
                elif caller == "public":
                    allowed_access_levels = ["public"]
                    accessible_category_ids = None
                else:
                    # Agent caller: public categories always accessible;
                    # agent-level categories only if trigger_keyword matches (or no keyword set)
                    allowed_access_levels = ["agent", "public"]

                    cats_resp = supabase.client.table("kb_categories").select(
                        "id, access_level, trigger_keyword"
                    ).eq("workspace_id", request.workspace_id).in_(
                        "access_level", ["agent", "public"]
                    ).execute()

                    accessible_category_ids: list[str] = []
                    for cat in (cats_resp.data or []):
                        if cat["access_level"] == "public":
                            # Public categories: always accessible to agent
                            accessible_category_ids.append(cat["id"])
                        else:
                            # Agent-level: check trigger_keyword
                            kw = cat.get("trigger_keyword")
                            if kw is None or kw.strip() == "":
                                # No keyword restriction — always accessible
                                accessible_category_ids.append(cat["id"])
                            elif kw.lower() in query_lower:
                                # Keyword present in query — grant access
                                accessible_category_ids.append(cat["id"])
                            # else: keyword required but not found — skip this category

                    logger.info(
                        f"   🔑 Accessible KB categories for query: {len(accessible_category_ids)}"
                    )

                # Generate text_1024 embedding for the query
                from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
                embeddings_service = RealEmbeddingsService()
                embedding_result = await embeddings_service.generate_all_embeddings(
                    entity_id="kb_search_query",
                    entity_type="search",
                    text_content=request.query
                )

                if embedding_result.get("success"):
                    query_embedding = embedding_result.get("embeddings", {}).get("text_1024")
                    if query_embedding:
                        kb_response = supabase.client.rpc(
                            "kb_match_docs",
                            {
                                "query_embedding": query_embedding,
                                "match_workspace_id": request.workspace_id,
                                "match_threshold": 0.5,
                                "match_count": request.top_k * 2,  # fetch extra, will post-filter
                                "allowed_access_levels": allowed_access_levels,
                            }
                        ).execute()

                        if kb_response.data:
                            kb_count = 0
                            for doc in kb_response.data:
                                cat_id = doc.get("category_id")
                                # Post-filter: skip if we have an accessible set and this cat isn't in it
                                if (
                                    accessible_category_ids is not None
                                    and cat_id is not None
                                    and cat_id not in accessible_category_ids
                                ):
                                    continue
                                results["chunks"].append({
                                    "id": doc.get("id"),
                                    "content": (doc.get("content") or "")[:800],
                                    "document_title": doc.get("title"),
                                    "category": cat_id,
                                    "metadata": {"source": "kb_docs", "visibility": doc.get("visibility")},
                                    "relevance_score": doc.get("similarity", 0.0),
                                    "type": "kb_doc",
                                })
                                kb_count += 1
                                if kb_count >= request.top_k:
                                    break
                            logger.info(f"   ✅ Found {kb_count} KB docs after keyword filtering")

            except Exception as e:
                logger.warning(f"KB docs search failed: {e}")

        processing_time = (datetime.utcnow() - start_time).total_seconds()
        total_results = len(results["products"]) + len(results["entities"]) + len(results["chunks"]) + len(results["images"])

        logger.info(f"✅ Knowledge base search complete: {total_results} results in {processing_time:.2f}s")

        return KnowledgeBaseSearchResponse(
            query=request.query,
            total_results=total_results,
            products=results["products"],
            entities=results["entities"],
            chunks=results["chunks"],
            images=results["images"],
            processing_time=processing_time,
            search_metadata={
                "search_types": request.search_types,
                "categories_filter": request.categories,
                "entity_types_filter": request.entity_types,
                "similarity_threshold": request.similarity_threshold
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge base search failed: {str(e)}"
        )
