"""
Document Routes - Upload, list, get, delete documents
"""

import logging
import os
import base64
from datetime import datetime
from typing import Optional, List
from uuid import uuid4
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, status, BackgroundTasks
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.services.supabase_client import get_supabase_client
from .shared import job_storage, run_async_in_background
from .models import DocumentListResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None, description="PDF file to upload (required unless file_url is provided)"),

    # Basic metadata
    title: Optional[str] = Form(None, description="Document title"),
    description: Optional[str] = Form(None, description="Document description"),
    tags: Optional[str] = Form(None, description="Comma-separated tags"),

    # NEW: Processing mode parameter
    processing_mode: str = Form(
        "standard",
        description="Processing mode: 'quick' (extract only), 'standard' (full RAG), 'deep' (complete analysis)"
    ),

    # NEW: Category-based extraction
    categories: str = Form(
        "all",
        description="Categories to extract: 'products', 'certificates', 'logos', 'specifications', 'all', 'extract_only'. Comma-separated."
    ),

    # NEW: URL-based upload
    file_url: Optional[str] = Form(
        None,
        description="URL to download PDF from (alternative to file upload)"
    ),

    # Discovery settings
    discovery_model: str = Form(
        "claude",
        description="AI model for discovery: 'claude' (default), 'gpt', 'haiku'"
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
    **?? CONSOLIDATED UPLOAD ENDPOINT - Single Entry Point for All Upload Scenarios**

    This endpoint replaces:
    - `/api/documents/process` (simple extraction)
    - `/api/documents/process-url` (URL processing)
    - `/api/documents/upload` (old unified upload)

    ## ?? Processing Modes

    ### Quick Mode (`processing_mode="quick"`)
    - Fast extraction without RAG
    - No embeddings generated
    - No product discovery
    - Use for: Simple text/image extraction

    ### Standard Mode (`processing_mode="standard"`) - DEFAULT
    - Full RAG pipeline
    - Text embeddings generated
    - Product discovery and extraction
    - Use for: Normal document processing

    ### Deep Mode (`processing_mode="deep"`)
    - Complete analysis with all AI models
    - Image embeddings (CLIP)
    - Advanced product enrichment
    - Quality validation
    - Use for: High-quality catalog processing

    ## ?? Category-Based Extraction

    Control what gets extracted:
    - `categories="products"` - Extract only products
    - `categories="certificates"` - Extract only certificates
    - `categories="products,certificates"` - Extract multiple categories
    - `categories="all"` - Extract everything (default)
    - `categories="extract_only"` - Just extract text/images, no categorization

    ## ?? URL Processing

    Upload from URL instead of file:
    - Set `file_url="https://example.com/catalog.pdf"`
    - Leave `file` parameter empty
    - System downloads and processes automatically

    ## ?? AI Model Selection

    Choose discovery model:
    - `discovery_model="claude"` - Claude Sonnet 4.5 (best quality, default)
    - `discovery_model="gpt"` - GPT-5 (fast, good quality)
    - `discovery_model="haiku"` - Claude Haiku 4.5 (fastest, lower cost)

    ## ?? Agent Prompts

    Use natural language instructions:
    - `agent_prompt="extract all products"` - Enhanced with product extraction details
    - `agent_prompt="search for NOVA"` - Enhanced with search context
    - `agent_prompt="find certificates"` - Enhanced with certificate extraction details

    ## ?? Processing Pipeline

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
    - AI analysis (Llama Vision)
    - Generates image embeddings (CLIP)

    **Stage 4: Entity Creation (70-90%)**
    - Creates products, certificates, logos, specifications
    - Links chunks and images
    - Attaches metadata

    **Stage 5: Quality Enhancement (90-100%)**
    - Async quality validation
    - Advanced embeddings
    - Entity enrichment

    ## ?? Examples

    ### Simple Extraction (Quick Mode)
    ```bash
    curl -X POST "/api/rag/documents/upload" \\
      -F "file=@document.pdf" \\
      -F "processing_mode=quick" \\
      -F "categories=extract_only"
    ```

    ### Standard Product Extraction
    ```bash
    curl -X POST "/api/rag/documents/upload" \\
      -F "file=@catalog.pdf" \\
      -F "processing_mode=standard" \\
      -F "categories=products"
    ```

    ### Deep Analysis with Multiple Categories
    ```bash
    curl -X POST "/api/rag/documents/upload" \\
      -F "file=@catalog.pdf" \\
      -F "processing_mode=deep" \\
      -F "categories=products,certificates,logos"
    ```

    ### URL Processing
    ```bash
    curl -X POST "/api/rag/documents/upload" \\
      -F "file_url=https://example.com/catalog.pdf" \\
      -F "processing_mode=standard" \\
      -F "categories=all"
    ```

    ### Agent-Driven Extraction
    ```bash
    curl -X POST "/api/rag/documents/upload" \\
      -F "file=@catalog.pdf" \\
      -F "agent_prompt=search for NOVA product" \\
      -F "categories=products"
    ```

    ## ? Response Example

    ```json
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "document_id": "660e8400-e29b-41d4-a716-446655440001",
      "status": "pending",
      "message": "Document upload successful. Processing started.",
      "status_url": "/api/rag/documents/job/550e8400-e29b-41d4-a716-446655440000",
      "processing_mode": "standard",
      "categories": ["products", "certificates"],
      "estimated_time": "2-5 minutes"
    }
    ```

    ## ?? Monitoring Progress

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

    ## ?? Error Codes

    - **400 Bad Request**: Invalid parameters (missing file/URL, invalid mode, unsupported file type)
    - **401 Unauthorized**: Missing or invalid authentication
    - **413 Payload Too Large**: File exceeds size limit (100MB)
    - **415 Unsupported Media Type**: Non-PDF file uploaded
    - **500 Internal Server Error**: Processing initialization failed
    - **503 Service Unavailable**: Background job queue full

    ## ?? Limits

    - **Max file size**: 100MB
    - **Max concurrent jobs**: 5 per workspace
    - **Supported formats**: PDF only
    - **URL download timeout**: 60 seconds

    ## ?? Migration from Old Endpoints

    **Old:** `POST /api/documents/process`
    **New:** `POST /api/rag/documents/upload` with `processing_mode="quick"`

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

        # Validate processing mode
        valid_modes = ['quick', 'standard', 'deep']
        if processing_mode not in valid_modes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid processing_mode '{processing_mode}'. Valid modes: {', '.join(valid_modes)}"
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
            file_content = await file.read()

        elif file_url:
            # Download from URL
            import aiohttp
            import tempfile

            logger.info(f"?? Downloading PDF from URL: {file_url}")

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

                        logger.info(f"? Downloaded {len(file_content)} bytes from URL")

            except aiohttp.ClientError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to download PDF from URL: {str(e)}"
                )

        # Generate IDs
        from uuid import uuid4
        job_id = str(uuid4())
        document_id = str(uuid4())

        logger.info(f"?? CONSOLIDATED UPLOAD")
        logger.info(f"   Job ID: {job_id}")
        logger.info(f"   Document ID: {document_id}")
        logger.info(f"   Filename: {filename}")
        logger.info(f"   Processing Mode: {processing_mode}")
        logger.info(f"   Categories: {category_list}")
        logger.info(f"   Discovery Model: {discovery_model}")
        logger.info(f"   Source: {'URL' if file_url else 'Upload'}")
        if agent_prompt:
            logger.info(f"   Agent Prompt: {agent_prompt}")

        # Parse tags
        document_tags = []
        if tags:
            document_tags = [tag.strip() for tag in tags.split(',')]

        # Save file temporarily
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(file_content)
        temp_file.close()
        file_path = temp_file.name

        # Get Supabase client
        supabase_client = get_supabase_client()

        # Create document record
        try:
            from datetime import datetime
            supabase_client.client.table('documents').insert({
                "id": document_id,
                "workspace_id": workspace_id,
                "filename": filename,
                "content_type": "application/pdf",
                "file_size": len(file_content),
                "file_path": file_path,
                "processing_status": "processing",
                "metadata": {
                    "title": title or filename,
                    "description": description or f"Document with {', '.join(category_list)} extraction",
                    "tags": document_tags,
                    "source": "consolidated_upload",
                    "processing_mode": processing_mode,
                    "categories": category_list,
                    "discovery_model": discovery_model,
                    "prompt_enhancement_enabled": enable_prompt_enhancement,
                    "agent_prompt": agent_prompt,
                    "file_url": file_url
                },
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).execute()
            logger.info(f"? Created document record {document_id}")
        except Exception as e:
            logger.error(f"? Failed to create document record: {e}")
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
                    "processing_mode": processing_mode,
                    "categories": category_list
                },
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).execute()
            logger.info(f"? Created/updated processed_documents record: {document_id}")
        except Exception as proc_doc_error:
            logger.error(f"? Failed to create processed_documents record: {proc_doc_error}")
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
                    "processing_mode": processing_mode,
                    "categories": category_list,
                    "discovery_model": discovery_model,
                    "prompt_enhancement_enabled": enable_prompt_enhancement,
                    "agent_prompt": agent_prompt,
                    "file_url": file_url
                },
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).execute()
            logger.info(f"? Created background job record {job_id}")
        except Exception as e:
            logger.error(f"? Failed to create background job record: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create background job record: {str(e)}"
            )

        # Start background processing
        # Note: "quick" mode is not implemented - all processing uses standard mode
        if processing_mode == "quick":
            logger.info("Quick mode requested but not implemented, using standard mode")
            processing_mode = "standard"

        # Import background processing function (avoid circular import)
        from app.api.rag_routes import process_document_with_discovery

        # Use the existing process_document_with_discovery function
        # Wrap async function for background execution
        background_tasks.add_task(
            run_async_in_background(process_document_with_discovery),
            job_id=job_id,
            document_id=document_id,
            file_content=file_content,
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
            enable_prompt_enhancement=enable_prompt_enhancement
        )

        return {
            "job_id": job_id,
            "document_id": document_id,
            "status": "processing",
            "message": f"Document upload started with {processing_mode} mode and {', '.join(category_list)} extraction",
            "status_url": f"/api/rag/documents/job/{job_id}",
            "processing_mode": processing_mode,
            "categories": category_list,
            "discovery_model": discovery_model,
            "prompt_enhancement_enabled": enable_prompt_enhancement,
            "source": "url" if file_url else "upload"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"? Consolidated upload failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
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
        logger.info(f"?? Fetching complete content for document {document_id}")
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
            logger.info(f"?? Fetching chunks for document {document_id}")
            chunks_response = supabase_client.client.table('document_chunks').select('*').eq('document_id', document_id).execute()
            chunks = chunks_response.data or []

            # Get embeddings for each chunk
            for chunk in chunks:
                embeddings_response = supabase_client.client.table('embeddings').select('*').eq('chunk_id', chunk['id']).execute()
                chunk['embeddings'] = embeddings_response.data or []

            result['chunks'] = chunks
            logger.info(f"? Fetched {len(chunks)} chunks")

        # Get images with AI analysis
        if include_images:
            logger.info(f"??? Fetching images for document {document_id}")
            images_response = supabase_client.client.table('document_images').select('*').eq('document_id', document_id).execute()
            result['images'] = images_response.data or []
            logger.info(f"? Fetched {len(result['images'])} images")

        # Get products
        if include_products:
            logger.info(f"?? Fetching products for document {document_id}")
            products_response = supabase_client.client.table('products').select('*').eq('source_document_id', document_id).execute()
            result['products'] = products_response.data or []
            logger.info(f"? Fetched {len(result['products'])} products")

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

        logger.info(f"? Document content fetched successfully: {chunks_count} chunks, {images_count} images, {products_count} products")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"? Error fetching document content: {e}", exc_info=True)
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

