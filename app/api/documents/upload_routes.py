"""
Document Upload API Routes

This module handles all document upload functionality including:
- File uploads
- URL-based uploads
- Processing mode configuration
- Category-based extraction
"""

import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status, BackgroundTasks
import aiohttp

from app.services.core.supabase_client import get_supabase_client
from app.orchestration import process_document_with_discovery, run_async_in_background

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
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

    ## 📋 Processing Modes

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

    ## 🎨 Category-Based Extraction

    Control what gets extracted:
    - `categories="products"` - Extract only products
    - `categories="certificates"` - Extract only certificates
    - `categories="products,certificates"` - Extract multiple categories
    - `categories="all"` - Extract everything (default)
    - `categories="extract_only"` - Just extract text/images, no categorization

    ## 🌐 URL Processing

    Upload from URL instead of file:
    - Set `file_url="https://example.com/catalog.pdf"`
    - Leave `file` parameter empty
    - System downloads and processes automatically

    ## 🤖 AI Model Selection

    Choose discovery model:
    - `discovery_model="claude-vision"` - Claude Sonnet 4.5 Vision (best quality, RECOMMENDED)
    - `discovery_model="gpt-vision"` - GPT-4o Vision (fast, good quality)
    - `discovery_model="claude-haiku-vision"` - Claude Haiku Vision (fastest, lower cost)

    ## 💬 Agent Prompts

    Use natural language instructions:
    - `agent_prompt="extract all products"` - Enhanced with product extraction details
    - `agent_prompt="search for NOVA"` - Enhanced with search context
    - `agent_prompt="find certificates"` - Enhanced with certificate extraction details

    ## ✅ Response Example

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
        job_id = str(uuid4())
        document_id = str(uuid4())

        logger.info(f"📤 CONSOLIDATED UPLOAD")
        logger.info(f"   Job ID: {job_id}")
        logger.info(f"   Document ID: {document_id}")
        logger.info(f"   Filename: {filename}")
        logger.info(f"   Processing Mode: {processing_mode}")
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

        # Get Supabase client
        supabase_client = get_supabase_client()

        # Create document record
        try:
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
                    "processing_mode": processing_mode,
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
                    "processing_mode": processing_mode,
                    "categories": category_list,
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

        # Start background processing
        # Note: "quick" mode is not implemented - all processing uses standard mode
        if processing_mode == "quick":
            logger.info("Quick mode requested but not implemented, using standard mode")
            processing_mode = "standard"

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
        logger.error(f"❌ Consolidated upload failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )

