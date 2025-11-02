"""
Unified Document Upload API

Single consolidated endpoint for all PDF upload scenarios.
Replaces /documents/upload, /documents/upload-async, /documents/upload-with-discovery, /documents/upload-focused
"""

import logging
import tempfile
import os
from typing import Optional, List
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field

from app.services.supabase_client import get_supabase_client
from app.services.prompt_enhancement_service import PromptEnhancementService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents - Unified Upload"])


# Request/Response Models
class UnifiedUploadResponse(BaseModel):
    """Unified upload response"""
    job_id: str
    document_id: str
    status: str
    message: str
    status_url: str
    categories: List[str]
    discovery_model: Optional[str] = None
    prompt_enhancement_enabled: bool


@router.post("/upload", response_model=UnifiedUploadResponse)
async def unified_document_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload"),
    
    # Basic metadata
    title: Optional[str] = Form(None, description="Document title"),
    description: Optional[str] = Form(None, description="Document description"),
    tags: Optional[str] = Form(None, description="Comma-separated tags"),
    
    # Category-based extraction (NEW)
    categories: str = Form(
        "products",
        description="Categories to extract: 'products', 'certificates', 'logos', 'specifications', 'all'. Comma-separated."
    ),
    
    # Discovery settings
    discovery_model: str = Form(
        "claude",
        description="AI model for discovery: 'claude' (default), 'gpt', 'haiku'"
    ),
    
    # Processing settings
    chunk_size: int = Form(1000, ge=100, le=4000, description="Chunk size for text processing"),
    chunk_overlap: int = Form(200, ge=0, le=1000, description="Chunk overlap"),
    
    # Prompt enhancement (NEW)
    enable_prompt_enhancement: bool = Form(
        True,
        description="Enable AI prompt enhancement with admin customizations"
    ),
    
    # Agent prompt (NEW) - Optional natural language instruction
    agent_prompt: Optional[str] = Form(
        None,
        description="Natural language instruction (e.g., 'extract products', 'search for NOVA')"
    ),
    
    # Workspace
    workspace_id: str = Form(
        "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
        description="Workspace ID"
    ),
    
    # Legacy compatibility (DEPRECATED)
    focused_extraction: Optional[bool] = Form(
        None,
        description="DEPRECATED: Use 'categories' parameter instead"
    ),
    extract_categories: Optional[str] = Form(
        None,
        description="DEPRECATED: Use 'categories' parameter instead"
    )
):
    """
    **Unified Document Upload Endpoint**
    
    Single endpoint that handles ALL PDF upload scenarios:
    - Simple document upload
    - Product discovery and extraction
    - Certificate extraction
    - Logo extraction
    - Specification extraction
    - Multi-category extraction
    - Agent-driven extraction with natural language prompts
    
    ## NEW FEATURES
    
    ### 1. Category-Based Extraction
    Instead of boolean `focused_extraction`, use `categories` parameter:
    - `categories="products"` - Extract only products
    - `categories="certificates"` - Extract only certificates
    - `categories="products,certificates"` - Extract multiple categories
    - `categories="all"` - Extract everything
    
    ### 2. Agent Prompt Enhancement
    Send simple natural language prompts that get enhanced automatically:
    - `agent_prompt="extract products"` → Enhanced with product extraction details
    - `agent_prompt="search for NOVA"` → Enhanced with search context
    - `agent_prompt="find certificates"` → Enhanced with certificate extraction details
    
    Admin can customize prompts via `/admin/extraction-prompts` endpoints.
    
    ## PROCESSING PIPELINE
    
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
    
    ## EXAMPLES
    
    ### Extract Products Only
    ```bash
    curl -X POST "/api/rag/documents/upload" \\
      -F "file=@catalog.pdf" \\
      -F "categories=products" \\
      -F "agent_prompt=extract all products"
    ```
    
    ### Extract Products and Certificates
    ```bash
    curl -X POST "/api/rag/documents/upload" \\
      -F "file=@catalog.pdf" \\
      -F "categories=products,certificates"
    ```
    
    ### Search for Specific Product
    ```bash
    curl -X POST "/api/rag/documents/upload" \\
      -F "file=@catalog.pdf" \\
      -F "categories=products" \\
      -F "agent_prompt=search for NOVA product"
    ```
    
    ### Extract Everything
    ```bash
    curl -X POST "/api/rag/documents/upload" \\
      -F "file=@catalog.pdf" \\
      -F "categories=all"
    ```
    
    ## BACKWARD COMPATIBILITY
    
    Old parameters still work but are deprecated:
    - `focused_extraction=True` → Converted to `categories="products"`
    - `extract_categories="products,certificates"` → Converted to `categories="products,certificates"`
    
    ## RETURNS
    
    Job ID and status URL for monitoring progress.
    Poll `/api/rag/documents/job/{job_id}` for status updates.
    """
    
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
        
        # Handle legacy parameters (backward compatibility)
        if extract_categories is not None:
            logger.warning("⚠️ DEPRECATED: 'extract_categories' parameter is deprecated. Use 'categories' instead.")
            categories = extract_categories
        
        if focused_extraction is not None:
            logger.warning("⚠️ DEPRECATED: 'focused_extraction' parameter is deprecated. Use 'categories' instead.")
            if not focused_extraction:
                categories = "all"
        
        # Parse categories
        category_list = [cat.strip() for cat in categories.split(',')]
        
        # Validate categories
        valid_categories = ['products', 'certificates', 'logos', 'specifications', 'all']
        for cat in category_list:
            if cat not in valid_categories:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid category '{cat}'. Valid categories: {', '.join(valid_categories)}"
                )
        
        # If 'all' is specified, expand to all categories
        if 'all' in category_list:
            category_list = ['products', 'certificates', 'logos', 'specifications']
        
        # Generate IDs
        job_id = str(uuid4())
        document_id = str(uuid4())
        
        logger.info(f"📤 UNIFIED UPLOAD")
        logger.info(f"   Job ID: {job_id}")
        logger.info(f"   Document ID: {document_id}")
        logger.info(f"   Filename: {file.filename}")
        logger.info(f"   Categories: {category_list}")
        logger.info(f"   Discovery Model: {discovery_model}")
        logger.info(f"   Prompt Enhancement: {enable_prompt_enhancement}")
        if agent_prompt:
            logger.info(f"   Agent Prompt: {agent_prompt}")
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Parse tags
        document_tags = []
        if tags:
            document_tags = [tag.strip() for tag in tags.split(',')]
        
        # Save file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(file_content)
        temp_file.close()
        file_path = temp_file.name
        
        # Get Supabase client
        supabase_client = get_supabase_client()
        
        # Create document record
        try:
            supabase_client.client.table('documents').insert({
                "id": document_id,
                "workspace_id": workspace_id,
                "filename": file.filename,
                "content_type": "application/pdf",
                "file_size": file_size,
                "file_path": file_path,
                "processing_status": "processing",
                "metadata": {
                    "title": title or file.filename,
                    "description": description or f"Document with {', '.join(category_list)} extraction",
                    "tags": document_tags,
                    "source": "unified_upload",
                    "categories": category_list,
                    "discovery_model": discovery_model,
                    "prompt_enhancement_enabled": enable_prompt_enhancement,
                    "agent_prompt": agent_prompt
                },
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).execute()
            logger.info(f"✅ Created document record {document_id}")
        except Exception as e:
            logger.error(f"❌ Failed to create document record: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create document record: {str(e)}"
            )

        # Create background job record
        try:
            supabase_client.client.table('background_jobs').insert({
                "id": job_id,
                "job_type": "pdf_processing",
                "status": "processing",
                "progress": 0,
                "document_id": document_id,
                "workspace_id": workspace_id,
                "metadata": {
                    "filename": file.filename,
                    "categories": category_list,
                    "discovery_model": discovery_model,
                    "prompt_enhancement_enabled": enable_prompt_enhancement,
                    "agent_prompt": agent_prompt
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
        
        # Import and start background processing
        from app.api.rag_routes import process_document_with_discovery
        
        background_tasks.add_task(
            process_document_with_discovery,
            job_id=job_id,
            document_id=document_id,
            file_content=file_content,
            filename=file.filename,
            title=title,
            description=description,
            document_tags=document_tags,
            discovery_model=discovery_model,
            focused_extraction=True,  # Always use focused extraction with categories
            extract_categories=category_list,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            workspace_id=workspace_id,
            agent_prompt=agent_prompt,  # NEW: Pass agent prompt for enhancement
            enable_prompt_enhancement=enable_prompt_enhancement  # NEW: Pass enhancement flag
        )
        
        return UnifiedUploadResponse(
            job_id=job_id,
            document_id=document_id,
            status="processing",
            message=f"Document upload started with {', '.join(category_list)} extraction",
            status_url=f"/api/rag/documents/job/{job_id}",
            categories=category_list,
            discovery_model=discovery_model,
            prompt_enhancement_enabled=enable_prompt_enhancement
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unified upload failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )

