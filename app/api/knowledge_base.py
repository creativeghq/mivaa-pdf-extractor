"""
Knowledge Base API Routes

This module provides API endpoints for Knowledge Base document management,
including CRUD operations, embedding generation, PDF text extraction, and semantic search.
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File
from pydantic import BaseModel, Field
from datetime import datetime

from app.services.supabase_client import SupabaseClient
from app.services.real_embeddings_service import RealEmbeddingsService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kb", tags=["knowledge-base"])


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateKBDocRequest(BaseModel):
    """Request model for creating a knowledge base document."""
    workspace_id: str = Field(..., description="UUID of the workspace")
    title: str = Field(..., description="Document title", min_length=1, max_length=255)
    content: str = Field(..., description="Document content", min_length=1)
    content_markdown: Optional[str] = Field(None, description="Markdown version of content")
    summary: Optional[str] = Field(None, description="Document summary")
    category_id: Optional[str] = Field(None, description="UUID of category")
    seo_keywords: Optional[List[str]] = Field(None, description="SEO keywords")
    status: str = Field(default="draft", description="Document status")
    visibility: str = Field(default="workspace", description="Document visibility")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Custom metadata")

    class Config:
        schema_extra = {
            "example": {
                "workspace_id": "uuid",
                "title": "Installation Guide",
                "content": "Step 1: Unpack the product...",
                "category_id": "uuid",
                "status": "draft",
                "visibility": "workspace"
            }
        }


class UpdateKBDocRequest(BaseModel):
    """Request model for updating a knowledge base document."""
    title: Optional[str] = Field(None, description="Document title")
    content: Optional[str] = Field(None, description="Document content")
    content_markdown: Optional[str] = Field(None, description="Markdown version")
    summary: Optional[str] = Field(None, description="Document summary")
    category_id: Optional[str] = Field(None, description="Category UUID")
    seo_keywords: Optional[List[str]] = Field(None, description="SEO keywords")
    status: Optional[str] = Field(None, description="Document status")
    visibility: Optional[str] = Field(None, description="Document visibility")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Custom metadata")


class KBDocResponse(BaseModel):
    """Response model for knowledge base document."""
    id: str
    workspace_id: str
    title: str
    content: str
    summary: Optional[str]
    category_id: Optional[str]
    status: str
    visibility: str
    embedding_status: str
    embedding_generated_at: Optional[str]
    created_by: Optional[str]
    created_at: str
    updated_at: str
    view_count: int


class KBSearchRequest(BaseModel):
    """Request model for knowledge base search."""
    workspace_id: str = Field(..., description="UUID of workspace")
    query: str = Field(..., description="Search query", min_length=1)
    search_type: str = Field(default="semantic", description="Search type: full-text, semantic, hybrid")
    limit: int = Field(default=10, description="Max results", ge=1, le=100)
    offset: int = Field(default=0, description="Result offset", ge=0)
    category_id: Optional[str] = Field(None, description="Filter by category")


class KBSearchResponse(BaseModel):
    """Response model for search results."""
    success: bool
    results: List[KBDocResponse]
    total_count: int
    search_time_ms: int
    search_type: str


# ============================================================================
# API Endpoints
# ============================================================================

@router.post(
    "/documents",
    response_model=KBDocResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new knowledge base document",
    description="Create a new document with automatic embedding generation"
)
async def create_kb_document(
    request: CreateKBDocRequest,
    supabase_client: SupabaseClient = Depends()
) -> KBDocResponse:
    """Create a new knowledge base document with embeddings."""
    try:
        # Generate embedding
        embeddings_service = RealEmbeddingsService()
        embedding_result = await embeddings_service.generate_all_embeddings(
            entity_id="temp",
            entity_type="kb_doc",
            text_content=request.content
        )
        
        text_embedding = None
        embedding_status = "pending"
        embedding_error = None
        
        if embedding_result.get("success"):
            text_embedding = embedding_result.get("embeddings", {}).get("text_1536")
            embedding_status = "success"
        else:
            embedding_error = embedding_result.get("error", "Unknown error")
            embedding_status = "failed"
        
        # Insert document
        doc_data = {
            "workspace_id": request.workspace_id,
            "title": request.title,
            "content": request.content,
            "content_markdown": request.content_markdown,
            "summary": request.summary,
            "category_id": request.category_id,
            "seo_keywords": request.seo_keywords,
            "status": request.status,
            "visibility": request.visibility,
            "metadata": request.metadata,
            "text_embedding": text_embedding,
            "embedding_status": embedding_status,
            "embedding_model": "text-embedding-3-small",
            "embedding_generated_at": datetime.utcnow().isoformat() if embedding_status == "success" else None,
            "embedding_error_message": embedding_error
        }
        
        result = supabase_client.client.table("kb_docs").insert(doc_data).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create document")
        
        doc = result.data[0]
        return KBDocResponse(**doc)
        
    except Exception as e:
        logger.error(f"Error creating KB document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/documents/{doc_id}",
    response_model=KBDocResponse,
    summary="Get a knowledge base document",
    description="Retrieve a specific document by ID"
)
async def get_kb_document(
    doc_id: str,
    supabase_client: SupabaseClient = Depends()
) -> KBDocResponse:
    """Get a knowledge base document by ID."""
    try:
        result = supabase_client.client.table("kb_docs").select("*").eq("id", doc_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return KBDocResponse(**result.data[0])
        
    except Exception as e:
        logger.error(f"Error fetching KB document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch(
    "/documents/{doc_id}",
    response_model=KBDocResponse,
    summary="Update a knowledge base document",
    description="Update document with smart embedding regeneration"
)
async def update_kb_document(
    doc_id: str,
    request: UpdateKBDocRequest,
    supabase_client: SupabaseClient = Depends()
) -> KBDocResponse:
    """Update a knowledge base document."""
    try:
        # Get current document
        current = supabase_client.client.table("kb_docs").select("*").eq("id", doc_id).execute()
        if not current.data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        current_doc = current.data[0]
        
        # Check if content changed
        content_changed = (
            request.title and request.title != current_doc.get("title") or
            request.content and request.content != current_doc.get("content") or
            request.summary and request.summary != current_doc.get("summary") or
            request.category_id and request.category_id != current_doc.get("category_id")
        )
        
        update_data = {}
        if request.title:
            update_data["title"] = request.title
        if request.content:
            update_data["content"] = request.content
        if request.content_markdown:
            update_data["content_markdown"] = request.content_markdown
        if request.summary:
            update_data["summary"] = request.summary
        if request.category_id:
            update_data["category_id"] = request.category_id
        if request.seo_keywords:
            update_data["seo_keywords"] = request.seo_keywords
        if request.status:
            update_data["status"] = request.status
        if request.visibility:
            update_data["visibility"] = request.visibility
        if request.metadata:
            update_data["metadata"] = request.metadata
        
        # Regenerate embedding if content changed
        if content_changed:
            embeddings_service = RealEmbeddingsService()
            embedding_result = await embeddings_service.generate_all_embeddings(
                entity_id="temp",
                entity_type="kb_doc",
                text_content=request.content or current_doc.get("content")
            )
            
            if embedding_result.get("success"):
                update_data["text_embedding"] = embedding_result.get("embeddings", {}).get("text_1536")
                update_data["embedding_status"] = "success"
                update_data["embedding_generated_at"] = datetime.utcnow().isoformat()
            else:
                update_data["embedding_status"] = "failed"
                update_data["embedding_error_message"] = embedding_result.get("error")
        
        update_data["updated_at"] = datetime.utcnow().isoformat()
        
        result = supabase_client.client.table("kb_docs").update(update_data).eq("id", doc_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to update document")
        
        return KBDocResponse(**result.data[0])
        
    except Exception as e:
        logger.error(f"Error updating KB document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/documents/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a knowledge base document",
    description="Delete a document and all related data"
)
async def delete_kb_document(
    doc_id: str,
    supabase_client: SupabaseClient = Depends()
) -> None:
    """Delete a knowledge base document."""
    try:
        supabase_client.client.table("kb_docs").delete().eq("id", doc_id).execute()
    except Exception as e:
        logger.error(f"Error deleting KB document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/search",
    response_model=KBSearchResponse,
    summary="Search knowledge base documents",
    description="Search using full-text, semantic, or hybrid search"
)
async def search_kb_documents(
    request: KBSearchRequest,
    supabase_client: SupabaseClient = Depends()
) -> KBSearchResponse:
    """Search knowledge base documents."""
    try:
        import time
        start_time = time.time()
        
        if request.search_type == "semantic":
            # Generate query embedding
            embeddings_service = RealEmbeddingsService()
            embedding_result = await embeddings_service.generate_all_embeddings(
                entity_id="temp",
                entity_type="query",
                text_content=request.query
            )
            
            if not embedding_result.get("success"):
                raise HTTPException(status_code=500, detail="Failed to generate query embedding")
            
            query_embedding = embedding_result.get("embeddings", {}).get("text_1536")
            
            # Vector similarity search
            query_sql = f"""
            SELECT * FROM kb_docs
            WHERE workspace_id = '{request.workspace_id}'
            AND text_embedding IS NOT NULL
            {'AND category_id = ' + f"'{request.category_id}'" if request.category_id else ''}
            ORDER BY text_embedding <=> '{query_embedding}'::vector
            LIMIT {request.limit}
            OFFSET {request.offset}
            """
            
            result = supabase_client.client.rpc("execute_query", {"query": query_sql}).execute()
            docs = result.data or []
        else:
            # Full-text search
            query_filter = supabase_client.client.table("kb_docs").select("*").eq("workspace_id", request.workspace_id)
            
            if request.category_id:
                query_filter = query_filter.eq("category_id", request.category_id)
            
            result = query_filter.ilike("title", f"%{request.query}%").limit(request.limit).offset(request.offset).execute()
            docs = result.data or []
        
        search_time_ms = int((time.time() - start_time) * 1000)
        
        return KBSearchResponse(
            success=True,
            results=[KBDocResponse(**doc) for doc in docs],
            total_count=len(docs),
            search_time_ms=search_time_ms,
            search_type=request.search_type
        )
        
    except Exception as e:
        logger.error(f"Error searching KB documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/documents/from-pdf",
    response_model=KBDocResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create document from PDF",
    description="Extract text from PDF and create document with embeddings"
)
async def create_kb_document_from_pdf(
    workspace_id: str,
    title: str,
    category_id: Optional[str] = None,
    file: UploadFile = File(...),
    supabase_client: SupabaseClient = Depends()
) -> KBDocResponse:
    """Create a knowledge base document from PDF file."""
    try:
        import fitz  # PyMuPDF

        # Read PDF file
        pdf_bytes = await file.read()

        # Extract text using PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_content = ""

        for page in doc:
            text_content += page.get_text()

        doc.close()

        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No text content found in PDF")

        # Create document with extracted text
        create_request = CreateKBDocRequest(
            workspace_id=workspace_id,
            title=title,
            content=text_content,
            category_id=category_id,
            status="draft",
            visibility="workspace"
        )

        return await create_kb_document(create_request, supabase_client)

    except Exception as e:
        logger.error(f"Error creating KB document from PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Category Endpoints
# ============================================================================

class CreateCategoryRequest(BaseModel):
    """Request model for creating a category."""
    workspace_id: str = Field(..., description="UUID of workspace")
    name: str = Field(..., description="Category name", min_length=1, max_length=100)
    slug: Optional[str] = Field(None, description="URL-friendly slug")
    description: Optional[str] = Field(None, description="Category description")
    icon: Optional[str] = Field(None, description="Icon name")
    color: Optional[str] = Field(None, description="Hex color code")
    parent_category_id: Optional[str] = Field(None, description="Parent category UUID")


class CategoryResponse(BaseModel):
    """Response model for category."""
    id: str
    workspace_id: str
    name: str
    slug: Optional[str]
    description: Optional[str]
    icon: Optional[str]
    color: Optional[str]
    parent_category_id: Optional[str]
    sort_order: int
    created_at: str


@router.post(
    "/categories",
    response_model=CategoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a category",
    description="Create a new knowledge base category"
)
async def create_category(
    request: CreateCategoryRequest,
    supabase_client: SupabaseClient = Depends()
) -> CategoryResponse:
    """Create a new category."""
    try:
        result = supabase_client.client.table("kb_categories").insert(request.dict()).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create category")

        return CategoryResponse(**result.data[0])

    except Exception as e:
        logger.error(f"Error creating category: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/categories",
    response_model=List[CategoryResponse],
    summary="List categories",
    description="Get all categories for a workspace"
)
async def list_categories(
    workspace_id: str,
    supabase_client: SupabaseClient = Depends()
) -> List[CategoryResponse]:
    """List all categories for a workspace."""
    try:
        result = supabase_client.client.table("kb_categories").select("*").eq("workspace_id", workspace_id).order("sort_order").execute()

        return [CategoryResponse(**cat) for cat in result.data or []]

    except Exception as e:
        logger.error(f"Error listing categories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Product Attachment Endpoints
# ============================================================================

class AttachProductRequest(BaseModel):
    """Request model for attaching document to product."""
    workspace_id: str = Field(..., description="UUID of workspace")
    document_id: str = Field(..., description="UUID of document")
    product_id: str = Field(..., description="UUID of product")
    relationship_type: str = Field(default="related", description="Relationship type")
    relevance_score: int = Field(default=3, description="Relevance score 1-5", ge=1, le=5)


class AttachmentResponse(BaseModel):
    """Response model for product attachment."""
    id: str
    workspace_id: str
    document_id: str
    product_id: str
    relationship_type: str
    relevance_score: int
    created_at: str


@router.post(
    "/attachments",
    response_model=AttachmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Attach document to product",
    description="Link a knowledge base document to a product"
)
async def attach_document_to_product(
    request: AttachProductRequest,
    supabase_client: SupabaseClient = Depends()
) -> AttachmentResponse:
    """Attach a document to a product."""
    try:
        result = supabase_client.client.table("kb_doc_attachments").insert(request.dict()).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create attachment")

        return AttachmentResponse(**result.data[0])

    except Exception as e:
        logger.error(f"Error creating attachment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/documents/{doc_id}/attachments",
    response_model=List[AttachmentResponse],
    summary="Get document attachments",
    description="Get all product attachments for a document"
)
async def get_document_attachments(
    doc_id: str,
    supabase_client: SupabaseClient = Depends()
) -> List[AttachmentResponse]:
    """Get all product attachments for a document."""
    try:
        result = supabase_client.client.table("kb_doc_attachments").select("*").eq("document_id", doc_id).execute()

        return [AttachmentResponse(**att) for att in result.data or []]

    except Exception as e:
        logger.error(f"Error fetching attachments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/products/{product_id}/documents",
    response_model=List[KBDocResponse],
    summary="Get product documents",
    description="Get all knowledge base documents attached to a product"
)
async def get_product_documents(
    product_id: str,
    supabase_client: SupabaseClient = Depends()
) -> List[KBDocResponse]:
    """Get all documents attached to a product."""
    try:
        # Get attachments
        attachments = supabase_client.client.table("kb_doc_attachments").select("document_id").eq("product_id", product_id).execute()

        if not attachments.data:
            return []

        doc_ids = [att["document_id"] for att in attachments.data]

        # Get documents
        result = supabase_client.client.table("kb_docs").select("*").in_("id", doc_ids).execute()

        return [KBDocResponse(**doc) for doc in result.data or []]

    except Exception as e:
        logger.error(f"Error fetching product documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class SearchKBRequest(BaseModel):
    """Request model for searching knowledge base documents."""
    workspace_id: str = Field(..., description="UUID of the workspace")
    query: str = Field(..., description="Search query", min_length=1)
    search_type: str = Field(default="semantic", description="Search type: semantic, full_text, or hybrid")
    limit: int = Field(default=20, description="Maximum number of results", ge=1, le=100)


class SearchKBResponse(BaseModel):
    """Response model for search results."""
    results: List[Dict[str, Any]]
    search_time_ms: float
    total_results: int


@router.post("/search", response_model=SearchKBResponse)
async def search_kb_documents(
    request: SearchKBRequest,
    supabase_client: SupabaseClient = Depends()
) -> SearchKBResponse:
    """
    Search knowledge base documents using semantic, full-text, or hybrid search.

    **Architecture:**
    1. Frontend calls MIVAA API with search query
    2. MIVAA generates embedding for query using OpenAI (text-embedding-3-small)
    3. MIVAA calls Supabase `match_kb_docs()` RPC function with query embedding
    4. Supabase performs vector similarity search using pgvector `<=>` operator
    5. Returns ranked results with similarity scores

    **Why MIVAA Backend is Required:**
    - Document embeddings already stored in `kb_docs.text_embedding` (generated when doc created)
    - Search only generates ONE embedding (for the query)
    - Cannot generate embeddings in Supabase RPC (requires OpenAI API call)
    - Uses pgvector's optimized cosine similarity for fast search

    **Search Types:**
    - **semantic**: Vector similarity using pgvector cosine distance
      - Generates query embedding via OpenAI
      - Compares against stored document embeddings
      - Returns results with similarity scores (0.0 - 1.0)
      - Minimum threshold: 0.5
    - **full_text**: ILIKE-based keyword matching
      - Searches title and content fields
      - Case-insensitive
    - **hybrid**: Combination of semantic + full-text
      - Weighted scoring for best results

    **Example Request:**
    ```json
    {
      "workspace_id": "uuid",
      "query": "sustainable wood materials",
      "search_type": "semantic",
      "limit": 20
    }
    ```

    **Example Response:**
    ```json
    {
      "results": [
        {
          "id": "uuid",
          "title": "Sustainable Wood Guide",
          "similarity": 0.87,
          "content": "...",
          "category_id": "uuid"
        }
      ],
      "search_time_ms": 145.3,
      "total_results": 5
    }
    ```
    """
    try:
        import time
        start_time = time.time()

        if request.search_type == "semantic":
            # Generate embedding for search query
            embeddings_service = RealEmbeddingsService()
            embedding_result = await embeddings_service.generate_all_embeddings(
                entity_id="search_query",
                entity_type="search",
                text_content=request.query
            )

            if not embedding_result.get("success"):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to generate query embedding: {embedding_result.get('error')}"
                )

            query_embedding = embedding_result.get("embeddings", {}).get("text_1536")
            if not query_embedding:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No embedding generated for query"
                )

            # Perform vector similarity search
            response = supabase_client.client.rpc(
                'match_kb_docs',
                {
                    'query_embedding': query_embedding,
                    'match_workspace_id': request.workspace_id,
                    'match_threshold': 0.5,
                    'match_count': request.limit
                }
            ).execute()

            results = response.data if response.data else []

        else:
            # Use the search_kb_docs RPC function for full-text and hybrid
            response = supabase_client.client.rpc(
                'search_kb_docs',
                {
                    'search_query': request.query,
                    'search_workspace_id': request.workspace_id,
                    'search_type': request.search_type,
                    'result_limit': request.limit
                }
            ).execute()

            results = response.data if response.data else []

        end_time = time.time()
        search_time_ms = (end_time - start_time) * 1000

        return SearchKBResponse(
            results=results,
            search_time_ms=search_time_ms,
            total_results=len(results)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/health")
async def kb_health_check() -> Dict[str, Any]:
    """Health check endpoint for Knowledge Base API."""
    return {
        "status": "healthy",
        "service": "knowledge-base-api",
        "version": "1.0.0",
        "features": {
            "document_crud": True,
            "embedding_generation": True,
            "semantic_search": True,
            "pdf_extraction": True,
            "product_attachment": True,
            "categories": True
        },
        "endpoints": {
            "create_document": "/api/kb/documents",
            "create_from_pdf": "/api/kb/documents/from-pdf",
            "search": "/api/kb/search",
            "categories": "/api/kb/categories",
            "attachments": "/api/kb/attachments"
        }
    }

