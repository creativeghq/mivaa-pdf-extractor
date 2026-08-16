"""
Knowledge Base API Routes

This module provides API endpoints for Knowledge Base document management,
including CRUD operations, embedding generation, PDF text extraction, and semantic search.
"""

import logging
from typing import Dict, Any, Optional, List
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File
from pydantic import BaseModel, Field
from datetime import datetime

from app.services.core.supabase_client import SupabaseClient
from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
from app.schemas.api_responses import KBHealthResponse
from app.dependencies import get_workspace_context, WorkspaceContext, get_current_user, resolve_workspace_id
from app.services.kb.kb_access import (
    kb_query_vector,
    resolve_kb_access_scope,
    resolve_kb_caller,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kb", tags=["Knowledge Base"])


def _assert_own_workspace(ctx: WorkspaceContext, body_workspace_id: str) -> None:
    """Pentest #250 G2: these routes insert with a service-role client (RLS-bypassing)
    and trust a body-supplied workspace_id. Reconcile it against the caller's JWT
    workspace (CLAUDE.md invariant #9) so an authenticated user of workspace A can't
    write into workspace B by passing its id. Admins of the target workspace pass."""
    caller_ws = getattr(ctx, "workspace_id", None)
    if not caller_ws or str(caller_ws) != str(body_workspace_id):
        raise HTTPException(status_code=403, detail="workspace_id does not match your session")


def _require_uuid(doc_id: str) -> None:
    """Reject non-UUID path segments before they reach the DB and trigger a
    Postgres 22P02 error. Stale frontends sometimes hit `/documents/from-pdf`
    (the POST path) with GET; treat that as a 404 instead of a 500."""
    try:
        UUID(doc_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=404, detail="Document not found")


def _load_own_doc(
    doc_id: str,
    ctx: WorkspaceContext,
    supabase_client: SupabaseClient,
) -> Dict[str, Any]:
    """Fetch a kb_doc and prove the CALLER's workspace owns it, or 404.

    Issue #15: `GET`/`PATCH`/`DELETE /api/kb/documents/{doc_id}` and the two
    attachment readers took a doc_id straight off the path and queried it with the
    service-role client — no workspace predicate anywhere. The audit asked whether
    those routes were shadowed by a gated twin the way `management_routes.py` is.
    They are NOT: `/api/kb` is a prefix no other router declares, so they were live,
    and any authenticated member of ANY workspace could read, edit or delete another
    tenant's KB document by id. MIVAA holds no RLS backstop (service-role client), so
    this predicate is the only thing between the two tenants.

    404 rather than 403 on mismatch, per invariant 1 — a 403 confirms the id exists
    and turns the endpoint into an existence oracle.
    """
    _require_uuid(doc_id)
    caller_ws = getattr(ctx, "workspace_id", None)
    if not caller_ws:
        raise HTTPException(status_code=404, detail="Document not found")

    result = supabase_client.client.table("kb_docs").select("*").eq("id", doc_id).execute()
    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Document not found")

    row = rows[0]
    if str(row.get("workspace_id")) != str(caller_ws):
        logger.warning(
            "KB doc %s requested by user %s of workspace %s — refusing (cross-tenant)",
            doc_id, getattr(ctx, "user_id", None), caller_ws,
        )
        raise HTTPException(status_code=404, detail="Document not found")
    return row


# `_caller_is_workspace_admin` lived here until the KB derivations were unified. It
# read `workspace_members.role` to decide admin-ness — correct, but it was the SECOND
# implementation of "who is this caller really", next to rag_routes' body-supplied
# `caller` field. Fixing MV2-12 by adding another private copy is what prompted this
# extraction. The one implementation is `resolve_kb_caller` in
# app/services/kb/kb_access.py, and it now also clamps the widening request that the
# rag path was honouring unchecked.


# ============================================================================
# Request/Response Models
# ============================================================================

PRICE_DOC_TYPES = {"price_list", "discount_rule", "contract_terms", "promotion"}


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
    price_doc_type: Optional[str] = Field(
        None,
        description="Sub-type for pricing docs: price_list | discount_rule | contract_terms | promotion"
    )

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
    price_doc_type: Optional[str] = Field(
        None,
        description="Sub-type for pricing docs: price_list | discount_rule | contract_terms | promotion"
    )


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
    price_doc_type: Optional[str] = None





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
    supabase_client: SupabaseClient = Depends(), current_user: dict = Depends(get_current_user)
) -> KBDocResponse:
    """Create a new knowledge base document with embeddings.

    Upserts by (workspace_id, title, category_id): if a doc with the same title
    and category already exists, updates it in place and re-embeds if content changed.
    """
    # Bind the caller-supplied workspace to the authenticated identity (invariant 1).
    workspace_id = await resolve_workspace_id(current_user, request.workspace_id)
    return await _upsert_kb_document(
        request, supabase_client, workspace_id,
        user_id=str((current_user or {}).get("sub") or "") or None,
    )


async def _upsert_kb_document(
    request: CreateKBDocRequest,
    supabase_client: SupabaseClient,
    workspace_id: Optional[str],
    user_id: Optional[str] = None,
) -> KBDocResponse:
    """Create-or-update a KB doc against an ALREADY-RESOLVED workspace_id.

    MV2-15: `create_kb_document_from_pdf` used to finish with
    `await create_kb_document(create_request, supabase_client)` — calling the route
    function directly, two positional args, so `current_user` kept its DEFAULT value,
    which is the `Depends(get_current_user)` marker object itself. That object was then
    handed to `resolve_workspace_id`, which reads `.get("sub")` off it. FastAPI only
    resolves dependencies for requests it ROUTES; a direct call gets the marker.
    `/api/kb/documents/from-pdf` therefore failed 100% of the time, silently, in the
    body of a `try` that reported it as a generic 500 — the third endpoint in this repo
    found sitting at a 100% failure rate with nothing complaining.

    The workspace is resolved by the CALLER and passed in, so this helper cannot be
    invoked without one having been bound to a verified identity first.
    """
    try:
        if request.price_doc_type is not None and request.price_doc_type not in PRICE_DOC_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid price_doc_type. Must be one of: {sorted(PRICE_DOC_TYPES)}"
            )

        existing_query = (
            supabase_client.client.table("kb_docs")
            .select("id, content, title, summary, category_id")
            .eq("workspace_id", workspace_id)
            .eq("title", request.title)
        )
        if request.category_id:
            existing_query = existing_query.eq("category_id", request.category_id)
        else:
            existing_query = existing_query.is_("category_id", "null")
        existing = existing_query.execute()

        if existing.data:
            existing_doc = existing.data[0]
            content_changed = existing_doc.get("content") != request.content
            update_payload: Dict[str, Any] = {
                "content": request.content,
                "content_markdown": request.content_markdown,
                "summary": request.summary,
                "category_id": request.category_id,
                "seo_keywords": request.seo_keywords,
                "status": request.status,
                "visibility": request.visibility,
                "metadata": request.metadata,
                "price_doc_type": request.price_doc_type,
                "updated_at": datetime.utcnow().isoformat(),
            }

            if content_changed:
                embeddings_service = RealEmbeddingsService()
                embedding_result = await embeddings_service.generate_all_embeddings(
                    entity_id="temp",
                    entity_type="kb_doc",
                    text_content=request.content,
                    # MV2-14: this is a paid Voyage call. It carried NO tenant and NO
                    # payer, so every KB embedding landed in ai_usage_logs with
                    # workspace_id NULL — invisible to per-workspace cost views and
                    # unmatchable by that table's own is_workspace_admin() policy.
                    workspace_id=workspace_id,
                    user_id=user_id,
                )
                if embedding_result.get("success"):
                    update_payload["text_embedding"] = embedding_result.get("embeddings", {}).get("text_1024")
                    update_payload["embedding_status"] = "success"
                    update_payload["embedding_generated_at"] = datetime.utcnow().isoformat()
                    update_payload["embedding_error_message"] = None
                else:
                    update_payload["embedding_status"] = "failed"
                    update_payload["embedding_error_message"] = embedding_result.get("error", "Unknown error")

            result = (
                supabase_client.client.table("kb_docs")
                .update(update_payload)
                .eq("id", existing_doc["id"])
                .execute()
            )
            if not result.data:
                raise HTTPException(status_code=500, detail="Failed to update existing document")
            return KBDocResponse(**result.data[0])

        # Fresh insert path — generate embedding
        embeddings_service = RealEmbeddingsService()
        embedding_result = await embeddings_service.generate_all_embeddings(
            entity_id="temp",
            entity_type="kb_doc",
            text_content=request.content,
            workspace_id=workspace_id,
            user_id=user_id,
        )

        text_embedding = None
        embedding_status = "pending"
        embedding_error = None

        if embedding_result.get("success"):
            text_embedding = embedding_result.get("embeddings", {}).get("text_1024")
            embedding_status = "success"
        else:
            embedding_error = embedding_result.get("error", "Unknown error")
            embedding_status = "failed"

        doc_data = {
            "workspace_id": workspace_id,
            "title": request.title,
            "content": request.content,
            "content_markdown": request.content_markdown,
            "summary": request.summary,
            "category_id": request.category_id,
            "seo_keywords": request.seo_keywords,
            "status": request.status,
            "visibility": request.visibility,
            "metadata": request.metadata,
            "price_doc_type": request.price_doc_type,
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

    except HTTPException:
        raise
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
    supabase_client: SupabaseClient = Depends(),
    ctx: WorkspaceContext = Depends(get_workspace_context),
) -> KBDocResponse:
    """Get a knowledge base document by ID."""
    try:
        return KBDocResponse(**_load_own_doc(doc_id, ctx, supabase_client))
    except HTTPException:
        raise  # 404 from the ownership check — never repackage it as a 500
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
    supabase_client: SupabaseClient = Depends(),
    ctx: WorkspaceContext = Depends(get_workspace_context),
) -> KBDocResponse:
    """Update a knowledge base document."""
    try:
        # Get current document — and prove this workspace owns it before editing.
        current_doc = _load_own_doc(doc_id, ctx, supabase_client)

        # Check if content changed OR embedding is missing/pending/failed
        force_regenerate = bool(
            request.metadata and request.metadata.get("force_regenerate")
        )
        embedding_missing = current_doc.get("embedding_status") != "success"
        content_changed = (
            request.title and request.title != current_doc.get("title") or
            request.content and request.content != current_doc.get("content") or
            request.summary and request.summary != current_doc.get("summary") or
            request.category_id and request.category_id != current_doc.get("category_id")
        )

        if request.price_doc_type is not None and request.price_doc_type not in PRICE_DOC_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid price_doc_type. Must be one of: {sorted(PRICE_DOC_TYPES)}"
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
            # Strip internal force_regenerate flag before persisting
            meta = {k: v for k, v in request.metadata.items() if k != "force_regenerate"}
            if meta:
                update_data["metadata"] = meta
        if request.price_doc_type is not None:
            update_data["price_doc_type"] = request.price_doc_type

        # Regenerate embedding if content changed, embedding is missing, or explicitly forced
        if content_changed or embedding_missing or force_regenerate:
            embeddings_service = RealEmbeddingsService()
            embedding_result = await embeddings_service.generate_all_embeddings(
                entity_id="temp",
                entity_type="kb_doc",
                text_content=request.content or current_doc.get("content"),
                workspace_id=str(ctx.workspace_id),
                user_id=str(getattr(ctx, "user_id", "") or "") or None,
            )
            
            if embedding_result.get("success"):
                update_data["text_embedding"] = embedding_result.get("embeddings", {}).get("text_1024")
                update_data["embedding_status"] = "success"
                update_data["embedding_generated_at"] = datetime.utcnow().isoformat()
            else:
                update_data["embedding_status"] = "failed"
                update_data["embedding_error_message"] = embedding_result.get("error")
        
        update_data["updated_at"] = datetime.utcnow().isoformat()
        
        # workspace_id predicate as well as the id: ownership is already proven
        # above, but a write that carries its tenant filter cannot be turned into a
        # cross-tenant write by a later edit that moves the check.
        result = supabase_client.client.table("kb_docs").update(update_data)\
            .eq("id", doc_id)\
            .eq("workspace_id", str(ctx.workspace_id))\
            .execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to update document")

        return KBDocResponse(**result.data[0])

    except HTTPException:
        raise  # 404 (not yours / not found) and 400 (bad price_doc_type) stay as-is
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
    supabase_client: SupabaseClient = Depends(),
    ctx: WorkspaceContext = Depends(get_workspace_context),
) -> None:
    """Delete a knowledge base document."""
    try:
        _load_own_doc(doc_id, ctx, supabase_client)
        supabase_client.client.table("kb_docs").delete()\
            .eq("id", doc_id)\
            .eq("workspace_id", str(ctx.workspace_id))\
            .execute()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting KB document: {str(e)}")
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
    supabase_client: SupabaseClient = Depends(), current_user: dict = Depends(get_current_user)
) -> KBDocResponse:
    """Create a knowledge base document from PDF file."""
    # Bind the caller-supplied workspace to the authenticated identity (invariant 1).
    workspace_id = await resolve_workspace_id(current_user, workspace_id)
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

        # The shared helper, NOT the route function — see `_upsert_kb_document`.
        # `workspace_id` here is already the resolved value from the top of this route.
        return await _upsert_kb_document(
            create_request, supabase_client, workspace_id,
            user_id=str((current_user or {}).get("sub") or "") or None,
        )

    except HTTPException:
        raise
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
    access_level: str = Field(default="agent", description="Access level: admin | agent | public")
    trigger_keyword: Optional[str] = Field(None, description="Agent searches this category only when query contains this keyword (agent-level only, case-insensitive)")


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
    access_level: str = "agent"
    trigger_keyword: Optional[str] = None


@router.post(
    "/categories",
    response_model=CategoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a category",
    description="Create a new knowledge base category"
)
async def create_category(
    request: CreateCategoryRequest,
    supabase_client: SupabaseClient = Depends(),
    ctx: WorkspaceContext = Depends(get_workspace_context),
) -> CategoryResponse:
    """Create a new category."""
    _assert_own_workspace(ctx, request.workspace_id)
    try:
        # Allowlisted payload, never `request.dict()` (invariant 8). The model's fields happen
        # to match the table today, so the spread looked harmless — but it binds the write to
        # whatever the model grows next. kb_categories carries `is_locked` and
        # `material_category_id`, both server-owned; a field added to the model with either
        # name would start setting them from the request body with no code change and no
        # review signal.
        payload = {
            "workspace_id": request.workspace_id,
            "name": request.name,
            "slug": request.slug,
            "description": request.description,
            "icon": request.icon,
            "color": request.color,
            "parent_category_id": request.parent_category_id,
            "access_level": request.access_level,
            "trigger_keyword": request.trigger_keyword,
        }
        result = supabase_client.client.table("kb_categories").insert(payload).execute()

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
    supabase_client: SupabaseClient = Depends(), current_user: dict = Depends(get_current_user)
) -> List[CategoryResponse]:
    """List all categories for a workspace."""
    # Bind the caller-supplied workspace to the authenticated identity (invariant 1).
    workspace_id = await resolve_workspace_id(current_user, workspace_id)
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
    supabase_client: SupabaseClient = Depends(),
    ctx: WorkspaceContext = Depends(get_workspace_context),
) -> AttachmentResponse:
    """Attach a document to a product."""
    _assert_own_workspace(ctx, request.workspace_id)
    try:
        # MV2-13: the workspace was reconciled against the JWT above, but document_id
        # and product_id were then inserted unchecked — two ids each individually
        # valid, never checked against EACH OTHER or against the workspace. That let a
        # caller staple another tenant's KB document onto their own product (and the
        # reverse), which `get_product_documents` above would then happily serve.
        # Verify both ends belong to the caller's workspace before writing the edge.
        _require_uuid(request.document_id)
        _require_uuid(request.product_id)
        _load_own_doc(request.document_id, ctx, supabase_client)

        product = supabase_client.client.table("products").select("id")\
            .eq("id", request.product_id)\
            .eq("workspace_id", request.workspace_id)\
            .execute()
        if not (product.data or []):
            logger.warning(
                "KB attachment refused: product %s is not in workspace %s",
                request.product_id, request.workspace_id,
            )
            raise HTTPException(status_code=404, detail="Product not found")

        # Allowlisted payload, never `request.dict()` (invariant 8) — same reasoning as
        # create_category above.
        payload = {
            "workspace_id": request.workspace_id,
            "document_id": request.document_id,
            "product_id": request.product_id,
            "relationship_type": request.relationship_type,
            "relevance_score": request.relevance_score,
        }
        result = supabase_client.client.table("kb_doc_attachments").insert(payload).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create attachment")

        return AttachmentResponse(**result.data[0])

    except HTTPException:
        raise  # the 404s above are the answer, not an internal error
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
    supabase_client: SupabaseClient = Depends(),
    ctx: WorkspaceContext = Depends(get_workspace_context),
) -> List[AttachmentResponse]:
    """Get all product attachments for a document."""
    try:
        _load_own_doc(doc_id, ctx, supabase_client)
        result = supabase_client.client.table("kb_doc_attachments").select("*")\
            .eq("document_id", doc_id)\
            .eq("workspace_id", str(ctx.workspace_id))\
            .execute()

        return [AttachmentResponse(**att) for att in result.data or []]

    except HTTPException:
        raise
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
    supabase_client: SupabaseClient = Depends(),
    ctx: WorkspaceContext = Depends(get_workspace_context),
) -> List[KBDocResponse]:
    """Get all documents attached to a product."""
    _require_uuid(product_id)
    try:
        workspace_id = str(ctx.workspace_id)

        # Get attachments — scoped to the caller's workspace. Unscoped, a product_id
        # from another tenant returned that tenant's attachment rows, and the doc
        # fetch below then returned their KB documents in full.
        attachments = supabase_client.client.table("kb_doc_attachments").select("document_id")\
            .eq("product_id", product_id)\
            .eq("workspace_id", workspace_id)\
            .execute()

        if not attachments.data:
            return []

        doc_ids = [att["document_id"] for att in attachments.data]

        # Get documents. The workspace predicate repeats here rather than relying on
        # the attachment scope: the two tables are joined by an id that the caller
        # influenced, so each side carries its own tenant filter.
        result = supabase_client.client.table("kb_docs").select("*")\
            .in_("id", doc_ids)\
            .eq("workspace_id", workspace_id)\
            .execute()

        return [KBDocResponse(**doc) for doc in result.data or []]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching product documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class SearchKBRequest(BaseModel):
    """Request model for searching knowledge base documents."""
    workspace_id: str = Field(..., description="UUID of the workspace")
    query: str = Field(..., description="Search query", min_length=1)
    search_type: str = Field(default="semantic", description="Search type: semantic, full_text, or hybrid")
    limit: int = Field(default=20, description="Maximum number of results", ge=1, le=100)
    category_id: Optional[str] = Field(None, description="Restrict search to a single category UUID")
    category_slug: Optional[str] = Field(None, description="Restrict search to a category by slug (e.g. 'pricing')")
    price_doc_type: Optional[str] = Field(
        None,
        description="Restrict to pricing sub-type: price_list | discount_rule | contract_terms | promotion"
    )
    require_published: bool = Field(
        default=False,
        description="When true, only published docs are returned (semantic only). "
                    "Default False preserves admin management search behavior — "
                    "agent-facing search should pass True explicitly."
    )
    # MV2-12: `is_admin_caller` and `allowed_access_levels` USED TO LIVE HERE, as
    # request-body fields fed straight into `kb_match_docs(include_private => ...)`
    # and its access-level gate — one of which was documented, in the API contract,
    # as "Overrides category access gating". Any authenticated member could send
    # `{"is_admin_caller": true}` and read the workspace's private KB.
    #
    # They are gone rather than defaulted-and-ignored: a field the published OpenAPI
    # still advertises is one a caller still sends, and the next person to wire it up
    # re-opens the hole. Both values are now derived server-side in the handler from
    # `workspace_members.role`. Pydantic ignores unknown keys, so an existing caller
    # that still sends them keeps working — it just no longer gets to decide.
    match_threshold: float = Field(default=0.5, description="Minimum similarity for semantic search", ge=0.0, le=1.0)


class SearchKBDocResult(BaseModel):
    """Individual search result document."""
    id: str
    workspace_id: str
    title: str
    content: str
    summary: Optional[str] = None
    category_id: Optional[str] = None
    category_slug: Optional[str] = None
    category_name: Optional[str] = None
    status: str
    visibility: str
    embedding_status: str
    embedding_generated_at: Optional[str] = None
    created_by: Optional[str] = None
    created_at: str
    updated_at: str
    view_count: int
    price_doc_type: Optional[str] = None
    similarity: Optional[float] = None  # Only for semantic search


class SearchKBResponse(BaseModel):
    """Response model for search results."""
    results: List[SearchKBDocResult]
    search_time_ms: float
    total_results: int


@router.post("/search", response_model=SearchKBResponse)
async def search_kb_documents(
    request: SearchKBRequest,
    supabase_client: SupabaseClient = Depends(), current_user: dict = Depends(get_current_user)
) -> SearchKBResponse:
    """
    Search knowledge base documents using semantic, full-text, or hybrid search.

    **Architecture:**
    1. Frontend calls MIVAA API with search query
    2. MIVAA generates embedding for query using OpenAI (text-embedding-3-small)
    3. MIVAA calls Supabase `kb_match_docs()` RPC function with query embedding
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
    # Bind the caller-supplied workspace to the authenticated identity (invariant 1).
    workspace_id = await resolve_workspace_id(current_user, request.workspace_id)

    # MV2-12: derive the read scope from who the caller IS, not from what they sent.
    # `include_private` and the access-level list are the same gate expressed twice, so
    # they are resolved together and once — a non-admin cannot widen either.
    #
    # Both derivations now come from app/services/kb/kb_access.py, shared with
    # /api/rag/search/knowledge-base. This endpoint used to keep private copies of
    # both, which is how its query vector ended up built in "document" mode while the
    # sibling built its in "query" mode against the same document-side vectors.
    #
    # per_doc_agent_gate=False: this route runs `kb_match_docs`, which — unlike the
    # agent path's `kb_match_doc_chunks` — applies NO per-doc `allowed_agents` or
    # category access_level gate. `include_private` is therefore the only thing between
    # a non-admin and private content here, so it must track admin-ness rather than
    # following the agent path's "private just means unpublished" rule.
    caller = await resolve_kb_caller(supabase_client, current_user, None, workspace_id)
    kb_scope = resolve_kb_access_scope(
        supabase_client, str(workspace_id), caller, request.query,
        per_doc_agent_gate=False,
    )
    is_admin_caller = kb_scope["include_private"]
    allowed_access_levels = kb_scope["allowed_access_levels"]

    try:
        import time
        start_time = time.time()

        if request.search_type == "semantic":
            # ONE derivation — `entity_type="search"` used to be passed here, which
            # falls through `input_type = "query" if entity_type == "query" else
            # "document"` and embedded the QUERY as a DOCUMENT. Same model, same 1024
            # dimensions, same column, plausible ranked results, quietly worse ranking.
            # The sibling endpoint had already found and fixed this; the fix did not
            # reach the copy. kb_query_vector is now the only way to build one.
            query_embedding = await kb_query_vector(
                request.query,
                workspace_id=str(workspace_id) if workspace_id else None,
                user_id=str((current_user or {}).get("sub") or "") or None,
            )
            if not query_embedding:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No embedding generated for query"
                )

            # Perform vector similarity search
            rpc_args: Dict[str, Any] = {
                'query_embedding': query_embedding,
                'match_workspace_id': workspace_id,
                'match_threshold': request.match_threshold,
                'match_count': request.limit,
                'require_published': request.require_published,
                'include_private': is_admin_caller,
                'allowed_access_levels': allowed_access_levels,
            }
            if request.category_id:
                rpc_args['match_category_id'] = request.category_id
            if request.category_slug:
                rpc_args['match_category_slug'] = request.category_slug
            if request.price_doc_type:
                rpc_args['match_price_doc_type'] = request.price_doc_type

            response = supabase_client.client.rpc('kb_match_docs', rpc_args).execute()

            raw_results = response.data if response.data else []

        else:
            # Use the kb_search_docs RPC function for full-text and hybrid
            rpc_args = {
                'search_query': request.query,
                'search_workspace_id': workspace_id,
                'search_type': request.search_type,
                'result_limit': request.limit,
                'include_private': is_admin_caller,
                'allowed_access_levels': allowed_access_levels,
            }
            if request.category_id:
                rpc_args['match_category_id'] = request.category_id
            if request.category_slug:
                rpc_args['match_category_slug'] = request.category_slug
            if request.price_doc_type:
                rpc_args['match_price_doc_type'] = request.price_doc_type

            response = supabase_client.client.rpc('kb_search_docs', rpc_args).execute()

            raw_results = response.data if response.data else []

        end_time = time.time()
        search_time_ms = (end_time - start_time) * 1000

        # Validate and convert raw results to typed models
        validated_results = []
        for raw_doc in raw_results:
            try:
                # Ensure all required fields exist with defaults
                doc_data = {
                    'id': raw_doc.get('id', ''),
                    'workspace_id': raw_doc.get('workspace_id', workspace_id),
                    'title': raw_doc.get('title', 'Untitled'),
                    'content': raw_doc.get('content', ''),
                    'summary': raw_doc.get('summary'),
                    'category_id': raw_doc.get('category_id'),
                    'category_slug': raw_doc.get('category_slug'),
                    'category_name': raw_doc.get('category_name'),
                    'status': raw_doc.get('status', 'draft'),
                    'visibility': raw_doc.get('visibility', 'workspace'),
                    'embedding_status': raw_doc.get('embedding_status', 'pending'),
                    'embedding_generated_at': raw_doc.get('embedding_generated_at'),
                    'created_by': raw_doc.get('created_by'),
                    'created_at': raw_doc.get('created_at', datetime.now().isoformat()),
                    'updated_at': raw_doc.get('updated_at', datetime.now().isoformat()),
                    'view_count': raw_doc.get('view_count', 0),
                    'price_doc_type': raw_doc.get('price_doc_type'),
                    'similarity': raw_doc.get('similarity')  # Only present in semantic search
                }
                validated_results.append(SearchKBDocResult(**doc_data))
            except Exception as validation_error:
                logger.warning(f"Failed to validate search result: {validation_error}, raw_doc: {raw_doc}")
                continue

        return SearchKBResponse(
            results=validated_results,
            search_time_ms=search_time_ms,
            total_results=len(validated_results)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/health", response_model=KBHealthResponse)
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


