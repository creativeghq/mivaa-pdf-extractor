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
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Query, status, BackgroundTasks, Request
from fastapi.responses import JSONResponse
import asyncio
import aiohttp
import sentry_sdk
from pydantic import BaseModel, Field, model_validator

from app.config import get_settings
from app.auth.workspace_resolution import billing_user_id
from app.services.search.rag_service import RAGService
from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
from app.services.products.product_creation_service import ProductCreationService
from app.services.tracking.job_recovery_service import JobRecoveryService
from app.services.tracking.checkpoint_recovery_service import checkpoint_recovery_service, ProcessingStage
from app.services.core.supabase_client import get_supabase_client, SupabaseClient
from app.services.products.product_relationship_service import ProductRelationshipService
from app.services.search.search_prompt_service import SearchPromptService
from app.services.tracking.stuck_job_analyzer import stuck_job_analyzer
from app.services.embeddings.vecs_service import get_vecs_service
from app.utils.resource_manager import get_resource_manager
from app.schemas.api_responses import (
    StatusResponse, ListDataResponse, JobInfoResponse,
    CheckpointListResponse, RelevancyListResponse, StatsResponse,
    AITrackingResponse, StuckJobsResponse, DocumentContentResponse,
)
from app.dependencies import current_user_id, get_current_user, get_optional_workspace_context, resolve_workspace_id, verify_internal_access
from app.utils.untrusted_content import as_untrusted_data
from app.utils.pdf_bounds import PdfBoundsError, assert_page_count
# NOTE: `authorize_rag_workspace` is imported at the BOTTOM of this module, not here.
# Importing it at the top triggers `app.api.documents.__init__` →
# `management_routes` → `app.orchestration` → back into this (still partially
# initialized) module, raising a circular-import ImportError on startup
# (Sentry MIVAA-5HQ). It is only used inside request handlers at runtime, so
# deferring the import to end-of-module — after `job_storage` and
# `process_document_with_discovery` are defined — breaks the cycle.

logger = logging.getLogger(__name__)


def _fail_job_terminally(job_id: str, reason: str, *, source: str) -> None:
    """Mark a job failed AND write the terminal stage_history event (#35 M20-1).

    Pipeline convention 9: the audit log must show why a job ended. Before this, the
    pre-stage validations raised BARE — a missing file, an unstattable file, a zero-byte
    PDF — outside any orchestrator handler, so the function exited with
    `background_jobs.status` still `processing` and no terminal event at all.

    That is the state the rest of the system handles worst, and the interaction is
    three-way:
      * auto-recovery hunts jobs stuck in `processing` and judges staleness on
        `updated_at` (#18 M5-5, mivaa#12)
      * `reprocess` refuses to act on a document whose job is pending/processing
        (#34 M19-3) — so a stranded job also BLOCKS the obvious way to retry it
      * and none of it is visible, because the job simply stops

    A file that does not exist is the most ordinary failure this pipeline has, and it
    produced the worst state. Best-effort and never raises: the caller is already
    failing, and a bookkeeping error must not replace the real one.
    """
    try:
        sb = get_supabase_client()
        sb.client.table("background_jobs").update({
            "status": "failed",
            "error_message": reason[:1000],
            "completed_at": datetime.utcnow().isoformat(),
        }).eq("id", job_id).execute()
    except Exception as e:  # noqa: BLE001
        logger.error("Could not mark job %s failed (%s): %s", job_id, source, e)
    try:
        sb = get_supabase_client()
        sb.client.rpc("append_stage_history", {
            "p_job_id": job_id,
            "p_event": {
                "stage": source,
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "data": {"error": reason[:500]},
                "source": source,
            },
        }).execute()
    except Exception as e:  # noqa: BLE001
        logger.error("Could not append terminal stage_history for %s: %s", job_id, e)


async def require_rag_resource_access(
    request: Request,
    workspace_context = Depends(get_optional_workspace_context),
):
    """FastAPI dependency authorizing a caller against the workspace that owns the resource a
    sensitive /api/rag route operates on — resolved from the request (job_id → background_jobs,
    document_id → documents, path param or ?document_id query).

    The /api/rag prefix is excluded from the global JWT middleware (it's called by edge functions
    with a service-role JWT), so destructive/read routes here would otherwise be reachable
    UNAUTHENTICATED by anyone who can guess a document_id/job_id. This mirrors resume_job's
    in-body guard: allow the x-cron-secret automation bypass, else require a JWT whose workspace
    owns the target. Attached via the route decorator's `dependencies=[...]` so it runs ONLY for
    real HTTP calls — internal callers (e.g. resume_job -> restart_job_from_checkpoint) invoke the
    handler directly and are unaffected. Fail-closed on any lookup error.
    """
    cron_secret_header = (request.headers.get("x-cron-secret") or "").strip()
    expected_cron_secret = (os.getenv("CRON_SECRET") or "").strip()
    if expected_cron_secret and cron_secret_header == expected_cron_secret:
        return  # trusted automation (auto-recovery cron)

    if workspace_context is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a Bearer JWT, or an x-cron-secret header for trusted automation.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    job_id = request.path_params.get("job_id")
    document_id = request.path_params.get("document_id") or request.query_params.get("document_id")
    if job_id:
        table, id_value, label = "background_jobs", job_id, "job"
    elif document_id:
        table, id_value, label = "documents", document_id, "document"
    else:
        # A list endpoint with no resource id would otherwise return rows across all tenants.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A document_id or job_id is required.",
        )

    try:
        sb = get_supabase_client()
        row = sb.client.table(table).select("workspace_id").eq("id", str(id_value)).single().execute()
        if not row.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{label} not found")
        resource_ws = str(row.data.get("workspace_id") or "")
        if resource_ws and resource_ws != str(workspace_context.workspace_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"You are not a member of the workspace that owns this {label}.",
            )
    except HTTPException:
        raise
    except Exception as ownership_err:
        logger.error(
            f"require_rag_resource_access({table}, {id_value}) raised: {ownership_err}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not verify {label} ownership; refusing the operation.",
        )


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

def _require_uuid(value: Any, field: str) -> str:
    """Return `value` as a string, or raise 400 when it is not a UUID.

    A document id reaches these routes from an AGENT, and an agent that has not yet
    looked one up will happily send the thing it was searching for — the live case was
    `kb_doc_id="tile"`. Handed straight to the RPC, Postgres raised `22P02 invalid
    input syntax for type uuid`, which the route's `except Exception` re-raised as a
    500. A caller sending the wrong TYPE is a 400; reporting it as a 500 puts a
    client-side mistake in the platform's error budget and buries the real ones.

    Deliberately only a SYNTAX check. Whether the id exists, and whether this caller
    may see it, is decided by the RPC and reported as 404 — proving a document exists
    before checking access would turn the route into an enumeration oracle.
    """
    from uuid import UUID

    try:
        UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field} must be a UUID",
        )
    return str(value)


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
    Initialize job recovery service AND auto-resume jobs that were interrupted
    by the previous shutdown.

    Why this matters:
    - The shutdown hook (main.py lifespan) marks `processing` jobs as
      'interrupted' because uvicorn cannot drain BackgroundTasks during a
      systemd restart.
    - Without auto-resume, those jobs sat 'interrupted' until the auto-
      recovery cron noticed (up to 5 min later) — and historically the cron
      filtered on status='processing' only, so they sat stuck *indefinitely*.
    - This startup hook closes the loop: after marking, we immediately
      re-dispatch eligible jobs into the new event loop using the same
      orchestrator the upload endpoint uses (process_document_with_discovery).
      The orchestrator's checkpoint logic resumes from the last completed stage.

    Eligibility for auto-resume:
      - status was just flipped to 'interrupted' by THIS startup pass (or by
        the previous shutdown — interrupted_at within the last 4 hours).
      - recovery_attempts < 3 (same cap the cron uses).
      - The original PDF still exists on disk (temp files survive a service
        restart but not a full host reboot — fall back gracefully if missing).
    """
    global job_recovery_service

    try:
        logger.info("🔄 Initializing job recovery service...")

        supabase_client = get_supabase_client()
        job_recovery_service = JobRecoveryService(supabase_client)

        # 1. Mark anything left in 'processing' as 'interrupted' so the DB
        #    reflects reality before we attempt resume.
        interrupted_count = await job_recovery_service.mark_all_processing_as_interrupted(
            reason="Service restart detected"
        )
        if interrupted_count > 0:
            logger.warning(f"🛑 Marked {interrupted_count} jobs as interrupted due to service restart")

        # 2. Auto-resume recently interrupted jobs (root-cause fix for the
        #    "deploy kills job, nothing picks it up" stuck-state bug).
        await _resume_recently_interrupted_jobs(supabase_client)

        # Get statistics
        stats = await job_recovery_service.get_job_statistics()
        logger.info(f"📊 Job statistics: {stats}")

        logger.info("✅ Job recovery service initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize job recovery service: {e}", exc_info=True)
        # Don't fail startup if job recovery fails
        job_recovery_service = None


async def _resume_recently_interrupted_jobs(supabase_client) -> None:
    """Re-dispatch jobs interrupted within the last 30 min back into the event loop.

    Uses the same orchestrator the upload endpoint uses, so checkpoint resume
    is automatic: process_document_with_discovery reads `last_checkpoint` and
    skips already-completed stages.
    """
    from datetime import datetime, timedelta
    import os

    cutoff = (datetime.utcnow() - timedelta(hours=4)).isoformat()

    try:
        rows = supabase_client.client.table('background_jobs') \
            .select('id, document_id, filename, metadata, recovery_attempts, interrupted_at') \
            .eq('status', 'interrupted') \
            .in_('job_type', ['product_discovery_upload', 'pdf_processing']) \
            .gte('interrupted_at', cutoff) \
            .lt('recovery_attempts', 3) \
            .execute()
    except Exception as e:
        logger.error(f"⚠️ Could not query interrupted jobs for auto-resume: {e}")
        return

    candidates = rows.data or []
    if not candidates:
        logger.info("✅ No recently-interrupted jobs to auto-resume")
        return

    logger.warning(f"🔄 Auto-resuming {len(candidates)} interrupted job(s) from last 4 hours")

    # Pull file_path + workspace_id from the documents table for each.
    doc_ids = [c['document_id'] for c in candidates if c.get('document_id')]
    docs_by_id: Dict[str, Dict[str, Any]] = {}
    if doc_ids:
        try:
            docs = supabase_client.client.table('documents') \
                .select('id, file_path, workspace_id') \
                .in_('id', doc_ids).execute()
            for d in (docs.data or []):
                docs_by_id[d['id']] = d
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch document file_paths: {e}")

    resumed = 0
    skipped_no_file = 0
    skipped_no_doc = 0

    for job in candidates:
        job_id = job['id']
        doc_id = job.get('document_id')
        meta = job.get('metadata') or {}

        if not doc_id or doc_id not in docs_by_id:
            logger.warning(f"   ⏭️  Skipping {job_id}: document row missing")
            skipped_no_doc += 1
            continue

        doc_row = docs_by_id[doc_id]
        file_path = doc_row.get('file_path')
        workspace_id = doc_row.get('workspace_id') or meta.get('workspace_id')

        if not file_path or not os.path.exists(file_path):
            # Temp PDF was wiped (full host reboot, /tmp on tmpfs, etc.).
            # We can't resume without the source file — leave as 'interrupted'
            # so the admin can choose to delete or re-upload.
            logger.warning(f"   ⏭️  Skipping {job_id}: source PDF missing on disk ({file_path})")
            skipped_no_file += 1
            continue

        # Atomic claim: flip back to processing + bump recovery counter.
        # Uses the same SQL function the auto-recovery cron calls, then
        # forces status back to 'processing' (the helper sets it to 'pending'
        # but we have everything we need to run it right now).
        # Audit fix (this PR): the second UPDATE that flips pending→processing
        # is now conditional on status='pending' so a parallel cron tick that
        # already claimed (and dispatched) this job doesn't double-dispatch.
        try:
            claimed = supabase_client.client.rpc(
                'mark_pdf_job_for_recovery',
                {'p_job_id': job_id, 'p_max_attempts': 3},
            ).execute()
            if not claimed.data:
                logger.info(f"   ⏭️  {job_id}: claim no-op (already recovered or attempts exhausted)")
                continue

            from app.schemas.jobs import JobStatus as _JS_recovery
            now_iso = datetime.utcnow().isoformat()
            promote_result = supabase_client.client.table('background_jobs').update({
                'status': _JS_recovery.PROCESSING.value,
                'last_heartbeat': now_iso,
                'updated_at': now_iso,
            }).eq('id', job_id).eq('status', 'pending').execute()

            if not promote_result.data:
                # Some other path (auto-recovery cron's /resume call) already
                # promoted this job to 'processing' between our mark and
                # promote. Skip dispatch — they're already running it.
                logger.info(
                    f"   ⏭️  {job_id}: pending→processing promote was no-op "
                    f"(another path picked it up); skipping dispatch"
                )
                continue

            try:
                supabase_client.client.rpc('append_recovery_history', {
                    'p_job_id': job_id,
                    'p_event': {
                        'attempted_at': now_iso,
                        'reason': 'startup_auto_resume',
                        'attempt_number': (job.get('recovery_attempts') or 0) + 1,
                        'succeeded': True,
                    },
                }).execute()
            except Exception as e:
                logger.debug(f"append_recovery_history failed for {job_id}: {e}")

        except Exception as e:
            logger.error(f"   ❌ Failed to claim {job_id} for resume: {e}")
            continue

        # Rehydrate orchestrator parameters from saved metadata.
        categories = meta.get('categories') or ['products']
        if isinstance(categories, str):
            categories = [c.strip() for c in categories.split(',')]

        try:
            asyncio.create_task(
                process_document_with_discovery(
                    job_id=job_id,
                    document_id=doc_id,
                    file_path=file_path,
                    filename=job.get('filename') or meta.get('filename') or 'resumed.pdf',
                    title=meta.get('title'),
                    description=meta.get('description'),
                    document_tags=meta.get('tags') or [],
                    discovery_model=meta.get('discovery_model') or 'claude-vision',
                    extract_categories=categories,
                    chunk_size=meta.get('chunk_size') or 1000,
                    chunk_overlap=meta.get('chunk_overlap') or 200,
                    workspace_id=workspace_id,
                    agent_prompt=meta.get('agent_prompt'),
                    enable_prompt_enhancement=meta.get('prompt_enhancement_enabled', True),
                    test_single_product=meta.get('test_single_product', False),
                )
            )
            resumed += 1
            logger.info(f"   🚀 Resumed {job_id} (attempt {(job.get('recovery_attempts') or 0) + 1}/3)")
        except Exception as e:
            logger.error(f"   ❌ Failed to dispatch resume for {job_id}: {e}", exc_info=True)

    logger.warning(
        f"🔄 Auto-resume summary: resumed={resumed}, "
        f"skipped_no_file={skipped_no_file}, skipped_no_doc={skipped_no_doc}"
    )

# Pydantic models for request/response validation
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
    include_history: bool = Field(True, description="Use conversation_history as context for this turn")
    # The handler already read `request.conversation_history` via hasattr — the field was
    # intended and never declared, so the check was permanently False and every chat turn ran
    # with no context at all. Declaring it restores the intended contract; this endpoint is
    # stateless and persists nothing, so history is supplied by the caller (#338).
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None,
        description=(
            "Prior turns, oldest first, e.g. [{'role': 'user', 'content': '…'}]. This endpoint "
            "stores nothing between calls, so the caller owns the transcript. Ignored when "
            "include_history is false."
        ),
    )
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
    # Empty is allowed, and meaningful: "I have a picture, not words." It used to be
    # min_length=1, which 422'd every image-only search — MaterialPickerModal sends
    # `query: ''` and had therefore never worked once. The alternative callers reached for
    # was inventing a string (the search page sends the image's filename), which is worse:
    # a fabricated query is embedded and ranked against as if it meant something.
    # `validate_query_or_image` below rejects the genuinely empty request.
    query: str = Field("", max_length=1000, description="Search query. May be empty when an image is supplied.")
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

    # MMR diversity re-ranking
    enable_mmr: bool = Field(False, description="Enable MMR diversity re-ranking on results")
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0, description="MMR relevance/diversity balance (1.0=pure relevance, 0.0=pure diversity)")

    # Aspect bias (#277) — when set, the multi_vector fusion is re-weighted to
    # emphasize this one visual aspect's per-aspect embedding collection
    # (color/texture/style/material) instead of the balanced 7-vector blend, so
    # "find similar colors / textures / styles / materials" biases toward it.
    # Explicit user intent → overrides the query-understanding weight profile.
    aspect: Optional[str] = Field(None, pattern="^(color|texture|style|material)$", description="Bias multi-vector fusion toward one visual aspect: color|texture|style|material (#277)")

    @model_validator(mode="after")
    def validate_query_or_image(self):
        """A search needs something to search WITH — words or a picture.

        Relaxing `query` to allow empty removed the only thing stopping a request with
        neither, which would have run the whole fusion against an empty string and returned
        a confident ordering of nothing in particular.
        """
        if not (self.query or "").strip() and not self.image_base64 and not self.image_url:
            raise ValueError("Provide a non-empty query, or an image (image_base64 / image_url)")
        return self

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
    request: Request,
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
        description="AI model for discovery: 'claude-vision' (Claude Opus 4.7 Vision - RECOMMENDED, 10x faster), 'claude-haiku-vision' (faster/cheaper), 'claude' (text-only, legacy), 'haiku' (text-only, legacy). 'gpt'/'gpt-vision' are REJECTED - there is no OpenAI path in MIVAA (audit #12 finding 4)."
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
        # Kept as a form default because the cron path sends no JWT and relies
        # on it; the JWT path below overrides it with the caller's own
        # workspace. Read from settings so an env override is honoured rather
        # than shadowed by a literal (M3-14, #16).
        default_factory=lambda: get_settings().default_workspace_id,
        description="Workspace ID"
    ),
    # Auth (added 2026-05-23). The route was previously anonymous, allowing
    # any caller to spend Anthropic/HF/Voyage budget. We now require a valid
    # JWT and derive the workspace from JWT claims — the form-supplied
    # `workspace_id` is treated as a hint and rejected when it doesn't match
    # the caller's workspace membership.
    #
    # The dependency is OPTIONAL (get_optional_workspace_context → returns None
    # instead of raising 403 on a missing/invalid bearer). The actual auth gate
    # is enforced in the body so a trusted internal trigger carrying an
    # `x-cron-secret` (the E2E harness, cron-initiated uploads) can bypass the
    # JWT requirement — mirroring resume_job. If we kept the hard
    # Depends(get_workspace_context) here, HTTPBearer(auto_error=True) would 403
    # at the dependency layer BEFORE we could check the secret.
    workspace_context = Depends(get_optional_workspace_context),
):
    """
    **🎯 CONSOLIDATED UPLOAD ENDPOINT — Single Entry Point**

    ## 🔐 Authentication (added 2026-05-23)

    Requires a valid JWT in `Authorization: Bearer <token>`. The form-supplied
    `workspace_id` is reconciled against the caller's JWT-derived workspace:
    if it doesn't match (and isn't the platform default), the request returns
    HTTP 403. Anonymous uploads are no longer accepted.

    For uploads via the Supabase MIVAA gateway: the gateway forwards your JWT
    transparently. Pass your session token to `mivaa-gateway`; do NOT pass
    the platform service key from the frontend.

    ## 📤 File handoff (private bucket)

    `pdf-documents` is a private bucket. The frontend must:
      1. Upload the file to `{user_id}/{timestamp}-{filename}` in `pdf-documents`.
      2. Mint a short-lived signed URL via `supabase.storage.from('pdf-documents').createSignedUrl(path, 3600)`.
      3. POST the SIGNED URL to this endpoint via the `file_url` form field
         (or attach the multipart `file` directly).
    NEVER pass `getPublicUrl()` on `pdf-documents` — that returns a 403 URL.
    The backend persists `storage_bucket` + `storage_object_path` (not `file_url`)
    so resume can re-mint a fresh signed URL via service role.

    ## 🎨 Category-Based Extraction

    Control what gets extracted:
    - `categories="products"` - Extract only products
    - `categories="certificates"` - Extract only certificates
    - `categories="logos"` - Extract only logos
    - `categories="specifications"` - Extract only specifications
    - `categories="products,certificates"` - Extract multiple categories
    - `categories="all"` - Extract everything (default - comprehensive deep analysis)
    - `categories="extract_only"` - **STAGE 1.5 ONLY mode**: runs document
      layout precompute and then completes. NO discovery, NO products, NO
      chunks, NO embeddings, NO quality. Cannot be combined with other
      categories (returns 400). Use when you want raw layout extraction
      without paying for the full pipeline.

    ## 🌐 URL Processing

    Upload from URL instead of file:
    - Set `file_url="<signed-url-on-pdf-documents>"`
    - Leave `file` parameter empty
    - System downloads and processes immediately

    ## 🤖 Discovery Model

    Stage 0 product discovery is currently **text-based** regardless of the
    `discovery_model` choice. The model parameter selects which Claude model
    handles the JSON discovery prompt; it does NOT send page images.
    Catalogs with product names rendered as part of page images
    (custom display fonts, logos) will silently miss those products —
    this is a known limitation, not a bug. See CLAUDE.md "Where vision
    actually runs" for the full breakdown.

    Choose:
    - `discovery_model="claude-vision"` - Claude Opus 4.7 (default)
    - `discovery_model="claude-haiku-vision"` - Claude Haiku 4.5 (cheaper, fast)

    Real vision DOES run at Stage 3 per-image material analysis (Claude Opus
    4.7 with Anthropic `tool_use` + the `VISION_ANALYSIS_TOOL` schema lock).

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
        # Fix B/C: refuse new uploads while a deploy drain is in progress so
        # we don't start a new job that's about to be killed by systemctl restart.
        try:
            from app.api.admin import is_draining
            if is_draining():
                raise HTTPException(
                    status_code=503,
                    detail="MIVAA is draining for a deploy — please retry in a few seconds.",
                    headers={"Retry-After": "30"},
                )
        except HTTPException:
            raise
        except Exception:
            # Don't block uploads if the draining check itself errors
            pass

        # ── Auth gate ──────────────────────────────────────────────────────
        # Public callers MUST present a valid JWT (workspace_context not None).
        # Trusted internal callers (the E2E harness, cron-initiated uploads)
        # present a valid `x-cron-secret` instead and bypass the JWT — same
        # mechanism as resume_job. CRON_SECRET unset ⇒ no bypass exists.
        _cron_secret_header = (request.headers.get("x-cron-secret") or "").strip()
        _expected_cron_secret = (os.getenv("CRON_SECRET") or "").strip()
        _is_cron_call = bool(_expected_cron_secret) and _cron_secret_header == _expected_cron_secret

        if not _is_cron_call and workspace_context is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "Authentication required. Provide a Bearer JWT, or an "
                    "x-cron-secret header for trusted internal callers."
                ),
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Workspace integrity check: the JWT-derived workspace from
        # workspace_context.workspace_id is authoritative. The form-supplied
        # `workspace_id` must either match it, or be left at the platform
        # default (which we replace with the JWT-derived value). This closes
        # the cross-workspace data-leak hole flagged in the 2026-05-23 audit.
        # On a cron-secret call there is no JWT workspace, so the form-supplied
        # `workspace_id` (defaulting to the platform default) is trusted as-is.
        _PLATFORM_DEFAULT_WS = get_settings().default_workspace_id
        if workspace_context is not None:
            _jwt_ws = str(workspace_context.workspace_id)
            if workspace_id == _PLATFORM_DEFAULT_WS:
                # Caller didn't override — bind to their actual workspace.
                workspace_id = _jwt_ws
            elif workspace_id != _jwt_ws:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=(
                        f"workspace_id '{workspace_id}' does not match the authenticated "
                        f"caller's workspace ('{_jwt_ws}'). Submit a workspace_id you "
                        f"are a member of, or omit it to use your default."
                    ),
                )

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

        # `extract_only` mode: skip discovery + Stage 4 (products) and only run
        # text/image extraction (Stages 1.5, 2, 3). Previously this was accepted
        # by validation but never branched on — the orchestrator ran the full
        # pipeline over an empty extract_categories list and produced nothing.
        # The 2026-05-23 audit flagged this as dead code; we now reject mixing
        # extract_only with other categories and route it through a leaner path.
        extract_only_mode = False
        if 'extract_only' in category_list:
            if len(category_list) > 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "'extract_only' cannot be combined with other categories — "
                        "it disables discovery. Submit it alone or remove it."
                    ),
                )
            extract_only_mode = True
            # The downstream orchestrator reads `category_list`; keeping it as
            # ['extract_only'] is the explicit signal to skip Stage 0 discovery.

        # Expand 'all' to all categories.
        # (The `focused_extraction` flag was removed 2026-05-24 — it was only
        # ever read by Stage 5 metadata and never gated any actual processing.
        # Page filtering already happens at Stage 0 discovery via
        # `extract_categories`, which scopes downstream work.)
        if 'all' in category_list:
            category_list = ['products', 'certificates', 'logos', 'specifications']

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

            # #250 H7/E: SSRF-guard the user-supplied URL (block internal/metadata hosts),
            # don't follow redirects, and cap the download size (memory-exhaustion defense).
            from app.utils.ssrf_guard import assert_safe_url, SSRFError
            try:
                file_url = assert_safe_url(file_url)
            except SSRFError as _e:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"file_url is not allowed: {_e}")
            _MAX_PDF_BYTES = 100 * 1024 * 1024  # 100MB (matches the documented cap)

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(file_url, timeout=aiohttp.ClientTimeout(total=60), allow_redirects=False) as response:
                        if response.status != 200:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Failed to download PDF from URL: HTTP {response.status}"
                            )
                        if response.content_length and response.content_length > _MAX_PDF_BYTES:
                            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                                detail="PDF exceeds the 100MB limit")

                        # Read at most _MAX_PDF_BYTES+1 so an unset/lied Content-Length can't
                        # blow memory; reject if it overflows.
                        file_content = await response.content.read(_MAX_PDF_BYTES + 1)
                        if len(file_content) > _MAX_PDF_BYTES:
                            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                                detail="PDF exceeds the 100MB limit")

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

        logger.info("📤 CONSOLIDATED UPLOAD")
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

        # #250 H7: enforce the 100MB cap for BOTH paths (the streamed upload had no cap).
        if file_size > 100 * 1024 * 1024:
            try:
                if file_path and os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception:
                pass
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                detail="PDF exceeds the 100MB limit")

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

            # Resolve the canonical storage location (bucket + object path) so
            # the AFTER DELETE trigger on `documents` can wipe the original PDF
            # without parsing URLs at delete time. We back-derive from
            # `file_url` because that's the only thing the frontend hands us.
            storage_bucket: Optional[str] = None
            storage_object_path: Optional[str] = None
            if file_url and "/storage/v1/object/" in file_url:
                try:
                    marker = "/storage/v1/object/"
                    tail = file_url.split(marker, 1)[1]            # public/<bucket>/<path>?token=
                    # Strip the public/sign segment
                    if tail.startswith("public/") or tail.startswith("sign/"):
                        tail = tail.split("/", 1)[1]
                    if "/" in tail:
                        storage_bucket, rest = tail.split("/", 1)
                        storage_object_path = rest.split("?", 1)[0]
                except Exception:
                    pass

            supabase_client.client.table('documents').insert({
                "id": document_id,
                "workspace_id": workspace_id,
                "filename": filename,
                "content_type": "application/pdf",
                "file_size": file_size,
                "file_path": file_path,
                "storage_bucket": storage_bucket,
                "storage_object_path": storage_object_path,
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
                    # NOTE: do NOT persist file_url here. It is a short-lived
                    # signed URL on a private bucket and will 403 the moment
                    # the token expires. Resume reads (storage_bucket,
                    # storage_object_path) and re-signs via service role.
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
                    # NOTE: do NOT persist file_url here — see documents.metadata
                    # comment above. Resume uses storage_bucket/storage_object_path.
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
            extract_categories=category_list,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            workspace_id=workspace_id,
            agent_prompt=agent_prompt,
            enable_prompt_enhancement=enable_prompt_enhancement,
            test_single_product=test_single_product,  # 🧪 TEST MODE flag
            extract_only_mode=extract_only_mode  # NEW: skip Stage 0 + downstream
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


# M19-1: `verify_internal_access` authenticates and stops there — its own docstring
# says it admits "ANY valid platform token, including an end user's". This route then
# read the job by id alone, so any token plus a job id returned that job's metadata,
# stage and recovery history. `require_rag_resource_access` already resolves
# job_id -> background_jobs and requires the caller's workspace to own it — it is
# what the seven sibling routes in this file use, and needs no new code.
@router.get("/documents/job/{job_id}", responses={200: {"model": JobInfoResponse}}, dependencies=[Depends(require_rag_resource_access)])
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


# M19-1 — see get_job_status above. This one returns checkpoint payloads and memory
# state, so it discloses more than the status route it sits beside.
@router.get("/documents/job/{job_id}/full-status", dependencies=[Depends(require_rag_resource_access)])
async def get_job_full_status(job_id: str):
    """Single round-trip endpoint for the full state of a job.

    `background_jobs` is the source of truth: stage_history (audit log) and
    recovery_history (auto-recovery attempts) live on the row itself.
    `product_processing_status` stays as a child table because cardinality
    differs (1 job → N products).
    """
    supabase_client = get_supabase_client()

    core_resp = supabase_client.client.table('background_jobs') \
        .select('*').eq('id', job_id).execute()
    if not core_resp.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    core = core_resp.data[0]

    try:
        products_resp = supabase_client.client.table('product_processing_status') \
            .select('*').eq('job_id', job_id) \
            .order('product_index', desc=False).execute()
        products = products_resp.data or []
    except Exception as products_err:
        logger.warning(f"Failed to load product_processing_status for {job_id}: {products_err}")
        products = []

    return JSONResponse(content={
        "job_id": job_id,
        "core": core,
        "stage_history": core.get('stage_history') or [],
        "recovery_history": core.get('recovery_history') or [],
        "products": products,
        "memory": job_storage.get(job_id),
    })


# M19-1 — see get_job_status above.
@router.get("/jobs/{job_id}/checkpoints", responses={200: {"model": CheckpointListResponse}}, dependencies=[Depends(require_rag_resource_access)])
async def get_job_checkpoints(job_id: str):
    """Returns the stage history for a job (alias maintained for older
    clients — `/documents/job/{id}/full-status` is the consolidated read).
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


@router.post("/jobs/{job_id}/restart", response_model=StatusResponse, dependencies=[Depends(require_rag_resource_access)])
async def restart_job_from_checkpoint(job_id: str, background_tasks: BackgroundTasks):
    """
    Manually restart a job from its last checkpoint.

    This endpoint allows manual recovery of stuck or failed jobs.
    The job will resume from the last successful checkpoint.

    Audit fix (this PR): re-entrancy guard. Two cron ticks (or a cron tick +
    a manual operator restart) can both POST /resume for the same job_id
    within seconds. Without an atomic claim, both would download the PDF and
    dispatch process_document_with_discovery, producing two orchestrators on
    the same row — which silently double-creates products + chunks and
    double-bills HF endpoint replicas. We claim the job by flipping
    status pending|interrupted → processing in the FIRST line of work; if
    the conditional UPDATE returns 0 rows, another caller already claimed it
    and we 409 out without dispatching.
    """
    try:
        # Atomic claim — must run before any expensive work (file download,
        # checkpoint verification). The .eq("status", "pending|interrupted")
        # filter ensures only one of N concurrent callers wins; the rest
        # fall through to the 409 path below.
        supabase_client = get_supabase_client()
        nowiso = datetime.utcnow().isoformat()
        claim_result = supabase_client.client.table('background_jobs').update({
            'status': 'processing',
            'last_heartbeat': nowiso,
            'updated_at': nowiso,
        }).eq('id', job_id).in_('status', ['pending', 'interrupted']).execute()

        if not claim_result.data:
            # Either job already 'processing' (someone else claimed it), or
            # status='completed'/'failed' (no recovery needed), or row missing.
            # 409 is correct because the operation isn't an error — another
            # caller is running it. The cron path treats !ok as a soft fail
            # and reverts pending→interrupted, so this is idempotent.
            row_check = supabase_client.client.table('background_jobs').select('id, status').eq('id', job_id).execute()
            if not row_check.data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Job {job_id} not found"
                )
            existing_status = row_check.data[0].get('status')
            logger.info(f"Resume rejected for {job_id}: status='{existing_status}' (already in flight or terminal)")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job {job_id} is in status '{existing_status}' — cannot resume (another caller may have claimed it)"
            )

        # Get last checkpoint
        last_checkpoint = await checkpoint_recovery_service.get_last_checkpoint(job_id)

        if not last_checkpoint:
            # We claimed the row but there's no checkpoint to resume from.
            # Revert the claim so the row's status doesn't stay 'processing'
            # with no orchestrator. The auto-recovery cron will then
            # eventually fail-out the job via fail_exhausted_pdf_jobs once
            # the stuck threshold elapses.
            try:
                supabase_client.client.table('background_jobs').update({
                    'status': 'interrupted',
                    'interrupted_at': datetime.utcnow().isoformat(),
                }).eq('id', job_id).execute()
            except Exception:
                pass
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No checkpoint found for job {job_id}"
            )

        # Verify checkpoint data exists
        resume_stage_str = last_checkpoint.get('stage')
        resume_stage = ProcessingStage(resume_stage_str)
        can_resume = await checkpoint_recovery_service.verify_checkpoint_data(job_id, resume_stage)

        if not can_resume:
            try:
                supabase_client.client.table('background_jobs').update({
                    'status': 'interrupted',
                    'interrupted_at': datetime.utcnow().isoformat(),
                }).eq('id', job_id).execute()
            except Exception:
                pass
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Checkpoint data verification failed for stage {resume_stage}"
            )

        # Get full job details from database (claim_result.data[0] is also valid
        # but a fresh read avoids any chance of stale fields between the claim
        # update and the rest of this handler).
        job_result = supabase_client.client.table('background_jobs').select('*').eq('id', job_id).execute()

        if not job_result.data or len(job_result.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found in database"
            )

        job_data = job_result.data[0]
        document_id = job_data['document_id']

        # NOTE: status='processing' is intentionally NOT flipped here. The
        # previous ordering set the flag BEFORE the download / temp-file
        # write, so transient Storage 5xx or /tmp-full errors left the row
        # at status='processing' with no orchestrator running. Auto-recovery
        # cron then back-offs (waiting for stuck-threshold) instead of
        # immediately reclaiming. We flip the status atomically with the
        # background_tasks.add_task at the bottom of the try block, so
        # any failure in the download path leaves the job at its prior
        # status ('failed' / 'interrupted') and the operator sees the
        # actual error.

        # Re-trigger the processing pipeline; process_document_with_discovery resumes from the checkpoint.

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
            doc_data.get('metadata', {})
            storage_bucket = doc_data.get('storage_bucket')
            storage_object_path = doc_data.get('storage_object_path')

            # If file_path points at a local temp file (gone after pod restart),
            # fall back to canonical (storage_bucket, storage_object_path). The
            # previously-persisted metadata.file_url is intentionally NOT read
            # here — it was a short-lived signed URL and would have expired by
            # the time auto-recovery runs.
            if file_path and file_path.startswith('/tmp/'):
                if storage_bucket and storage_object_path:
                    logger.info(
                        f"⚠️ file_path is local temp file ({file_path}); "
                        f"resolving from storage bucket={storage_bucket} path={storage_object_path}"
                    )
                    file_path = None  # force the storage-download branch below
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Document {document_id} has local temp file_path but no storage_bucket/storage_object_path. Cannot resume."
                    )

            if not file_path and not (storage_bucket and storage_object_path):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Document {document_id} has no file_path or storage location"
                )

            # Download file from storage or URL
            logger.info(f"📥 Downloading file: bucket={storage_bucket} path={storage_object_path} fallback={file_path}")

            if storage_bucket and storage_object_path:
                # Service-role download from the canonical storage location.
                # Bypasses signed-URL TTLs entirely.
                file_response = supabase_client.client.storage.from_(storage_bucket).download(storage_object_path)
            elif file_path and (file_path.startswith('http://') or file_path.startswith('https://')):
                # Legacy URL-based path (pre-storage_object_path migration).
                # `documents.file_path` is a DB column, so this is an invariant-7 fetch:
                # guarded helper, every redirect hop re-validated, body capped while
                # streaming at the same limit upload accepts.
                #
                # `http` is allowed here and nowhere else: these rows predate the storage
                # migration and some carry plaintext URLs. Refusing them would turn a
                # resumable job into a permanently unresumable one, which is a worse
                # outcome than a plaintext fetch whose host has still been DNS-resolved
                # and checked against every private range. New code uses https only.
                from app.utils.ssrf_guard import MAX_PDF_BYTES, SSRFError, safe_fetch_bytes

                try:
                    fetched = await safe_fetch_bytes(
                        file_path,
                        max_bytes=MAX_PDF_BYTES,
                        timeout=60.0,
                        allow_schemes=("http", "https"),
                    )
                except SSRFError as ssrf_err:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Blocked document file_path: {ssrf_err}",
                    )
                if not fetched.ok:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"Failed to download file from URL: HTTP {fetched.status_code}",
                    )
                file_response = fetched.content
                logger.info(f"✅ Downloaded file from URL: {len(file_response)} bytes")
            else:
                # Legacy: file_path encodes "{bucket}/{key}"
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
            discovery_model = job_metadata.get('discovery_model', 'claude-opus-4-8')
            categories = job_metadata.get('categories', ['products'])
            enable_prompt_enhancement = job_metadata.get('prompt_enhancement_enabled', False)
            agent_prompt = job_metadata.get('agent_prompt')
            test_single_product = job_metadata.get('test_single_product', False)

            logger.info(f"   Resume parameters: discovery_model={discovery_model}, categories={categories}, test_mode={test_single_product}")

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
                extract_categories=categories,
                chunk_size=1000,
                chunk_overlap=200,
                agent_prompt=agent_prompt,
                enable_prompt_enhancement=enable_prompt_enhancement,
                test_single_product=test_single_product
            )

            # Status is already 'processing' from the atomic claim at the top
            # of this handler. We just need to record the restart context
            # (clear `error`, stamp `started_at`, append restart metadata).
            # Audit fix (this PR): use merge_background_job_metadata RPC for
            # the metadata write so we don't read-modify-write — concurrent
            # heartbeat / stage_history updates were racing with this overwrite.
            supabase_client.client.table('background_jobs').update({
                "error": None,
                "interrupted_at": None,
                "started_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }).eq('id', job_id).execute()
            try:
                supabase_client.client.rpc(
                    'merge_background_job_metadata',
                    {
                        'p_job_id': job_id,
                        'p_metadata': {
                            "restart_from_stage": resume_stage.value,
                            "restart_reason": "manual_restart",
                            "restart_at": datetime.utcnow().isoformat(),
                        },
                    },
                ).execute()
            except Exception as e:
                logger.warning(f"merge_background_job_metadata failed for {job_id}: {e}")

            logger.info(f"✅ Job {job_id} marked for restart from {resume_stage} and background task triggered (type: {job_type})")

        except HTTPException:
            # Audit fix (this PR): the atomic claim above flipped status to
            # 'processing'. If the file-download path raises, no orchestrator
            # is running but the row says we're processing — auto-recovery
            # waits the full stuck threshold before reclaiming. Revert the
            # claim so the cron sees an 'interrupted' row immediately.
            try:
                supabase_client.client.table('background_jobs').update({
                    'status': 'interrupted',
                    'interrupted_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat(),
                }).eq('id', job_id).eq('status', 'processing').execute()
            except Exception:
                pass
            raise
        except Exception as e:
            logger.error(f"Failed to download file for restart: {e}", exc_info=True)
            try:
                supabase_client.client.table('background_jobs').update({
                    'status': 'interrupted',
                    'interrupted_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat(),
                    'error': f"Resume download failed: {str(e)[:1000]}",
                }).eq('id', job_id).eq('status', 'processing').execute()
            except Exception:
                pass
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


@router.post("/documents/job/{job_id}/resume", response_model=StatusResponse)
async def resume_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    request: Request,
    workspace_context = Depends(get_optional_workspace_context),
):
    """
    Resume a job from its last checkpoint (alias for restart).

    Auth: requires a JWT. The caller's workspace must own the job — anyone
    able to guess a job UUID could otherwise re-spawn the orchestrator and
    double-bill HF/Anthropic. The auto-recovery cron path uses a separate
    cron-secret header (see `x-cron-secret` accept logic below) to bypass
    the JWT check for legitimate automated recovery.

    The workspace dependency is OPTIONAL on purpose: the auto-recovery-cron
    presents ONLY an `x-cron-secret` header and no bearer token. A hard
    Depends(get_workspace_context) would 403 at HTTPBearer(auto_error=True)
    before this body's cron check could run — which silently broke automated
    recovery (the cron just logged the 403 and moved on). Optional + in-body
    enforcement is the correct shape.
    """
    # Cron bypass: if the call carries a valid x-cron-secret header, skip the
    # workspace ownership check. This is the only path the supabase
    # `auto-recovery-cron` edge function uses.
    import os as _os
    cron_secret_header = (request.headers.get("x-cron-secret") or "").strip()
    expected_cron_secret = (_os.getenv("CRON_SECRET") or "").strip()
    is_cron_call = bool(expected_cron_secret) and cron_secret_header == expected_cron_secret

    if not is_cron_call:
        # Non-cron callers MUST authenticate with a JWT.
        if workspace_context is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "Authentication required. Provide a Bearer JWT, or an "
                    "x-cron-secret header for trusted automated recovery."
                ),
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Enforce workspace ownership.
        try:
            supabase_client = get_supabase_client()
            job_row = supabase_client.client.table('background_jobs') \
                .select('workspace_id') \
                .eq('id', job_id) \
                .single().execute()
            if not job_row.data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Job {job_id} not found",
                )
            job_ws = str(job_row.data.get('workspace_id') or '')
            if job_ws and job_ws != str(workspace_context.workspace_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You are not a member of the workspace that owns this job.",
                )
        except HTTPException:
            raise
        except Exception as ownership_err:
            # Don't let a DB hiccup turn into a silent bypass.
            logger.error(f"resume_job: ownership check raised: {ownership_err}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not verify job ownership; refusing resume.",
            )

    return await restart_job_from_checkpoint(job_id, background_tasks)


@router.post("/documents/{document_id}/reprocess", dependencies=[Depends(require_rag_resource_access)])
async def reprocess_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    clear_intermediate: bool = True,
) -> Dict[str, Any]:
    """
    Full Stage 0→4.7 reprocess of an existing document without re-uploading
    the PDF. Use this to validate pipeline changes against a known document.

    Flow:
      1. Find the latest background_jobs row for this document_id.
      2. Resolve the PDF on disk (from /tmp/pdf_processor_{doc_id}/).
      3. If `clear_intermediate=True` (default): delete derived data
         (products, chunks, images, checkpoints, catalog_layout,
         catalog_legends) so the new run starts from a clean slate.
         The `documents` row itself is preserved.
      4. Create a NEW background_jobs row (status=pending, progress=0).
      5. Launch `process_document_with_discovery()` as a background task.
      6. Return the new job_id + handoff URL for the monitor.

    Query params:
      clear_intermediate (bool, default True): whether to wipe derived data.
        Set False for a "resume fresh" that just starts a new job against
        existing intermediate state — useful if you want to test idempotency.
    """
    import uuid

    try:
        supabase = get_supabase_client()

        # ── 1. Resolve the document and its latest job ─────────────────
        doc_resp = supabase.client.table("documents") \
            .select("id, filename, workspace_id, metadata") \
            .eq("id", document_id) \
            .limit(1) \
            .execute()
        if not doc_resp.data:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        doc = doc_resp.data[0]
        filename = doc.get("filename") or f"{document_id}.pdf"
        workspace_id = doc.get("workspace_id")
        doc_metadata = doc.get("metadata") or {}

        # Find the latest job to preserve user-specified discovery options
        # `discovery_model` and `extract_categories` are upload-FORM parameters, not
        # columns on background_jobs (#26 M13-2) — naming them made PostgREST reject the
        # read, so this whole reprocess route 500'd before reaching the PDF. They are
        # carried in the job's `metadata` jsonb.
        jobs_resp = supabase.client.table("background_jobs") \
            .select("id, status, metadata") \
            .eq("document_id", document_id) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
        prev_job = (jobs_resp.data or [{}])[0] if jobs_resp.data else {}

        # Refuse while a job for this document is still in flight (#34 M19-3).
        # Everything below this point DELETES: products, chunks, images, VECS
        # embeddings, tile storage and document metadata. Issuing a reprocess against a
        # document mid-ingestion pulls the running job's outputs out from under it, and
        # the running job then carries on writing into the wreckage.
        #
        # `restart` — 400 lines up in this same file — already does exactly this, as a
        # status-scoped compare-and-swap: `.in_('status', ['pending', 'interrupted'])`.
        # This is that guard stated as a precondition, because the delete is not a single
        # UPDATE that could carry one.
        if (prev_job.get("status") or "") in ("pending", "processing"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"A job for document {document_id} is {prev_job.get('status')}. "
                    "Reprocess deletes this document's derived data, so it refuses while "
                    "one is in flight — wait for it to finish, or cancel it first."
                ),
            )
        prev_job_meta = prev_job.get("metadata") or {}

        discovery_model = prev_job_meta.get("discovery_model") or "claude-opus-4-8"
        extract_categories = prev_job_meta.get("extract_categories") or []

        # ── 2. Resolve PDF on disk ─────────────────────────────────────
        from app.services.products.product_spec_vision_extractor import _get_source_pdf_path
        pdf_path = _get_source_pdf_path(document_id)
        if not pdf_path:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"PDF source not on disk for document {document_id}. "
                    f"Expected at /tmp/pdf_processor_{document_id}/{document_id}.pdf. "
                    f"Re-download from Supabase Storage first."
                ),
            )

        # ── 3. Clear intermediate data (optional) ──────────────────────
        if clear_intermediate:
            logger.info(f"🧹 reprocess: clearing intermediate data for {document_id}")

            # A SELECT of this document's product ids used to sit here, commented
            # "so we can delete their joins" — and nothing ever deleted a join with
            # them. A read whose result was discarded, so removing it changes no
            # behaviour. NOTE the intent it recorded was never implemented: if
            # product join rows do need clearing on reprocess, that is a real gap and
            # not something this cleanup should invent a fix for.

            # 3a. VECS embeddings — these are NOT FK-cascaded from
            # document_images / products. Without explicit cleanup the
            # vector index keeps returning hits to deleted rows and the
            # next reprocess writes new vectors alongside the stale ones.
            try:
                from app.services.embeddings.vecs_service import get_vecs_service
                vecs = get_vecs_service()
                vec_count = await vecs.delete_document_embeddings(document_id)
                logger.info(f"🧹 reprocess: cleared {vec_count} VECS embeddings for {document_id}")
            except Exception as vec_err:
                logger.warning(f"reprocess: VECS cleanup failed (continuing): {vec_err}")

            # 3b. Storage objects — pdf-tiles under extracted/{document_id}/.
            # The original PDF is preserved (we need it on disk to reprocess).
            # We deliberately do NOT call cleanup_document_storage here, since
            # the trigger on the documents row would also try to fire if we
            # ended up touching the row. Use cleanup_storage_bucket directly.
            try:
                from app.services.utilities.cleanup_service import CleanupService
                cs = CleanupService()
                tiles_deleted = cs.cleanup_storage_bucket(
                    "pdf-tiles", f"extracted/{document_id}", supabase
                )
                legacy_deleted = cs.cleanup_storage_bucket(
                    "documents", document_id, supabase
                )
                logger.info(
                    f"🧹 reprocess: deleted {tiles_deleted + legacy_deleted} extracted tiles"
                )
            except Exception as storage_err:
                logger.warning(f"reprocess: storage cleanup failed (continuing): {storage_err}")

            # 3c. Tables. document_images delete fires the per-row trigger
            # which scrubs any image_url that points at a storage object —
            # so any stragglers the prefix-based delete missed get cleaned
            # row-by-row.
            #
            # image_product_associations / kb_doc_attachments cascade off
            # both products and document_images, so they go automatically.
            supabase.client.table("products") \
                .delete().eq("source_document_id", document_id).execute()
            supabase.client.table("document_chunks") \
                .delete().eq("document_id", document_id).execute()
            supabase.client.table("document_images") \
                .delete().eq("document_id", document_id).execute()

            # stage_history lives on background_jobs.stage_history.
            # We're spawning a brand-new job row below; the old row's
            # history stays as audit data and naturally drops out of any
            # job-id-scoped queries. Nothing to clear here.

            # Wipe the Layer 1/2 cached output on the document so they re-run
            doc_metadata.pop("catalog_layout", None)
            doc_metadata.pop("catalog_legends", None)
            supabase.client.table("documents") \
                .update({"metadata": doc_metadata}).eq("id", document_id).execute()

        # ── 4. Create a fresh background_jobs row ──────────────────────
        new_job_id = str(uuid.uuid4())
        supabase.client.table("background_jobs").insert({
            "id": new_job_id,
            "document_id": document_id,
            "workspace_id": workspace_id,
            "job_type": "pdf_processing",
            "status": "pending",
            "progress": 0,
            "filename": filename,
            "discovery_model": discovery_model,
            "extract_categories": extract_categories,
            "metadata": {
                "reprocess_of": prev_job.get("id"),
                "triggered_by": "reprocess_endpoint",
                "clear_intermediate": clear_intermediate,
            },
        }).execute()

        # ── 5. Kick off process_document_with_discovery in the background
        background_tasks.add_task(
            process_document_with_discovery,
            job_id=new_job_id,
            document_id=document_id,
            file_path=pdf_path,
            filename=filename,
            title=doc_metadata.get("title") or filename,
            description=doc_metadata.get("description") or "",
            document_tags=doc_metadata.get("tags") or [],
            discovery_model=discovery_model,
            extract_categories=extract_categories,
            chunk_size=1000,
            chunk_overlap=200,
            workspace_id=workspace_id,
            agent_prompt=doc_metadata.get("agent_prompt"),
            enable_prompt_enhancement=True,
            test_single_product=False,
        )

        logger.info(
            f"🔄 reprocess: launched new job {new_job_id} for document {document_id} "
            f"(previous job: {prev_job.get('id') or 'none'}, cleared: {clear_intermediate})"
        )

        return {
            "success": True,
            "document_id": document_id,
            "new_job_id": new_job_id,
            "previous_job_id": prev_job.get("id"),
            "cleared_intermediate": clear_intermediate,
            "message": (
                f"Reprocess dispatched. Monitor via "
                f"/api/rag/documents/job/{new_job_id} or the admin AsyncJobQueueMonitor."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ reprocess_document failed for {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Reprocess failed: {str(e)}",
        )


# M19-1, with a correction to the fix the audit proposed. It says to swap all FOUR
# weakly-gated routes to `require_rag_resource_access`. That works for the three that
# carry a job_id and NOT for this one: the gate resolves an id from the path or query
# and raises 400 when there is none ("A list endpoint with no resource id would
# otherwise return rows across all tenants"). Swapping it here would 400 every call.
#
# A list route needs a PREDICATE, not a resource lookup. The workspace comes from the
# caller's context; the cron path has none and keeps its platform-wide view, which is
# what auto-recovery needs.
@router.get("/documents/jobs", responses={200: {"model": ListDataResponse}}, dependencies=[Depends(verify_internal_access)])
async def list_jobs(
    request: Request,
    limit: int = 10,
    offset: int = 0,
    status_filter: Optional[str] = None,
    sort: str = "created_at:desc",
    workspace_context = Depends(get_optional_workspace_context),
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

        # Bind the list to the caller's workspace (M19-1). Without this, any valid
        # platform token listed every tenant's jobs. A cron caller (x-cron-secret, no
        # workspace context) is deliberately unfiltered: auto-recovery has to see the
        # whole estate, and it is not a user.
        caller_workspace = getattr(workspace_context, "workspace_id", None)
        if caller_workspace:
            query = query.eq('workspace_id', caller_workspace)
        elif not (request.headers.get("x-cron-secret") or "").strip():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="A workspace context is required to list jobs.",
            )

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


@router.delete("/documents/jobs/{job_id}", dependencies=[Depends(require_rag_resource_access)])
async def delete_job(
    job_id: str,
    preserve_outputs: Optional[bool] = Query(
        None,
        description=(
            "Override the deletion mode. None (default) = decide based on job.status: "
            "'completed' jobs preserve outputs, 'cancelled'/'failed'/'stuck' jobs wipe everything. "
            "Pass true/false to force a specific mode."
        ),
    ),
):
    """
    Delete a job with two distinct semantics, decided by `job.status` (or
    overridden via the `preserve_outputs` query param).

    ──────────────────────────────────────────────────────────────────────
    `status='completed'` (or `preserve_outputs=true`) → PRESERVE_OUTPUTS
    ──────────────────────────────────────────────────────────────────────
    Removes the job tracking row and per-product status. KEEPS all the
    catalog data the job produced: documents, chunks, products, images,
    embeddings, storage files. Use case: "user removes a finished job from
    'recent jobs' but the products are now in their catalog and must stay."

    ──────────────────────────────────────────────────────────────────────
    `status` ∈ {'cancelled', 'failed', 'stuck'} (or `preserve_outputs=false`) → FULL_WIPE
    ──────────────────────────────────────────────────────────────────────
    Wipes everything: job, document, chunks, products, images, embeddings,
    associations, storage files, temp files. Use case: "this job's output
    is bad/partial — get it out of the catalog entirely."
    """
    try:
        logger.info(f"🗑️ DELETE /documents/jobs/{job_id} - resolving deletion mode")

        # Remove from in-memory storage if exists
        if job_id in job_storage:
            del job_storage[job_id]
            logger.info(f"   ✅ Removed job {job_id} from job_storage")

        # Get services
        supabase_client = get_supabase_client()
        vecs_service = get_vecs_service()

        # Decide the mode. Explicit override wins; otherwise infer from status.
        # The enum-derived sets are the canonical source of truth — see
        # app/schemas/jobs.py for the contract.
        from app.schemas.jobs import (
            JOB_STATUS_PRESERVE_OUTPUTS,
            JOB_STATUS_WIPE_OUTPUTS,
        )
        COMPLETED_STATUSES = {s.value for s in JOB_STATUS_PRESERVE_OUTPUTS}
        WIPE_STATUSES = {s.value for s in JOB_STATUS_WIPE_OUTPUTS}

        if preserve_outputs is not None:
            mode_preserve = bool(preserve_outputs)
            mode_source = "explicit_query_param"
        else:
            try:
                job_row = supabase_client.client.table('background_jobs')\
                    .select('status')\
                    .eq('id', job_id)\
                    .single()\
                    .execute()
                job_status = (job_row.data or {}).get('status') or 'unknown'
            except Exception:
                job_status = 'unknown'

            if job_status in COMPLETED_STATUSES:
                mode_preserve = True
                mode_source = f"status={job_status}"
            elif job_status in WIPE_STATUSES:
                mode_preserve = False
                mode_source = f"status={job_status}"
            else:
                # Unknown / processing / pending — be conservative: full wipe.
                # If the user is deleting a job mid-processing, they almost
                # certainly want the partial output gone.
                mode_preserve = False
                mode_source = f"status={job_status}_default_wipe"

        logger.info(
            f"   📋 Mode: preserve_outputs={mode_preserve} (decided by {mode_source})"
        )

        from app.services.utilities.cleanup_service import CleanupService
        cleanup_service = CleanupService()

        stats = await cleanup_service.delete_job_completely(
            job_id=job_id,
            supabase_client=supabase_client,
            vecs_service=vecs_service,
            delete_storage_files=True,
            preserve_outputs=mode_preserve,
        )

        # Check if job was actually deleted
        if not stats['job_deleted']:
            logger.warning(f"   ⚠️ Job {job_id} not found or deletion failed")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found or deletion failed"
            )

        logger.info(f"   ✅ Deletion finished for job {job_id} [mode={stats.get('mode')}]")
        logger.info(f"   📊 Stats: {stats}")

        msg_suffix = (
            "tracking removed; produced catalog data preserved"
            if mode_preserve
            else "and all associated data deleted"
        )
        return {
            "success": True,
            "message": f"Job {job_id} {msg_suffix}",
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


@router.get("/chunks", responses={200: {"model": ListDataResponse}}, dependencies=[Depends(require_rag_resource_access)])
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

        # `include_embeddings` has to REMOVE, not just add (#34 M19-2). The base query
        # is `select('*')`, so `text_embedding` was in every row regardless and this
        # block only layered derived fields on top — a caller explicitly asking not to
        # receive embeddings received them anyway. That is payload bloat rather than a
        # leak (the gate above binds the document to its owner), but a control that does
        # nothing is worse than no control: it reads as if the choice were being honoured.
        #
        # `has_embedding` is computed either way, because "does this chunk have one" is
        # the question `include_embeddings=false` callers are usually asking.
        for chunk in chunks:
            text_embedding = chunk.pop('text_embedding', None)
            chunk['has_embedding'] = text_embedding is not None
            if include_embeddings:
                chunk['embedding'] = text_embedding
                chunk['embeddings'] = (
                    [{'embedding': text_embedding, 'type': 'text'}] if text_embedding else []
                )

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


@router.get("/images", responses={200: {"model": ListDataResponse}}, dependencies=[Depends(require_rag_resource_access)])
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


@router.get("/products", responses={200: {"model": ListDataResponse}}, dependencies=[Depends(require_rag_resource_access)])
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


@router.get("/embeddings", responses={200: {"model": ListDataResponse}}, dependencies=[Depends(require_rag_resource_access)])
async def get_embeddings(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    embedding_type: Optional[str] = Query(None, description="Filter by embedding type (text, visual, color, texture, style, material, understanding)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of embeddings to return"),
    offset: int = Query(0, ge=0, description="Number of embeddings to skip")
):
    """
    Get embeddings for a document.

    Returns presence summary for image embeddings (read from document_images
    has_*_slig boolean flags — VECS is the canonical store) and the actual
    text embedding vectors for document_chunks.

    Args:
        document_id: Document ID to filter embeddings
        embedding_type: Optional type filter (text, visual, color, texture, style, material, understanding)
        limit: Maximum number of embeddings to return
        offset: Pagination offset

    Returns:
        List of embeddings (text embeddings inline, image embeddings as presence flags)
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

        # ── Image embeddings: read presence from document_images boolean flags ──
        # VECS is the source of truth for the actual vectors. We expose only the
        # presence flags here because returning raw 768D float arrays for every
        # image would balloon the response size for no admin-UI benefit.
        wants_image = not embedding_type or embedding_type in (
            'visual', 'slig', 'color', 'texture', 'style', 'material', 'understanding'
        )
        if wants_image:
            # Aspect collections post-2026-05-04 are 1024D Voyage embeddings of
            # VisionAnalysis text; provenance columns on document_images carry
            # the model + schema_version that produced each row's vector.
            image_result = supabase_client.client.table('document_images').select(
                'id, image_url, '
                'has_slig_embedding, has_color_slig, has_texture_slig, '
                'has_style_slig, has_material_slig, has_understanding_embedding, '
                'color_aspect_embedding_model, color_aspect_schema_version, '
                'texture_aspect_embedding_model, texture_aspect_schema_version, '
                'style_aspect_embedding_model, style_aspect_schema_version, '
                'material_aspect_embedding_model, material_aspect_schema_version'
            ).eq('document_id', document_id).range(offset, offset + limit - 1).execute()

            aspect_rows = {
                'has_color_slig':    ('color_aspect_1024',    'color'),
                'has_texture_slig':  ('texture_aspect_1024',  'texture'),
                'has_style_slig':    ('style_aspect_1024',    'style'),
                'has_material_slig': ('material_aspect_1024', 'material'),
            }
            non_aspect_rows = {
                'has_slig_embedding':         ('visual_slig_768',     768,  'siglip2',  'visual'),
                'has_understanding_embedding':('understanding_1024',  1024, 'voyage-4', 'understanding'),
            }

            for img in (image_result.data or []):
                for flag, (label, dim, model, type_alias) in non_aspect_rows.items():
                    if not img.get(flag):
                        continue
                    if embedding_type and embedding_type != type_alias and embedding_type not in ('visual', 'slig'):
                        continue
                    embeddings.append({
                        'id': f"{img['id']}_{type_alias}",
                        'entity_id': img['id'],
                        'entity_type': 'image',
                        'embedding_type': label,
                        'dimension': dim,
                        'model': model,
                        'present': True,
                        'storage': f"vecs.image_{type_alias if type_alias != 'visual' else 'slig'}_embeddings",
                    })
                    embedding_stats[label] = embedding_stats.get(label, 0) + 1

                for flag, (label, type_alias) in aspect_rows.items():
                    if not img.get(flag):
                        continue
                    if embedding_type and embedding_type != type_alias and embedding_type not in ('visual', 'slig'):
                        continue
                    prov_model = img.get(f'{type_alias}_aspect_embedding_model') or 'voyage-3'
                    prov_schema = img.get(f'{type_alias}_aspect_schema_version')
                    embeddings.append({
                        'id': f"{img['id']}_{type_alias}",
                        'entity_id': img['id'],
                        'entity_type': 'image',
                        'embedding_type': label,
                        'dimension': 1024,
                        'model': prov_model,
                        'schema_version': prov_schema,
                        'present': True,
                        'storage': f"vecs.image_{type_alias}_embeddings",
                    })
                    embedding_stats[label] = embedding_stats.get(label, 0) + 1

        # ── Text chunk embeddings: still inline (1024D Voyage AI) ──
        if not embedding_type or embedding_type == 'text':
            chunk_query = supabase_client.client.table('document_chunks').select(
                'id, content, text_embedding, embedding_dimension'
            ).eq('document_id', document_id).not_('text_embedding', 'is', None).range(offset, offset + limit - 1)

            chunk_result = chunk_query.execute()

            if chunk_result.data:
                for chunk in chunk_result.data:
                    dimension = chunk.get('embedding_dimension', 1024)
                    embeddings.append({
                        'id': f"{chunk['id']}_text",
                        'entity_id': chunk['id'],
                        'entity_type': 'chunk',
                        'embedding_type': f'text_{dimension}',
                        'dimension': dimension,
                        'model': 'voyage-4',
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


@router.get("/relevancies", responses={200: {"model": RelevancyListResponse}}, dependencies=[Depends(verify_internal_access)])
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
    sub_job_id = str(uuid4())

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
        candidates = product_result.get('candidates_detected', 0)
        logger.info(f"✅ Background product creation completed: {products_created} products created")

        # Candidates found and nothing created is the silent zero at the level that
        # matters most to a user: products ARE the point of the pipeline, and an
        # ingestion that finishes green with none looks identical to a catalogue that
        # genuinely had none (#35 M20-4). Zero candidates is a different, legitimate
        # answer and is left alone.
        if candidates and not products_created:
            logger.error(
                "product creation produced 0 products from %s candidates for document %s "
                "(job %s) — the run completed, so nothing else will report this",
                candidates, document_id, job_id,
            )
            sentry_sdk.capture_message(
                f"Ingestion completed with 0 products from {candidates} candidates "
                f"(document {document_id})",
                level="warning",
            )

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

        # Persist products_created onto the background_jobs row so the API response reflects it.
        try:
            job_recovery_service = JobRecoveryService(supabase_client)

            current_job = await job_recovery_service.get_job_status(job_id)
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

        # Record the failure on the PARENT job (#35 M20-4). Before this, only the
        # sub-job was marked failed, and the parent had already been persisted
        # `completed` — so an ingestion whose product creation blew up reported a clean
        # success with zero products, and products are the point of the pipeline.
        #
        # Recorded in METADATA, not as a new status. `background_jobs_status_check`
        # admits seven values and `completed_with_failures` is not one of them; adding
        # it would need every reader — the edge runner, the admin UI, auto-recovery — to
        # learn it, and audit #217 M7 is the record of what happens when they do not
        # (writing 'running' made job-research runs invisible and mis-bucketed).
        #
        # Nor is the parent marked `failed`: the document WAS processed — chunks, images
        # and embeddings all landed — and failing it would send auto-recovery to restart
        # a job that mostly succeeded.
        try:
            parent = supabase_client.client.table('background_jobs')                 .select('metadata')                 .eq('id', job_id)                 .limit(1)                 .execute()
            parent_meta = ((parent.data or [{}])[0] or {}).get('metadata') or {}
            parent_meta.update({
                'product_creation_failed': True,
                'product_creation_error': str(e)[:500],
                'products_created': 0,
                'product_creation_failed_at': datetime.utcnow().isoformat(),
            })
            supabase_client.client.table('background_jobs').update({
                'metadata': parent_meta,
                'updated_at': datetime.utcnow().isoformat(),
            }).eq('id', job_id).execute()
            logger.error(
                "marked parent job %s as product_creation_failed — the ingestion "
                "otherwise reports success with zero products", job_id,
            )
        except Exception as parent_error:
            logger.error(
                "could not record product-creation failure on parent job %s: %s — the "
                "ingestion will now report a clean success it did not earn",
                job_id, parent_error, exc_info=True,
            )
            sentry_sdk.capture_exception(parent_error)

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
        except Exception as checkpoint_error:
            # Resumability is lost when this fails — surface it loudly so ops
            # know the failed-job state was not persisted.
            logger.error(
                f"⚠️ Failed to write failure checkpoint for job {job_id}: {checkpoint_error}",
                exc_info=True,
            )
            sentry_sdk.capture_exception(checkpoint_error)


async def process_document_with_discovery(
    job_id: str,
    document_id: str,
    file_path: str,  # CHANGED: file_path instead of file_content
    filename: str,
    title: Optional[str],
    description: Optional[str],
    document_tags: List[str],
    discovery_model: str,
    extract_categories: List[str],
    chunk_size: int,
    chunk_overlap: int,
    workspace_id: str = None,
    agent_prompt: Optional[str] = None,
    enable_prompt_enhancement: bool = True,
    test_single_product: bool = False,  # 🧪 TEST MODE: Process only first product
    extract_only_mode: bool = False  # 2026-05-23: skip discovery + downstream, run extract-only
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
        extract_categories: List of categories to extract (e.g., ['products'], ['certificates', 'logos']).
                          Categories: 'products', 'certificates', 'logos', 'specifications', 'all'
    """
    # Use default workspace ID from config if not provided
    workspace_id = workspace_id or get_settings().default_workspace_id
    start_time = datetime.utcnow()

    # ── Tenancy preflight ────────────────────────────────────────────────────────────
    # The three ids arrive independently and every stage below reads and writes by bare
    # id under service role (#35 M20-3). A mismatched tuple here does not corrupt one
    # row — it misroutes an ENTIRE ingestion: products written under the wrong
    # workspace, another tenant's document marked completed.
    #
    # Validated ONCE, first, before the credit preflight and before any file is touched.
    # The finding also asks for the validated workspace to be threaded through every
    # stage instead of the parameter being re-trusted; that half is a refactor of this
    # 1,800-line function and is not attempted here. It is also the less important half:
    # rethreading changes nothing about a tuple that has already been proven coherent,
    # and this is what makes the parameter trustworthy in the first place.
    #
    # A failure marks the job failed rather than raising bare — a job that stops with
    # `status='processing'` and no terminal event is the state the rest of the system
    # handles worst, which is #35 M20-1 in this same file.
    try:
        from app.utils.tenancy import assert_job_tuple

        assert_job_tuple(get_supabase_client(), job_id, document_id, workspace_id)
    except Exception as _tenancy_err:
        _fail_job_terminally(
            job_id,
            f"Tenancy preflight failed: {_tenancy_err}",
            source="tenancy_preflight",
        )
        raise

    # ── Credit preflight ─────────────────────────────────────────────────────────────
    # Invariant 10 says debit BEFORE the upstream call. This pipeline cannot: every
    # debit_credits_* call fires after its Claude-vision / Replicate / Voyage request, so a
    # failed debit is money already spent and the logger can only record "UNBILLED". A
    # 0-credit account could therefore upload a large PDF and have the platform absorb the
    # entire multi-dollar job. (audit #286)
    #
    # This is the cheapest correction that actually bounds the loss: refuse to START when the
    # account is empty. It deliberately does NOT try to predict the job's cost — a preflight
    # that guesses either blocks paying customers or waves through the expensive jobs it was
    # meant to stop. It answers one question: is this account funded at all?
    #
    # An unreadable balance PROCEEDS. A transient RPC failure must not stop paying customers
    # from processing documents; the failure being removed is unbounded spend by an unfunded
    # account, not one job on a flaky read.
    try:
        _min_credits = float(os.getenv("PDF_JOB_MIN_CREDITS", "10"))
        # Own client binding: the module-level name `supabase` is not bound until much later in
        # this function, so referencing it here would raise NameError, be caught below, and turn
        # the whole guard into a permanent silent no-op.
        _sb = get_supabase_client()
        _job_row = _sb.client.table("background_jobs")             .select("user_id").eq("id", job_id).maybe_single().execute()
        _job_user = (_job_row.data or {}).get("user_id") if _job_row else None
        if _job_user:
            from app.services.integrations.credits_integration_service import get_credits_service
            _available = await get_credits_service().get_available_credits(
                user_id=str(_job_user), workspace_id=str(workspace_id) if workspace_id else None,
            )
            if _available is not None and _available < _min_credits:
                logger.error(
                    "🛑 [CREDIT PREFLIGHT] job=%s user=%s ws=%s has %.2f credits "
                    "(floor %.2f) — refusing to start rather than running the pipeline unbilled",
                    job_id, _job_user, workspace_id, _available, _min_credits,
                )
                _sb.client.table("background_jobs").update({
                    "status": "failed",
                    "error_message": (
                        f"Insufficient credits: {_available:.2f} available, "
                        f"{_min_credits:.2f} required to start processing."
                    ),
                    "completed_at": datetime.utcnow().isoformat(),
                }).eq("id", job_id).execute()
                return
    except Exception as _preflight_err:  # noqa: BLE001
        # Never let the guard itself break ingestion — it is a cost bound, not a correctness gate.
        logger.warning("[CREDIT PREFLIGHT] skipped (non-fatal): %s", _preflight_err)

    # Initialize lazy loading for this job
    from app.services.utilities.lazy_loader import get_component_manager
    component_manager = get_component_manager()

    # Track which components are loaded for cleanup
    loaded_components = []

    # Bind correlation IDs to ContextVars so every log record from this
    # background task is stamped with job_id / document_id (see
    # pipeline_observability.JobContextLogFilter).
    from app.utils.pipeline_observability import (
        _current_job_id,
        _current_document_id,
    )
    _current_job_id.set(job_id)
    _current_document_id.set(document_id)

    logger.info("=" * 80)
    logger.info("🔍 [PRODUCT DISCOVERY] STARTING")
    logger.info("=" * 80)
    logger.info(f"📋 Job ID: {job_id}")
    logger.info(f"📄 Document ID: {document_id}")
    logger.info(f"🤖 Discovery Model: {discovery_model.upper()}")
    logger.info(f"📦 Extract Categories: {', '.join(extract_categories).upper()}")

    # Open a Sentry transaction for the whole job; per-stage spans nest
    # under it. push_scope tags remain so existing breadcrumbs continue
    # to carry job_id even outside spans.
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("job_id", job_id)
        scope.set_tag("document_id", document_id)
        scope.set_tag("discovery_model", discovery_model)
        scope.set_context("job_config", {
            "extract_categories": extract_categories,
            "chunk_size": chunk_size
        })

        scope.set_context("job_details", {
            "filename": filename,
            "discovery_model": discovery_model,
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
         _fail_job_terminally(job_id, f"File not found at {file_path}", source="pre_stage_validation")
         raise FileNotFoundError(f"File not found at {file_path}")
    # Defensive: a previous worker may have left a 0-byte file behind
    # (e.g. mid-write SIGKILL from kernel OOM). pymupdf raises a confusing
    # "Cannot open empty file" later if we proceed; fail clearly here so
    # the caller can re-download from Supabase storage.
    try:
        _file_size_check = os.path.getsize(file_path)
    except OSError as e:
        logger.error(f"❌ [BACKGROUND TASK] Cannot stat {file_path}: {e}")
        _fail_job_terminally(job_id, f"Cannot stat PDF at {file_path}: {e}", source="pre_stage_validation")
        raise FileNotFoundError(f"Cannot stat PDF at {file_path}: {e}")
    if _file_size_check == 0:
        logger.error(f"❌ [BACKGROUND TASK] PDF at {file_path} is 0 bytes — likely truncated by prior crash")
        try:
            os.unlink(file_path)
        except OSError:
            pass
        _fail_job_terminally(
            job_id,
            f"PDF at {file_path} is 0 bytes (likely truncated by prior crash). Re-upload required.",
            source="pre_stage_validation",
        )
        raise FileNotFoundError(
            f"PDF at {file_path} is 0 bytes (likely truncated by prior crash). "
            f"Re-upload required."
        )

    logger.info("🔧 [BACKGROUND TASK] Opening file for reading...")
    with open(file_path, 'rb') as f:
        file_content = f.read()

    file_size = len(file_content)
    logger.info(f"✅ [BACKGROUND TASK] File read successfully: {file_size} bytes ({file_size / (1024*1024):.1f} MB)")
    logger.info("=" * 80)

    # Bound the document before anything iterates it (#35 M20-2). The upload size cap
    # does NOT cover this: a small compressed PDF can declare thousands of pages, and
    # every stage below is per-page. Third location for this gap after #22 M9-6 and
    # #24 M11-6, which is why the rule now lives in one place.
    try:
        import fitz as _fitz_bounds
        with _fitz_bounds.open(stream=file_content, filetype="pdf") as _probe:
            assert_page_count(len(_probe))
    except PdfBoundsError as _bounds_err:
        logger.error("❌ [BACKGROUND TASK] %s", _bounds_err)
        _fail_job_terminally(job_id, str(_bounds_err), source="pre_stage_validation")
        raise
    except Exception as _probe_err:  # noqa: BLE001
        # A PDF that will not open is a real failure, but it is the NEXT stage's to
        # report with its own diagnostics — this probe exists only to bound the size.
        logger.warning("[BACKGROUND TASK] page-count probe skipped: %s", _probe_err)

    # Stable catalog-cache key — hash the ORIGINAL uploaded PDF bytes, BEFORE
    # page numbering. The numbered PDF's bytes are not deterministic across runs,
    # so keying Stage 0's catalog cache on them caused a miss (and full ~$0.20
    # re-discovery) on every resume. Captured here while file_content is still
    # the original upload.
    import hashlib as _hashlib_orig
    original_pdf_sha256 = _hashlib_orig.sha256(file_content).hexdigest()

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

    from app.utils.pipeline_observability import pipeline_stage_span
    try:
        with pipeline_stage_span("preprocess.page_numbering"):
            numbered_pdf_path, numbering_stats = await preprocess_pdf_with_page_numbers(
                pdf_path=file_path,
                job_id=job_id,
                progress_callback=page_numbering_progress
            )
            logger.info(f"✅ Page numbering complete: {numbering_stats['pages_numbered']} pages")
            logger.info(f"   📄 Numbered PDF: {numbered_pdf_path}")

            # Use the numbered PDF for all subsequent processing
            file_path = numbered_pdf_path

            # P2-1: explicitly drop the original PDF bytes before re-reading
            # the numbered PDF, so we don't hold both copies in RAM
            # simultaneously (numbered PDFs are 10-50% larger).
            try:
                del file_content
                import gc as _gc_after_numbering
                _gc_after_numbering.collect()
            except Exception:
                pass

            # Re-read file content from numbered PDF
            with open(file_path, 'rb') as f:
                file_content = f.read()
            logger.info(f"   📖 Re-read numbered PDF: {len(file_content)} bytes")

    except Exception as e:
        # Hard-fail: every downstream stage (discovery, image extraction, vision
        # analysis, entity linking) assumes pages have visible numbers. Continuing
        # with the un-numbered PDF silently produces wrong page references.
        logger.error(f"❌ Page numbering failed — aborting job: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)
        job_storage[job_id]["status"] = "failed"
        job_storage[job_id]["error"] = f"Page numbering failure: {e}"
        if job_recovery_service:
            await job_recovery_service.persist_job(
                job_id=job_id,
                document_id=document_id,
                filename=filename,
                status="failed",
                progress=job_storage[job_id].get("progress", 5),
                metadata={**job_storage[job_id].get("metadata", {}), "error": str(e)},
            )
        raise

    # Update progress to 10% (page numbering complete)
    job_storage[job_id]["progress"] = 10
    job_storage[job_id]["metadata"]["current_step"] = "Page numbering complete"
    logger.info("=" * 80)

    # Get AI model configuration
    settings = get_settings()
    image_analysis_model = settings.image_analysis_model

    from app.models.ai_config import DEFAULT_AI_CONFIG
    product_creation_model = DEFAULT_AI_CONFIG.discovery_model
    quality_validation_model = DEFAULT_AI_CONFIG.classification_validation_model

    # Background heartbeat (every JOB_HEARTBEAT_INTERVAL_SECONDS) so the
    # auto-recovery cron can detect a dead orchestrator even when a single
    # stage stalls for many minutes.
    from app.services.tracking.job_heartbeat import JobHeartbeat
    supabase = get_supabase_client()
    heartbeat = JobHeartbeat(job_id=job_id, supabase_client=supabase)
    await heartbeat.__aenter__()
    try:
        # ============================================================================
        # POSTURE: warm-replica floor for the duration of this job
        # ============================================================================
        # Sets min_replica=1, max_replica=4, scaleToZeroTimeout=30min on all 4
        # HF endpoints. Floor=1 guarantees one warm replica throughout the job
        # (no per-stage cold-start tax), ceiling=4 allows HF to burst on load.
        # Paired with scale_all_to_zero() in the finally block, which sets
        # min=0 and lets HF's 30-min idle timer drain the last replica.
        try:
            from app.services.core.endpoint_controller import endpoint_controller
            await endpoint_controller.prepare_for_processing(
                reason=f"pdf_job_{job_id}",
                min_replica=1,
                max_replica=4,
                scale_to_zero_timeout=30,
            )
        except Exception as prep_err:
            logger.warning(
                f"⚠️ prepare_for_processing failed (job will continue, "
                f"may incur cold-start latency): {prep_err}"
            )

        # ============================================================================
        # WARMUP ALL HUGGINGFACE ENDPOINTS IN PARALLEL
        # ============================================================================
        logger.info("=" * 80)
        logger.info("🔥 WARMING UP ALL HUGGINGFACE ENDPOINTS (PARALLEL)")
        logger.info("=" * 80)

        # Create WARMUP_STARTED checkpoint
        warmup_endpoints_to_start = ["slig", "paddleocr"]
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

        async def warmup_slig():
            try:
                from app.services.embeddings.slig_endpoint_manager import SLIGEndpointManager
                slig_config = settings.get_slig_config()
                if slig_config.get("enabled", False) and slig_config.get("modal_url"):
                    # Modal-hosted: from_config builds the Modal endpoint provider.
                    # resume_if_needed() short-circuits when already warm + healthy,
                    # then warmup() /health-probes (auto-wakes a cold container).
                    manager = SLIGEndpointManager.from_config(slig_config)
                    endpoint_managers['slig'] = manager

                    def resume_and_warmup_slig():
                        if manager.resume_if_needed():
                            return manager.warmup()
                        return False

                    success = await asyncio.to_thread(resume_and_warmup_slig)
                    if success:
                        warmup_results["success"].append("slig")
                        logger.info("   ✅ SLIG warmup complete")
                    else:
                        warmup_results["failed"].append({"endpoint": "slig", "error": "warmup timed out"})
                        logger.warning("   ⚠️ SLIG warmup timed out (will retry on first use)")
                else:
                    warmup_results["skipped"].append("slig")
            except Exception as e:
                warmup_results["failed"].append({"endpoint": "slig", "error": str(e)})
                logger.warning(f"⚠️ Failed to warmup SLIG endpoint: {e}")

        async def warmup_paddleocr():
            try:
                from app.services.pdf.paddleocr_endpoint_manager import PaddleOCRManager
                paddle_config = settings.get_paddleocr_config()
                if paddle_config.get("enabled", False) and paddle_config.get("endpoint_url"):
                    # Provider-aware build (huggingface | modal). from_config picks
                    # the Modal endpoint provider.
                    manager = PaddleOCRManager.from_config(paddle_config)
                    endpoint_managers['paddleocr'] = manager

                    # Provider-agnostic warmup: resume_if_needed() short-circuits
                    # when the endpoint is already up + healthy (HF checks SDK
                    # status + /health and skips re-warm; Modal /health-probes and
                    # auto-wakes), then warmup() confirms readiness.
                    def resume_and_warmup_paddleocr():
                        if manager.resume_if_needed():
                            return manager.warmup()
                        return False

                    success = await asyncio.to_thread(resume_and_warmup_paddleocr)
                    if success:
                        warmup_results["success"].append("paddleocr")
                        logger.info("   ✅ PaddleOCR warmup complete")
                    else:
                        warmup_results["failed"].append({"endpoint": "paddleocr", "error": "warmup timed out"})
                        logger.warning("   ⚠️ PaddleOCR warmup timed out (will retry on first use)")
                else:
                    warmup_results["skipped"].append("paddleocr")
            except Exception as e:
                warmup_results["failed"].append({"endpoint": "paddleocr", "error": str(e)})
                logger.warning(f"⚠️ Failed to warmup PaddleOCR endpoint: {e}")

        # Execute all warmups in parallel. Both endpoints are Modal-hosted, so
        # warmup is just a /health probe that auto-wakes a cold container — there
        # is no HF-billing fast-fail path anymore.
        gather_results = await asyncio.gather(
            warmup_slig(),
            warmup_paddleocr(),
            return_exceptions=True,
        )
        # Re-raise any exception that gather captured so the orchestrator's
        # normal error handling kicks in.
        for r in gather_results:
            if isinstance(r, BaseException):
                raise r

        logger.info("=" * 80)
        logger.info(f"✅ WARMUP PHASE COMPLETE - {len(endpoint_managers)} endpoints resumed")
        logger.info(f"   Success: {warmup_results['success']}")
        logger.info(f"   Skipped (already running): {warmup_results['skipped']}")
        if warmup_results['failed']:
            logger.warning(f"   Failed: {warmup_results['failed']}")

            # Check if SLIG (critical for SLIG embeddings) failed
            failed_endpoints = [f.get('endpoint') if isinstance(f, dict) else f for f in warmup_results['failed']]
            if 'slig' in failed_endpoints:
                logger.error("❌ CRITICAL: SLIG endpoint warmup failed - cannot generate SLIG embeddings")
                logger.error("   Stopping processing to prevent incomplete data")

                # Update job status to failed
                job_storage[job_id]["status"] = "failed"
                job_storage[job_id]["error"] = "SLIG endpoint warmup failed - SLIG embeddings unavailable"
                if job_recovery_service:
                    await job_recovery_service.persist_job(
                        job_id=job_id,
                        document_id=document_id,
                        filename=filename,
                        status="failed",
                        progress=job_storage[job_id].get("progress", 0),
                        metadata=job_storage[job_id].get("metadata", {}),
                        error="SLIG endpoint warmup failed - SLIG embeddings unavailable"
                    )
                # NOT HTTPException — we are inside a BackgroundTask, no HTTP
                # response to attach a 503 to. Raise a plain RuntimeError so
                # the orchestrator's outer except + finally runs and triggers
                # scale-to-zero cleanup of the endpoints we already woke.
                raise RuntimeError(
                    "SLIG endpoint warmup failed - SLIG embeddings unavailable"
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
        # UNIFIED ENDPOINT CONTROLLER — align AIMD gates with auto-scaler
        # ============================================================================
        # The controller owns per-endpoint concurrency gates (slig/paddleocr).
        # Calling warm_all() here:
        #   1. Force-shrinks gates to minimum for any endpoint whose manager is
        #      missing or whose warmup failed (no piling in on a broken endpoint)
        #   2. Aligns per-gate maximums with current HF replica counts via the
        #      auto-scaler (supply-side info feeds demand-side throttling)
        # This is the last step before Stage 1 starts issuing real work.
        try:
            from app.services.core.endpoint_controller import endpoint_controller
            controller_outcome = await endpoint_controller.warm_all(job_id)
            endpoint_controller.log_stats(f"   Gates ready for job {job_id}")
            logger.info(f"🎛️  Endpoint controller aligned: {controller_outcome}")
        except Exception as e:
            # Controller alignment is best-effort — the pipeline can still run
            # with default gate values. Never fail the job on controller init.
            logger.warning(f"⚠️ Endpoint controller alignment skipped: {e}")

        # ============================================================================
        # HEALTH VALIDATION — confirm each warmed endpoint answers GET /health
        # ============================================================================
        # Both endpoints are Modal-hosted; a /health 200 means a container is
        # serving. warm_all() above already probed + force-minimumed broken gates.
        # This is the blocking gate: stop the job if a REQUIRED endpoint
        # (SLIG visual embeddings, PaddleOCR structural pass) isn't answering.
        # Vision analysis + classification run on Anthropic Claude (no warmup), so
        # they are not in this set.
        logger.info("=" * 80)
        logger.info("🔍 VALIDATING ENDPOINT HEALTH (/health probe)")
        logger.info("=" * 80)

        health_results = {}
        for name, manager in endpoint_managers.items():
            try:
                ok = await asyncio.to_thread(manager._test_inference)
            except Exception as probe_err:
                logger.warning(f"   ⚠️ {name} /health probe raised: {probe_err}")
                ok = False
            health_results[name] = ok
            logger.info(f"   {'✅' if ok else '❌'} {name}: {'healthy' if ok else 'UNHEALTHY'}")

        required_endpoints = [n for n in ('slig', 'paddleocr') if n in endpoint_managers]
        all_healthy = all(health_results.get(n, False) for n in required_endpoints)
        endpoint_registry.set_health_validated(
            all_healthy, {k: {"healthy": v} for k, v in health_results.items()}
        )

        if not all_healthy:
            failed_endpoints = [n for n in required_endpoints if not health_results.get(n, False)]
            error_msg = f"Required endpoints failed health check: {failed_endpoints}"
            logger.error(f"❌ {error_msg}")
            logger.error("   Pipeline cannot proceed without healthy endpoints")
            job_storage[job_id]["status"] = "failed"
            job_storage[job_id]["error"] = error_msg
            job_storage[job_id]["failed_at"] = datetime.utcnow().isoformat()
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

        logger.info("=" * 80)
        logger.info("✅ ALL REQUIRED ENDPOINTS HEALTHY - Ready to process")
        logger.info("=" * 80)

        # Mark processing as started (prevents auto-pause)
        endpoint_registry.start_processing(job_id)

        # ============================================================================
        # INITIALIZE PROGRESS TRACKING
        # ============================================================================
        logger.info("🔧 [BACKGROUND TASK] Initializing progress tracking components...")
        # Initialize Progress Tracker
        from app.services.tracking.progress_tracker import ProgressTracker
        # ProcessingStage already imported from checkpoint_recovery_service at module level (line 36)
        # checkpoint_recovery_service already imported at module level (line 36)
        from app.services.tracking.job_progress_monitor import JobProgressMonitor

        logger.info("🔧 [BACKGROUND TASK] Creating ProgressTracker...")
        tracker = ProgressTracker(
            job_id=job_id,
            document_id=document_id,
            total_pages=0,  # Will update after PDF extraction
            job_storage=job_storage
        )
        logger.info("🔧 [BACKGROUND TASK] Starting processing...")
        await tracker.start_processing()
        logger.info("✅ [BACKGROUND TASK] Processing started")

        # 🫀 Heartbeat: the JobHeartbeat thread (entered above via async-with)
        # is the canonical heartbeat — it survives blocked event loops, which
        # is the whole reason it exists. The asyncio-based
        # `tracker.start_heartbeat` was a second mechanism that wrote to the
        # SAME row (4 selects/min per job for liveness), and was fragile on
        # long sync work. Removed 2026-05-23 per round-3 audit.
        logger.info("🫀 [BACKGROUND TASK] JobHeartbeat thread is the active liveness signal")

        logger.info("🔧 [BACKGROUND TASK] Starting progress monitor...")
        # 📊 Start detailed progress monitoring (reports every 60s to logs + Sentry)
        progress_monitor = JobProgressMonitor(job_id=job_id, document_id=document_id, total_stages=9)
        await progress_monitor.start()
        logger.info(f"✅ [BACKGROUND TASK] Progress monitoring started for job {job_id}")

        logger.info("🔧 [BACKGROUND TASK] Creating INITIALIZED checkpoint...")
        # Create INITIALIZED checkpoint
        await checkpoint_recovery_service.create_checkpoint(
            job_id=job_id,
            stage=ProcessingStage.INITIALIZED,
            data={
                "document_id": document_id,
                "filename": filename,
                "file_size": len(file_content),
                "pdf_path": file_path  # vision-guided extraction needs the on-disk path
            },
            metadata={
                "title": title or filename,
                "description": description,
                "tags": document_tags,
                "discovery_model": discovery_model,
            }
        )
        logger.info(f"✅ [BACKGROUND TASK] INITIALIZED checkpoint created for job {job_id}")
        logger.info(f"   📄 PDF path saved: {file_path}")

        # ============================================================================
        # EXTRACT-ONLY MODE: skip discovery + Stages 1.5/2/3/4/4.5/4.6/4.7/5
        # ============================================================================
        # When the upload route receives categories='extract_only', the caller
        # wants raw text/image extraction without any AI categorization. The
        # rest of the pipeline (discovery, products, embeddings, quality) is
        # skipped — Stage 1.5 is still run because layout precompute is a
        # prerequisite for any downstream search use, even without products.
        if extract_only_mode:
            logger.info(
                "📄 [EXTRACT-ONLY MODE] Skipping discovery + product pipeline. "
                "Running Stage 1.5 layout precompute only, then completing."
            )
            try:
                from app.api.pdf_processing.stage_1_layout_precompute import precompute_document_layout
                _layout_summary = await precompute_document_layout(
                    document_id=document_id,
                    pdf_path=file_path,
                    supabase=get_supabase_client(),
                    logger=logger,
                    job_id=job_id,
                    tracker=tracker,
                )
                logger.info(f"✅ [EXTRACT-ONLY MODE] Stage 1.5 complete: {_layout_summary}")
                # Mark complete via the documented tracker signatures.
                await tracker.update_progress(
                    progress_percentage=100,
                    details={
                        'current_step': 'Extract-only mode complete',
                        'completion_reason': 'extract_only_mode',
                    },
                )
                try:
                    get_supabase_client().client.rpc(
                        'merge_background_job_metadata',
                        {
                            'p_job_id': job_id,
                            'p_metadata': {
                                'completion_reason': 'extract_only_mode',
                                'layout_summary': _layout_summary,
                            },
                        }
                    ).execute()
                except Exception:
                    pass
                await tracker.complete_job(result={
                    'mode': 'extract_only',
                    'layout_summary': _layout_summary,
                })
            except Exception as eo_err:
                logger.error(f"❌ extract-only mode failed: {eo_err}", exc_info=True)
                await tracker.fail_job(error=eo_err)
            return

        # ============================================================================
        # STAGE 1: DOCUMENT-LEVEL STRUCTURAL PASS (PaddleOCR) — runs BEFORE discovery
        # ============================================================================
        # Structure-first architecture: PaddleOCR renders every physical page once
        # and returns layout regions + OCR text + figure boxes, persisted to
        # document_layout_analysis. Running it first means:
        #   - discovery reads PaddleOCR's reading-order text (cleaner, multilingual,
        #     layout-ordered) instead of raw page.get_text(),
        #   - Stage 2 chunking reads the same cache,
        #   - Stage 3 crops come from PaddleOCR's figure boxes (cache hit).
        # One pass feeds the whole pipeline.
        # Imported before the try so the `except LayoutPrecomputeFatalError` clause
        # below always has the name bound, even if the inner import were to fail.
        from app.api.pdf_processing.stage_1_layout_precompute import (
            LayoutPrecomputeFatalError,
        )
        try:
            _settings_for_precompute = get_settings()
            if getattr(_settings_for_precompute, "layout_precompute_enabled", True):
                from app.api.pdf_processing.stage_1_layout_precompute import (
                    precompute_document_layout,
                )
                logger.info("=" * 80)
                logger.info("📐 STAGE 1: DOCUMENT-LEVEL STRUCTURAL PASS (PaddleOCR)")
                logger.info("=" * 80)
                precompute_summary = await precompute_document_layout(
                    document_id=document_id,
                    pdf_path=file_path,
                    supabase=get_supabase_client(),
                    logger=logger,
                    job_id=job_id,
                    tracker=tracker,  # nest-safe slow_op stack
                )
                job_storage[job_id]["metadata"] = {
                    **job_storage[job_id].get("metadata", {}),
                    "layout_precompute": precompute_summary,
                }
            else:
                logger.info("⏭️  [STAGE 1] Skipped (LAYOUT_PRECOMPUTE_ENABLED=False)")
        except LayoutPrecomputeFatalError:
            # Intended-fatal: endpoint misconfigured, or circuit-breaker tripped
            # with zero successes. Propagate so the outer handler fails the job
            # (SPN-3) — do NOT downgrade to a warning and silently process the
            # whole catalog on inferior PyMuPDF text.
            logger.error("❌ [STAGE 1] structural pass hit a FATAL condition — failing job", exc_info=True)
            raise
        except Exception as precompute_err:
            # Best-effort: discovery falls back to PyMuPDF text, chunker to its
            # text path. A transient per-page failure shouldn't sink the job.
            logger.warning(
                f"⚠️ [STAGE 1] structural pass failed (non-fatal, falling back to PyMuPDF text): {precompute_err}",
                exc_info=True,
            )

        # ============================================================================
        # STAGE 0: PRODUCT DISCOVERY (MODULAR)
        # ============================================================================
        logger.info("🚀 [BACKGROUND TASK] ========================================")
        logger.info("🚀 [BACKGROUND TASK] STARTING STAGE 0: PRODUCT DISCOVERY")
        logger.info("🚀 [BACKGROUND TASK] ========================================")
        progress_monitor.update_stage("product_discovery", {"discovery_model": discovery_model})
        from app.api.pdf_processing.stage_0_discovery import process_stage_0_discovery

        logger.info("🔧 [BACKGROUND TASK] Calling process_stage_0_discovery...")
        with pipeline_stage_span("stage_0.product_discovery", extra_data={"discovery_model": discovery_model}):
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
                temp_pdf_path=file_path,
                test_single_product=test_single_product,
                catalog_cache_key=original_pdf_sha256,
            )

        catalog = stage_0_result["catalog"]
        page_count = stage_0_result["page_count"]
        file_size_mb = stage_0_result["file_size_mb"]
        temp_pdf_path = stage_0_result["temp_pdf_path"]

        # ============================================================================
        # ZERO-ENTITIES EARLY EXIT
        # ============================================================================
        # If Stage 0 discovered no products AND no other entities in the requested
        # categories, downstream stages have nothing to operate on. Running them
        # anyway pays full HF / Voyage / Anthropic cost for empty output. This is
        # the explicit short-circuit the round-2 audit flagged as missing.
        _products_n        = len(catalog.products)
        _certificates_n    = len(catalog.certificates) if "certificates"    in extract_categories else 0
        _logos_n           = len(catalog.logos)        if "logos"           in extract_categories else 0
        _specifications_n  = len(catalog.specifications) if "specifications" in extract_categories else 0
        _total_entities    = _products_n + _certificates_n + _logos_n + _specifications_n

        if _total_entities == 0:
            logger.warning(
                f"⚠️  [STAGE 0] Discovery returned ZERO entities for categories "
                f"{extract_categories}. Likely causes: (a) PDF text layer is empty "
                f"(scanned-only catalog — discovery is text-based, won't find "
                f"image-baked product names), (b) LLM JSON parsing failed and all "
                f"items were filtered, (c) catalog actually contains nothing in the "
                f"requested categories. Short-circuiting downstream stages."
            )
            # Mark the job complete with an explicit reason rather than racing
            # through Stages 1.5/2/3/4 over an empty set. Use the documented
            # tracker signatures: update_progress(progress_percentage, details),
            # complete_job(result). Merge metadata via the RPC so existing
            # keys (filename, discovery_model, cost summaries) aren't clobbered.
            try:
                await tracker.update_progress(
                    progress_percentage=100,
                    details={
                        'current_step': 'No entities discovered — short-circuiting downstream stages',
                        'completion_reason': 'zero_entities_discovered',
                    },
                )
                # Merge metadata BEFORE complete_job so the reason is durable
                # even if complete_job's own metadata writes happen first.
                # CORRECTION (post-round-3): the in-scope variable name is
                # `supabase`, not `supabase_client` — the latter is undefined
                # inside this function and was NameError'ing every RPC call,
                # which the outer try/except silently swallowed. Same fix
                # applied to the stage_history RPC + fallback write below.
                try:
                    supabase.client.rpc(
                        'merge_background_job_metadata',
                        {
                            'p_job_id': job_id,
                            'p_metadata': {
                                'completion_reason': 'zero_entities_discovered',
                                'extract_categories': extract_categories,
                                'zero_entities_at': datetime.utcnow().isoformat(),
                            },
                        }
                    ).execute()
                except Exception as _merge_err:
                    logger.debug(f"merge_background_job_metadata failed (non-fatal): {_merge_err}")
                # Emit a discrete stage_history event so the audit log shows
                # why the job ended — append_stage_history caps at 100 entries
                # so this is the LAST one operators see for the job.
                try:
                    supabase.client.rpc(
                        'append_stage_history',
                        {
                            'p_job_id': job_id,
                            'p_event': {
                                'stage': 'stage_0_discovery',
                                'status': 'completed_no_entities',
                                'data': {
                                    'reason': 'zero_entities_discovered',
                                    'categories': extract_categories,
                                    'products_discovered': _products_n,
                                    'certificates_discovered': _certificates_n,
                                    'logos_discovered': _logos_n,
                                    'specifications_discovered': _specifications_n,
                                },
                                'completed_at': datetime.utcnow().isoformat(),
                            },
                        }
                    ).execute()
                except Exception as _hist_err:
                    logger.debug(f"append_stage_history failed (non-fatal): {_hist_err}")
                # Terminal: complete_job(result) — the result dict surfaces in
                # /full-status.core.result_data and the admin UI.
                await tracker.complete_job(result={
                    'reason': 'zero_entities_discovered',
                    'categories': extract_categories,
                    'products_discovered': _products_n,
                    'certificates_discovered': _certificates_n,
                    'logos_discovered': _logos_n,
                    'specifications_discovered': _specifications_n,
                })
            except Exception as _early_exit_err:
                logger.error(
                    f"❌ Zero-entities early-exit bookkeeping failed: {_early_exit_err}",
                    exc_info=True,
                )
                # Don't let bookkeeping failure leave the job at 'processing'.
                # Best-effort direct write so the job is marked terminal.
                try:
                    supabase.client.table('background_jobs').update({
                        'status': 'completed',
                        'progress': 100,
                        'completed_at': datetime.utcnow().isoformat(),
                        'updated_at': datetime.utcnow().isoformat(),
                    }).eq('id', job_id).execute()
                except Exception:
                    pass
            return  # don't run downstream stages

        # ============================================================================
        # PRODUCT-CENTRIC PIPELINE: Process each product individually
        # ============================================================================
        logger.info(f"\n{'='*80}")
        logger.info(f"🏭 PRODUCT-CENTRIC PIPELINE: Processing {len(catalog.products)} products")
        logger.info(f"{'='*80}\n")

        # Replica provisioning is Modal-managed now (both SLIG + PaddleOCR-VL on
        # Modal autoscale on demand up to max_containers) — no proactive HF
        # scale-up call needed. The AdaptiveConcurrency gates still throttle
        # in-flight load under backpressure.

        # NOTE: the document-level structural pass (PaddleOCR) now runs as STAGE 1
        # BEFORE discovery (see above) — structure-first. It is no longer run
        # here after discovery.

        # Initialize product progress tracker
        from app.services.tracking.product_progress_tracker import ProductProgressTracker
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
            'extract_categories': extract_categories,
            'material_category': material_category  # For image categorization
        }

        # Track overall metrics
        total_products = len(catalog.products)

        # ========================================================================
        # CATALOG-WIDE ICON EXTRACTION (runs ONCE before the product loop)
        # ========================================================================
        # Ceramic catalogs typically put shared legend/iconography pages
        # (slip ratings, PEI wear classes, fire ratings, shade variation) on
        # pages not assigned to any product. If we only look at per-product
        # pages in Stage 3, those icons are never OCR'd and the rollup to
        # `product.metadata.slip_resistance` / `.pei_rating` / ... fails with
        # all-nulls for legitimately-populated catalogs.
        #
        # This pre-pass scans `catalog.supplementary_pages` (computed by
        # Stage 0 discovery) for icon-shaped images and routes them through
        # the same OCR + Claude icon pipeline as per-product icons. The
        # resulting document_images rows share the same document_id, and the
        # Stage 4 `_merge_icon_metadata_into_product` rollup already queries
        # all images for the document — so every product in this catalog
        # automatically inherits the catalog-wide spec defaults.
        # P2-9: skip the catalog-wide icon pass when the user's category
        # selection makes it irrelevant. The pass calls Claude on every
        # supplementary page — pure waste if the user only asked for
        # categories whose products have no compliance icons.
        #
        # 2026-05-03 fix: `'products'` MUST trigger the pass. Ceramic
        # catalogs put slip resistance / PEI / fire rating / water absorption /
        # frost / shade-variation icons on shared legend pages, NOT on each
        # product's own pages. Stage 4's `_merge_icon_metadata_into_product`
        # rollup queries all icon images for the document and copies their
        # spec verdicts onto every product's metadata. Skip the pass and
        # those product fields all stay null — exactly the all-empty
        # `compliance / performance` blocks we saw on FOLD (job acff9ebb).
        _icon_relevant_categories = {'products', 'specifications', 'certificates', 'logos'}
        _icon_pass_relevant = bool(set(extract_categories) & _icon_relevant_categories)

        # Test-mode skip (added 2026-05-03): the icon pre-pass is a catalog-level
        # operation that scans ALL supplementary pages — typically 30+ pages of
        # OCR + Anthropic icon extraction. In `test_single_product=True` runs we
        # only process one product, so spending 20-100 minutes on a catalog-wide
        # pre-pass before that single product's Stage 3 is the wrong tradeoff
        # (job 051e1dda timed out here without ever reaching VALENOVA). Skip
        # it; the missing compliance/PEI rollups can be backfilled by a full run.
        if test_single_product and _icon_pass_relevant:
            logger.info(
                "🔖 Skipping catalog-wide icon pass — test_single_product=True "
                "(catalog-level pre-pass would burn the per-product budget)"
            )
            catalog_icon_stats = {
                'supplementary_pages_scanned': 0,
                'icon_candidates_found': 0,
                'icon_metadata_extracted': 0,
                'icon_extraction_failed': 0,
                'skipped': True,
                'skipped_reason': 'test_single_product',
            }
        elif not _icon_pass_relevant:
            logger.info(
                f"🔖 Skipping catalog-wide icon pass — "
                f"categories={extract_categories} don't include icon-relevant types"
            )
            catalog_icon_stats = {
                'supplementary_pages_scanned': 0,
                'icon_candidates_found': 0,
                'icon_metadata_extracted': 0,
                'icon_extraction_failed': 0,
                'skipped': True,
                'skipped_reason': 'icon-irrelevant categories',
            }
        else:
            try:
                from app.api.pdf_processing.stage_3_images import process_catalog_wide_icons

                logger.info("🔖 Running catalog-wide icon extraction pass (supplementary pages)...")
                if tracker:
                    tracker.current_step = "Catalog-wide icon extraction"
                    await tracker.update_heartbeat()
                    # This pre-pass can take 30-60+ minutes on large catalogs
                    # (job 051e1dda timed out here). Mark it as a known slow
                    # operation so auto-recovery cron's stuck-job detector
                    # doesn't false-positive while it runs. Always cleared in
                    # the finally block, even on exception, so a failed icon
                    # pass doesn't poison the next stage's auto-recovery.
                    try:
                        await tracker.set_slow_operation(
                            operation="stage_3_images:process_catalog_wide_icons",
                            expected_max_seconds=3600,
                        )
                    except Exception as _slow_err:
                        logger.debug(f"   set_slow_operation failed (non-fatal): {_slow_err}")

                catalog_icon_stats = await process_catalog_wide_icons(
                    file_content=file_content,
                    document_id=document_id,
                    workspace_id=workspace_id,
                    job_id=job_id,
                    catalog=catalog,
                    logger=logger,
                )
                logger.info(
                    f"🔖 Catalog-wide icon pass complete: "
                    f"pages={catalog_icon_stats.get('supplementary_pages_scanned', 0)}, "
                    f"icons_found={catalog_icon_stats.get('icon_candidates_found', 0)}, "
                    f"icons_with_spec_items={catalog_icon_stats.get('icon_metadata_extracted', 0)}, "
                    f"failed={catalog_icon_stats.get('icon_extraction_failed', 0)}"
                )
            except Exception as cat_icons_err:
                # This is a best-effort pre-pass — the rest of the pipeline MUST
                # keep running if it fails. We surface to Sentry so ops can see it.
                logger.warning(
                    f"⚠️ Catalog-wide icon pass failed (non-blocking): {cat_icons_err}"
                )
                try:
                    sentry_sdk.capture_exception(cat_icons_err)
                except Exception:
                    pass
            finally:
                # Always clear the slow-op marker so it doesn't leak into the
                # next stage and poison auto-recovery.
                if tracker:
                    try:
                        await tracker.clear_slow_operation()
                    except Exception:
                        pass

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
        # Concurrency cap, batch size and memory threshold come from settings
        # (env: MAX_CONCURRENT_PRODUCTS / PRODUCT_BATCH_SIZE / PRODUCT_MEMORY_THRESHOLD_MB)
        # so capacity is tunable per server. file_content is passed by reference
        # to all product workers — never copied per-product.
        parallel_config = ParallelProcessingConfig(
            enable_parallel=len(products_to_process) > 2,
        )

        logger.info(f"🚀 Processing mode: {'PARALLEL' if parallel_config.enable_parallel else 'SEQUENTIAL'}")
        if parallel_config.enable_parallel:
            logger.info(f"   Max concurrent products: {parallel_config.max_concurrent}")

        # Process all products (parallel or sequential based on config)
        with pipeline_stage_span(
            "products.parallel_processing",
            extra_data={
                "products": len(products_to_process),
                "max_concurrent": parallel_config.max_concurrent,
                "parallel": parallel_config.enable_parallel,
            },
        ):
            # `physical_page_upper_bound` is the upper bound used by
            # stage_1_focused_extraction to validate `physical_page > bound`.
            # For spread-layout catalogs (e.g. art-book layouts where each
            # PDF sheet contains 2 physical pages side-by-side),
            # `page_count` returns PDF sheet count (e.g. 71) while physical
            # page numbers go up to e.g. 140. Using `page_count` as the
            # bound silently drops every product whose pages live past the
            # sheet count — i.e. the back half of any spread-layout
            # catalog. Prefer `catalog.total_pages` (physical page count)
            # when present, falling back to page_count only for non-spread
            # layouts where they're equal.
            #
            # The `as_physical_page_bound` wrapper is a NewType-style guard
            # that documents the intended meaning of the int and validates
            # it's ≥1. See app/schemas/page_types.py for the rationale.
            from app.schemas.page_types import as_physical_page_bound
            _raw_bound = getattr(catalog, "total_pages", None) or page_count
            physical_page_bound = as_physical_page_bound(_raw_bound)
            if _raw_bound != page_count:
                logger.info(
                    f"📐 Spread layout: using catalog.total_pages={physical_page_bound} "
                    f"(physical) for page validation instead of page_count={page_count} "
                    f"(PDF sheets)"
                )

            parallel_result = await process_products_parallel(
                products=products_to_process,
                file_content=file_content,
                document_id=document_id,
                workspace_id=workspace_id,
                job_id=job_id,
                catalog=catalog,
                tracker=tracker,
                product_tracker=product_tracker,
                checkpoint_recovery_service=checkpoint_recovery_service,
                supabase=supabase,
                config=processing_config,
                logger_instance=logger,
                physical_page_upper_bound=physical_page_bound,
                temp_pdf_path=file_path,
                parallel_config=parallel_config,
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
        logger.info("🏭 PRODUCT-CENTRIC PIPELINE COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"✅ Products completed: {products_completed}/{total_products}")
        logger.info(f"❌ Products failed: {products_failed}/{total_products}")
        logger.info(f"📝 Total chunks created: {total_chunks_created}")
        logger.info(f"🖼️  Total images processed: {total_images_processed}")
        logger.info(f"🎨 Total SLIG embeddings: {total_clip_embeddings}")
        logger.info(f"🔗 Total relationships created: {total_relationships_created}")
        logger.info(f"⏱️  Processing time: {parallel_result.processing_time_seconds:.1f}s")
        logger.info(f"{'='*80}\n")

        # Anomaly check — a job that produced products but zero chunks is
        # technically "completed" yet returns nothing on chunk-level RAG
        # search. Surface as a Sentry warning so ops sees it before users do.
        if products_completed > 0 and total_chunks_created == 0:
            warn_msg = (
                f"⚠️ Job {job_id} completed with {products_completed} product(s) "
                f"but ZERO chunks. Document will not be findable via chunk search."
            )
            logger.warning(warn_msg)
            try:
                sentry_sdk.capture_message(warn_msg, level="warning")
            except Exception:
                pass

        # Hard failure gate: if EVERY product failed, the document produced no
        # usable output. Proceeding to Stage 5 + complete_job here would mark the
        # job green/100% and mask total data loss with no retry. Raise so the
        # outer handler (except @ ~4430) runs fail_job + flips the document out
        # of 'processing'. (The anomaly check above only fires when products
        # SUCCEEDED but produced zero chunks — it does NOT cover all-failed.)
        if total_products > 0 and products_completed == 0:
            raise RuntimeError(
                f"All {total_products} product(s) failed in the product-centric "
                f"pipeline (products_failed={products_failed}) — failing job {job_id} "
                f"instead of completing with zero output."
            )

        # Partial-loss signal (SPN-2): SOME products failed but not all. The job
        # still completes (the succeeded products are usable), but a bare
        # status='completed'/100% masks the loss — the only prior signal was the
        # per-product `product_processing_status='failed'` rows, which don't touch
        # the job. Surface it at the job level: a queryable metadata flag + audit
        # event + Sentry warning, so ops sees a half-lost catalog before users do.
        if total_products > 0 and products_failed > 0:
            partial_msg = (
                f"Job {job_id} completed with PARTIAL product loss: "
                f"{products_failed}/{total_products} product(s) failed "
                f"({products_completed} succeeded)."
            )
            logger.warning(f"⚠️ {partial_msg}")
            try:
                supabase.client.rpc('merge_background_job_metadata', {
                    'p_job_id': job_id,
                    'p_metadata': {
                        'completion_reason': 'completed_with_failures',
                        'products_failed': products_failed,
                        'products_completed': products_completed,
                        'total_products': total_products,
                    },
                }).execute()
            except Exception as _meta_err:
                logger.debug(f"partial-loss metadata stamp failed (non-fatal): {_meta_err}")
            try:
                supabase.client.rpc('append_stage_history', {
                    'p_job_id': job_id,
                    'p_event': {
                        'stage': 'product_processing',
                        'status': 'completed_with_failures',
                        'data': {
                            'products_failed': products_failed,
                            'products_completed': products_completed,
                            'total_products': total_products,
                        },
                        'completed_at': datetime.utcnow().isoformat(),
                    },
                }).execute()
            except Exception as _hist_err:
                logger.debug(f"partial-loss stage_history append failed (non-fatal): {_hist_err}")
            try:
                sentry_sdk.capture_message(partial_msg, level="warning")
            except Exception:
                pass

        # Update tracker with final counts. These are the AUTHORITATIVE
        # per-job totals — set them from the parallel_result aggregates (each
        # summed once from per-product results) so the persisted job row is
        # correct regardless of any transient over-count or concurrent-write
        # race in the in-memory counters during processing.
        tracker.chunks_created = total_chunks_created
        tracker.images_extracted = total_images_processed
        tracker.clip_embeddings_generated = total_clip_embeddings
        tracker.image_embeddings_generated = total_clip_embeddings
        tracker.products_created = products_completed
        tracker.relations_created = total_relationships_created
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
            enrich_products_from_chunks_and_vision,
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

        # ============================================================================
        # STAGE 4.7: DETERMINISTIC ENRICHMENT FROM CHUNKS + VISION_ANALYSIS
        # ============================================================================
        # For products that still have empty factory_name / designers / SKUs /
        # grout_suppliers / body_type / material_category / finish after AI
        # extraction + Stage 4.5 propagation + Stage 4.6 dimension extraction,
        # run deterministic regex extractors over the chunks AND majority-vote
        # rollup from document_images.vision_analysis.
        #
        # Only fills null/empty fields. Never overwrites AI values. No AI calls.
        progress_monitor.update_stage("product_enrichment", {
            "description": "Filling empty fields from chunks and vision_analysis"
        })
        await tracker.update_progress(78, {
            "current_step": "Enriching products from chunks and vision_analysis"
        })

        enrichment_result = await enrich_products_from_chunks_and_vision(
            document_id=document_id,
            supabase=supabase,
            logger=logger,
            # Plumb the job_id through so Layers 1/2/3 can emit progress
            # updates via ProgressTrackingService and the frontend
            # AsyncJobQueueMonitor can render the catalog-layout /
            # legend-extraction / spec-vision sub-stages in real time.
            job_id=getattr(tracker, "job_id", None),
        )
        logger.info(
            f"🧩 Product enrichment: "
            f"{enrichment_result.get('products_updated', 0)}/{enrichment_result.get('products_checked', 0)} "
            f"products updated — fields_filled: {enrichment_result.get('fields_filled', [])}"
        )

        await tracker.update_progress(80, {
            "current_step": (
                f"Enrichment done — "
                f"{enrichment_result.get('products_updated', 0)} products filled from chunks/vision"
            )
        })
        await tracker._sync_to_database(stage="product_enrichment")

        # ── Factory enrichment trigger (async, non-blocking) ─────────────────
        try:
            from app.api.pdf_processing.stage_4_products import _trigger_factory_enrichment
            # Filter to products that actually completed processing — running
            # enrichment over a product whose Stage 2 (chunks) or Stage 3 (images)
            # failed means feeding garbage to Voyage. The 2026-05-23 audit
            # surfaced this orphan-enrichment risk.
            _ok_products_resp = (
                supabase.client.table('products')
                .select('id')
                .eq('source_document_id', document_id)
                .execute()
            )
            all_product_ids = [p['id'] for p in (_ok_products_resp.data or [])]
            # Exclude products whose product_processing_status is 'failed'.
            if all_product_ids:
                try:
                    _failed_resp = (
                        supabase.client.table('product_processing_status')
                        .select('product_id, status')
                        .in_('product_id', all_product_ids)
                        .eq('status', 'failed')
                        .execute()
                    )
                    _failed_ids = {row['product_id'] for row in (_failed_resp.data or []) if row.get('product_id')}
                    if _failed_ids:
                        before = len(all_product_ids)
                        all_product_ids = [pid for pid in all_product_ids if pid not in _failed_ids]
                        logger.info(
                            f"Factory enrichment: excluded {before - len(all_product_ids)} "
                            f"failed product(s); processing {len(all_product_ids)} successful ones"
                        )
                except Exception as _filter_err:
                    logger.debug(f"product_processing_status filter failed (non-fatal): {_filter_err}")

            if all_product_ids:
                # Do NOT re-import asyncio in this function — asyncio is module-scoped (line 20).
                # A local `import asyncio` rebinds it function-scoped and breaks the warm-up
                # `await asyncio.to_thread(...)` calls earlier in the body with UnboundLocalError.
                #
                # Hard cap factory enrichment so a hung downstream call can't keep HF endpoints
                # pinned — endpoint_registry treats this fire-and-forget task as part of the job.
                async def _factory_enrichment_with_timeout():
                    try:
                        await asyncio.wait_for(
                            _trigger_factory_enrichment(
                                workspace_id=workspace_id,
                                product_ids=all_product_ids,
                                scope_column='source_document_id',
                                scope_value=document_id,
                                logger=logger,
                            ),
                            timeout=120,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"⏱️ Factory enrichment for document {document_id} "
                            f"exceeded 120s — abandoning so endpoints can scale down"
                        )

                _fe_task = asyncio.create_task(_factory_enrichment_with_timeout())
                _fe_task.add_done_callback(lambda t: logger.error(
                    f"❌ Factory enrichment task failed: {t.exception()}",
                    exc_info=t.exception(),
                ) if not t.cancelled() and t.exception() else None)
        except Exception as _fe:
            logger.warning(f"⚠️ Factory enrichment trigger failed (non-blocking): {_fe}")

        # ── Rebuild gold-layer product relationship edges ────────────────────
        # SQL rule-derived edges are rebuilt once per job (whole workspace, cheap);
        # the LLM pass extracts text-stated complementary/alternative edges over
        # just this document's products. Both non-blocking.
        try:
            from app.services.products.product_relationship_service import ProductRelationshipService
            _rel_service = ProductRelationshipService(supabase.client)
            _edge_count = await _rel_service.rebuild_edges(workspace_id)
            logger.info(f"🔗 Product edges rebuilt (rules): {_edge_count} for workspace {workspace_id}")

            _doc_pids_resp = supabase.client.table('products').select('id').eq(
                'source_document_id', document_id
            ).execute()
            _doc_pids = [r['id'] for r in (_doc_pids_resp.data or [])]
            if _doc_pids:
                _llm_edges = await _rel_service.extract_llm_edges(
                    workspace_id, product_ids=_doc_pids, job_id=job_id
                )
                logger.info(f"🔗 Product edges extracted (LLM): {_llm_edges} for {len(_doc_pids)} products")
        except Exception as _ee:
            logger.warning(f"⚠️ Product edge rebuild failed (non-blocking): {_ee}")

        # ── Page embeddings — the 8th fusion vector (#239) ───────────────────
        # Runs HERE, after chunking, because the page text it embeds alongside the
        # render comes from `document_chunks` (silver). Running it earlier would mean
        # re-extracting text from the PDF, which is the layer violation the Medallion
        # rule exists to prevent.
        #
        # Awaited rather than fire-and-forget: `document_page_embeddings` rows are what
        # the silent-zero probe reads, and a background task that loses its race with
        # job completion would leave the document looking permanently unembedded.
        # Failures are non-fatal — a catalog with no page vectors is a catalog with
        # seven working channels, not a failed ingest.
        try:
            from app.services.embeddings.page_embedding_service import get_page_embedding_service

            with pipeline_stage_span("stage_4.7.page_embeddings"):
                _page_result = await asyncio.wait_for(
                    get_page_embedding_service().embed_document_pages(
                        document_id=document_id,
                        workspace_id=workspace_id,
                        job_id=job_id,
                    ),
                    # Bounded so a Voyage outage cannot hold the whole job open: the
                    # per-page rows already written stay, and the backfill picks up
                    # whatever this run did not reach.
                    timeout=1800,
                )
            logger.info(
                f"📄 Page embeddings: {_page_result.get('embedded', 0)} embedded, "
                f"{_page_result.get('failed', 0)} failed, "
                f"{_page_result.get('skipped_blank', 0)} blank "
                f"(of {_page_result.get('pages_considered', 0)} pages)"
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"⏱️ Page embeddings for {document_id} exceeded 30min — abandoning; "
                f"already-written pages are kept and the rest stay retryable"
            )
        except Exception as _pe:
            logger.warning(f"⚠️ Page embeddings failed (non-blocking): {_pe}")

        # ============================================================================
        # STAGE 5: QUALITY ENHANCEMENT (MODULAR)
        # ============================================================================
        progress_monitor.update_stage("quality_enhancement", {"products_created": products_created})
        from app.api.pdf_processing.stage_5_quality import process_stage_5_quality
        from app.utils.circuit_breaker import claude_breaker

        with pipeline_stage_span("stage_5.quality_enhancement"):
            stage_5_result = await process_stage_5_quality(
                document_id=document_id,
                job_id=job_id,
                workspace_id=workspace_id,
                catalog=catalog,
                physical_pages=list(all_physical_pages),
                products_created=products_created,
                images_processed=images_saved_count,
                quality_validation_model=quality_validation_model,
                start_time=start_time,
                tracker=tracker,
                checkpoint_recovery_service=checkpoint_recovery_service,
                component_manager=component_manager,
                loaded_components=loaded_components,
                claude_breaker=claude_breaker,
                logger=logger,
            )

        # Stage 5 handles all SUCCESS cleanup (component unloading, resource cleanup, job completion)
        logger.info("✅ [MODULAR PIPELINE] All stages completed successfully")

        # Flip the document record so it's no longer stuck in 'processing'.
        try:
            _supabase = get_supabase_client()
            _supabase.client.table('documents').update({
                'processing_status': 'completed',
            }).eq('id', document_id).execute()
            logger.info(f"   ✅ Marked document {document_id} as completed")
        except Exception as status_error:
            logger.warning(f"   ⚠️ Failed to mark document completed: {status_error}")

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

        # Flip the document record so it stops showing as 'processing' forever.
        try:
            _supabase = get_supabase_client()
            existing = _supabase.client.table('documents').select('metadata').eq('id', document_id).single().execute()
            merged_metadata = (existing.data or {}).get('metadata') or {}
            merged_metadata['error'] = str(e)[:2000]
            merged_metadata['error_type'] = type(e).__name__
            merged_metadata['failed_at'] = datetime.utcnow().isoformat()
            merged_metadata['failed_stage'] = current_stage if 'current_stage' in locals() else 'unknown'
            _supabase.client.table('documents').update({
                'processing_status': 'failed',
                'metadata': merged_metadata,
            }).eq('id', document_id).execute()
            logger.info(f"   ✅ Marked document {document_id} as failed")
        except Exception as status_error:
            logger.warning(f"   ⚠️ Failed to update document status to failed: {status_error}")

        # Rollback products created during discovery — but ONLY when this is the
        # terminal failure (no more recovery attempts left). Previously this ran
        # on every exception, so the next auto-recovery dispatch had to
        # re-discover and re-create every product from scratch, and products
        # that did succeed before the failure got duplicated. The 2026-05-23
        # audit flagged this. Gate on attempt count: only nuke when we're past
        # the max-recovery threshold.
        try:
            from app.services.utilities.cleanup_service import CleanupService
            from app.config import get_settings as _gs

            # Look up the job's recovery attempt count + threshold.
            # CORRECTION (2026-05-23 round-3): the Settings field is
            # `job_max_recovery_attempts` (not `auto_recovery_max_attempts`)
            # and the column is `recovery_attempts` (not `attempts`).
            # The earlier names were guessed wrong; getattr/select silently
            # returned defaults, defeating the env override.
            _max_attempts = int(getattr(_gs(), 'job_max_recovery_attempts', 3) or 3)
            _job_attempt = 0
            try:
                _supabase = get_supabase_client()
                _job_row = _supabase.client.table('background_jobs') \
                    .select('recovery_attempts, recovery_history') \
                    .eq('id', job_id).single().execute()
                if _job_row.data:
                    _job_attempt = int(_job_row.data.get('recovery_attempts') or 0)
                    # Some versions track attempts via recovery_history length;
                    # take the max so we never undershoot.
                    _rh = _job_row.data.get('recovery_history') or []
                    _job_attempt = max(_job_attempt, len(_rh) if isinstance(_rh, list) else 0)
            except Exception:
                pass

            if _job_attempt >= _max_attempts:
                cleanup_service = CleanupService()
                rollback_stats = await cleanup_service.rollback_discovered_products(
                    document_id=document_id,
                    product_db_ids=None,  # Delete all products for this document
                    supabase_client=None
                )
                logger.info(
                    f"🔄 Terminal failure (attempt {_job_attempt}/{_max_attempts}) — "
                    f"product rollback completed: {rollback_stats}"
                )
            else:
                logger.info(
                    f"⏭️  Resumable failure (attempt {_job_attempt}/{_max_attempts}) — "
                    f"preserving products for recovery"
                )
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

        # 4. Scale HuggingFace endpoints to zero (stop billing).
        # Single coordination point — manager-first, direct-HF fallback for
        # endpoints whose manager wasn't registered (partial warmup, etc).
        logger.info("📉 [CLEANUP] Scaling AI endpoints to zero...")
        try:
            from app.services.core.endpoint_controller import endpoint_controller
            outcome = await endpoint_controller.scale_all_to_zero(reason=f"pdf_job_{job_id}")
            scaled_count = sum(1 for ok in outcome.values() if ok)
        except Exception as scale_error:
            logger.warning(f"   ⚠️ scale_all_to_zero raised: {scale_error}")
            scaled_count = 0

        # 5. Clear endpoint registry (cleanup singleton state)
        try:
            endpoint_registry.clear_all()
            logger.info("   ✅ Endpoint registry cleared")
        except Exception as registry_error:
            logger.warning(f"   ⚠️ Failed to clear endpoint registry: {registry_error}")

        # 6. Final Garbage Collection
        gc.collect()
        logger.info(f"✨ [CLEANUP] Finished. Endpoints scaled to zero: {scaled_count}")

        # Stop heartbeat — final write inside __aexit__ updates last_heartbeat
        # so the recovery cron sees a fresh timestamp on natural completion.
        try:
            await heartbeat.__aexit__(None, None, None)
        except Exception as hb_err:
            logger.warning(f"Heartbeat shutdown raised: {hb_err}")


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
    claims: Dict[str, Any] = Depends(get_current_user),
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
        await authorize_rag_workspace(claims, request.workspace_id)

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
    rag_service: RAGService = Depends(get_rag_service),
    claims: Dict[str, Any] = Depends(get_current_user),
):
    """
    Conversational interface for document Q&A.

    This endpoint maintains conversation context and provides
    contextual responses based on the document collection.
    """
    start_time = datetime.utcnow()

    try:
        await authorize_rag_workspace(claims, request.workspace_id)

        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid4())

        # Build conversation context from the caller-supplied history. `include_history` is
        # now what actually gates it — it was declared and read nowhere, so callers could not
        # turn context off, and the hasattr guard below it tested a field that did not exist,
        # so context was never on either (#338).
        conversation_context = (
            request.conversation_history if request.include_history else None
        )

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

    except HTTPException:
        raise
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
                images_response = supabase_client.client.table('image_product_associations').select(
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
    rag_service: RAGService = Depends(get_rag_service),
    claims: Dict[str, Any] = Depends(get_current_user),
):
    """
    **🔍 SEARCH ENDPOINT - Multi-Vector Search with AI Query Understanding**

    ## 🎯 Supported Search Strategies

    ### Multi-Vector Search (`strategy="multi_vector"`) - ⭐ DEFAULT & RECOMMENDED ✅
    - 🎯 **ENHANCED**: 7-vector fusion search + JSONB metadata filtering
    - **Embeddings Combined:**
      - Text (15%) - Voyage AI 1024D semantic understanding
      - Visual (15%) - SLIG 768D visual similarity
      - Understanding (20%) - Voyage AI 1024D from Claude Opus 4.7 vision analysis
      - Color (12.5%) - SLIG 768D color palette matching
      - Texture (12.5%) - SLIG 768D texture pattern matching
      - Style (12.5%) - SLIG 768D design style matching
      - Material (12.5%) - SLIG 768D material type matching
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
    - Visual similarity using SLIG (SigLIP2) embeddings
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
    import time as _time
    start_time = datetime.utcnow()
    _t_total_start = _time.time()
    # Per-stage timing accumulator (used for observability tracking)
    stage_timings: Dict[str, int] = {}
    qu_was_cache_hit = False
    qu_was_product_name = False

    try:
        await authorize_rag_workspace(claims, request.workspace_id)

        # Validate strategy
        valid_strategies = ['multi_vector', 'material', 'image']
        if strategy not in valid_strategies:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid strategy '{strategy}'. Valid strategies: {', '.join(valid_strategies)}"
            )

        # Initialize services
        supabase_client = get_supabase_client()
        # The ids are passed HERE because the service cannot invent them, and every
        # Claude call it makes was previously logged with neither (audit #19 M6-6).
        search_prompt_service = SearchPromptService(
            supabase_client=supabase_client.client,
            workspace_id=request.workspace_id,
            user_id=current_user_id(claims),
        )
        ProductRelationshipService(supabase_client=supabase_client.client)

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
        unified_service = None
        _t_qu_start = _time.time()
        if enable_query_understanding:
            try:
                # asyncio is imported at module level (line 20). Do NOT re-import
                # locally — same UnboundLocalError trap as the discovery pipeline
                # function (see comment at the previous fix site). The earlier
                # `asyncio.create_task` / `asyncio.TimeoutError` references in
                # this function would all crash before this branch is reached.
                from app.services.search.unified_search_service import UnifiedSearchService

                # Create temporary service instance for query parsing
                unified_service = UnifiedSearchService()
                visual_query, parsed_filters, weight_profile, dynamic_weights = await asyncio.wait_for(
                    unified_service._parse_query_with_ai(query_to_use, workspace_id=request.workspace_id),
                    timeout=8  # 8s timeout — don't let query understanding block search
                )
                qu_was_cache_hit = getattr(unified_service, '_last_query_understanding_was_cache_hit', False)
                # If parser returned no filters and visual_query == original, it was a product-name search
                qu_was_product_name = (not parsed_filters) and (visual_query == query_to_use)

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
                logger.warning("Query understanding timed out after 8s, continuing with original query")
            except Exception as e:
                logger.error(f"Query understanding failed: {e}, continuing with original query")
                # Continue with original query if parsing fails
        stage_timings['query_understanding_ms'] = int((_time.time() - _t_qu_start) * 1000)

        # 🔍 STEP 2: Route to appropriate search method based on strategy
        _t_search_start = _time.time()

        # `aspect` is honored by exactly ONE branch below — multi_vector, which re-weights
        # the fusion toward that aspect's channel. `image` ranks on the SLIG visual vector
        # alone and `material` is JSONB filtering; neither has anywhere to apply it, and
        # both used to accept it and drop it on the floor, so an aspect-biased image search
        # returned plain visual similarity while looking like it had worked.
        #
        # Checked here rather than inside a branch so a strategy added later cannot quietly
        # inherit the silent drop — a new branch has to opt in by name (#277).
        _requested_aspect = getattr(request, 'aspect', None)
        if _requested_aspect and strategy != "multi_vector":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"strategy='{strategy}' cannot honor aspect='{_requested_aspect}'; only "
                    f"multi_vector can. For a single-aspect match use POST "
                    f"/api/search/by-{_requested_aspect}, which queries "
                    f"image_{_requested_aspect}_embeddings directly — pass query_image (the "
                    f"server runs vision_analysis on it) or query_text."
                ),
            )

        # All strategies now use the parsed query + extracted filters + dynamic weights
        if strategy == "multi_vector":
            # 🎯 Enhanced multi-vector search with dynamic weight profiles.
            # The 7-aspect → 9-source mapping lives in weight_profiles so this route
            # and /api/documents/query cannot drift apart (they held identical copies).
            from app.services.search.weight_profiles import (
                profile_to_source_weights,
                aspect_bias_weights,
            )
            rag_weights = profile_to_source_weights(dynamic_weights) if dynamic_weights else None
            material_filters = getattr(request, 'material_filters', None)
            # Build search config with optional MMR and weight overrides
            sc = {}
            if rag_weights:
                sc["weights"] = rag_weights
            if getattr(request, 'enable_mmr', False):
                sc["enable_mmr"] = True
                sc["mmr_lambda"] = getattr(request, 'mmr_lambda', 0.7)
            # Aspect bias (#277) — explicit user aspect wins over the balanced /
            # query-understanding weights: emphasize the chosen per-aspect vector.
            _aspect = _requested_aspect
            if _aspect in ('color', 'texture', 'style', 'material'):
                sc["weights"] = aspect_bias_weights(_aspect)
                # Named, not inferred. rag_service decides whether to spend a vision call on
                # image-derived query vectors, and "were weights supplied?" is the wrong
                # signal — query understanding sets weights on ordinary searches too, so
                # sniffing that would bill a Claude call for every text+image search.
                sc["aspect"] = _aspect
                weight_profile = f"aspect:{_aspect}"
                logger.info(
                    f"🎨 Aspect bias applied: {_aspect} "
                    f"(weight {sc['weights'][_aspect]:.3f} of a normalized 1.0)"
                )
            results = await rag_service.multi_vector_search(
                query=query_to_use,
                workspace_id=request.workspace_id,
                top_k=request.top_k,
                material_filters=material_filters,
                search_config=sc or None,
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
            # Image similarity search using visual embeddings (SLIG (SigLIP2))
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

        # Combined search time (vector + fulltext + scoring all happen inside rag_service)
        stage_timings['vector_search_ms'] = int((_time.time() - _t_search_start) * 1000)

        # Get raw results
        raw_results = results.get('results', [])

        # Apply formatting, filtering, enrichment prompts if enabled
        _t_enhancement_start = _time.time()
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

        # Honor include_content. It is declared on SearchRequest, documented as
        # "Include chunk content in results", and sent by unifiedSearchService on EVERY
        # search — and until now it was read nowhere, so a caller asking to drop the chunk
        # bodies got them anyway (#338). Dropped last so the content is still available to
        # the prompt/enrichment stages above, which legitimately need it to rank and
        # summarize; only the payload sent back to the caller loses it.
        if not request.include_content:
            for _r in enhanced_results:
                if isinstance(_r, dict):
                    _r.pop('content', None)

        stage_timings['enhancement_ms'] = int((_time.time() - _t_enhancement_start) * 1000)
        stage_timings['total_ms'] = int((_time.time() - _t_total_start) * 1000)

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # ── Track search query with full per-stage timings (fire-and-forget) ──
        try:
            from app.services.search.search_query_tracker import get_search_tracker
            tracker = get_search_tracker()
            asyncio.create_task(tracker.track_query(
                workspace_id=request.workspace_id,
                query_text=request.query,
                query_metadata=parsed_filters or None,
                search_type=strategy,
                result_count=len(enhanced_results),
                response_time_ms=stage_timings.get('total_ms', 0),
                weight_profile=weight_profile or "balanced",
                dynamic_weights=dynamic_weights,
                weight_profile_source="query_understanding" if enable_query_understanding else "default",
                stage_timings=stage_timings,
                cache_hit=qu_was_cache_hit,
                is_product_name_search=qu_was_product_name,
                strategy=strategy,
            ))
        except Exception as track_err:
            logger.debug(f"Search tracking failed (non-fatal): {track_err}")

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

@router.get("/documents/documents/{document_id}/content", responses={200: {"model": DocumentContentResponse}}, dependencies=[Depends(verify_internal_access)])
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
    - All images with AI analysis (SLIG embeddings + Claude Vision)
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

        # Count embeddings via the canonical has_*_slig boolean flags on document_images.
        # The actual embedding vectors live in vecs.image_*_embeddings collections.
        text_embeddings = sum(1 for chunk in result['chunks'] if chunk.get('embeddings'))
        slig_embeddings = sum(1 for img in result['images'] if img.get('has_slig_embedding'))
        understanding_embeddings = sum(1 for img in result['images'] if img.get('has_understanding_embedding'))
        color_embeddings = sum(1 for img in result['images'] if img.get('has_color_slig'))
        texture_embeddings = sum(1 for img in result['images'] if img.get('has_texture_slig'))
        style_embeddings = sum(1 for img in result['images'] if img.get('has_style_slig'))
        material_embeddings = sum(1 for img in result['images'] if img.get('has_material_slig'))
        vision_analysis = sum(1 for img in result['images'] if img.get('vision_analysis'))
        claude_validation = sum(1 for img in result['images'] if img.get('claude_validation'))

        result['statistics'] = {
            "chunks_count": chunks_count,
            "images_count": images_count,
            "products_count": products_count,
            "ai_usage": {
                "openai_calls": text_embeddings,
                "vision_calls": vision_analysis,
                "claude_calls": claude_validation,
                "slig_embeddings": slig_embeddings
            },
            "embeddings_generated": {
                "text": text_embeddings,
                "visual": slig_embeddings,
                "color": color_embeddings,
                "texture": texture_embeddings,
                "style": style_embeddings,
                "material": material_embeddings,
                "understanding": understanding_embeddings,
                "total": (text_embeddings + slig_embeddings + color_embeddings
                          + texture_embeddings + style_embeddings + material_embeddings
                          + understanding_embeddings)
            },
            "completion_rates": {
                "text_embeddings": f"{(text_embeddings / chunks_count * 100) if chunks_count > 0 else 0:.1f}%",
                "image_analysis": f"{(slig_embeddings / images_count * 100) if images_count > 0 else 0:.1f}%"
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
                # `services` is Dict[str, Dict[str, Any]]. This entry was the bare string
                # "Direct Vector DB", so pydantic rejected the response and the endpoint
                # answered 500 — while `overall_status` sitting right above it said
                # "healthy". A health check that reports itself unhealthy because it cannot
                # serialise its own healthy answer is worse than no health check: it was
                # firing from 2026-04-07 to 2026-08-15 and read as a real outage every time.
                # The vector store IS a service, so it gets a service entry rather than a
                # loose descriptor.
                "vector_store": {"status": "available", "type": "Direct Vector DB"},
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

@router.get("/stats", responses={200: {"model": StatsResponse}}, dependencies=[Depends(verify_internal_access)])
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
                "embeddings": "SLIG SigLIP2 768D / Voyage AI 1024D",
                "rag_synthesis": "Claude Opus 4.7",
                "vision": "claude-opus-4-8"
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

@router.get("/workspace-stats", responses={200: {"model": StatsResponse}})
async def get_workspace_statistics(
    workspace_id: str,
    supabase: SupabaseClient = Depends(get_supabase_client), current_user: dict = Depends(get_current_user)
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
    # Bind the caller-supplied workspace to the authenticated identity (invariant 1).
    workspace_id = await resolve_workspace_id(current_user, workspace_id)
    try:

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

        # Image embedding count via SQL function (bypasses VECS connection issues).
        # Cast workspace_id to UUID — count_vecs_embeddings is overloaded and the str cast picks the right variant.
        try:
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




@router.get("/job/{job_id}/ai-tracking", responses={200: {"model": AITrackingResponse}}, dependencies=[Depends(verify_internal_access)])
async def get_job_ai_tracking(job_id: str):
    """
    Get detailed AI model tracking information for a job.

    Returns comprehensive metrics on:
    - Which AI models were used (Anthropic Claude, SLIG, Voyage, OpenAI)
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


@router.get("/job/{job_id}/ai-tracking/stage/{stage}", responses={200: {"model": AITrackingResponse}}, dependencies=[Depends(verify_internal_access)])
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


@router.get("/job/{job_id}/ai-tracking/model/{model_name}", responses={200: {"model": AITrackingResponse}}, dependencies=[Depends(verify_internal_access)])
async def get_job_ai_tracking_by_model(job_id: str, model_name: str):
    """
    Get AI model tracking information for a specific AI model.

    Args:
        job_id: Job identifier
        model_name: AI model name (Anthropic Claude, SLIG, Voyage, OpenAI)

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

    # A `return AdvancedQueryResponse(...)` block used to sit here, after the `raise`
    # above and inside its `except` — unreachable, and referencing four names
    # (`start_time`, `request`, `results`, `datetime`) that do not exist in this
    # function at all. Paste residue from a different handler. Removed; it was four
    # of this file's F821s and it made the error path harder to read than it is.

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


@router.get("/admin/stuck-jobs/analyze/{job_id}", responses={200: {"model": StuckJobsResponse}}, dependencies=[Depends(verify_internal_access)])
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


@router.get("/admin/stuck-jobs/statistics", responses={200: {"model": StuckJobsResponse}}, dependencies=[Depends(verify_internal_access)])
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

# Per-hit ceiling on an expanded PDF chunk (hit + neighbours), in characters.
# ~1.5k tokens. Without it, `expand_neighbors=3` over top_k=10 could hand the
# LLM most of a catalog and call it a search result.
EXPANDED_CHUNK_CHAR_BUDGET = 6000

# Cosine floor for the Voyage-1024D text searches on this endpoint (chunks, entities).
#
# Deliberately NOT `request.similarity_threshold`, which defaults to 0.7: that knob was
# tuned for the word-count scorer the chunk branch replaced (+0.15 per matched word,
# 0..1). 0.7 is a different SCALE, not a stricter setting — as a cosine cutoff on
# Voyage vectors it rejects near enough every real hit, which would swap a
# wrong-results bug for a no-results one that looks exactly like an empty corpus.
#
# 0.4 is measured on kb_doc_chunks (a bull's-eye query lands ~0.50, unrelated text
# <=0.33) and BORROWED on document_chunks / document_entities, which are empty. Same
# model and the same text-vs-text comparison, so it is a reasonable transfer — but it
# is not a verified one, and a floor is the one parameter whose failure mode is
# invisible: too high and the endpoint returns nothing, which looks exactly like an
# empty corpus.
#
# So rather than guess again later, this is (a) overridable without a redeploy and
# (b) self-reporting. `search_metadata.similarity_floor` ships the candidate scores
# and, critically, the HIGHEST score the floor rejected. One real query then answers
# "is this value right?" — if the best rejected hit sits just under the floor, it is
# too high; if nothing is ever rejected, it is doing no work.
_SIMILARITY_FLOOR_DEFAULT = 0.4
_SIMILARITY_FLOOR_ENV = "MIVAA_TEXT_SEARCH_SIMILARITY_FLOOR"


def text_search_similarity_floor() -> tuple:
    """Return (floor, source). Env wins so it can be tuned against real data."""
    raw = os.getenv(_SIMILARITY_FLOOR_ENV)
    if raw:
        try:
            value = float(raw)
            if 0.0 <= value <= 1.0:
                return value, "env"
            logger.warning(
                f"{_SIMILARITY_FLOOR_ENV}={raw!r} is outside 0..1 — using the default"
            )
        except ValueError:
            logger.warning(f"{_SIMILARITY_FLOOR_ENV}={raw!r} is not a number — using the default")
    return _SIMILARITY_FLOOR_DEFAULT, "default"


def summarize_similarity_floor(scores: List[float], floor: float, source: str) -> Dict[str, Any]:
    """Describe what the floor actually did to this result set.

    `top_rejected` is the number that matters: the best hit the floor threw away.
    If it sits just below the floor, the floor is too high; if it is None, the floor
    did nothing and the ANN limit is the real constraint.
    """
    kept = [s for s in scores if s >= floor]
    rejected = [s for s in scores if s < floor]
    return {
        "value": round(floor, 4),
        "source": source,
        "candidates": len(scores),
        "kept": len(kept),
        "rejected": len(rejected),
        "best_score": round(max(scores), 4) if scores else None,
        "worst_kept": round(min(kept), 4) if kept else None,
        "top_rejected": round(max(rejected), 4) if rejected else None,
    }


class KnowledgeBaseSearchRequest(BaseModel):
    """Request model for knowledge base search.

    Every field is bounded (#29 M15-5). This route creates a Voyage vector per call and
    fans out across several search types, so an unbounded `top_k` or a 10,000-element
    filter list is paid work sized by the caller. Third instance of unbounded batch
    input on a paid path, after #23 M10-2 and the rechunk loop below.

    The bounds are generous on purpose — a cap that fires on real usage gets raised by
    the next person who hits it, and then it is not a cap.
    """
    query: str = Field(..., max_length=2_000, description="Search query")
    workspace_id: str = Field(..., description="Workspace ID to search within")
    search_types: List[str] = Field(
        default=["products", "entities", "chunks"],
        max_length=10,
        description="Types to search: products, entities, chunks, images, kb_docs"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        max_length=50,
        description="Filter by categories: product, certificate, logo, specification, general"
    )
    entity_types: Optional[List[str]] = Field(
        default=None,
        max_length=50,
        description="Filter by entity types: certificate, logo, specification"
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return per type")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    caller: str = Field(
        default="agent",
        description="Caller context: 'admin' (all levels), 'agent' (agent+public), 'public' (public only)"
    )
    agent_id: Optional[str] = Field(
        default=None,
        description=(
            "Identity of the querying agent (e.g. 'kai', 'interior-designer', "
            "'demo', or a background-agent type). When set on an agent caller, "
            "kb_docs with a non-empty allowed_agents list are returned only if "
            "this agent_id is in that list. Ignored for 'admin' callers."
        )
    )
    category_id: Optional[str] = Field(
        default=None,
        description="Restrict kb_docs search to a single category UUID"
    )
    category_slug: Optional[str] = Field(
        default=None,
        description="Restrict kb_docs search to a category by slug (e.g. 'pricing')"
    )
    price_doc_type: Optional[str] = Field(
        default=None,
        description="Restrict to pricing sub-type: price_list | discount_rule | contract_terms | promotion"
    )
    expand_neighbors: int = Field(
        default=1, ge=0, le=3,
        description=(
            "Structure expansion for PDF chunk hits: also return the +/-N chunks "
            "adjacent in reading order, so an answer that runs past a chunk boundary "
            "(a table split in two, a spec whose heading landed in the previous chunk) "
            "arrives whole. 0 disables it. Applies to `chunks` only — kb_docs are "
            "already section-sized and have /search/read-section for reading outward."
        )
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


class KBRechunkRequest(BaseModel):
    """Bounded (#29 M15-5). Rechunking embeds every chunk of every named doc, so
    `doc_ids` and `limit` size a Voyage bill directly and the route loops over the whole
    list. `all=True` is the intended way to process more than a page at a time — it
    exists precisely so a caller does not need an unbounded `doc_ids`."""
    doc_id: Optional[str] = Field(default=None, description="Rechunk a single kb_doc")
    doc_ids: Optional[List[str]] = Field(
        default=None, max_length=500, description="Rechunk an explicit set"
    )
    all: bool = Field(default=False, description="Backfill mode: page through all kb_docs")
    workspace_id: Optional[str] = Field(default=None, description="Restrict backfill to a workspace")
    limit: int = Field(default=25, ge=1, le=500, description="Max docs per call (backfill paging)")
    offset: int = Field(default=0, ge=0, description="Backfill paging offset")


@router.post("/kb-docs/rechunk")
async def kb_docs_rechunk(request: KBRechunkRequest, http_request: Request):
    """Chunk + embed kb_docs into kb_doc_chunks (section-level retrieval). Idempotent
    (delete+reinsert per doc). Internal/admin only — gated on x-cron-secret, since the
    /api/rag prefix is excluded from the JWT middleware. Called on-write (per doc) and by
    the one-time backfill (all=true, paged by limit/offset).

    Deliberately NOT using `resolve_workspace_id`: this route self-guards below on the
    cron secret or the service-role bearer and is not user-reachable, so `workspace_id`
    is a backfill filter from a trusted internal caller, not a tenancy claim. Adding
    `Depends(get_current_user)` here would 401 the cron path, which sends no bearer at
    all."""
    secret = (http_request.headers.get("x-cron-secret") or "").strip()
    expected = (os.getenv("CRON_SECRET") or "").strip()
    authz = (http_request.headers.get("authorization") or "").strip()
    svc = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY") or "").strip()
    # Trusted internal callers only: the x-cron-secret (backfill/cron) OR the service-role
    # bearer (the kb-generate-embedding edge fn already holds it). Not user-reachable.
    ok = (expected and secret == expected) or (svc and authz == f"Bearer {svc}")
    if not ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="x-cron-secret or service-role bearer required for kb-docs/rechunk")

    sb = get_supabase_client()
    from app.services.kb.kb_chunk_service import rechunk_doc

    ids: List[str] = []
    if request.doc_id:
        ids = [request.doc_id]
    elif request.doc_ids:
        ids = request.doc_ids
    elif request.all:
        q = sb.client.table("kb_docs").select("id").neq("content", "").order("id")
        if request.workspace_id:
            q = q.eq("workspace_id", request.workspace_id)
        rows = q.range(request.offset, request.offset + request.limit - 1).execute()
        ids = [r["id"] for r in (rows.data or [])]
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Provide doc_id, doc_ids, or all=true")

    results = []
    for did in ids:
        try:
            results.append(await rechunk_doc(sb, did))
        except Exception as e:  # per-doc isolation — one bad doc can't abort the batch
            logger.warning("rechunk failed for %s: %s", did, e)
            results.append({"doc_id": did, "error": str(e)[:200], "chunks": 0})

    return {
        "processed": len(results),
        "chunks_total": sum(r.get("chunks", 0) for r in results),
        "failed_embeds": sum(r.get("failed", 0) for r in results),
        "errors": [r for r in results if r.get("error")],
        "next_offset": (request.offset + request.limit) if request.all else None,
        "results": results,
    }


# KB access derivations live in app/services/kb/kb_access.py — ONE definition shared
# with /api/kb/search, which used to keep its own. See that module's docstring: the two
# endpoints are genuinely different (whole-doc vs section-level corpus, admin UI vs
# agent tool) but the query vector, the caller identity and the read scope must agree,
# and the query vector had already drifted.
from app.services.kb.kb_access import (  # noqa: E402
    kb_query_vector,
    resolve_kb_access_scope,
    resolve_kb_caller,
)


def _resolve_kb_access_scope(
    supabase: SupabaseClient,
    workspace_id: str,
    caller: str,
    query: str,
) -> Dict[str, Any]:
    """Thin binding of the shared resolver to THIS file's corpus.

    `per_doc_agent_gate=True` because both RPCs reached from here
    (`kb_match_doc_chunks`, `kb_read_doc_section`) enforce category access_level AND
    per-doc `allowed_agents` internally, which is what makes it correct for an agent to
    read a `visibility='private'` doc — private means "not published to the public KB
    website", not "hidden from agents". `/api/kb/search` runs `kb_match_docs`, which has
    no such gate, so it passes False. Keeping that argument explicit at each corpus is
    the point; there is no default that is right for both.
    """
    return resolve_kb_access_scope(
        supabase, workspace_id, caller, query, per_doc_agent_gate=True
    )


@router.post("/search/knowledge-base", response_model=KnowledgeBaseSearchResponse)
async def search_knowledge_base(
    request: KnowledgeBaseSearchRequest,
    rag_service: RAGService = Depends(get_rag_service),
    supabase: SupabaseClient = Depends(get_supabase_client),
    claims: Dict[str, Any] = Depends(get_current_user),
):
    """
    🔍 Search existing knowledge base without uploading a PDF.

    Uses the same **7-vector fusion search** as the main search endpoint, combining:
    - Text (15%) - Voyage AI 1024D semantic understanding
    - Visual (15%) - SLIG 768D visual similarity
    - Understanding (20%) - Voyage AI 1024D from Claude Opus 4.7 analysis
    - Color (12.5%) - SLIG 768D color palette matching
    - Texture (12.5%) - SLIG 768D texture pattern matching
    - Style (12.5%) - SLIG 768D design style matching
    - Material (12.5%) - SLIG 768D material type matching

    Performs unified semantic search across:
    - **Products** (with all metadata, embeddings, and material properties)
    - **Document entities** (certificates, logos, specifications)
    - **Chunks** (text content from PDFs with category tags)
    - **Images** (visual content with SLIG embeddings)

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
        await authorize_rag_workspace(claims, request.workspace_id)

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

        # Per-branch outcome, so a caller can tell "this corpus found nothing" from
        # "this corpus broke" (#29 M15-4). Each branch overwrites its own entry on
        # failure; anything still "ok" ran to completion. Bound HERE rather than in
        # the branches, because a branch that never ran must still report something.
        branch_status: Dict[str, str] = {
            "products": "ok",
            "entities": "ok",
            "chunks": "ok",
            "kb_docs": "ok",
        }

        # One query embedding, shared by every branch that needs it. `chunks`
        # (document_chunks.text_embedding) and `kb_docs` (kb_doc_chunks.embedding)
        # are both halfvec(1024) Voyage columns, so generating it per-branch would
        # be a second paid API call for the identical vector.
        _query_embedding_memo: Dict[str, Any] = {}
        # Reported in search_metadata. Expansion that silently never fires is the
        # platform's dominant failure shape, so the counters ship with the response
        # rather than living only in logs.
        chunk_expansion_stats: Dict[str, Any] = {
            "requested": 0, "hits": 0, "expanded_hits": 0, "neighbors_added": 0,
        }
        # Populated by the chunk / entity branches; shipped in search_metadata so the
        # floor can be judged from one real query instead of another investigation.
        chunk_floor_stats: Optional[Dict[str, Any]] = None
        entity_floor_stats: Optional[Dict[str, Any]] = None

        async def _query_embedding_1024() -> Optional[List[float]]:
            """Voyage 1024D embedding of the query. Memoized; returns None on failure
            so each branch can degrade rather than fail the whole search."""
            if "value" in _query_embedding_memo:
                return _query_embedding_memo["value"]
            # `kb_query_vector` owns the input_type decision — see kb_access. The
            # comment that used to live here explained why entity_type must be "query"
            # and not "search"; /api/kb/search had the same code with the wrong value,
            # which is what one derivation in one place prevents.
            value = await kb_query_vector(
                request.query,
                workspace_id=request.workspace_id,
                user_id=billing_user_id(claims),
            )
            _query_embedding_memo["value"] = value
            return value

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
                                # Only product-level embedding column post-cleanup is text_embedding_1024.
                                # Image-level visual embeddings live in vecs.image_*_embeddings.
                                "text": bool(result.get('text_embedding_1024')),
                                "understanding": bool(result.get('vision_analysis')),
                            }
                        })

                logger.info(f"   ✅ Found {len(results['products'])} products")

            except Exception as e:
                # A failed branch is not an empty one (#29 M15-4). The response used to
                # carry counts and no per-branch status, so an RPC error, an embedding
                # outage or schema drift was indistinguishable from "found nothing" —
                # and "nothing" is the answer nobody investigates.
                logger.error(f"Product search failed: {e}")
                branch_status["products"] = f"failed: {str(e)[:200]}"

        # Search entities (certificates / logos / specifications) by vector similarity.
        #
        # What this replaced: `await vecs_service.search_similar(collection_name=
        # "embeddings", ...)` — `vecs_service` was never instantiated in this function,
        # `search_similar` is not a method of VecsService (only `search_similar_images`
        # is), and no VECS collection named "embeddings" exists. So every call raised
        # NameError, the `except` below logged "Entity search failed", and
        # results["entities"] was ALWAYS []. The producer matched it: entity embeddings
        # were generated, counted and thrown away (see DocumentEntityService.
        # generate_entity_embeddings). Both halves reported success; neither worked.
        if "entities" in request.search_types:
            logger.info("   Searching entities...")
            try:
                query_embedding = await _query_embedding_1024()
                if not query_embedding:
                    logger.warning("   ⚠️ Entity search skipped — query embedding unavailable")
                else:
                    entity_rpc_args: Dict[str, Any] = {
                        "query_embedding": query_embedding,
                        "p_workspace_id": request.workspace_id,
                        "p_limit": request.top_k,
                        # Floor applied in Python for the same reason as the chunk
                        # branch — see text_search_similarity_floor().
                        "p_similarity_threshold": 0.0,
                    }
                    if request.entity_types:
                        # Filter in SQL, not after the fact: post-filtering an ANN page
                        # silently shrinks top_k to however many survived.
                        entity_rpc_args["p_entity_types"] = request.entity_types

                    entity_candidates = (
                        supabase.client
                        .rpc("search_document_entities_by_embedding", entity_rpc_args)
                        .execute().data
                    ) or []

                    _floor, _floor_src = text_search_similarity_floor()
                    _escores = [(r.get("similarity_score") or 0.0) for r in entity_candidates]
                    entity_floor_stats = summarize_similarity_floor(_escores, _floor, _floor_src)
                    entity_rows = [
                        r for r in entity_candidates
                        if (r.get("similarity_score") or 0.0) >= _floor
                    ]

                    for erow in entity_rows:
                        results["entities"].append({
                            "id": erow.get("entity_id"),
                            "entity_type": erow.get("entity_type"),
                            "name": erow.get("name"),
                            "description": erow.get("description"),
                            "content": erow.get("content"),
                            "source_document_id": erow.get("source_document_id"),
                            "page_range": erow.get("page_range"),
                            "metadata": erow.get("metadata") or {},
                            "relevance_score": erow.get("similarity_score") or 0.0,
                            "type": "entity",
                        })

                    logger.info(f"   ✅ Found {len(entity_rows)} entities")

            except Exception as e:
                # A failed branch is not an empty one (#29 M15-4). The response used to
                # carry counts and no per-branch status, so an RPC error, an embedding
                # outage or schema drift was indistinguishable from "found nothing" —
                # and "nothing" is the answer nobody investigates.
                logger.error(f"Entity search failed: {e}")
                branch_status["entities"] = f"failed: {str(e)[:200]}"

        # Search PDF-derived chunks (document_chunks) by vector similarity, then
        # expand each hit with its reading-order neighbours (issue #318).
        #
        # What this replaced: an UNORDERED `select * limit top_k*3` sample of the
        # workspace, scored at +0.15 per query word found as a substring against a
        # threshold defaulting to 0.7 — an arbitrary 30-row sample, a hit needing 5+
        # matching words to survive, content cut at 500 chars, and no document_id /
        # chunk_index in the result, so a retrieved chunk could not be read outward
        # from. text_embedding was never touched despite the hnsw index existing.
        if "chunks" in request.search_types:
            logger.info("   Searching chunks...")
            try:
                query_embedding = await _query_embedding_1024()
                if not query_embedding:
                    logger.warning("   ⚠️ Chunk search skipped — query embedding unavailable")
                else:
                    chunk_rpc_args: Dict[str, Any] = {
                        "query_embedding": query_embedding,
                        "p_workspace_id": request.workspace_id,
                        "p_limit": request.top_k,
                        # 0.0, and the floor is applied below in Python. The RPC could
                        # do it, but then the rejected scores never leave the database
                        # and the floor stays unmeasurable — which is how it came to be
                        # wrong in the first place. See text_search_similarity_floor().
                        "p_similarity_threshold": 0.0,
                    }
                    if request.categories:
                        chunk_rpc_args["p_categories"] = request.categories

                    chunk_candidates = (
                        supabase.client
                        .rpc("search_document_chunks_by_embedding", chunk_rpc_args)
                        .execute().data
                    ) or []

                    _floor, _floor_src = text_search_similarity_floor()
                    _scores = [(r.get("similarity_score") or 0.0) for r in chunk_candidates]
                    chunk_floor_stats = summarize_similarity_floor(_scores, _floor, _floor_src)
                    chunk_rows = [
                        r for r in chunk_candidates
                        if (r.get("similarity_score") or 0.0) >= _floor
                    ]
                    chunk_expansion_stats["hits"] = len(chunk_rows)

                    # Structure expansion, one round trip for the whole result page.
                    # Adjacency is resolved server-side inside each hit's own
                    # (document, product) namespace — chunk_index restarts at 0 per
                    # product, so a document-wide walk would interleave products.
                    expand_n = request.expand_neighbors
                    chunk_expansion_stats["requested"] = expand_n
                    neighbors_by_hit: Dict[str, List[Dict[str, Any]]] = {}
                    if expand_n and chunk_rows:
                        try:
                            expanded_rows = (
                                supabase.client.rpc("expand_document_chunk_hits", {
                                    "p_chunk_ids": [r["chunk_id"] for r in chunk_rows],
                                    "p_workspace_id": request.workspace_id,
                                    "p_before": expand_n,
                                    "p_after": expand_n,
                                }).execute().data
                            ) or []
                            for nrow in expanded_rows:
                                neighbors_by_hit.setdefault(
                                    nrow["source_chunk_id"], []
                                ).append(nrow)
                        except Exception as expand_err:
                            # Expansion is an enhancement: never let it cost the hit.
                            logger.warning(
                                f"   ⚠️ Chunk expansion failed, returning bare hits: {expand_err}"
                            )

                    # Total order even if an index is NULL. A plain
                    # `(idx is None, idx)` tuple key raises TypeError comparing
                    # None to None, which the outer handler would swallow into
                    # "Chunk search failed" — a whole search lost to a sort key.
                    def _reading_order(idx: Optional[int]) -> int:
                        return idx if idx is not None else 2_147_483_647

                    for row in chunk_rows:
                        matched = (row.get("content") or "").strip()
                        # (chunk_index, text, is_the_hit) — reassembled in reading order.
                        pieces: List[tuple] = [(row.get("chunk_index"), matched, True)]
                        budget = EXPANDED_CHUNK_CHAR_BUDGET - len(matched)
                        for nrow in sorted(
                            neighbors_by_hit.get(row["chunk_id"], []),
                            key=lambda n: _reading_order(n.get("chunk_index")),
                        ):
                            text = (nrow.get("content") or "").strip()
                            # Skip rather than break: a single oversized neighbour must
                            # not block the smaller one on its other side.
                            if not text or len(text) > budget:
                                continue
                            budget -= len(text)
                            pieces.append((nrow.get("chunk_index"), text, False))

                        pieces.sort(key=lambda p: _reading_order(p[0]))
                        added_indexes = [p[0] for p in pieces if not p[2]]
                        if added_indexes:
                            chunk_expansion_stats["expanded_hits"] += 1
                            chunk_expansion_stats["neighbors_added"] += len(added_indexes)

                        results["chunks"].append({
                            # `id` is the DOCUMENT id, not the chunk's — it is the
                            # address /search/read-section (source='pdf') reads from.
                            "id": row.get("document_id"),
                            "chunk_id": row.get("chunk_id"),
                            "chunk_index": row.get("chunk_index"),
                            "product_id": row.get("product_id"),
                            "page_number": row.get("page_number"),
                            # Full text, expanded — no 500-char cut.
                            "content": "\n\n".join(p[1] for p in pieces),
                            # The hit on its own, so a caller can tell what actually
                            # matched from what expansion pulled in around it.
                            "matched_content": matched if added_indexes else None,
                            "document_title": row.get("document_title"),
                            "product_name": row.get("product_name"),
                            "category": row.get("chunk_type"),
                            "metadata": {
                                "source": "document_chunks",
                                "page_number": row.get("page_number"),
                                "chunk_type": row.get("chunk_type"),
                                "region_types": row.get("region_types"),
                                "product_id": row.get("product_id"),
                                "expanded": bool(added_indexes),
                                "expanded_chunk_indexes": added_indexes,
                            },
                            "relevance_score": row.get("similarity_score") or 0.0,
                            "source": "pdf",
                            "type": "chunk",
                        })

                    logger.info(
                        f"   ✅ {len(chunk_rows)} PDF chunk hits, "
                        f"{chunk_expansion_stats['expanded_hits']} expanded "
                        f"(+{chunk_expansion_stats['neighbors_added']} neighbours)"
                    )

            except Exception as e:
                # A failed branch is not an empty one (#29 M15-4). The response used to
                # carry counts and no per-branch status, so an RPC error, an embedding
                # outage or schema drift was indistinguishable from "found nothing" —
                # and "nothing" is the answer nobody investigates.
                logger.error(f"Chunk search failed: {e}")
                branch_status["chunks"] = f"failed: {str(e)[:200]}"

        # Search KB docs (kb_docs table, filtered by category access_level + trigger_keyword)
        if "kb_docs" in request.search_types:
            logger.info("   📚 Searching KB docs...")
            try:
                # DERIVED, never taken from the body. `caller="admin"` grants admin
                # access levels + private docs, and PriceLookupDrawer sends exactly that
                # from the FRONTEND — through mivaa-gateway, which forwards the end
                # user's own JWT for /api/rag/* paths. So the assertion arrived on an
                # ordinary user token and was honoured unchecked. resolve_kb_caller
                # honours a platform service credential (price-tools.ts calls MIVAA
                # directly with MIVAA_API_KEY and asserts admin deliberately), lets any
                # caller NARROW, and clamps a widening request to 'agent'.
                caller = await resolve_kb_caller(
                    supabase, claims, request.caller, request.workspace_id
                )
                # Access gate (levels + trigger-keyword category allow-list + shared-KB
                # scope) is resolved by the SAME helper the read-section endpoint uses.
                # Cross-workspace exposure is additionally gated inside kb_match_doc_chunks.
                kb_scope = _resolve_kb_access_scope(
                    supabase, request.workspace_id, caller, request.query
                )
                allowed_access_levels = kb_scope["allowed_access_levels"]
                accessible_category_ids = kb_scope["accessible_category_ids"]

                # Shared with the `chunks` branch above — same vector, one API call.
                query_embedding = await _query_embedding_1024()

                if query_embedding:
                    rpc_args: Dict[str, Any] = {
                        "query_embedding": query_embedding,
                        "match_workspace_id": request.workspace_id,
                        # 0.4, not 0.5. A long KB doc (e.g. a 7k-char company
                        # overview) has ONE averaged embedding, so even a bull's-eye
                        # query ("Materials Hub") lands ~0.50 — right on a 0.5 cutoff,
                        # flickering in/out with float noise. Measured separation is
                        # clean: the true match sits ~0.50 while the next-best
                        # unrelated docs sit ≤0.33, so 0.4 admits the real hit without
                        # letting noise in. (Proper long-term fix: chunk long KB docs.)
                        "match_threshold": 0.4,
                        "match_count": request.top_k * 2,  # fetch extra, will post-filter
                        "allowed_access_levels": allowed_access_levels,
                        # 'private' visibility means "not published to the public KB
                        # website" — it is NOT an agent gate. Agent readability is
                        # governed by category access_level (agent/public) + per-doc
                        # allowed_agents, both applied above/below. So admin AND agent
                        # callers include private docs; only the public-website caller
                        # ('public') is restricted to visibility='public'. Without this,
                        # every "private but agent-allowed" doc was invisible to the agent.
                        "include_private": kb_scope["include_private"],
                    }
                    # Per-agent allow-list: only agent callers filter by identity.
                    # Admin/public callers pass no agent_id, so allowed_agents is ignored.
                    if caller != "admin" and request.agent_id:
                        rpc_args["match_agent_id"] = request.agent_id
                    if request.category_id:
                        rpc_args["match_category_id"] = request.category_id
                    if request.category_slug:
                        rpc_args["match_category_slug"] = request.category_slug
                    if request.price_doc_type:
                        rpc_args["match_price_doc_type"] = request.price_doc_type
                    # Cross-workspace shared KB: tenant callers (agent/admin — NOT the
                    # public-website caller) also pull the operator root workspace's
                    # published + non-private docs. The RPC forces published+non-private on
                    # the shared branch and no-ops when shared == caller, so an operator
                    # (root) caller never bleeds its own drafts/private through this path.
                    if kb_scope["shared_workspace_id"]:
                        rpc_args["shared_workspace_id"] = kb_scope["shared_workspace_id"]

                    # Section-level retrieval: match on kb_doc_chunks (each ~1.3k-char
                    # section), gated via the parent doc INSIDE the RPC (workspace,
                    # published, category access_level, visibility, per-doc allowed_agents
                    # via match_agent_id). So a big manual returns its most relevant
                    # SECTIONS in full instead of a truncated head, and matching is
                    # per-section rather than one weak whole-doc vector.
                    kb_response = supabase.client.rpc("kb_match_doc_chunks", rpc_args).execute()

                    if kb_response.data:
                        kb_count = 0
                        for ch in kb_response.data:
                            cat_id = ch.get("category_id")
                            # Trigger-keyword gate: agent-level categories only unlock when
                            # their keyword is in the query (public categories always in set).
                            if (
                                accessible_category_ids is not None
                                and cat_id is not None
                                and cat_id not in accessible_category_ids
                            ):
                                continue
                            heading = ch.get("heading")
                            results["chunks"].append({
                                "id": ch.get("kb_doc_id"),
                                "chunk_id": ch.get("chunk_id"),
                                "chunk_index": ch.get("chunk_index"),
                                "heading": heading,
                                # Full section content — no truncation (chunks are
                                # section-sized) — wrapped as DATA (invariant 9, #29
                                # M15-1). Every consumer of this endpoint hands the text
                                # to a model, and a KB document is a PERSISTENT injection
                                # primitive: written once, replayed into every future turn
                                # that retrieves it. The delimiter goes on here so no
                                # caller has to remember.
                                "content": as_untrusted_data(
                                    ch.get("content"),
                                    source=f"knowledge base: {ch.get('document_title') or 'untitled'}",
                                ),
                                "document_title": ch.get("document_title"),
                                "category": cat_id,
                                "category_slug": ch.get("category_slug"),
                                "category_name": ch.get("category_name"),
                                "price_doc_type": ch.get("price_doc_type"),
                                "metadata": {"source": "kb_doc_chunks", "visibility": ch.get("visibility"), "heading": heading},
                                "relevance_score": ch.get("similarity", 0.0),
                                # `source` tells the caller WHICH corpus this hit is
                                # addressed in, and therefore which read-section mode
                                # ('kb' vs 'pdf') can read outward from it. The two
                                # share this list but not their id spaces.
                                "source": "kb",
                                "type": "kb_doc",
                            })
                            kb_count += 1
                            if kb_count >= request.top_k:
                                break
                        logger.info(f"   ✅ Found {kb_count} KB doc sections after keyword filtering")

            except Exception as e:
                # A failed branch is not an empty one (#29 M15-4). The response used to
                # carry counts and no per-branch status, so an RPC error, an embedding
                # outage or schema drift was indistinguishable from "found nothing" —
                # and "nothing" is the answer nobody investigates.
                logger.error(f"KB docs search failed: {e}")
                branch_status["kb_docs"] = f"failed: {str(e)[:200]}"

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
                "similarity_threshold": request.similarity_threshold,
                # Did structure expansion actually fire, and how much did it add?
                # Shipped with the response so "expansion is on" can be verified
                # from a single call instead of inferred from log archaeology.
                "chunk_expansion": chunk_expansion_stats,
                # What the cosine floor did to each corpus. `top_rejected` vs
                # `worst_kept` is the whole diagnosis: adjacent values mean the floor
                # is cutting at the margin, a null top_rejected means it never fired.
                "similarity_floor": {
                    "chunks": chunk_floor_stats,
                    "entities": entity_floor_stats,
                },
                # "ok" per corpus, or the reason it failed. Without this, a total of 0
                # is the same response whether every branch ran clean or every branch
                # raised — which is the silent-zero shape at the top of the read path.
                "branch_status": branch_status,
                "degraded": any(v != "ok" for v in branch_status.values()),
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


class ReadSectionRequest(BaseModel):
    """Request model for contiguous section reading (locate-then-read)."""
    source: str = Field(
        default="kb",
        description=(
            "Which corpus the id addresses: 'kb' (kb_docs, authored articles) or "
            "'pdf' (document_chunks, text extracted from ingested PDFs). Defaults to "
            "'kb' so pre-existing callers are unaffected."
        )
    )
    kb_doc_id: Optional[str] = Field(
        default=None,
        description="kb_doc to read from (as returned by knowledge-base search). Required when source='kb'."
    )
    document_id: Optional[str] = Field(
        default=None,
        description="document to read from, as returned by knowledge-base search. Required when source='pdf'."
    )
    product_id: Optional[str] = Field(
        default=None,
        description=(
            "PDF only. chunk_index restarts at 0 for EACH product within a document, "
            "so the product scopes the index namespace. Pass the product_id the search "
            "hit carried; omitting it reads the document-level namespace (product_id IS "
            "NULL), which is a different set of chunks — not a wildcard."
        )
    )
    workspace_id: str = Field(..., description="Caller's workspace ID")
    from_chunk_index: int = Field(default=0, ge=0, description="First section index to read (inclusive)")
    to_chunk_index: Optional[int] = Field(
        default=None,
        description="Last section index to read (inclusive). Defaults to from_chunk_index + 3."
    )
    query: str = Field(
        default="",
        description=(
            "The user's original question. Required for parity with search: agent-level "
            "categories are trigger_keyword-gated against it, so passing an empty query "
            "can make a keyword-gated document unreadable."
        )
    )
    caller: str = Field(default="agent", description="Caller context: 'admin' | 'agent' | 'public'")
    agent_id: Optional[str] = Field(default=None, description="Querying agent identity (allowed_agents gate)")
    max_tokens: int = Field(
        default=6000, ge=200, le=20000,
        description="Token budget for the returned span. Over budget, the span is cut short and truncated=true."
    )


class ReadSectionResponse(BaseModel):
    """Response model for contiguous section reading."""
    source: str = "kb"
    kb_doc_id: Optional[str] = None
    document_id: Optional[str] = None
    product_id: Optional[str] = None
    document_title: Optional[str] = None
    chunks: List[Dict[str, Any]] = []
    doc_chunk_count: int = 0
    returned_chunk_indexes: List[int] = []
    token_total: int = 0
    truncated: bool = False
    outline: List[Dict[str, Any]] = []


@router.post("/search/read-section", response_model=ReadSectionResponse)
async def read_document_section(
    request: ReadSectionRequest,
    supabase: SupabaseClient = Depends(get_supabase_client),
    claims: Dict[str, Any] = Depends(get_current_user),
):
    """
    📖 Read a contiguous run of sections from ONE document, in order.

    The companion to `/search/knowledge-base`: search **locates** a section (it returns
    an id + `chunk_index` per hit), this **reads outward** from it. Answers that
    straddle a section boundary — a spec continued under the next heading, a table
    whose caption sits in the previous chunk — are otherwise unreachable except by
    guessing new keywords and hoping the missing part scores above threshold.

    Two corpora, selected by `source`:

    - **`kb`** (default) — authored `kb_docs`. Access is gated by
      `_resolve_kb_access_scope` + `kb_read_doc_section`, the SAME predicate
      `/search/knowledge-base` uses.
    - **`pdf`** — `document_chunks` extracted from ingested PDFs, via
      `read_document_chunk_span`. Scoped to the caller's workspace, and to the
      `(document, product)` index namespace because `chunk_index` restarts at 0 for
      each product inside a document.

    The caller supplies the id either way, so an object it cannot read returns **404**
    (not 403 — a 403 would confirm the id exists).

    Cheap by construction: pure SQL, no embedding and no LLM call.
    """
    try:
        await authorize_rag_workspace(claims, request.workspace_id)

        source = (request.source or "kb").strip().lower()
        if source not in ("kb", "pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="source must be 'kb' or 'pdf'",
            )

        # Derived, never body-supplied — same reasoning as the search route above.
        # This one matters more: the caller also supplies `kb_doc_id`, so a
        # self-declared admin here is a read-by-id of a document the gate would
        # otherwise have refused.
        caller = await resolve_kb_caller(
            supabase, claims, request.caller, request.workspace_id
        )
        from_idx = max(0, request.from_chunk_index)
        # Default span: the located section plus a little either side of it.
        to_idx = request.to_chunk_index if request.to_chunk_index is not None else from_idx + 3
        if to_idx < from_idx:
            to_idx = from_idx

        if source == "pdf":
            if not request.document_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="document_id is required when source='pdf'",
                )
            pdf_resp = supabase.client.rpc("read_document_chunk_span", {
                "p_document_id": _require_uuid(request.document_id, "document_id"),
                "match_workspace_id": request.workspace_id,
                "p_from_chunk_index": from_idx,
                "p_to_chunk_index": to_idx,
                "p_product_id": request.product_id,
            }).execute()
            # Normalize onto the kb row shape so the budget/outline logic below is
            # shared. PDF chunks carry no heading — the page is their human locator —
            # and no token_count column, so length/4 stands in for the budget.
            rows = [
                {
                    "chunk_id": r.get("chunk_id"),
                    "chunk_index": r.get("chunk_index"),
                    "heading": (
                        f"page {r.get('page_number')}"
                        if r.get("page_number") is not None else None
                    ),
                    "content": r.get("content") or "",
                    "token_count": max(1, len(r.get("content") or "") // 4),
                    "document_title": r.get("document_title") or r.get("product_name"),
                    "page_number": r.get("page_number"),
                    "category_id": None,  # no category gate on the PDF corpus
                    "doc_chunk_count": r.get("doc_chunk_count"),
                }
                for r in (pdf_resp.data or [])
            ]
        else:
            if not request.kb_doc_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="kb_doc_id is required when source='kb'",
                )
            scope = _resolve_kb_access_scope(supabase, request.workspace_id, caller, request.query)

            rpc_args: Dict[str, Any] = {
                "p_kb_doc_id": _require_uuid(request.kb_doc_id, "kb_doc_id"),
                "match_workspace_id": request.workspace_id,
                "p_from_chunk_index": from_idx,
                "p_to_chunk_index": to_idx,
                "allowed_access_levels": scope["allowed_access_levels"],
                "include_private": scope["include_private"],
            }
            if caller != "admin" and request.agent_id:
                rpc_args["match_agent_id"] = request.agent_id
            if scope["shared_workspace_id"]:
                rpc_args["shared_workspace_id"] = scope["shared_workspace_id"]

            resp = supabase.client.rpc("kb_read_doc_section", rpc_args).execute()
            rows = resp.data or []

        if not rows:
            # Either the doc is invisible to this caller, or the span is out of range.
            # Both answer 404 so the endpoint can't be used to probe for doc ids.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Section not found or not accessible",
            )

        # Trigger-keyword gate — the same post-filter search applies to its RPC rows.
        # kb only: the PDF corpus has no categories, and read_document_chunk_span
        # already scopes its rows to the caller's workspace.
        if source == "kb":
            accessible = scope["accessible_category_ids"]
            cat_id = rows[0].get("category_id")
            if accessible is not None and cat_id is not None and cat_id not in accessible:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Section not found or not accessible",
                )

        # Token budget. Sections are returned whole (a half-section is worse than a
        # missing one), so the budget cuts at a section boundary and reports it.
        budget = request.max_tokens
        spent = 0
        kept: List[Dict[str, Any]] = []
        truncated = False
        for row in rows:
            tokens = row.get("token_count") or 0
            if kept and spent + tokens > budget:
                truncated = True
                break
            kept.append({
                "chunk_id": row.get("chunk_id"),
                "chunk_index": row.get("chunk_index"),
                "heading": row.get("heading"),
                # Same corpus, larger spans — same treatment (#29 M15-1).
                "content": as_untrusted_data(
                    row.get("content"),
                    source=f"{source} section",
                ),
                "token_count": tokens,
                # PDF only; None on kb rows.
                "page_number": row.get("page_number"),
            })
            spent += tokens

        doc_chunk_count = int(rows[0].get("doc_chunk_count") or 0)

        # Outline of the whole requested span (including anything the budget cut), so the
        # agent can narrow its next call instead of blindly re-reading.
        outline = [
            {
                "chunk_index": r.get("chunk_index"),
                "heading": r.get("heading"),
                "token_count": r.get("token_count") or 0,
            }
            for r in rows
        ]

        logger.info(
            f"📖 read-section source={source} "
            f"doc={request.kb_doc_id or request.document_id} span=[{from_idx}..{to_idx}] "
            f"returned={len(kept)}/{len(rows)} tokens={spent} truncated={truncated}"
        )

        return ReadSectionResponse(
            source=source,
            kb_doc_id=request.kb_doc_id,
            document_id=request.document_id,
            product_id=request.product_id,
            document_title=rows[0].get("document_title"),
            chunks=kept,
            doc_chunk_count=doc_chunk_count,
            returned_chunk_indexes=[c["chunk_index"] for c in kept],
            token_total=spent,
            truncated=truncated,
            outline=outline,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Read section failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Read section failed: {str(e)}"
        )


# ============================================================================
# Deferred imports (break circular-import cycle — see Sentry MIVAA-5HQ)
# ============================================================================
# `app.api.documents` pulls in `management_routes` → `app.orchestration`, which
# re-exports `process_document_with_discovery` / `job_storage` from THIS module.
# Importing from `app.api.documents` at the top of the file would re-enter this
# module before those symbols exist. By the time this runs at end-of-module they
# are defined, so the cycle resolves. `authorize_rag_workspace` is only used
# inside request handlers (runtime), so a late binding is safe.
from app.api.documents.query_routes import authorize_rag_workspace  # noqa: E402
