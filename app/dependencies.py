"""
Central dependency injection module for the Mivaa PDF Extractor service.

This module provides centralized dependency injection functions for:
- Service instances (Supabase, RAG Service, MaterialKai, PDF Processor)
- Authentication and authorization
- Request context and workspace validation

Security Note: Authentication dependencies enforce JWT validation and workspace isolation.
"""

from typing import Optional, Dict, Any
import os

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.auth.workspace_resolution import (
    WorkspaceAccessDenied,
    is_service_caller,  # noqa: F401  (re-exported for routes)
    resolve_workspace_id as _resolve_workspace_id,
)
from app.config import get_settings
from app.middleware.jwt_auth import JWTAuthMiddleware
from app.schemas.auth import WorkspaceContext
from app.services.core.supabase_client import get_supabase_client as _get_supabase_client
from app.services.search.rag_service import RAGService
from app.services.pdf.pdf_processor import PDFProcessor

# Initialize security scheme
security = HTTPBearer()

# Initialize settings
settings = get_settings()

# ============================================================================
# Service Dependencies (Using app.state for consistency)
# ============================================================================

def get_supabase_client():
    """
    Get Supabase client instance.

    Uses the global singleton instance initialized at startup.

    Returns:
        SupabaseClient instance

    Raises:
        HTTPException: If Supabase client is not initialized
    """
    client = _get_supabase_client()

    if not client.client:
        raise HTTPException(
            status_code=503,
            detail="Database service is not available. Please check configuration."
        )

    return client


async def get_material_kai_service(request: Request):
    """
    Get Material Kai service instance from app state.

    Args:
        request: FastAPI request object to access app.state

    Returns:
        MaterialKaiService instance

    Raises:
        HTTPException: If Material Kai service is not available
    """
    if hasattr(request.app.state, 'material_kai_service') and request.app.state.material_kai_service:
        return request.app.state.material_kai_service

    raise HTTPException(
        status_code=503,
        detail="Material Kai service is not available. Please check service configuration."
    )


async def get_rag_service(request: Request) -> RAGService:
    """
    Get RAG service instance from app state (lazy-loaded).

    The RAG service is initialized on first use through the component manager.
    This ensures efficient resource usage and proper lifecycle management.

    Args:
        request: FastAPI request object to access app.state

    Returns:
        RAGService instance

    Raises:
        HTTPException: If RAG service is not available
    """
    # Check if RAG service is already loaded in app state
    if hasattr(request.app.state, 'rag_service') and request.app.state.rag_service:
        service = request.app.state.rag_service
        if service.available:
            return service

    # Try to lazy load via component manager
    if hasattr(request.app.state, 'component_manager') and request.app.state.component_manager:
        try:
            service = await request.app.state.component_manager.get("rag_service")
            if service and service.available:
                request.app.state.rag_service = service
                return service
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to lazy load RAG service: {e}")

    raise HTTPException(
        status_code=503,
        detail="RAG service is not available. Please check service configuration."
    )


_pdf_processor_singleton: Optional[PDFProcessor] = None

def get_pdf_processor():
    """
    Get PDF processor singleton instance.

    Reuses the same PDFProcessor across requests to avoid re-creating
    the ThreadPoolExecutor and re-importing heavy CV2/scipy modules.

    Returns:
        PDFProcessor instance
    """
    global _pdf_processor_singleton
    if _pdf_processor_singleton is None:
        _pdf_processor_singleton = PDFProcessor()
    return _pdf_processor_singleton


# ============================================================================
# Authentication Dependencies
# ============================================================================

# Singleton JWT middleware — avoids re-instantiation + settings parse per request
_jwt_middleware_singleton: Optional[JWTAuthMiddleware] = None

def _get_jwt_middleware() -> JWTAuthMiddleware:
    global _jwt_middleware_singleton
    if _jwt_middleware_singleton is None:
        _jwt_middleware_singleton = JWTAuthMiddleware(None)
    return _jwt_middleware_singleton


async def verify_internal_access(request: Request) -> Optional[Dict[str, Any]]:
    """Auth gate for routes the JWT middleware cannot cover.

    Some prefixes are in ``JWTAuthMiddleware.exclude_paths`` because their real
    callers cannot present a Supabase *user* JWT -- edge functions calling with a
    service-role token, pg_cron calling with ``x-cron-secret``, the ``mk_``
    platform key. Excluding them from the middleware is correct; leaving them with
    no gate at all is not, and invariant 5 requires both.

    Accepts EITHER the shared ``x-cron-secret`` OR any token
    ``JWTAuthMiddleware._validate_token`` accepts (user JWT, service-role JWT, or
    the ``mk_`` platform key). Rejects anonymous callers, fail-closed.

    Returns the validated claims for the token path and ``None`` for the
    ``x-cron-secret`` path. Routes that scope by a body-supplied ``workspace_id``
    need that distinction: this gate admits ANY valid platform token, including an
    end user's, so without the claims a route cannot tell a trusted cron call from
    a user naming someone else's workspace. Declaring it as a parameter as well as
    a decorator dependency is free -- FastAPI caches a dependency per request.

    Originally ``internal_routes.verify_internal_access`` (pentest #250 D19/D20);
    promoted here by audit #12 so the 15 other excluded-and-ungated routes it
    found could use the same gate instead of growing a second copy.
    """
    secret = os.getenv("CRON_SECRET")
    if secret and request.headers.get("x-cron-secret") == secret:
        return None
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            claims = await _get_jwt_middleware()._validate_token(auth.split(" ", 1)[1])
            if claims:
                return claims
        except Exception:
            pass
    raise HTTPException(status_code=401, detail="Unauthorized: internal endpoint requires a valid token or x-cron-secret")


def current_user_id(user: Any) -> str:
    """The caller's user id, from whatever shape the auth dependency handed back.

    `get_current_user` returns the raw JWT CLAIMS DICT, whose id lives under `sub`. Three
    route files nevertheless annotated it `user: User` (the pydantic model in
    `middleware/jwt_auth`, which does have `.id`) and then wrote `str(user.id)` — 81 times.
    FastAPI does not coerce a `Depends()` parameter to its annotation, so the dict passed
    straight through and every one of those routes raised
    `AttributeError: 'dict' object has no attribute 'id'` on its first call. Sentry caught
    it on `/api/v1/price-monitoring/products/{product_id}`; the other 80 sites were the
    same bug waiting for traffic.

    Accepts the model form too, so this stays correct if the dependency is ever changed to
    return one. Raises 401 rather than returning None: a route that reached here has
    already authenticated, so an unresolvable id is a broken token, not an anonymous call —
    and returning None would push a `str(None)` = `"None"` user id into an ownership check.

    NOT for the metering path: `credit_metering._user_id` deliberately returns Optional
    because "no payer" is a legitimate state there (cron sweeps). Here it never is.
    """
    if isinstance(user, dict):
        uid = user.get("sub") or user.get("user_id")
    else:
        uid = getattr(user, "id", None) or getattr(user, "sub", None)
    if not uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated token carries no user id",
        )
    return str(uid)


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Extract and validate the current authenticated user from JWT token.

    Args:
        request: FastAPI request object
        credentials: HTTP Bearer token credentials

    Returns:
        Dict containing user information and claims

    Raises:
        HTTPException: If token is invalid or user is not authenticated
    """
    try:
        jwt_middleware = _get_jwt_middleware()

        # Validate token and extract claims
        claims = await jwt_middleware._validate_token(credentials.credentials)

        if not claims:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return claims

    except HTTPException:
        raise  # Re-raise auth exceptions directly without wrapping
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_workspace_context(
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user)
) -> WorkspaceContext:
    """
    Extract and validate workspace context from authenticated user.

    Args:
        request: FastAPI request object
        user: Authenticated user information (JWT claims)

    Returns:
        WorkspaceContext with validated workspace information

    Raises:
        HTTPException: If workspace context is invalid or missing
    """
    try:
        jwt_middleware = _get_jwt_middleware()

        # Extract workspace context
        workspace_context = await jwt_middleware._extract_workspace_context(user)

        if not workspace_context:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or missing workspace context"
            )

        return workspace_context

    except HTTPException:
        raise  # already a clean client-facing status (e.g. the 403 above)
    except Exception as e:
        # #250 J3: don't leak internal exception text (DB/JWT internals) to the
        # client — log it server-side, return a generic 403.
        import logging
        logging.getLogger(__name__).warning("Workspace validation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing workspace context",
        )


# ============================================================================
# Workspace resolution (BOLA — pentest #250 invariant 1)
# ============================================================================


async def resolve_workspace_id(
    claims: Dict[str, Any],
    requested_workspace_id: Optional[str],
    request: Optional[Request] = None,
) -> Optional[str]:
    """Bind a caller-supplied `workspace_id` to the authenticated identity.

    Thin wrapper over `app.auth.workspace_resolution.resolve_workspace_id` — that
    module holds the rule and the reasoning and is unit-tested; this supplies the two
    things it cannot import: the real membership check, and the workspace the
    middleware already validated for this request.

    Use it in any route that reads a `workspace_id` off the request::

        workspace_id = await resolve_workspace_id(claims, request.workspace_id, http_request)
    """
    jwt_middleware = _get_jwt_middleware()

    async def _is_member(user_id: str, workspace_id: str) -> bool:
        return await jwt_middleware._validate_workspace_access(user_id, workspace_id)

    try:
        return await _resolve_workspace_id(
            claims,
            requested_workspace_id,
            is_member=_is_member,
            validated_workspace_id=getattr(getattr(request, "state", None), "workspace_id", None),
        )
    except WorkspaceAccessDenied as denied:
        # The rule module raises a plain exception so it stays importable with pytest
        # alone; the HTTP mapping belongs here. Detail is deliberately bare — naming
        # the workspace back to a caller who does not belong to it is the leak.
        import logging
        logging.getLogger(__name__).warning(
            "Workspace access denied: user %s requested workspace %s",
            denied.user_id, denied.workspace_id,
        )
        raise HTTPException(status_code=denied.status_code, detail="Not found")


# ============================================================================
# Permission Dependencies
# ============================================================================

def require_permission(permission: str):
    """
    Create a dependency that requires a specific permission.
    
    Args:
        permission: Required permission string (e.g., 'pdf:read', 'pdf:write')
        
    Returns:
        Dependency function that validates the permission
    """
    async def permission_dependency(
        workspace_context: WorkspaceContext = Depends(get_workspace_context)
    ) -> WorkspaceContext:
        """Validate that the user has the required permission."""
        if not workspace_context.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        return workspace_context
    
    return permission_dependency


# ============================================================================
# Convenience Permission Dependencies
# ============================================================================

# PDF processing permissions
require_pdf_read = require_permission("pdf:read")
require_pdf_write = require_permission("pdf:write")
require_pdf_delete = require_permission("pdf:delete")

# Document management permissions
require_document_read = require_permission("document:read")
require_document_write = require_permission("document:write")
require_document_delete = require_permission("document:delete")

# Search permissions
require_search_read = require_permission("search:read")

# Image processing permissions
require_image_read = require_permission("image:read")
require_image_write = require_permission("image:write")

# Admin permissions
require_admin = require_permission("admin:all")


# ============================================================================
# Optional Authentication Dependencies
# ============================================================================

async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[Dict[str, Any]]:
    """
    Extract user information if authentication is provided, but don't require it.
    
    Args:
        request: FastAPI request object
        credentials: Optional HTTP Bearer token credentials
        
    Returns:
        User information if authenticated, None otherwise
    """
    if not credentials:
        return None
        
    try:
        return await get_current_user(request, credentials)
    except HTTPException:
        return None


async def get_optional_workspace_context(
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user)
) -> Optional[WorkspaceContext]:
    """
    Extract workspace context if user is authenticated, but don't require it.
    
    Args:
        request: FastAPI request object
        user: Optional authenticated user information
        
    Returns:
        WorkspaceContext if authenticated, None otherwise
    """
    if not user:
        return None
        
    try:
        return await get_workspace_context(request, user)
    except HTTPException:
        return None


# ────────────────────────────────────────────────────────────────────────────
# Authorization for the /api/internal surface (audit #13)
# ────────────────────────────────────────────────────────────────────────────
#
# `verify_internal_access` above proves AUTHENTICATION and says so in its own
# docstring: it admits any valid platform token, including an end user's. Seven
# `/api/internal` mutation routes used it as their only gate and then took the
# object id straight from the path, so any authenticated user could name another
# tenant's document, product or job and have MIVAA mutate it with service-role
# DB access and no RLS behind it (audit #13 MI-1).
#
# There are two correct answers, and which one applies depends on whether the
# route has a real user caller:
#
#   * no user caller  -> `require_trusted_service`, which admits ONLY the cron
#     secret, the `mk_` platform key or a service-role token.
#   * real user caller (the Admin UI sends the operator's own Supabase JWT
#     straight to MIVAA) -> keep the user in, and check that they own the id.
#     That is `assert_job_in_workspace` / `assert_document_in_workspace` /
#     `assert_product_in_workspace` below.
#
# Applying the first to a route that has a user caller breaks it; applying the
# second to a route that has none is busywork. The split is recorded per route.


def is_trusted_service_caller(claims: Optional[Dict[str, Any]]) -> bool:
    """True for the cron secret, the `mk_` platform key, or a service-role token.

    `verify_internal_access` returns None for the x-cron-secret path, which is by
    construction a trusted caller. Otherwise the token's own claims say what it is:
    the platform key stamps `service='mivaa'`, and a service-role JWT carries
    `role`/`aud` of `service_role`. An end user's Supabase JWT has neither.
    """
    if claims is None:
        return True
    return (
        claims.get("service") == "mivaa"
        or claims.get("role") == "service_role"
        or claims.get("aud") == "service_role"
    )


async def require_trusted_service(request: Request) -> Optional[Dict[str, Any]]:
    """Gate for internal routes that have NO user caller.

    Fails closed, and refuses an ordinary user JWT rather than admitting it the way
    `verify_internal_access` does. Use this only where a search across both repos
    shows no frontend/edge caller — otherwise the route needs an ownership check
    instead, or this will 403 a legitimate operator.
    """
    claims = await verify_internal_access(request)
    if not is_trusted_service_caller(claims):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is callable only by the platform itself",
        )
    return claims


def _caller_workspace_or_none(claims: Optional[Dict[str, Any]]) -> Optional[str]:
    """The workspace a NON-trusted caller is confined to, else None (no confinement).

    A trusted caller is not confined — a cron sweep legitimately spans tenants.
    """
    if is_trusted_service_caller(claims):
        return None
    ws = claims.get("workspace_id") if claims else None
    return str(ws) if ws else None


async def _assert_row_in_caller_workspace(
    *,
    table: str,
    column: str,
    row_id: str,
    claims: Optional[Dict[str, Any]],
    what: str,
) -> Dict[str, Any]:
    """Fetch `row_id` from `table` and confirm the caller may act on it.

    Returns the row. Raises 404 — never 403 — on a mismatch: telling an attacker
    "that exists but is not yours" is an id oracle (invariant 1).
    """
    sb = _get_supabase_client().client
    res = (
        sb.table(table)
        .select(f"id, {column}")
        .eq("id", row_id)
        .maybe_single()
        .execute()
    )
    row = getattr(res, "data", None)
    if not row:
        raise HTTPException(status_code=404, detail=f"{what} not found")

    caller_ws = _caller_workspace_or_none(claims)
    if caller_ws is None:
        return row

    owner_ws = row.get(column)
    if not owner_ws:
        # An unowned row is not "everyone's". Only a trusted caller may touch it —
        # `reset-job` could reset null-workspace jobs for any caller (audit #13).
        raise HTTPException(status_code=404, detail=f"{what} not found")
    if str(owner_ws) != str(caller_ws):
        raise HTTPException(status_code=404, detail=f"{what} not found")
    return row


async def assert_job_in_workspace(job_id: str, claims: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return await _assert_row_in_caller_workspace(
        table="background_jobs", column="workspace_id", row_id=job_id,
        claims=claims, what="Job",
    )


async def assert_document_in_workspace(document_id: str, claims: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return await _assert_row_in_caller_workspace(
        table="documents", column="workspace_id", row_id=document_id,
        claims=claims, what="Document",
    )


async def assert_product_in_workspace(product_id: str, claims: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return await _assert_row_in_caller_workspace(
        table="products", column="workspace_id", row_id=product_id,
        claims=claims, what="Product",
    )


async def assert_document_belongs_to(document_id: str, workspace_id: Optional[str]) -> None:
    """The document must live in the workspace the caller was already authorized for.

    `authorize_rag_workspace(claims, body.workspace_id)` proves the caller may act in
    the workspace they named. It says nothing about the OTHER id in the same request.
    Naming your own workspace and another tenant's `document_id` produced chunks,
    image rows, embeddings and relationships with cross-tenant references baked in
    (audit #13 MI-2) — the "two ids each individually valid, never checked against
    each other" class, and its sixth confirmed instance across the two repos.

    404, not 403: see _assert_row_in_caller_workspace.
    """
    if not document_id or not workspace_id:
        return
    sb = _get_supabase_client().client
    res = (
        sb.table("documents")
        .select("id, workspace_id")
        .eq("id", document_id)
        .maybe_single()
        .execute()
    )
    row = getattr(res, "data", None)
    if not row or str(row.get("workspace_id") or "") != str(workspace_id):
        raise HTTPException(status_code=404, detail="Document not found")


async def assert_job_belongs_to(job_id: Optional[str], workspace_id: Optional[str]) -> None:
    """The job must live in the workspace the caller was already authorized for.

    `regenerate-image-embeddings` scoped the IMAGES it processed to the caller's
    workspace correctly, then wrote `background_jobs` by id alone — so passing
    another tenant's job UUID marked their job processing/completed/failed with
    attacker-controlled progress metadata, which then fed their dashboards and any
    recovery logic keyed on status (audit #13 MI-3).
    """
    if not job_id or not workspace_id:
        return
    sb = _get_supabase_client().client
    res = (
        sb.table("background_jobs")
        .select("id, workspace_id")
        .eq("id", job_id)
        .maybe_single()
        .execute()
    )
    row = getattr(res, "data", None)
    if not row or str(row.get("workspace_id") or "") != str(workspace_id):
        raise HTTPException(status_code=404, detail="Job not found")


def caller_workspace_or_none(claims: Optional[Dict[str, Any]]) -> Optional[str]:
    """Public form of _caller_workspace_or_none, for routes that filter a query.

    Returns the workspace a non-trusted caller must be confined to, or None when
    the caller is trusted (cron / platform key / service role) and legitimately
    spans tenants.
    """
    return _caller_workspace_or_none(claims)


async def assert_job_readable(job_id: str, claims: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Alias of assert_job_in_workspace, named for read paths.

    `/api/jobs/{id}` and `/api/jobs/{id}/products` are not admin-only despite the
    path — they used `verify_internal_access` and no workspace filter, so any
    authenticated caller could read another tenant's filenames, document ids,
    errors and product-processing metadata (audit #13 MI-4).
    """
    return await assert_job_in_workspace(job_id, claims)
