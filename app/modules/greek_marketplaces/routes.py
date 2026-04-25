"""
Greek Marketplaces module — admin-only HTTP routes.

Mounted at `/api/v1/modules/greek-marketplaces/*` by the module registry
(see `app/modules/__init__.py::mount_module_routers`).

Endpoints:
  GET  /status   — source credentials + 7-day usage stats
  POST /search   — admin test query against all three adapters
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.modules import is_module_enabled
from app.modules.greek_marketplaces.adapters.skroutz import get_skroutz_adapter
from app.modules.greek_marketplaces.service import (
    MODULE_SLUG,
    get_greek_marketplaces_service,
)
from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.firecrawl_client import get_firecrawl_client
from app.services.integrations.perplexity_price_search_service import PriceHit

logger = logging.getLogger(__name__)


def _extract_bearer(request: Request) -> str:
    header = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if not header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )
    return header.split(" ", 1)[1].strip()


def _admin_user_id_from_request(request: Request) -> str:
    """
    Self-contained auth for module routes:
      * Pull the Supabase JWT off the Authorization header.
      * Ask Supabase (not our own middleware) to validate it — this works even
        when SUPABASE_JWT_SECRET isn't configured locally.
      * Verify the user has an admin / super_admin role in `user_profiles`.

    Returns the authenticated user's id. Raises 401 on invalid token,
    403 on non-admin users.
    """
    token = _extract_bearer(request)
    supabase = get_supabase_client().client
    try:
        response = supabase.auth.get_user(token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {exc}",
        )

    user = getattr(response, "user", None)
    if user is None or not getattr(user, "id", None):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not resolve authenticated user.",
        )
    user_id = str(user.id)

    if not _is_admin(supabase, user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required.",
        )
    return user_id


def _is_admin(sb, user_id: str) -> bool:
    """Same role check used by price_monitoring_routes — reads
    user_profiles.role_id → roles.name IN ('admin', 'super_admin').
    """
    try:
        res = (
            sb.table("user_profiles")
            .select("role_id")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        row = (res.data if res else None) or {}
        role_id = row.get("role_id")
        if not role_id:
            return False
        role = sb.table("roles").select("name").eq("id", role_id).maybe_single().execute()
        name = ((role.data if role else None) or {}).get("name")
        return name in ("admin", "super_admin")
    except Exception:  # noqa: BLE001
        return False


# ── Models ─────────────────────────────────────────────────────────────────────


class SourceStatus(BaseModel):
    key: str
    name: str
    configured: bool
    details: str


class ModuleStats(BaseModel):
    queries_7d: int = 0
    credits_7d: float = 0.0
    per_source_7d: Dict[str, int] = Field(default_factory=dict)


class ModuleStatus(BaseModel):
    slug: str = MODULE_SLUG
    enabled: bool
    sources: List[SourceStatus]
    stats: ModuleStats


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    country_code: str = Field("GR", min_length=2, max_length=2)
    limit: int = Field(15, ge=1, le=25)


class SearchResponse(BaseModel):
    hits: List[PriceHit]
    counts: Dict[str, int]
    elapsed_ms: int


# ── Router ─────────────────────────────────────────────────────────────────────


router = APIRouter(
    prefix="/api/v1/modules/greek-marketplaces",
    tags=["modules:greek-marketplaces"],
)


@router.get("/status", response_model=ModuleStatus)
async def module_status(request: Request) -> ModuleStatus:
    """Return source-credential state + 7-day usage stats for this module."""
    _admin_user_id_from_request(request)
    skroutz = get_skroutz_adapter()
    firecrawl = get_firecrawl_client()

    # Lazy import keeps the routes module light and avoids surprising
    # the module loader on first import.
    from app.modules.greek_marketplaces.adapters.shopflix import ENABLED as shopflix_enabled

    sources = [
        SourceStatus(
            key="skroutz",
            name="Skroutz",
            configured=skroutz.is_configured,
            details=(
                "Firecrawl scrape of skroutz.gr/search?keyphrase=… (price-asc sort) — "
                "aggregator URL (merchants one click away)."
                if skroutz.is_configured
                else "FIRECRAWL_API_KEY not set — adapter skips."
            ),
        ),
        SourceStatus(
            key="bestprice",
            name="Bestprice.gr",
            configured=bool(firecrawl.api_key),
            details=(
                "Firecrawl scrape of bestprice.gr/search?q=…&o=2 (price-asc sort)."
                if firecrawl.api_key
                else "FIRECRAWL_API_KEY not set — adapter skips."
            ),
        ),
        SourceStatus(
            key="shopflix",
            name="Shopflix.gr",
            configured=shopflix_enabled and bool(firecrawl.api_key),
            details=(
                "Firecrawl scrape of shopflix.gr search."
                if (shopflix_enabled and firecrawl.api_key)
                else "Disabled — search URL pattern unconfirmed. Set ENABLED=True in shopflix.py once the URL is verified."
            ),
        ),
    ]

    stats = await _fetch_module_stats(MODULE_SLUG)

    return ModuleStatus(
        enabled=is_module_enabled(MODULE_SLUG),
        sources=sources,
        stats=stats,
    )


@router.post("/search", response_model=SearchResponse)
async def test_search(
    payload: SearchRequest,
    request: Request,
) -> SearchResponse:
    """
    Run the Greek Marketplaces service directly against all three adapters.
    Intended as an admin diagnostic tool — does not write to competitor_sources
    or price_history. Credit debits still fire (Skroutz is free; Firecrawl calls
    consume credits per the normal pricing).
    """
    user_id = _admin_user_id_from_request(request)

    if not is_module_enabled(MODULE_SLUG):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Module '{MODULE_SLUG}' is disabled. Enable it at /admin/modules first.",
        )

    if payload.country_code.upper() != "GR":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Greek Marketplaces only supports country_code='GR'.",
        )

    start = time.monotonic()
    service = get_greek_marketplaces_service()
    hits = await service.search(
        query=payload.query,
        country_code=payload.country_code,
        user_id=user_id,
        workspace_id=None,
        limit=payload.limit,
    )

    counts: Dict[str, int] = {}
    for hit in hits:
        counts[hit.source] = counts.get(hit.source, 0) + 1

    return SearchResponse(
        hits=hits,
        counts=counts,
        elapsed_ms=int((time.monotonic() - start) * 1000),
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


async def _fetch_module_stats(slug: str) -> ModuleStats:
    """Read ai_usage_logs to build 7-day usage stats for this module."""
    try:
        supabase = get_supabase_client().client
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        response = (
            supabase.table("ai_usage_logs")
            .select("credits_debited, metadata", count="exact")
            .eq("module_slug", slug)
            .gte("created_at", cutoff)
            .execute()
        )
        rows = response.data or []
        total_credits = 0.0
        per_source: Dict[str, int] = {}
        for row in rows:
            total_credits += float(row.get("credits_debited") or 0)
            metadata = row.get("metadata") or {}
            if not isinstance(metadata, dict):
                continue
            # Adapter identity is pushed into the Firecrawl `request_data` at
            # scrape time, then mirrored into `metadata.request.source` by
            # AICallLogger.log_firecrawl_call. Fall back to api_provider so
            # legacy rows still aggregate as "firecrawl".
            source = None
            req = metadata.get("request")
            if isinstance(req, dict):
                source = req.get("source")
            if not source:
                source = metadata.get("source") or metadata.get("api_provider")
            if source:
                per_source[source] = per_source.get(source, 0) + 1

        return ModuleStats(
            queries_7d=response.count or 0,
            credits_7d=round(total_credits, 2),
            per_source_7d=per_source,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("greek-marketplaces stats fetch failed: %s", exc)
        return ModuleStats()
