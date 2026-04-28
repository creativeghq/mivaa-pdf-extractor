"""
Idealo module — admin-only HTTP routes.

Mounted at `/api/v1/modules/idealo/*` by the module registry
(see `app/modules/__init__.py::mount_module_routers`).

Endpoints:
  GET  /status   — locale support + 7-day usage stats
  POST /search   — admin test query against a single locale
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.modules import is_module_enabled
from app.modules.idealo.service import (
    MODULE_SLUG,
    _LOCALE_HOST,
    get_idealo_service,
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
    country_code: str = Field(..., min_length=2, max_length=2)
    limit: int = Field(5, ge=1, le=15)


class SearchResponse(BaseModel):
    hits: List[PriceHit]
    counts: Dict[str, int]
    elapsed_ms: int


# ── Router ─────────────────────────────────────────────────────────────────────


router = APIRouter(
    prefix="/api/v1/modules/idealo",
    tags=["modules:idealo"],
)


@router.get("/status", response_model=ModuleStatus)
async def module_status(request: Request) -> ModuleStatus:
    _admin_user_id_from_request(request)
    firecrawl = get_firecrawl_client()
    fc_ok = bool(firecrawl.api_key)

    sources: List[SourceStatus] = []
    seen_hosts: set[str] = set()
    for cc, host in _LOCALE_HOST.items():
        if host in seen_hosts:
            continue
        seen_hosts.add(host)
        sources.append(
            SourceStatus(
                key=host,
                name=host,
                configured=fc_ok,
                details=(
                    f"Firecrawl scrape of {host}/preisvergleich/MainSearchProductCategory.html "
                    f"(price-asc sort). Markets: {cc}."
                    if fc_ok
                    else "FIRECRAWL_API_KEY not set — adapter skips."
                ),
            )
        )

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
    Run the Idealo service for a single locale. Admin diagnostic only —
    does not write to competitor_sources or price_history. Firecrawl
    credits debit normally.
    """
    user_id = _admin_user_id_from_request(request)

    if not is_module_enabled(MODULE_SLUG):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Module '{MODULE_SLUG}' is disabled. Enable it at /admin/modules first.",
        )

    cc = payload.country_code.upper()
    if cc not in _LOCALE_HOST:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Idealo has no locale for country_code='{cc}'. "
                f"Supported: {sorted(_LOCALE_HOST.keys())}."
            ),
        )

    start = time.monotonic()
    service = get_idealo_service()
    hits = await service.search(
        query=payload.query,
        country_code=cc,
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
        logger.warning("idealo stats fetch failed: %s", exc)
        return ModuleStats()
