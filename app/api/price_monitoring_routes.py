"""
Price Monitoring API Routes — internal product flow.

Thin wrapper over `tracked_queries_service`. Every internal product that gets
enrolled becomes a `tracked_queries` row with `api_key_id IS NULL` and
`product_id NOT NULL`. The legacy `competitor_sources` / `price_history` /
`price_monitoring_products` tables were dropped 2026-05-01; everything routes
through `tracked_query_price_history` now.

Endpoint surface:
  POST   /products/{product_id}/track          — get-or-create + refresh
  DELETE /products/{product_id}/track          — deactivate (soft delete)
  GET    /products/{product_id}                — read tracked_query summary
  POST   /products/{product_id}/refresh        — force refresh
  GET    /products/{product_id}/sources        — latest retailer rows (split)
  GET    /products/{product_id}/history        — historical price rows
  POST   /products/{product_id}/exclude        — exclude URL/domain
  POST   /products/{product_id}/include        — undo exclusion
  GET    /products/{product_id}/exclusions     — list exclusions
  POST   /products/{product_id}/verify         — re-verify URLs (Firecrawl)
  POST   /products/{product_id}/url-only       — add pinned URL (custom monitoring)
  GET    /products/{product_id}/url-only       — list pinned URLs

Cross-flow endpoints (also serve external API consumers):
  POST   /market-check                         — stateless market scan
  POST   /classifier-correction                — feed few-shot classifier loop
  POST   /promote-family-row                   — admin override (sticky)
  POST   /demote-to-family                     — undo promotion
  POST   /tracked-queries/cron-refresh         — cron-target batch refresh
  POST   /broadcast-api-announcement           — admin email broadcast

Legacy aliases kept for short-term frontend compatibility:
  /start, /stop, /check-now, /discover, /status/{product_id},
  /history/{product_id}, /sources/{product_id}
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx as _httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from app.dependencies import get_current_user, get_workspace_context
from app.middleware.jwt_auth import User, WorkspaceContext
from app.schemas.api_responses import (
    DataResponse,
    MonitoringActionResponse,
    PriceHistoryResponse,
    PriceSourceResponse,
    StatusResponse,
)
from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.perplexity_price_search_service import (
    PriceHit,
    get_perplexity_price_search_service,
)
from app.services.integrations.product_identity_service import (
    facets_from_catalog,
    normalize_model_token,
)
from app.services.integrations.tracked_queries_service import get_tracked_queries_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/price-monitoring",
    tags=["Price Monitoring"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not found"},
    },
)


# ============================================================================
# Helpers — country resolution + admin role check
# ============================================================================

_COUNTRY_NAME_TO_CODE = {
    "greece": "GR", "ελλάδα": "GR", "ellada": "GR",
    "germany": "DE", "deutschland": "DE",
    "united kingdom": "GB", "great britain": "GB", "uk": "GB", "england": "GB",
    "united states": "US", "usa": "US", "us": "US", "america": "US",
    "italy": "IT", "italia": "IT",
    "spain": "ES", "españa": "ES", "espana": "ES",
    "france": "FR",
    "netherlands": "NL", "holland": "NL",
    "belgium": "BE",
    "austria": "AT", "switzerland": "CH",
    "portugal": "PT",
    "ireland": "IE",
    "bulgaria": "BG", "romania": "RO", "cyprus": "CY",
    "poland": "PL", "czechia": "CZ", "czech republic": "CZ",
    "slovakia": "SK", "hungary": "HU",
    "lithuania": "LT", "latvia": "LV", "estonia": "EE",
    "sweden": "SE", "denmark": "DK", "norway": "NO", "finland": "FI",
    "turkey": "TR", "türkiye": "TR",
    "canada": "CA", "australia": "AU",
}

DEFAULT_COUNTRY_CODE = "GR"


def _resolve_user_country_code(profile: Optional[Dict[str, Any]]) -> str:
    prof = profile or {}
    code = (prof.get("location_country_code") or "").strip().upper()
    if len(code) == 2 and code.isalpha():
        return code

    location_text = (prof.get("location") or "").strip()
    if location_text:
        import re as _re
        trailing_iso = _re.search(r',\s*([A-Za-z]{2})\s*$', location_text)
        if trailing_iso:
            guess = trailing_iso.group(1).upper()
            if guess.isalpha():
                return guess

        lower = location_text.lower()
        for name, iso in _COUNTRY_NAME_TO_CODE.items():
            if name in lower:
                return iso

    return DEFAULT_COUNTRY_CODE


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
        rn = ((role.data if role else None) or {}).get("name")
        return rn in ("admin", "super_admin")
    except Exception:
        return False


def _require_admin(user: User) -> None:
    if not _is_admin(get_supabase_client().client, str(user.id)):
        raise HTTPException(status_code=403, detail="admin role required")


def _resolve_product_context(sb, product_id: str) -> Dict[str, Any]:
    """Fetch product name/manufacturer/dimensions for find_or_create."""
    prod = (
        sb.table("products")
        .select("id, name, metadata")
        .eq("id", product_id)
        .maybe_single()
        .execute()
    )
    row = (prod.data if prod else None) or None
    if not row:
        raise HTTPException(status_code=404, detail=f"product {product_id} not found")
    metadata = row.get("metadata") or {}
    raw_dims = metadata.get("dimensions") or metadata.get("size") or metadata.get("product_size")
    dims: Optional[str] = None
    if isinstance(raw_dims, list) and raw_dims:
        first = raw_dims[0]
        dims = (first.get("metric_cm") or first.get("imperial_in")) if isinstance(first, dict) else str(first)
    elif isinstance(raw_dims, dict):
        dims = raw_dims.get("metric_cm") or raw_dims.get("imperial_in")
    elif isinstance(raw_dims, str):
        dims = raw_dims
    return {
        "name": row.get("name") or "",
        "manufacturer": (
            metadata.get("factory_name")
            or metadata.get("manufacturer")
            or metadata.get("brand")
        ),
        "dimensions": dims,
    }


def _normalize_domain(domain: Optional[str]) -> Optional[str]:
    if not domain:
        return None
    d = domain.strip().lower()
    d = d.removeprefix("http://").removeprefix("https://").removeprefix("www.")
    return d.split("/")[0] or None


# ============================================================================
# Request / response models
# ============================================================================


class TrackProductRequest(BaseModel):
    """POST /products/{product_id}/track — enroll a catalog product into monitoring."""
    country_code: Optional[str] = Field(default=None, description="Override country (defaults to user profile location).")
    force_refresh: bool = Field(default=False, description="Re-run discovery now even if a row already exists.")


class RefreshProductRequest(BaseModel):
    force_refresh: bool = Field(default=False, description="Bypass the volatility cadence (admin only).")
    verify_prices: bool = Field(default=True, description="Run the Firecrawl verification pass.")


class ProductExcludeRequest(BaseModel):
    url: Optional[str] = Field(default=None, description="Specific competitor URL.")
    domain: Optional[str] = Field(default=None, description="Whole retailer domain.")
    reason: Optional[str] = Field(default=None, description="Audit note.")


class ProductExclusionRow(BaseModel):
    id: str
    url: Optional[str] = None
    domain: Optional[str] = None
    reason: Optional[str] = None
    excluded_at: str


class ProductVerifyRequest(BaseModel):
    urls: Optional[List[str]] = Field(
        default=None,
        description="Optional whitelist of URLs to re-verify. Each must already exist in the latest refresh run.",
    )


class AddUrlOnlyRequest(BaseModel):
    url: str = Field(..., min_length=1)
    country_code: Optional[str] = None


class MarketCheckRequest(BaseModel):
    product_id: Optional[str] = Field(default=None)
    product_name: Optional[str] = Field(default=None)
    dimensions: Optional[str] = Field(default=None)
    manufacturer: Optional[str] = Field(default=None)
    verify_prices: bool = Field(default=True)


class MarketStats(BaseModel):
    count: int
    verified_count: int
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    currency: Optional[str] = None


class MarketCheckResponse(BaseModel):
    success: bool
    product_id: Optional[str] = None
    query: str
    country_code: str
    results: List[PriceHit] = []
    total_results: int = 0
    stats: MarketStats
    summary: Optional[str] = None
    credits_used: int = 0
    latency_ms: int = 0
    from_monitoring_cache: bool = False
    cache_age_seconds: Optional[int] = None
    error: Optional[str] = None


class MatchCorrectionRequest(BaseModel):
    tracked_query_history_id: Optional[str] = None
    corrected_match_kind: str = Field(..., pattern="^(exact|variant|family|mismatch|unverifiable|should_drop)$")
    correction_note: Optional[str] = None


class PromoteFamilyRequest(BaseModel):
    tracked_query_history_id: str = Field(..., description="ID from tracked_query_price_history.")
    override_kind: str = Field(..., pattern="^(exact|variant)$")
    reason: Optional[str] = None


class DemoteFamilyRequest(BaseModel):
    tracked_query_history_id: str
    reason: Optional[str] = None


class BroadcastApiAnnouncementRequest(BaseModel):
    template_slug: str = Field(default="api_broadcast.price_tracking_v6")
    docs_url: str = Field(default="https://github.com/creativeghq/material-kai-vision-platform/blob/main/docs/api/price-monitoring-api.md")
    api_base_url: str = Field(default="https://v1api.materialshub.gr")
    support_email: str = Field(default="support@materialshub.gr")
    dry_run: bool = Field(default=True)


# ============================================================================
# Internal product enrollment + lifecycle
# ============================================================================


@router.post(
    "/products/{product_id}/track",
    response_model=DataResponse,
    summary="Enroll a catalog product into monitoring (or re-run discovery if already enrolled)",
)
async def track_product(
    product_id: str,
    body: Optional[TrackProductRequest] = None,
    user: User = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    body = body or TrackProductRequest()
    sb = get_supabase_client().client
    ctx = _resolve_product_context(sb, product_id)

    profile = (
        sb.table("user_profiles")
        .select("location_country_code, location")
        .eq("user_id", user.id)
        .maybe_single()
        .execute()
    )
    country = body.country_code or _resolve_user_country_code((profile.data if profile else None) or {})

    service = get_tracked_queries_service()
    tq = await service.find_or_create_for_product(
        product_id=product_id,
        product_name=ctx["name"],
        manufacturer=ctx["manufacturer"],
        dimensions=ctx["dimensions"],
        country_code=country,
        user_id=str(user.id),
        workspace_id=workspace.workspace_id if workspace else None,
        run_first_refresh=True,
        force=body.force_refresh,
    )
    return {"success": True, "data": {"product_id": product_id, "tracked_query_id": tq.get("id"), "tracked_query": tq}}


@router.delete(
    "/products/{product_id}/track",
    response_model=StatusResponse,
    summary="Stop monitoring this product (soft delete — history is kept).",
)
async def untrack_product(
    product_id: str,
    user: User = Depends(get_current_user),
):
    service = get_tracked_queries_service()
    tq = await service.find_for_product(product_id)
    if not tq:
        raise HTTPException(status_code=404, detail="product is not enrolled in monitoring")
    ok = await service.deactivate(tq["id"])
    return {"success": ok, "message": "monitoring stopped" if ok else "failed to stop"}


@router.get(
    "/products/{product_id}",
    response_model=DataResponse,
    summary="Read the tracked_query summary row for this product.",
)
async def get_product_monitoring(
    product_id: str,
    user: User = Depends(get_current_user),
):
    service = get_tracked_queries_service()
    tq = await service.find_for_product(product_id)
    return {"success": True, "data": tq}


@router.post(
    "/products/{product_id}/refresh",
    response_model=DataResponse,
    summary="Re-run discovery for this product (uses cached query_facets, applies cost optimizations).",
)
async def refresh_product(
    product_id: str,
    body: Optional[RefreshProductRequest] = None,
    user: User = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    body = body or RefreshProductRequest()
    service = get_tracked_queries_service()
    tq = await service.find_for_product(product_id)
    if not tq:
        # Auto-enroll on first call so the UI's Refresh button works without a
        # separate /track click.
        sb = get_supabase_client().client
        ctx = _resolve_product_context(sb, product_id)
        profile = (
            sb.table("user_profiles")
            .select("location_country_code, location")
            .eq("user_id", user.id)
            .maybe_single()
            .execute()
        )
        country = _resolve_user_country_code((profile.data if profile else None) or {})
        tq = await service.find_or_create_for_product(
            product_id=product_id,
            product_name=ctx["name"],
            manufacturer=ctx["manufacturer"],
            dimensions=ctx["dimensions"],
            country_code=country,
            user_id=str(user.id),
            workspace_id=workspace.workspace_id if workspace else None,
            run_first_refresh=True,
            force=True,
        )
        return {"success": True, "data": tq}

    # Force refresh requires admin (matches the old /discover semantics).
    if body.force_refresh:
        _require_admin(user)
    # Toggle verify_prices on the row in case the caller wants to flip it.
    if body.verify_prices != bool(tq.get("verify_prices", True)):
        await service.update(tq["id"], verify_prices=body.verify_prices)

    outcome = await service.refresh(tq["id"], force=body.force_refresh)
    return {"success": outcome.get("status") == "refreshed", "data": outcome}


@router.get(
    "/products/{product_id}/sources",
    response_model=DataResponse,
    summary="Latest retailer rows for this product, split into primary vs family.",
)
async def get_product_sources(
    product_id: str,
    user: User = Depends(get_current_user),
):
    service = get_tracked_queries_service()
    tq = await service.find_for_product(product_id)
    if not tq:
        return {"success": True, "data": {"results": [], "family_results": []}}
    split = await service.latest_results_split(tq["id"])
    return {"success": True, "data": {**split, "tracked_query_id": tq["id"]}}


@router.get(
    "/products/{product_id}/history",
    response_model=PriceHistoryResponse,
    summary="Historical price rows for this product, newest first.",
)
async def get_product_history(
    product_id: str,
    limit: int = Query(default=200, ge=1, le=2000),
    user: User = Depends(get_current_user),
):
    service = get_tracked_queries_service()
    tq = await service.find_for_product(product_id)
    if not tq:
        return {"success": True, "history": [], "count": 0}
    rows = await service.history(tq["id"], limit=limit)
    return {"success": True, "history": rows, "count": len(rows)}


# ============================================================================
# Per-product result exclusions
# ============================================================================


@router.post(
    "/products/{product_id}/exclude",
    response_model=ProductExclusionRow,
    summary="Exclude a retailer URL or domain from this product's monitoring.",
    status_code=status.HTTP_201_CREATED,
)
async def exclude_product_result(
    product_id: str,
    body: ProductExcludeRequest,
    user: User = Depends(get_current_user),
):
    if not body.url and not body.domain:
        raise HTTPException(status_code=400, detail="Either `url` or `domain` is required.")
    service = get_tracked_queries_service()
    tq = await service.find_for_product(product_id)
    if not tq:
        raise HTTPException(status_code=404, detail="product is not enrolled in monitoring")
    row = await service.add_exclusion(
        tq["id"],
        url=body.url,
        domain=_normalize_domain(body.domain),
        reason=body.reason,
        api_key_id=None,
    )
    return ProductExclusionRow(
        id=row.get("id", ""),
        url=row.get("url"),
        domain=row.get("domain"),
        reason=row.get("reason"),
        excluded_at=row.get("excluded_at") or datetime.utcnow().isoformat(),
    )


@router.post("/products/{product_id}/include", summary="Undo a previous exclusion.")
async def include_product_result(
    product_id: str,
    body: ProductExcludeRequest,
    user: User = Depends(get_current_user),
):
    if not body.url and not body.domain:
        raise HTTPException(status_code=400, detail="Either `url` or `domain` is required.")
    service = get_tracked_queries_service()
    tq = await service.find_for_product(product_id)
    if not tq:
        return {"success": True, "product_id": product_id, "removed_count": 0}
    removed = await service.remove_exclusion(
        tq["id"], url=body.url, domain=_normalize_domain(body.domain),
    )
    return {"success": True, "product_id": product_id, "removed_count": removed}


@router.get(
    "/products/{product_id}/exclusions",
    response_model=List[ProductExclusionRow],
    summary="List every exclusion attached to this product.",
)
async def list_product_exclusions(
    product_id: str,
    user: User = Depends(get_current_user),
):
    service = get_tracked_queries_service()
    tq = await service.find_for_product(product_id)
    if not tq:
        return []
    rows = await service.list_exclusions(tq["id"])
    return [
        ProductExclusionRow(
            id=r["id"],
            url=r.get("url"),
            domain=r.get("domain"),
            reason=r.get("reason"),
            excluded_at=r.get("excluded_at") or datetime.utcnow().isoformat(),
        )
        for r in rows
    ]


@router.post(
    "/products/{product_id}/verify",
    summary="Re-verify retailer prices on demand (Firecrawl only — no new discovery).",
)
async def verify_product_sources(
    product_id: str,
    body: ProductVerifyRequest,
    user: User = Depends(get_current_user),
):
    service = get_tracked_queries_service()
    tq = await service.find_for_product(product_id)
    if not tq:
        return {
            "success": True,
            "status": "no_results",
            "rows_processed": 0,
            "credits_used": 0,
            "results": [],
            "message": "Product is not enrolled in monitoring.",
        }
    return await service.reverify(tq["id"], urls=body.urls)


# ============================================================================
# Mode = 'url-only' (Custom Monitoring — user pastes specific retailer URLs)
# ============================================================================


@router.post(
    "/products/{product_id}/url-only",
    response_model=DataResponse,
    summary="Add a pinned retailer URL to monitor for this product (no Perplexity discovery).",
)
async def add_url_only(
    product_id: str,
    body: AddUrlOnlyRequest,
    user: User = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    sb = get_supabase_client().client
    ctx = _resolve_product_context(sb, product_id)
    service = get_tracked_queries_service()
    tq = await service.add_url_only(
        product_id=product_id,
        url=body.url,
        product_name=ctx["name"],
        user_id=str(user.id),
        workspace_id=workspace.workspace_id if workspace else None,
        country_code=body.country_code,
    )
    return {"success": True, "data": tq}


@router.get(
    "/products/{product_id}/url-only",
    response_model=DataResponse,
    summary="List pinned-URL tracked queries for this product.",
)
async def list_url_only_for_product(
    product_id: str,
    user: User = Depends(get_current_user),
):
    service = get_tracked_queries_service()
    rows = await service.list_url_only_for_product(product_id)
    return {"success": True, "data": rows}


# ============================================================================
# Stateless market scan (cross-flow — survives the consolidation)
# ============================================================================


def _compute_market_stats(hits: List[PriceHit]) -> MarketStats:
    priced = [h for h in hits if h.price is not None]
    if not priced:
        return MarketStats(count=len(hits), verified_count=0)

    stat_hits = [
        h for h in priced
        if (h.match_kind is None or h.match_kind == "exact")
        and (h.availability != "out_of_stock")
    ]
    if not stat_hits:
        return MarketStats(
            count=len(priced),
            verified_count=sum(1 for h in priced if h.verified),
        )

    values = sorted(float(h.price) for h in stat_hits)

    if len(values) >= 4:
        provisional_median = (
            values[len(values) // 2]
            if len(values) % 2
            else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2
        )
        lo_bound = provisional_median / 3.0
        hi_bound = provisional_median * 3.0
        trimmed = [v for v in values if lo_bound <= v <= hi_bound]
        if trimmed:
            values = trimmed

    n = len(values)
    median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
    currencies = [h.currency for h in stat_hits if h.currency]
    currency = max(set(currencies), key=currencies.count) if currencies else None
    verified = sum(1 for h in priced if h.verified)
    return MarketStats(
        count=len(priced),
        verified_count=verified,
        min=values[0],
        max=values[-1],
        median=median,
        currency=currency,
    )


@router.post(
    "/market-check",
    response_model=MarketCheckResponse,
    summary="One-shot market scan for pricing decisions (stateless).",
)
async def market_check(
    body: MarketCheckRequest,
    user: User = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
) -> MarketCheckResponse:
    sb = get_supabase_client().client

    if not _is_admin(sb, user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="market-check requires admin or super_admin role.",
        )

    product_id = body.product_id
    product_name: Optional[str] = body.product_name
    dimensions: Optional[str] = body.dimensions
    manufacturer: Optional[str] = body.manufacturer

    if product_id and not product_name:
        ctx = _resolve_product_context(sb, product_id)
        product_name = ctx["name"]
        dimensions = dimensions or ctx["dimensions"]
        manufacturer = manufacturer or ctx["manufacturer"]

    if not product_name or not product_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either product_id (with a row that has a name) or product_name is required.",
        )

    profile = (
        sb.table("user_profiles")
        .select("location_country_code, location")
        .eq("user_id", user.id)
        .maybe_single()
        .execute()
    )
    country_code = _resolve_user_country_code((profile.data if profile else None) or {})

    query_text = product_name.strip()
    if manufacturer and manufacturer.lower() not in query_text.lower():
        query_text = f"{manufacturer} {query_text}".strip()
    if dimensions:
        query_text = f"{query_text} {dimensions}".strip()

    # Cache shortcut: if this product is already enrolled and the tracked_query
    # was refreshed in the last 6h, return the latest results without spending.
    if product_id:
        service = get_tracked_queries_service()
        tq = await service.find_for_product(product_id)
        if tq and tq.get("last_refreshed_at"):
            try:
                last_at = datetime.fromisoformat(
                    tq["last_refreshed_at"].replace("Z", "+00:00")
                )
                age_s = int((datetime.now(last_at.tzinfo) - last_at).total_seconds())
                if age_s <= 6 * 3600:
                    rows = await service.latest_results(tq["id"])
                    cached_hits: List[PriceHit] = []
                    for r in rows:
                        if r.get("price") is None:
                            continue
                        cached_hits.append(PriceHit(
                            retailer_name=r.get("retailer_name") or "Unknown",
                            product_url=r.get("product_url") or "",
                            price=float(r["price"]),
                            original_price=float(r["original_price"]) if r.get("original_price") is not None else None,
                            verified=bool(r.get("verified") or False),
                            source=r.get("source") or "perplexity",
                            currency=r.get("currency") or "EUR",
                            availability=r.get("availability") or "unknown",
                            notes=r.get("notes"),
                            match_kind=r.get("match_kind"),
                            match_score=r.get("match_score"),
                            match_note=r.get("match_note"),
                            product_title=r.get("product_title"),
                        ))
                    if cached_hits:
                        return MarketCheckResponse(
                            success=True,
                            product_id=product_id,
                            query=query_text,
                            country_code=country_code,
                            results=cached_hits,
                            total_results=len(cached_hits),
                            stats=_compute_market_stats(cached_hits),
                            summary=None,
                            credits_used=0,
                            latency_ms=0,
                            from_monitoring_cache=True,
                            cache_age_seconds=age_s,
                        )
            except Exception as e:
                logger.debug(f"market-check cache read failed (non-fatal): {e}")

    # Fresh scan. When called with a product_id we have structured metadata —
    # use it as the reference facets, no need to pay Haiku to re-decompose.
    catalog_facets = None
    if product_id:
        prod_row_for_facets = (
            sb.table("products")
            .select("id, name, metadata")
            .eq("id", product_id)
            .maybe_single()
            .execute()
        )
        catalog_facets = facets_from_catalog((prod_row_for_facets.data if prod_row_for_facets else None) or {})

    enriched_query = query_text
    if catalog_facets:
        prefix_parts: List[str] = []
        if catalog_facets.brand and normalize_model_token(catalog_facets.brand) not in normalize_model_token(query_text):
            prefix_parts.append(catalog_facets.brand)
        if catalog_facets.sku_tokens:
            prefix_parts.append(catalog_facets.sku_tokens[0])
        if prefix_parts:
            enriched_query = " ".join([*prefix_parts, query_text])

    service_search = get_perplexity_price_search_service()
    result = await service_search.search_prices(
        product_name=enriched_query,
        dimensions=None,
        country_code=country_code,
        limit=10,
        user_id=user.id,
        workspace_id=workspace.workspace_id if workspace else None,
        verify_prices=body.verify_prices,
        query_facets=catalog_facets,
        manufacturer_hint=manufacturer,
    )

    if not result.success:
        return MarketCheckResponse(
            success=False,
            product_id=product_id,
            query=query_text,
            country_code=country_code,
            stats=MarketStats(count=0, verified_count=0),
            credits_used=result.credits_used,
            latency_ms=result.latency_ms,
            error=result.error or "market scan failed",
        )

    return MarketCheckResponse(
        success=True,
        product_id=product_id,
        query=query_text,
        country_code=country_code,
        results=result.hits,
        total_results=len(result.hits),
        stats=_compute_market_stats(result.hits),
        summary=result.summary,
        credits_used=result.credits_used,
        latency_ms=result.latency_ms,
        from_monitoring_cache=False,
    )


# ============================================================================
# Cron-target batch refresh
# ============================================================================


@router.post(
    "/tracked-queries/cron-refresh",
    summary="Refresh all due tracked_queries (called by Supabase cron).",
)
async def cron_refresh_tracked_queries(request: Request, limit: int = Query(default=50, ge=1, le=500)) -> Dict[str, Any]:
    expected = os.getenv("CRON_SECRET") or ""
    provided = request.headers.get("x-cron-secret") or ""
    if not expected or provided != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing x-cron-secret")

    service = get_tracked_queries_service()
    due = await service.due_for_refresh(limit=limit)
    processed = 0
    succeeded = 0
    failed = 0
    total_credits = 0
    results: List[Dict[str, Any]] = []

    for row in due:
        tracking_id = row.get("id")
        if not tracking_id:
            continue
        try:
            outcome = await service.refresh(tracking_id, force=False)
            processed += 1
            total_credits += int(outcome.get("credits_used", 0) or 0)
            if outcome.get("status") == "refreshed":
                succeeded += 1
            else:
                failed += 1
            results.append({
                "tracking_id": tracking_id,
                "status": outcome.get("status"),
                "credits_used": outcome.get("credits_used", 0),
                "results_count": len(outcome.get("results") or []),
                "error": outcome.get("error"),
            })
        except Exception as e:
            logger.error(f"cron refresh crashed for {tracking_id}: {e}")
            failed += 1
            processed += 1
            results.append({"tracking_id": tracking_id, "status": "crashed", "error": str(e)})

    return {
        "success": True,
        "due_count": len(due),
        "processed": processed,
        "succeeded": succeeded,
        "failed": failed,
        "total_credits_used": total_credits,
        "results": results,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ============================================================================
# Classifier feedback loop + admin overrides (cross-flow)
# ============================================================================


@router.post("/classifier-correction", response_model=StatusResponse)
async def submit_classifier_correction(
    body: MatchCorrectionRequest,
    user: User = Depends(get_current_user),
):
    """Admin feedback on the identity classifier. Writes to `match_corrections`.
    Recent rows are pulled into the classifier system prompt on every classify
    call (cached 5 min) — closes the loop without retraining.
    """
    if not body.tracked_query_history_id:
        raise HTTPException(status_code=400, detail="tracked_query_history_id required")
    sb = get_supabase_client().client
    snapshot: Dict[str, Any] = {}
    try:
        ph = sb.table("tracked_query_price_history").select(
            "tracked_query_id, retailer_name, product_url, product_title, match_kind"
        ).eq("id", body.tracked_query_history_id).maybe_single().execute()
        row = (ph.data if ph else None) or {}
        tq = sb.table("tracked_queries").select("query_facets").eq(
            "id", row.get("tracked_query_id") or ""
        ).maybe_single().execute()
        tq_row = (tq.data if tq else None) or {}
        snapshot = {
            "tracked_query_id": row.get("tracked_query_id"),
            "tracked_query_history_id": body.tracked_query_history_id,
            "retailer_name": row.get("retailer_name"),
            "product_url": row.get("product_url"),
            "product_title": row.get("product_title"),
            "original_match_kind": row.get("match_kind"),
            "query_facets": tq_row.get("query_facets"),
            "page_facets": {"product_name": row.get("product_title")},
        }
    except Exception as e:
        logger.warning(f"correction snapshot fetch failed: {e}")

    try:
        sb.table("match_corrections").insert({
            **snapshot,
            "corrected_match_kind": body.corrected_match_kind,
            "correction_note": body.correction_note,
            "created_by": str(user.id),
        }).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"insert failed: {e}")
    return {"success": True, "message": "correction recorded"}


@router.post("/promote-family-row", response_model=StatusResponse)
async def promote_family_row(
    body: PromoteFamilyRequest,
    user: User = Depends(get_current_user),
):
    """Promote a family/mismatch row to tracked. Sticky — every future refresh
    of the same URL keeps the override until the admin demotes it back.
    """
    _require_admin(user)
    sb = get_supabase_client().client

    ph = sb.table("tracked_query_price_history").select(
        "tracked_query_id, product_url, retailer_name, match_kind"
    ).eq("id", body.tracked_query_history_id).maybe_single().execute()
    row = (ph.data if ph else None) or {}
    if not row:
        raise HTTPException(status_code=404, detail="history row not found")
    tracked_query_id = row.get("tracked_query_id")
    product_url = row.get("product_url")
    if not tracked_query_id or not product_url:
        raise HTTPException(status_code=400, detail="row missing tracked_query_id or product_url")
    original_kind = row.get("match_kind")

    sb.table("tracked_query_promoted_urls").upsert({
        "tracked_query_id": tracked_query_id,
        "product_url": product_url,
        "override_kind": body.override_kind,
        "reason": body.reason,
        "created_by": str(user.id),
    }, on_conflict="tracked_query_id,product_url").execute()

    sb.table("tracked_query_price_history").update({
        "match_kind": body.override_kind,
        "match_note": body.reason or "Manually promoted by admin",
        "manual_override": True,
    }).eq("tracked_query_id", tracked_query_id).eq("product_url", product_url).execute()

    sb.table("match_corrections").insert({
        "tracked_query_id": tracked_query_id,
        "tracked_query_history_id": body.tracked_query_history_id,
        "retailer_name": row.get("retailer_name"),
        "product_url": product_url,
        "original_match_kind": original_kind,
        "corrected_match_kind": body.override_kind,
        "correction_note": body.reason or "promoted via UI",
        "created_by": str(user.id),
    }).execute()

    return {"success": True, "message": "row promoted"}


@router.post("/demote-to-family", response_model=StatusResponse)
async def demote_to_family(
    body: DemoteFamilyRequest,
    user: User = Depends(get_current_user),
):
    """Undo a prior promotion. The URL goes back to whatever the classifier
    decides on the next refresh.
    """
    _require_admin(user)
    sb = get_supabase_client().client

    ph = sb.table("tracked_query_price_history").select(
        "tracked_query_id, product_url, retailer_name"
    ).eq("id", body.tracked_query_history_id).maybe_single().execute()
    row = (ph.data if ph else None) or {}
    if not row:
        raise HTTPException(status_code=404, detail="history row not found")
    sb.table("tracked_query_promoted_urls").delete().eq(
        "tracked_query_id", row.get("tracked_query_id")
    ).eq("product_url", row.get("product_url")).execute()
    sb.table("tracked_query_price_history").update({
        "match_kind": "family",
        "match_note": body.reason or "Demoted to family by admin",
        "manual_override": False,
    }).eq("tracked_query_id", row.get("tracked_query_id")).eq("product_url", row.get("product_url")).execute()
    sb.table("match_corrections").insert({
        "tracked_query_id": row.get("tracked_query_id"),
        "tracked_query_history_id": body.tracked_query_history_id,
        "retailer_name": row.get("retailer_name"),
        "product_url": row.get("product_url"),
        "corrected_match_kind": "should_drop",
        "correction_note": body.reason or "demoted via UI",
        "created_by": str(user.id),
    }).execute()
    return {"success": True, "message": "row demoted"}


# ============================================================================
# Email broadcast (admin one-shot)
# ============================================================================


@router.post("/broadcast-api-announcement", response_model=DataResponse)
async def broadcast_api_announcement(
    body: BroadcastApiAnnouncementRequest,
    user: User = Depends(get_current_user),
):
    """Admin-only one-shot broadcast to every distinct user_id that owns at
    least one active api_key. Renders the template against each recipient and
    dispatches via the `email-api` edge function (Resend-backed).
    """
    sb = get_supabase_client().client
    if not _is_admin(sb, user.id):
        raise HTTPException(status_code=403, detail="admin role required")

    keys = (
        sb.table("api_keys")
        .select("user_id")
        .eq("is_active", True)
        .execute()
    )
    user_ids = sorted({r["user_id"] for r in (keys.data or []) if r.get("user_id")})
    if not user_ids:
        return {"success": True, "data": {"recipients": 0, "message": "no api_key owners found"}}

    profiles = (
        sb.table("user_profiles")
        .select("user_id, email, full_name")
        .in_("user_id", user_ids)
        .execute()
    )
    by_uid = {p["user_id"]: p for p in (profiles.data or []) if p.get("email")}

    already: set = set()
    try:
        prior = (
            sb.table("email_logs")
            .select("user_id")
            .eq("template_slug", body.template_slug)
            .eq("status", "sent")
            .execute()
        )
        already = {r.get("user_id") for r in (prior.data or []) if r.get("user_id")}
    except Exception:
        pass

    targets = [
        {"user_id": uid, "email": p["email"], "full_name": p.get("full_name") or "there"}
        for uid, p in by_uid.items()
        if uid not in already
    ]

    if body.dry_run:
        return {"success": True, "data": {
            "recipients": len(targets),
            "skipped_already_sent": len(already),
            "missing_email": len(user_ids) - len(by_uid),
            "preview": targets[:5],
        }}

    sent = 0
    failed = 0
    supabase_url = os.getenv("SUPABASE_URL") or ""
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""
    if not supabase_url or not service_key:
        raise HTTPException(status_code=500, detail="SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY missing")

    with _httpx.Client(timeout=15.0) as client:
        for t in targets:
            try:
                resp = client.post(
                    f"{supabase_url}/functions/v1/email-api",
                    headers={
                        "Authorization": f"Bearer {service_key}",
                        "apikey": service_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "action": "send",
                        "to": t["email"],
                        "templateSlug": body.template_slug,
                        "variables": {
                            "name": t["full_name"],
                            "user_name": t["full_name"],
                            "docs_url": body.docs_url,
                            "api_base_url": body.api_base_url,
                            "support_email": body.support_email,
                        },
                        "tags": [
                            {"name": "category", "value": "api_broadcast"},
                            {"name": "template", "value": body.template_slug},
                        ],
                    },
                )
                if resp.status_code < 400:
                    sent += 1
                else:
                    failed += 1
                    logger.warning(f"broadcast send failed for {t['email']}: {resp.status_code} {resp.text[:200]}")
            except Exception as e:
                failed += 1
                logger.warning(f"broadcast send raised for {t['email']}: {e}")

    return {"success": True, "data": {
        "recipients": len(targets),
        "sent": sent,
        "failed": failed,
        "skipped_already_sent": len(already),
    }}


# ============================================================================
# Legacy aliases — kept short-term for the frontend rewire window.
# ============================================================================
# These map the old shape (POST /start, /stop, /check-now, /discover, /sources/{id})
# onto the new product-scoped endpoints. Frontend should migrate to the new
# /products/{product_id}/* surface and these aliases can be deleted.


class _LegacyProductBody(BaseModel):
    product_id: str
    product_name: Optional[str] = None
    frequency: Optional[str] = None
    enabled: Optional[bool] = None
    force_refresh: Optional[bool] = None
    verify_prices: Optional[bool] = None


@router.post("/start", response_model=MonitoringActionResponse, deprecated=True)
async def legacy_start(
    body: _LegacyProductBody,
    user: User = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    sb = get_supabase_client().client
    ctx = _resolve_product_context(sb, body.product_id)
    profile = (
        sb.table("user_profiles")
        .select("location_country_code, location")
        .eq("user_id", user.id)
        .maybe_single()
        .execute()
    )
    country = _resolve_user_country_code((profile.data if profile else None) or {})
    service = get_tracked_queries_service()
    tq = await service.find_or_create_for_product(
        product_id=body.product_id,
        product_name=ctx["name"],
        manufacturer=ctx["manufacturer"],
        dimensions=ctx["dimensions"],
        country_code=country,
        user_id=str(user.id),
        workspace_id=workspace.workspace_id if workspace else None,
        run_first_refresh=False,
    )
    if not tq.get("is_active"):
        await service.reactivate(tq["id"])
    return {"success": True, "message": "monitoring active", "data": {"tracked_query_id": tq.get("id")}}


@router.post("/stop", response_model=MonitoringActionResponse, deprecated=True)
async def legacy_stop(
    product_id: str = Query(...),
    user: User = Depends(get_current_user),
):
    service = get_tracked_queries_service()
    tq = await service.find_for_product(product_id)
    if not tq:
        return {"success": True, "message": "not enrolled"}
    await service.deactivate(tq["id"])
    return {"success": True, "message": "monitoring stopped"}


@router.post("/check-now", response_model=MonitoringActionResponse, deprecated=True)
async def legacy_check_now(
    body: _LegacyProductBody,
    user: User = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    sb = get_supabase_client().client
    service = get_tracked_queries_service()
    tq = await service.find_for_product(body.product_id)
    if not tq:
        ctx = _resolve_product_context(sb, body.product_id)
        profile = (
            sb.table("user_profiles")
            .select("location_country_code, location")
            .eq("user_id", user.id)
            .maybe_single()
            .execute()
        )
        country = _resolve_user_country_code((profile.data if profile else None) or {})
        tq = await service.find_or_create_for_product(
            product_id=body.product_id,
            product_name=ctx["name"],
            manufacturer=ctx["manufacturer"],
            dimensions=ctx["dimensions"],
            country_code=country,
            user_id=str(user.id),
            workspace_id=workspace.workspace_id if workspace else None,
            run_first_refresh=True,
            force=True,
        )
        return {"success": True, "message": "first refresh complete", "data": tq}
    outcome = await service.refresh(tq["id"], force=True)
    return {"success": outcome.get("status") == "refreshed", "message": outcome.get("status"), "data": outcome}


@router.post("/discover", deprecated=True)
async def legacy_discover(
    body: _LegacyProductBody,
    user: User = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    """Deprecated — use POST /products/{product_id}/refresh."""
    refresh_body = RefreshProductRequest(
        force_refresh=bool(body.force_refresh),
        verify_prices=bool(body.verify_prices) if body.verify_prices is not None else True,
    )
    res = await refresh_product(body.product_id, refresh_body, user, workspace)
    data = res.get("data") or {}
    return {
        "success": res.get("success", False),
        "source": "perplexity_web_search",
        "product_id": body.product_id,
        "results": data.get("results") or [],
        "total_results": len(data.get("results") or []),
        "credits_used": data.get("credits_used", 0),
        "latency_ms": data.get("latency_ms", 0),
        "summary": data.get("summary"),
        "throttled": data.get("status") == "throttled",
        "throttle_until": data.get("throttle_until"),
        "last_search_at": None,
        "cached": False,
        "error": data.get("error"),
    }


@router.get("/status/{product_id}", response_model=DataResponse, deprecated=True)
async def legacy_status(
    product_id: str,
    user: User = Depends(get_current_user),
):
    return await get_product_monitoring(product_id, user)


@router.get("/history/{product_id}", response_model=PriceHistoryResponse, deprecated=True)
async def legacy_history(
    product_id: str,
    limit: int = Query(default=200, ge=1, le=2000),
    user: User = Depends(get_current_user),
):
    return await get_product_history(product_id, limit, user)


@router.get("/sources/{product_id}", response_model=PriceSourceResponse, deprecated=True)
async def legacy_sources(
    product_id: str,
    user: User = Depends(get_current_user),
):
    res = await get_product_sources(product_id, user)
    data = res.get("data") or {}
    return {
        "success": True,
        "sources": (data.get("results") or []) + (data.get("family_results") or []),
        "count": len(data.get("results") or []) + len(data.get("family_results") or []),
    }
