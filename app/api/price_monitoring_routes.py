"""
Price Monitoring API Routes

FastAPI endpoints for price monitoring functionality:
- Start/stop monitoring
- On-demand price checks
- Get price history and statistics
- Manage competitor sources
- Configure price alerts
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status, Query, Request
from pydantic import BaseModel, Field

from app.services.integrations.price_monitoring_service import get_price_monitoring_service
from app.services.integrations.perplexity_price_search_service import (
    get_perplexity_price_search_service,
    PriceHit,
)
from app.services.integrations.tracked_queries_service import get_tracked_queries_service
from app.services.integrations.product_identity_service import (
    facets_from_catalog,
    normalize_model_token,
)
import os
from app.services.core.supabase_client import get_supabase_client
from app.dependencies import get_current_user, get_workspace_context
from app.middleware.jwt_auth import User, WorkspaceContext
from app.schemas.api_responses import (
    StatusResponse, DataResponse, MonitoringActionResponse,
    PriceHistoryResponse, PriceStatisticsResponse,
    PriceSourceResponse, PriceAlertResponse, PriceJobsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/price-monitoring",
    tags=["Price Monitoring"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not found"}
    }
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class StartMonitoringRequest(BaseModel):
    """Request to start price monitoring for a product"""
    product_id: str = Field(..., description="Product UUID to monitor")
    frequency: str = Field(
        default="daily",
        description="Monitoring frequency: hourly, daily, weekly, on_demand"
    )
    enabled: bool = Field(default=True, description="Enable monitoring")


class CheckPricesRequest(BaseModel):
    """Request to perform on-demand price check"""
    product_id: str = Field(..., description="Product UUID")
    product_name: str = Field(..., description="Product name for scraping context")


class AddCompetitorSourceRequest(BaseModel):
    """Request to add a competitor source"""
    product_id: str = Field(..., description="Product UUID")
    source_name: str = Field(..., description="Competitor name")
    source_url: str = Field(..., description="Competitor product URL")
    scraping_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional Firecrawl configuration"
    )


class DiscoverSourcesRequest(BaseModel):
    """Request to run Claude web_search discovery for a monitored product."""
    product_id: str = Field(..., description="Product UUID to discover retailers for.")
    force_refresh: bool = Field(
        default=False,
        description="Bypass the 6h throttle. Admin/super_admin role required.",
    )
    verify_prices: bool = Field(
        default=True,
        description=(
            "When true (default), every discovered hit is re-fetched via Firecrawl to "
            "confirm the price on the live page. Set false for faster/cheaper discovery "
            "at the cost of accuracy (prices may be stale / LLM-hallucinated)."
        ),
    )


class DiscoverSourcesResponse(BaseModel):
    """Response from /discover. Returns the retailer list + throttle state."""
    success: bool
    source: str = "perplexity_web_search"
    product_id: str
    results: List[PriceHit] = []
    total_results: int = 0
    credits_used: int = 0
    latency_ms: int = 0
    summary: Optional[str] = Field(
        default=None,
        description="2-3 sentence Perplexity summary: closest retailer, manufacturer showroom presence, pricing outliers to question.",
    )
    throttled: bool = False
    throttle_until: Optional[str] = Field(
        default=None,
        description="ISO timestamp: if throttled, next allowed refresh time (unless admin force-refreshes).",
    )
    last_search_at: Optional[str] = None
    cached: bool = Field(
        default=False,
        description="True if the throttle prevented a new search and we returned existing sources.",
    )
    error: Optional[str] = None


class CreatePriceAlertRequest(BaseModel):
    """Request to create a price alert"""
    product_id: str = Field(..., description="Product UUID")
    alert_type: str = Field(
        ...,
        description="Alert type: price_drop, price_increase, any_change, availability"
    )
    threshold_percentage: Optional[float] = Field(
        default=None,
        description="Percentage threshold for alert"
    )
    threshold_amount: Optional[float] = Field(
        default=None,
        description="Amount threshold for alert"
    )
    notification_channels: List[str] = Field(
        default=["email"],
        description="Notification channels: email, in_app, sms"
    )


# ============================================================================
# MONITORING ENDPOINTS
# ============================================================================

@router.post("/start", response_model=MonitoringActionResponse)
async def start_monitoring(
    request: StartMonitoringRequest,
    user: User = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context)
):
    """
    Start price monitoring for a product.
    
    Requires Factory or Store role.
    """
    try:
        service = get_price_monitoring_service()
        
        result = await service.start_monitoring(
            product_id=request.product_id,
            user_id=str(user.id),
            workspace_id=str(workspace.workspace_id),
            frequency=request.frequency,
            enabled=request.enabled
        )
        
        if result.get("success"):
            return {
                "success": True,
                "message": f"Started {request.frequency} monitoring",
                "monitoring": result.get("monitoring")
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to start monitoring")
            )
            
    except Exception as e:
        logger.error(f"❌ Failed to start monitoring: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/stop", response_model=MonitoringActionResponse)
async def stop_monitoring(
    product_id: str = Query(..., description="Product UUID"),
    user: User = Depends(get_current_user)
):
    """
    Stop price monitoring for a product.
    """
    try:
        service = get_price_monitoring_service()
        
        result = await service.stop_monitoring(
            product_id=product_id,
            user_id=str(user.id)
        )
        
        if result.get("success"):
            return {
                "success": True,
                "message": "Monitoring stopped"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to stop monitoring")
            )
            
    except Exception as e:
        logger.error(f"❌ Failed to stop monitoring: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/check-now", response_model=MonitoringActionResponse)
async def check_prices_now(
    request: CheckPricesRequest,
    user: User = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context)
):
    """
    Perform on-demand price check for a product.

    Scrapes all active competitor sources and saves price history.
    Debits Firecrawl credits from user's account.
    """
    try:
        service = get_price_monitoring_service()

        result = await service.check_prices_now(
            product_id=request.product_id,
            user_id=str(user.id),
            workspace_id=str(workspace.workspace_id),
            product_name=request.product_name
        )

        if result.get("success"):
            return {
                "success": True,
                "message": f"Checked {result.get('sources_checked', 0)} sources",
                "job_id": result.get("job_id"),
                "sources_checked": result.get("sources_checked", 0),
                "prices_found": result.get("prices_found", 0),
                "credits_consumed": result.get("credits_consumed", 0)
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Price check failed")
            )

    except Exception as e:
        logger.error(f"❌ Failed to check prices: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/status/{product_id}", response_model=DataResponse)
async def get_monitoring_status(
    product_id: str,
    user: User = Depends(get_current_user)
):
    """
    Get monitoring status for a product.
    """
    try:
        supabase = get_supabase_client()

        response = supabase.client.table("price_monitoring_products").select("*").eq(
            "product_id", product_id
        ).eq("user_id", str(user.id)).execute()

        if response.data:
            return {
                "success": True,
                "monitoring": response.data[0]
            }
        else:
            return {
                "success": True,
                "monitoring": None,
                "message": "No monitoring configured"
            }

    except Exception as e:
        logger.error(f"❌ Failed to get monitoring status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# PRICE HISTORY ENDPOINTS
# ============================================================================

@router.get("/history/{product_id}", response_model=PriceHistoryResponse)
async def get_price_history(
    product_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    source_name: Optional[str] = Query(default=None),
    user: User = Depends(get_current_user)
):
    """
    Get price history for a product.

    Optionally filter by source_name.
    """
    try:
        supabase = get_supabase_client()

        query = supabase.client.table("price_history").select("*").eq(
            "product_id", product_id
        ).order("scraped_at", desc=True).limit(limit)

        if source_name:
            query = query.eq("source_name", source_name)

        response = query.execute()

        return {
            "success": True,
            "history": response.data or [],
            "count": len(response.data or [])
        }

    except Exception as e:
        logger.error(f"❌ Failed to get price history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/statistics/{product_id}", response_model=PriceStatisticsResponse)
async def get_price_statistics(
    product_id: str,
    user: User = Depends(get_current_user)
):
    """
    Get price statistics for a product.

    Returns min, max, avg prices and trend analysis.
    """
    try:
        service = get_price_monitoring_service()

        stats = await service.get_price_statistics(product_id)

        return {
            "success": True,
            "statistics": stats
        }

    except Exception as e:
        logger.error(f"❌ Failed to get price statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# COMPETITOR SOURCES ENDPOINTS
# ============================================================================

@router.post("/sources", response_model=PriceSourceResponse)
async def add_competitor_source(
    request: AddCompetitorSourceRequest,
    user: User = Depends(get_current_user)
):
    """
    Add a competitor source for price monitoring.
    """
    try:
        supabase = get_supabase_client()

        response = supabase.client.table("competitor_sources").insert({
            "product_id": request.product_id,
            "source_name": request.source_name,
            "source_url": request.source_url,
            "scraping_config": request.scraping_config or {},
            "is_active": True,
            "error_count": 0,
            "created_by": str(user.id)
        }).execute()

        if response.data:
            return {
                "success": True,
                "message": "Competitor source added",
                "source": response.data[0]
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add competitor source"
            )

    except Exception as e:
        logger.error(f"❌ Failed to add competitor source: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/sources/{product_id}", response_model=PriceSourceResponse)
async def get_competitor_sources(
    product_id: str,
    user: User = Depends(get_current_user)
):
    """
    Get all competitor sources for a product.
    """
    try:
        supabase = get_supabase_client()

        response = supabase.client.table("competitor_sources").select("*").eq(
            "product_id", product_id
        ).order("created_at", desc=True).execute()

        return {
            "success": True,
            "sources": response.data or [],
            "count": len(response.data or [])
        }

    except Exception as e:
        logger.error(f"❌ Failed to get competitor sources: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/sources/{source_id}", response_model=StatusResponse)
async def delete_competitor_source(
    source_id: str,
    user: User = Depends(get_current_user)
):
    """
    Delete a competitor source.
    """
    try:
        supabase = get_supabase_client()

        response = supabase.client.table("competitor_sources").delete().eq(
            "id", source_id
        ).execute()

        return {
            "success": True,
            "message": "Competitor source deleted"
        }

    except Exception as e:
        logger.error(f"❌ Failed to delete competitor source: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# PRICE ALERTS ENDPOINTS
# ============================================================================

@router.post("/alerts", response_model=PriceAlertResponse)
async def create_price_alert(
    request: CreatePriceAlertRequest,
    user: User = Depends(get_current_user)
):
    """
    Create a price alert for a product.
    """
    try:
        supabase = get_supabase_client()

        response = supabase.client.table("price_alerts").insert({
            "user_id": str(user.id),
            "product_id": request.product_id,
            "alert_type": request.alert_type,
            "threshold_percentage": request.threshold_percentage,
            "threshold_amount": request.threshold_amount,
            "notification_channels": request.notification_channels,
            "is_active": True,
            "trigger_count": 0
        }).execute()

        if response.data:
            return {
                "success": True,
                "message": "Price alert created",
                "alert": response.data[0]
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create price alert"
            )

    except Exception as e:
        logger.error(f"❌ Failed to create price alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/alerts/{product_id}", response_model=PriceAlertResponse)
async def get_price_alerts(
    product_id: str,
    user: User = Depends(get_current_user)
):
    """
    Get all price alerts for a product.
    """
    try:
        supabase = get_supabase_client()

        response = supabase.client.table("price_alerts").select("*").eq(
            "product_id", product_id
        ).eq("user_id", str(user.id)).order("created_at", desc=True).execute()

        return {
            "success": True,
            "alerts": response.data or [],
            "count": len(response.data or [])
        }

    except Exception as e:
        logger.error(f"❌ Failed to get price alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/alerts/{alert_id}", response_model=StatusResponse)
async def delete_price_alert(
    alert_id: str,
    user: User = Depends(get_current_user)
):
    """
    Delete a price alert.
    """
    try:
        supabase = get_supabase_client()

        response = supabase.client.table("price_alerts").delete().eq(
            "id", alert_id
        ).eq("user_id", str(user.id)).execute()

        return {
            "success": True,
            "message": "Price alert deleted"
        }

    except Exception as e:
        logger.error(f"❌ Failed to delete price alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# JOB MONITORING ENDPOINTS
# ============================================================================

@router.post(
    "/discover",
    response_model=DiscoverSourcesResponse,
    summary="Discover retailers via Claude web_search for a monitored product",
    description=(
        "Runs Claude's web_search tool to find up to 10 retailers selling the "
        "specified product. Biases results toward the user's country (from profile) "
        "but does not restrict. Throttled to once per 6h per product; set "
        "`force_refresh=true` (admin/super_admin only) to bypass. Inserts discovered "
        "retailers into `competitor_sources` with `source_type='perplexity_web_search'` "
        "and writes price snapshots to `price_history`."
    ),
)
async def discover_sources(
    request: DiscoverSourcesRequest,
    user: User = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
) -> DiscoverSourcesResponse:
    sb = get_supabase_client().client
    service = get_perplexity_price_search_service()
    product_id = request.product_id

    # ── Admin check for force_refresh ──
    if request.force_refresh and not _is_admin(sb, user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="force_refresh requires admin or super_admin role.",
        )

    # ── Confirm the product belongs to a monitoring subscription owned by this user ──
    monitoring = (
        sb.table("price_monitoring_products")
        .select("id, product_id, user_id, workspace_id, last_claude_search_at")
        .eq("product_id", product_id)
        .eq("user_id", user.id)
        .maybe_single()
        .execute()
    )
    mon_row = (monitoring.data if monitoring else None) or None
    if not mon_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No monitoring subscription found for this product. Call /start first.",
        )

    last_at_raw = mon_row.get("last_claude_search_at")
    last_at = datetime.fromisoformat(last_at_raw.replace("Z", "+00:00")) if last_at_raw else None
    throttled, throttle_until = service.check_throttle(last_at, force_refresh=request.force_refresh)

    if throttled:
        # Return existing sources instead of making a fresh call.
        existing = (
            sb.table("competitor_sources")
            .select("source_name, source_url, source_type, current_price, current_original_price, current_price_verified, current_currency, current_availability, current_metadata, match_kind, match_score, match_note, last_seen_at")
            .eq("product_id", product_id)
            .in_("source_type", ["perplexity_web_search", "dataforseo_shopping"])
            .eq("is_active", True)
            .order("last_seen_at", desc=True)
            .limit(20)
            .execute()
        )
        rows = existing.data or []
        cached_hits = []
        for r in rows:
            if r.get("current_price") is None:
                continue
            meta = r.get("current_metadata") or {}
            cached_hits.append(PriceHit(
                retailer_name=r.get("source_name") or "Unknown",
                product_url=r.get("source_url") or "",
                price=float(r.get("current_price") or 0),
                original_price=float(r["current_original_price"]) if r.get("current_original_price") is not None else None,
                verified=bool(r.get("current_price_verified") or False),
                source="dataforseo" if r.get("source_type") == "dataforseo_shopping" else "perplexity",
                currency=r.get("current_currency") or "USD",
                availability=r.get("current_availability") or "unknown",
                match_kind=r.get("match_kind"),
                match_score=r.get("match_score"),
                match_note=r.get("match_note"),
                product_title=meta.get("product_title"),
            ))
        return DiscoverSourcesResponse(
            success=True,
            product_id=product_id,
            results=cached_hits,
            total_results=len(cached_hits),
            throttled=True,
            throttle_until=throttle_until.isoformat() if throttle_until else None,
            last_search_at=last_at_raw,
            cached=True,
        )

    # ── Fetch product details for the prompt ──
    prod = (
        sb.table("products")
        .select("id, name, metadata")
        .eq("id", product_id)
        .maybe_single()
        .execute()
    )
    prod_row = (prod.data if prod else None) or {}
    product_name = prod_row.get("name") or "unknown product"
    metadata = prod_row.get("metadata") or {}
    # `metadata.dimensions` from the PDF ingestion pipeline is a list[dict]
    # (e.g. [{"metric_cm":"11.8x11.8","imperial_in":"4.65x4.65"}]). Flatten to
    # a freeform string so it can be appended to the Perplexity/DataForSEO
    # query without dragging dict repr into the prompt.
    raw_dims = (
        metadata.get("dimensions")
        or metadata.get("size")
        or metadata.get("product_size")
    )
    if isinstance(raw_dims, list) and raw_dims:
        first = raw_dims[0]
        if isinstance(first, dict):
            dimensions = first.get("metric_cm") or first.get("imperial_in") or None
        else:
            dimensions = str(first)
    elif isinstance(raw_dims, dict):
        dimensions = raw_dims.get("metric_cm") or raw_dims.get("imperial_in") or None
    else:
        dimensions = raw_dims

    # ── User's country for regional preference ──
    # Priority: location_country_code (ISO) → derived from free-text `location`
    # (e.g. "Greece" → "GR") → DEFAULT_COUNTRY_CODE (GR).
    profile = (
        sb.table("user_profiles")
        .select("location_country_code, location")
        .eq("user_id", user.id)
        .maybe_single()
        .execute()
    )
    prof_row = (profile.data if profile else None) or {}
    country_code = _resolve_user_country_code(prof_row)

    # Build reference facets from catalog metadata — structured signal beats
    # re-parsing the free-text name with Haiku when we already have the data.
    catalog_facets = facets_from_catalog(prod_row)

    # Imports for the price-monitoring-notifications module — sanity band +
    # alert detection. Local import avoids a hard dependency from this route
    # file when the module is disabled.
    from app.modules.price_monitoring_notifications.service import (
        get_price_alert_dispatcher,
        _domain_of as _alert_domain_of,
    )

    # Build the Stage A query string with the strongest identity tokens up
    # front: brand + collection + (one strong SKU when known). Retailer search
    # ranking on Perplexity/DataForSEO is keyword-driven, so seeding the query
    # with the SKU pulls the right merchants to the top of the results.
    enriched_name = product_name
    if catalog_facets:
        prefix_parts: List[str] = []
        if catalog_facets.brand and normalize_model_token(catalog_facets.brand) not in normalize_model_token(product_name):
            prefix_parts.append(catalog_facets.brand)
        if catalog_facets.sku_tokens:
            prefix_parts.append(catalog_facets.sku_tokens[0])
        if prefix_parts:
            enriched_name = " ".join([*prefix_parts, product_name])

    # Pull known retailer domains so the prompt can ask for ADDITIONAL retailers.
    known_domains: List[str] = []
    try:
        known = (
            sb.table("competitor_sources")
            .select("source_domain, source_url")
            .eq("product_id", product_id)
            .eq("is_active", True)
            .limit(50)
            .execute()
        )
        seen: set = set()
        for r in known.data or []:
            d = r.get("source_domain") or _alert_domain_of(r.get("source_url"))
            if d and d not in seen:
                seen.add(d)
                known_domains.append(d)
    except Exception as e:
        logger.debug(f"known-retailer pull failed: {e}")

    # ── Run Stage A discovery ──
    result = await service.search_prices(
        product_name=enriched_name,
        dimensions=dimensions,
        country_code=country_code,
        limit=10,
        user_id=user.id,
        workspace_id=mon_row.get("workspace_id") or (workspace.workspace_id if workspace else None),
        verify_prices=request.verify_prices,
        query_facets=catalog_facets,
        manufacturer_hint=(
            metadata.get("factory_name")
            or metadata.get("manufacturer")
            or metadata.get("brand")
        ) if metadata else None,
        known_retailer_domains=known_domains,
    )

    if not result.success:
        return DiscoverSourcesResponse(
            success=False,
            product_id=product_id,
            error=result.error or "Claude search failed",
            credits_used=result.credits_used,
            latency_ms=result.latency_ms,
        )

    # ── Per-product exclusions (2026-04-28): drop hits whose URL/domain has
    #    been explicitly excluded by an admin. Filtering BEFORE persist means
    #    excluded URLs never feed competitor_sources or price_history, so
    #    they never appear in the chart/median/alerts.
    try:
        ex = (
            sb.table("product_excluded_urls")
            .select("url, domain")
            .eq("product_id", product_id)
            .execute()
        )
        excluded_urls = {e["url"] for e in (ex.data or []) if e.get("url")}
        excluded_domains = {e["domain"] for e in (ex.data or []) if e.get("domain")}
        if excluded_urls or excluded_domains:
            before = len(result.hits)
            result.hits = [
                h for h in result.hits
                if h.product_url not in excluded_urls
                and (_alert_domain_of(h.product_url) or "") not in excluded_domains
            ]
            dropped = before - len(result.hits)
            if dropped > 0:
                logger.info(f"discover: product={product_id} dropped {dropped} hits via exclusions")
    except Exception as e:
        logger.warning(f"product exclusion filter failed (non-fatal): {e}")

    # ── Persist: upsert competitor_sources by (product_id, source_url),
    #    then insert price_history rows, then stamp last_claude_search_at.
    #    Hit source maps to competitor_source_type: perplexity → perplexity_web_search,
    #    dataforseo → dataforseo_shopping. So the UI can split them into
    #    "Discovered retailers" vs "Merchants" sections by source_type.
    #    Each price reading also passes through the sanity band before insert
    #    so outliers get flagged (is_anomaly=true) without polluting medians.
    alert_dispatcher = get_price_alert_dispatcher()
    persisted_history_rows: List[Dict[str, Any]] = []
    now_iso = datetime.utcnow().isoformat()
    for hit in result.hits:
        source_type = "dataforseo_shopping" if hit.source == "dataforseo" else "perplexity_web_search"
        # Build current_metadata — only non-null fields, so the JSON stays tidy.
        meta: Dict[str, Any] = {}
        if hit.image_url:
            meta["image_url"] = hit.image_url
        if hit.rating_value is not None:
            meta["rating_value"] = float(hit.rating_value)
        if hit.rating_votes is not None:
            meta["rating_votes"] = int(hit.rating_votes)
        if hit.notes:
            meta["notes"] = hit.notes
        if hit.product_title:
            meta["product_title"] = hit.product_title
        # Sanity band check on the new reading.
        price_val = float(hit.price) if hit.price is not None else None
        domain = _alert_domain_of(hit.product_url)
        is_anomaly = False
        anomaly_reason: Optional[str] = None
        rolling_med: Optional[float] = None
        if price_val is not None:
            verdict = alert_dispatcher.check_sanity(
                product_id=product_id,
                retailer_domain=domain or "",
                new_price=price_val,
            )
            is_anomaly = verdict.is_anomaly
            anomaly_reason = verdict.reason if is_anomaly else None
            rolling_med = verdict.rolling_median

        # Anomaly readings DO NOT overwrite the cached current_price on
        # competitor_sources — admin must accept via manual_override first.
        upsert_row = {
            "product_id": product_id,
            "source_name": hit.retailer_name,
            "source_url": hit.product_url,
            "source_domain": domain,
            "source_type": source_type,
            "discovered_via": source_type,
            "auto_discovered": True,
            "is_active": True,
            "current_metadata": meta or None,
            "match_kind": hit.match_kind,
            "match_score": hit.match_score,
            "match_note": hit.match_note,
            "last_seen_at": now_iso,
            "last_successful_scrape": now_iso,
            "error_count": 0,
            "last_error": None,
            "created_by": user.id,
        }
        if not is_anomaly:
            upsert_row.update({
                "current_price": price_val,
                "current_original_price": float(hit.original_price) if hit.original_price is not None else None,
                "current_price_verified": bool(getattr(hit, "verified", False)),
                "current_currency": hit.currency,
                "current_availability": hit.availability or "unknown",
                "current_price_updated_at": now_iso,
            })
        try:
            sb.table("competitor_sources").upsert(upsert_row, on_conflict="product_id,source_url").execute()
        except Exception as e:
            logger.warning(f"Upsert competitor_source failed for {hit.product_url}: {e}")
            continue

        history_row = {
            "product_id": product_id,
            "source_name": hit.retailer_name,
            "source_url": hit.product_url,
            "price": price_val,
            "original_price": float(hit.original_price) if hit.original_price is not None else None,
            "verified": bool(getattr(hit, "verified", False)),
            "currency": hit.currency,
            "availability": hit.availability or "unknown",
            "match_kind": hit.match_kind,
            "match_score": hit.match_score,
            "match_note": hit.match_note,
            "product_title": hit.product_title,
            "scraped_at": now_iso,
            "is_anomaly": is_anomaly,
            "anomaly_reason": anomaly_reason,
            "rolling_median_at_check": rolling_med,
            "metadata": {
                "via": source_type,
                "notes": hit.notes,
                "image_url": hit.image_url,
                "rating_value": hit.rating_value,
                "rating_votes": hit.rating_votes,
            },
        }
        try:
            sb.table("price_history").insert(history_row).execute()
            persisted_history_rows.append(history_row)
        except Exception as e:
            logger.warning(f"price_history insert failed for {hit.product_url}: {e}")

    # Stamp throttle timestamp + running credit counter
    try:
        sb.table("price_monitoring_products").update({
            "last_claude_search_at": now_iso,
            "last_claude_credits_used": result.credits_used,
            "total_claude_credits_used": (
                (mon_row.get("total_claude_credits_used") or 0) + result.credits_used
            ),
            "updated_at": now_iso,
        }).eq("id", mon_row["id"]).execute()
    except Exception as e:
        logger.warning(f"Failed to stamp last_claude_search_at on monitoring row: {e}")

    # Alert detection — module-gated + credit-metered + dedupe inside the
    # dispatcher. Runs after persistence so detection sees committed state.
    if persisted_history_rows:
        try:
            candidates = alert_dispatcher.detect_after_refresh(
                product_id=product_id,
                new_rows=persisted_history_rows,
            )
            if candidates:
                fired = alert_dispatcher.dispatch(candidates)
                logger.info(
                    f"price-alerts: product={product_id} fired {fired}/{len(candidates)} alerts"
                )
        except Exception as e:
            logger.warning(f"price-alerts: dispatch failed (non-fatal): {e}")

    return DiscoverSourcesResponse(
        success=True,
        product_id=product_id,
        results=result.hits,
        total_results=len(result.hits),
        credits_used=result.credits_used,
        latency_ms=result.latency_ms,
        summary=result.summary,
        last_search_at=now_iso,
        cached=False,
    )


class MarketCheckRequest(BaseModel):
    """
    Stateless one-shot market scan for pricing decisions.

    Either pass `product_id` (we'll read name + dimensions from the catalog)
    or pass free-text `product_name` + optional `dimensions` directly.
    """
    product_id: Optional[str] = Field(default=None, description="Catalog product UUID (preferred — gives the richest context).")
    product_name: Optional[str] = Field(default=None, description="Free-text product name — used when no catalog product_id is available.")
    dimensions: Optional[str] = Field(default=None, description="Optional size spec (e.g. '60x60 cm'), appended to the query.")
    manufacturer: Optional[str] = Field(default=None, description="Optional manufacturer to disambiguate generic names.")
    verify_prices: bool = Field(default=True, description="Run the Firecrawl verification pass. Defaults to true for pricing accuracy.")


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
    from_monitoring_cache: bool = Field(
        default=False,
        description="True when this product already has active monitoring and we returned the cached competitor_sources rows instead of spending credits.",
    )
    cache_age_seconds: Optional[int] = None
    error: Optional[str] = None


def _compute_market_stats(hits: List[PriceHit]) -> MarketStats:
    """
    Build min/max/median + verified count. Per 2026-04-25 product decision,
    ONLY `exact` identity matches count toward statistics — variants are a
    different SKU (different color/finish) and would inflate the range;
    unverifiable rows lack identity confirmation so we can't trust them
    either. `count` still returns the total number of priced rows because
    the UI shows "3 of 8 priced and matched" style breakdowns.

    Additional exclusions (2026-04-25 round 2):
      * out_of_stock rows — stale prices, not actionable as competitor data
      * extreme outliers (>3× or <1/3 of the median) — almost certainly a
        different product (shower-system vs spout, set vs single piece).
        Only applied when n >= 4 — too few hits and one valid premium
        retailer would skew the median.
    """
    priced = [h for h in hits if h.price is not None]
    if not priced:
        return MarketStats(count=len(hits), verified_count=0)

    # Stats are computed only from exact matches (or legacy rows where
    # identity wasn't evaluated — match_kind is None) AND in-stock rows.
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

    # Outlier trim — only meaningful when we have enough data points to know
    # what "typical" is. Below 4 hits, every row is signal.
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
    summary="One-shot market scan for pricing decisions (stateless)",
    description=(
        "Admin-only. Runs Perplexity + DataForSEO + Firecrawl verification against "
        "the retailer market and returns min/max/median so the admin can compare "
        "against a KB-based AI price proposal. **Stateless** — does NOT enroll the "
        "product into continuous monitoring, does NOT write to competitor_sources "
        "or price_history. "
        "However: if the product is already enrolled in monitoring AND its last "
        "refresh is ≤6h old, we return the cached competitor_sources snapshot "
        "(marked `from_monitoring_cache: true`) to save credits. "
        "Refreshing monitoring for continuous tracking is a separate action — see "
        "`POST /start` and `POST /discover`."
    ),
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

    # ── Resolve query + product context ──
    product_id = body.product_id
    product_name: Optional[str] = body.product_name
    dimensions: Optional[str] = body.dimensions
    manufacturer: Optional[str] = body.manufacturer

    if product_id and not product_name:
        prod = (
            sb.table("products")
            .select("id, name, metadata")
            .eq("id", product_id)
            .maybe_single()
            .execute()
        )
        prod_row = (prod.data if prod else None) or {}
        product_name = prod_row.get("name") or product_name
        metadata = prod_row.get("metadata") or {}
        # `metadata.dimensions` is a list[dict] from the PDF ingestion pipeline;
        # flatten to a freeform string so it can be appended to the query.
        raw_dims = (
            dimensions
            or metadata.get("dimensions")
            or metadata.get("size")
            or metadata.get("product_size")
        )
        if isinstance(raw_dims, list) and raw_dims:
            first = raw_dims[0]
            if isinstance(first, dict):
                dimensions = first.get("metric_cm") or first.get("imperial_in") or None
            else:
                dimensions = str(first)
        elif isinstance(raw_dims, dict):
            dimensions = raw_dims.get("metric_cm") or raw_dims.get("imperial_in") or None
        else:
            dimensions = raw_dims
        manufacturer = (
            manufacturer
            or metadata.get("factory_name")
            or metadata.get("manufacturer")
            or metadata.get("brand")
        )

    if not product_name or not product_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either product_id (with a row that has a name) or product_name is required.",
        )

    # ── Country from user profile ──
    profile = (
        sb.table("user_profiles")
        .select("location_country_code, location")
        .eq("user_id", user.id)
        .maybe_single()
        .execute()
    )
    prof_row = (profile.data if profile else None) or {}
    country_code = _resolve_user_country_code(prof_row)

    query_text = product_name.strip()
    if manufacturer and manufacturer.lower() not in query_text.lower():
        query_text = f"{manufacturer} {query_text}".strip()
    if dimensions:
        query_text = f"{query_text} {dimensions}".strip()

    # ── Monitoring cache shortcut ──
    # If this product is already enrolled in monitoring AND the last refresh is
    # fresh (≤6h), return the cached competitor_sources rows rather than
    # spending credits again. Respects the same 6h throttle as /discover.
    if product_id:
        mon = (
            sb.table("price_monitoring_products")
            .select("id, last_claude_search_at")
            .eq("product_id", product_id)
            .eq("user_id", user.id)
            .maybe_single()
            .execute()
        )
        mon_row = (mon.data if mon else None) or None
        if mon_row and mon_row.get("last_claude_search_at"):
            last_at = datetime.fromisoformat(mon_row["last_claude_search_at"].replace("Z", "+00:00"))
            age_s = int((datetime.now(last_at.tzinfo) - last_at).total_seconds())
            if age_s <= 6 * 3600:
                existing = (
                    sb.table("competitor_sources")
                    .select("source_name, source_url, source_type, current_price, current_original_price, current_price_verified, current_currency, current_availability, current_metadata, match_kind, match_score, match_note")
                    .eq("product_id", product_id)
                    .in_("source_type", ["perplexity_web_search", "dataforseo_shopping"])
                    .eq("is_active", True)
                    .order("current_price", desc=False)
                    .limit(20)
                    .execute()
                )
                rows = existing.data or []
                cached_hits: List[PriceHit] = []
                for r in rows:
                    if r.get("current_price") is None:
                        continue
                    meta = r.get("current_metadata") or {}
                    cached_hits.append(PriceHit(
                        retailer_name=r.get("source_name") or "Unknown",
                        product_url=r.get("source_url") or "",
                        price=float(r["current_price"]),
                        original_price=float(r["current_original_price"]) if r.get("current_original_price") is not None else None,
                        verified=bool(r.get("current_price_verified") or False),
                        source="dataforseo" if r.get("source_type") == "dataforseo_shopping" else "perplexity",
                        currency=r.get("current_currency") or "EUR",
                        availability=r.get("current_availability") or "unknown",
                        image_url=meta.get("image_url"),
                        rating_value=meta.get("rating_value"),
                        rating_votes=meta.get("rating_votes"),
                        notes=meta.get("notes"),
                        match_kind=r.get("match_kind"),
                        match_score=r.get("match_score"),
                        match_note=r.get("match_note"),
                        product_title=meta.get("product_title"),
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

    # ── Fresh scan ──
    # When called with a product_id we have structured metadata — use it as
    # the reference facets, no need to pay Haiku to re-decompose.
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

    # Stage A query enrichment: prepend brand + strongest SKU when we have
    # them from catalog facets, so retailer ranking gets the SKU anchor.
    enriched_query = query_text
    if catalog_facets:
        prefix_parts: List[str] = []
        if catalog_facets.brand and normalize_model_token(catalog_facets.brand) not in normalize_model_token(query_text):
            prefix_parts.append(catalog_facets.brand)
        if catalog_facets.sku_tokens:
            prefix_parts.append(catalog_facets.sku_tokens[0])
        if prefix_parts:
            enriched_query = " ".join([*prefix_parts, query_text])

    service = get_perplexity_price_search_service()
    result = await service.search_prices(
        product_name=enriched_query,
        dimensions=None,  # already folded into query_text
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


# Country-name → ISO-3166-1 alpha-2 fallback table for when the user has typed
# "Greece" / "Germany" / etc. into their profile's free-text location field but
# the dedicated location_country_code column was never populated. Narrow on
# purpose — primary markets for the platform. Extend as needed.
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

DEFAULT_COUNTRY_CODE = "GR"  # platform's primary market — Greece


def _resolve_user_country_code(profile: Optional[Dict[str, Any]]) -> str:
    """
    Pick ISO-3166-1 alpha-2 from the user profile, in priority order:
      1. location_country_code if already populated (from earlier normalize pass)
      2. Match the trailing ", XX" pattern in free-text `location` (e.g. "Athens, GR")
      3. Match the free-text against the country-name fallback table
      4. Default GR
    """
    prof = profile or {}
    code = (prof.get("location_country_code") or "").strip().upper()
    if len(code) == 2 and code.isalpha():
        return code

    location_text = (prof.get("location") or "").strip()
    if location_text:
        # Trailing ", GR" / " GR" / ", United Kingdom" patterns
        import re as _re
        trailing_iso = _re.search(r',\s*([A-Za-z]{2})\s*$', location_text)
        if trailing_iso:
            guess = trailing_iso.group(1).upper()
            if guess.isalpha():
                return guess

        # Country name match (whole string or part)
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


@router.post(
    "/tracked-queries/cron-refresh",
    summary="Refresh all due tracked_queries (called by Supabase cron)",
    description=(
        "Iterates tracked_queries where last_refreshed_at + refresh_interval_hours < now(), "
        "runs Perplexity for each, writes to tracked_query_price_history. "
        "Auth: `x-cron-secret` header must match server-side CRON_SECRET. "
        "Not for end-user callers — use POST /api/v1/prices/track/{id}/refresh for that."
    ),
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


class BroadcastApiAnnouncementRequest(BaseModel):
    """Body for POST /broadcast-api-announcement — admin one-shot send."""
    template_slug: str = Field(default="api_broadcast.price_tracking_v6",
        description="Slug of the email template in `email_templates` to use.")
    docs_url: str = Field(default="https://github.com/creativeghq/material-kai-vision-platform/blob/main/docs/api/price-monitoring-api.md")
    api_base_url: str = Field(default="https://v1api.materialshub.gr",
        description="Base URL of the price-tracking API — substituted into the template body.")
    support_email: str = Field(default="support@materialshub.gr")
    dry_run: bool = Field(default=True,
        description="When true (default), returns the recipient list without sending.")


@router.post("/broadcast-api-announcement", response_model=DataResponse)
async def broadcast_api_announcement(
    body: BroadcastApiAnnouncementRequest,
    user: User = Depends(get_current_user),
):
    """
    Admin-only one-shot broadcast to every distinct user_id that owns at
    least one active api_key. Renders the chosen email template against
    each recipient's name + the supplied URLs and dispatches via the
    `email-api` edge function (Resend-backed). Recipients who already got
    this slug are skipped (idempotent within the same template version).
    """
    sb = get_supabase_client().client

    # Recipient list — distinct user_ids that own an active api_key.
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

    # Skip recipients we've already sent this template to. The check uses
    # `email_logs` (the platform's existing email send log). If the table is
    # absent we just allow re-send.
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

    import httpx as _httpx
    with _httpx.Client(timeout=15.0) as client:
        for t in targets:
            try:
                resp = client.post(
                    f"{supabase_url}/functions/v1/email-api",
                    headers={
                        # Both headers required — email-api accepts service-role
                        # via `apikey` (level='secret'); Authorization alone fails.
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


class MatchCorrectionRequest(BaseModel):
    """Body for POST /classifier-correction — admin marks a misclassified row."""
    competitor_source_id: Optional[str] = None
    tracked_query_history_id: Optional[str] = None
    corrected_match_kind: str = Field(..., pattern="^(exact|variant|family|mismatch|unverifiable|should_drop)$")
    correction_note: Optional[str] = None


@router.post("/classifier-correction", response_model=StatusResponse)
async def submit_classifier_correction(
    body: MatchCorrectionRequest,
    user: User = Depends(get_current_user),
):
    """
    Admin feedback on the identity classifier. Writes to `match_corrections`.
    Recent rows from this table are pulled into the classifier system prompt
    on every classify call (cached 5 min) — closes the loop without retraining.
    """
    sb = get_supabase_client().client
    snapshot: Dict[str, Any] = {}
    try:
        if body.competitor_source_id:
            cs = sb.table("competitor_sources").select(
                "product_id, source_name, source_url, match_kind, current_metadata"
            ).eq("id", body.competitor_source_id).maybe_single().execute()
            row = (cs.data if cs else None) or {}
            meta = row.get("current_metadata") or {}
            snapshot = {
                "product_id": row.get("product_id"),
                "competitor_source_id": body.competitor_source_id,
                "retailer_name": row.get("source_name"),
                "product_url": row.get("source_url"),
                "product_title": meta.get("product_title"),
                "original_match_kind": row.get("match_kind"),
                "page_facets": {"product_name": meta.get("product_title")},
            }
        elif body.tracked_query_history_id:
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


class PromoteFamilyRequest(BaseModel):
    """Body for POST /promote-family-row — admin promotes a family/mismatch row to tracked.

    The promoted URL becomes "sticky": the classifier override is honored on
    every future refresh (until the admin demotes it back). The override is
    also written to `match_corrections` so the few-shot classifier loop learns
    from it globally.
    """
    competitor_source_id: Optional[str] = None
    tracked_query_history_id: Optional[str] = None
    override_kind: str = Field(..., pattern="^(exact|variant)$")
    reason: Optional[str] = None


class DemoteFamilyRequest(BaseModel):
    """Body for POST /demote-to-family — undo a previously-applied promotion."""
    competitor_source_id: Optional[str] = None
    tracked_query_history_id: Optional[str] = None
    reason: Optional[str] = None


def _require_admin(user: User) -> None:
    """Promotion endpoints are admin-only — they affect chart + median + alerts."""
    role_resp = (
        get_supabase_client().client.table("user_profiles")
        .select("role_id, roles(name)")
        .eq("id", str(user.id))
        .maybe_single()
        .execute()
    )
    profile = (role_resp.data if role_resp else None) or {}
    role_name = (profile.get("roles") or {}).get("name")
    if role_name not in ("admin", "super_admin"):
        raise HTTPException(status_code=403, detail="admin role required")


@router.post("/promote-family-row", response_model=StatusResponse)
async def promote_family_row(
    body: PromoteFamilyRequest,
    user: User = Depends(get_current_user),
):
    """
    Promote a family/mismatch row to tracked. The override flips the row's
    `match_kind` immediately AND records a sticky promotion so future refreshes
    of the same URL keep the override.

    Pass either `competitor_source_id` (internal product flow) or
    `tracked_query_history_id` (external API tracked-query flow).
    """
    _require_admin(user)
    sb = get_supabase_client().client
    now_iso = datetime.utcnow().isoformat()

    if body.tracked_query_history_id:
        ph = sb.table("tracked_query_price_history").select(
            "tracked_query_id, product_url, retailer_name, match_kind"
        ).eq("id", body.tracked_query_history_id).maybe_single().execute()
        row = (ph.data if ph else None) or {}
        if not row:
            raise HTTPException(status_code=404, detail="history row not found")
        tracked_query_id = row.get("tracked_query_id")
        product_url = row.get("product_url")
        original_kind = row.get("match_kind")
        if not tracked_query_id or not product_url:
            raise HTTPException(status_code=400, detail="row missing tracked_query_id or product_url")

        # Sticky promotion (overrides every future refresh of this URL).
        sb.table("tracked_query_promoted_urls").upsert({
            "tracked_query_id": tracked_query_id,
            "product_url": product_url,
            "override_kind": body.override_kind,
            "reason": body.reason,
            "created_by": str(user.id),
        }, on_conflict="tracked_query_id,product_url").execute()

        # Update the current row + manual_override flag so the chart/UI
        # reflect the change immediately.
        sb.table("tracked_query_price_history").update({
            "match_kind": body.override_kind,
            "match_note": body.reason or "Manually promoted by admin",
            "manual_override": True,
        }).eq("tracked_query_id", tracked_query_id).eq("product_url", product_url).execute()

        # Feed the few-shot classifier loop globally.
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

    elif body.competitor_source_id:
        cs = sb.table("competitor_sources").select(
            "product_id, source_url, source_name, match_kind"
        ).eq("id", body.competitor_source_id).maybe_single().execute()
        row = (cs.data if cs else None) or {}
        if not row:
            raise HTTPException(status_code=404, detail="competitor source not found")
        product_id = row.get("product_id")
        source_url = row.get("source_url")
        original_kind = row.get("match_kind")
        if not product_id or not source_url:
            raise HTTPException(status_code=400, detail="source missing product_id or source_url")

        sb.table("competitor_source_promoted_urls").upsert({
            "product_id": product_id,
            "product_url": source_url,
            "override_kind": body.override_kind,
            "reason": body.reason,
            "created_by": str(user.id),
        }, on_conflict="product_id,product_url").execute()

        sb.table("competitor_sources").update({
            "match_kind": body.override_kind,
            "match_note": body.reason or "Manually promoted by admin",
        }).eq("id", body.competitor_source_id).execute()

        sb.table("match_corrections").insert({
            "product_id": product_id,
            "competitor_source_id": body.competitor_source_id,
            "retailer_name": row.get("source_name"),
            "product_url": source_url,
            "original_match_kind": original_kind,
            "corrected_match_kind": body.override_kind,
            "correction_note": body.reason or "promoted via UI",
            "created_by": str(user.id),
        }).execute()
    else:
        raise HTTPException(status_code=400, detail="competitor_source_id or tracked_query_history_id required")

    return {"success": True, "message": "row promoted"}


@router.post("/demote-to-family", response_model=StatusResponse)
async def demote_to_family(
    body: DemoteFamilyRequest,
    user: User = Depends(get_current_user),
):
    """Undo a prior promotion. The URL goes back to whatever the classifier decides next refresh."""
    _require_admin(user)
    sb = get_supabase_client().client

    if body.tracked_query_history_id:
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
    elif body.competitor_source_id:
        cs = sb.table("competitor_sources").select(
            "product_id, source_url, source_name"
        ).eq("id", body.competitor_source_id).maybe_single().execute()
        row = (cs.data if cs else None) or {}
        if not row:
            raise HTTPException(status_code=404, detail="competitor source not found")
        sb.table("competitor_source_promoted_urls").delete().eq(
            "product_id", row.get("product_id")
        ).eq("product_url", row.get("source_url")).execute()
        sb.table("competitor_sources").update({
            "match_kind": "family",
            "match_note": body.reason or "Demoted to family by admin",
        }).eq("id", body.competitor_source_id).execute()
        sb.table("match_corrections").insert({
            "product_id": row.get("product_id"),
            "competitor_source_id": body.competitor_source_id,
            "retailer_name": row.get("source_name"),
            "product_url": row.get("source_url"),
            "corrected_match_kind": "should_drop",
            "correction_note": body.reason or "demoted via UI",
            "created_by": str(user.id),
        }).execute()
    else:
        raise HTTPException(status_code=400, detail="competitor_source_id or tracked_query_history_id required")

    return {"success": True, "message": "row demoted"}


@router.get("/jobs/{product_id}", response_model=PriceJobsResponse)
async def get_monitoring_jobs(
    product_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    user: User = Depends(get_current_user)
):
    """
    Get monitoring job history for a product.
    """
    try:
        supabase = get_supabase_client()

        response = supabase.client.table("price_monitoring_jobs").select("*").eq(
            "product_id", product_id
        ).eq("user_id", str(user.id)).order("created_at", desc=True).limit(limit).execute()

        return {
            "success": True,
            "jobs": response.data or [],
            "count": len(response.data or [])
        }

    except Exception as e:
        logger.error(f"❌ Failed to get monitoring jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# Per-product result exclusions (catalog flow, 2026-04-28)
# ============================================================================
# Mirrors the external /api/v1/prices/track/{id}/exclude endpoints. Admin-only
# via RLS on `product_excluded_urls`. Excluded URLs/domains are filtered
# before competitor_sources rows get persisted (in the discover flow) and
# excluded from `GET /api/v1/price-monitoring/sources/{product_id}` reads.


class ProductExcludeRequest(BaseModel):
    """Body for POST /products/{product_id}/exclude / /include."""
    url: Optional[str] = Field(
        default=None,
        description="Specific competitor URL to exclude. Either `url` or `domain` is required.",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Whole retailer domain to exclude (wildcard).",
    )
    reason: Optional[str] = Field(default=None, description="Optional audit note.")


class ProductExclusionRow(BaseModel):
    id: str
    url: Optional[str] = None
    domain: Optional[str] = None
    reason: Optional[str] = None
    excluded_at: str


def _normalize_domain(domain: Optional[str]) -> Optional[str]:
    if not domain:
        return None
    d = domain.strip().lower()
    d = d.removeprefix("http://").removeprefix("https://").removeprefix("www.")
    return d.split("/")[0] or None


@router.post(
    "/products/{product_id}/exclude",
    response_model=ProductExclusionRow,
    summary="Exclude a retailer URL or domain from this product's monitoring",
    description=(
        "Marks a URL or whole retailer domain as excluded for THIS product. "
        "The next discover/refresh will not persist hits matching the exclusion, "
        "and `/sources/{product_id}` hides them. Admin-only — enforced by RLS "
        "on `product_excluded_urls`. Idempotent."
    ),
    status_code=status.HTTP_201_CREATED,
)
async def exclude_product_result(
    product_id: str,
    body: ProductExcludeRequest,
    user: User = Depends(get_current_user),
):
    if not body.url and not body.domain:
        raise HTTPException(status_code=400, detail="Either `url` or `domain` is required.")
    sb = get_supabase_client()
    payload = {
        "product_id": product_id,
        "url": body.url.strip() if body.url else None,
        "domain": _normalize_domain(body.domain),
        "reason": body.reason,
        "excluded_by": user.id,
    }
    try:
        res = (
            sb.client.table("product_excluded_urls")
            .upsert(
                payload,
                on_conflict="product_id,url" if body.url else "product_id,domain",
            )
            .execute()
        )
    except Exception as e:
        logger.error(f"product exclusion upsert failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Also flip is_active=false on the matching competitor_sources rows so
    # they disappear from the live retailer list immediately. Future refreshes
    # would re-discover them; the exclusion filter prevents re-persistence.
    try:
        q = sb.client.table("competitor_sources").update({"is_active": False}).eq("product_id", product_id)
        if body.url:
            q = q.eq("source_url", body.url)
        else:
            # Match all rows whose source_url contains the domain. Supabase
            # doesn't expose a regex op via the JS client API, so use ilike.
            q = q.ilike("source_url", f"%{payload['domain']}%")
        q.execute()
    except Exception as e:
        logger.warning(f"competitor_sources deactivate after exclusion failed (non-fatal): {e}")

    row = (res.data or [payload])[0]
    return ProductExclusionRow(
        id=row.get("id", ""),
        url=row.get("url"),
        domain=row.get("domain"),
        reason=row.get("reason"),
        excluded_at=row.get("excluded_at") or datetime.utcnow().isoformat(),
    )


@router.post("/products/{product_id}/include", summary="Undo a previous exclusion")
async def include_product_result(
    product_id: str,
    body: ProductExcludeRequest,
    user: User = Depends(get_current_user),
):
    if not body.url and not body.domain:
        raise HTTPException(status_code=400, detail="Either `url` or `domain` is required.")
    sb = get_supabase_client()
    q = sb.client.table("product_excluded_urls").delete().eq("product_id", product_id)
    if body.url:
        q = q.eq("url", body.url.strip())
    else:
        q = q.eq("domain", _normalize_domain(body.domain))
    res = q.execute()

    # Re-activate any matching competitor_sources row so it reappears.
    try:
        cs = sb.client.table("competitor_sources").update({"is_active": True}).eq("product_id", product_id)
        if body.url:
            cs = cs.eq("source_url", body.url)
        else:
            cs = cs.ilike("source_url", f"%{_normalize_domain(body.domain)}%")
        cs.execute()
    except Exception as e:
        logger.warning(f"competitor_sources reactivate after include failed (non-fatal): {e}")

    return {"success": True, "product_id": product_id, "removed_count": len(res.data or [])}


@router.get(
    "/products/{product_id}/exclusions",
    response_model=List[ProductExclusionRow],
    summary="List every exclusion attached to this product",
)
async def list_product_exclusions(
    product_id: str,
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client()
    res = (
        sb.client.table("product_excluded_urls")
        .select("id, url, domain, reason, excluded_at")
        .eq("product_id", product_id)
        .order("excluded_at", desc=True)
        .execute()
    )
    return [
        ProductExclusionRow(
            id=r["id"],
            url=r.get("url"),
            domain=r.get("domain"),
            reason=r.get("reason"),
            excluded_at=r["excluded_at"],
        )
        for r in (res.data or [])
    ]


# ============================================================================
# On-demand re-verification for catalog products (2026-04-28)
# ============================================================================
# Mirrors POST /api/v1/prices/track/{id}/verify on the external API. Re-runs
# Firecrawl on selected (or all active) competitor_sources URLs and refreshes
# current_price + current_price_verified + current_metadata in place. Does
# NOT call Perplexity / DataForSEO / marketplace adapters — narrower scope
# than /discover, cheaper, faster.


class ProductVerifyRequest(BaseModel):
    """Body for POST /products/{product_id}/verify."""
    urls: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional whitelist of URLs to re-verify. Each must already exist "
            "in this product's competitor_sources. When omitted, every active "
            "row for this product is re-verified."
        ),
    )


@router.post("/products/{product_id}/verify", summary="Re-verify retailer prices on demand (no new discovery)")
async def verify_product_sources(
    product_id: str,
    body: ProductVerifyRequest,
    user: User = Depends(get_current_user),
):
    """
    Re-runs Firecrawl on every (or specified) competitor_sources row for this
    product. Refreshes current_price, current_original_price,
    current_price_verified, current_availability, and current_metadata.notes
    in place — does NOT touch the discovery pipeline. Useful for clearing
    stale 'unverified' flags or refreshing one specific retailer without
    spending the full discovery cost.

    Cost: 1 Firecrawl credit per URL. No LLM cost. ~1-3s per URL (parallel).
    Returns the freshly-updated rows.
    """
    from app.services.integrations.perplexity_price_search_service import (
        get_perplexity_price_search_service, PriceHit,
    )

    sb = get_supabase_client().client
    started = datetime.utcnow()

    # Pull active competitor_sources for the product, optionally narrowed.
    q = (
        sb.table("competitor_sources")
        .select(
            "id, source_name, source_url, source_type, current_price, current_original_price, "
            "current_price_verified, current_currency, current_availability, current_metadata, "
            "match_kind, match_score, match_note"
        )
        .eq("product_id", product_id)
        .eq("is_active", True)
    )
    if body.urls:
        urls = [u.strip() for u in body.urls if u and isinstance(u, str)]
        q = q.in_("source_url", urls)
    rows_resp = q.execute()
    rows = rows_resp.data or []
    if not rows:
        return {
            "success": True,
            "status": "no_results",
            "rows_processed": 0,
            "credits_used": 0,
            "results": [],
            "message": "No active retailer rows to verify for this product."
        }

    # Build PriceHit objects, run Firecrawl verify in parallel, write back.
    hits: List[PriceHit] = []
    row_by_url: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        url = r.get("source_url")
        if not url:
            continue
        row_by_url[url] = r
        meta = r.get("current_metadata") or {}
        hits.append(PriceHit(
            retailer_name=r.get("source_name") or "",
            product_url=url,
            price=float(r["current_price"]) if r.get("current_price") is not None else None,
            original_price=float(r["current_original_price"]) if r.get("current_original_price") is not None else None,
            currency=r.get("current_currency"),
            availability=r.get("current_availability"),
            verified=False,
            source=r.get("source_type") or "perplexity",
            notes=(meta or {}).get("notes"),
            match_kind=r.get("match_kind"),
            match_score=r.get("match_score"),
            match_note=r.get("match_note"),
            product_title=(meta or {}).get("product_title"),
        ))

    svc = get_perplexity_price_search_service()
    try:
        credits_used = await svc._verify_hits_with_firecrawl(
            hits, extractions_out=None,
            user_id=user.id, workspace_id=None, double_read=False,
        )
    except Exception as e:
        logger.exception(f"verify_product_sources failed for {product_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Persist updates back to competitor_sources. Updates current_price,
    # current_original_price, current_price_verified, current_availability,
    # and merges notes into current_metadata.
    now_iso = datetime.utcnow().isoformat()
    updated_rows = []
    for h in hits:
        r = row_by_url.get(h.product_url)
        if not r:
            continue
        meta = r.get("current_metadata") or {}
        if h.notes:
            meta = {**meta, "notes": h.notes}
        update = {
            "current_price": float(h.price) if h.price is not None else None,
            "current_original_price": float(h.original_price) if h.original_price is not None else None,
            "current_price_verified": bool(h.verified),
            "current_availability": h.availability,
            "current_currency": h.currency,
            "current_metadata": meta or None,
            "current_price_updated_at": now_iso,
            "last_seen_at": now_iso,
        }
        try:
            res = sb.table("competitor_sources").update(update).eq("id", r["id"]).execute()
            if res.data:
                updated_rows.append(res.data[0])
        except Exception as e:
            logger.warning(f"competitor_sources update after verify failed (id={r['id']}): {e}")

    latency_ms = int((datetime.utcnow() - started).total_seconds() * 1000)
    return {
        "success": True,
        "status": "verified",
        "rows_processed": len(hits),
        "verified_count": sum(1 for h in hits if h.verified),
        "unverified_count": sum(1 for h in hits if not h.verified),
        "credits_used": int(credits_used or 0),
        "latency_ms": latency_ms,
        "results": updated_rows,
    }

