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


class DiscoverSourcesResponse(BaseModel):
    """Response from /discover. Returns the retailer list + throttle state."""
    success: bool
    source: str = "perplexity_web_search"
    product_id: str
    results: List[PriceHit] = []
    total_results: int = 0
    credits_used: int = 0
    latency_ms: int = 0
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
            .select("source_name, source_url, current_price, current_currency, current_availability, last_seen_at")
            .eq("product_id", product_id)
            .eq("source_type", "perplexity_web_search")
            .eq("is_active", True)
            .order("last_seen_at", desc=True)
            .limit(10)
            .execute()
        )
        rows = existing.data or []
        cached_hits = [
            PriceHit(
                retailer_name=r.get("source_name") or "Unknown",
                product_url=r.get("source_url") or "",
                price=float(r.get("current_price") or 0),
                currency=r.get("current_currency") or "USD",
                availability=r.get("current_availability") or "unknown",
            )
            for r in rows
            if r.get("current_price") is not None
        ]
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
    dimensions = (
        metadata.get("dimensions")
        or metadata.get("size")
        or metadata.get("product_size")
    )

    # ── User's country for regional preference ──
    profile = (
        sb.table("user_profiles")
        .select("location_country_code, location")
        .eq("user_id", user.id)
        .maybe_single()
        .execute()
    )
    prof_row = (profile.data if profile else None) or {}
    country_code = prof_row.get("location_country_code")

    # ── Run Claude web search ──
    result = await service.search_prices(
        product_name=product_name,
        dimensions=dimensions,
        country_code=country_code,
        limit=10,
        user_id=user.id,
        workspace_id=mon_row.get("workspace_id") or (workspace.workspace_id if workspace else None),
    )

    if not result.success:
        return DiscoverSourcesResponse(
            success=False,
            product_id=product_id,
            error=result.error or "Claude search failed",
            credits_used=result.credits_used,
            latency_ms=result.latency_ms,
        )

    # ── Persist: upsert competitor_sources by (product_id, source_url),
    #    then insert price_history rows, then stamp last_claude_search_at.
    #    Hit source maps to competitor_source_type: perplexity → perplexity_web_search,
    #    dataforseo → dataforseo_shopping. So the UI can split them into
    #    "Discovered retailers" vs "Merchants" sections by source_type.
    now_iso = datetime.utcnow().isoformat()
    for hit in result.hits:
        source_type = "dataforseo_shopping" if hit.source == "dataforseo" else "perplexity_web_search"
        upsert_row = {
            "product_id": product_id,
            "source_name": hit.retailer_name,
            "source_url": hit.product_url,
            "source_type": source_type,
            "discovered_via": source_type,
            "auto_discovered": True,
            "is_active": True,
            "current_price": float(hit.price) if hit.price is not None else None,
            "current_currency": hit.currency,
            "current_availability": hit.availability or "unknown",
            "current_price_updated_at": now_iso,
            "last_seen_at": now_iso,
            "last_successful_scrape": now_iso,
            "error_count": 0,
            "last_error": None,
            "created_by": user.id,
        }
        try:
            sb.table("competitor_sources").upsert(upsert_row, on_conflict="product_id,source_url").execute()
        except Exception as e:
            logger.warning(f"Upsert competitor_source failed for {hit.product_url}: {e}")
            continue

        try:
            sb.table("price_history").insert({
                "product_id": product_id,
                "source_name": hit.retailer_name,
                "source_url": hit.product_url,
                "price": float(hit.price) if hit.price is not None else None,
                "currency": hit.currency,
                "availability": hit.availability or "unknown",
                "scraped_at": now_iso,
                "metadata": {
                    "via": source_type,
                    "notes": hit.notes,
                    "image_url": hit.image_url,
                    "rating_value": hit.rating_value,
                    "rating_votes": hit.rating_votes,
                },
            }).execute()
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

    return DiscoverSourcesResponse(
        success=True,
        product_id=product_id,
        results=result.hits,
        total_results=len(result.hits),
        credits_used=result.credits_used,
        latency_ms=result.latency_ms,
        last_search_at=now_iso,
        cached=False,
    )


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




