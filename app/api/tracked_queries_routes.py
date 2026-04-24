"""
Public Price Tracking API — /api/v1/prices/track/*

External projects authenticate with an api_keys Bearer token, register
tracked queries by product name, and control the refresh cadence. Our
cron refreshes on their schedule; they poll the GET endpoints for the
latest results + history.

Deleting the api_key CASCADEs out every tracked query + price history
tied to it (enforced at the DB level).
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.price_lookup_routes import ApiKeyContext, authenticate_api_key
from app.services.integrations.tracked_queries_service import get_tracked_queries_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/prices/track",
    tags=["Price Tracking (Public API)"],
    responses={
        401: {"description": "Invalid or missing API key"},
        403: {"description": "API key does not own this tracking_id"},
        404: {"description": "tracking_id not found"},
        429: {"description": "Rate limit exceeded"},
    },
)


# ────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ────────────────────────────────────────────────────────────────────────────


class CreateTrackRequest(BaseModel):
    search_query: str = Field(..., description="Product name, e.g. 'Ferrara Beige Keros'.")
    dimensions: Optional[str] = Field(
        default=None,
        description="Optional size spec appended to the query (e.g. '60x120').",
    )
    country_code: Optional[str] = Field(
        default=None,
        description="ISO-3166 alpha-2 to bias results toward a market (GR, DE, GB, etc.). Omit for global.",
    )
    manufacturer: Optional[str] = Field(
        default=None,
        description="Optional manufacturer name — included in the query to disambiguate generic product names.",
    )
    preferred_retailer_domains: Optional[List[str]] = Field(
        default=None,
        description="Optional list of domains to prioritize (e.g. ['youbath.gr', 'fshome.gr']). Used as Perplexity search_domain_filter.",
    )
    refresh_interval_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="How often our cron will refresh this query. Min 1h, max 720h (30 days). Default 24h.",
    )


class UpdateTrackRequest(BaseModel):
    refresh_interval_hours: Optional[int] = Field(default=None, ge=1, le=720)
    country_code: Optional[str] = None
    preferred_retailer_domains: Optional[List[str]] = None
    dimensions: Optional[str] = None
    manufacturer: Optional[str] = None


class TrackedQueryResultRow(BaseModel):
    retailer_name: str
    product_url: str
    price: Optional[float] = None
    currency: Optional[str] = None
    price_unit: Optional[str] = None
    availability: Optional[str] = None
    city: Optional[str] = None
    ships_from_abroad: bool = False
    notes: Optional[str] = None
    scraped_at: Optional[str] = None


class TrackedQueryResponse(BaseModel):
    tracking_id: str
    search_query: str
    dimensions: Optional[str] = None
    country_code: Optional[str] = None
    manufacturer: Optional[str] = None
    preferred_retailer_domains: Optional[List[str]] = None
    refresh_interval_hours: int
    last_refreshed_at: Optional[str] = None
    last_error: Optional[str] = None
    is_active: bool = True
    total_credits_used: int = 0
    created_at: str
    # Embedded for convenience on create/refresh responses
    results: Optional[List[TrackedQueryResultRow]] = None
    throttle_until: Optional[str] = None


class RefreshResponse(BaseModel):
    tracking_id: str
    status: str  # 'refreshed' | 'throttled' | 'inactive' | 'error'
    credits_used: int = 0
    latency_ms: int = 0
    results: List[TrackedQueryResultRow] = []
    summary: Optional[str] = None
    throttle_until: Optional[str] = None
    error: Optional[str] = None


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _ensure_owner(tq: Optional[Dict[str, Any]], ctx: ApiKeyContext) -> Dict[str, Any]:
    """404 if missing, 403 if the tracking_id wasn't created by this api_key."""
    if not tq:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="tracking_id not found")
    if tq.get("api_key_id") != ctx.api_key_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This api_key does not own this tracking_id.",
        )
    return tq


def _to_response(
    tq: Dict[str, Any], results: Optional[List[Dict[str, Any]]] = None
) -> TrackedQueryResponse:
    return TrackedQueryResponse(
        tracking_id=tq["id"],
        search_query=tq.get("search_query") or "",
        dimensions=tq.get("dimensions"),
        country_code=tq.get("country_code"),
        manufacturer=tq.get("manufacturer"),
        preferred_retailer_domains=tq.get("preferred_retailer_domains"),
        refresh_interval_hours=int(tq.get("refresh_interval_hours") or 24),
        last_refreshed_at=tq.get("last_refreshed_at"),
        last_error=tq.get("last_error"),
        is_active=bool(tq.get("is_active", True)),
        total_credits_used=int(tq.get("total_credits_used") or 0),
        created_at=tq.get("created_at") or datetime.now(timezone.utc).isoformat(),
        results=[TrackedQueryResultRow(**r) for r in (results or [])] if results else None,
    )


# ────────────────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=TrackedQueryResponse,
    summary="Register a new price-tracked query (runs first refresh immediately)",
    description=(
        "Create a tracked query and run the first Perplexity search synchronously "
        "so the caller gets initial results in the response. Subsequent refreshes "
        "happen on our cron at `refresh_interval_hours` cadence. Deleting the api_key "
        "used to create this tracking_id will CASCADE-delete the query and all its "
        "price history — there's no way to transfer ownership."
    ),
    status_code=status.HTTP_201_CREATED,
)
async def create_tracked_query(
    body: CreateTrackRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
) -> TrackedQueryResponse:
    service = get_tracked_queries_service()
    created = await service.create(
        api_key_id=ctx.api_key_id,
        user_id=ctx.user_id,
        workspace_id=ctx.workspace_id,
        search_query=body.search_query,
        dimensions=body.dimensions,
        country_code=body.country_code,
        manufacturer=body.manufacturer,
        preferred_retailer_domains=body.preferred_retailer_domains,
        refresh_interval_hours=body.refresh_interval_hours,
    )
    results = await service.latest_results(created["id"])
    return _to_response(created, results)


@router.get(
    "",
    response_model=List[TrackedQueryResponse],
    summary="List all tracked queries owned by this api_key",
)
async def list_tracked_queries(
    include_inactive: bool = False,
    limit: int = 100,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
) -> List[TrackedQueryResponse]:
    service = get_tracked_queries_service()
    rows = await service.list_for_api_key(ctx.api_key_id, include_inactive=include_inactive, limit=limit)
    return [_to_response(r) for r in rows]


@router.get(
    "/{tracking_id}",
    response_model=TrackedQueryResponse,
    summary="Get one tracked query + its latest results",
)
async def get_tracked_query(
    tracking_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
) -> TrackedQueryResponse:
    service = get_tracked_queries_service()
    tq = _ensure_owner(await service.get(tracking_id), ctx)
    results = await service.latest_results(tracking_id)
    return _to_response(tq, results)


@router.get(
    "/{tracking_id}/history",
    response_model=List[TrackedQueryResultRow],
    summary="Get the full price history for a tracked query",
)
async def get_tracked_query_history(
    tracking_id: str,
    limit: int = 500,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
) -> List[TrackedQueryResultRow]:
    service = get_tracked_queries_service()
    tq = _ensure_owner(await service.get(tracking_id), ctx)
    history = await service.history(tracking_id, limit=limit)
    return [TrackedQueryResultRow(**row) for row in history]


@router.put(
    "/{tracking_id}",
    response_model=TrackedQueryResponse,
    summary="Update cadence / country / preferred retailers",
)
async def update_tracked_query(
    tracking_id: str,
    body: UpdateTrackRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
) -> TrackedQueryResponse:
    service = get_tracked_queries_service()
    _ensure_owner(await service.get(tracking_id), ctx)
    updated = await service.update(
        tracking_id,
        refresh_interval_hours=body.refresh_interval_hours,
        country_code=body.country_code,
        preferred_retailer_domains=body.preferred_retailer_domains,
        dimensions=body.dimensions,
        manufacturer=body.manufacturer,
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="update failed")
    return _to_response(updated)


@router.post(
    "/{tracking_id}/refresh",
    response_model=RefreshResponse,
    summary="Force a refresh now (bypasses the cadence)",
    description=(
        "Run Perplexity immediately regardless of when the last refresh happened. "
        "Useful when the external project wants prices *now*. Respect per-key "
        "rate limits — default 60 req/min."
    ),
)
async def refresh_tracked_query(
    tracking_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
) -> RefreshResponse:
    service = get_tracked_queries_service()
    _ensure_owner(await service.get(tracking_id), ctx)
    outcome = await service.refresh(tracking_id, force=True)
    return RefreshResponse(
        tracking_id=tracking_id,
        status=outcome.get("status", "unknown"),
        credits_used=int(outcome.get("credits_used", 0) or 0),
        latency_ms=int(outcome.get("latency_ms", 0) or 0),
        results=[TrackedQueryResultRow(**r) for r in (outcome.get("results") or [])],
        summary=outcome.get("summary"),
        throttle_until=outcome.get("throttle_until"),
        error=outcome.get("error"),
    )


@router.delete(
    "/{tracking_id}",
    summary="Stop tracking (soft delete — history is preserved)",
    description=(
        "Marks the tracked query inactive so cron stops refreshing it, but keeps "
        "the row and its price history. To permanently wipe everything, delete the "
        "api_key that created it — that CASCADEs out the tracked_query and all "
        "its history rows."
    ),
)
async def delete_tracked_query(
    tracking_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
) -> Dict[str, Any]:
    service = get_tracked_queries_service()
    _ensure_owner(await service.get(tracking_id), ctx)
    ok = await service.deactivate(tracking_id)
    return {"success": bool(ok), "tracking_id": tracking_id, "is_active": False}
