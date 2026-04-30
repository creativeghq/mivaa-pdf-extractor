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
    search_query: str = Field(
        ...,
        description=(
            "Product identity string. RECOMMENDED format: '{ProductName} {Model/Series} {SKU}' "
            "concatenated together — e.g. 'ORABELLA PRECIOSA 10202 Modern Chrome Single Lever Basin "
            "Mixer'. The SKU is the strongest disambiguator: when present, our identity classifier "
            "drops sibling SKUs in the same series (e.g. shower outlets when you asked for the basin "
            "mixer). Brand-only or series-only queries (no SKU) work too but return a wider set of "
            "retailers and may include sibling SKUs flagged as 'family'."
        ),
    )
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
    verify_prices: bool = Field(
        default=True,
        description=(
            "When true (default), every refresh re-fetches each retailer URL via Firecrawl to "
            "confirm the price on the live page. Set false to skip verification (faster, cheaper, "
            "but prices may be stale / hallucinated)."
        ),
    )
    alert_channels: Optional[List[str]] = Field(
        default=None,
        description=(
            "Delivery channels for price alerts. Allowed values: 'bell', 'email', 'webhook'. "
            "Bell + webhook are free; email costs 1 credit per send. Default ['bell']."
        ),
    )
    alert_on_price_drop: Optional[bool] = Field(
        default=None,
        description="Fire an alert when the trailing 7-day median drops ≥10% W/W (per retailer).",
    )
    alert_on_new_retailer: Optional[bool] = Field(
        default=None,
        description="Fire an alert when discovery surfaces a retailer domain we have never tracked.",
    )
    alert_on_promo: Optional[bool] = Field(
        default=None,
        description="Fire an alert when original_price becomes non-null on a row that previously had it null.",
    )
    alert_webhook_url: Optional[str] = Field(
        default=None,
        description=(
            "Per-tracked-query webhook destination. Required when 'webhook' is in alert_channels. "
            "Receives POST {alert_type, title, body, retailer_name, retailer_domain, payload, fired_at}. "
            "24h dedupe per (alert_type, retailer_domain)."
        ),
    )


class UpdateTrackRequest(BaseModel):
    refresh_interval_hours: Optional[int] = Field(default=None, ge=1, le=720)
    country_code: Optional[str] = None
    preferred_retailer_domains: Optional[List[str]] = None
    dimensions: Optional[str] = None
    manufacturer: Optional[str] = None
    verify_prices: Optional[bool] = None
    alert_channels: Optional[List[str]] = None
    alert_on_price_drop: Optional[bool] = None
    alert_on_new_retailer: Optional[bool] = None
    alert_on_promo: Optional[bool] = None
    alert_webhook_url: Optional[str] = None


class TrackedQueryResultRow(BaseModel):
    retailer_name: str
    product_url: str
    price: Optional[float] = None
    original_price: Optional[float] = None  # on-page "was" price (promo)
    currency: Optional[str] = None
    price_unit: Optional[str] = None
    availability: Optional[str] = None
    city: Optional[str] = None
    ships_from_abroad: bool = False
    verified: bool = False  # True when Firecrawl confirmed the price from the live page
    notes: Optional[str] = None
    scraped_at: Optional[str] = None
    # Product-identity verification. Nullable on legacy rows.
    match_kind: Optional[str] = None       # 'exact' | 'variant' | 'unverifiable'
    match_score: Optional[int] = None      # 0-100
    match_note: Optional[str] = None       # e.g. 'Color differs: asked BLACK MATT, page shows WHITE'
    product_title: Optional[str] = None    # Exact name shown on retailer page. Disambiguates multiple rows from the same retailer (different variants).
    # Sanity-band fields. Nullable on legacy rows.
    is_anomaly: Optional[bool] = None
    anomaly_reason: Optional[str] = None
    rolling_median_at_check: Optional[float] = None


class TrackedQueryResponse(BaseModel):
    tracking_id: str
    search_query: str
    dimensions: Optional[str] = None
    country_code: Optional[str] = None
    manufacturer: Optional[str] = None
    preferred_retailer_domains: Optional[List[str]] = None
    refresh_interval_hours: int
    verify_prices: bool = True
    last_refreshed_at: Optional[str] = None
    last_error: Optional[str] = None
    is_active: bool = True
    total_credits_used: int = 0
    created_at: str
    # Alert opt-ins
    alert_channels: Optional[List[str]] = None
    alert_on_price_drop: bool = False
    alert_on_new_retailer: bool = False
    alert_on_promo: bool = False
    alert_webhook_url: Optional[str] = None
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
        verify_prices=bool(tq.get("verify_prices", True)),
        last_refreshed_at=tq.get("last_refreshed_at"),
        last_error=tq.get("last_error"),
        is_active=bool(tq.get("is_active", True)),
        total_credits_used=int(tq.get("total_credits_used") or 0),
        created_at=tq.get("created_at") or datetime.now(timezone.utc).isoformat(),
        alert_channels=tq.get("alert_channels"),
        alert_on_price_drop=bool(tq.get("alert_on_price_drop") or False),
        alert_on_new_retailer=bool(tq.get("alert_on_new_retailer") or False),
        alert_on_promo=bool(tq.get("alert_on_promo") or False),
        alert_webhook_url=tq.get("alert_webhook_url"),
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
        verify_prices=body.verify_prices,
        alert_channels=body.alert_channels,
        alert_on_price_drop=body.alert_on_price_drop,
        alert_on_new_retailer=body.alert_on_new_retailer,
        alert_on_promo=body.alert_on_promo,
        alert_webhook_url=body.alert_webhook_url,
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
        verify_prices=body.verify_prices,
        alert_channels=body.alert_channels,
        alert_on_price_drop=body.alert_on_price_drop,
        alert_on_new_retailer=body.alert_on_new_retailer,
        alert_on_promo=body.alert_on_promo,
        alert_webhook_url=body.alert_webhook_url,
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


# ────────────────────────────────────────────────────────────────────────────
# Per-query result exclusions
# ────────────────────────────────────────────────────────────────────────────


class ExcludeRequest(BaseModel):
    url: Optional[str] = Field(
        default=None,
        description="Specific product URL to exclude. Either `url` or `domain` is required.",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Whole retailer domain to exclude (wildcard). Either `url` or `domain` is required.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Optional free-text note (e.g. 'wrong SKU', 'aggregator noise'). Stored for audit.",
    )


class ExclusionRow(BaseModel):
    id: str
    url: Optional[str] = None
    domain: Optional[str] = None
    reason: Optional[str] = None
    excluded_at: str


@router.post(
    "/{tracking_id}/exclude",
    response_model=ExclusionRow,
    summary="Exclude a result (URL or domain) from this tracked query",
    description=(
        "Mark a URL or whole retailer domain as excluded for THIS tracking_id. "
        "The next refresh will not persist hits matching the exclusion, and "
        "GET /track/{id} / /history will hide them by default. Other tracked "
        "queries (yours or other consumers') are unaffected — exclusions are "
        "scoped, never global. Idempotent: re-excluding the same URL updates "
        "the reason but doesn't error."
    ),
    status_code=status.HTTP_201_CREATED,
)
async def exclude_result(
    tracking_id: str,
    body: ExcludeRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
) -> ExclusionRow:
    if not body.url and not body.domain:
        raise HTTPException(status_code=400, detail="Either `url` or `domain` is required.")
    service = get_tracked_queries_service()
    _ensure_owner(await service.get(tracking_id), ctx)
    row = await service.add_exclusion(
        tracking_id,
        url=body.url,
        domain=body.domain,
        reason=body.reason,
        api_key_id=ctx.api_key_id,
    )
    return ExclusionRow(
        id=row["id"],
        url=row.get("url"),
        domain=row.get("domain"),
        reason=row.get("reason"),
        excluded_at=row.get("excluded_at") or datetime.now(timezone.utc).isoformat(),
    )


@router.post(
    "/{tracking_id}/include",
    summary="Undo a previous exclusion (re-include a URL or domain)",
    description=(
        "Removes an exclusion entry. Future refreshes will surface results matching "
        "the URL/domain again. Returns `{success, removed_count}` — `removed_count=0` "
        "means there was no matching exclusion to remove."
    ),
)
async def include_result(
    tracking_id: str,
    body: ExcludeRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
) -> Dict[str, Any]:
    if not body.url and not body.domain:
        raise HTTPException(status_code=400, detail="Either `url` or `domain` is required.")
    service = get_tracked_queries_service()
    _ensure_owner(await service.get(tracking_id), ctx)
    removed = await service.remove_exclusion(tracking_id, url=body.url, domain=body.domain)
    return {"success": True, "tracking_id": tracking_id, "removed_count": removed}


@router.get(
    "/{tracking_id}/exclusions",
    response_model=List[ExclusionRow],
    summary="List every exclusion on this tracked query",
)
async def list_exclusions(
    tracking_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
) -> List[ExclusionRow]:
    service = get_tracked_queries_service()
    _ensure_owner(await service.get(tracking_id), ctx)
    rows = await service.list_exclusions(tracking_id)
    return [
        ExclusionRow(
            id=r["id"],
            url=r.get("url"),
            domain=r.get("domain"),
            reason=r.get("reason"),
            excluded_at=r["excluded_at"],
        )
        for r in rows
    ]


# ────────────────────────────────────────────────────────────────────────────
# On-demand re-verification
# ────────────────────────────────────────────────────────────────────────────


class VerifyRequest(BaseModel):
    urls: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional whitelist of URLs to re-verify. Each URL must already "
            "exist in the latest refresh run for this tracking_id. When "
            "omitted, every URL in the latest run is re-verified."
        ),
    )


class VerifyResponse(BaseModel):
    tracking_id: str
    status: str  # 'verified' | 'no_results' | 'no_match' | 'inactive' | 'error'
    rows_processed: int = 0
    verified_count: int = 0
    unverified_count: int = 0
    credits_used: int = 0
    latency_ms: int = 0
    results: List[TrackedQueryResultRow] = []
    error: Optional[str] = None


@router.post(
    "/{tracking_id}/verify",
    response_model=VerifyResponse,
    summary="Re-verify the latest results — refreshes verified flag + price by re-scraping each URL",
    description=(
        "Re-runs Firecrawl verification on every URL (or just the URLs you "
        "specify) in the latest refresh run for this tracked query. **Does "
        "not** re-run discovery — no new retailers will be added. Use this "
        "to clear stale `verified=false` flags and refresh prices on rows "
        "you suspect changed without doing a full new discovery.\n\n"
        "Cost: 1 Firecrawl credit per URL re-verified. No LLM cost. Counts "
        "toward your `total_credits_used` accumulator. ~1-3s per URL "
        "(parallel). Returns the freshly-updated rows.\n\n"
        "Differences vs `/refresh`:\n"
        "- `/refresh` runs Perplexity + DataForSEO + marketplaces (full discovery + classifier + verify).\n"
        "- `/verify` only re-runs Firecrawl on URLs you already track. Cheaper, faster, narrower scope."
    ),
)
async def verify_tracked_query(
    tracking_id: str,
    body: VerifyRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
) -> VerifyResponse:
    service = get_tracked_queries_service()
    _ensure_owner(await service.get(tracking_id), ctx)
    outcome = await service.reverify(tracking_id, urls=body.urls)
    return VerifyResponse(
        tracking_id=tracking_id,
        status=outcome.get("status", "unknown"),
        rows_processed=int(outcome.get("rows_processed", 0) or 0),
        verified_count=int(outcome.get("verified_count", 0) or 0),
        unverified_count=int(outcome.get("unverified_count", 0) or 0),
        credits_used=int(outcome.get("credits_used", 0) or 0),
        latency_ms=int(outcome.get("latency_ms", 0) or 0),
        results=[TrackedQueryResultRow(**r) for r in (outcome.get("results") or [])],
        error=outcome.get("error"),
    )
