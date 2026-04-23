"""
Public Price Lookup API — `POST /api/v1/prices/lookup`

External callers authenticate with a key from the `api_keys` table
(`Authorization: Bearer <key>`), pass a URL (and optional product name hint),
and get back structured price data.

Billing: debited from the API key owner's workspace via the shared
FirecrawlClient + AICallLogger path. Rate limit: per-key sliding window
backed by `price_lookups` row count.

This is a one-shot lookup — it does NOT create a `competitor_sources` row.
For ongoing monitoring, users still go through the price monitoring flow.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field, HttpUrl

from app.models.extraction import PriceExtraction
from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.firecrawl_client import get_firecrawl_client
from app.utils.price_parsing import parse_price

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/prices",
    tags=["Price Lookup (Public API)"],
    responses={
        401: {"description": "Invalid or missing API key"},
        403: {"description": "API key lacks access to this endpoint"},
        429: {"description": "Rate limit exceeded"},
    },
)

ENDPOINT_PATH = "/api/v1/prices/lookup"
DEFAULT_RATE_LIMIT_PER_MIN = 60
MAX_RATE_LIMIT_PER_MIN = 600


# ────────────────────────────────────────────────────────────────────────────
# Auth context + dependency
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class ApiKeyContext:
    api_key_id: str
    user_id: Optional[str]
    workspace_id: Optional[str]
    rate_limit_per_min: int


async def authenticate_api_key(
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> ApiKeyContext:
    """
    Validate Authorization: Bearer <key> against the api_keys table.
    Enforces is_active, expires_at, and allowed_endpoints. Resolves the
    caller's workspace via workspace_members for billing.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization: Bearer <api_key> header",
        )

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Empty API key")

    sb = get_supabase_client().client  # sync client — this runs in request hot path
    row = (
        sb.table("api_keys")
        .select("id, user_id, is_active, expires_at, allowed_endpoints, rate_limit_override")
        .eq("api_key", token)
        .maybe_single()
        .execute()
    )
    key = (row.data if row else None) or {}

    if not key or not key.get("is_active"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    expires_at = key.get("expires_at")
    if expires_at and datetime.fromisoformat(expires_at.replace("Z", "+00:00")) < datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key expired")

    # allowed_endpoints: None means allow-all. A non-empty list must include
    # the lookup endpoint path (or a trailing-wildcard prefix match).
    allowed = key.get("allowed_endpoints")
    if allowed:
        if not _endpoint_allowed(ENDPOINT_PATH, allowed):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This API key does not permit access to /api/v1/prices/lookup",
            )

    user_id = key.get("user_id")
    workspace_id = _resolve_workspace_for_user(sb, user_id) if user_id else None

    # Touch last_used_at (fire and forget — failure here shouldn't block the request)
    try:
        sb.table("api_keys").update({"last_used_at": datetime.now(timezone.utc).isoformat()}).eq(
            "id", key["id"]
        ).execute()
    except Exception as e:
        logger.debug(f"Could not update last_used_at for api_key {key['id']}: {e}")

    rate_limit = int(key.get("rate_limit_override") or DEFAULT_RATE_LIMIT_PER_MIN)
    rate_limit = max(1, min(rate_limit, MAX_RATE_LIMIT_PER_MIN))

    return ApiKeyContext(
        api_key_id=key["id"],
        user_id=user_id,
        workspace_id=workspace_id,
        rate_limit_per_min=rate_limit,
    )


def _endpoint_allowed(path: str, allowed_patterns: list) -> bool:
    """Exact match or trailing-`*` prefix match."""
    for pat in allowed_patterns or []:
        if not isinstance(pat, str):
            continue
        if pat == path:
            return True
        if pat.endswith("*") and path.startswith(pat[:-1]):
            return True
    return False


def _resolve_workspace_for_user(sb, user_id: str) -> Optional[str]:
    """Pick the user's primary active workspace. Oldest membership wins."""
    try:
        res = (
            sb.table("workspace_members")
            .select("workspace_id, joined_at, status")
            .eq("user_id", user_id)
            .eq("status", "active")
            .order("joined_at", desc=False)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        return rows[0]["workspace_id"] if rows else None
    except Exception as e:
        logger.warning(f"Could not resolve workspace for user {user_id}: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ────────────────────────────────────────────────────────────────────────────


class PriceLookupRequest(BaseModel):
    url: HttpUrl = Field(
        ...,
        description="Product page URL to scrape.",
        examples=["https://example.com/products/oak-flooring"],
    )
    product_name: Optional[str] = Field(
        default=None,
        description="Optional product name hint — helps the extractor pick the main price when the page lists related items.",
    )
    use_javascript_render: bool = Field(
        default=False,
        description="Set true for JS-heavy / single-page-app sites. Slower and costs more credits.",
    )


class PriceLookupResponse(BaseModel):
    success: bool
    price: Optional[float] = None
    currency: Optional[str] = None
    availability: Optional[str] = None
    shipping_cost: Optional[str] = None
    product_name: Optional[str] = None
    scraped_at: Optional[str] = None
    credits_used: int = 0
    latency_ms: int = 0
    source: str = "firecrawl_url"
    error: Optional[str] = None


# ────────────────────────────────────────────────────────────────────────────
# Endpoint
# ────────────────────────────────────────────────────────────────────────────


@router.post(
    "/lookup",
    response_model=PriceLookupResponse,
    summary="One-shot price lookup from a URL",
    description=(
        "Scrape a product page and return structured price data. Authenticate with "
        "`Authorization: Bearer <api_key>` from an `api_keys` row you own. Billed "
        "against your workspace's Firecrawl credits. Does NOT create a monitoring "
        "subscription — for ongoing tracking, use the price monitoring flow."
    ),
)
async def lookup_price(
    body: PriceLookupRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
) -> PriceLookupResponse:
    sb = get_supabase_client().client

    # ── Rate limit: sliding 60s window ──
    since = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
    recent = (
        sb.table("price_lookups")
        .select("id", count="exact")
        .eq("api_key_id", ctx.api_key_id)
        .gte("created_at", since)
        .execute()
    )
    used = recent.count or 0
    if used >= ctx.rate_limit_per_min:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {ctx.rate_limit_per_min} requests/min",
            headers={"Retry-After": "60"},
        )

    # ── Scrape ──
    firecrawl = get_firecrawl_client()
    prompt = (
        f"Extract the current price, currency, and availability status for the product "
        f"'{body.product_name}'. Use the main product price, not related items."
        if body.product_name
        else "Extract the current price of the main product on this page, not related items or strike-through prices."
    )

    result = await firecrawl.scrape(
        url=str(body.url),
        extraction_model=PriceExtraction,
        user_id=ctx.user_id or "anonymous",
        workspace_id=ctx.workspace_id,
        extraction_prompt=prompt,
        use_javascript_render=body.use_javascript_render,
    )

    response = _build_response(result, body)
    _log_lookup(sb, ctx, body, result, response)
    return response


def _build_response(result, body: PriceLookupRequest) -> PriceLookupResponse:
    if not result.success:
        return PriceLookupResponse(
            success=False,
            error=result.error or "scrape failed",
            credits_used=result.credits_used,
            latency_ms=result.latency_ms,
        )

    extracted = result.data
    amount, currency = parse_price(
        extracted.price if extracted else None,
        hint_currency=extracted.currency if extracted else None,
    )

    return PriceLookupResponse(
        success=True,
        price=float(amount) if amount is not None else None,
        currency=currency,
        availability=extracted.availability if extracted else None,
        shipping_cost=extracted.shipping_cost if extracted else None,
        product_name=extracted.product_name if extracted else None,
        scraped_at=datetime.now(timezone.utc).isoformat(),
        credits_used=result.credits_used,
        latency_ms=result.latency_ms,
    )


def _log_lookup(
    sb,
    ctx: ApiKeyContext,
    body: PriceLookupRequest,
    result,
    response: PriceLookupResponse,
) -> None:
    """Write the lookup row. Failure here must not fail the request."""
    try:
        sb.table("price_lookups").insert(
            {
                "api_key_id": ctx.api_key_id,
                "user_id": ctx.user_id,
                "workspace_id": ctx.workspace_id,
                "source": "firecrawl_url",
                "url": str(body.url),
                "product_name_input": body.product_name,
                "success": response.success,
                "price": response.price,
                "currency": response.currency,
                "availability": response.availability,
                "product_name_extracted": response.product_name,
                "shipping_cost": response.shipping_cost,
                "credits_used": response.credits_used,
                "latency_ms": response.latency_ms,
                "error_message": response.error,
                "use_javascript_render": body.use_javascript_render,
                "raw_extract": (result.raw_extract or {}) if result.success else None,
            }
        ).execute()
    except Exception as e:
        logger.warning(f"Failed to log price_lookup row for api_key {ctx.api_key_id}: {e}")
