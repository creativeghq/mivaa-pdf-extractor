"""
Public Tools API — backs the `/tools` lead-gen page.

Authentication is optional but changes the billing model:

  • Anonymous (no Authorization header) — 2 free scans / day per IP, captcha-
    gated, cache-shielded.
  • Authenticated (Bearer JWT) — debits `SCAN_CREDIT_COST` credits per scan
    from `user_credits.balance`, no daily cap, captcha still required.
    Cache hits do NOT debit. Failed/no-result scans are refunded.

Endpoints (all stateless — do NOT write to tracked_queries / tracked_mentions):

  POST /api/v1/public/price-scan       — turnstile_token + product_name(+facets)
  POST /api/v1/public/mention-scan     — turnstile_token + subject_label
  GET  /api/v1/public/quota            — quota + balance + turnstile_site_key
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.perplexity_price_search_service import (
    get_perplexity_price_search_service,
    PriceHit as InternalPriceHit,
)
from app.services.integrations.mention_identity_service import SubjectFacets
from app.services.integrations.mention_search_service import MentionSearchService
from app.services.integrations.platform_secret_resolver import resolve_secret
from app.services.integrations.public_lookup_service import (
    ANONYMOUS_DAILY_QUOTA,
    QuotaStatus,
    check_quota,
    log_scan,
    normalize_query,
    query_hash,
    read_cache,
    write_cache,
)
from app.services.integrations.turnstile_verifier import verify_token

logger = logging.getLogger(__name__)

# Credits debited per authenticated scan. Mirrors the partner-API operation
# cost (price/mention refresh = 5 credits in CLAUDE.md). Anonymous visitors
# pay nothing — they hit the 2/day cap instead.
SCAN_CREDIT_COST = 5

router = APIRouter(
    prefix="/api/v1/public",
    tags=["Public Tools"],
)


# ============================================================================
# Request / Response models
# ============================================================================

class PublicPriceScanRequest(BaseModel):
    turnstile_token: str = Field(..., min_length=1, max_length=4096)
    product_name: str = Field(..., min_length=2, max_length=200)
    manufacturer: Optional[str] = Field(default=None, max_length=120)
    dimensions: Optional[str] = Field(default=None, max_length=80)
    country_code: Optional[str] = Field(default=None, max_length=2)


class PublicPriceResult(BaseModel):
    retailer_name: str
    product_url: str
    price: Optional[float] = None
    original_price: Optional[float] = None
    currency: Optional[str] = None
    availability: Optional[str] = None
    verified: bool = False
    source: Optional[str] = None
    product_title: Optional[str] = None
    match_kind: Optional[str] = None


class PublicMarketStats(BaseModel):
    count: int
    verified_count: int
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    currency: Optional[str] = None


class PublicPriceScanResponse(BaseModel):
    success: bool
    query: str
    country_code: Optional[str] = None
    results: List[PublicPriceResult] = []
    stats: PublicMarketStats
    summary: Optional[str] = None
    from_cache: bool = False
    quota: "PublicQuotaResponse"
    error: Optional[str] = None


class PublicMentionScanRequest(BaseModel):
    turnstile_token: str = Field(..., min_length=1, max_length=4096)
    subject_label: str = Field(..., min_length=2, max_length=200)
    aliases: Optional[List[str]] = Field(default=None, max_length=10)
    country_code: Optional[str] = Field(default=None, max_length=2)


class PublicMentionResult(BaseModel):
    url: str
    title: Optional[str] = None
    excerpt: Optional[str] = None
    outlet_domain: Optional[str] = None
    outlet_name: Optional[str] = None
    published_at: Optional[str] = None
    source: Optional[str] = None
    language_code: Optional[str] = None
    country_code: Optional[str] = None


class PublicMentionScanResponse(BaseModel):
    success: bool
    subject_label: str
    country_code: Optional[str] = None
    results: List[PublicMentionResult] = []
    total_results: int = 0
    top_outlets: List[dict] = []
    from_cache: bool = False
    quota: "PublicQuotaResponse"
    error: Optional[str] = None


class PublicQuotaResponse(BaseModel):
    # Anonymous billing surface (still populated for signed-in users so the
    # frontend can show "X free scans converted into Y credits" if it wants):
    used: int
    remaining: int
    limit: int
    reset_at: str
    turnstile_site_key: Optional[str] = None
    is_authenticated: bool = False
    # Signed-in billing surface — Null for anonymous callers.
    credits_balance: Optional[int] = None
    credits_per_scan: int = 0


PublicPriceScanResponse.model_rebuild()
PublicMentionScanResponse.model_rebuild()


# ============================================================================
# Helpers
# ============================================================================

def _extract_ip(request: Request) -> Optional[str]:
    """Resolve client IP, honoring CF / proxy headers."""
    fwd = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


def _resolve_user_id(request: Request) -> Optional[str]:
    """Best-effort user_id extraction from the Authorization header.

    Public endpoint — no enforcement. If JWT is malformed we silently treat
    the visitor as anonymous and key the quota on IP."""
    auth = request.headers.get("authorization") or ""
    if not auth.lower().startswith("bearer "):
        return None
    token = auth.split(" ", 1)[1].strip()
    if not token:
        return None
    try:
        sb = get_supabase_client().client
        resp = sb.auth.get_user(token)
        if resp and resp.user:
            return str(resp.user.id)
    except Exception as e:
        logger.debug(f"public-tools: optional auth lookup failed: {e}")
    return None


def _read_credit_balance(user_id: str) -> Optional[int]:
    """Return the user's credit balance, or None if no row exists."""
    sb = get_supabase_client().client
    try:
        resp = (
            sb.table("user_credits")
            .select("balance")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        if resp and resp.data:
            return int(resp.data.get("balance") or 0)
    except Exception as e:
        logger.warning(f"public-tools: credit balance read failed for {user_id}: {e}")
    return 0


def _debit_credits(user_id: str, *, operation_type: str, qhash: str, scan_type: str) -> tuple[bool, Optional[int], Optional[str]]:
    """Call the debit_user_credits RPC. Returns (success, new_balance, error)."""
    sb = get_supabase_client().client
    try:
        resp = sb.rpc("debit_user_credits", {
            "p_user_id": user_id,
            "p_amount": SCAN_CREDIT_COST,
            "p_operation_type": operation_type,
            "p_description": f"Public {scan_type} scan",
            "p_metadata": {"query_hash": qhash, "scan_type": scan_type},
        }).execute()
        row = (resp.data or [None])[0]
        if not row:
            return False, None, "no_response"
        success = bool(row.get("success"))
        return success, (int(row["new_balance"]) if row.get("new_balance") is not None else None), row.get("error_message")
    except Exception as e:
        logger.warning(f"public-tools: debit_user_credits failed: {e}")
        return False, None, str(e)[:200]


def _quota_to_response(
    q: QuotaStatus,
    *,
    is_authenticated: bool,
    credits_balance: Optional[int] = None,
) -> PublicQuotaResponse:
    site_key = resolve_secret("TURNSTILE_SITE_KEY").value
    return PublicQuotaResponse(
        used=q.used,
        remaining=q.remaining,
        limit=q.limit,
        reset_at=q.reset_at.astimezone(timezone.utc).isoformat(),
        turnstile_site_key=site_key,
        is_authenticated=is_authenticated,
        credits_balance=credits_balance,
        credits_per_scan=SCAN_CREDIT_COST if is_authenticated else 0,
    )


def _compute_stats(hits: List[InternalPriceHit]) -> PublicMarketStats:
    priced = [h for h in hits if h.price is not None]
    if not priced:
        return PublicMarketStats(count=0, verified_count=0)
    values = sorted([float(h.price) for h in priced])
    n = len(values)
    median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
    currencies = [h.currency for h in priced if h.currency]
    currency = max(set(currencies), key=currencies.count) if currencies else None
    return PublicMarketStats(
        count=n,
        verified_count=sum(1 for h in priced if h.verified),
        min=values[0],
        max=values[-1],
        median=median,
        currency=currency,
    )


# ============================================================================
# Endpoints
# ============================================================================

@router.get(
    "/quota",
    response_model=PublicQuotaResponse,
    summary="Read current quota for this caller (IP-based or user-based).",
)
async def get_quota(request: Request) -> PublicQuotaResponse:
    user_id = _resolve_user_id(request)
    ip = _extract_ip(request) if not user_id else None
    q = check_quota(ip_address=ip, user_id=user_id)
    balance = _read_credit_balance(user_id) if user_id else None
    return _quota_to_response(q, is_authenticated=bool(user_id), credits_balance=balance)


@router.post(
    "/price-scan",
    response_model=PublicPriceScanResponse,
    summary="One-shot price scan for any product name (public, captcha-gated, 2/day).",
)
async def price_scan(body: PublicPriceScanRequest, request: Request) -> PublicPriceScanResponse:
    start = time.time()
    user_id = _resolve_user_id(request)
    ip = _extract_ip(request) if not user_id else None
    user_agent = request.headers.get("user-agent")

    # 1. Verify Turnstile FIRST. Cheapest filter — bots fail here.
    verdict = await verify_token(
        body.turnstile_token,
        remote_ip=ip,
        expected_action="price_scan",
    )
    qhash = query_hash("price", f"{body.manufacturer or ''} {body.product_name} {body.dimensions or ''}", body.country_code)
    if not verdict.success:
        log_scan(
            scan_type="price", ip_address=ip, user_id=user_id, qhash=qhash,
            query_text=body.product_name, cache_hit=False, upstream_cost_usd=0,
            latency_ms=int((time.time() - start) * 1000), outcome="captcha_failed",
            error_message=",".join(verdict.error_codes), user_agent=user_agent,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Captcha verification failed: {','.join(verdict.error_codes) or 'unknown'}",
        )

    # 2. Quota check
    #    Anonymous → 2/day cap. Authenticated → must have ≥ SCAN_CREDIT_COST credits.
    q = check_quota(ip_address=ip, user_id=user_id)
    balance_before = _read_credit_balance(user_id) if user_id else None
    if user_id:
        # Authenticated path: skip the 2/day cap, gate on credits.
        if (balance_before or 0) < SCAN_CREDIT_COST:
            log_scan(
                scan_type="price", ip_address=ip, user_id=user_id, qhash=qhash,
                query_text=body.product_name, cache_hit=False, upstream_cost_usd=0,
                latency_ms=int((time.time() - start) * 1000), outcome="rate_limited",
                error_message="insufficient_credits", user_agent=user_agent,
            )
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "message": "Insufficient credits.",
                    "quota": _quota_to_response(q, is_authenticated=True, credits_balance=balance_before).model_dump(mode="json"),
                },
            )
    else:
        # Anonymous path: 2/day cap
        if not q.allowed:
            log_scan(
                scan_type="price", ip_address=ip, user_id=user_id, qhash=qhash,
                query_text=body.product_name, cache_hit=False, upstream_cost_usd=0,
                latency_ms=int((time.time() - start) * 1000), outcome="rate_limited",
                user_agent=user_agent,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "message": "Daily scan quota reached.",
                    "quota": _quota_to_response(q, is_authenticated=False).model_dump(mode="json"),
                },
            )

    # 3. Cache lookup — identical query within 24h serves from cache (no quota burn, no debit, no upstream spend)
    cached = read_cache(scan_type="price", qhash=qhash)
    if cached:
        log_scan(
            scan_type="price", ip_address=ip, user_id=user_id, qhash=qhash,
            query_text=body.product_name, cache_hit=True, upstream_cost_usd=0,
            latency_ms=int((time.time() - start) * 1000), outcome="success",
            user_agent=user_agent,
        )
        cached["from_cache"] = True
        cached["quota"] = _quota_to_response(
            q, is_authenticated=bool(user_id), credits_balance=balance_before
        ).model_dump(mode="json")
        return PublicPriceScanResponse(**cached)

    # 4. Fresh scan
    query_text = body.product_name.strip()
    if body.manufacturer and body.manufacturer.lower() not in query_text.lower():
        query_text = f"{body.manufacturer} {query_text}".strip()
    if body.dimensions:
        query_text = f"{query_text} {body.dimensions}".strip()

    service = get_perplexity_price_search_service()
    try:
        result = await service.search_prices(
            product_name=query_text,
            dimensions=None,
            country_code=(body.country_code or "").upper() or None,
            limit=10,
            user_id=None,
            workspace_id=None,
            verify_prices=True,
            manufacturer_hint=body.manufacturer,
        )
    except Exception as e:
        logger.warning(f"public price-scan failed: {e}")
        log_scan(
            scan_type="price", ip_address=ip, user_id=user_id, qhash=qhash,
            query_text=body.product_name, cache_hit=False, upstream_cost_usd=0,
            latency_ms=int((time.time() - start) * 1000), outcome="failed",
            error_message=str(e)[:500], user_agent=user_agent,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Price scan upstream failed. Try again in a few minutes.",
        )

    if not result.success:
        log_scan(
            scan_type="price", ip_address=ip, user_id=user_id, qhash=qhash,
            query_text=body.product_name, cache_hit=False, upstream_cost_usd=0,
            latency_ms=result.latency_ms or int((time.time() - start) * 1000),
            outcome="failed", error_message=(result.error or "")[:500],
            user_agent=user_agent,
        )
        # Failure → no credit debit for authenticated users; anonymous quota
        # already burned at the log_scan above only counts success rows, so
        # failure is implicitly free for them too.
        q_after = check_quota(ip_address=ip, user_id=user_id)
        return PublicPriceScanResponse(
            success=False, query=query_text, country_code=body.country_code,
            results=[], stats=PublicMarketStats(count=0, verified_count=0),
            from_cache=False, error=result.error or "scan failed",
            quota=_quota_to_response(q_after, is_authenticated=bool(user_id), credits_balance=balance_before),
        )

    # 5. Format response
    public_hits = [
        PublicPriceResult(
            retailer_name=h.retailer_name,
            product_url=h.product_url,
            price=h.price,
            original_price=h.original_price,
            currency=h.currency,
            availability=h.availability,
            verified=h.verified,
            source=h.source,
            product_title=h.product_title,
            match_kind=h.match_kind,
        )
        for h in result.hits
    ]
    stats = _compute_stats(result.hits)
    response_payload = {
        "success": True,
        "query": query_text,
        "country_code": (body.country_code or "").upper() or None,
        "results": [h.model_dump(mode="json") for h in public_hits],
        "stats": stats.model_dump(mode="json"),
        "summary": result.summary,
        "from_cache": False,
    }

    write_cache(scan_type="price", qhash=qhash, result=response_payload)
    log_scan(
        scan_type="price", ip_address=ip, user_id=user_id, qhash=qhash,
        query_text=body.product_name, cache_hit=False,
        upstream_cost_usd=(result.credits_used or 0) / 1000.0,
        latency_ms=result.latency_ms, outcome="success", user_agent=user_agent,
    )

    # Debit credits AFTER successful scan (authenticated users only).
    balance_after = balance_before
    if user_id:
        debited, new_balance, _err = _debit_credits(
            user_id, operation_type="public_price_scan", qhash=qhash, scan_type="price",
        )
        if debited and new_balance is not None:
            balance_after = new_balance

    q_after = check_quota(ip_address=ip, user_id=user_id)
    return PublicPriceScanResponse(
        **response_payload,
        quota=_quota_to_response(q_after, is_authenticated=bool(user_id), credits_balance=balance_after),
    )


@router.post(
    "/mention-scan",
    response_model=PublicMentionScanResponse,
    summary="One-shot mention scan for any brand/product (public, captcha-gated, 2/day).",
)
async def mention_scan(body: PublicMentionScanRequest, request: Request) -> PublicMentionScanResponse:
    start = time.time()
    user_id = _resolve_user_id(request)
    ip = _extract_ip(request) if not user_id else None
    user_agent = request.headers.get("user-agent")

    verdict = await verify_token(
        body.turnstile_token,
        remote_ip=ip,
        expected_action="mention_scan",
    )
    qhash = query_hash("mention", body.subject_label, body.country_code)
    if not verdict.success:
        log_scan(
            scan_type="mention", ip_address=ip, user_id=user_id, qhash=qhash,
            query_text=body.subject_label, cache_hit=False, upstream_cost_usd=0,
            latency_ms=int((time.time() - start) * 1000), outcome="captcha_failed",
            error_message=",".join(verdict.error_codes), user_agent=user_agent,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Captcha verification failed: {','.join(verdict.error_codes) or 'unknown'}",
        )

    q = check_quota(ip_address=ip, user_id=user_id)
    balance_before = _read_credit_balance(user_id) if user_id else None
    if user_id:
        if (balance_before or 0) < SCAN_CREDIT_COST:
            log_scan(
                scan_type="mention", ip_address=ip, user_id=user_id, qhash=qhash,
                query_text=body.subject_label, cache_hit=False, upstream_cost_usd=0,
                latency_ms=int((time.time() - start) * 1000), outcome="rate_limited",
                error_message="insufficient_credits", user_agent=user_agent,
            )
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "message": "Insufficient credits.",
                    "quota": _quota_to_response(q, is_authenticated=True, credits_balance=balance_before).model_dump(mode="json"),
                },
            )
    else:
        if not q.allowed:
            log_scan(
                scan_type="mention", ip_address=ip, user_id=user_id, qhash=qhash,
                query_text=body.subject_label, cache_hit=False, upstream_cost_usd=0,
                latency_ms=int((time.time() - start) * 1000), outcome="rate_limited",
                user_agent=user_agent,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "message": "Daily scan quota reached.",
                    "quota": _quota_to_response(q, is_authenticated=False).model_dump(mode="json"),
                },
            )

    cached = read_cache(scan_type="mention", qhash=qhash)
    if cached:
        log_scan(
            scan_type="mention", ip_address=ip, user_id=user_id, qhash=qhash,
            query_text=body.subject_label, cache_hit=True, upstream_cost_usd=0,
            latency_ms=int((time.time() - start) * 1000), outcome="success",
            user_agent=user_agent,
        )
        cached["from_cache"] = True
        cached["quota"] = _quota_to_response(
            q, is_authenticated=bool(user_id), credits_balance=balance_before
        ).model_dump(mode="json")
        return PublicMentionScanResponse(**cached)

    # Build facets deterministically — no LLM call, no Anthropic dependency
    label = body.subject_label.strip()
    aliases = [a.strip() for a in (body.aliases or []) if a and a.strip()]
    facets = SubjectFacets(
        label=label,
        aliases=aliases,
        brand=None,
        product_type=None,
        must_have_tokens=[label] + aliases,
        competitor_brands=[],
        language_codes=["en"],
    )

    sources_enabled = {"news": True, "blogs": True, "rss": False, "youtube": False}
    country_codes = [body.country_code.upper()] if body.country_code else []

    search_service = MentionSearchService()
    try:
        result = await search_service.search(
            facets=facets,
            sources_enabled=sources_enabled,
            source_config={},
            country_codes=country_codes,
            recency_days=30,
            force_full_discovery=False,
            attribution=None,
        )
    except Exception as e:
        logger.warning(f"public mention-scan failed: {e}")
        log_scan(
            scan_type="mention", ip_address=ip, user_id=user_id, qhash=qhash,
            query_text=body.subject_label, cache_hit=False, upstream_cost_usd=0,
            latency_ms=int((time.time() - start) * 1000), outcome="failed",
            error_message=str(e)[:500], user_agent=user_agent,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Mention scan upstream failed. Try again in a few minutes.",
        )

    public_hits: List[PublicMentionResult] = []
    outlet_counts: dict[str, int] = {}
    for h in result.hits[:20]:
        domain = h.outlet_domain or ""
        if domain:
            outlet_counts[domain] = outlet_counts.get(domain, 0) + 1
        public_hits.append(PublicMentionResult(
            url=h.url,
            title=h.title,
            excerpt=(h.excerpt or "")[:280] if h.excerpt else None,
            outlet_domain=h.outlet_domain,
            outlet_name=h.outlet_name,
            published_at=h.published_at,
            source=h.source,
            language_code=h.language_code,
            country_code=h.country_code,
        ))

    top_outlets = [
        {"domain": d, "count": c}
        for d, c in sorted(outlet_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    ]
    response_payload = {
        "success": True,
        "subject_label": label,
        "country_code": (body.country_code or "").upper() or None,
        "results": [h.model_dump(mode="json") for h in public_hits],
        "total_results": len(public_hits),
        "top_outlets": top_outlets,
        "from_cache": False,
    }

    write_cache(scan_type="mention", qhash=qhash, result=response_payload)
    log_scan(
        scan_type="mention", ip_address=ip, user_id=user_id, qhash=qhash,
        query_text=body.subject_label, cache_hit=False,
        upstream_cost_usd=(result.credits_used or 0) / 1000.0,
        latency_ms=result.latency_ms, outcome="success", user_agent=user_agent,
    )

    balance_after = balance_before
    if user_id:
        debited, new_balance, _err = _debit_credits(
            user_id, operation_type="public_mention_scan", qhash=qhash, scan_type="mention",
        )
        if debited and new_balance is not None:
            balance_after = new_balance

    q_after = check_quota(ip_address=ip, user_id=user_id)
    return PublicMentionScanResponse(
        **response_payload,
        quota=_quota_to_response(q_after, is_authenticated=bool(user_id), credits_balance=balance_after),
    )
