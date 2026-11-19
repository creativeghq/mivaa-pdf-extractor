"""Public Tools API — backs the `/tools` lead-gen page."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.services.core.supabase_client import get_supabase_client, read_rpc
from app.services.integrations.perplexity_price_search_service import (
    get_perplexity_price_search_service,
    PriceHit as InternalPriceHit,
)
from app.services.integrations.mention_identity_service import SubjectFacets
from app.services.integrations.mention_search_service import MentionSearchService
from app.services.integrations.platform_secret_resolver import resolve_secret
from app.services.integrations.public_lookup_service import (
    QuotaStatus,
    QuotaUnavailable,
    Reservation,
    check_quota,
    finalize_scan,
    log_scan,
    query_hash,
    read_cache,
    reserve_scan,
    write_cache,
)
from app.services.integrations.turnstile_verifier import verify_token
from app.services.integrations.credit_router import debit_row

logger = logging.getLogger(__name__)

# Credits debited per authenticated scan. Mirrors the anonymous free-tier cap
# (2 scans/day) so users see one consistent number across both modes.
# Anonymous visitors pay nothing — they hit the 2/day cap instead.
SCAN_CREDIT_COST = 2
# The catalog product-search tool is a cheap DB lookup (no paid upstream), so it
# costs a single credit. Per-scan-type cost is surfaced via /quota?scan_type=product.
PRODUCT_SCAN_CREDIT_COST = 1

# Per-scan-type credit cost, keyed by the `scan_type` string used in logs/cache.
SCAN_COST_BY_TYPE = {
    "price": SCAN_CREDIT_COST,
    "mention": SCAN_CREDIT_COST,
    "product": PRODUCT_SCAN_CREDIT_COST,
}

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


class PublicProductSearchRequest(BaseModel):
    turnstile_token: str = Field(..., min_length=1, max_length=4096)
    query: str = Field(..., min_length=2, max_length=200)


class PublicProductResult(BaseModel):
    id: str
    name: str
    brand: Optional[str] = None
    category: Optional[str] = None
    image_url: Optional[str] = None
    snippet: Optional[str] = None


class PublicProductSearchResponse(BaseModel):
    success: bool
    query: str
    results: List[PublicProductResult] = []
    total_results: int = 0
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
PublicProductSearchResponse.model_rebuild()


# ============================================================================
# Helpers
# ============================================================================

QUOTA_UNAVAILABLE_MESSAGE = "Quota service unavailable. Try again in a few minutes."
CREDITS_UNAVAILABLE_MESSAGE = "Credit service unavailable. Try again in a few minutes."

_INSUFFICIENT_RE = re.compile(r"insufficient|member_limit_exceeded", re.IGNORECASE)


def _extract_ip(request: Request) -> Optional[str]:
    """Resolve the client IP for quota keying.

    The socket peer as uvicorn reports it: it runs with proxy headers behind our own nginx,
    which sets X-Real-IP / X-Forwarded-For, so `request.client.host` is the real client.
    Nothing fronts nginx, so a `cf-connecting-ip` header is caller-supplied and never read.
    """
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
    """The user's credit balance; 0 when no row exists, None when it could not be read."""
    sb = get_supabase_client().client
    try:
        resp = (
            sb.table("user_credits")
            .select("balance")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
    except Exception as e:
        logger.warning(f"public-tools: credit balance read failed for {user_id}: {e}")
        return None
    if resp and resp.data:
        return int(resp.data.get("balance") or 0)
    return 0


def _debit_credits(user_id: str, *, operation_type: str, qhash: str, scan_type: str, cost: int = SCAN_CREDIT_COST) -> tuple[bool, Optional[int], Optional[str]]:
    """Debit for one public scan through `credit_router.debit_row`. Returns (success, new_balance, error)."""
    row = debit_row(
        user_id=user_id,
        amount=cost,
        operation_type=operation_type,
        description=f"Public {scan_type} scan",
        metadata={"query_hash": qhash, "scan_type": scan_type},
        workspace_id=None,  # public lead-gen scan → the scanning user's personal wallet
        module="public-tools",
    )
    success = bool(row.get("success"))
    balance = int(row["new_balance"]) if row.get("new_balance") is not None else None
    return success, balance, row.get("error_message")


def _is_insufficient_balance(error_message: Optional[str]) -> bool:
    """True for the RPC's own refusal of a short balance; anything else is a server fault."""
    return bool(error_message) and bool(_INSUFFICIENT_RE.search(error_message or ""))


def _refund_credits(user_id: str, *, qhash: str, scan_type: str, cost: int = SCAN_CREDIT_COST) -> None:
    """Best-effort refund of a debit taken before a paid upstream call that produced no result."""
    sb = get_supabase_client().client
    try:
        sb.rpc("refund_credits", {
            "p_user_id": user_id,
            "p_amount": cost,
            "p_operation_type": f"public_{scan_type}_scan_refund",
            "p_description": f"Refund: public {scan_type} scan produced no result",
            "p_metadata": {"query_hash": qhash, "scan_type": scan_type, "refund": True},
            "p_workspace_id": None,
        }).execute()
    except Exception as e:
        logger.warning(f"public-tools: refund_credits failed: {e}")


def _quota_to_response(
    q: QuotaStatus,
    *,
    is_authenticated: bool,
    credits_balance: Optional[int] = None,
    cost: int = SCAN_CREDIT_COST,
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
        credits_per_scan=cost if is_authenticated else 0,
    )


def _quota_or_503(*, ip: Optional[str], user_id: Optional[str]) -> QuotaStatus:
    try:
        return check_quota(ip_address=ip, user_id=user_id)
    except QuotaUnavailable:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=QUOTA_UNAVAILABLE_MESSAGE)


def _display_quota(*, ip: Optional[str], user_id: Optional[str], fallback: QuotaStatus) -> QuotaStatus:
    """Quota for a response body once the work is done — display only, so a read
    failure here must not throw away a result the caller already paid for."""
    try:
        return check_quota(ip_address=ip, user_id=user_id)
    except QuotaUnavailable:
        return fallback


@dataclass
class _Admission:
    """What the gate let through: the quota seen, and what to settle afterwards."""

    q: QuotaStatus
    balance_before: Optional[int] = None
    balance_after: Optional[int] = None
    reservation: Optional[Reservation] = None


def _admit(
    *,
    scan_type: str,
    operation_type: str,
    ip: Optional[str],
    user_id: Optional[str],
    qhash: str,
    query_text: str,
    user_agent: Optional[str],
    start: float,
    cost: int,
) -> _Admission:
    """Gate a fresh (non-cached) scan BEFORE any upstream spend.

    Signed-in: debit `cost` credits (402 on a short balance, 503 when the debit itself
    failed). Anonymous: reserve a `pending` quota row, then re-count the rows ahead of it,
    so concurrent requests cannot both spend the last slot (429), and an unreadable or
    unwritable quota refuses (503).
    """
    def _log(outcome: str, error: Optional[str] = None) -> None:
        log_scan(
            scan_type=scan_type, ip_address=ip, user_id=user_id, qhash=qhash,
            query_text=query_text, cache_hit=False, upstream_cost_usd=0,
            latency_ms=int((time.time() - start) * 1000), outcome=outcome,
            error_message=error, user_agent=user_agent,
        )

    q = _quota_or_503(ip=ip, user_id=user_id)

    if user_id:
        balance_before = _read_credit_balance(user_id)

        def _insufficient() -> HTTPException:
            return HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "message": "Insufficient credits.",
                    "quota": _quota_to_response(
                        q, is_authenticated=True, credits_balance=balance_before, cost=cost,
                    ).model_dump(mode="json"),
                },
            )

        # An unreadable balance (None) is not a short one: the atomic debit decides.
        if balance_before is not None and balance_before < cost:
            _log("rate_limited", "insufficient_credits")
            raise _insufficient()
        debited, new_balance, derr = _debit_credits(
            user_id, operation_type=operation_type, qhash=qhash, scan_type=scan_type, cost=cost,
        )
        if not debited:
            if _is_insufficient_balance(derr):
                _log("rate_limited", derr)
                raise _insufficient()
            logger.warning(f"public-tools: {scan_type} debit failed (not a balance refusal): {derr}")
            _log("failed", f"debit_failed: {derr or 'unknown'}")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=CREDITS_UNAVAILABLE_MESSAGE)
        if new_balance is not None:
            balance_after = new_balance
        elif balance_before is not None:
            balance_after = balance_before - cost
        else:
            balance_after = None
        return _Admission(q=q, balance_before=balance_before, balance_after=balance_after)

    def _quota_exceeded(qs: QuotaStatus) -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "message": "Daily scan quota reached.",
                "quota": _quota_to_response(qs, is_authenticated=False, cost=cost).model_dump(mode="json"),
            },
        )

    if not q.allowed:
        _log("rate_limited")
        raise _quota_exceeded(q)

    try:
        reservation = reserve_scan(
            scan_type=scan_type, ip_address=ip, user_id=None, qhash=qhash,
            query_text=query_text, user_agent=user_agent,
        )
    except QuotaUnavailable:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=QUOTA_UNAVAILABLE_MESSAGE)

    try:
        q_reserved = check_quota(ip_address=ip, user_id=None, created_up_to=reservation.created_at)
    except QuotaUnavailable:
        finalize_scan(reservation, outcome="failed", error_message="quota_unavailable")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=QUOTA_UNAVAILABLE_MESSAGE)
    if q_reserved.used > q_reserved.limit:
        finalize_scan(
            reservation, outcome="rate_limited", error_message="quota_exceeded",
            latency_ms=int((time.time() - start) * 1000),
        )
        raise _quota_exceeded(q_reserved)
    return _Admission(q=q_reserved, reservation=reservation)


def _settle(
    admission: _Admission,
    *,
    succeeded: bool,
    scan_type: str,
    ip: Optional[str],
    user_id: Optional[str],
    qhash: str,
    query_text: str,
    user_agent: Optional[str],
    latency_ms: int,
    upstream_cost_usd: float = 0,
    error_message: Optional[str] = None,
    cost: int = SCAN_CREDIT_COST,
) -> None:
    """Record the final outcome. A failed scan refunds the debit and releases the
    anonymous slot — an honest failure never eats a visitor's daily scans."""
    outcome = "success" if succeeded else "failed"
    if not succeeded and user_id:
        _refund_credits(user_id, qhash=qhash, scan_type=scan_type, cost=cost)
        admission.balance_after = admission.balance_before
    if admission.reservation is not None:
        finalize_scan(
            admission.reservation, outcome=outcome, upstream_cost_usd=upstream_cost_usd,
            latency_ms=latency_ms, error_message=error_message,
        )
        return
    log_scan(
        scan_type=scan_type, ip_address=ip, user_id=user_id, qhash=qhash,
        query_text=query_text, cache_hit=False, upstream_cost_usd=upstream_cost_usd,
        latency_ms=latency_ms, outcome=outcome, error_message=error_message,
        user_agent=user_agent,
    )


def _serve_cached(
    *,
    scan_type: str,
    ip: Optional[str],
    user_id: Optional[str],
    qhash: str,
    query_text: str,
    user_agent: Optional[str],
    start: float,
    cost: int,
) -> Optional[dict]:
    """A cached result with its quota attached, or None. Free: no gate, no debit, no slot."""
    cached = read_cache(scan_type=scan_type, qhash=qhash)
    if not cached:
        return None
    q = _quota_or_503(ip=ip, user_id=user_id)
    balance = _read_credit_balance(user_id) if user_id else None
    log_scan(
        scan_type=scan_type, ip_address=ip, user_id=user_id, qhash=qhash,
        query_text=query_text, cache_hit=True, upstream_cost_usd=0,
        latency_ms=int((time.time() - start) * 1000), outcome="success",
        user_agent=user_agent,
    )
    cached["from_cache"] = True
    cached["quota"] = _quota_to_response(
        q, is_authenticated=bool(user_id), credits_balance=balance, cost=cost,
    ).model_dump(mode="json")
    return cached


def _captcha_failed(
    verdict, *, scan_type: str, ip: Optional[str], user_id: Optional[str], qhash: str,
    query_text: str, user_agent: Optional[str], start: float,
) -> HTTPException:
    log_scan(
        scan_type=scan_type, ip_address=ip, user_id=user_id, qhash=qhash,
        query_text=query_text, cache_hit=False, upstream_cost_usd=0,
        latency_ms=int((time.time() - start) * 1000), outcome="captcha_failed",
        error_message=",".join(verdict.error_codes), user_agent=user_agent,
    )
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Captcha verification failed: {','.join(verdict.error_codes) or 'unknown'}",
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


def _mention_alias_list(aliases: Optional[List[str]]) -> List[str]:
    return [a.strip() for a in (aliases or []) if a and a.strip()]


# ============================================================================
# Endpoints
# ============================================================================

@router.get(
    "/quota",
    response_model=PublicQuotaResponse,
    summary="Read current quota for this caller (IP-based or user-based).",
)
async def get_quota(request: Request, scan_type: Optional[str] = None) -> PublicQuotaResponse:
    # `scan_type` lets a tool ask for its own per-scan credit cost (e.g. product=1 vs
    # price/mention=2). Defaults to the standard cost for back-compat with existing callers.
    user_id = _resolve_user_id(request)
    ip = _extract_ip(request) if not user_id else None
    q = _quota_or_503(ip=ip, user_id=user_id)
    balance = _read_credit_balance(user_id) if user_id else None
    cost = SCAN_COST_BY_TYPE.get((scan_type or "").lower(), SCAN_CREDIT_COST)
    return _quota_to_response(q, is_authenticated=bool(user_id), credits_balance=balance, cost=cost)


@router.post(
    "/price-scan",
    response_model=PublicPriceScanResponse,
    summary="One-shot price scan for any product name (public, captcha-gated, 2/day).",
)
async def price_scan(body: PublicPriceScanRequest, request: Request) -> PublicPriceScanResponse:
    start = time.time()
    cost = SCAN_COST_BY_TYPE["price"]
    user_id = _resolve_user_id(request)
    ip = _extract_ip(request) if not user_id else None
    user_agent = request.headers.get("user-agent")
    ctx = dict(scan_type="price", ip=ip, user_id=user_id, user_agent=user_agent)

    verdict = await verify_token(
        body.turnstile_token,
        remote_ip=ip,
        expected_action="price_scan",
    )
    qhash = query_hash("price", f"{body.manufacturer or ''} {body.product_name} {body.dimensions or ''}", body.country_code)
    if not verdict.success:
        raise _captcha_failed(verdict, qhash=qhash, query_text=body.product_name, start=start, **ctx)

    cached = _serve_cached(qhash=qhash, query_text=body.product_name, start=start, cost=cost, **ctx)
    if cached:
        return PublicPriceScanResponse(**cached)

    admission = _admit(
        operation_type="public_price_scan", qhash=qhash, query_text=body.product_name,
        start=start, cost=cost, **ctx,
    )

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
        _settle(
            admission, succeeded=False, qhash=qhash, query_text=body.product_name,
            latency_ms=int((time.time() - start) * 1000), error_message=str(e)[:500],
            cost=cost, **ctx,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Price scan upstream failed. Try again in a few minutes.",
        )

    source_errors = dict(result.source_errors or {})
    if source_errors:
        logger.warning(f"public price-scan: sources failed: {source_errors}")

    if not result.hits and (not result.success or result.all_primary_sources_failed()):
        error = result.error if not result.success and result.error else (
            "No price source could be searched right now. Try again in a few minutes."
        )
        _settle(
            admission, succeeded=False, qhash=qhash, query_text=body.product_name,
            latency_ms=result.latency_ms or int((time.time() - start) * 1000),
            error_message=(error + (f" | {source_errors}" if source_errors else ""))[:500],
            cost=cost, **ctx,
        )
        q_after = _display_quota(ip=ip, user_id=user_id, fallback=admission.q)
        return PublicPriceScanResponse(
            success=False, query=query_text, country_code=body.country_code,
            results=[], stats=PublicMarketStats(count=0, verified_count=0),
            from_cache=False, error=error,
            quota=_quota_to_response(q_after, is_authenticated=bool(user_id), credits_balance=admission.balance_after, cost=cost),
        )

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

    # A partial answer is served but never cached: the next caller should get a full try.
    if result.success and not source_errors:
        write_cache(scan_type="price", qhash=qhash, result=response_payload)
    _settle(
        admission, succeeded=True, qhash=qhash, query_text=body.product_name,
        latency_ms=result.latency_ms, upstream_cost_usd=(result.credits_used or 0) / 1000.0,
        error_message=f"partial: {source_errors}"[:500] if source_errors else None,
        cost=cost, **ctx,
    )

    q_after = _display_quota(ip=ip, user_id=user_id, fallback=admission.q)
    return PublicPriceScanResponse(
        **response_payload,
        quota=_quota_to_response(q_after, is_authenticated=bool(user_id), credits_balance=admission.balance_after, cost=cost),
    )


@router.post(
    "/mention-scan",
    response_model=PublicMentionScanResponse,
    summary="One-shot mention scan for any brand/product (public, captcha-gated, 2/day).",
)
async def mention_scan(body: PublicMentionScanRequest, request: Request) -> PublicMentionScanResponse:
    start = time.time()
    cost = SCAN_COST_BY_TYPE["mention"]
    user_id = _resolve_user_id(request)
    ip = _extract_ip(request) if not user_id else None
    user_agent = request.headers.get("user-agent")
    ctx = dict(scan_type="mention", ip=ip, user_id=user_id, user_agent=user_agent)

    verdict = await verify_token(
        body.turnstile_token,
        remote_ip=ip,
        expected_action="mention_scan",
    )
    label = body.subject_label.strip()
    aliases = _mention_alias_list(body.aliases)
    qhash = query_hash("mention", body.subject_label, body.country_code, aliases=aliases)
    if not verdict.success:
        raise _captcha_failed(verdict, qhash=qhash, query_text=body.subject_label, start=start, **ctx)

    cached = _serve_cached(qhash=qhash, query_text=body.subject_label, start=start, cost=cost, **ctx)
    if cached:
        return PublicMentionScanResponse(**cached)

    # Facets are built deterministically — no LLM call.
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

    admission = _admit(
        operation_type="public_mention_scan", qhash=qhash, query_text=body.subject_label,
        start=start, cost=cost, **ctx,
    )

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
        _settle(
            admission, succeeded=False, qhash=qhash, query_text=body.subject_label,
            latency_ms=int((time.time() - start) * 1000), error_message=str(e)[:500],
            cost=cost, **ctx,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Mention scan upstream failed. Try again in a few minutes.",
        )

    source_errors = dict(result.errors or {})
    if source_errors:
        logger.warning(f"public mention-scan: sources failed: {source_errors}")

    # `by_source` only carries sources that answered, so empty + errors = nobody answered.
    if not result.hits and source_errors and not result.by_source:
        error = "No mention source could be searched right now. Try again in a few minutes."
        _settle(
            admission, succeeded=False, qhash=qhash, query_text=body.subject_label,
            latency_ms=result.latency_ms or int((time.time() - start) * 1000),
            error_message=f"{error} | {source_errors}"[:500], cost=cost, **ctx,
        )
        q_after = _display_quota(ip=ip, user_id=user_id, fallback=admission.q)
        return PublicMentionScanResponse(
            success=False, subject_label=label,
            country_code=(body.country_code or "").upper() or None,
            results=[], total_results=0, top_outlets=[], from_cache=False, error=error,
            quota=_quota_to_response(q_after, is_authenticated=bool(user_id), credits_balance=admission.balance_after, cost=cost),
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

    if not source_errors:
        write_cache(scan_type="mention", qhash=qhash, result=response_payload)
    _settle(
        admission, succeeded=True, qhash=qhash, query_text=body.subject_label,
        latency_ms=result.latency_ms, upstream_cost_usd=(result.credits_used or 0) / 1000.0,
        error_message=f"partial: {source_errors}"[:500] if source_errors else None,
        cost=cost, **ctx,
    )

    q_after = _display_quota(ip=ip, user_id=user_id, fallback=admission.q)
    return PublicMentionScanResponse(
        **response_payload,
        quota=_quota_to_response(q_after, is_authenticated=bool(user_id), credits_balance=admission.balance_after, cost=cost),
    )


@router.post(
    "/product-search",
    response_model=PublicProductSearchResponse,
    summary="Search the public material catalog (public, captcha-gated, 2/day or 1 credit).",
)
async def product_search(body: PublicProductSearchRequest, request: Request) -> PublicProductSearchResponse:
    """Public, metered search over the operator's ROOT-workspace catalog (its public-facing
    products only — NOT tenant private catalogs; enforced by the `search_public_products` RPC).

    Same billing model as the other public tools: anonymous → 2 free scans/day, authenticated →
    1 credit each (PRODUCT_SCAN_CREDIT_COST). Turnstile-gated; identical queries within 24h are
    cache-served free.
    """
    start = time.time()
    cost = PRODUCT_SCAN_CREDIT_COST
    user_id = _resolve_user_id(request)
    ip = _extract_ip(request) if not user_id else None
    user_agent = request.headers.get("user-agent")
    ctx = dict(scan_type="product", ip=ip, user_id=user_id, user_agent=user_agent)

    verdict = await verify_token(
        body.turnstile_token,
        remote_ip=ip,
        expected_action="product_search",
    )
    qhash = query_hash("product", body.query)
    if not verdict.success:
        raise _captcha_failed(verdict, qhash=qhash, query_text=body.query, start=start, **ctx)

    cached = _serve_cached(qhash=qhash, query_text=body.query, start=start, cost=cost, **ctx)
    if cached:
        return PublicProductSearchResponse(**cached)

    admission = _admit(
        operation_type="public_product_search", qhash=qhash, query_text=body.query,
        start=start, cost=cost, **ctx,
    )

    query_text = body.query.strip()
    sb = get_supabase_client().client
    try:
        resp = read_rpc(sb, "search_public_products", {"p_query": query_text, "p_limit": 24}).execute()
        rows = resp.data or []
    except Exception as e:
        logger.warning(f"public product-search failed: {e}")
        _settle(
            admission, succeeded=False, qhash=qhash, query_text=body.query,
            latency_ms=int((time.time() - start) * 1000), error_message=str(e)[:500],
            cost=cost, **ctx,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Product search failed. Try again in a few minutes.",
        )

    results = [
        PublicProductResult(
            id=str(r.get("id")),
            name=r.get("name") or "",
            brand=r.get("brand"),
            category=r.get("category"),
            image_url=r.get("image_url"),
            snippet=r.get("snippet"),
        )
        for r in rows
    ]
    response_payload = {
        "success": True,
        "query": query_text,
        "results": [r.model_dump(mode="json") for r in results],
        "total_results": len(results),
        "from_cache": False,
    }

    write_cache(scan_type="product", qhash=qhash, result=response_payload)
    _settle(
        admission, succeeded=True, qhash=qhash, query_text=body.query,
        latency_ms=int((time.time() - start) * 1000), cost=cost, **ctx,
    )

    q_after = _display_quota(ip=ip, user_id=user_id, fallback=admission.q)
    return PublicProductSearchResponse(
        **response_payload,
        quota=_quota_to_response(q_after, is_authenticated=bool(user_id), credits_balance=admission.balance_after, cost=cost),
    )
