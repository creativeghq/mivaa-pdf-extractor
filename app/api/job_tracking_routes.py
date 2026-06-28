"""
Public Job Tracking API — /api/v1/jobs/track/*

External projects authenticate with an `api_keys` Bearer token (`kai_*`),
register tracked job searches, and control the refresh cadence themselves
(no platform cron polls these — partners pay per call).

Mirror of `mention_tracking_routes.py` for job research. Uses the same
`authenticate_api_key` dependency.

Routing:
  api_key_id NOT NULL  → external API consumer (this file)
  api_key_id IS NULL   → internal session-JWT flow (job_research_routes.py)

Deleting the api_key CASCADEs out every tracked_jobs row + job_listings tied
to it (enforced at the DB level via ON DELETE CASCADE).

Per-call partner billing (debit on entry, refund on hard failure / no-op):
  refresh           → 5 credits
  digest_preview    → 1 credit
  regenerate-keys   → 2 credits
  classifier-correct→ 0 credits (read-write but cheap)
  CRUD reads        → 0 credits
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.price_lookup_routes import ApiKeyContext, authenticate_api_key
from app.services.integrations import job_cost_logger as costs
from app.services.integrations.job_research_service import get_job_research_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/jobs/track",
    tags=["Job Tracking (Public API)"],
    responses={
        401: {"description": "Invalid or missing API key"},
        402: {"description": "Insufficient credits"},
        403: {"description": "API key does not own this tracking_id"},
        404: {"description": "tracking_id not found"},
        429: {"description": "Rate limit exceeded"},
    },
)


# ────────────────────────────────────────────────────────────────────────────
# Request models
# ────────────────────────────────────────────────────────────────────────────

class CreateExternalJobTrackRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=200, description="Display name for this tracked search.")
    keywords: List[str] = Field(..., min_length=1, max_length=20)
    excluded_keywords: Optional[List[str]] = None
    location: Optional[str] = None
    country_code: Optional[str] = Field(None, description="ISO-2 country code; biases DataForSEO Google Jobs.")
    remote_only: bool = False
    seniority: Optional[str] = Field(None, pattern="^(junior|mid|senior|lead|principal|any)$")
    employment_type: Optional[List[str]] = None
    salary_min: Optional[int] = None
    salary_currency: Optional[str] = Field("USD", description="ISO-4217.")
    excluded_companies: Optional[List[str]] = None
    preferred_companies: Optional[List[str]] = None
    sources_enabled: Optional[Dict[str, bool]] = Field(
        None,
        description="Per-source toggles. Defaults: google_jobs=true, perplexity=true, careers_pages=false, rss_feeds=false.",
    )
    careers_page_urls: Optional[List[str]] = Field(None, description="Required when sources_enabled.careers_pages=true.")
    rss_feed_urls: Optional[List[str]] = Field(None, description="Required when sources_enabled.rss_feeds=true.")
    refresh_interval_hours: int = Field(24, ge=1, le=168)
    max_age_days: int = Field(14, ge=1, le=365, description="Recency gate: drop listings older than this (and undated aggregator listings). Default 14.")
    alert_webhook_url: Optional[str] = Field(None, description="Per-tracked_job webhook POST'd at digest tick + on burst.")
    digest_hour_utc: int = Field(7, ge=0, le=23, description="Reserved for future digest support on external flow.")
    run_first_refresh: bool = Field(True, description="Run discovery + classifier inline before returning.")


class UpdateExternalJobTrackRequest(BaseModel):
    label: Optional[str] = None
    keywords: Optional[List[str]] = None
    excluded_keywords: Optional[List[str]] = None
    location: Optional[str] = None
    country_code: Optional[str] = None
    remote_only: Optional[bool] = None
    seniority: Optional[str] = None
    employment_type: Optional[List[str]] = None
    salary_min: Optional[int] = None
    salary_currency: Optional[str] = None
    excluded_companies: Optional[List[str]] = None
    preferred_companies: Optional[List[str]] = None
    sources_enabled: Optional[Dict[str, bool]] = None
    careers_page_urls: Optional[List[str]] = None
    rss_feed_urls: Optional[List[str]] = None
    refresh_interval_hours: Optional[int] = Field(None, ge=1, le=168)
    max_age_days: Optional[int] = Field(None, ge=1, le=365)
    alert_webhook_url: Optional[str] = None
    is_active: Optional[bool] = None


class ClassifierCorrectionRequest(BaseModel):
    corrected_relevance: str = Field(..., pattern="^(match|tangential|mismatch)$")
    reason: Optional[str] = None


# ────────────────────────────────────────────────────────────────────────────
# Helper: enforce ownership
# ────────────────────────────────────────────────────────────────────────────

def _load_owned(api_key_id: str, tracking_id: str) -> Dict[str, Any]:
    svc = get_job_research_service()
    row = svc.get(tracking_id, api_key_id=api_key_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"tracking_id {tracking_id} not found for this API key")
    return row


# ────────────────────────────────────────────────────────────────────────────
# CRUD
# ────────────────────────────────────────────────────────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED, summary="Create a tracked job search")
async def create_tracked(
    body: CreateExternalJobTrackRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """First-refresh debit happens here (5 credits if `run_first_refresh=true`).
    Refunded if the first refresh fails."""
    svc = get_job_research_service()
    user_id = getattr(ctx, "user_id", None)

    debit_amount = costs.JOB_OP_CREDIT_COST.get("refresh", 5) if body.run_first_refresh else 0
    if debit_amount and user_id:
        if not costs.debit_credits(user_id=user_id, amount=debit_amount, operation_type="job_research.refresh"):
            raise HTTPException(status_code=402, detail="Insufficient credits")

    try:
        row = await svc.create(
            api_key_id=ctx.api_key_id,
            workspace_id=getattr(ctx, "workspace_id", None),
            # Attribute the background_agents row to the api_key's owner so the
            # search shows up in their /admin/background-agents + saved-jobs panel.
            api_key_owner_user_id=getattr(ctx, "user_id", None),
            **body.model_dump(exclude_none=True),
        )
    except ValueError as e:
        if debit_amount and user_id:
            costs.refund_credits(user_id=user_id, amount=debit_amount, operation_type="job_research.refresh")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if debit_amount and user_id:
            costs.refund_credits(user_id=user_id, amount=debit_amount, operation_type="job_research.refresh")
        raise HTTPException(status_code=500, detail=str(e)[:200])

    # Refund the first-refresh credit on a true no-op (audit #217 H15): an explicit
    # error, OR new candidates were found but the classifier persisted none of them.
    fr = (row.get("first_refresh") or {})
    no_op = bool(fr.get("error")) or (
        fr.get("candidates_after_exclusions", 0) > 0 and fr.get("persisted", 0) == 0
    )
    credits_debited = debit_amount
    if debit_amount and user_id and no_op:
        costs.refund_credits(user_id=user_id, amount=debit_amount, operation_type="job_research.refresh")
        credits_debited = 0

    return {"data": row, "partner_credits_debited": credits_debited}


@router.get("", summary="List your tracked job searches")
async def list_tracked(
    only_active: bool = Query(True),
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    svc = get_job_research_service()
    return {"data": svc.list_for_api_key(ctx.api_key_id, only_active=only_active)}


@router.get("/{tracking_id}", summary="Read one tracked job search")
async def get_tracked(
    tracking_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    return {"data": _load_owned(ctx.api_key_id, tracking_id)}


@router.put("/{tracking_id}", summary="Update tracked job config")
async def update_tracked(
    tracking_id: str,
    body: UpdateExternalJobTrackRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    _load_owned(ctx.api_key_id, tracking_id)
    svc = get_job_research_service()
    # Use the same update path as the internal flow but bypass owner_user_id check
    # (ownership already verified above).
    try:
        row = svc.update(tracking_id, owner_user_id=None, patch=body.model_dump(exclude_none=True))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"data": row}


@router.delete("/{tracking_id}", summary="Soft-delete (deactivate). Hard delete via api_key revocation cascades.")
async def delete_tracked(
    tracking_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    _load_owned(ctx.api_key_id, tracking_id)
    svc = get_job_research_service()
    ok = svc.deactivate(tracking_id, owner_user_id=None)
    if not ok:
        raise HTTPException(status_code=404, detail="tracking_id not found")
    return {"ok": True}


# ────────────────────────────────────────────────────────────────────────────
# Refresh + listings + summary
# ────────────────────────────────────────────────────────────────────────────

@router.post("/{tracking_id}/refresh", summary="Force a refresh (debits 5 credits)")
async def refresh_tracked(
    tracking_id: str,
    force_full_discovery: bool = Query(False),
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    _load_owned(ctx.api_key_id, tracking_id)
    user_id = getattr(ctx, "user_id", None)
    debit_amount = costs.JOB_OP_CREDIT_COST.get("refresh", 5)

    if user_id and not costs.debit_credits(user_id=user_id, amount=debit_amount, operation_type="job_research.refresh"):
        raise HTTPException(status_code=402, detail="Insufficient credits")

    svc = get_job_research_service()
    try:
        outcome = await svc.refresh(tracking_id, force=True, force_full_discovery=force_full_discovery)
    except Exception as e:
        if user_id:
            costs.refund_credits(user_id=user_id, amount=debit_amount, operation_type="job_research.refresh")
        raise HTTPException(status_code=500, detail=str(e)[:200])

    if outcome.get("skipped"):
        if user_id:
            costs.refund_credits(user_id=user_id, amount=debit_amount, operation_type="job_research.refresh")
        return {"data": outcome, "partner_credits_debited": 0}

    # No-op refund (audit #217 H15): if the refresh found NEW candidates but persisted
    # none of them, the classifier dropped everything (wholesale failure or all-mismatch
    # bug) — the partner paid and got zero listings. A legitimate "nothing new" run
    # short-circuits earlier with no `candidates_after_exclusions` key, so it keeps the
    # credit (upstream calls genuinely ran).
    if user_id and outcome.get("error"):
        costs.refund_credits(user_id=user_id, amount=debit_amount, operation_type="job_research.refresh")
        return {"data": outcome, "partner_credits_debited": 0}
    if user_id and outcome.get("candidates_after_exclusions", 0) > 0 and outcome.get("persisted", 0) == 0:
        costs.refund_credits(user_id=user_id, amount=debit_amount, operation_type="job_research.refresh")
        return {"data": outcome, "partner_credits_debited": 0}

    return {"data": outcome, "partner_credits_debited": debit_amount}


@router.get("/{tracking_id}/listings", summary="List discovered listings")
async def list_listings(
    tracking_id: str,
    relevance: str = Query("match", pattern="^(match|tangential|unverifiable|all)$"),
    days: int = Query(30, ge=1, le=365),
    only_actionable: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    _load_owned(ctx.api_key_id, tracking_id)
    svc = get_job_research_service()
    rows = svc.list_listings(
        tracking_id, relevance=relevance, days=days,
        only_actionable=only_actionable, limit=limit,
    )
    return {"data": rows, "count": len(rows)}


@router.get("/{tracking_id}/summary", summary="Aggregate snapshot")
async def get_summary(
    tracking_id: str,
    days: int = Query(30, ge=1, le=365),
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    _load_owned(ctx.api_key_id, tracking_id)
    svc = get_job_research_service()
    return {"data": svc.summary(tracking_id, days=days)}


# ────────────────────────────────────────────────────────────────────────────
# Exclusions + listing actions + classifier correction
# ────────────────────────────────────────────────────────────────────────────

class ExcludeBody(BaseModel):
    url: Optional[str] = None
    domain: Optional[str] = None
    company: Optional[str] = None
    reason: Optional[str] = None


@router.post("/{tracking_id}/exclude", summary="Add a URL/domain/company exclusion")
async def add_exclusion(
    tracking_id: str,
    body: ExcludeBody,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    _load_owned(ctx.api_key_id, tracking_id)
    svc = get_job_research_service()
    try:
        return {"data": svc.add_exclusion(tracking_id, **body.model_dump(exclude_none=True))}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{tracking_id}/exclusions", summary="List exclusions")
async def list_exclusions(
    tracking_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    _load_owned(ctx.api_key_id, tracking_id)
    svc = get_job_research_service()
    return {"data": svc.list_exclusions(tracking_id)}


@router.post("/{tracking_id}/regenerate-keywords", summary="Re-run Haiku keyword expansion (2 credits)")
async def regenerate_keywords(
    tracking_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    _load_owned(ctx.api_key_id, tracking_id)
    user_id = getattr(ctx, "user_id", None)
    debit_amount = 2  # Lighter than a refresh — single Haiku call
    if user_id and not costs.debit_credits(user_id=user_id, amount=debit_amount, operation_type="job_research.regenerate_keywords"):
        raise HTTPException(status_code=402, detail="Insufficient credits")
    svc = get_job_research_service()
    try:
        result = await svc.regenerate_keywords(tracking_id, owner_user_id=None)
    except Exception as e:
        if user_id:
            costs.refund_credits(user_id=user_id, amount=debit_amount, operation_type="job_research.regenerate_keywords")
        raise HTTPException(status_code=500, detail=str(e)[:200])
    return {"data": result, "partner_credits_debited": debit_amount}


@router.post("/listings/{listing_id}/correct-match", summary="Classifier correction (free)")
async def correct_match(
    listing_id: str,
    body: ClassifierCorrectionRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Free — just inserts a row into job_match_corrections + flips the listing's
    relevance. Recent corrections become Haiku few-shot on the next refresh."""
    svc = get_job_research_service()
    listing_res = (
        svc.sb.table("job_listings")
        .select("id, tracked_job_id, relevance")
        .eq("id", listing_id)
        .maybe_single()
        .execute()
    )
    listing = (listing_res.data if listing_res else None) or None
    if not listing:
        raise HTTPException(status_code=404, detail="listing not found")

    # Verify the parent tracked_job belongs to this api_key
    parent = svc.get(listing["tracked_job_id"], api_key_id=ctx.api_key_id)
    if not parent:
        raise HTTPException(status_code=403, detail="not your listing")

    svc.sb.table("job_match_corrections").insert({
        "tracked_job_id": listing["tracked_job_id"],
        "job_listing_id": listing_id,
        "user_id": getattr(ctx, "user_id", None),
        "original_relevance": listing.get("relevance"),
        "corrected_relevance": body.corrected_relevance,
        "reason": (body.reason or None),
    }).execute()
    svc.sb.table("job_listings").update({
        "relevance": body.corrected_relevance,
        "match_note": f"User corrected: {body.reason or 'no reason given'}"[:240],
    }).eq("id", listing_id).execute()
    return {"ok": True, "applied_immediately": True}
