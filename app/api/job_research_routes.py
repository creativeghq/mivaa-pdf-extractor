"""
Job Research API Routes — internal flow (session JWT) + cron endpoints.

Endpoint inventory:
  POST   /api/v1/job-research/track                      — create tracked_job
  GET    /api/v1/job-research/track                      — list user's tracked_jobs
  GET    /api/v1/job-research/track/{id}                 — read one
  PUT    /api/v1/job-research/track/{id}                 — update
  DELETE /api/v1/job-research/track/{id}                 — soft delete (deactivate)

  POST   /api/v1/job-research/track/{id}/refresh         — re-run discovery
  GET    /api/v1/job-research/track/{id}/listings        — list job_listings rows
  GET    /api/v1/job-research/track/{id}/summary         — aggregate snapshot
  POST   /api/v1/job-research/track/{id}/exclude         — add exclusion (url/domain/company)
  GET    /api/v1/job-research/track/{id}/exclusions      — list exclusions
  DELETE /api/v1/job-research/exclusions/{exclusion_id}  — remove exclusion

  POST   /api/v1/job-research/listings/{listing_id}/action  — mark saved/applied/dismissed

  POST   /api/v1/job-research/cron-refresh               — internal cron tick (x-cron-secret)
  POST   /api/v1/job-research/cron-digest                — digest tick (x-cron-secret)

External (api_key) flow lives in `job_tracking_routes.py` (added in a follow-up).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.dependencies import get_current_user, get_workspace_context
from app.middleware.jwt_auth import User, WorkspaceContext
from app.modules.job_research_notifications.service import get_job_digest_dispatcher
from app.services.integrations.job_research_service import get_job_research_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/job-research",
    tags=["Job Research"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not found"},
    },
)


# ─── Request models ──────────────────────────────────────────────────────

class CreateTrackedJobRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=200)
    keywords: List[str] = Field(..., min_length=1, max_length=20)
    excluded_keywords: Optional[List[str]] = None
    location: Optional[str] = None
    country_code: Optional[str] = None
    remote_only: bool = False
    seniority: Optional[str] = None
    employment_type: Optional[List[str]] = None
    salary_min: Optional[int] = None
    salary_currency: Optional[str] = None
    excluded_companies: Optional[List[str]] = None
    preferred_companies: Optional[List[str]] = None
    sources_enabled: Optional[Dict[str, bool]] = None
    careers_page_urls: Optional[List[str]] = None
    digest_hour_utc: int = Field(7, ge=0, le=23)
    digest_day_of_week: Optional[int] = Field(None, ge=0, le=6, description="0=Sunday..6=Saturday. NULL = daily.")
    alert_channels: Optional[List[str]] = None
    alert_webhook_url: Optional[str] = None
    refresh_interval_hours: int = Field(24, ge=1, le=168)
    source_conversation_id: Optional[str] = Field(None, description="agent_chat_conversations.id where the user set up the search; daily digest will chat-post into it.")
    run_first_refresh: bool = Field(True, description="If true, run discovery + classifier synchronously so the response includes real listings.")


class UpdateTrackedJobRequest(BaseModel):
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
    digest_enabled: Optional[bool] = None
    digest_hour_utc: Optional[int] = Field(None, ge=0, le=23)
    digest_day_of_week: Optional[int] = Field(None, ge=0, le=6)
    alert_channels: Optional[List[str]] = None
    alert_webhook_url: Optional[str] = None
    refresh_interval_hours: Optional[int] = Field(None, ge=1, le=168)
    is_active: Optional[bool] = None


class ExcludeRequest(BaseModel):
    url: Optional[str] = None
    domain: Optional[str] = None
    company: Optional[str] = None
    reason: Optional[str] = None


class ListingActionRequest(BaseModel):
    action: str = Field(..., description="saved | applied | dismissed | interested")
    notes: Optional[str] = None


class ClassifierCorrectionRequest(BaseModel):
    corrected_relevance: str = Field(..., description="match | tangential | mismatch")
    reason: Optional[str] = Field(None, description="Optional free-text the user types into the prompt")


# ─── CRUD ────────────────────────────────────────────────────────────────

@router.post("/track")
async def create_tracked_job(
    body: CreateTrackedJobRequest,
    user: User = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    svc = get_job_research_service()
    try:
        row = await svc.create(
            owner_user_id=str(user.id),
            workspace_id=str(workspace.id) if workspace else None,
            **body.model_dump(exclude_none=True),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"tracked_job": row}


@router.post("/track/{tracked_job_id}/regenerate-keywords")
async def regenerate_keywords(
    tracked_job_id: str,
    user: User = Depends(get_current_user),
):
    """Re-run Haiku keyword expansion. Returns the new expanded list + the rejected suggestions."""
    svc = get_job_research_service()
    try:
        result = await svc.regenerate_keywords(tracked_job_id, owner_user_id=str(user.id))
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result


@router.get("/track")
async def list_tracked_jobs(
    only_active: bool = Query(True),
    user: User = Depends(get_current_user),
):
    svc = get_job_research_service()
    return {"tracked_jobs": svc.list_for_user(str(user.id), only_active=only_active)}


@router.get("/track/{tracked_job_id}")
async def get_tracked_job(
    tracked_job_id: str,
    user: User = Depends(get_current_user),
):
    svc = get_job_research_service()
    row = svc.get(tracked_job_id, owner_user_id=str(user.id))
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return {"tracked_job": row}


@router.put("/track/{tracked_job_id}")
async def update_tracked_job(
    tracked_job_id: str,
    body: UpdateTrackedJobRequest,
    user: User = Depends(get_current_user),
):
    svc = get_job_research_service()
    row = svc.update(tracked_job_id, str(user.id), body.model_dump(exclude_none=True))
    return {"tracked_job": row}


@router.delete("/track/{tracked_job_id}")
async def delete_tracked_job(
    tracked_job_id: str,
    user: User = Depends(get_current_user),
):
    svc = get_job_research_service()
    ok = svc.deactivate(tracked_job_id, str(user.id))
    if not ok:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


# ─── Refresh + listings + summary ────────────────────────────────────────

@router.post("/track/{tracked_job_id}/refresh")
async def refresh_tracked_job(
    tracked_job_id: str,
    force: bool = Query(False),
    force_full_discovery: bool = Query(False),
    user: User = Depends(get_current_user),
):
    svc = get_job_research_service()
    # Ownership check
    if not svc.get(tracked_job_id, owner_user_id=str(user.id)):
        raise HTTPException(status_code=404, detail="Not found")
    outcome = await svc.refresh(tracked_job_id, force=force, force_full_discovery=force_full_discovery)
    return outcome


@router.get("/track/{tracked_job_id}/listings")
async def list_listings(
    tracked_job_id: str,
    relevance: str = Query("match"),
    days: int = Query(30, ge=1, le=365),
    only_actionable: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
    user: User = Depends(get_current_user),
):
    svc = get_job_research_service()
    if not svc.get(tracked_job_id, owner_user_id=str(user.id)):
        raise HTTPException(status_code=404, detail="Not found")
    rows = svc.list_listings(
        tracked_job_id, relevance=relevance, days=days,
        only_actionable=only_actionable, limit=limit,
    )
    return {"listings": rows, "count": len(rows)}


@router.get("/track/{tracked_job_id}/summary")
async def get_summary(
    tracked_job_id: str,
    days: int = Query(30, ge=1, le=365),
    user: User = Depends(get_current_user),
):
    svc = get_job_research_service()
    if not svc.get(tracked_job_id, owner_user_id=str(user.id)):
        raise HTTPException(status_code=404, detail="Not found")
    return svc.summary(tracked_job_id, days=days)


# ─── Exclusions ──────────────────────────────────────────────────────────

@router.post("/track/{tracked_job_id}/exclude")
async def add_exclusion(
    tracked_job_id: str,
    body: ExcludeRequest,
    user: User = Depends(get_current_user),
):
    svc = get_job_research_service()
    if not svc.get(tracked_job_id, owner_user_id=str(user.id)):
        raise HTTPException(status_code=404, detail="Not found")
    try:
        row = svc.add_exclusion(tracked_job_id, **body.model_dump(exclude_none=True))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"exclusion": row}


@router.get("/track/{tracked_job_id}/exclusions")
async def list_exclusions(
    tracked_job_id: str,
    user: User = Depends(get_current_user),
):
    svc = get_job_research_service()
    if not svc.get(tracked_job_id, owner_user_id=str(user.id)):
        raise HTTPException(status_code=404, detail="Not found")
    return {"exclusions": svc.list_exclusions(tracked_job_id)}


@router.delete("/exclusions/{exclusion_id}")
async def remove_exclusion(
    exclusion_id: str,
    user: User = Depends(get_current_user),
):
    # RLS enforces ownership
    svc = get_job_research_service()
    ok = svc.remove_exclusion(exclusion_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


# ─── Per-listing user actions ────────────────────────────────────────────

@router.post("/listings/{listing_id}/action")
async def mark_listing(
    listing_id: str,
    body: ListingActionRequest,
    user: User = Depends(get_current_user),
):
    svc = get_job_research_service()
    try:
        row = svc.mark_listing(listing_id, action=body.action, user_id=str(user.id), notes=body.notes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"listing": row}


@router.post("/listings/{listing_id}/correct-match")
async def correct_match(
    listing_id: str,
    body: ClassifierCorrectionRequest,
    user: User = Depends(get_current_user),
):
    """User feedback for the classifier. Inserts a `job_match_corrections` row;
    the classifier service prepends the most recent corrections per tracked_job
    as Haiku few-shot examples on the next refresh."""
    if body.corrected_relevance not in ("match", "tangential", "mismatch"):
        raise HTTPException(status_code=400, detail="corrected_relevance must be one of match|tangential|mismatch")
    svc = get_job_research_service()
    try:
        # Verify ownership via parent tracked_job; svc already RLS-bounded
        listing = (
            svc.sb.table("job_listings")
            .select("id, tracked_job_id, relevance, title, company")
            .eq("id", listing_id)
            .maybe_single()
            .execute()
        )
        listing_row = (listing.data if listing else None) or None
        if not listing_row:
            raise HTTPException(status_code=404, detail="listing not found")

        # Confirm tracked_job is owned by the user (RLS enforces, but we want a clean 403)
        owner_check = svc.get(listing_row["tracked_job_id"], owner_user_id=str(user.id))
        if not owner_check:
            raise HTTPException(status_code=403, detail="not your listing")

        svc.sb.table("job_match_corrections").insert({
            "tracked_job_id": listing_row["tracked_job_id"],
            "job_listing_id": listing_id,
            "user_id": str(user.id),
            "original_relevance": listing_row.get("relevance"),
            "corrected_relevance": body.corrected_relevance,
            "reason": (body.reason or None),
        }).execute()

        # Also update the listing's relevance + match_note inline so the user sees
        # immediate feedback in the UI instead of waiting for the next refresh.
        svc.sb.table("job_listings").update({
            "relevance": body.corrected_relevance,
            "match_note": f"User corrected: {body.reason or 'no reason given'}"[:240],
        }).eq("id", listing_id).execute()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])
    return {"ok": True, "applied_immediately": True}


# ─── Cron endpoints (x-cron-secret) ──────────────────────────────────────

def _verify_cron_secret(x_cron_secret: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("CRON_SECRET") or os.getenv("PRICE_MONITORING_CRON_SECRET") or ""
    if not expected:
        raise HTTPException(status_code=500, detail="CRON_SECRET not configured")
    if x_cron_secret != expected:
        raise HTTPException(status_code=403, detail="Invalid x-cron-secret")


@router.post("/cron-refresh")
async def cron_refresh(
    limit: int = Query(50, ge=1, le=200),
    _: None = Depends(_verify_cron_secret),
):
    """Pick due tracked_jobs and run refresh for each."""
    svc = get_job_research_service()
    try:
        res = svc.sb.rpc("get_internal_tracked_jobs_due", {"p_limit": limit}).execute()
        rows = res.data or []
    except Exception as e:
        logger.warning(f"job-cron: get_internal_tracked_jobs_due failed: {e}")
        return {"error": str(e)[:200]}

    outcomes: List[Dict[str, Any]] = []
    for r in rows:
        try:
            o = await svc.refresh(r["id"])
            outcomes.append({"tracked_job_id": r["id"], **o})
        except Exception as e:
            logger.warning(f"job-cron: refresh {r.get('id')} failed: {e}")
            outcomes.append({"tracked_job_id": r["id"], "error": str(e)[:200]})

    return {"due": len(rows), "outcomes": outcomes}


@router.post("/cron-digest")
async def cron_digest(
    current_hour_utc: int = Query(..., ge=0, le=23),
    _: None = Depends(_verify_cron_secret),
):
    """Send the consolidated daily digest for users whose digest_hour_utc matches."""
    dispatcher = get_job_digest_dispatcher()
    return await dispatcher.dispatch_due_users(current_hour_utc=current_hour_utc)
