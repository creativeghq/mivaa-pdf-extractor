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

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app.dependencies import get_current_user, get_workspace_context, require_admin
from app.middleware.jwt_auth import User, WorkspaceContext
from app.modules.job_research_notifications.service import get_job_digest_dispatcher
from app.services.integrations.job_research_service import get_job_research_service
from app.services.integrations import job_cost_logger as costs
from app.services.integrations.cron_billing import charge_cron
from app.utils.paid_door import metered_door
from app.services.integrations.job_sites_kb_sync import sync_one_site_type as _sync_kb

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
    careers_page_urls: Optional[List[str]] = Field(None, description="Per-tracked_job careers pages. UNIONed with operator-curated global defaults (job_research_sites WHERE site_type='careers_page_default') at refresh time.")
    rss_feed_urls: Optional[List[str]] = Field(None, description="Per-tracked_job RSS/Atom feeds. UNIONed with operator-curated global defaults (job_research_sites WHERE site_type='rss_feed_default') at refresh time.")
    digest_hour_utc: int = Field(7, ge=0, le=23)
    digest_day_of_week: Optional[int] = Field(None, ge=0, le=6, description="0=Sunday..6=Saturday. NULL = daily.")
    alert_channels: Optional[List[str]] = None
    alert_webhook_url: Optional[str] = None
    refresh_interval_hours: int = Field(24, ge=1, le=168)
    max_age_days: int = Field(14, ge=1, le=365, description="Recency gate: drop listings older than this (and undated aggregator listings). Default 14.")
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
    rss_feed_urls: Optional[List[str]] = None
    digest_enabled: Optional[bool] = None
    digest_hour_utc: Optional[int] = Field(None, ge=0, le=23)
    digest_day_of_week: Optional[int] = Field(None, ge=0, le=6)
    alert_channels: Optional[List[str]] = None
    alert_webhook_url: Optional[str] = None
    # Burst alerting. The notifier implements this fully (threshold + 2h cooldown), but the
    # flags had no write path at all, so the feature was unreachable code. (audit #305)
    alert_on_burst: Optional[bool] = None
    burst_threshold: Optional[int] = Field(None, ge=1, le=1000)
    refresh_interval_hours: Optional[int] = Field(None, ge=1, le=168)
    max_age_days: Optional[int] = Field(None, ge=1, le=365)
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


# ─── job_research_sites (operator-curated job-board list) ────────────────

class JobSiteCreateRequest(BaseModel):
    site_type: str = Field(..., description="perplexity_domain | rss_feed_default | careers_page_default")
    url_or_domain: str = Field(..., min_length=1, max_length=400)
    display_name: Optional[str] = None
    country_code: Optional[str] = Field(None, max_length=3)
    category: Optional[str] = Field(None, max_length=40)
    is_enabled: bool = True
    notes: Optional[str] = None


class JobSiteUpdateRequest(BaseModel):
    url_or_domain: Optional[str] = None
    display_name: Optional[str] = None
    country_code: Optional[str] = None
    category: Optional[str] = None
    is_enabled: Optional[bool] = None
    notes: Optional[str] = None


# ─── CRUD ────────────────────────────────────────────────────────────────

@router.post("/track")
async def create_tracked_job(
    body: CreateTrackedJobRequest,
    user: User = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    """Create a tracked job search. Debits the first refresh BEFORE running it.

    `run_first_refresh` defaults to True, so this door triggers paid discovery on the
    usual path and was charging nothing for it (#21 M8-3). Its partner-key twin in
    `job_tracking_routes` has metered this correctly all along — the internal route is
    the copy that never got it.
    """
    debit_amount = costs.JOB_OP_CREDIT_COST.get("refresh", 5) if body.run_first_refresh else 0
    async with metered_door(
        user_id=str(user.get("sub")),
        workspace_id=str(workspace.id) if workspace else None,
        cost=debit_amount,
        operation_type="job_research.refresh",
        debit=costs.debit_credits, refund=costs.refund_credits,
    ) as paid:
        svc = get_job_research_service()
        try:
            row = await svc.create(
                owner_user_id=str(user.get("sub")),
                workspace_id=str(workspace.id) if workspace else None,
                **body.model_dump(exclude_none=True),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Refund a true no-op, matching the partner route (audit #217 H15): an explicit
        # error, or candidates were found and the classifier persisted none of them.
        fr = (row.get("first_refresh") or {})
        if debit_amount and (
            fr.get("error")
            or (fr.get("candidates_after_exclusions", 0) > 0 and fr.get("persisted", 0) == 0)
        ):
            paid.refund("first refresh produced nothing")
    return {"tracked_job": row, "credits_debited": paid.charged}


@router.post("/track/{tracked_job_id}/regenerate-keywords")
async def regenerate_keywords(
    tracked_job_id: str,
    user: User = Depends(get_current_user),
):
    """Re-run Haiku keyword expansion. Returns the new expanded list + the rejected suggestions.

    Metered (#21 M8-3): this makes a model call, and made it for free.
    """
    svc = get_job_research_service()
    owner = svc.get(tracked_job_id, owner_user_id=str(user.get("sub")))
    if not owner:
        # Checked before the debit so a caller is never charged for someone else's id.
        raise HTTPException(status_code=404, detail="Not found")
    async with metered_door(
        user_id=str(user.get("sub")),
        workspace_id=owner.get("workspace_id"),
        cost=costs.JOB_OP_CREDIT_COST.get("regenerate_keywords", 1),
        operation_type="job_research.regenerate_keywords",
        debit=costs.debit_credits, refund=costs.refund_credits,
    ) as paid:
        try:
            result = await svc.regenerate_keywords(tracked_job_id, owner_user_id=str(user.get("sub")))
        except RuntimeError as e:
            raise HTTPException(status_code=404, detail=str(e))
        if not (result or {}).get("expanded"):
            paid.refund("keyword expansion returned nothing")
    return {**result, "credits_debited": paid.charged}


@router.get("/track")
async def list_tracked_jobs(
    only_active: bool = Query(True),
    user: User = Depends(get_current_user),
):
    svc = get_job_research_service()
    return {"tracked_jobs": svc.list_for_user(str(user.get("sub")), only_active=only_active)}


@router.get("/track/{tracked_job_id}")
async def get_tracked_job(
    tracked_job_id: str,
    user: User = Depends(get_current_user),
):
    svc = get_job_research_service()
    # get_readable: own job OR a job owned by an api_key this user owns (external
    # flow surfaced into the premade pages).
    row = svc.get_readable(tracked_job_id, str(user.get("sub")))
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
    row = svc.update(tracked_job_id, str(user.get("sub")), body.model_dump(exclude_none=True))
    return {"tracked_job": row}


@router.delete("/track/{tracked_job_id}")
async def delete_tracked_job(
    tracked_job_id: str,
    user: User = Depends(get_current_user),
):
    svc = get_job_research_service()
    ok = svc.deactivate(tracked_job_id, str(user.get("sub")))
    if not ok:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


# ─── Refresh + listings + summary ────────────────────────────────────────

@router.post("/track/{tracked_job_id}/refresh", summary="Force a refresh (debits 5 credits)")
async def refresh_tracked_job(
    tracked_job_id: str,
    force: bool = Query(False),
    force_full_discovery: bool = Query(False),
    user: User = Depends(get_current_user),
):
    """Session-JWT twin of `POST /api/v1/job-tracking/{id}/refresh`.

    Both doors run the identical paid fan-out — DataForSEO SERP, Perplexity Sonar, Firecrawl
    career-page scrapes, Haiku classification. The partner door has debited 5 credits up front
    since it shipped; this one debited nothing, so the cheapest way to spend the operator's
    DataForSEO/Firecrawl/Anthropic budget was to press "Refresh" in the app rather than call the
    API. Two doors onto one paid operation, one of which checks — the shape that keeps costing
    this platform money.

    Metering mirrors the partner route (invariant 10: debit BEFORE the upstream call, refund
    when the work could not be delivered), with one addition — the debit passes `workspace_id`,
    so a funded workspace pool pays before the member's personal balance.
    """
    user_id = user.get("sub")
    if not user_id:
        # No payer means no debit. Refuse rather than fall through to a free paid refresh —
        # "fail open on a missing identity" is how a metered route becomes an unmetered one.
        raise HTTPException(status_code=401, detail="Unauthenticated")
    user_id = str(user_id)

    svc = get_job_research_service()
    # Ownership check. Keep the row: it carries the workspace the charge should route to.
    job = svc.get(tracked_job_id, owner_user_id=user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")

    workspace_id = job.get("workspace_id")
    debit_amount = costs.JOB_OP_CREDIT_COST.get("refresh", 5)

    if not costs.debit_credits(
        user_id=user_id, amount=debit_amount,
        operation_type="job_research.refresh", workspace_id=workspace_id,
    ):
        raise HTTPException(status_code=402, detail="Insufficient credits")

    def _refund() -> None:
        costs.refund_credits(
            user_id=user_id, amount=debit_amount,
            operation_type="job_research.refresh", workspace_id=workspace_id,
        )

    try:
        outcome = await svc.refresh(tracked_job_id, force=force, force_full_discovery=force_full_discovery)
    except Exception as e:
        _refund()
        raise HTTPException(status_code=500, detail=str(e)[:200])

    # Refund cases, identical to the partner route: nothing ran, it errored, or the classifier
    # dropped every candidate it found (paid the upstreams, delivered zero listings). A genuine
    # "nothing new" run short-circuits without `candidates_after_exclusions` and keeps the credit,
    # because the upstream calls really did happen.
    if outcome.get("skipped") or outcome.get("error") or (
        outcome.get("candidates_after_exclusions", 0) > 0 and outcome.get("persisted", 0) == 0
    ):
        _refund()
        return {**outcome, "credits_debited": 0}

    return {**outcome, "credits_debited": debit_amount}


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
    if not svc.get_readable(tracked_job_id, str(user.get("sub"))):
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
    if not svc.get_readable(tracked_job_id, str(user.get("sub"))):
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
    if not svc.get(tracked_job_id, owner_user_id=str(user.get("sub"))):
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
    if not svc.get(tracked_job_id, owner_user_id=str(user.get("sub"))):
        raise HTTPException(status_code=404, detail="Not found")
    return {"exclusions": svc.list_exclusions(tracked_job_id)}


@router.delete("/exclusions/{exclusion_id}")
async def remove_exclusion(
    exclusion_id: str,
    user: User = Depends(get_current_user),
):
    # Ownership is checked in the service, via the parent tracked_job (#21 M8-1). The
    # comment that was here said "RLS enforces ownership"; there is no connection in
    # this process for a row-level policy to apply to.
    svc = get_job_research_service()
    ok = svc.remove_exclusion(exclusion_id, owner_user_id=str(user.get("sub")))
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
        row = svc.mark_listing(listing_id, action=body.action, user_id=str(user.get("sub")), notes=body.notes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError:
        # 404, not 403: a listing that belongs to someone else must be indistinguishable
        # from one that does not exist, or the id becomes an enumeration oracle.
        raise HTTPException(status_code=404, detail="Not found")
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
        # Verify ownership via the parent tracked_job. (The tail of this comment used
        # to read "svc already RLS-bounded" — it is not: this process connects as
        # service role. The `get_readable` check below is the real gate, and unlike its
        # three siblings this route always had one.)
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

        # Confirm tracked_job is readable by the user (own job OR via owned api_key)
        owner_check = svc.get_readable(listing_row["tracked_job_id"], str(user.get("sub")))
        if not owner_check:
            # 404, not 403 (invariant 1): "not your listing" confirms the id exists.
            raise HTTPException(status_code=404, detail="listing not found")

        svc.sb.table("job_match_corrections").insert({
            "tracked_job_id": listing_row["tracked_job_id"],
            "job_listing_id": listing_id,
            "user_id": str(user.get("sub")),
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

def _module_enabled(sb, slug: str) -> bool:
    """Is a platform module enabled in public.modules? Mirrors the edge-side isModuleEnabled.
    FAIL-OPEN on missing row / read error — the edge cron is the authoritative fail-closed
    gate, so this chokepoint check only converts an explicit enabled=false into a skip.
    """
    try:
        res = sb.table("modules").select("enabled").eq("slug", slug).maybe_single().execute()
        row = res.data if res else None
        return True if row is None else bool(row.get("enabled"))
    except Exception:
        return True


def _verify_cron_secret(x_cron_secret: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("CRON_SECRET") or os.getenv("PRICE_MONITORING_CRON_SECRET") or ""
    if not expected:
        raise HTTPException(status_code=500, detail="CRON_SECRET not configured")
    if x_cron_secret != expected:
        raise HTTPException(status_code=403, detail="Invalid x-cron-secret")


# ─── Sites configuration (operator-curated list of job-board sites) ──────
# These endpoints back the hidden admin page at /admin/knowledge-base/job-sources.
#
# The write routes take `require_admin` (#21 M8-1). They used to take only
# `get_current_user` under a comment claiming "RLS on `job_research_sites` enforces
# admin-only writes". The policy is real and well-formed — and void here, because MIVAA
# connects as service role, which bypasses row-level security entirely. So any
# authenticated user could add, alter or delete rows in a PLATFORM-WIDE operator-curated
# list that feeds scheduled discovery defaults for everyone.
#
# The tell was in the error messages: both 404s said "no permission (admin-only writes)"
# — written by someone who believed a check was happening one layer down. "RLS enforces
# X" is never a valid justification in this codebase; there is no connection here for it
# to apply to.
#
# Reads stay open to any authenticated user: the list is operator-curated, not secret.

@router.get("/sites")
async def list_job_sites(
    site_type: Optional[str] = Query(None, description="Filter by site_type"),
    user: User = Depends(get_current_user),
):
    svc = get_job_research_service()
    q = svc.sb.table("job_research_sites").select("*").order("site_type").order("url_or_domain")
    if site_type:
        q = q.eq("site_type", site_type)
    return {"sites": q.execute().data or []}


@router.post("/sites")
async def create_job_site(
    body: JobSiteCreateRequest,
    user: User = Depends(require_admin),
):
    if body.site_type not in ("perplexity_domain", "rss_feed_default", "careers_page_default"):
        raise HTTPException(status_code=400, detail="invalid site_type")
    svc = get_job_research_service()
    try:
        res = svc.sb.table("job_research_sites").insert({
            "site_type": body.site_type,
            "url_or_domain": body.url_or_domain.strip().lower() if body.site_type == "perplexity_domain" else body.url_or_domain.strip(),
            "display_name": body.display_name,
            "country_code": (body.country_code or "").upper() or None,
            "category": body.category,
            "is_enabled": body.is_enabled,
            "notes": body.notes,
            "created_by": str(user.get("sub")),
        }).execute()
    except Exception as e:
        msg = str(e)
        if "duplicate" in msg.lower() or "unique" in msg.lower():
            raise HTTPException(status_code=409, detail="site already exists for this site_type")
        raise HTTPException(status_code=500, detail=msg[:200])
    _sync_kb(body.site_type)  # mirror into the 'Job Sources' KB category
    return {"site": (res.data or [{}])[0]}


class JobSitesBulkCreateRequest(BaseModel):
    site_type: str = Field(..., description="perplexity_domain | rss_feed_default | careers_page_default")
    urls: List[str] = Field(..., min_length=1, max_length=200, description="One URL or domain per item. Newline-split happens on the frontend; backend takes the array.")
    country_code: Optional[str] = Field(None, max_length=3)
    category: Optional[str] = Field(None, max_length=40)
    notes: Optional[str] = None


@router.post("/sites/_resync")
async def resync_job_sites_kb_doc(
    user: User = Depends(require_admin),
):
    """v0.5.1: trigger the KB doc resync. Called by the frontend after it does
    a direct-Supabase CRUD on `job_research_sites`. Cheap (~50ms — one DB
    SELECT + one kb_docs UPDATE). Returns the sections that were touched.

    Admin-only for the same reason as its siblings (#21 M8-1): it rewrites a
    PLATFORM-WIDE KB document. The direct-Supabase CRUD it follows runs in the browser
    under the user's own key, where the admin-write policy genuinely applies — so the
    only legitimate caller of this is already an admin.
    """
    try:
        from app.services.integrations.job_sites_kb_sync import sync_all
        result = sync_all()
        return {"ok": True, "result": result}
    except Exception as e:
        logger.warning(f"sites _resync failed (non-fatal): {e}")
        return {"ok": False, "error": str(e)[:200]}


@router.post("/sites/bulk")
async def create_job_sites_bulk(
    body: JobSitesBulkCreateRequest,
    user: User = Depends(require_admin),
):
    """v0.5: bulk-insert multiple sites at once into the platform-wide default list.
    Skips entries that already exist (idempotent). Syncs the KB doc ONCE at the
    end instead of per-row for efficiency. Returns counts of created/skipped/failed."""
    if body.site_type not in ("perplexity_domain", "rss_feed_default", "careers_page_default"):
        raise HTTPException(status_code=400, detail="invalid site_type")
    svc = get_job_research_service()
    cleaned: List[str] = []
    seen: set = set()
    for raw in body.urls:
        u = (raw or "").strip()
        if body.site_type == "perplexity_domain":
            u = u.lower()
        if not u:
            continue
        key = u.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(u)
    if not cleaned:
        raise HTTPException(status_code=400, detail="No valid URLs / domains after trimming and dedup")

    created = 0
    skipped = 0
    failed: List[Dict[str, Any]] = []
    for u in cleaned:
        try:
            svc.sb.table("job_research_sites").insert({
                "site_type": body.site_type,
                "url_or_domain": u,
                "country_code": (body.country_code or "").upper() or None,
                "category": body.category,
                "notes": body.notes,
                "is_enabled": True,
                "created_by": str(user.get("sub")),
            }).execute()
            created += 1
        except Exception as e:
            msg = str(e).lower()
            if "duplicate" in msg or "unique" in msg or "23505" in msg:
                skipped += 1
            else:
                failed.append({"url_or_domain": u, "error": str(e)[:200]})

    # Sync the KB doc ONCE at the end (not per row)
    try:
        _sync_kb(body.site_type)
    except Exception as e:
        logger.warning(f"bulk-add: kb sync at end failed: {e}")

    return {
        "site_type": body.site_type,
        "requested": len(cleaned),
        "created": created,
        "skipped": skipped,
        "failed": failed,
    }


@router.put("/sites/{site_id}")
async def update_job_site(
    site_id: str,
    body: JobSiteUpdateRequest,
    user: User = Depends(require_admin),
):
    patch = body.model_dump(exclude_none=True)
    if not patch:
        return {"site": None, "no_op": True}
    svc = get_job_research_service()
    if "country_code" in patch and patch["country_code"]:
        patch["country_code"] = patch["country_code"].upper()
    res = svc.sb.table("job_research_sites").update(patch).eq("id", site_id).execute()
    row = (res.data or [None])[0]
    if not row:
        raise HTTPException(status_code=404, detail="site not found")
    if row.get("site_type"):
        _sync_kb(row["site_type"])
    return {"site": row}


@router.delete("/sites/{site_id}")
async def delete_job_site(
    site_id: str,
    user: User = Depends(require_admin),
):
    svc = get_job_research_service()
    # Read first so we know which site_type to re-sync after deletion
    pre = svc.sb.table("job_research_sites").select("site_type").eq("id", site_id).maybe_single().execute()
    pre_row = (pre.data if pre else None) or {}
    res = svc.sb.table("job_research_sites").delete().eq("id", site_id).execute()
    if not (res.data or []):
        raise HTTPException(status_code=404, detail="site not found")
    if pre_row.get("site_type"):
        _sync_kb(pre_row["site_type"])
    return {"ok": True}


@router.post("/cron-refresh")
async def cron_refresh(
    limit: int = Query(50, ge=1, le=200),
    _: None = Depends(_verify_cron_secret),
):
    """Pick due tracked_jobs and run refresh for each."""
    svc = get_job_research_service()
    # Honor the module toggle (defense-in-depth; the edge cron gates too).
    if not _module_enabled(svc.sb, "job-research"):
        return {"skipped": "module_disabled", "due": 0, "outcomes": []}
    try:
        res = svc.sb.rpc("get_internal_tracked_jobs_due", {"p_limit": limit}).execute()
        rows = res.data or []
    except Exception as e:
        logger.warning(f"job-cron: get_internal_tracked_jobs_due failed: {e}")
        return {"error": str(e)[:200]}

    # Resolve owners for metering (the due RPC doesn't return them).
    ids = [r["id"] for r in rows if r.get("id")]
    owner_by_id: Dict[str, Dict[str, Any]] = {}
    if ids:
        try:
            ores = svc.sb.table("tracked_jobs").select("id, workspace_id, user_id").in_("id", ids).execute()
            owner_by_id = {o["id"]: o for o in (ores.data or [])}
        except Exception as e:
            # Fail CLOSED on the cron path (#21 M8-2). This used to log
            # "metering fails open" and carry on: every owner would then be {}, every
            # charge would take the no-payer branch, and the whole due batch would run
            # as free provider spend. Failing open is a defensible default for a USER
            # request, where blocking a paying customer is the greater harm; cron is the
            # highest-volume caller here, unattended, and nobody is waiting on it — so a
            # billing-infrastructure outage must stop the spend, not silently absorb it.
            logger.error(f"job-cron: owner lookup failed — refusing to refresh unmetered: {e}")
            return {
                "error": "owner_lookup_failed",
                "due": len(rows),
                "refreshed": 0,
                "detail": str(e)[:200],
            }

    outcomes: List[Dict[str, Any]] = []
    skipped_unpaid = 0
    skipped_no_payer = 0
    for r in rows:
        owner = owner_by_id.get(r["id"], {})
        if not owner.get("workspace_id") and not owner.get("user_id"):
            # The lookup SUCCEEDED and this subject still resolves to nobody. On a user
            # request charge_cron proceeds and records `no_payer`; on the unattended
            # loop that is a standing invitation to spend forever on a row nobody owns.
            skipped_no_payer += 1
            outcomes.append({"tracked_job_id": r["id"], "status": "skipped_no_payer"})
            continue
        # Meter the owner BEFORE the paid refresh. Registered cron_key
        # 'job-research-refresh' (3 cr). charge_cron fails CLOSED when the charge itself
        # fails; the no-payer case it would otherwise wave through is handled above.
        if not charge_cron(
            svc.sb, "job-research-refresh",
            workspace_id=owner.get("workspace_id"), user_id=owner.get("user_id"),
            description="Tracked-job search refresh",
            # Attributes the charge in credit_transactions.metadata so
            # stamp_job_refresh_cost can total it per subject. Without this the
            # ledger knew only {cron_key, workspace_id, units} and
            # total_partner_credits_debited was unfillable (audit #305 finding 6).
            subject={"tracked_job_id": str(r["id"])},
        ):
            skipped_unpaid += 1
            outcomes.append({"tracked_job_id": r["id"], "status": "skipped_insufficient_credits"})
            continue
        try:
            o = await svc.refresh(r["id"])
            outcomes.append({"tracked_job_id": r["id"], **o})
        except Exception as e:
            logger.warning(f"job-cron: refresh {r.get('id')} failed: {e}")
            outcomes.append({"tracked_job_id": r["id"], "error": str(e)[:200]})

    return {
        "due": len(rows),
        "skipped_insufficient_credits": skipped_unpaid,
        "skipped_no_payer": skipped_no_payer,
        "outcomes": outcomes,
    }


@router.post("/cron-digest")
async def cron_digest(
    current_hour_utc: int = Query(..., ge=0, le=23),
    _: None = Depends(_verify_cron_secret),
):
    """Send the consolidated daily digest for users whose digest_hour_utc matches."""
    dispatcher = get_job_digest_dispatcher()
    return await dispatcher.dispatch_due_users(current_hour_utc=current_hour_utc)
