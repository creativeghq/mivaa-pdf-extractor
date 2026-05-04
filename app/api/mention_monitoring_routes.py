"""
Mention Monitoring API Routes — internal product/brand flow + external API.

Mirror of `price_monitoring_routes.py` for the mention-monitoring path.

Internal flow (product enrollment, session JWT):
  POST   /api/v1/mention-monitoring/products/{product_id}/track
  DELETE /api/v1/mention-monitoring/products/{product_id}/track
  GET    /api/v1/mention-monitoring/products/{product_id}
  POST   /api/v1/mention-monitoring/products/{product_id}/refresh
  GET    /api/v1/mention-monitoring/products/{product_id}/feed
  GET    /api/v1/mention-monitoring/products/{product_id}/history
  GET    /api/v1/mention-monitoring/products/{product_id}/summary
  GET    /api/v1/mention-monitoring/products/{product_id}/llm-visibility
  POST   /api/v1/mention-monitoring/products/{product_id}/probe-llm

Subject-id flow (brand/keyword + admin lookups):
  POST   /api/v1/mention-monitoring/track
  GET    /api/v1/mention-monitoring/track/{tracked_mention_id}
  PUT    /api/v1/mention-monitoring/track/{tracked_mention_id}
  DELETE /api/v1/mention-monitoring/track/{tracked_mention_id}
  POST   /api/v1/mention-monitoring/track/{tracked_mention_id}/refresh
  GET    /api/v1/mention-monitoring/track/{tracked_mention_id}/feed
  GET    /api/v1/mention-monitoring/track/{tracked_mention_id}/history
  GET    /api/v1/mention-monitoring/track/{tracked_mention_id}/summary
  GET    /api/v1/mention-monitoring/track/{tracked_mention_id}/llm-visibility
  POST   /api/v1/mention-monitoring/track/{tracked_mention_id}/probe-llm
  POST   /api/v1/mention-monitoring/track/{tracked_mention_id}/exclude
  POST   /api/v1/mention-monitoring/track/{tracked_mention_id}/include
  GET    /api/v1/mention-monitoring/track/{tracked_mention_id}/exclusions
  POST   /api/v1/mention-monitoring/track/{tracked_mention_id}/promote
  GET    /api/v1/mention-monitoring/track/{tracked_mention_id}/share-of-voice

Cross-flow:
  POST   /api/v1/mention-monitoring/classifier-correction
  POST   /api/v1/mention-monitoring/cron-refresh        (x-cron-secret)
  POST   /api/v1/mention-monitoring/cron-probe-llm      (x-cron-secret)
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from app.dependencies import get_current_user, get_workspace_context
from app.middleware.jwt_auth import User, WorkspaceContext
from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.tracked_mentions_service import (
    get_tracked_mentions_service,
)
from app.services.integrations.llm_mention_probe_service import (
    build_probes, get_llm_mention_probe_service,
)
from app.services.integrations.mention_identity_service import SubjectFacets
from app.services.integrations.mention_opportunity_service import (
    get_mention_opportunity_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/mention-monitoring",
    tags=["Mention Monitoring"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not found"},
    },
)


# ============================================================================
# Helpers
# ============================================================================

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


def _require_admin(user: User) -> None:
    if not _is_admin(get_supabase_client().client, str(user.id)):
        raise HTTPException(status_code=403, detail="admin role required")


def _resolve_product(sb, product_id: str) -> Dict[str, Any]:
    prod = (
        sb.table("products")
        .select("id, name, manufacturer, metadata")
        .eq("id", product_id)
        .maybe_single()
        .execute()
    )
    row = (prod.data if prod else None) or None
    if not row:
        raise HTTPException(status_code=404, detail=f"product {product_id} not found")
    return row


def _check_owner_or_admin(sb, *, tracked_mention_id: str, user_id: str) -> Dict[str, Any]:
    r = (
        sb.table("tracked_mentions")
        .select("*")
        .eq("id", tracked_mention_id)
        .maybe_single()
        .execute()
    )
    row = (r.data if r else None) or None
    if not row:
        raise HTTPException(status_code=404, detail="tracked_mention not found")
    if str(row.get("user_id")) != str(user_id) and not _is_admin(sb, user_id):
        raise HTTPException(status_code=403, detail="not the owner")
    return row


# ============================================================================
# Request/response models
# ============================================================================

class TrackRequest(BaseModel):
    subject_type: str = Field("product", pattern="^(product|brand|keyword)$")
    subject_label: Optional[str] = None
    product_id: Optional[str] = None
    brand_name: Optional[str] = None
    aliases: Optional[List[str]] = None
    # Opt-in: when true, Haiku expands the label into per-word aliases on the
    # first refresh. Default off — discovery uses only label + supplied aliases.
    auto_expand_aliases: bool = False
    sources_enabled: Optional[Dict[str, bool]] = None
    source_config: Optional[Dict[str, Any]] = None
    language_codes: Optional[List[str]] = None
    country_codes: Optional[List[str]] = None
    refresh_interval_hours: int = 24
    recency_days: int = 30
    homepage_domain: Optional[str] = None
    alert_channels: Optional[List[str]] = None
    alert_on_spike: Optional[bool] = None
    alert_on_negative_sentiment: Optional[bool] = None
    alert_on_new_outlet: Optional[bool] = None
    alert_on_llm_visibility_change: Optional[bool] = None
    alert_webhook_url: Optional[str] = None
    run_first_refresh: bool = True


class UpdateRequest(BaseModel):
    subject_label: Optional[str] = None
    aliases: Optional[List[str]] = None
    auto_expand_aliases: Optional[bool] = None
    sources_enabled: Optional[Dict[str, bool]] = None
    source_config: Optional[Dict[str, Any]] = None
    language_codes: Optional[List[str]] = None
    country_codes: Optional[List[str]] = None
    refresh_interval_hours: Optional[int] = None
    recency_days: Optional[int] = None
    homepage_domain: Optional[str] = None
    alert_channels: Optional[List[str]] = None
    alert_on_spike: Optional[bool] = None
    alert_on_negative_sentiment: Optional[bool] = None
    alert_on_new_outlet: Optional[bool] = None
    alert_on_llm_visibility_change: Optional[bool] = None
    alert_webhook_url: Optional[str] = None
    is_active: Optional[bool] = None


class RefreshRequest(BaseModel):
    force: bool = False


class ExcludeRequest(BaseModel):
    url: Optional[str] = None
    domain: Optional[str] = None
    reason: Optional[str] = None


class PromoteRequest(BaseModel):
    url: str
    override_relevance: str = Field(..., pattern="^(exact|tangential|unverifiable)$")
    reason: Optional[str] = None


class ClassifierCorrectionRequest(BaseModel):
    mention_history_id: str
    corrected_relevance: Optional[str] = Field(None, pattern="^(exact|tangential|mismatch|unverifiable)$")
    corrected_sentiment: Optional[str] = Field(None, pattern="^(positive|neutral|negative)$")
    correction_note: Optional[str] = None


class ProbeLlmRequest(BaseModel):
    models: Optional[List[str]] = None


class OpportunitiesRequest(BaseModel):
    types: Optional[List[str]] = None
    days: int = Field(30, ge=1, le=180)
    limit_per_type: int = Field(5, ge=1, le=20)
    use_llm_summary: bool = False


# ============================================================================
# Internal flow — product
# ============================================================================

@router.post("/products/{product_id}/track")
async def track_product(
    product_id: str,
    body: Optional[TrackRequest] = None,
    user: User = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    sb = get_supabase_client().client
    product = _resolve_product(sb, product_id)
    svc = get_tracked_mentions_service()
    # Pull manufacturer from product metadata for brand_hint
    metadata = product.get("metadata") or {}
    brand = product.get("manufacturer") or metadata.get("brand") or metadata.get("manufacturer")
    aliases = (body.aliases if body else None) or []
    if not aliases and (metadata.get("sku") or metadata.get("model")):
        aliases = [str(metadata.get("sku") or metadata.get("model"))]

    row = await svc.find_or_create_for_product(
        product_id=product_id,
        product_name=product.get("name") or product_id,
        brand_name=brand,
        aliases=aliases,
        auto_expand_aliases=(body.auto_expand_aliases if body else False),
        user_id=str(user.id),
        workspace_id=getattr(workspace, "workspace_id", None) if workspace else None,
        country_codes=(body.country_codes if body else None) or [],
        run_first_refresh=(body.run_first_refresh if body else True),
    )
    # Apply alert prefs from body, if any
    if body:
        updates = body.model_dump(exclude_unset=True, exclude={"product_id", "subject_type", "subject_label", "brand_name", "run_first_refresh"})
        if updates:
            row = svc.update(row["id"], **updates) or row
    return {"success": True, "data": row}


@router.delete("/products/{product_id}/track")
async def untrack_product(
    product_id: str,
    user: User = Depends(get_current_user),
):
    svc = get_tracked_mentions_service()
    existing = svc.find_for_product(product_id)
    if not existing:
        return {"success": True, "message": "not tracked"}
    sb = get_supabase_client().client
    if str(existing.get("user_id")) != str(user.id) and not _is_admin(sb, str(user.id)):
        raise HTTPException(status_code=403, detail="not the owner")
    ok = svc.deactivate(existing["id"])
    return {"success": ok}


@router.get("/products/{product_id}")
async def get_product_monitoring(
    product_id: str,
    user: User = Depends(get_current_user),
):
    svc = get_tracked_mentions_service()
    row = svc.find_for_product(product_id)
    if not row:
        return {"success": True, "data": None}
    sb = get_supabase_client().client
    if str(row.get("user_id")) != str(user.id) and not _is_admin(sb, str(user.id)):
        raise HTTPException(status_code=403, detail="not the owner")
    return {"success": True, "data": row}


@router.post("/products/{product_id}/refresh")
async def refresh_product(
    product_id: str,
    body: Optional[RefreshRequest] = None,
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    svc = get_tracked_mentions_service()
    existing = svc.find_for_product(product_id)
    if not existing:
        product = _resolve_product(sb, product_id)
        existing = await svc.find_or_create_for_product(
            product_id=product_id,
            product_name=product.get("name") or product_id,
            user_id=str(user.id),
            run_first_refresh=False,
        )
    if str(existing.get("user_id")) != str(user.id) and not _is_admin(sb, str(user.id)):
        raise HTTPException(status_code=403, detail="not the owner")
    force = bool(body.force) if body else False
    if force and not _is_admin(sb, str(user.id)):
        raise HTTPException(status_code=403, detail="force_refresh requires admin")
    outcome = await svc.refresh(existing["id"], force=force)
    return {"success": True, "data": outcome}


@router.get("/products/{product_id}/feed")
async def get_product_feed(
    product_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    svc = get_tracked_mentions_service()
    existing = svc.find_for_product(product_id)
    if not existing:
        return {"success": True, "data": []}
    if str(existing.get("user_id")) != str(user.id) and not _is_admin(sb, str(user.id)):
        raise HTTPException(status_code=403, detail="not the owner")
    rows = svc.latest_results(existing["id"], limit=limit)
    return {"success": True, "data": rows}


@router.get("/products/{product_id}/history")
async def get_product_history(
    product_id: str,
    days: int = Query(default=30, ge=1, le=180),
    sentiment: Optional[str] = None,
    outlet_type: Optional[str] = None,
    limit: int = Query(default=200, ge=1, le=2000),
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    svc = get_tracked_mentions_service()
    existing = svc.find_for_product(product_id)
    if not existing:
        return {"success": True, "data": []}
    if str(existing.get("user_id")) != str(user.id) and not _is_admin(sb, str(user.id)):
        raise HTTPException(status_code=403, detail="not the owner")
    rows = svc.history(existing["id"], days=days, limit=limit,
                       sentiment=sentiment, outlet_type=outlet_type)
    return {"success": True, "data": rows}


@router.get("/products/{product_id}/summary")
async def get_product_summary(
    product_id: str,
    days: int = Query(default=30, ge=1, le=180),
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    svc = get_tracked_mentions_service()
    existing = svc.find_for_product(product_id)
    if not existing:
        return {"success": True, "data": None}
    if str(existing.get("user_id")) != str(user.id) and not _is_admin(sb, str(user.id)):
        raise HTTPException(status_code=403, detail="not the owner")
    summary = svc.summary(existing["id"], days=days)
    return {"success": True, "data": summary}


@router.get("/products/{product_id}/llm-visibility")
async def get_product_llm_visibility(
    product_id: str,
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    svc = get_tracked_mentions_service()
    existing = svc.find_for_product(product_id)
    if not existing:
        return {"success": True, "data": {"present": False}}
    if str(existing.get("user_id")) != str(user.id) and not _is_admin(sb, str(user.id)):
        raise HTTPException(status_code=403, detail="not the owner")
    snapshot = get_llm_mention_probe_service().visibility_snapshot(existing["id"])
    return {"success": True, "data": snapshot}


@router.post("/products/{product_id}/probe-llm")
async def probe_product_llm(
    product_id: str,
    body: Optional[ProbeLlmRequest] = None,
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _require_admin(user)
    svc = get_tracked_mentions_service()
    existing = svc.find_for_product(product_id)
    if not existing:
        product = _resolve_product(sb, product_id)
        existing = await svc.find_or_create_for_product(
            product_id=product_id,
            product_name=product.get("name") or product_id,
            user_id=str(user.id),
            run_first_refresh=False,
        )
    facets = SubjectFacets.from_dict(existing.get("subject_facets") or {
        "label": existing.get("subject_label"),
        "aliases": existing.get("aliases") or [],
        "brand": existing.get("brand_name"),
    })
    res = await get_llm_mention_probe_service().probe(
        tracked_mention_id=existing["id"],
        facets=facets,
        models=(body.models if body else None),
    )
    # After probe, check for visibility-shift alerts
    try:
        snapshot = get_llm_mention_probe_service().visibility_snapshot(
            existing["id"], run_id=res.get("probe_run_id")
        )
        from app.modules.mention_monitoring_notifications.service import (
            get_mention_alert_dispatcher,
        )
        dispatcher = get_mention_alert_dispatcher()
        cands = dispatcher.detect_after_llm_probe(
            tracked_mention_id=existing["id"], current_snapshot=snapshot,
        )
        dispatcher.dispatch(cands)
    except Exception as e:
        logger.warning(f"llm-probe alert dispatch failed: {e}")
    return {"success": True, "data": res}


# ============================================================================
# Subject-id flow
# ============================================================================

@router.post("/track")
async def create_tracked_mention(
    body: TrackRequest,
    user: User = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    if not body.subject_label and not body.product_id and not body.brand_name:
        raise HTTPException(
            status_code=400,
            detail="one of subject_label, product_id, brand_name is required",
        )
    sb = get_supabase_client().client
    label = body.subject_label
    product_id = body.product_id
    brand = body.brand_name
    if product_id and not label:
        product = _resolve_product(sb, product_id)
        label = product.get("name") or product_id
        brand = brand or product.get("manufacturer")

    svc = get_tracked_mentions_service()
    row = await svc.create(
        api_key_id=None,
        user_id=str(user.id),
        workspace_id=getattr(workspace, "workspace_id", None) if workspace else None,
        subject_type=body.subject_type,
        subject_label=label or (brand or "untitled"),
        product_id=product_id,
        brand_name=brand,
        aliases=body.aliases,
        auto_expand_aliases=body.auto_expand_aliases,
        sources_enabled=body.sources_enabled,
        source_config=body.source_config,
        language_codes=body.language_codes,
        country_codes=body.country_codes,
        refresh_interval_hours=body.refresh_interval_hours,
        recency_days=body.recency_days,
        homepage_domain=body.homepage_domain,
        alert_channels=body.alert_channels,
        alert_on_spike=body.alert_on_spike,
        alert_on_negative_sentiment=body.alert_on_negative_sentiment,
        alert_on_new_outlet=body.alert_on_new_outlet,
        alert_on_llm_visibility_change=body.alert_on_llm_visibility_change,
        alert_webhook_url=body.alert_webhook_url,
        run_first_refresh=body.run_first_refresh,
    )
    return {"success": True, "data": row}


@router.get("/track/{tracked_mention_id}")
async def get_tracked_mention(
    tracked_mention_id: str,
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    row = _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=str(user.id))
    return {"success": True, "data": row}


@router.put("/track/{tracked_mention_id}")
async def update_tracked_mention(
    tracked_mention_id: str,
    body: UpdateRequest,
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=str(user.id))
    updates = body.model_dump(exclude_unset=True)
    row = get_tracked_mentions_service().update(tracked_mention_id, **updates)
    return {"success": True, "data": row}


@router.delete("/track/{tracked_mention_id}")
async def delete_tracked_mention(
    tracked_mention_id: str,
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=str(user.id))
    ok = get_tracked_mentions_service().deactivate(tracked_mention_id)
    return {"success": ok}


@router.post("/track/{tracked_mention_id}/refresh")
async def refresh_tracked_mention(
    tracked_mention_id: str,
    body: Optional[RefreshRequest] = None,
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=str(user.id))
    force = bool(body.force) if body else False
    if force and not _is_admin(sb, str(user.id)):
        raise HTTPException(status_code=403, detail="force_refresh requires admin")
    outcome = await get_tracked_mentions_service().refresh(tracked_mention_id, force=force)
    return {"success": True, "data": outcome}


@router.get("/track/{tracked_mention_id}/feed")
async def get_tracked_feed(
    tracked_mention_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=str(user.id))
    rows = get_tracked_mentions_service().latest_results(tracked_mention_id, limit=limit)
    return {"success": True, "data": rows}


@router.get("/track/{tracked_mention_id}/history")
async def get_tracked_history(
    tracked_mention_id: str,
    days: int = Query(default=30, ge=1, le=180),
    sentiment: Optional[str] = None,
    outlet_type: Optional[str] = None,
    limit: int = Query(default=200, ge=1, le=2000),
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=str(user.id))
    rows = get_tracked_mentions_service().history(
        tracked_mention_id, days=days, sentiment=sentiment, outlet_type=outlet_type, limit=limit,
    )
    return {"success": True, "data": rows}


@router.get("/track/{tracked_mention_id}/summary")
async def get_tracked_summary(
    tracked_mention_id: str,
    days: int = Query(default=30, ge=1, le=180),
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=str(user.id))
    summary = get_tracked_mentions_service().summary(tracked_mention_id, days=days)
    return {"success": True, "data": summary}


@router.get("/track/{tracked_mention_id}/llm-visibility")
async def get_tracked_llm_visibility(
    tracked_mention_id: str,
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=str(user.id))
    snapshot = get_llm_mention_probe_service().visibility_snapshot(tracked_mention_id)
    return {"success": True, "data": snapshot}


@router.post("/track/{tracked_mention_id}/probe-llm")
async def probe_tracked_llm(
    tracked_mention_id: str,
    body: Optional[ProbeLlmRequest] = None,
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _require_admin(user)
    row = _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=str(user.id))
    facets = SubjectFacets.from_dict(row.get("subject_facets") or {
        "label": row.get("subject_label"),
        "aliases": row.get("aliases") or [],
        "brand": row.get("brand_name"),
    })
    res = await get_llm_mention_probe_service().probe(
        tracked_mention_id=tracked_mention_id,
        facets=facets,
        models=(body.models if body else None),
    )
    return {"success": True, "data": res}


@router.post("/track/{tracked_mention_id}/exclude")
async def exclude_url(
    tracked_mention_id: str,
    body: ExcludeRequest,
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=str(user.id))
    if not body.url and not body.domain:
        raise HTTPException(status_code=400, detail="url or domain required")
    out = get_tracked_mentions_service().add_exclusion(
        tracked_mention_id, url=body.url, domain=body.domain,
        reason=body.reason, user_id=str(user.id),
    )
    return {"success": True, "data": out}


@router.post("/track/{tracked_mention_id}/include")
async def include_url(
    tracked_mention_id: str,
    body: ExcludeRequest,
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=str(user.id))
    removed = get_tracked_mentions_service().remove_exclusion(
        tracked_mention_id, url=body.url, domain=body.domain,
    )
    return {"success": True, "removed_count": removed}


@router.get("/track/{tracked_mention_id}/exclusions")
async def list_exclusions(
    tracked_mention_id: str,
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=str(user.id))
    rows = get_tracked_mentions_service().list_exclusions(tracked_mention_id)
    return {"success": True, "data": rows}


@router.post("/track/{tracked_mention_id}/promote")
async def promote_url(
    tracked_mention_id: str,
    body: PromoteRequest,
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _require_admin(user)
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=str(user.id))
    out = get_tracked_mentions_service().add_promoted_url(
        tracked_mention_id,
        url=body.url, override_relevance=body.override_relevance,
        reason=body.reason, user_id=str(user.id),
    )
    return {"success": True, "data": out}


@router.get("/track/{tracked_mention_id}/share-of-voice")
async def share_of_voice(
    tracked_mention_id: str,
    days: int = Query(default=30, ge=1, le=180),
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=str(user.id))
    # Aggregate competitor-mentions across LLM probes for this subject
    try:
        r = (
            sb.table("llm_mention_probes")
            .select("competitors_mentioned, run_at")
            .eq("tracked_mention_id", tracked_mention_id)
            .order("run_at", desc=True)
            .limit(500)
            .execute()
        )
        rows = r.data or []
    except Exception:
        rows = []
    counts: Dict[str, int] = {}
    for r in rows:
        for c in r.get("competitors_mentioned") or []:
            cn = (c or "").strip()
            if cn:
                counts[cn] = counts.get(cn, 0) + 1
    return {"success": True, "data": {
        "tracked_mention_id": tracked_mention_id,
        "competitor_mentions": [
            {"name": k, "count": v}
            for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
        ],
    }}


@router.post("/products/{product_id}/opportunities")
async def get_product_opportunities(
    product_id: str,
    body: Optional[OpportunitiesRequest] = None,
    user: User = Depends(get_current_user),
):
    """Generate content + outreach opportunities for a product's tracked mentions."""
    sb = get_supabase_client().client
    svc = get_tracked_mentions_service()
    existing = svc.find_for_product(product_id)
    if not existing:
        return {"success": True, "data": {
            "tracked_mention_id": None, "opportunities": [],
            "errors": {"subject": "product is not enrolled in mention monitoring"},
        }}
    if str(existing.get("user_id")) != str(user.id) and not _is_admin(sb, str(user.id)):
        raise HTTPException(status_code=403, detail="not the owner")
    body = body or OpportunitiesRequest()
    out = await get_mention_opportunity_service().generate(
        tracked_mention_id=existing["id"],
        types=body.types,
        days=body.days,
        limit_per_type=body.limit_per_type,
        use_llm_summary=body.use_llm_summary,
    )
    return {"success": True, "data": out}


@router.post("/track/{tracked_mention_id}/opportunities")
async def get_tracked_opportunities(
    tracked_mention_id: str,
    body: Optional[OpportunitiesRequest] = None,
    user: User = Depends(get_current_user),
):
    """Generate content + outreach opportunities for any tracked subject."""
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=str(user.id))
    body = body or OpportunitiesRequest()
    out = await get_mention_opportunity_service().generate(
        tracked_mention_id=tracked_mention_id,
        types=body.types,
        days=body.days,
        limit_per_type=body.limit_per_type,
        use_llm_summary=body.use_llm_summary,
    )
    return {"success": True, "data": out}


# ============================================================================
# Cross-flow
# ============================================================================

@router.post("/classifier-correction")
async def classifier_correction(
    body: ClassifierCorrectionRequest,
    user: User = Depends(get_current_user),
):
    sb = get_supabase_client().client
    # Look up the row, copy snapshot fields for few-shot examples
    r = (
        sb.table("mention_history")
        .select("tracked_mention_id, outlet_domain, url, title, relevance, sentiment")
        .eq("id", body.mention_history_id)
        .maybe_single()
        .execute()
    )
    row = (r.data if r else None) or None
    if not row:
        raise HTTPException(status_code=404, detail="mention_history row not found")
    _check_owner_or_admin(sb, tracked_mention_id=row["tracked_mention_id"], user_id=str(user.id))
    sb.table("mention_match_corrections").insert({
        "tracked_mention_id": row["tracked_mention_id"],
        "mention_history_id": body.mention_history_id,
        "outlet_domain": row.get("outlet_domain"),
        "url": row.get("url"),
        "title": row.get("title"),
        "original_relevance": row.get("relevance"),
        "corrected_relevance": body.corrected_relevance,
        "original_sentiment": row.get("sentiment"),
        "corrected_sentiment": body.corrected_sentiment,
        "correction_note": body.correction_note,
        "created_by": str(user.id),
    }).execute()
    # Apply the correction directly to the row so the UI updates immediately
    if body.corrected_relevance or body.corrected_sentiment:
        patch: Dict[str, Any] = {"manual_override": True}
        if body.corrected_relevance:
            patch["relevance"] = body.corrected_relevance
        if body.corrected_sentiment:
            patch["sentiment"] = body.corrected_sentiment
        sb.table("mention_history").update(patch).eq("id", body.mention_history_id).execute()
    return {"success": True}


@router.post("/cron-refresh")
async def cron_refresh(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
):
    """Cron-target batch refresh. Authentication via x-cron-secret header."""
    secret = request.headers.get("x-cron-secret")
    expected = os.getenv("CRON_SECRET")
    if not expected or secret != expected:
        raise HTTPException(status_code=401, detail="bad cron secret")
    sb = get_supabase_client().client
    try:
        r = sb.rpc("get_internal_tracked_mentions_due", {"p_limit": limit}).execute()
        due = r.data or []
    except Exception as e:
        return {"success": False, "error": str(e)}
    svc = get_tracked_mentions_service()
    processed = succeeded = failed = 0
    results: List[Dict[str, Any]] = []
    for row in due:
        try:
            outcome = await svc.refresh(row["id"], force=False)
            processed += 1
            if outcome.get("status") == "refreshed":
                succeeded += 1
            else:
                failed += 1
            results.append({"id": row["id"], "status": outcome.get("status"),
                            "credits": outcome.get("credits_used"),
                            "hits_count": outcome.get("hits_count")})
        except Exception as e:
            failed += 1
            processed += 1
            results.append({"id": row["id"], "status": "error", "error": str(e)[:200]})
    return {"success": True, "due_count": len(due),
            "processed": processed, "succeeded": succeeded, "failed": failed,
            "results": results}


@router.post("/cron-probe-llm")
async def cron_probe_llm(
    request: Request,
    limit: int = Query(default=25, ge=1, le=100),
    min_age_days: int = Query(default=7, ge=1, le=30),
):
    secret = request.headers.get("x-cron-secret")
    expected = os.getenv("CRON_SECRET")
    if not expected or secret != expected:
        raise HTTPException(status_code=401, detail="bad cron secret")
    sb = get_supabase_client().client
    try:
        r = sb.rpc("get_tracked_mentions_due_for_llm_probe", {
            "p_limit": limit, "p_min_age_days": min_age_days,
        }).execute()
        due = r.data or []
    except Exception as e:
        return {"success": False, "error": str(e)}
    svc = get_tracked_mentions_service()
    probe = get_llm_mention_probe_service()
    processed = succeeded = failed = 0
    for row in due:
        tm_id = row["id"]
        try:
            full = svc.get(tm_id) or {}
            facets = SubjectFacets.from_dict(full.get("subject_facets") or {
                "label": full.get("subject_label"),
                "aliases": full.get("aliases") or [],
                "brand": full.get("brand_name"),
            })
            await probe.probe(tracked_mention_id=tm_id, facets=facets)
            succeeded += 1
        except Exception as e:
            failed += 1
            logger.warning(f"cron-probe-llm subject {tm_id} failed: {e}")
        processed += 1
    return {"success": True, "due_count": len(due), "processed": processed,
            "succeeded": succeeded, "failed": failed}


# ============================================================================
# Stateless opportunities — no DB row, used by SEO pipeline edge functions
# ============================================================================

class StatelessOpportunitiesRequest(BaseModel):
    """Body for `POST /opportunities-stateless`. Bypasses the persisted
    `tracked_mentions` row entirely. Used by the SEO pipeline (seo-research)
    so a content-research run doesn't have to spawn an ephemeral row.

    Auth: `x-cron-secret` header (same secret as the cron endpoints) — this
    endpoint is internal-only, called by edge functions on the platform's
    own infrastructure. Not exposed to external API consumers.
    """
    subject_label: str = Field(..., min_length=1, max_length=200,
                               description="The keyword / topic / brand to research.")
    brand_name: Optional[str] = Field(None, description="Brand string when known.")
    aliases: Optional[List[str]] = Field(None, description="Alternate strings to fall back on if the label has no SERP data.")
    language_codes: Optional[List[str]] = Field(None, description="e.g. ['en'], ['el','en']. Default ['en'].")
    country_codes: Optional[List[str]] = Field(None, description="ISO-3166 alpha-2. e.g. ['US'], ['GR']. Default ['US'].")
    homepage_domain: Optional[str] = Field(None, description="Brand homepage domain — required for `domain_snapshot` type.")
    types: Optional[List[str]] = Field(None, description="Subset of opportunity types. Default = all subject-driven (mention-derived auto-skip).")
    limit_per_type: int = Field(5, ge=1, le=20)
    use_llm_summary: bool = Field(False, description="When true, Haiku polishes rationales/actions.")


@router.post("/opportunities-stateless")
async def opportunities_stateless(
    request: Request,
    body: StatelessOpportunitiesRequest,
):
    """Generate opportunities for an inline subject — no DB row required.

    Mention-derived types (`trending_topic`, `outlet_pitch`, `author_relationship`,
    `sentiment_response`, `llm_visibility`) auto-skip in this mode since they
    need data that only exists on a real tracked subject.
    """
    secret = request.headers.get("x-cron-secret")
    expected = os.getenv("CRON_SECRET")
    if not expected or secret != expected:
        raise HTTPException(status_code=401, detail="bad cron secret")
    subject_override = {
        "subject_label": body.subject_label,
        "brand_name": body.brand_name,
        "aliases": body.aliases or [],
        "language_codes": body.language_codes or ["en"],
        "country_codes": body.country_codes or ["US"],
        "homepage_domain": body.homepage_domain,
    }
    out = await get_mention_opportunity_service().generate(
        subject_override=subject_override,
        types=body.types,
        limit_per_type=body.limit_per_type,
        use_llm_summary=body.use_llm_summary,
    )
    return {"success": True, "data": out}
