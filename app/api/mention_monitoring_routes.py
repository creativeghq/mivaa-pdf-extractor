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
  GET    /api/v1/mention-monitoring/products/{product_id}/llm-visibility-trend
  GET    /api/v1/mention-monitoring/products/{product_id}/ai-overview-history
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
  GET    /api/v1/mention-monitoring/track/{tracked_mention_id}/llm-visibility-trend
  GET    /api/v1/mention-monitoring/track/{tracked_mention_id}/ai-overview-history
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
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.dependencies import current_user_id, get_current_user, get_workspace_context
from app.middleware.jwt_auth import User, WorkspaceContext
from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.tracked_mentions_service import (
    get_tracked_mentions_service,
)
from app.services.integrations.llm_mention_probe_service import (
    get_llm_mention_probe_service,
)
from app.services.integrations.mention_identity_service import SubjectFacets
from app.services.integrations.cron_billing import charge_cron
from app.services.integrations.mention_cost_logger import (
    MENTION_OP_CREDIT_COST, debit_credits, refund_credits,
)
from app.utils.paid_door import metered_door
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
    if not _is_admin(get_supabase_client().client, current_user_id(user)):
        raise HTTPException(status_code=403, detail="admin role required")


def _brand_of(product: Dict[str, Any]) -> Optional[str]:
    """The brand to monitor for a product, from one place.

    `products.manufacturer` is NOT a column — it moved into `attributes` (#26 M13-2),
    which made every `_resolve_product` read fail outright. Two call sites derived the
    brand slightly differently around it; this is the single derivation both now use,
    so a future move needs one edit rather than two that can disagree.
    """
    attributes = product.get("attributes") or {}
    metadata = product.get("metadata") or {}
    for candidate in (
        attributes.get("manufacturer"), attributes.get("brand"),
        metadata.get("brand"), metadata.get("manufacturer"),
    ):
        if candidate:
            return candidate
    return None


def _resolve_product(sb, product_id: str) -> Dict[str, Any]:
    prod = (
        sb.table("products")
        .select("id, name, attributes, metadata")
        .eq("id", product_id)
        .maybe_single()
        .execute()
    )
    row = (prod.data if prod else None) or None
    if not row:
        raise HTTPException(status_code=404, detail=f"product {product_id} not found")
    return row


def _module_enabled(sb, slug: str) -> bool:
    """Is a platform module enabled in public.modules? Mirrors the edge-side isModuleEnabled.

    FAIL-OPEN on a missing row / read error: the edge cron already gates on this flag
    (fail-closed) before it ever calls us, so this defense-in-depth chokepoint check must
    never break a legitimately-enabled run — it only exists to stop a FUTURE internal
    caller of the cron endpoints from bypassing the toggle.
    """
    try:
        res = sb.table("modules").select("enabled").eq("slug", slug).maybe_single().execute()
        row = res.data if res else None
        return True if row is None else bool(row.get("enabled"))
    except Exception:
        return True


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
    probe_tier: Optional[str] = Field(
        None, pattern="^(cheap|frontier)$",
        description="Which model tier the LLM probe uses. Defaults to cheap.",
    )
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
    probe_tier: Optional[str] = Field(None, pattern="^(cheap|frontier)$")
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
    user: Dict[str, Any] = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    sb = get_supabase_client().client
    product = _resolve_product(sb, product_id)
    svc = get_tracked_mentions_service()
    # Pull manufacturer from product metadata for brand_hint
    metadata = product.get("metadata") or {}
    brand = _brand_of(product)
    aliases = (body.aliases if body else None) or []
    if not aliases and (metadata.get("sku") or metadata.get("model")):
        aliases = [str(metadata.get("sku") or metadata.get("model"))]

    # Only the first-refresh sweep costs money; enrolling without it is free (#18 M5-3).
    run_first = (body.run_first_refresh if body else True)
    async with metered_door(
        user_id=current_user_id(user),
        workspace_id=getattr(workspace, "workspace_id", None) if workspace else None,
        cost=MENTION_OP_CREDIT_COST["track"] if run_first else 0,
        operation_type="mention_monitoring.track",
        debit=debit_credits, refund=refund_credits,
    ) as paid:
        row = await svc.find_or_create_for_product(
            product_id=product_id,
            product_name=product.get("name") or product_id,
            brand_name=brand,
            aliases=aliases,
            auto_expand_aliases=(body.auto_expand_aliases if body else False),
            user_id=current_user_id(user),
            workspace_id=getattr(workspace, "workspace_id", None) if workspace else None,
            country_codes=(body.country_codes if body else None) or [],
            run_first_refresh=run_first,
        )
        if not row:
            paid.refund("enrolment produced no tracked mention")
    # Apply alert prefs from body, if any
    if body:
        updates = body.model_dump(exclude_unset=True, exclude={"product_id", "subject_type", "subject_label", "brand_name", "run_first_refresh"})
        if updates:
            row = svc.update(row["id"], **updates) or row
    return {"success": True, "credits_debited": paid.charged, "data": row}


@router.delete("/products/{product_id}/track")
async def untrack_product(
    product_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
):
    svc = get_tracked_mentions_service()
    existing = svc.find_for_product(product_id)
    if not existing:
        return {"success": True, "message": "not tracked"}
    sb = get_supabase_client().client
    if str(existing.get("user_id")) != current_user_id(user) and not _is_admin(sb, current_user_id(user)):
        raise HTTPException(status_code=403, detail="not the owner")
    ok = svc.deactivate(existing["id"])
    return {"success": ok}


@router.get("/products/{product_id}")
async def get_product_monitoring(
    product_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
):
    svc = get_tracked_mentions_service()
    row = svc.find_for_product(product_id)
    if not row:
        return {"success": True, "data": None}
    sb = get_supabase_client().client
    if str(row.get("user_id")) != current_user_id(user) and not _is_admin(sb, current_user_id(user)):
        raise HTTPException(status_code=403, detail="not the owner")
    return {"success": True, "data": row}


@router.post("/products/{product_id}/refresh")
async def refresh_product(
    product_id: str,
    body: Optional[RefreshRequest] = None,
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    svc = get_tracked_mentions_service()
    existing = svc.find_for_product(product_id)
    if not existing:
        product = _resolve_product(sb, product_id)
        existing = await svc.find_or_create_for_product(
            product_id=product_id,
            product_name=product.get("name") or product_id,
            user_id=current_user_id(user),
            run_first_refresh=False,
        )
    if str(existing.get("user_id")) != current_user_id(user) and not _is_admin(sb, current_user_id(user)):
        raise HTTPException(status_code=403, detail="not the owner")
    force = bool(body.force) if body else False
    if force and not _is_admin(sb, current_user_id(user)):
        raise HTTPException(status_code=403, detail="force_refresh requires admin")
    async with metered_door(
        user_id=current_user_id(user),
        workspace_id=existing.get("workspace_id"),
        cost=MENTION_OP_CREDIT_COST["refresh"],
        operation_type="mention_monitoring.refresh",
        debit=debit_credits, refund=refund_credits,
    ) as paid:
        outcome = await svc.refresh(existing["id"], force=force)
        if (outcome or {}).get("skipped") or (outcome or {}).get("error"):
            paid.refund("refresh never reached a provider")
    return {"success": True, "credits_debited": paid.charged, "data": outcome}


@router.get("/products/{product_id}/feed")
async def get_product_feed(
    product_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    svc = get_tracked_mentions_service()
    existing = svc.find_for_product(product_id)
    if not existing:
        return {"success": True, "data": []}
    if str(existing.get("user_id")) != current_user_id(user) and not _is_admin(sb, current_user_id(user)):
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
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    svc = get_tracked_mentions_service()
    existing = svc.find_for_product(product_id)
    if not existing:
        return {"success": True, "data": []}
    if str(existing.get("user_id")) != current_user_id(user) and not _is_admin(sb, current_user_id(user)):
        raise HTTPException(status_code=403, detail="not the owner")
    rows = svc.history(existing["id"], days=days, limit=limit,
                       sentiment=sentiment, outlet_type=outlet_type)
    return {"success": True, "data": rows}


@router.get("/products/{product_id}/summary")
async def get_product_summary(
    product_id: str,
    days: int = Query(default=30, ge=1, le=180),
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    svc = get_tracked_mentions_service()
    existing = svc.find_for_product(product_id)
    if not existing:
        return {"success": True, "data": None}
    if str(existing.get("user_id")) != current_user_id(user) and not _is_admin(sb, current_user_id(user)):
        raise HTTPException(status_code=403, detail="not the owner")
    summary = svc.summary(existing["id"], days=days)
    return {"success": True, "data": summary}


@router.get("/products/{product_id}/llm-visibility")
async def get_product_llm_visibility(
    product_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    svc = get_tracked_mentions_service()
    existing = svc.find_for_product(product_id)
    if not existing:
        return {"success": True, "data": {"present": False}}
    if str(existing.get("user_id")) != current_user_id(user) and not _is_admin(sb, current_user_id(user)):
        raise HTTPException(status_code=403, detail="not the owner")
    snapshot = get_llm_mention_probe_service().visibility_snapshot(existing["id"])
    return {"success": True, "data": snapshot}


@router.get("/products/{product_id}/ai-overview-history")
async def get_product_ai_overview_history(
    product_id: str,
    days: int = Query(default=90, ge=1, le=365),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Google AI Overview presence over a window (issue #349 A6)."""
    sb = get_supabase_client().client
    svc = get_tracked_mentions_service()
    existing = svc.find_for_product(product_id)
    if not existing:
        return {"success": True, "data": {"present": False, "days": days, "checks": []}}
    if str(existing.get("user_id")) != current_user_id(user) and not _is_admin(sb, current_user_id(user)):
        raise HTTPException(status_code=403, detail="not the owner")
    data = get_mention_opportunity_service().ai_overview_history(existing["id"], days=days)
    return {"success": True, "data": data}


@router.get("/products/{product_id}/llm-visibility-trend")
async def get_product_llm_visibility_trend(
    product_id: str,
    days: int = Query(default=90, ge=1, le=365),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Visibility across probe RUNS, not the latest one (issue #349 A2)."""
    sb = get_supabase_client().client
    svc = get_tracked_mentions_service()
    existing = svc.find_for_product(product_id)
    if not existing:
        return {"success": True, "data": {"present": False, "days": days, "points": []}}
    if str(existing.get("user_id")) != current_user_id(user) and not _is_admin(sb, current_user_id(user)):
        raise HTTPException(status_code=403, detail="not the owner")
    trend = get_llm_mention_probe_service().visibility_trend(existing["id"], days=days)
    return {"success": True, "data": trend}


@router.post("/products/{product_id}/probe-llm")
async def probe_product_llm(
    product_id: str,
    body: Optional[ProbeLlmRequest] = None,
    user: Dict[str, Any] = Depends(get_current_user),
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
            user_id=current_user_id(user),
            run_first_refresh=False,
        )
    facets = SubjectFacets.from_dict(existing.get("subject_facets") or {
        "label": existing.get("subject_label"),
        "aliases": existing.get("aliases") or [],
        "brand": existing.get("brand_name"),
    })
    # Class #5: tag every external call inside this probe matrix with the
    # tracked_mention_id + product_id so per-subject cost dashboards work.
    from app.services.integrations.mention_cost_logger import CostAttribution as _CA
    probe_attribution = _CA(
        user_id=current_user_id(user),
        tracked_mention_id=existing["id"],
        product_id=existing.get("product_id"),
    )
    async with metered_door(
        user_id=current_user_id(user),
        workspace_id=existing.get("workspace_id"),
        # The frontier tier runs the same templates against models costing ~25x per
        # token. Charging the cheap price for it would make the expensive option the
        # cheap one — a cost control that is really a cost incentive (#349 A7).
        cost=MENTION_OP_CREDIT_COST[
            "probe_llm_frontier" if (existing.get("probe_tier") == "frontier") else "probe_llm"
        ],
        operation_type="mention_monitoring.probe_llm",
        debit=debit_credits, refund=refund_credits,
    ) as paid:
        res = await get_llm_mention_probe_service().probe(
            tracked_mention_id=existing["id"],
            facets=facets,
            models=(body.models if body else None),
            attribution=probe_attribution,
            homepage_domain=existing.get("homepage_domain"),
            tier=(existing.get("probe_tier") or "cheap"),
            **_probe_overrides(existing),
        )
        if not res:
            paid.refund("the probe matrix returned nothing")
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
    user: Dict[str, Any] = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    if not body.subject_label and not body.product_id and not body.brand_name:
        raise HTTPException(
            status_code=400,
            detail="one of subject_label, product_id, brand_name is required",
        )
    # `chk_tracked_mentions_subject` requires brand_name on a brand/keyword subject
    # that carries no product_id. Without this the insert reaches Postgres, fails with
    # a raw 23514, and the caller gets a 500 naming a constraint they cannot see —
    # which is exactly what the website AI-visibility panel hit. The route validated a
    # payload the table then rejected: an offered vocabulary wider than the enforced one.
    if (
        body.subject_type in ("brand", "keyword")
        and not body.product_id
        and not body.brand_name
        and not body.subject_label
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"a {body.subject_type} subject needs brand_name (or a subject_label to "
                "derive it from) — the database requires it for any subject without a product"
            ),
        )
    sb = get_supabase_client().client
    label = body.subject_label
    product_id = body.product_id
    brand = body.brand_name
    if product_id and not label:
        product = _resolve_product(sb, product_id)
        label = product.get("name") or product_id
        brand = brand or _brand_of(product)
    # A brand/keyword subject with no product must carry brand_name. Deriving it from
    # the label is what the caller meant in every case: for these subject types the
    # label IS the brand or the phrase being tracked.
    if not product_id and body.subject_type in ("brand", "keyword") and not brand:
        brand = label

    svc = get_tracked_mentions_service()
    row = await svc.create(
        api_key_id=None,
        user_id=current_user_id(user),
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
        probe_tier=(body.probe_tier or "cheap"),
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
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    row = _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=current_user_id(user))
    return {"success": True, "data": row}


@router.put("/track/{tracked_mention_id}")
async def update_tracked_mention(
    tracked_mention_id: str,
    body: UpdateRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=current_user_id(user))
    updates = body.model_dump(exclude_unset=True)
    row = get_tracked_mentions_service().update(tracked_mention_id, **updates)
    return {"success": True, "data": row}


@router.delete("/track/{tracked_mention_id}")
async def delete_tracked_mention(
    tracked_mention_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=current_user_id(user))
    ok = get_tracked_mentions_service().deactivate(tracked_mention_id)
    return {"success": ok}


@router.post("/track/{tracked_mention_id}/refresh")
async def refresh_tracked_mention(
    tracked_mention_id: str,
    body: Optional[RefreshRequest] = None,
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _owner_row = _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=current_user_id(user))
    force = bool(body.force) if body else False
    if force and not _is_admin(sb, current_user_id(user)):
        raise HTTPException(status_code=403, detail="force_refresh requires admin")
    async with metered_door(
        user_id=current_user_id(user),
        workspace_id=(_owner_row or {}).get("workspace_id"),
        cost=MENTION_OP_CREDIT_COST["refresh"],
        operation_type="mention_monitoring.refresh",
        debit=debit_credits, refund=refund_credits,
    ) as paid:
        outcome = await get_tracked_mentions_service().refresh(tracked_mention_id, force=force)
        if (outcome or {}).get("skipped") or (outcome or {}).get("error"):
            paid.refund("refresh never reached a provider")
    return {"success": True, "credits_debited": paid.charged, "data": outcome}


@router.get("/track/{tracked_mention_id}/feed")
async def get_tracked_feed(
    tracked_mention_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=current_user_id(user))
    rows = get_tracked_mentions_service().latest_results(tracked_mention_id, limit=limit)
    return {"success": True, "data": rows}


@router.get("/track/{tracked_mention_id}/history")
async def get_tracked_history(
    tracked_mention_id: str,
    days: int = Query(default=30, ge=1, le=180),
    sentiment: Optional[str] = None,
    outlet_type: Optional[str] = None,
    limit: int = Query(default=200, ge=1, le=2000),
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=current_user_id(user))
    rows = get_tracked_mentions_service().history(
        tracked_mention_id, days=days, sentiment=sentiment, outlet_type=outlet_type, limit=limit,
    )
    return {"success": True, "data": rows}


@router.get("/track/{tracked_mention_id}/summary")
async def get_tracked_summary(
    tracked_mention_id: str,
    days: int = Query(default=30, ge=1, le=180),
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=current_user_id(user))
    summary = get_tracked_mentions_service().summary(tracked_mention_id, days=days)
    return {"success": True, "data": summary}


@router.get("/track/{tracked_mention_id}/llm-visibility")
async def get_tracked_llm_visibility(
    tracked_mention_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=current_user_id(user))
    snapshot = get_llm_mention_probe_service().visibility_snapshot(tracked_mention_id)
    return {"success": True, "data": snapshot}


@router.get("/track/{tracked_mention_id}/ai-overview-history")
async def get_tracked_ai_overview_history(
    tracked_mention_id: str,
    days: int = Query(default=90, ge=1, le=365),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Google AI Overview presence over a window (issue #349 A6).

    A read of already-recorded observations — no SERP call, no credits. The checks are
    written by the opportunity pass, so history only accumulates for subjects somebody
    actually refreshes.
    """
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=current_user_id(user))
    data = get_mention_opportunity_service().ai_overview_history(tracked_mention_id, days=days)
    return {"success": True, "data": data}


@router.get("/track/{tracked_mention_id}/llm-visibility-trend")
async def get_tracked_llm_visibility_trend(
    tracked_mention_id: str,
    days: int = Query(default=90, ge=1, le=365),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Visibility across probe RUNS, not the latest one (issue #349 A2)."""
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=current_user_id(user))
    trend = get_llm_mention_probe_service().visibility_trend(tracked_mention_id, days=days)
    return {"success": True, "data": trend}


@router.post("/track/{tracked_mention_id}/probe-llm")
async def probe_tracked_llm(
    tracked_mention_id: str,
    body: Optional[ProbeLlmRequest] = None,
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _require_admin(user)
    row = _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=current_user_id(user))
    facets = SubjectFacets.from_dict(row.get("subject_facets") or {
        "label": row.get("subject_label"),
        "aliases": row.get("aliases") or [],
        "brand": row.get("brand_name"),
    })
    from app.services.integrations.mention_cost_logger import CostAttribution as _CA
    probe_attribution = _CA(
        user_id=current_user_id(user),
        tracked_mention_id=tracked_mention_id,
        product_id=row.get("product_id"),
    )
    async with metered_door(
        user_id=current_user_id(user),
        workspace_id=row.get("workspace_id"),
        cost=MENTION_OP_CREDIT_COST[
            "probe_llm_frontier" if (row.get("probe_tier") == "frontier") else "probe_llm"
        ],
        operation_type="mention_monitoring.probe_llm",
        debit=debit_credits, refund=refund_credits,
    ) as paid:
        res = await get_llm_mention_probe_service().probe(
            tracked_mention_id=tracked_mention_id,
            facets=facets,
            models=(body.models if body else None),
            attribution=probe_attribution,
            homepage_domain=row.get("homepage_domain"),
            tier=(row.get("probe_tier") or "cheap"),
            **_probe_overrides(row),
        )
        if not res:
            paid.refund("the probe matrix returned nothing")
    return {"success": True, "credits_debited": paid.charged, "data": res}


@router.post("/track/{tracked_mention_id}/exclude")
async def exclude_url(
    tracked_mention_id: str,
    body: ExcludeRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=current_user_id(user))
    if not body.url and not body.domain:
        raise HTTPException(status_code=400, detail="url or domain required")
    out = get_tracked_mentions_service().add_exclusion(
        tracked_mention_id, url=body.url, domain=body.domain,
        reason=body.reason, user_id=current_user_id(user),
    )
    return {"success": True, "data": out}


@router.post("/track/{tracked_mention_id}/include")
async def include_url(
    tracked_mention_id: str,
    body: ExcludeRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=current_user_id(user))
    removed = get_tracked_mentions_service().remove_exclusion(
        tracked_mention_id, url=body.url, domain=body.domain,
    )
    return {"success": True, "removed_count": removed}


@router.get("/track/{tracked_mention_id}/exclusions")
async def list_exclusions(
    tracked_mention_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=current_user_id(user))
    rows = get_tracked_mentions_service().list_exclusions(tracked_mention_id)
    return {"success": True, "data": rows}


@router.post("/track/{tracked_mention_id}/promote")
async def promote_url(
    tracked_mention_id: str,
    body: PromoteRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    sb = get_supabase_client().client
    _require_admin(user)
    _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=current_user_id(user))
    out = get_tracked_mentions_service().add_promoted_url(
        tracked_mention_id,
        url=body.url, override_relevance=body.override_relevance,
        reason=body.reason, user_id=current_user_id(user),
    )
    return {"success": True, "data": out}


@router.get("/track/{tracked_mention_id}/share-of-voice")
async def share_of_voice(
    tracked_mention_id: str,
    days: int = Query(default=30, ge=1, le=180),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Share of voice: the subject against its competitors, bucketed per probe run.

    Two defects fixed here (issue #349 A4). It counted competitors ONLY, so the one
    brand whose page this is had no share of its own voice; and `days` was declared,
    validated and then never applied to the query, which filtered on the subject and
    took the newest 500 rows whatever window you asked for. Both produce a number
    that looks like an answer, which is why neither surfaced on its own.
    """
    sb = get_supabase_client().client
    row = _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=current_user_id(user))
    data = get_llm_mention_probe_service().share_of_voice_series(
        tracked_mention_id,
        subject_label=(row.get("subject_label") or row.get("brand_name") or ""),
        days=days,
    )
    # `competitor_mentions` at the top level is what the pre-#349 clients read. Kept
    # pointing at the same totals rather than dropped, so an old caller reading it
    # gets the windowed answer instead of a 500.
    data["competitor_mentions"] = (data.get("totals") or {}).get("competitor_mentions", [])
    return {"success": True, "data": data}


@router.post("/products/{product_id}/opportunities")
async def get_product_opportunities(
    product_id: str,
    body: Optional[OpportunitiesRequest] = None,
    user: Dict[str, Any] = Depends(get_current_user),
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
    if str(existing.get("user_id")) != current_user_id(user) and not _is_admin(sb, current_user_id(user)):
        raise HTTPException(status_code=403, detail="not the owner")
    # A disabled module must not run paid discovery. The refresh routes below already gate
    # on this; these three did not, so `mention-monitoring` kept billing DataForSEO Labs
    # while switched off — 3 calls on 2026-08-02 alone, months after the toggle went false.
    # (audit #272: "disabled != stopped")
    if not _module_enabled(sb, "mention-monitoring"):
        return {"success": True, "data": {"opportunities": [], "errors": {}},
                "skipped": "module_disabled"}
    body = body or OpportunitiesRequest()
    async with metered_door(
        user_id=current_user_id(user),
        workspace_id=existing.get("workspace_id"),
        cost=MENTION_OP_CREDIT_COST[
            "opportunities_with_llm" if body.use_llm_summary else "opportunities"
        ],
        operation_type="mention_monitoring.opportunities",
        debit=debit_credits, refund=refund_credits,
    ) as paid:
        out = await get_mention_opportunity_service().generate(
            tracked_mention_id=existing["id"],
            types=body.types,
            days=body.days,
            limit_per_type=body.limit_per_type,
            use_llm_summary=body.use_llm_summary,
            # Re-stated to the service so the tenancy check exists at the layer that
            # spends the money too (#21 M8-4), not only at the route.
            caller_user_id=current_user_id(user),
            caller_is_admin=_is_admin(sb, current_user_id(user)),
        )
        if not (out or {}).get("opportunities"):
            paid.refund("generated no opportunities")
    return {"success": True, "credits_debited": paid.charged, "data": out}


@router.post("/track/{tracked_mention_id}/opportunities")
async def get_tracked_opportunities(
    tracked_mention_id: str,
    body: Optional[OpportunitiesRequest] = None,
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Generate content + outreach opportunities for any tracked subject."""
    sb = get_supabase_client().client
    _owner_row = _check_owner_or_admin(sb, tracked_mention_id=tracked_mention_id, user_id=current_user_id(user))
    # A disabled module must not run paid discovery. The refresh routes below already gate
    # on this; these three did not, so `mention-monitoring` kept billing DataForSEO Labs
    # while switched off — 3 calls on 2026-08-02 alone, months after the toggle went false.
    # (audit #272: "disabled != stopped")
    if not _module_enabled(sb, "mention-monitoring"):
        return {"success": True, "data": {"opportunities": [], "errors": {}},
                "skipped": "module_disabled"}
    body = body or OpportunitiesRequest()
    async with metered_door(
        user_id=current_user_id(user),
        workspace_id=(_owner_row or {}).get("workspace_id"),
        cost=MENTION_OP_CREDIT_COST[
            "opportunities_with_llm" if body.use_llm_summary else "opportunities"
        ],
        operation_type="mention_monitoring.opportunities",
        debit=debit_credits, refund=refund_credits,
    ) as paid:
        out = await get_mention_opportunity_service().generate(
            tracked_mention_id=tracked_mention_id,
            types=body.types,
            days=body.days,
            limit_per_type=body.limit_per_type,
            use_llm_summary=body.use_llm_summary,
            caller_user_id=current_user_id(user),
            caller_is_admin=_is_admin(sb, current_user_id(user)),
        )
        if not (out or {}).get("opportunities"):
            paid.refund("generated no opportunities")
    return {"success": True, "credits_debited": paid.charged, "data": out}


# ============================================================================
# Cross-flow
# ============================================================================

@router.post("/classifier-correction")
async def classifier_correction(
    body: ClassifierCorrectionRequest,
    user: Dict[str, Any] = Depends(get_current_user),
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
    _check_owner_or_admin(sb, tracked_mention_id=row["tracked_mention_id"], user_id=current_user_id(user))
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
        "created_by": current_user_id(user),
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
    # Honor the module toggle (defense-in-depth; the edge cron gates too). A disabled
    # module must not run paid discovery.
    if not _module_enabled(sb, "mention-monitoring"):
        return {"success": True, "skipped": "module_disabled", "due_count": 0,
                "processed": 0, "succeeded": 0, "failed": 0, "results": []}
    try:
        r = sb.rpc("get_internal_tracked_mentions_due", {"p_limit": limit}).execute()
        due = r.data or []
    except Exception as e:
        return {"success": False, "error": str(e)}
    svc = get_tracked_mentions_service()
    processed = succeeded = failed = 0
    skipped_unpaid = 0
    results: List[Dict[str, Any]] = []
    # Resolve owners for metering (the due RPC doesn't return them).
    ids = [row["id"] for row in due if row.get("id")]
    owner_by_id: Dict[str, Dict[str, Any]] = {}
    if ids:
        try:
            ores = sb.table("tracked_mentions").select("id, workspace_id, user_id").in_("id", ids).execute()
            owner_by_id = {o["id"]: o for o in (ores.data or [])}
        except Exception as e:
            logger.warning(f"mention-cron: owner lookup failed (metering fails open): {e}")
    for row in due:
        owner = owner_by_id.get(row["id"], {})
        # Meter the owner BEFORE the paid discovery refresh. Registered cron_key
        # 'mention-monitoring' (3 cr); fails open, False only when out of credits.
        if not charge_cron(
            sb, "mention-monitoring",
            workspace_id=owner.get("workspace_id"), user_id=owner.get("user_id"),
            description="Tracked-subject mention refresh",
            subject={"tracked_mention_id": str(row["id"])},
        ):
            skipped_unpaid += 1
            results.append({"id": row["id"], "status": "skipped_insufficient_credits"})
            continue
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
            "skipped_insufficient_credits": skipped_unpaid,
            "results": results}


def _probe_overrides(row: dict) -> dict:
    """The workspace's own probe questions, off `source_config`.

    One reader for all three call sites (product, subject, cron). Three separate
    `.get('source_config', {}).get(...)` chains is exactly how one path keeps
    asking the stock questions while the UI says the custom ones are saved.
    """
    cfg = (row or {}).get("source_config") or {}
    if not isinstance(cfg, dict):
        return {"custom_probes": None, "include_default_probes": True}
    return {
        "custom_probes": cfg.get("custom_probes"),
        # Explicit opt-out only. Absent means keep the stock questions, so an
        # existing subject cannot silently lose its baseline measurement.
        "include_default_probes": cfg.get("include_default_probes", True) is not False,
    }


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
    # Honor the module toggle (defense-in-depth; the edge cron gates too). A disabled
    # module must not run paid LLM probes.
    if not _module_enabled(sb, "mention-monitoring"):
        return {"success": True, "skipped": "module_disabled", "due_count": 0,
                "processed": 0, "succeeded": 0, "failed": 0}
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
    skipped_unpaid = 0
    from app.services.integrations.mention_cost_logger import CostAttribution as _CA
    for row in due:
        tm_id = row["id"]
        try:
            full = svc.get(tm_id) or {}
            # Meter the owner BEFORE the paid LLM probe. Registered cron_key
            # 'llm-mention-probe' (3 cr); fails open, False only when out of credits.
            _tier = full.get("probe_tier") or "cheap"
            if not charge_cron(
                sb, "llm-mention-probe",
                workspace_id=full.get("workspace_id"), user_id=full.get("user_id"),
                description=f"LLM-visibility probe run ({_tier} tier)",
            ):
                skipped_unpaid += 1
                logger.info(f"cron-probe-llm: skipped {tm_id} (insufficient credits)")
                continue
            facets = SubjectFacets.from_dict(full.get("subject_facets") or {
                "label": full.get("subject_label"),
                "aliases": full.get("aliases") or [],
                "brand": full.get("brand_name"),
            })
            cron_attr = _CA(
                user_id=full.get("user_id"),
                workspace_id=full.get("workspace_id"),
                tracked_mention_id=tm_id,
                product_id=full.get("product_id"),
                api_key_id=full.get("api_key_id"),
            )
            await probe.probe(
                tracked_mention_id=tm_id, facets=facets, attribution=cron_attr,
                homepage_domain=full.get("homepage_domain"),
                tier=(full.get("probe_tier") or "cheap"),
                **_probe_overrides(full),
            )
            succeeded += 1
        except Exception as e:
            failed += 1
            logger.warning(f"cron-probe-llm subject {tm_id} failed: {e}")
        processed += 1
    return {"success": True, "due_count": len(due), "processed": processed,
            "succeeded": succeeded, "failed": failed,
            "skipped_insufficient_credits": skipped_unpaid}


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
    workspace_id: Optional[str] = Field(
        None,
        description=(
            "Whose research this is, for cost attribution. There is no tracked-subject row on this "
            "path for the service to read an owner off, so without it the DataForSEO Labs spend "
            "this triggers lands in ai_usage_logs owned by nobody. Trusted because the endpoint is "
            "x-cron-secret gated and internal-only — the edge function sends the session's "
            "workspace, not anything a user supplied."
        ),
    )


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
    sb = get_supabase_client().client
    # A disabled module must not run paid discovery. The refresh routes below already gate
    # on this; these three did not, so `mention-monitoring` kept billing DataForSEO Labs
    # while switched off — 3 calls on 2026-08-02 alone, months after the toggle went false.
    # (audit #272: "disabled != stopped")
    if not _module_enabled(sb, "mention-monitoring"):
        return {"success": True, "data": {"opportunities": [], "errors": {}},
                "skipped": "module_disabled"}
    subject_override = {
        "subject_label": body.subject_label,
        "brand_name": body.brand_name,
        "aliases": body.aliases or [],
        "language_codes": body.language_codes or ["en"],
        "country_codes": body.country_codes or ["US"],
        "homepage_domain": body.homepage_domain,
        # Picked up by CostAttribution, which derives its owner from this dict when the caller
        # did not pre-build one. Absent, every external call in this run logs with no tenant.
        "workspace_id": body.workspace_id,
    }
    # Cron-authenticated, so it bills the subject's workspace through the shared
    # cron meter rather than a user wallet — but it MUST still meter (#18 M5-3).
    if not charge_cron(
        sb, "mention_opportunities_stateless",
        workspace_id=body.workspace_id,
        units=1,
        description="stateless opportunity generation",
        subject={"subject_label": body.subject_label},
    ):
        return {"success": True, "data": {"opportunities": [], "errors": {}},
                "skipped": "insufficient_credits"}
    out = await get_mention_opportunity_service().generate(
        subject_override=subject_override,
        types=body.types,
        limit_per_type=body.limit_per_type,
        use_llm_summary=body.use_llm_summary,
    )
    return {"success": True, "data": out}
