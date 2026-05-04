"""
Public Mention Tracking API — /api/v1/mentions/track/*

External projects authenticate with an `api_keys` Bearer token, register
tracked mention subjects (brand / keyword / product name), and control
the refresh cadence.

Mirror of `tracked_queries_routes.py` for mention monitoring. Uses the
same `authenticate_api_key` dependency so api_keys with the Bearer prefix
`kai_*` are accepted exactly the same way as the price-tracking flow.

Routing:
  api_key_id NOT NULL  → external API consumer (this file)
  api_key_id IS NULL   → internal product/brand flow (mention_monitoring_routes.py)

Deleting the api_key CASCADEs out every tracked subject + mention history
tied to it (enforced at the DB level via ON DELETE CASCADE).
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.price_lookup_routes import ApiKeyContext, authenticate_api_key
from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.tracked_mentions_service import (
    get_tracked_mentions_service,
)
from app.services.integrations.llm_mention_probe_service import (
    get_llm_mention_probe_service,
)
from app.services.integrations.mention_identity_service import SubjectFacets
from app.services.integrations.mention_opportunity_service import (
    get_mention_opportunity_service,
)
from app.services.integrations.mention_cost_logger import (
    MENTION_OP_CREDIT_COST, debit_credits, refund_credits,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/mentions/track",
    tags=["Mention Tracking (Public API)"],
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

class CreateMentionTrackRequest(BaseModel):
    """Body for `POST /api/v1/mentions/track` — register a new tracked subject.
    First refresh runs synchronously; the response includes the initial results."""

    subject_type: str = Field(
        "brand", pattern="^(product|brand|keyword)$",
        description="What kind of subject this is. Currently informational; doesn't gate behavior.",
        examples=["brand"],
    )
    subject_label: str = Field(
        ..., min_length=1, max_length=200,
        description="The literal string to track. Discovery searches for this exact phrase.",
        examples=["Flobali"],
    )
    brand_name: Optional[str] = Field(
        None, description="Brand string when known. Used as a hint and stored on the row.",
    )
    aliases: Optional[List[str]] = Field(
        None,
        description=(
            "Alternate strings to also search for (e.g. brand abbreviations, model SKUs, "
            "Greek/Latin spellings). Each alias is matched as a literal phrase. "
            "Recommended for multi-word labels where articles split the words apart."
        ),
        examples=[["Flobali Tiles", "Flobali Hellas"]],
    )
    auto_expand_aliases: bool = Field(
        False,
        description=(
            "Default false. When true, an LLM (Claude Haiku) expands the subject_label "
            "into per-word aliases on first refresh — broader recall on multi-word "
            "subjects but higher cost and a dependency on Anthropic. When false, "
            "discovery uses only the subject_label and any aliases supplied above."
        ),
    )
    sources_enabled: Optional[Dict[str, bool]] = Field(
        None,
        description=(
            "Per-source toggles. Defaults: news=true, blogs=true, rss=true, llm=true, "
            "youtube=false. Only the keys you set are overridden."
        ),
        examples=[{"news": True, "blogs": True, "youtube": False, "rss": True, "llm": True}],
    )
    source_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Per-source config blob. Currently supports `rss_feeds: string[]` for user-curated RSS URLs.",
    )
    language_codes: Optional[List[str]] = Field(
        None, description="ISO 639-1 codes the subject is likely discussed in. Default ['en'].",
        examples=[["en", "el"]],
    )
    country_codes: Optional[List[str]] = Field(
        None,
        description=(
            "ISO 3166-1 alpha-2 codes. When set, results are STRICTLY filtered to TLDs "
            "matching these countries (plus the curated per-country outlet allowlist). "
            "Drop the field for global discovery."
        ),
        examples=[["GR", "DE"]],
    )
    refresh_interval_hours: int = Field(
        24, ge=1, le=720,
        description="Volatility-aware base cadence. Stable subjects auto-stretch up to weekly.",
    )
    recency_days: int = Field(
        30, ge=1, le=730,
        description=(
            "Discovery window in days. Articles older than this are dropped. "
            "Default 30. Bump higher (90/180/365) for niche brands whose coverage "
            "cadence is slower than monthly — e.g. regional outlets that publish "
            "quarterly, trade press with monthly issues."
        ),
        examples=[30, 90, 180],
    )
    homepage_domain: Optional[str] = Field(
        None,
        description=(
            "Optional. The brand's primary domain (no scheme, no path) — e.g. "
            "'flobali.gr'. Powers the `domain_snapshot` opportunity type, which "
            "returns Domain Rank, organic-traffic estimation, referring-domains "
            "count, and total backlinks. When unset, `domain_snapshot` returns "
            "a configuration card explaining how to enable."
        ),
        examples=["flobali.gr", "yourbrand.com"],
    )
    alert_channels: Optional[List[str]] = Field(
        None,
        description="Channels to fire alerts on. Subset of: 'bell', 'email', 'webhook'. Default ['bell'].",
        examples=[["webhook"]],
    )
    alert_on_spike: bool = Field(False, description="Fire when today's count ≥ 2× trailing 7d daily-average baseline.")
    alert_on_negative_sentiment: bool = Field(False, description="Fire on a new negative-sentiment mention from a high-DA outlet (DA ≥ 30).")
    alert_on_new_outlet: bool = Field(False, description="Fire on the first-ever mention from a new domain.")
    alert_on_llm_visibility_change: bool = Field(False, description="Fire when average LLM rank shifts ≥ 2 positions week-over-week.")
    alert_webhook_url: Optional[str] = Field(
        None,
        description="HTTPS URL to receive POSTed alert payloads. Required when 'webhook' is in alert_channels.",
        examples=["https://your.api/webhooks/material-kai-mentions"],
    )


class UpdateMentionTrackRequest(BaseModel):
    subject_label: Optional[str] = None
    aliases: Optional[List[str]] = None
    auto_expand_aliases: Optional[bool] = None
    sources_enabled: Optional[Dict[str, bool]] = None
    source_config: Optional[Dict[str, Any]] = None
    language_codes: Optional[List[str]] = None
    country_codes: Optional[List[str]] = None
    refresh_interval_hours: Optional[int] = Field(None, ge=1, le=720)
    recency_days: Optional[int] = Field(None, ge=1, le=730)
    homepage_domain: Optional[str] = None
    alert_channels: Optional[List[str]] = None
    alert_on_spike: Optional[bool] = None
    alert_on_negative_sentiment: Optional[bool] = None
    alert_on_new_outlet: Optional[bool] = None
    alert_on_llm_visibility_change: Optional[bool] = None
    alert_webhook_url: Optional[str] = None
    is_active: Optional[bool] = None


class RefreshRequest(BaseModel):
    """Body for `POST /api/v1/mentions/track/{id}/refresh`. Cost: 5 credits."""
    force: bool = Field(
        False,
        description="When true, bypass the volatility cadence even if next_check_at is in the future.",
    )


class ExcludeRequest(BaseModel):
    """Body for `POST /track/{id}/exclude` and `/include`. Either url or domain is required."""
    url: Optional[str] = Field(None, description="Specific URL to exclude. Mutually exclusive with `domain`.")
    domain: Optional[str] = Field(None, description="Whole domain to exclude (e.g. 'spammer.com').")
    reason: Optional[str] = Field(None, description="Free-text reason saved on the exclusion row.")


class OpportunitiesRequest(BaseModel):
    """Body for `POST /track/{id}/opportunities`. Cost: 2 credits (or 5 with `use_llm_summary`)."""
    types: Optional[List[str]] = Field(
        None,
        description=(
            "Subset of opportunity types to return. Default = all six: "
            "['trending_topic','outlet_pitch','keyword_opportunity','pao_question',"
            "'author_relationship','sentiment_response']."
        ),
    )
    days: int = Field(30, ge=1, le=180, description="Discovery window for source signals.")
    limit_per_type: int = Field(5, ge=1, le=20, description="Max opportunities returned per type.")
    use_llm_summary: bool = Field(
        False,
        description=(
            "When true, Haiku rewrites each opportunity's rationale and suggested_action in "
            "tighter, more actionable language (+3 credits over the default cost)."
        ),
    )


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _check_owner(sb, *, tracking_id: str, api_key_id: str) -> Dict[str, Any]:
    r = (
        sb.table("tracked_mentions")
        .select("*")
        .eq("id", tracking_id)
        .maybe_single()
        .execute()
    )
    row = (r.data if r else None) or None
    if not row:
        raise HTTPException(status_code=404, detail="tracking_id not found")
    if row.get("api_key_id") != api_key_id:
        raise HTTPException(status_code=403, detail="api key does not own this tracking_id")
    return row


# ────────────────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_tracking(
    body: CreateMentionTrackRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Create a new tracked mention subject. First refresh runs synchronously
    so the response includes initial results."""
    svc = get_tracked_mentions_service()
    row = await svc.create(
        api_key_id=ctx.api_key_id,
        user_id=ctx.user_id,
        workspace_id=ctx.workspace_id,
        subject_type=body.subject_type,
        subject_label=body.subject_label,
        product_id=None,
        brand_name=body.brand_name or (body.subject_label if body.subject_type == "brand" else None),
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
        run_first_refresh=True,
    )
    return {"success": True, "data": row}


@router.get("")
async def list_tracking(
    ctx: ApiKeyContext = Depends(authenticate_api_key),
    include_inactive: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=500),
):
    """List all tracked subjects owned by this API key."""
    sb = get_supabase_client().client
    q = (
        sb.table("tracked_mentions")
        .select("*")
        .eq("api_key_id", ctx.api_key_id)
        .order("created_at", desc=True)
        .limit(limit)
    )
    if not include_inactive:
        q = q.eq("is_active", True)
    r = q.execute()
    return {"success": True, "data": r.data or [], "count": len(r.data or [])}


@router.get("/{tracking_id}")
async def get_tracking(
    tracking_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    sb = get_supabase_client().client
    row = _check_owner(sb, tracking_id=tracking_id, api_key_id=ctx.api_key_id)
    return {"success": True, "data": row}


@router.put("/{tracking_id}")
async def update_tracking(
    tracking_id: str,
    body: UpdateMentionTrackRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    sb = get_supabase_client().client
    _check_owner(sb, tracking_id=tracking_id, api_key_id=ctx.api_key_id)
    updates = body.model_dump(exclude_unset=True)
    row = get_tracked_mentions_service().update(tracking_id, **updates)
    return {"success": True, "data": row}


@router.delete("/{tracking_id}")
async def delete_tracking(
    tracking_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Soft delete (deactivates the row but preserves history)."""
    sb = get_supabase_client().client
    _check_owner(sb, tracking_id=tracking_id, api_key_id=ctx.api_key_id)
    ok = get_tracked_mentions_service().deactivate(tracking_id)
    return {"success": ok}


@router.post("/{tracking_id}/refresh")
async def refresh_tracking(
    tracking_id: str,
    body: Optional[RefreshRequest] = None,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Re-run discovery + classification. Bills against the API key's quota."""
    sb = get_supabase_client().client
    _check_owner(sb, tracking_id=tracking_id, api_key_id=ctx.api_key_id)
    force = bool(body.force) if body else False

    # Layer B: debit credits before doing the work; refund on hard failure or
    # when the refresh short-circuits (throttled / inactive). Successful refreshes
    # keep the credits even if 0 hits land — the external API calls still ran.
    cost = MENTION_OP_CREDIT_COST["refresh"]
    if not debit_credits(user_id=ctx.user_id or "", amount=cost,
                         operation_type="mention_monitoring.refresh"):
        raise HTTPException(status_code=402, detail="insufficient credits")
    try:
        outcome = await get_tracked_mentions_service().refresh(tracking_id, force=force)
    except Exception:
        refund_credits(user_id=ctx.user_id or "", amount=cost,
                       operation_type="mention_monitoring.refresh")
        raise

    if outcome.get("status") in ("throttled", "inactive", "not_found", "error"):
        refund_credits(user_id=ctx.user_id or "", amount=cost,
                       operation_type="mention_monitoring.refresh")
    else:
        outcome["partner_credits_debited"] = cost
    return {"success": True, "data": outcome}


@router.get("/{tracking_id}/feed")
async def get_feed(
    tracking_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Latest refresh run as a row list."""
    sb = get_supabase_client().client
    _check_owner(sb, tracking_id=tracking_id, api_key_id=ctx.api_key_id)
    rows = get_tracked_mentions_service().latest_results(tracking_id, limit=limit)
    return {"success": True, "data": rows, "count": len(rows)}


@router.get("/{tracking_id}/history")
async def get_history(
    tracking_id: str,
    days: int = Query(default=30, ge=1, le=180),
    sentiment: Optional[str] = Query(default=None, pattern="^(positive|neutral|negative)$"),
    outlet_type: Optional[str] = None,
    limit: int = Query(default=200, ge=1, le=2000),
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Paginated mention history."""
    sb = get_supabase_client().client
    _check_owner(sb, tracking_id=tracking_id, api_key_id=ctx.api_key_id)
    rows = get_tracked_mentions_service().history(
        tracking_id, days=days, sentiment=sentiment, outlet_type=outlet_type, limit=limit,
    )
    return {"success": True, "data": rows, "count": len(rows)}


@router.get("/{tracking_id}/summary")
async def get_summary(
    tracking_id: str,
    days: int = Query(default=30, ge=1, le=180),
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Aggregate snapshot: total count, sentiment breakdown, top outlets."""
    sb = get_supabase_client().client
    _check_owner(sb, tracking_id=tracking_id, api_key_id=ctx.api_key_id)
    summary = get_tracked_mentions_service().summary(tracking_id, days=days)
    return {"success": True, "data": summary}


@router.get("/{tracking_id}/llm-visibility")
async def get_llm_visibility(
    tracking_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Most recent LLM probe snapshot."""
    sb = get_supabase_client().client
    _check_owner(sb, tracking_id=tracking_id, api_key_id=ctx.api_key_id)
    snapshot = get_llm_mention_probe_service().visibility_snapshot(tracking_id)
    return {"success": True, "data": snapshot}


@router.post("/{tracking_id}/probe-llm")
async def probe_llm(
    tracking_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Fire a fresh LLM probe matrix. Bills against the API key's quota."""
    sb = get_supabase_client().client
    row = _check_owner(sb, tracking_id=tracking_id, api_key_id=ctx.api_key_id)
    facets = SubjectFacets.from_dict(row.get("subject_facets") or {
        "label": row.get("subject_label"),
        "aliases": row.get("aliases") or [],
        "brand": row.get("brand_name"),
    })
    cost = MENTION_OP_CREDIT_COST["probe_llm"]
    if not debit_credits(user_id=ctx.user_id or "", amount=cost,
                         operation_type="mention_monitoring.probe_llm"):
        raise HTTPException(status_code=402, detail="insufficient credits")
    try:
        result = await get_llm_mention_probe_service().probe(
            tracked_mention_id=tracking_id, facets=facets,
        )
    except Exception:
        refund_credits(user_id=ctx.user_id or "", amount=cost,
                       operation_type="mention_monitoring.probe_llm")
        raise
    if result.get("status") != "completed":
        refund_credits(user_id=ctx.user_id or "", amount=cost,
                       operation_type="mention_monitoring.probe_llm")
    else:
        result["partner_credits_debited"] = cost
    return {"success": True, "data": result}


@router.post("/{tracking_id}/exclude")
async def exclude_url(
    tracking_id: str,
    body: ExcludeRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    sb = get_supabase_client().client
    _check_owner(sb, tracking_id=tracking_id, api_key_id=ctx.api_key_id)
    if not body.url and not body.domain:
        raise HTTPException(status_code=400, detail="url or domain required")
    out = get_tracked_mentions_service().add_exclusion(
        tracking_id, url=body.url, domain=body.domain,
        reason=body.reason, user_id=ctx.user_id,
    )
    return {"success": True, "data": out}


@router.post("/{tracking_id}/include")
async def include_url(
    tracking_id: str,
    body: ExcludeRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    sb = get_supabase_client().client
    _check_owner(sb, tracking_id=tracking_id, api_key_id=ctx.api_key_id)
    removed = get_tracked_mentions_service().remove_exclusion(
        tracking_id, url=body.url, domain=body.domain,
    )
    return {"success": True, "removed_count": removed}


@router.get("/{tracking_id}/exclusions")
async def list_exclusions(
    tracking_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    sb = get_supabase_client().client
    _check_owner(sb, tracking_id=tracking_id, api_key_id=ctx.api_key_id)
    rows = get_tracked_mentions_service().list_exclusions(tracking_id)
    return {"success": True, "data": rows}


@router.post("/{tracking_id}/opportunities")
async def get_opportunities(
    tracking_id: str,
    body: Optional[OpportunitiesRequest] = None,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Generate content + outreach opportunities from existing mention data.

    Returns a ranked list of typed opportunities the user can act on:
      - trending_topic        (write a post on a recurring theme)
      - outlet_pitch          (warm outlets to pitch)
      - keyword_opportunity   (high-volume related keywords)
      - pao_question          (questions readers are asking)
      - author_relationship   (warm author contacts)
      - sentiment_response    (negative mentions to address)

    Read-only — does not mutate state or trigger a refresh.
    """
    sb = get_supabase_client().client
    _check_owner(sb, tracking_id=tracking_id, api_key_id=ctx.api_key_id)
    body = body or OpportunitiesRequest()
    cost_key = "opportunities_with_llm" if body.use_llm_summary else "opportunities"
    cost = MENTION_OP_CREDIT_COST[cost_key]
    if not debit_credits(user_id=ctx.user_id or "", amount=cost,
                         operation_type=f"mention_monitoring.{cost_key}"):
        raise HTTPException(status_code=402, detail="insufficient credits")
    try:
        out = await get_mention_opportunity_service().generate(
            tracked_mention_id=tracking_id,
            types=body.types,
            days=body.days,
            limit_per_type=body.limit_per_type,
            use_llm_summary=body.use_llm_summary,
        )
    except Exception:
        refund_credits(user_id=ctx.user_id or "", amount=cost,
                       operation_type=f"mention_monitoring.{cost_key}")
        raise
    out["partner_credits_debited"] = cost
    return {"success": True, "data": out}
