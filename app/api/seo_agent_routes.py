"""
SEO Agent Routes — internal-only DataForSEO dispatch surface.

Used exclusively by the KAI agent's SEO toolkit (supabase/functions/_shared/
tools/seo-agent-tools.ts and friends). Auth: `x-cron-secret` header — same
secret used by mention-monitoring + price-monitoring crons.

Endpoint design:
  - One generic dispatcher: `POST /api/v1/seo-agent/dataforseo/{kind}`
    where {kind} maps to a method on `DataForSEOUnifiedClient`. The body is
    forwarded as **kwargs. This lets us add new endpoints without adding a
    new HTTP route every time.
  - Plus a handful of composed routes for multi-step audits (URL audit,
    site review, brand search audit) that orchestrate multiple DataForSEO
    calls + external services (Firecrawl, the existing seo-analyze edge
    function).

Cost: every call goes through the unified client which logs to ai_usage_logs
with per-attribution metadata. Caller (the agent tool) supplies user_id +
workspace_id in the request body.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.services.integrations.dataforseo_unified_client import (
    DataForSEOUnifiedClient, get_dataforseo_client,
)
from app.services.integrations.mention_cost_logger import CostAttribution

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/seo-agent",
    tags=["SEO Agent (Internal)"],
    responses={
        401: {"description": "Bad cron secret"},
        404: {"description": "Unknown endpoint kind"},
    },
)


def _check_secret(request: Request) -> None:
    secret = request.headers.get("x-cron-secret")
    expected = os.getenv("CRON_SECRET")
    if not expected or secret != expected:
        raise HTTPException(status_code=401, detail="bad cron secret")


def _attribution_from(body_attr: Optional[Dict[str, Any]]) -> Optional[CostAttribution]:
    if not body_attr:
        return None
    return CostAttribution(
        user_id=body_attr.get("user_id"),
        workspace_id=body_attr.get("workspace_id"),
        api_key_id=body_attr.get("api_key_id"),
        product_id=body_attr.get("product_id"),
        tracked_mention_id=body_attr.get("tracked_mention_id"),
    )


# Whitelist of method names the agent can call via the generic dispatcher.
# Anything not in this set returns 404 — prevents the agent from accidentally
# hitting unrelated DataForSEOUnifiedClient internals.
_ALLOWED_METHODS = {
    # SERP — Google
    "serp_google_organic", "serp_google_maps", "serp_google_local_finder",
    "serp_google_news", "serp_google_images", "serp_google_jobs",
    "serp_google_autocomplete", "serp_google_finance", "serp_google_ai_summary",
    # SERP — other engines
    "serp_bing_organic", "serp_youtube_organic",
    "serp_youtube_video_info", "serp_youtube_video_subtitles", "serp_youtube_video_comments",
    "serp_baidu_organic", "serp_yahoo_organic", "serp_naver_organic", "serp_seznam_organic",
    # AI Optimization
    "ai_keyword_search_volume",
    "ai_llm_mentions_search", "ai_llm_mentions_top_pages",
    "ai_llm_mentions_top_domains", "ai_llm_mentions_aggregated_metrics",
    "ai_llm_response", "ai_llm_models",
    # Keywords Data
    "kw_google_ads_search_volume", "kw_google_ads_keywords_for_site",
    "kw_google_ads_keywords_for_keywords", "kw_google_ads_traffic_by_keywords",
    "kw_bing_ads_search_volume",
    "kw_google_trends_explore", "kw_dataforseo_trends_explore",
    "kw_clickstream_search_volume",
    # Labs — Google
    "labs_related_keywords", "labs_keyword_suggestions", "labs_keyword_ideas",
    "labs_bulk_keyword_difficulty", "labs_search_intent", "labs_keyword_overview",
    "labs_historical_keyword_data", "labs_serp_competitors",
    "labs_ranked_keywords", "labs_competitors_domain", "labs_domain_intersection",
    "labs_subdomains", "labs_relevant_pages", "labs_domain_rank_overview",
    "labs_historical_serps", "labs_historical_rank_overview", "labs_page_intersection",
    "labs_bulk_traffic_estimation", "labs_categories_for_domain", "labs_keywords_for_site",
    # Labs — Amazon + Apps
    "labs_amazon_related_keywords", "labs_amazon_ranked_keywords", "labs_app_keywords",
    # Backlinks
    "backlinks_summary", "backlinks_history", "backlinks_backlinks",
    "backlinks_anchors", "backlinks_referring_domains", "backlinks_competitors",
    "backlinks_domain_intersection", "backlinks_bulk_spam_score",
    "backlinks_bulk_ranks", "backlinks_bulk_referring_domains",
    "backlinks_timeseries_summary",
    # OnPage
    "onpage_task_post", "onpage_summary", "onpage_pages",
    "onpage_duplicate_tags", "onpage_duplicate_content", "onpage_redirect_chains",
    "onpage_non_indexable", "onpage_links", "onpage_keyword_density",
    "onpage_microdata", "onpage_lighthouse", "onpage_instant_pages",
    "onpage_content_parsing",
    # Content Analysis
    "content_search", "content_summary", "content_sentiment_analysis",
    "content_phrase_trends", "content_rating_distribution",
    # Domain Analytics
    "domain_technologies", "domain_whois", "domain_domains_by_technology",
    # Merchant
    "merchant_google_shopping_products", "merchant_amazon_products", "merchant_amazon_asin",
    # App Data
    "app_data_search", "app_data_app_info", "app_data_app_reviews",
    # Business Data
    "business_google_my_business_info", "business_google_reviews",
    "business_trustpilot_search", "business_trustpilot_reviews",
    "business_tripadvisor_search", "business_pinterest", "business_reddit",
    "business_listings_search",
    # Escape hatch
    "call_arbitrary",
}


class DataForSEORequest(BaseModel):
    """Generic dispatch body. `params` is forwarded to the client method
    as **kwargs. `attribution` carries cost-logging context."""
    params: Dict[str, Any] = Field(default_factory=dict, description="Method kwargs.")
    attribution: Optional[Dict[str, Any]] = Field(default=None, description="Cost attribution.")


@router.post("/dataforseo/{kind}")
async def dataforseo_dispatch(
    kind: str,
    request: Request,
    body: DataForSEORequest,
):
    """Generic DataForSEO dispatcher.

    `kind` is a method name on `DataForSEOUnifiedClient`. The body's `params`
    are forwarded as kwargs. Returns the normalized `DataForSEOResult` shape
    plus a `cost_logged` flag.

    Whitelisted to ~70 client methods — anything else returns 404 to stop the
    agent from calling unrelated internals.
    """
    _check_secret(request)
    if kind not in _ALLOWED_METHODS:
        raise HTTPException(status_code=404, detail=f"unknown endpoint kind: {kind}")
    client = get_dataforseo_client()
    method = getattr(client, kind, None)
    if not method or not callable(method):
        raise HTTPException(status_code=404, detail=f"client method missing: {kind}")
    attribution = _attribution_from(body.attribution)
    try:
        result = await method(attribution=attribution, **(body.params or {}))
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"bad params: {e}")
    return {
        "success": result.ok,
        "data": {
            "kind": kind,
            "items": result.items,
            "raw": result.raw,
            "status_code": result.status_code,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
            "error": result.error,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Composed audits — multi-call orchestrations that produce a single coherent
# report. These wrap several DataForSEO calls + occasionally external services
# so the agent can fire one tool to get a meaningful artifact.
# ─────────────────────────────────────────────────────────────────────────────

class SiteReviewRequest(BaseModel):
    """Body for `POST /site-review`. Composite domain audit."""
    domain: str = Field(..., min_length=3, description="Domain to audit, e.g. 'flobali.gr'.")
    country_code: Optional[str] = Field(None, description="ISO-3166 alpha-2.")
    language_code: str = Field("en")
    include_backlinks: bool = Field(True)
    include_top_keywords: bool = Field(True)
    include_competitors: bool = Field(True)
    top_keywords_limit: int = Field(20, ge=5, le=100)
    competitors_limit: int = Field(10, ge=3, le=30)
    attribution: Optional[Dict[str, Any]] = None


@router.post("/site-review")
async def site_review(request: Request, body: SiteReviewRequest):
    """Composite domain audit:
       - domain_rank_overview   (rank, traffic, ranking-keywords count)
       - ranked_keywords        (top organic keywords by traffic)
       - competitors_domain     (top competitors)
       - backlinks_summary      (referring domains, total backlinks, spam score)

    Calls run in parallel. Per-section failures don't fail the whole call.
    Internal cost: ~$0.005-0.012 depending on which sections are enabled.
    """
    _check_secret(request)
    import asyncio
    client = get_dataforseo_client()
    attribution = _attribution_from(body.attribution)

    # Build the parallel task set
    tasks = {
        "domain_rank_overview": client.labs_domain_rank_overview(
            target=body.domain, country_code=body.country_code,
            language_code=body.language_code, attribution=attribution,
        ),
    }
    if body.include_top_keywords:
        tasks["ranked_keywords"] = client.labs_ranked_keywords(
            target=body.domain, country_code=body.country_code,
            language_code=body.language_code, limit=body.top_keywords_limit,
            attribution=attribution,
        )
    if body.include_competitors:
        tasks["competitors"] = client.labs_competitors_domain(
            target=body.domain, country_code=body.country_code,
            language_code=body.language_code, limit=body.competitors_limit,
            attribution=attribution,
        )
    if body.include_backlinks:
        tasks["backlinks_summary"] = client.backlinks_summary(
            target=body.domain, attribution=attribution,
        )
        tasks["backlinks_anchors"] = client.backlinks_anchors(
            target=body.domain, limit=10, attribution=attribution,
        )

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    out: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for (key, _), result in zip(tasks.items(), results):
        if isinstance(result, Exception):
            errors[key] = str(result)[:200]
            continue
        if not result.ok:
            errors[key] = result.error or "unknown error"
            continue
        out[key] = {"items": result.items, "cost_usd": result.cost_usd, "latency_ms": result.latency_ms}

    return {
        "success": True,
        "data": {
            "domain": body.domain,
            "country": body.country_code,
            "language": body.language_code,
            "sections": out,
            "errors": errors,
        },
    }


class KeywordGapRequest(BaseModel):
    """Body for `POST /keyword-gap`. Returns keywords that competitor ranks for
    but the user's domain doesn't — pure content-gap delta."""
    your_domain: str
    competitor_domain: str
    country_code: Optional[str] = None
    language_code: str = "en"
    limit: int = Field(100, ge=10, le=1000)
    attribution: Optional[Dict[str, Any]] = None


@router.post("/keyword-gap")
async def keyword_gap(request: Request, body: KeywordGapRequest):
    """Keyword-gap analysis using DataForSEO Labs `domain_intersection`.
    `intersections=False` returns keywords where target1 (you) does NOT
    rank but target2 (competitor) does."""
    _check_secret(request)
    client = get_dataforseo_client()
    attribution = _attribution_from(body.attribution)
    result = await client.labs_domain_intersection(
        targets=[body.your_domain, body.competitor_domain],
        country_code=body.country_code, language_code=body.language_code,
        limit=body.limit, intersections=False,
        attribution=attribution,
    )
    return {
        "success": result.ok,
        "data": {
            "your_domain": body.your_domain,
            "competitor_domain": body.competitor_domain,
            "items": result.items,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
            "error": result.error,
        },
    }


class BrandSearchAuditRequest(BaseModel):
    """Composite SERP audit for the brand's own name. Reveals KP state,
    own organic listings, AI Overview brand mention, paid bids on own
    brand. Used for brand-defensive SEO."""
    brand_name: str
    country_code: Optional[str] = None
    language_code: str = "en"
    attribution: Optional[Dict[str, Any]] = None


@router.post("/brand-search-audit")
async def brand_search_audit(request: Request, body: BrandSearchAuditRequest):
    _check_secret(request)
    client = get_dataforseo_client()
    attribution = _attribution_from(body.attribution)
    # Single SERP call already returns Knowledge Panel, AI Overview, organic, paid
    result = await client.serp_google_organic(
        keyword=body.brand_name,
        country_code=body.country_code,
        language_code=body.language_code,
        attribution=attribution,
    )
    return {
        "success": result.ok,
        "data": {
            "brand_name": body.brand_name,
            "items": result.items,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
            "error": result.error,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# OnPage long-running orchestration helpers
#
# OnPage crawls are async (POST → poll). The agent tool can either:
#  (a) call /onpage/start then poll /onpage/status until ready
#  (b) call /onpage/quick-page for a single-page instant audit (Lighthouse +
#      content_parsing in parallel) — returns ~10s, no polling needed.
# ─────────────────────────────────────────────────────────────────────────────

class OnpageStartRequest(BaseModel):
    target: str
    max_crawl_pages: int = Field(50, ge=1, le=1000)
    enable_javascript: bool = True
    load_resources: bool = True
    attribution: Optional[Dict[str, Any]] = None


@router.post("/onpage/start")
async def onpage_start(request: Request, body: OnpageStartRequest):
    """Kick off a site crawl. Returns the task_id; caller polls /onpage/summary/{id}."""
    _check_secret(request)
    client = get_dataforseo_client()
    attribution = _attribution_from(body.attribution)
    result = await client.onpage_task_post(
        target=body.target,
        max_crawl_pages=body.max_crawl_pages,
        enable_javascript=body.enable_javascript,
        load_resources=body.load_resources,
        attribution=attribution,
    )
    task_id = None
    for task in (result.raw.get("tasks") or []):
        task_id = task.get("id")
        if task_id:
            break
    return {
        "success": result.ok and bool(task_id),
        "data": {
            "task_id": task_id,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
            "error": result.error,
        },
    }


class QuickPageAuditRequest(BaseModel):
    """Body for `POST /onpage/quick-page`. Single-page instant audit."""
    url: str
    for_mobile: bool = False
    include_lighthouse: bool = True
    include_content_parsing: bool = True
    attribution: Optional[Dict[str, Any]] = None


@router.post("/onpage/quick-page")
async def onpage_quick_page(request: Request, body: QuickPageAuditRequest):
    """Single-URL instant audit: instant_pages + lighthouse + content_parsing
    in parallel. Returns immediately — no task polling. Used by the agent's
    `seo_audit_url` tool."""
    _check_secret(request)
    import asyncio
    client = get_dataforseo_client()
    attribution = _attribution_from(body.attribution)

    tasks: Dict[str, Any] = {
        "instant_page": client.onpage_instant_pages(
            url=body.url, attribution=attribution,
        ),
    }
    if body.include_lighthouse:
        tasks["lighthouse"] = client.onpage_lighthouse(
            url=body.url, for_mobile=body.for_mobile, attribution=attribution,
        )
    if body.include_content_parsing:
        tasks["content_parsing"] = client.onpage_content_parsing(
            url=body.url, attribution=attribution,
        )

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    out: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for (key, _), result in zip(tasks.items(), results):
        if isinstance(result, Exception):
            errors[key] = str(result)[:200]
            continue
        if not result.ok:
            errors[key] = result.error or "unknown error"
            continue
        out[key] = {"items": result.items, "cost_usd": result.cost_usd, "latency_ms": result.latency_ms}

    return {
        "success": True,
        "data": {
            "url": body.url,
            "for_mobile": body.for_mobile,
            "sections": out,
            "errors": errors,
        },
    }
