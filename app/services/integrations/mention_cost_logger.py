"""
Mention Monitoring — cost logging + credit metering helpers.

Single chokepoint for writing `ai_usage_logs` entries from the mention
monitoring services. Every external API call (DataForSEO, Perplexity,
Anthropic, OpenAI, Gemini) goes through `log_external_call()` so we get:

  - Per-subject cost attribution (metadata.tracked_mention_id)
  - Per-product cost attribution (product_id when internal-flow)
  - Per-run cost attribution (metadata.refresh_run_id when refreshing)
  - Module-level rollup via module_slug='mention-monitoring'

Also exposes credit debit/refund helpers for the partner-billing layer
(Layer B) — endpoints debit credits before doing work and refund on
failure, mirroring how price-tracking handles partner usage.

Why a dedicated module instead of using AICallLogger directly:
  - AICallLogger.log_ai_call requires a confidence_score + breakdown that's
    meaningful for catalog AI (vision, classification, extraction) but
    awkward for mention discovery / classifier calls.
  - We want a flat "log this external call with these costs" interface
    that doesn't pretend to compute confidence.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx

from app.services.core.supabase_client import get_supabase_client
from app.modules._core.cost_logger import (
    CostAttribution as _CoreCostAttribution,
    log_external_call as _core_log_external_call,
    DEFAULT_MARKUP,
)

logger = logging.getLogger(__name__)

MODULE_SLUG = "mention-monitoring"


def _slug_for(attribution: Optional["CostAttribution"]) -> str:
    """Which module's budget a DataForSEO call belongs to.

    The two `log_dataforseo_*` helpers below are called by the SHARED
    `dataforseo_unified_client`, which is a singleton used by both mention-monitoring and the SEO
    agent toolkit. They lived in this module and hardcoded MODULE_SLUG, so every SEO DataForSEO
    call was filed under 'mention-monitoring': `seo-toolkit` had 0 rows in ai_usage_logs over 30
    days while the operator dashboard summed `seo_research_runs.cost_usd`, which is a hardcoded 0.
    Real spend, attributed to the wrong module, invisible in both places it was looked for.

    The caller sets `module_slug` on its attribution; anything that doesn't stays on the historical
    default, so existing mention-monitoring rows keep their slug and remain comparable. (#286)
    """
    return getattr(attribution, "module_slug", None) or MODULE_SLUG


def _op_prefix(attribution: Optional["CostAttribution"]) -> str:
    """`operation_type` must agree with `module_slug` — an 'seo-toolkit' row reading
    `mention_monitoring.opportunities.…` is the same mislabelling one column over."""
    slug = _slug_for(attribution)
    return "mention_monitoring.opportunities" if slug == MODULE_SLUG else slug.replace("-", "_")


# ────────────────────────────────────────────────────────────────────────────
# Attribution context — threaded through service callers
# ────────────────────────────────────────────────────────────────────────────

class CostAttribution(_CoreCostAttribution):
    """Mention-monitoring attribution — keeps the `tracked_mention_id=` (+ optional
    `product_id=`) keyword API while delegating storage to the shared core."""
    __slots__ = ()

    def __init__(self, *, tracked_mention_id: Optional[str] = None, **kwargs):
        super().__init__(subject_key="tracked_mention_id", subject_id=tracked_mention_id, **kwargs)


# ────────────────────────────────────────────────────────────────────────────
# Pricing tables — used when we only know the count of "calls" not tokens.
# Per-call costs (USD) for each external provider. Markup is applied on top.
# ────────────────────────────────────────────────────────────────────────────

# DataForSEO: tiny per-request cost. The exact number depends on plan tier,
# but $0.0006 is the documented standard rate for SERP / News.
DATAFORSEO_NEWS_PER_CALL = 0.0006
DATAFORSEO_LABS_PER_CALL = 0.001     # related-keywords endpoint

# Perplexity Sonar: per-request fixed cost. Token cost is on top of this.
SONAR_PER_CALL = 0.005
SONAR_PRO_PER_CALL = 0.01

# YouTube Data API: free quota; cost is 0 for our purposes.
YOUTUBE_PER_CALL = 0.0

# Anthropic Haiku 4.5: per 1K tokens. Used by classifier + facet extraction
# + opportunity LLM polish.
HAIKU_INPUT_PER_1K = 0.001
HAIKU_OUTPUT_PER_1K = 0.005

# OpenAI gpt-4o-mini: per 1K tokens (LLM probe).
GPT4O_MINI_INPUT_PER_1K = 0.00015
GPT4O_MINI_OUTPUT_PER_1K = 0.0006

# Gemini 2.0 Flash: per 1K tokens (LLM probe).
GEMINI_FLASH_INPUT_PER_1K = 0.00010
GEMINI_FLASH_OUTPUT_PER_1K = 0.0004


# ────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ────────────────────────────────────────────────────────────────────────────

def log_external_call(**kwargs) -> None:
    """Insert one row into ai_usage_logs — delegates to the shared core, defaulting
    module_slug='mention-monitoring'. Signature unchanged for existing callers.

    `module_slug` is a DEFAULT, not a constant: the two `log_dataforseo_*` helpers below are
    reached from the SHARED unified client and pass the caller's own slug. Injecting it
    unconditionally (`_core_log_external_call(module_slug=MODULE_SLUG, **kwargs)`) raised
    `TypeError: got multiple values for keyword argument 'module_slug'` the moment one of them
    did — a runtime failure inside a best-effort logger, i.e. swallowed, i.e. cost logging
    silently stops. `setdefault` keeps every existing caller identical and lets an explicit one
    win. (#286)

    Per-row cost rollups (Layer C `recompute_mention_cost`) still sum these by
    tracked_mention_id; product_id is set at the column level from attribution.
    """
    kwargs.setdefault("module_slug", MODULE_SLUG)
    _core_log_external_call(**kwargs)


def log_dataforseo_news_call(
    *,
    attribution: Optional[CostAttribution],
    query: str,
    hits_returned: int,
    latency_ms: int,
    success: bool = True,
    error_message: Optional[str] = None,
) -> None:
    log_external_call(
        operation_type="mention_monitoring.discovery.dataforseo_news",
        model_name="dataforseo-news",
        raw_cost_usd=DATAFORSEO_NEWS_PER_CALL,
        attribution=attribution,
        latency_ms=latency_ms,
        extra_metadata={"query": query[:120], "hits_returned": hits_returned},
        success=success,
        error_message=error_message,
    )


def log_dataforseo_labs_call(
    *,
    attribution: Optional[CostAttribution],
    seed_keyword: str,
    items_returned: int,
    latency_ms: int,
    success: bool = True,
    error_message: Optional[str] = None,
) -> None:
    log_external_call(
        module_slug=_slug_for(attribution),
        operation_type=f"{_op_prefix(attribution)}.dataforseo_labs",
        model_name="dataforseo-labs-related-keywords",
        raw_cost_usd=DATAFORSEO_LABS_PER_CALL,
        attribution=attribution,
        latency_ms=latency_ms,
        extra_metadata={"seed": seed_keyword[:120], "items_returned": items_returned},
        success=success,
        error_message=error_message,
    )


def log_dataforseo_serp_call(
    *,
    attribution: Optional[CostAttribution],
    operation: str,                    # 'pao_question' / 'serp_organic'
    query: str,
    items_returned: int,
    latency_ms: int,
    success: bool = True,
    error_message: Optional[str] = None,
) -> None:
    log_external_call(
        module_slug=_slug_for(attribution),
        operation_type=f"{_op_prefix(attribution)}.dataforseo_serp.{operation}",
        model_name="dataforseo-serp-google-organic",
        raw_cost_usd=DATAFORSEO_NEWS_PER_CALL,  # SERP API priced same as News
        attribution=attribution,
        latency_ms=latency_ms,
        extra_metadata={"query": query[:120], "items_returned": items_returned},
        success=success,
        error_message=error_message,
    )


def log_perplexity_call(
    *,
    attribution: Optional[CostAttribution],
    model: str,                        # 'sonar' or 'sonar-pro'
    input_tokens: int,
    output_tokens: int,
    hits_returned: int,
    latency_ms: int,
    success: bool = True,
    error_message: Optional[str] = None,
) -> None:
    per_call = SONAR_PRO_PER_CALL if model == "sonar-pro" else SONAR_PER_CALL
    # Perplexity pricing: per_call + token cost. Token-cost rates are roughly
    # equal to OpenAI's small-model band; we approximate at $0.001/1K both ways.
    token_cost = ((input_tokens + output_tokens) / 1000.0) * 0.001
    raw = per_call + token_cost
    log_external_call(
        operation_type=f"mention_monitoring.discovery.perplexity_{model}",
        model_name=model,
        raw_cost_usd=raw,
        attribution=attribution,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        extra_metadata={"hits_returned": hits_returned},
        success=success,
        error_message=error_message,
    )


def log_haiku_call(
    *,
    attribution: Optional[CostAttribution],
    operation: str,                    # 'facet_extraction' / 'classifier' / 'opportunity_polish'
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    success: bool = True,
    error_message: Optional[str] = None,
) -> None:
    raw = (
        (input_tokens / 1000.0) * HAIKU_INPUT_PER_1K
        + (output_tokens / 1000.0) * HAIKU_OUTPUT_PER_1K
    )
    log_external_call(
        operation_type=f"mention_monitoring.{operation}",
        model_name="claude-haiku-4-5-20251001",
        raw_cost_usd=raw,
        attribution=attribution,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        success=success,
        error_message=error_message,
    )


def log_llm_probe_call(
    *,
    attribution: Optional[CostAttribution],
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    success: bool = True,
    error_message: Optional[str] = None,
) -> None:
    if model.startswith("claude-haiku"):
        rates = (HAIKU_INPUT_PER_1K, HAIKU_OUTPUT_PER_1K)
    elif model == "gpt-4o-mini":
        rates = (GPT4O_MINI_INPUT_PER_1K, GPT4O_MINI_OUTPUT_PER_1K)
    elif model.startswith("gemini"):
        rates = (GEMINI_FLASH_INPUT_PER_1K, GEMINI_FLASH_OUTPUT_PER_1K)
    elif model == "sonar":
        rates = (0.001, 0.001)
    else:
        rates = (0.0005, 0.0015)  # conservative default
    raw = (input_tokens / 1000.0) * rates[0] + (output_tokens / 1000.0) * rates[1]
    log_external_call(
        operation_type="mention_monitoring.llm_probe",
        model_name=model,
        raw_cost_usd=raw,
        attribution=attribution,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        success=success,
        error_message=error_message,
    )


def log_youtube_call(
    *,
    attribution: Optional[CostAttribution],
    query: str,
    hits_returned: int,
    latency_ms: int,
    success: bool = True,
    error_message: Optional[str] = None,
) -> None:
    log_external_call(
        operation_type="mention_monitoring.discovery.youtube",
        model_name="youtube-data-api-v3",
        raw_cost_usd=YOUTUBE_PER_CALL,
        attribution=attribution,
        latency_ms=latency_ms,
        extra_metadata={"query": query[:120], "hits_returned": hits_returned},
        success=success,
        error_message=error_message,
    )


# ────────────────────────────────────────────────────────────────────────────
# Layer B: partner credit metering
# ────────────────────────────────────────────────────────────────────────────

# Per-operation credit cost. Tunable.
MENTION_OP_CREDIT_COST: Dict[str, int] = {
    "refresh": 5,
    "probe_llm": 15,
    "opportunities": 2,
    "opportunities_with_llm": 5,
    "market_check": 3,           # reserved for future stateless endpoint
}


def debit_credits(
    *, user_id: str, amount: int, operation_type: str, workspace_id: Optional[str] = None,
) -> bool:
    """Atomic credit debit via the shared credit router.
    Routes to the workspace pool when funded (workspace_id set + pool exists), else the
    personal wallet. Partner (api_key) callers pass no workspace_id → personal (unchanged).
    Returns True on success, False on insufficient balance / failure.
    """
    if amount <= 0:
        return True
    if not user_id:
        return False
    try:
        sb = get_supabase_client().client
        result = sb.rpc("debit_credits", {
            "p_user_id": user_id,
            "p_amount": amount,
            "p_operation_type": operation_type,
            "p_workspace_id": workspace_id,
        }).execute()
        # The RPC returns [{success: bool, ...}] — insufficient balance yields
        # success=false (a non-empty, truthy row). bool(data) would treat that as a
        # successful debit → paid op served free (audit #217 H3).
        data = getattr(result, "data", None)
        if not data:
            return False
        row = data[0] if isinstance(data, list) else data
        return bool(row.get("success")) if isinstance(row, dict) else bool(row)
    except Exception as e:
        logger.info(f"mention-cost: credit debit skipped (likely insufficient): {e}")
        return False


def refund_credits(
    *, user_id: str, amount: int, operation_type: str, workspace_id: Optional[str] = None,
) -> None:
    """Best-effort refund via the shared credit router. Never raises."""
    if amount <= 0 or not user_id:
        return
    try:
        sb = get_supabase_client().client
        sb.rpc("refund_credits", {
            "p_user_id": user_id,
            "p_amount": amount,
            "p_operation_type": f"{operation_type}.refund",
            "p_workspace_id": workspace_id,
        }).execute()
    except Exception as e:
        logger.info(f"mention-cost: refund failed (non-fatal): {e}")


# ────────────────────────────────────────────────────────────────────────────
# Layer C: per-row rollup helpers
# ────────────────────────────────────────────────────────────────────────────

def stamp_refresh_cost(*, tracked_mention_id: str, refresh_run_id: str) -> None:
    """After a refresh persists rows, call this to update last_refresh_*
    counters on tracked_mentions and recompute the lifetime totals."""
    if not tracked_mention_id or not refresh_run_id:
        return
    try:
        sb = get_supabase_client().client
        sb.rpc("stamp_mention_refresh_cost", {
            "p_tracked_mention_id": tracked_mention_id,
            "p_refresh_run_id": refresh_run_id,
        }).execute()
    except Exception as e:
        logger.warning(f"mention-cost: stamp_mention_refresh_cost failed: {e}")


def recompute_lifetime_cost(*, tracked_mention_id: str) -> None:
    """Sum all ai_usage_logs entries for this tracked_mention_id and write the
    total back to tracked_mentions.total_billed_usd. Useful after probe-llm
    or opportunities calls (which don't have a refresh_run_id)."""
    if not tracked_mention_id:
        return
    try:
        sb = get_supabase_client().client
        sb.rpc("recompute_mention_cost", {
            "p_tracked_mention_id": tracked_mention_id,
        }).execute()
    except Exception as e:
        logger.warning(f"mention-cost: recompute_mention_cost failed: {e}")
