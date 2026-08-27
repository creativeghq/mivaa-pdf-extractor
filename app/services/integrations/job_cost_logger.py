"""
Job Research — cost logging + credit metering helpers.

Single chokepoint for writing `ai_usage_logs` entries from the job-research
services. Every external API call (DataForSEO Jobs, Perplexity Sonar,
Firecrawl, Anthropic Haiku) goes through `log_external_call()` so we get:

  - Per-subject cost attribution (metadata.tracked_job_id)
  - Per-run cost attribution (metadata.refresh_run_id)
  - Module-level rollup via module_slug='job-research'

Mirrors mention_cost_logger.py — same ai_usage_logs schema, different
operation_type prefix and credit table.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.credit_router import (
    debit_credits as _router_debit_credits,
    refund_credits as _router_refund_credits,
)
from app.modules._core.provider_pricing import (
    DATAFORSEO_SERP_PER_CALL,
    FIRECRAWL_PER_CREDIT,
    haiku_token_cost,
    sonar_rates,
)
from app.modules._core.cost_logger import (
    CostAttribution as _CoreCostAttribution,
    log_external_call as _core_log_external_call,
)

logger = logging.getLogger(__name__)

MODULE_SLUG = "job-research"

# Provider rates live in ONE place — app/modules/_core/provider_pricing.py. This file used to
# carry its own copies of the six Sonar constants, the two Haiku constants and the DataForSEO SERP
# rate, all duplicated verbatim in mention_cost_logger. Two copies of a price cannot disagree
# loudly: a wrong rate is a valid number, nothing raises, and the totals look plausible. That is
# precisely how Sonar Pro ended up 3x/15x light and DataForSEO Labs ~12x light, and why fixing them
# meant editing two files that nothing tied together. (main repo #365)
#
# `DATAFORSEO_SERP_PER_CALL` is gone: it held the same $0.0006 standard-queue rate as the News
# constant next door, under a second name, which read as two independent facts and was one.


class CostAttribution(_CoreCostAttribution):
    """Job-research attribution — keeps the `tracked_job_id=` keyword API while
    delegating storage/serialization to the shared core."""
    __slots__ = ()

    def __init__(self, *, tracked_job_id: Optional[str] = None, **kwargs):
        super().__init__(subject_key="tracked_job_id", subject_id=tracked_job_id, **kwargs)


def log_external_call(**kwargs) -> None:
    """Best-effort insert into ai_usage_logs — delegates to the shared core,
    stamping module_slug='job-research'. Signature unchanged for callers."""
    _core_log_external_call(module_slug=MODULE_SLUG, **kwargs)


def log_dataforseo_jobs_call(
    *,
    attribution: Optional[CostAttribution],
    query: str,
    location: str,
    hits_returned: int,
    latency_ms: int,
    success: bool = True,
    error_message: Optional[str] = None,
) -> None:
    log_external_call(
        operation_type="job_research.discovery.dataforseo_jobs",
        model_name="dataforseo-google-jobs",
        raw_cost_usd=DATAFORSEO_SERP_PER_CALL,
        attribution=attribution,
        latency_ms=latency_ms,
        extra_metadata={"query": query[:120], "location": location[:80], "hits_returned": hits_returned},
        success=success,
        error_message=error_message,
    )


def log_perplexity_call(
    *,
    attribution: Optional[CostAttribution],
    model: str,
    input_tokens: int,
    output_tokens: int,
    hits_returned: int,
    latency_ms: int,
    success: bool = True,
    error_message: Optional[str] = None,
) -> None:
    per_call, in_rate, out_rate = sonar_rates(model)
    token_cost = (input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate
    raw = per_call + token_cost
    log_external_call(
        operation_type=f"job_research.discovery.perplexity_{model}",
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


def log_firecrawl_call(
    *,
    attribution: Optional[CostAttribution],
    url: str,
    credits_used: int,
    listings_extracted: int,
    latency_ms: int,
    success: bool = True,
    error_message: Optional[str] = None,
) -> None:
    raw = float(credits_used) * FIRECRAWL_PER_CREDIT
    log_external_call(
        operation_type="job_research.discovery.firecrawl_careers",
        model_name="firecrawl-v2",
        raw_cost_usd=raw,
        attribution=attribution,
        latency_ms=latency_ms,
        extra_metadata={"url": url[:200], "credits_used": credits_used, "listings_extracted": listings_extracted},
        success=success,
        error_message=error_message,
    )


def log_haiku_call(
    *,
    attribution: Optional[CostAttribution],
    operation: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    success: bool = True,
    error_message: Optional[str] = None,
) -> None:
    # Resolved through ai_model_pricing, not restated. `claude-haiku-4-5` has a row there, and a
    # literal beside it is a second USD source that keeps the old number after any admin edit.
    raw = haiku_token_cost(input_tokens, output_tokens)
    log_external_call(
        operation_type=f"job_research.{operation}",
        model_name="claude-haiku-4-5-20251001",
        raw_cost_usd=raw,
        attribution=attribution,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        success=success,
        error_message=error_message,
    )


# ─── Layer B: partner credit metering for external (api_key) flow ──────────
JOB_OP_CREDIT_COST: Dict[str, int] = {
    "refresh": 5,
    "digest_preview": 1,
}


def debit_credits(
    *, user_id: str, amount: int, operation_type: str,
    workspace_id: Optional[str] = None,
) -> bool:
    """Thin delegate to the shared credit router.

    The implementation used to live here, byte-identical to the copies in the two
    sibling cost loggers apart from a log prefix. Audit #30 M16-1 had to be fixed in
    all three at once, which is what a copy costs. The NAME stays so callers do not
    change; the rule lives in `credit_router`.
    """
    return _router_debit_credits(
        user_id=user_id, amount=amount, operation_type=operation_type,
        workspace_id=workspace_id, module="job-cost",
    )


def refund_credits(
    *, user_id: str, amount: int, operation_type: str,
    workspace_id: Optional[str] = None,
) -> None:
    """Thin delegate to the shared credit router.

    The implementation used to live here, byte-identical to the copies in the two
    sibling cost loggers apart from a log prefix. Audit #30 M16-1 had to be fixed in
    all three at once, which is what a copy costs. The NAME stays so callers do not
    change; the rule lives in `credit_router`.
    """
    _router_refund_credits(
        user_id=user_id, amount=amount, operation_type=operation_type,
        workspace_id=workspace_id, module="job-cost",
    )


# ─── Layer C: per-row rollup helpers ───────────────────────────────────────
def stamp_refresh_cost(*, tracked_job_id: str, refresh_run_id: str) -> None:
    if not tracked_job_id or not refresh_run_id:
        return
    try:
        sb = get_supabase_client().client
        sb.rpc("stamp_job_refresh_cost", {
            "p_tracked_job_id": tracked_job_id,
            "p_refresh_run_id": refresh_run_id,
        }).execute()
    except Exception as e:
        logger.warning(f"job-cost: stamp_job_refresh_cost failed: {e}")


def recompute_lifetime_cost(*, tracked_job_id: str) -> None:
    if not tracked_job_id:
        return
    try:
        sb = get_supabase_client().client
        sb.rpc("recompute_job_cost", {"p_tracked_job_id": tracked_job_id}).execute()
    except Exception as e:
        logger.warning(f"job-cost: recompute_job_cost failed: {e}")
