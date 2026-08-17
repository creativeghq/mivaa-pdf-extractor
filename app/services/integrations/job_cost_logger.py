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
from app.modules._core.cost_logger import (
    CostAttribution as _CoreCostAttribution,
    log_external_call as _core_log_external_call,
)

logger = logging.getLogger(__name__)

MODULE_SLUG = "job-research"

# DataForSEO Jobs SERP — flat per-call cost
DATAFORSEO_JOBS_PER_CALL = 0.0006

# Perplexity Sonar — per-request search fee + token cost.
#
# The token rate is PER MODEL and PER DIRECTION. It used to be one constant applied to
# (input + output) for both models, which is correct for Sonar ($1/$1 per 1M) and wrong for
# Sonar Pro ($3 in / $15 out per 1M) — so every Sonar Pro call was recorded 3x light on input and
# 15x light on output. Caught by solving `raw_cost_usd` back out of ai_usage_logs over 76 calls and
# comparing with Perplexity's published rates (main repo #365).
#
# A wrong rate is a valid number: nothing raises, the row inserts, the dashboard totals look fine.
# The only thing that catches it is checking the arithmetic against the provider.
# Source: https://docs.perplexity.ai/getting-started/pricing (verified 2026-08-17)
SONAR_PER_CALL = 0.005          # $5 / 1000 requests
SONAR_PRO_PER_CALL = 0.01       # inside the published $6-14 / 1000 band
SONAR_INPUT_PER_1K = 0.001      # $1 / 1M
SONAR_OUTPUT_PER_1K = 0.001     # $1 / 1M
SONAR_PRO_INPUT_PER_1K = 0.003  # $3 / 1M
SONAR_PRO_OUTPUT_PER_1K = 0.015 # $15 / 1M

# Firecrawl — per credit, ~1 credit per scrape on standard pages, 5 on JS render
FIRECRAWL_PER_CREDIT = 0.002

# Anthropic Haiku 4.5
HAIKU_INPUT_PER_1K = 0.001
HAIKU_OUTPUT_PER_1K = 0.005


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
        raw_cost_usd=DATAFORSEO_JOBS_PER_CALL,
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
    is_pro = model == "sonar-pro"
    per_call = SONAR_PRO_PER_CALL if is_pro else SONAR_PER_CALL
    in_rate = SONAR_PRO_INPUT_PER_1K if is_pro else SONAR_INPUT_PER_1K
    out_rate = SONAR_PRO_OUTPUT_PER_1K if is_pro else SONAR_OUTPUT_PER_1K
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
    raw = (
        (input_tokens / 1000.0) * HAIKU_INPUT_PER_1K
        + (output_tokens / 1000.0) * HAIKU_OUTPUT_PER_1K
    )
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


def debit_credits(*, user_id: str, amount: int, operation_type: str, workspace_id: Optional[str] = None) -> bool:
    if amount <= 0 or not user_id:
        return amount <= 0
    try:
        sb = get_supabase_client().client
        # Route through the shared credit router: workspace pool when funded, else personal.
        # Partner (api_key) callers pass no workspace_id → personal wallet (unchanged).
        result = sb.rpc("debit_credits", {
            "p_user_id": user_id,
            "p_amount": amount,
            "p_operation_type": operation_type,
            "p_workspace_id": workspace_id,
        }).execute()
        # The RPC returns a row [{success: bool, ...}] — on insufficient balance it
        # returns success=false (NOT an empty result). Reading bool(data) treats that
        # truthy row as a successful debit → paid op served free (audit #217 H3).
        data = getattr(result, "data", None)
        if not data:
            return False
        row = data[0] if isinstance(data, list) else data
        return bool(row.get("success")) if isinstance(row, dict) else bool(row)
    except Exception as e:
        logger.info(f"job-cost: credit debit skipped: {e}")
        return False


def refund_credits(*, user_id: str, amount: int, operation_type: str, workspace_id: Optional[str] = None) -> None:
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
        logger.info(f"job-cost: refund failed (non-fatal): {e}")


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
