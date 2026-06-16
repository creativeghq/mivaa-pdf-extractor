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
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

MODULE_SLUG = "job-research"
DEFAULT_MARKUP = 1.5

# DataForSEO Jobs SERP — flat per-call cost
DATAFORSEO_JOBS_PER_CALL = 0.0006

# Perplexity Sonar — base + token cost
SONAR_PER_CALL = 0.005
SONAR_PRO_PER_CALL = 0.01
SONAR_TOKEN_PER_1K = 0.001

# Firecrawl — per credit, ~1 credit per scrape on standard pages, 5 on JS render
FIRECRAWL_PER_CREDIT = 0.002

# Anthropic Haiku 4.5
HAIKU_INPUT_PER_1K = 0.001
HAIKU_OUTPUT_PER_1K = 0.005


class CostAttribution:
    """Bag of identifiers tagged onto every ai_usage_logs row."""
    __slots__ = ("user_id", "workspace_id", "tracked_job_id", "refresh_run_id", "api_key_id")

    def __init__(
        self,
        *,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        tracked_job_id: Optional[str] = None,
        refresh_run_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
    ):
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.tracked_job_id = tracked_job_id
        self.refresh_run_id = refresh_run_id
        self.api_key_id = api_key_id

    def to_metadata(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if self.tracked_job_id:
            meta["tracked_job_id"] = self.tracked_job_id
        if self.refresh_run_id:
            meta["refresh_run_id"] = self.refresh_run_id
        if self.api_key_id:
            meta["api_key_id"] = self.api_key_id
        return meta


def log_external_call(
    *,
    operation_type: str,
    model_name: str,
    raw_cost_usd: float,
    attribution: Optional[CostAttribution] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: int = 0,
    credits_debited: int = 0,
    extra_metadata: Optional[Dict[str, Any]] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    markup_multiplier: float = DEFAULT_MARKUP,
) -> None:
    """Best-effort insert into ai_usage_logs. Never raises."""
    try:
        sb = get_supabase_client().client
        billed = round(float(raw_cost_usd) * float(markup_multiplier), 6)
        meta: Dict[str, Any] = {"latency_ms": latency_ms, "success": success}
        if error_message:
            meta["error"] = (error_message or "")[:240]
        if attribution:
            meta.update(attribution.to_metadata())
        if extra_metadata:
            meta.update(extra_metadata)

        sb.table("ai_usage_logs").insert({
            "user_id": attribution.user_id if attribution else None,
            "workspace_id": attribution.workspace_id if attribution else None,
            "operation_type": operation_type,
            "model_name": model_name,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "raw_cost_usd": round(float(raw_cost_usd), 6),
            "markup_multiplier": float(markup_multiplier),
            "billed_cost_usd": billed,
            "credits_debited": int(credits_debited or 0),
            "module_slug": MODULE_SLUG,
            "metadata": meta,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.warning(f"job-cost: ai_usage_logs insert failed: {e}")


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
    per_call = SONAR_PRO_PER_CALL if model == "sonar-pro" else SONAR_PER_CALL
    token_cost = ((input_tokens + output_tokens) / 1000.0) * SONAR_TOKEN_PER_1K
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


def debit_credits(*, user_id: str, amount: int, operation_type: str) -> bool:
    if amount <= 0 or not user_id:
        return amount <= 0
    try:
        sb = get_supabase_client().client
        result = sb.rpc("debit_user_credits", {
            "p_user_id": user_id,
            "p_amount": amount,
            "p_operation_type": operation_type,
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


def refund_credits(*, user_id: str, amount: int, operation_type: str) -> None:
    if amount <= 0 or not user_id:
        return
    try:
        sb = get_supabase_client().client
        sb.rpc("credit_user_credits", {
            "p_user_id": user_id,
            "p_amount": amount,
            "p_operation_type": f"{operation_type}.refund",
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
