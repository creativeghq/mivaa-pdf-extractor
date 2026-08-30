"""
Shared monitoring cost-logging core.

The per-module cost loggers (job_cost_logger.py, mention_cost_logger.py, and any
future monitoring module) had a byte-for-byte duplicate `CostAttribution` bag +
`log_external_call()` ai_usage_logs insert, differing only in `module_slug` and
the subject-id metadata key (tracked_job_id / tracked_mention_id). This is the
single implementation.

Each module keeps its OWN pricing tables, per-source `log_*_call` wrappers, and
credit debit/refund/stamp RPCs (those genuinely differ) and delegates the insert
here via a thin `CostAttribution` subclass + a `log_external_call` passthrough
that fills in `module_slug`. No call site changes — behaviour is identical.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.modules._core.cost_accounting import intended_cost_usd, settle_call_cost
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

DEFAULT_MARKUP = 1.5


class CostAttribution:
    """Bag of identifiers tagged onto every ai_usage_logs row.

    `subject_key`/`subject_id` are the monitoring subject (e.g.
    ('tracked_job_id', <uuid>)) written into metadata. `product_id` is a column
    (internal-flow rows have it; external-API rows don't). Per-module subclasses
    pass `subject_key` so their existing keyword API (tracked_job_id=…) is kept.
    """
    __slots__ = (
        "user_id", "workspace_id", "subject_key", "subject_id",
        "product_id", "refresh_run_id", "api_key_id", "module_slug",
    )

    def __init__(
        self,
        *,
        subject_key: str,
        subject_id: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        product_id: Optional[str] = None,
        refresh_run_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        module_slug: Optional[str] = None,
    ):
        self.subject_key = subject_key
        self.subject_id = subject_id
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.product_id = product_id
        self.refresh_run_id = refresh_run_id
        self.api_key_id = api_key_id
        # Which module's budget this call belongs to. Optional and defaulted by the caller's
        # logger, because SHARED clients cannot know it: the DataForSEO unified client is a
        # singleton used by both mention-monitoring and the SEO toolkit, so a module baked into
        # the client (or into its logger module) files every caller's spend under one slug. That
        # is exactly what happened — all SEO DataForSEO spend landed under 'mention-monitoring'
        # and `seo-toolkit` showed 0 rows while the operator dashboard read a hardcoded 0.
        # It travels with the ATTRIBUTION because the attribution is built by the call site.
        self.module_slug = module_slug

    def to_metadata(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if self.subject_id:
            meta[self.subject_key] = self.subject_id
        if self.refresh_run_id:
            meta["refresh_run_id"] = self.refresh_run_id
        if self.api_key_id:
            meta["api_key_id"] = self.api_key_id
        return meta


#: Marker written into `ai_usage_logs.error_message` when a call reports success and
#: no usage at all (#30 M16-6). Named rather than inlined so a cost view can select on
#: it, and so the two loggers cannot drift to two spellings of the same fact.
ZERO_USAGE_MARKER = "usage_missing: provider reported 0 input and 0 output tokens"


def usage_anomaly(input_tokens: int, output_tokens: int, success: bool) -> Optional[str]:
    """A marker when a SUCCESSFUL call reports no tokens, else None.

    A model call that succeeded consumed tokens. Zero of both is not a cheap call — it
    is an accounting failure, and it does not fail: `int(x or 0)` turns a missing usage
    block into 0, 0 tokens produce 0 raw cost, and the row is still written
    `success=True`. That feeds #30 M16-1 directly — a zero cost becomes a zero amount
    becomes a free operation reporting success, all the way down.

    `success` stays whatever the caller reported. What is being marked is the USAGE, not
    the call: the provider may genuinely have answered. Flipping success to False would
    trade a wrong cost for a wrong outcome.
    """
    if success and not int(input_tokens or 0) and not int(output_tokens or 0):
        return ZERO_USAGE_MARKER
    return None


def log_external_call(
    *,
    module_slug: str,
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
        # A call that did not happen did not cost anything. This used to be
        # `raw_cost_usd * markup` regardless of `success`, so every flat-rate provider booked
        # its per-call price for refusals — $1.22 of billed spend in seven days that nobody
        # ever spent. `success` lived only in metadata, so no cost view could tell.
        raw_usd, billed, unbilled_reason = settle_call_cost(raw_cost_usd, markup_multiplier, success)

        meta: Dict[str, Any] = {"latency_ms": latency_ms, "success": success}
        if unbilled_reason:
            # Keep the price it WOULD have carried. Zeroing the cost must not destroy the
            # difference between "this provider is free" and "this provider refused us".
            meta["would_have_cost_usd"] = intended_cost_usd(raw_cost_usd)
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
            "raw_cost_usd": raw_usd,
            "markup_multiplier": float(markup_multiplier),
            "billed_cost_usd": billed,
            "unbilled_reason": unbilled_reason,
            "credits_debited": int(credits_debited or 0),
            "module_slug": module_slug,
            "product_id": attribution.product_id if attribution else None,
            "metadata": meta,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.warning(f"[{module_slug}] ai_usage_logs insert failed: {e}")
