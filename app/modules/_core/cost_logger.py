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
        "product_id", "refresh_run_id", "api_key_id",
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
    ):
        self.subject_key = subject_key
        self.subject_id = subject_id
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.product_id = product_id
        self.refresh_run_id = refresh_run_id
        self.api_key_id = api_key_id

    def to_metadata(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if self.subject_id:
            meta[self.subject_key] = self.subject_id
        if self.refresh_run_id:
            meta["refresh_run_id"] = self.refresh_run_id
        if self.api_key_id:
            meta["api_key_id"] = self.api_key_id
        return meta


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
            "module_slug": module_slug,
            "product_id": attribution.product_id if attribution else None,
            "metadata": meta,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.warning(f"[{module_slug}] ai_usage_logs insert failed: {e}")
