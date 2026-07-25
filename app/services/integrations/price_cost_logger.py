"""
Price Tracking / Lookup — partner credit metering helpers (Layer B).

The external `kai_*` price flows (`/api/v1/prices/track/*`, `/api/v1/prices/lookup`)
debit credits before running the paid upstream work (Perplexity + DataForSEO +
Firecrawl) and refund on hard failure / no-op — mirroring the mention and job
tracking flows (`mention_cost_logger` / `job_cost_logger`).

Per-call ai_usage_logs rows are already written deeper in the pipeline by the
Perplexity / Firecrawl services (via AICallLogger), so this module only owns the
partner-facing credit debit/refund + the per-operation cost table.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


# ─── Layer B: partner credit metering for external (api_key) flow ──────────
# Per-operation credit cost. Tunable. Mirrors MENTION_OP_CREDIT_COST / JOB_OP_CREDIT_COST.
PRICE_OP_CREDIT_COST: Dict[str, int] = {
    "track": 5,           # create + synchronous first refresh (full discovery + verify)
    "refresh": 5,         # forced refresh (full discovery + verify)
    "lookup_search": 3,   # /lookup search-query mode (Perplexity + DataForSEO + Firecrawl verify)
    "lookup_url": 1,      # /lookup url mode (single Firecrawl scrape)
}


def debit_credits(*, user_id: str, amount: int, operation_type: str, workspace_id: Optional[str] = None) -> bool:
    """Atomic credit debit via the shared credit router. Returns True on success,
    False on insufficient balance / failure. Partner (api_key) callers pass no
    workspace_id → personal wallet."""
    if amount <= 0 or not user_id:
        return amount <= 0
    try:
        sb = get_supabase_client().client
        result = sb.rpc("debit_credits", {
            "p_user_id": user_id,
            "p_amount": amount,
            "p_operation_type": operation_type,
            "p_workspace_id": workspace_id,
        }).execute()
        # The RPC returns [{success: bool, ...}] — insufficient balance yields
        # success=false (a truthy row). bool(data) would treat that as a
        # successful debit → paid op served free (audit #217 H3).
        data = getattr(result, "data", None)
        if not data:
            return False
        row = data[0] if isinstance(data, list) else data
        return bool(row.get("success")) if isinstance(row, dict) else bool(row)
    except Exception as e:
        logger.info(f"price-cost: credit debit skipped: {e}")
        return False


def refund_credits(*, user_id: str, amount: int, operation_type: str, workspace_id: Optional[str] = None) -> None:
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
        logger.info(f"price-cost: refund failed (non-fatal): {e}")
