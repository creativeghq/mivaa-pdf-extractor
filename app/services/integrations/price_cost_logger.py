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

from app.services.integrations.credit_router import (
    debit_credits as _router_debit_credits,
    refund_credits as _router_refund_credits,
)

logger = logging.getLogger(__name__)


# ─── Layer B: partner credit metering for external (api_key) flow ──────────
# Per-operation credit cost. Tunable. Mirrors MENTION_OP_CREDIT_COST / JOB_OP_CREDIT_COST.
PRICE_OP_CREDIT_COST: Dict[str, int] = {
    "track": 5,           # create + synchronous first refresh (full discovery + verify)
    "refresh": 5,         # forced refresh (full discovery + verify)
    "lookup_search": 3,   # /lookup search-query mode (Perplexity + DataForSEO + Firecrawl verify)
    "lookup_url": 1,      # /lookup url mode (single Firecrawl scrape)
    "verify": 2,          # on-demand re-verify of known URLs (Firecrawl only, no discovery)
    "url_only": 1,        # pin one retailer URL + its first scrape
    "market_check": 3,    # stateless market scan (Perplexity + optional Firecrawl verify)
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
        workspace_id=workspace_id, module="price-cost",
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
        workspace_id=workspace_id, module="price-cost",
    )
