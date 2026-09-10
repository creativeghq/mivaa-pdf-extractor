"""What a call actually cost, once you know whether it happened."""
from __future__ import annotations

from typing import Optional, Tuple

#: `unbilled_reason` written when the provider call did not succeed. Matches the edge-side
#: constant in `_shared/ai-logger.ts` so one query explains both runtimes.
CALL_FAILED = "call_failed"


def settle_call_cost(
    raw_cost_usd: float,
    markup_multiplier: float,
    success: bool,
) -> Tuple[float, float, Optional[str]]:
    """
    Resolve one provider call into ``(raw_usd, billed_usd, unbilled_reason)``.

    A failed call costs zero and says so. A successful one is priced as before, so this is a
    no-op on every row that was already correct.
    """
    if not success:
        return 0.0, 0.0, CALL_FAILED

    raw = round(float(raw_cost_usd or 0.0), 6)
    billed = round(raw * float(markup_multiplier or 0.0), 6)
    return raw, billed, None


def intended_cost_usd(raw_cost_usd: float) -> float:
    """
    The price the call would have carried had it succeeded — recorded in metadata on a failure.

    Kept so that zeroing the cost never destroys information: the difference between "this
    provider is free" and "this provider charges $0.005 and refused us 35 times" is the whole
    signal when a metric goes flat.
    """
    return round(float(raw_cost_usd or 0.0), 6)
