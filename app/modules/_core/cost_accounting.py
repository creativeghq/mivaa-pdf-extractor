"""
What a call actually cost, once you know whether it happened.

`log_external_call` computed `billed = raw_cost_usd * markup` and wrote it whatever `success`
said, and `success` went only into `metadata`. Every flat-rate provider therefore booked its
per-call price for calls that never happened. Measured 2026-08-30, over seven days:

    dataforseo-labs-related-keywords   53 failed calls   $0.954 billed
    sonar (Perplexity, HTTP 401)       35 failed calls   $0.263 billed
    dataforseo-serp-google-organic      4 failed calls   $0.004 billed
    firecrawl-v2                        1 failed call    $0.003 billed

Token-priced models were never affected: their cost derives from `response.usage`, and a call
that failed returned no tokens, so the arithmetic already produced zero. Only the flat
`per_call` component survives a failure, which is why this reads as a provider-specific bug
and is actually one line in the shared writer.

A 401 is the clearest case — nothing was delivered and nothing was charged upstream — but the
rule is not really about who billed whom. It is the platform's own rule that a metric is a
VALUE or a stated REASON there is no value: a cost dashboard that quietly includes money nobody
spent is wrong in the direction that is hardest to notice, because the number looks plausible.
So the cost goes to zero and `unbilled_reason` says why, which is exactly the contract the edge
side already uses (`ai-logger.ts → logFailedCall`). The price the call WOULD have carried is
kept in metadata rather than destroyed, so if a provider is later found to bill its own
failures, the evidence to reprice is still there.

This module imports nothing. MIVAA's CI installs pytest and no application dependencies, so a
test can only exercise real logic if that logic sits somewhere importable without a database
client — which is the whole reason this is not inlined in `cost_logger.py`.
"""
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
