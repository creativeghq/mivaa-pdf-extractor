"""One wrapper for "this route spends real money", so the debit cannot drift after the spend.

Invariant 10: debit and rate-limit BEFORE the upstream (LLM / Perplexity / DataForSEO /
Firecrawl) call, not after; on debit failure, do not perform the work.

Thirteen user-triggered doors in `price_monitoring_routes` and `mention_monitoring_routes`
entered paid provider work with no checked debit at all (audit #18 M5-3). Each one was
individually plausible — the callee logs usage to `ai_usage_logs` afterwards, so cost
telemetry looked healthy while nobody was ever charged. The reference implementation each
of them should have copied lived in `job_research_routes` as fourteen hand-written lines
(debit / define `_refund` / try / except-refund / outcome-refund), and hand-copying it
thirteen more times is how the fourteenth copy ends up subtly different.

So it lives here once:

    async with metered_door(
        user_id=current_user_id(user), workspace_id=ws,
        cost=costs.PRICE_OP_CREDIT_COST["refresh"],
        operation_type="price_monitoring.refresh",
        debit=costs.debit_credits, refund=costs.refund_credits,
    ) as paid:
        outcome = await service.refresh(tq["id"])
        if outcome.get("status") != "refreshed":
            paid.refund("nothing was delivered")

* The debit happens before the block runs; insufficient balance raises 402 and the block
  never executes.
* An exception inside the block refunds and re-raises.
* `paid.refund(...)` is idempotent, so the exception path cannot double-refund work a door
  already refunded itself. That idempotency is the other half of M5-3: without a wrapper,
  a door that grew a second refund branch would quietly hand back the credit twice.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Optional

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# The signatures of price_cost_logger / mention_cost_logger / job_cost_logger, which are
# already keyword-only and already identical to each other.
DebitFn = Callable[..., bool]
RefundFn = Callable[..., Any]


class PaidWork:
    """Handle yielded into a metered block; refunds at most once."""

    def __init__(
        self,
        *,
        user_id: Optional[str],
        workspace_id: Optional[str],
        cost: int,
        operation_type: str,
        refund: RefundFn,
    ) -> None:
        self.cost = cost
        self.operation_type = operation_type
        self._user_id = user_id
        self._workspace_id = workspace_id
        self._refund_fn = refund
        self._refunded = False

    @property
    def charged(self) -> int:
        """What the caller actually kept — 0 once refunded. Put this in the response
        rather than the quoted price, so `credits_debited` never claims a charge that
        was handed back."""
        return 0 if self._refunded else self.cost

    def refund(self, reason: str = "work not delivered") -> None:
        """Hand the credit back. Safe to call more than once."""
        if self._refunded or self.cost <= 0 or not self._user_id:
            return
        self._refunded = True
        logger.info(
            "[paid-door] refunding %s credits for %s: %s",
            self.cost, self.operation_type, reason,
        )
        self._refund_fn(
            user_id=self._user_id,
            amount=self.cost,
            operation_type=self.operation_type,
            workspace_id=self._workspace_id,
        )


@asynccontextmanager
async def metered_door(
    *,
    user_id: Optional[str],
    workspace_id: Optional[str],
    cost: int,
    operation_type: str,
    debit: DebitFn,
    refund: RefundFn,
) -> AsyncIterator[PaidWork]:
    """Debit up front, refund on failure. See the module docstring."""
    if cost > 0 and not user_id:
        # `if user_id and debit(...)` reads as metering and behaves as a free pass —
        # test_paid_route_metering already pins that shape as a bug on the job-research
        # doors. A paid door with no resolvable payer refuses.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No payer could be resolved for a paid operation",
        )

    if cost > 0 and not debit(
        user_id=user_id, amount=cost,
        operation_type=operation_type, workspace_id=workspace_id,
    ):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Insufficient credits",
        )

    work = PaidWork(
        user_id=user_id, workspace_id=workspace_id,
        cost=cost, operation_type=operation_type, refund=refund,
    )
    try:
        yield work
    except Exception:
        work.refund("the operation raised")
        raise
