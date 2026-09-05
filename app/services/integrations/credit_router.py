"""The one credit debit/refund. Every paid door in this service routes through here.

WHY IT IS ONE FILE
------------------
It was three, in `mention_cost_logger`, `price_cost_logger` and `job_cost_logger` — the
same twenty lines copied, differing only in a log prefix. Audit #30 M16-1 was found in
one of them and had to be fixed in all three, and the same audit had already recorded
why: identical contracts in separate files are exactly how one of them drifts. Two of
them had ALSO drifted once before, spelling the zero-amount rule differently
(`if amount <= 0: return True` vs `return amount <= 0`) for the same behaviour.

A fourth copy was about to be written for the AI-services doors (#23 M10-4). That is the
moment to stop copying, so the three now delegate here and keep their names — callers
are unchanged, and there is one place where the rule lives.

WHAT THE RULES ARE
------------------
* A NON-POSITIVE amount is refused, loudly. It used to return True — success, nothing
  debited — which is the shape audit #217 H3 found and fixed in the same files: a
  billing helper reporting success without charging. Work that is genuinely free must
  not call the debit path at all.
* NO PAYER is a refusal, never a free pass. `if user_id and debit(...)` reads as
  metering and behaves as a free pass whenever the identity is missing.
* The RPC returns `[{success: bool, ...}]` and an insufficient balance is a NON-EMPTY,
  truthy row. `bool(data)` would read that as a successful debit — a paid operation
  served free (audit #217 H3). The flag is read explicitly.
* A refund of nothing is a no-op, not an error: unlike a debit, a zero refund is a
  legitimate state (a door that refunds unconditionally after a free operation).

TWO ENTRY POINTS, ON PURPOSE
----------------------------
`debit_credits` answers yes/no and is what most doors want. `debit_row` returns the
RPC's whole row, because some doors legitimately need more than a boolean: one raises
402 carrying the RPC's own `error_message`, another reports the caller's `new_balance`.
Both go through the SAME call and the same rules — what differs is only what the caller
does with the answer, which belongs at the call site.

The sweep in `test_audit_30_gates_hold` derives its scope from the tree, so it found
FOUR more copies of this RPC call than the audit had (seven in total, not three). They
also passed `p_description` and `p_metadata`, which is why those are parameters here
rather than being dropped in the move.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.services.core.supabase_client import get_supabase_client
from app.utils.retry_helper import describe_exception, should_retry_exception

logger = logging.getLogger(__name__)


def debit_row(
    *,
    user_id: str,
    amount: int,
    operation_type: str,
    workspace_id: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[dict] = None,
    module: str = "credit-router",
) -> dict:
    """The debit, returning the RPC's row: `{success, new_balance?, error_message?}`.

    A refusal is returned as a row too (`success=False` with a reason), never raised, so
    every caller decides its own response — 402, skip-the-unit, or report the balance.
    """
    if amount <= 0:
        logger.error(
            "%s: refusing a non-positive debit of %s for %s — a free operation must "
            "not go through the debit path (#30 M16-1)",
            module, amount, operation_type,
        )
        return {"success": False, "error_message": "non-positive amount"}
    if not user_id:
        logger.error(
            "%s: refusing a debit of %s for %s with no payer — a missing identity is a "
            "refusal, not a free pass",
            module, amount, operation_type,
        )
        return {"success": False, "error_message": "no payer"}

    payload = {
        "p_user_id": user_id,
        "p_amount": amount,
        "p_operation_type": operation_type,
        "p_workspace_id": workspace_id,
    }
    if description is not None:
        payload["p_description"] = description
    if metadata is not None:
        payload["p_metadata"] = metadata

    try:
        result = get_supabase_client().client.rpc("debit_credits", payload).execute()
        data = getattr(result, "data", None)
        if not data:
            return {"success": False, "error_message": "no_response"}
        row = data[0] if isinstance(data, list) else data
        if not isinstance(row, dict):
            return {"success": bool(row)}
        return row
    except Exception as e:
        # A raised RPC is never "insufficient". `debit_credits` answers an empty balance
        # with a ROW (`success=false`, handled above); what lands here is the transport —
        # "Server disconnected" mid-POST on the first tick of the rank tracker (MIVAA-5KJ),
        # a Cloudflare 52x, a timeout. The retry patch cannot repeat a debit (a second
        # attempt could charge twice), so for a transient fault the outcome is UNKNOWN:
        # the credits may or may not have been taken, and the paid call they were for did
        # not happen. Logging that as "likely insufficient" at INFO named the one cause it
        # could not be, at a level nobody reads. The caller still gets a refusal — the
        # conservative answer — but the reason now says which of the two facts this was.
        if should_retry_exception(e):
            logger.warning(
                "%s: credit debit outcome UNKNOWN for %s (%s credits) — transport failure "
                "on a write that must not be repeated; treated as refused: %s",
                module, operation_type, amount, describe_exception(e),
            )
            return {
                "success": False,
                "error_message": f"debit_unknown: {describe_exception(e)}"[:200],
            }
        logger.warning(
            "%s: credit debit RAISED for %s (%s credits); treated as refused: %s",
            module, operation_type, amount, describe_exception(e),
        )
        return {"success": False, "error_message": describe_exception(e)[:200]}


def debit_credits(
    *,
    user_id: str,
    amount: int,
    operation_type: str,
    workspace_id: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[dict] = None,
    module: str = "credit-router",
) -> bool:
    """Atomic credit debit via the shared credit router.

    Routes to the workspace pool when funded (workspace_id set + pool exists), else the
    personal wallet. Partner (api_key) callers pass no workspace_id → personal.

    Returns True on success, False on insufficient balance / failure / refusal.
    `module` only labels the logs, so a caller can still tell which door failed.
    """
    row = debit_row(
        user_id=user_id, amount=amount, operation_type=operation_type,
        workspace_id=workspace_id, description=description, metadata=metadata,
        module=module,
    )
    # The RPC returns `[{success: bool, ...}]` and an insufficient balance is a
    # NON-EMPTY, truthy row — `bool(data)` would read that as a successful debit.
    return bool(row.get("success"))


def refund_credits(
    *,
    user_id: str,
    amount: int,
    operation_type: str,
    workspace_id: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[dict] = None,
    suffix: str = ".refund",
    module: str = "credit-router",
) -> None:
    """Best-effort refund via the shared credit router. Never raises.

    `suffix` exists because the callers do not agree on how a refund names itself —
    most append `.refund`, `public_tools` uses `_refund`. Preserved rather than
    normalised: those strings are already in `credit_transactions` and renaming them
    would split each operation's history in two.
    """
    if amount <= 0 or not user_id:
        return
    payload = {
        "p_user_id": user_id,
        "p_amount": amount,
        "p_operation_type": f"{operation_type}{suffix}",
        "p_workspace_id": workspace_id,
    }
    if description is not None:
        payload["p_description"] = description
    if metadata is not None:
        payload["p_metadata"] = metadata
    try:
        get_supabase_client().client.rpc("refund_credits", payload).execute()
    except Exception as e:
        logger.info("%s: refund failed (non-fatal): %s", module, e)
