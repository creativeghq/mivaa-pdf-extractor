"""One implementation of "stamp the cost rollup and say whether it happened".

WHAT IT REPLACES
----------------
`stamp_refresh_cost` and `recompute_lifetime_cost` existed twice — once in
`job_cost_logger`, once in `mention_cost_logger` — structurally identical apart from the
id column and the RPC name. This session has already unified a debit rule that had seven
copies (#30 M16-1), a PDF bounds check that had three (#35 M20-2) and a forced tool call
that had four (#32). Same argument, made before the drift rather than after it.

WHY IT MATTERS MORE HERE THAN THE COPY COUNT SUGGESTS
-----------------------------------------------------
`stamp_job_refresh_cost` is the platform's canonical example of the silent-zero defect.
It referenced a column that did not exist, the exception was swallowed at WARNING, and
per-subject billing sat at zero for months while every health signal stayed green. The
copy was fixed; the SHAPE that hid it was not:

  * a bare `return` when an id is missing — a caller that has just written an
    `ai_usage_logs` row and cannot name the run it belongs to is a bug, and it looked
    identical to "nothing to do"
  * `except Exception: logger.warning(...)` and return None — so no caller could
    distinguish a rollup that ran from one that did not
  * a `None` return type, which meant no caller could react even if it wanted to

The rollup is what makes spend VISIBLE. When it stops, the numbers do not go wrong —
they stop moving, which looks exactly like a quiet month.

WHAT THIS DOES DIFFERENTLY
--------------------------
Returns a bool. Logs at ERROR, not WARNING — a lost rollup is not a warning, it is spend
that has already happened and is now unattributed, and the severity is what decides
whether it survives the DB log sink's noise filter. A missing id is reported as the
defect it is rather than treated as a no-op.

It still does not RAISE. The caller has already made and been billed for the provider
call; a bookkeeping failure must not destroy the result it is bookkeeping for.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def stamp_rollup(
    *,
    rpc: str,
    params: Dict[str, Any],
    module: str,
    required: Optional[Dict[str, Any]] = None,
) -> bool:
    """Call a cost-rollup RPC. True if it ran, False if it did not.

    `required` names the ids the rollup cannot work without, so a missing one is
    reported with the NAME of what was missing instead of a silent early return.
    """
    missing = [k for k, v in (required or {}).items() if not v]
    if missing:
        # Not a no-op. Reaching a rollup helper without the id of the thing being rolled
        # up means the caller has spend it cannot attribute, and that is worth a line in
        # the log rather than a silent return.
        logger.error(
            f"{module}: {rpc} skipped — missing {missing}. Spend for this subject will "
            "not appear in its rollup."
        )
        return False

    try:
        from app.services.core.supabase_client import get_supabase_client

        get_supabase_client().client.rpc(rpc, params).execute()
        return True
    except Exception as e:
        # ERROR, not WARNING. The provider call already happened and was already
        # billed; what failed is the step that makes it visible. Sub-WARNING records
        # from noisy libraries are dropped by the DB log sink, and this must never be.
        logger.error(f"{module}: {rpc} failed — rollup not updated: {e}")
        return False
