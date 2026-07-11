"""Cron credit metering (Python side of public.cron_charge_workspace / cron_charge_user).

A workspace-scoped cron's per-subject loop calls charge_cron() once per UNIT OF WORK (one
price/mention/job refresh, one digest, one probe). It charges the workspace owner (pool -> personal
via debit_credits) or, when the subject has no workspace, the user's personal balance. Returns:
  - True  -> proceed with the work.
  - False -> skip this subject (payer out of credits). The next cron tick re-charges and
             auto-resumes the moment the owner tops up.

Fails OPEN on any error (missing payer, RPC failure) so metering can NEVER block or mis-charge
scheduled work -- worst case it silently no-ops.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def charge_cron(
    supabase_client: Any,
    cron_key: str,
    *,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    units: int = 1,
    description: Optional[str] = None,
) -> bool:
    """Charge one unit of a metered cron's work; return True to proceed, False to skip.

    Pass the RAW supabase client (e.g. ``service.supabase.client``). Workspace subjects bill the
    workspace owner; workspace-less subjects bill ``user_id``. Fails OPEN on any error.
    """
    try:
        if workspace_id:
            res = supabase_client.rpc(
                "cron_charge_workspace",
                {
                    "p_workspace_id": str(workspace_id),
                    "p_cron_key": cron_key,
                    "p_units": int(units),
                    "p_description": description,
                },
            ).execute()
        elif user_id:
            res = supabase_client.rpc(
                "cron_charge_user",
                {
                    "p_user_id": str(user_id),
                    "p_cron_key": cron_key,
                    "p_units": int(units),
                    "p_description": description,
                },
            ).execute()
        else:
            return True  # no payer to bill -> free pass

        data = getattr(res, "data", None)
        row: Any = None
        if isinstance(data, list):
            row = data[0] if data else None
        elif isinstance(data, dict):
            row = data
        if not row:
            return True  # nothing returned -> fail open
        return bool(row.get("allowed", True))
    except Exception as e:  # noqa: BLE001 -- metering must never break the cron
        logger.warning("[cron-billing] %s charge failed (failing open): %s", cron_key, e)
        return True
