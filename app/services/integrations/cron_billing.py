"""Cron credit metering (Python side of public.cron_charge_workspace / cron_charge_user)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def charge_cron(
    supabase_client: Any,
    cron_key: str,
    *,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    units: int = 1,
    description: Optional[str] = None,
    subject: Optional[Dict[str, Any]] = None,
) -> bool:
    """Charge one unit of a metered cron's work; return True to proceed, False to skip."""
    try:
        if workspace_id:
            res = supabase_client.rpc(
                "cron_charge_workspace",
                {
                    "p_workspace_id": str(workspace_id),
                    "p_cron_key": cron_key,
                    "p_units": int(units),
                    "p_description": description,
                    "p_subject": subject,
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
                    "p_subject": subject,
                },
            ).execute()
        else:
            # Nobody to bill. Proceed -- but say so, so an unbillable subject is a
            # visible fact rather than an invisible one.
            _record_unmetered(supabase_client, cron_key, "no_payer", subject)
            return True

        data = getattr(res, "data", None)
        row: Any = None
        if isinstance(data, list):
            row = data[0] if data else None
        elif isinstance(data, dict):
            row = data
        if not row:
            # The RPC answered with nothing. We do not know whether the payer was
            # charged, so we must not spend on their behalf.
            _record_unmetered(supabase_client, cron_key, "empty_charge_response", subject)
            return False
        return bool(row.get("allowed", True))
    except Exception as e:  # noqa: BLE001 -- a metering fault must not raise into the cron
        logger.warning("[cron-billing] %s charge failed (skipping this unit): %s", cron_key, e)
        _record_unmetered(supabase_client, cron_key, f"charge_error: {e}"[:300], subject)
        return False


def _record_unmetered(
    supabase_client: Any,
    cron_key: str,
    reason: str,
    subject: Optional[Dict[str, Any]],
) -> None:
    """Leave a durable trace that a scheduled unit was not metered.

    Best-effort and never raises: this runs on the path that already failed, and a
    logging failure must not be the thing that breaks the cron.
    """
    try:
        supabase_client.table("system_logs").insert({
            "level": "WARNING",
            "logger_name": "cron_billing",
            "message": f"cron unit not metered: {cron_key} ({reason})",
            # `context`, not `metadata` — system_logs has no metadata column, and this
            # insert is inside a bare except, so the wrong name would fail invisibly.
            "context": {"cron_key": cron_key, "reason": reason, "subject": subject},
        }).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning("[cron-billing] could not record unmetered unit for %s: %s", cron_key, e)
