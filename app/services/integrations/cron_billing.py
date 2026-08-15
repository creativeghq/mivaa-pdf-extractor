"""Cron credit metering (Python side of public.cron_charge_workspace / cron_charge_user).

A workspace-scoped cron's per-subject loop calls charge_cron() once per UNIT OF WORK (one
price/mention/job refresh, one digest, one probe). It charges the workspace owner (pool -> personal
via debit_credits) or, when the subject has no workspace, the user's personal balance. Returns:
  - True  -> proceed with the work.
  - False -> skip this subject (payer out of credits). The next cron tick re-charges and
             auto-resumes the moment the owner tops up.

FAILURE MODES ARE NOT THE SAME THING, and this used to treat them as one (audit #18 M5-4).

  - No payer at all (subject has neither workspace nor user): proceeds. There is nobody to
    bill; skipping would just stop the work forever. Recorded, not silent.
  - The charge RPC failed, or returned nothing: SKIPS. Invariant 10 is explicit -- "on debit
    failure, do not perform the work" -- and a cron is the highest-volume caller of these
    endpoints, so failing open converts a billing outage into unbounded free provider spend
    across every scheduled run. The next tick re-charges and auto-resumes, exactly as it does
    for an out-of-credit payer, so a transient outage costs a delay and not a bill.

Both non-charging outcomes leave a durable breadcrumb in `system_logs`, because "we could not
meter this run" that exists only in a container log is indistinguishable from "we metered it"
by the time anyone looks (#17 M4-2).
"""
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
    """Charge one unit of a metered cron's work; return True to proceed, False to skip.

    Pass the RAW supabase client (e.g. ``service.supabase.client``). Workspace subjects bill the
    workspace owner; workspace-less subjects bill ``user_id``. Fails CLOSED when the charge
    itself fails -- see the module docstring for why that is not the same as having no payer.

    ``subject`` is merged into ``credit_transactions.metadata`` so the charge can be attributed
    back to what it paid for -- e.g. ``{"tracked_job_id": "..."}``. Without it the ledger recorded
    only ``{cron_key, workspace_id, units}``, so per-subject credit totals were unfillable and
    ``tracked_jobs.total_partner_credits_debited`` read 0 forever while real credits were being
    spent (audit #305 finding 6).
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
