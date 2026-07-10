"""Credit metering for paid MIVAA AI operations (pentest #250 H1).

Endpoints that call a paid upstream (Replicate / Claude vision) must debit credits
BEFORE the call (CLAUDE.md invariant #10) and refund if it fails. Prices are
admin-editable in the `ai_model_pricing` table (model_key == service_name); the
Python fallback lives in app/config/ai_pricing.py `EXTERNAL_SERVICE_PRICING`.

Policy:
  - insufficient balance  -> HTTPException(402) (do NOT do the work)
  - billing-infra error   -> log + proceed (don't block real work on a credits outage)
  - upstream failure       -> refund what was debited
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def _user_id(user: Optional[Dict[str, Any]]) -> Optional[str]:
    if isinstance(user, dict):
        return user.get("sub") or user.get("user_id")
    return None


def _workspace_id(user: Optional[Dict[str, Any]]) -> Optional[str]:
    """Active workspace from the auth context, if the token carried one. When absent the
    credit router falls back to the personal wallet — so this is safe to pass through."""
    if isinstance(user, dict):
        return user.get("workspace_id") or user.get("active_workspace_id")
    return None


async def meter_operation(user: Optional[Dict[str, Any]], service_name: str, operation_type: str) -> float:
    """Debit credits up-front. Returns credits debited (0.0 if none). 402 on insufficient."""
    uid = _user_id(user)
    if not uid:
        return 0.0
    try:
        from app.services.integrations.credits_integration_service import get_credits_service
        res = await get_credits_service().debit_credits_for_external_service(
            user_id=uid, workspace_id=_workspace_id(user), operation_type=operation_type,
            service_name=service_name, units=1, metadata={"endpoint": operation_type},
        )
        if not res.get("success"):
            err = str(res.get("error") or "").lower()
            if "insufficient" in err or res.get("credits_required"):
                raise HTTPException(status_code=402, detail="Insufficient credits")
            logger.warning("metering non-fatal for %s: %s", service_name, res.get("error"))
            return 0.0
        return float(res.get("credits_debited") or 0.0)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("metering skipped (infra) for %s: %s", service_name, e)
        return 0.0


def refund_operation(user: Optional[Dict[str, Any]], credits: float, operation_type: str) -> None:
    """Best-effort refund of a prior debit when the paid upstream produced no result."""
    uid = _user_id(user)
    if not uid or not credits:
        return
    try:
        from app.services.core.supabase_client import get_supabase_client
        get_supabase_client().client.rpc("refund_credits", {
            "p_user_id": uid, "p_amount": credits,
            "p_operation_type": f"{operation_type}_refund",
            "p_description": f"Refund: {operation_type} produced no result",
            "p_metadata": {"refund": True},
            "p_workspace_id": _workspace_id(user),
        }).execute()
    except Exception as e:
        logger.warning("refund failed for %s: %s", operation_type, e)
