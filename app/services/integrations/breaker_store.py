"""Read and record `provider_credential_health` for the breaker.

Kept apart from `provider_breaker` so the POLICY stays importable with pytest alone: everything
here needs a Supabase client, and MIVAA's CI does not install one.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.services.integrations.provider_breaker import (
    BreakerVerdict,
    classify_provider_failure,
    verdict_from_state,
)

logger = logging.getLogger(__name__)


def _client():
    from app.services.core.supabase_client import get_supabase_client

    return get_supabase_client().client


def breaker_verdict(provider: str, workspace_id: Optional[str] = None) -> BreakerVerdict:
    """Should we spend on this provider right now? An unreadable state CALLS -- see the policy."""
    try:
        res = _client().rpc(
            "provider_breaker_state",
            {"p_provider": provider, "p_workspace_id": workspace_id},
        ).execute()
        return verdict_from_state(res.data if isinstance(res.data, dict) else None)
    except Exception as e:  # noqa: BLE001 - the breaker must never break the caller
        logger.debug(f"provider_breaker_state({provider}) unreadable: {e}")
        return verdict_from_state(None)


def record_outcome(
    provider: str,
    *,
    ok: bool,
    http_status: Optional[int] = None,
    error: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> None:
    """Close the breaker on any success; count only refusals the credential causes.

    A transport error or a 500 is deliberately NOT counted: opening on those turns a provider
    outage into a self-inflicted one that outlives it.
    """
    try:
        sb = _client()
        if ok:
            sb.rpc(
                "record_provider_success",
                {"p_provider": provider, "p_workspace_id": workspace_id},
            ).execute()
            return
        kind = classify_provider_failure(http_status, error)
        if kind is None:
            return
        sb.rpc(
            "record_provider_auth_failure",
            {
                "p_provider": provider,
                "p_workspace_id": workspace_id,
                "p_http_status": http_status,
                "p_error": error,
            },
        ).execute()
        logger.warning(
            f"provider {provider}: {kind} refusal recorded (HTTP {http_status}): {error}"
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"could not record provider outcome for {provider}: {e}")
