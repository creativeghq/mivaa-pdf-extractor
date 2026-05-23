"""
Cloudflare Turnstile verifier.

Server-side verification of the cf-turnstile-response token returned by the
widget on the /tools page. Without this, anyone could POST directly to the
public scan endpoints and bypass the captcha.

Docs: https://developers.cloudflare.com/turnstile/get-started/server-side-validation/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from app.services.integrations.platform_secret_resolver import resolve_secret

logger = logging.getLogger(__name__)

_SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


@dataclass
class TurnstileVerdict:
    success: bool
    error_codes: list[str]
    challenge_ts: Optional[str] = None
    hostname: Optional[str] = None
    action: Optional[str] = None


async def verify_token(
    token: str,
    *,
    remote_ip: Optional[str] = None,
    expected_action: Optional[str] = None,
) -> TurnstileVerdict:
    """Verify a Turnstile token against Cloudflare's siteverify endpoint.

    Returns success=False with a 'configuration_error' code if the secret key
    isn't set — the caller should fail closed (treat as captcha failure).
    """
    secret = resolve_secret("TURNSTILE_SECRET_KEY").value
    if not secret:
        logger.warning("TURNSTILE_SECRET_KEY not configured — failing closed.")
        return TurnstileVerdict(success=False, error_codes=["configuration_error"])

    if not token or not token.strip():
        return TurnstileVerdict(success=False, error_codes=["missing-input-response"])

    data = {"secret": secret, "response": token}
    if remote_ip:
        data["remoteip"] = remote_ip

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(_SITEVERIFY_URL, data=data)
            payload = resp.json()
    except Exception as e:
        logger.warning(f"Turnstile siteverify call failed: {e}")
        return TurnstileVerdict(success=False, error_codes=["network_error"])

    success = bool(payload.get("success"))
    error_codes = list(payload.get("error-codes") or [])
    action = payload.get("action")

    if success and expected_action and action != expected_action:
        return TurnstileVerdict(
            success=False,
            error_codes=["action_mismatch"],
            challenge_ts=payload.get("challenge_ts"),
            hostname=payload.get("hostname"),
            action=action,
        )

    return TurnstileVerdict(
        success=success,
        error_codes=error_codes,
        challenge_ts=payload.get("challenge_ts"),
        hostname=payload.get("hostname"),
        action=action,
    )
