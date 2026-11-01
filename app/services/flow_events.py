from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

HTTP_TIMEOUT_SECONDS = 15.0


def emit_flow_event(
    event_type: str,
    data: Dict[str, Any],
    supabase_url: str,
    service_role_key: str,
    client: Optional[httpx.Client] = None,
) -> bool:
    if not supabase_url or not service_role_key:
        logger.warning("flow event %s not sent: SUPABASE_URL or the service key is not configured", event_type)
        return False
    headers = {
        "Authorization": f"Bearer {service_role_key}",
        "apikey": service_role_key,
        "Content-Type": "application/json",
    }
    body = {"action": "trigger-event", "event_type": event_type, "data": data}
    try:
        if client is not None:
            response = client.post(f"{supabase_url}/functions/v1/flow-engine", headers=headers, json=body)
        else:
            with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as http:
                response = http.post(f"{supabase_url}/functions/v1/flow-engine", headers=headers, json=body)
        if response.status_code >= 400:
            logger.warning("flow event %s rejected: HTTP %s %s", event_type, response.status_code, response.text[:200])
            return False
        return True
    except Exception as exc:
        logger.warning("flow event %s failed: %s", event_type, exc)
        return False
