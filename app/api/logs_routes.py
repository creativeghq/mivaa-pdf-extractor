"""Logs API Routes"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, field_validator

from app.services.core.supabase_client import get_supabase_client
from app.schemas.api_responses import StatusResponse

logger = logging.getLogger(__name__)

#: Bounds on the one unauthenticated write in this repository. Sizes are generous
#: enough for a real stack trace and small enough that the route is not a write
#: amplifier into a TTL'd table.
MAX_MESSAGE_CHARS = 4_000
MAX_LOGGER_NAME_CHARS = 200
MAX_URL_CHARS = 2_000
MAX_USER_AGENT_CHARS = 500
MAX_CONTEXT_BYTES = 16_000


router = APIRouter(prefix="/api/admin/logs", tags=["Admin", "Logs"])


class FrontendLogRequest(BaseModel):
    """Request model for frontend log submission.

    There is deliberately no `user_id` field. Attribution is derived from the bearer
    token when the caller presents one — a body-supplied id is a claim, not a fact,
    and this route is open.
    """
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    message: str = Field(..., max_length=MAX_MESSAGE_CHARS)
    logger_name: str = Field("frontend", max_length=MAX_LOGGER_NAME_CHARS)
    context: Optional[dict] = None
    url: Optional[str] = Field(None, max_length=MAX_URL_CHARS)
    user_agent: Optional[str] = Field(None, max_length=MAX_USER_AGENT_CHARS)

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, v: Any) -> Any:
        """Accept the frontend's casing without widening what is accepted."""
        return v.upper() if isinstance(v, str) else v

    @field_validator("context")
    @classmethod
    def _bound_context(cls, v: Optional[dict]) -> Optional[dict]:
        """`context` is a free-form dict, so its SIZE is the only thing that can be
        bounded — and it has to be, or the field is the amplifier the other caps
        removed."""
        if v is None:
            return v
        import json
        try:
            encoded = json.dumps(v, default=str)
        except (TypeError, ValueError):
            raise ValueError("context is not JSON-serialisable")
        if len(encoded.encode("utf-8")) > MAX_CONTEXT_BYTES:
            raise ValueError(f"context exceeds {MAX_CONTEXT_BYTES} bytes")
        return v


async def _attributed_user_id(request: Request) -> Optional[str]:
    """The caller's user id IF they presented a valid token, else None.

    Best-effort by design: this route stays open (see the module docstring), so an
    anonymous frontend error is a legitimate outcome and must record no user rather
    than an asserted one.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    try:
        from app.dependencies import _get_jwt_middleware
        claims = await _get_jwt_middleware()._validate_token(auth.split(" ", 1)[1])
    except Exception:
        return None
    if not claims:
        return None
    uid = claims.get("sub") or claims.get("user_id")
    return str(uid) if uid else None


@router.post("/frontend", responses={200: {"model": StatusResponse}})
async def log_frontend_error(log_request: FrontendLogRequest, request: Request):
    """
    Log a frontend error to the database.

    This endpoint allows the frontend to send errors to the same logging
    system as the backend, enabling unified error tracking.

    The logs are tagged with source='frontend' to distinguish them from backend logs.
    """
    # system_logs holds WARNING and above only; every reader filters to that floor.
    if log_request.level in ("DEBUG", "INFO"):
        return {"success": True, "message": "Below WARNING; not recorded"}
    try:
        supabase = get_supabase_client()
        user_id = await _attributed_user_id(request)

        # Prepare log entry
        log_entry: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "level": log_request.level.upper(),
            "logger_name": log_request.logger_name,
            "message": log_request.message,
            "context": {
                **(log_request.context or {}),
                "source": "frontend",  # Tag as frontend
                "url": log_request.url,
                "user_agent": log_request.user_agent or request.headers.get("user-agent"),
                "ip_address": request.client.host if request.client else None,
            },
            # From the token or nothing. Never from the body (M11-4).
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat()
        }

        # Insert into database
        supabase.client.table('system_logs').insert(log_entry).execute()

        return {
            "success": True,
            "log_id": log_entry["id"],
            "message": "Frontend log recorded successfully"
        }

    except Exception as e:
        # Don't fail the frontend if logging fails. `logger`, not `print`: a print
        # reaches no sink anyone can query, so the failure would be invisible.
        logger.warning("Failed to record frontend log: %s", e)
        return {
            "success": False,
            "error": str(e)
        }
