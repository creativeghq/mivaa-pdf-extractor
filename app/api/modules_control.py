"""
Module-system control endpoints.

These are NOT per-module routes (those live under each module's own
`routes.py` and mount at `/api/v1/modules/<slug>/*`). These are about the
module system itself — currently just a cache invalidation hook called by
the admin /admin/modules toggle UI to drop the in-process enabled-flag cache
without waiting for the 5-minute TTL.

Mounted at `/api/v1/modules/_invalidate` (prefix uses an underscore so it
can't ever collide with a real module slug).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status

from app.modules._core.registry import invalidate_cache
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/modules", tags=["modules:control"])


def _extract_bearer(request: Request) -> str:
    header = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if not header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )
    return header.split(" ", 1)[1].strip()


def _is_admin(sb, user_id: str) -> bool:
    try:
        res = (
            sb.table("user_profiles")
            .select("role_id")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        row = (res.data if res else None) or {}
        role_id = row.get("role_id")
        if not role_id:
            return False
        role = sb.table("roles").select("name").eq("id", role_id).maybe_single().execute()
        name = ((role.data if role else None) or {}).get("name")
        return name in ("admin", "super_admin")
    except Exception:  # noqa: BLE001
        return False


@router.post("/_invalidate", status_code=status.HTTP_204_NO_CONTENT)
async def invalidate_module_cache(request: Request) -> None:
    """Drop the in-process enabled-flag cache. Admin only."""
    token = _extract_bearer(request)
    supabase = get_supabase_client().client
    try:
        response = supabase.auth.get_user(token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {exc}",
        )

    user = getattr(response, "user", None)
    if user is None or not getattr(user, "id", None):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not resolve authenticated user.",
        )
    user_id = str(user.id)

    if not _is_admin(supabase, user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required.",
        )

    invalidate_cache()
    logger.info("modules: cache invalidated by user %s", user_id)
