"""Shared authorization helper for the RAG query surface."""

import logging
from typing import Any, Dict

from fastapi import HTTPException, status

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


async def authorize_rag_workspace(claims: Dict[str, Any], workspace_id: str) -> None:
    """Authorize an authenticated caller for a body-supplied `workspace_id`."""
    if claims.get("service") == "mivaa" or claims.get("is_test_user"):
        return

    user_id = claims.get("sub") or claims.get("user_id")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Missing user identity")

    try:
        client = get_supabase_client()
        resp = (
            client.client.table("workspace_members")
            .select("status")
            .eq("user_id", user_id)
            .eq("workspace_id", workspace_id)
            .eq("status", "active")
            .execute()
        )
        is_member = bool(resp.data)
    except Exception as e:
        logger.error(f"Workspace authorization check failed: {e}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Workspace authorization failed")

    if not is_member:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Not authorized for workspace {workspace_id}",
        )
