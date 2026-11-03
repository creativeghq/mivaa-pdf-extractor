"""Who pays for document work no signed-in person started: the uploader, else the workspace's owner, else its admin."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

PAYER_ROLES = ("owner", "admin")


def billing_principal(db: Any, workspace_id: Optional[str], document_id: Optional[str] = None) -> Optional[str]:
    """The document's `created_by`, else the longest-standing active owner, else active admin, else None."""
    db = getattr(db, "client", db)
    if document_id:
        rows = db.table("documents").select("created_by, workspace_id").eq("id", document_id).limit(1).execute().data or []
        if rows and rows[0].get("created_by"):
            return str(rows[0]["created_by"])
        workspace_id = workspace_id or (rows[0].get("workspace_id") if rows else None)
    if not workspace_id:
        return None
    for role in PAYER_ROLES:
        rows = (db.table("workspace_members").select("user_id").eq("workspace_id", workspace_id)
                .eq("status", "active").eq("role", role).order("joined_at").limit(1).execute().data or [])
        if rows:
            return str(rows[0]["user_id"])
    logger.warning("no billing principal for workspace %s (document %s)", workspace_id, document_id)
    return None
