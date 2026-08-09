"""
Shared authorization helper for the RAG query surface.

THIS MODULE NO LONGER DECLARES ROUTES.

It used to declare `POST /query`, `POST /chat` and `POST /search`, and it was mounted
at the same `/api/rag` prefix as `rag_routes.py`, which declares all three as well.
Starlette serves the FIRST matching route and main.py includes `rag_router` before
this one, so every request went to `rag_routes.py` and the copies here were
unreachable. They were not harmless dead code:

  * OpenAPI generation is last-write-wins per path, so the PUBLISHED contract for
    `/api/rag/search` was this module's `SearchRequest` — which has no `aspect` field.
    The aspect-aware search shipped in #277 was therefore invisible to anyone reading
    the docs, while working perfectly in the served handler.
  * Reordering two `include_router` lines in main.py — a change with no visible
    connection to search — would have silently swapped the served implementation for
    one that drops `aspect` on the floor, turning the feature off with no edit to
    either file and nothing raising.

This is the third time this exact shape has bitten this repo: `upload_routes.py` was
deleted for it on 2026-05-23 (see the note in this package's `__init__.py`), and
`/search/knowledge-base` for it again, which is why
tests/unit/test_kb_chunk_retrieval.py asserts single declaration. That guard now
covers `/query`, `/chat` and `/search` too, so the ordering can never decide behaviour
again.

`authorize_rag_workspace` stays because four modules import it (rag_routes,
anthropic_routes, data_import_routes, internal_routes). The router object is gone
deliberately: an empty-but-registered router is an invitation for the next route added
here to become a shadowing duplicate all over again.
"""

import logging
from typing import Any, Dict

from fastapi import HTTPException, status

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


async def authorize_rag_workspace(claims: Dict[str, Any], workspace_id: str) -> None:
    """
    Authorize an authenticated caller for a body-supplied `workspace_id`.

    MIVAA is internet-reachable and the `/api/rag` prefix is excluded from the JWT
    middleware (routes self-guard). Before this gate, /query, /chat and /search had
    NO auth dependency and trusted the body `workspace_id` → any internet caller could
    read any tenant's documents/chunks (audit C4). Two trusted caller shapes:

    - Material Kai platform service (the edge mivaa-gateway / internal callers) —
      `claims['service']=='mivaa'`. The gateway has already authenticated the end user
      and debited credits, so the body `workspace_id` is trusted as-is. Test users
      (dev/test envs) are likewise trusted.
    - A real Supabase user JWT — must be an ACTIVE member of `workspace_id`, else 403.

    `get_current_user` (HTTPBearer auto_error) already rejects callers with no/invalid
    bearer with 401, which is what closes the unauthenticated-access hole.
    """
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
