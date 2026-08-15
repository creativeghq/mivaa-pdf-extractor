"""
Document API routes module.

This package declares NO routes. It exports `authorize_rag_workspace` (from
query_routes) and nothing else.

Removed 2026-05-23: upload_routes.py — was a duplicate of rag_routes.py's
`POST /documents/upload`. FastAPI's first-registered handler wins, and
`rag_routes.py` is included first at main.py:2041, so the upload_routes
handler was unreachable. The actual upload route lives in
[rag_routes.py:554](../rag_routes.py#L554).

Removed likewise: query_routes' `POST /query`, `/chat` and `/search`, for exactly
the same reason — all three were also declared in rag_routes.py, which is included
first, so these were unreachable. They still shaped the OpenAPI (last-write-wins per
path), so the published contract for `/api/rag/search` advertised a body with no
`aspect` field while the served handler accepted one. `query_router` is gone rather
than left empty: a registered router with no routes is how the next addition here
quietly becomes a shadowing duplicate again.

Removed 2026-08-15 (issue #15, MV2-11): management_routes.py — the third instance,
and the one the paragraph above predicted. It declared a bare `APIRouter()` mounted
at `prefix="/api/rag"` (main.py) — the SAME final namespace as rag_routes, included
second, so all 10 of its handlers were unreachable. Every one of those 10 paths is
declared in rag_routes.py, gated: nine by a route decorator dependency
(`verify_internal_access` / `require_rag_resource_access`), and
`POST /documents/job/{job_id}/resume` by an explicit in-body JWT + workspace-ownership
check. The management_routes copies had NO gate of any kind — no decorator dependency,
no signature dependency, no resolve_workspace_id — including
`DELETE /documents/jobs/{job_id}` and `POST /jobs/{job_id}/restart`.

Two things kept it dangerous while it was unreachable:

  1. The published OpenAPI contract for all 10 paths was generated from the UNGATED
     definitions — last-write-wins per path, and management_routes was registered
     second. That is the documented, already-experienced bug from the paragraph above.
  2. It was loaded. Reordering the includes, adding a path here that rag_routes lacks,
     or removing one from rag_routes would have put an ungated destructive handler live
     instantly — with no diff to either file to explain why.

Do not re-add a router in this package. Routes under /api/rag belong in rag_routes.py,
where the gates are.
"""

__all__ = []
