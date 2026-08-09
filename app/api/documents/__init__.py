"""
Document API routes module.

This module contains all document-related API endpoints, organized by functionality:
- query_routes: `authorize_rag_workspace` only — no longer declares routes
- management_routes: Job management, document content, AI tracking ✅

Removed 2026-05-23: upload_routes.py — was a duplicate of rag_routes.py's
`POST /documents/upload`. FastAPI's first-registered handler wins, and
`rag_routes.py` is included first at main.py:2052, so the upload_routes
handler was unreachable. The actual upload route lives in
[rag_routes.py:539](../rag_routes.py#L539).

Removed likewise: query_routes' `POST /query`, `/chat` and `/search`, for exactly
the same reason — all three were also declared in rag_routes.py, which is included
first, so these were unreachable. They still shaped the OpenAPI (last-write-wins per
path), so the published contract for `/api/rag/search` advertised a body with no
`aspect` field while the served handler accepted one. `query_router` is gone rather
than left empty: a registered router with no routes is how the next addition here
quietly becomes a shadowing duplicate again.
"""

from .management_routes import router as management_router

__all__ = [
    'management_router',
]

