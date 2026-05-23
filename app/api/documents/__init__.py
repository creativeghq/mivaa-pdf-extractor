"""
Document API routes module.

This module contains all document-related API endpoints, organized by functionality:
- query_routes: Document querying and search ✅
- management_routes: Job management, document content, AI tracking ✅

Removed 2026-05-23: upload_routes.py — was a duplicate of rag_routes.py's
`POST /documents/upload`. FastAPI's first-registered handler wins, and
`rag_routes.py` is included first at main.py:2052, so the upload_routes
handler was unreachable. The actual upload route lives in
[rag_routes.py:539](../rag_routes.py#L539).
"""

from .query_routes import router as query_router
from .management_routes import router as management_router

__all__ = [
    'query_router',
    'management_router',
]

