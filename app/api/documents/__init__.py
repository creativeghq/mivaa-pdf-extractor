"""
Document API routes module.

This module contains all document-related API endpoints, organized by functionality:
- upload_routes: Document upload and processing ✅
- query_routes: Document querying and search ✅
- management_routes: Job management, document content, AI tracking ✅
"""

from .upload_routes import router as upload_router
from .query_routes import router as query_router
from .management_routes import router as management_router

__all__ = [
    'upload_router',
    'query_router',
    'management_router',
]

