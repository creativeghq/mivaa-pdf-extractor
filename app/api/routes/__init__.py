"""
RAG API Routes Package

This package contains modular route definitions for the RAG API.
Each module focuses on a specific domain (documents, jobs, search, etc.)
"""

from .data import router as data_router
from .system import router as system_router
from .jobs import router as jobs_router
from .search import router as search_router
from .documents import router as documents_router

__all__ = [
    "data_router",
    "system_router",
    "jobs_router",
    "search_router",
    "documents_router",
]

