"""
Document Processing Orchestration Services

This module provides access to document processing orchestration functions.

NOTE: The actual implementation remains in app.api.rag_routes for now.
This will be fully refactored in a future iteration to move the logic
into proper service classes.
"""

from app.api.rag_routes import (
    process_document_with_discovery,
    run_async_in_background,
)

__all__ = [
    'process_document_with_discovery',
    'run_async_in_background',
]
