"""
Document Processing Orchestration Services

This module provides access to document processing orchestration functions.

NOTE: The actual implementation remains in app.api.rag_routes for now.
This will be fully refactored in a future iteration to move the logic
into proper service classes.

For now, this module serves as a clean import point for orchestration functions.
"""

# Import orchestration functions from rag_routes
# These are complex 500+ line functions that will be refactored later
from app.api.rag_routes import (
    process_document_background,
    process_document_with_discovery,
    run_async_in_background,
)

__all__ = [
    'process_document_background',
    'process_document_with_discovery',
    'run_async_in_background',
]

