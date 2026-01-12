"""
Document Processing Orchestration

This module provides a clean import point for document processing orchestration functions.

The actual implementations are in app.api.rag_routes for now, but this module
provides a stable API for importing these functions from other parts of the application.

This will be fully refactored in a future iteration to move the logic into proper
service classes under app.services.orchestration.
"""

# Re-export orchestration functions from rag_routes
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

