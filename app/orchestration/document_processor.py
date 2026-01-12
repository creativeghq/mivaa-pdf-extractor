"""
Document Processing Orchestration

This module contains the main orchestration logic for document processing.
Currently imports from rag_routes.py for backward compatibility.

TODO: Extract the full process_document_with_discovery function here
"""

# Temporary import from original location for backward compatibility
from app.api.rag_routes import process_document_with_discovery

__all__ = ['process_document_with_discovery']

