"""
Document processing orchestration module.
"""

from app.api.rag_routes import process_document_with_discovery
from .utils import run_async_in_background

__all__ = [
    'process_document_with_discovery',
    'run_async_in_background',
]

