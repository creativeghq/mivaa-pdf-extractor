"""
Document processing orchestration module.

This module contains the core orchestration logic for document processing:
- document_processor: Main background task orchestration
- pipeline_coordinator: Stage coordination and progress tracking
"""

from .document_processor import process_document_with_discovery
from .pipeline_coordinator import PipelineCoordinator
from .utils import run_async_in_background

__all__ = [
    'process_document_with_discovery',
    'PipelineCoordinator',
    'run_async_in_background',
]

