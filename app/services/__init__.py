"""
Services Package for MIVAA PDF Extractor

This package contains all service classes and modules for the PDF processing application,
including Supabase integration, LlamaIndex RAG services, PDF processing, and external API integrations.
"""

# Import key services for convenience
from .supabase_client import SupabaseClient, get_supabase_client, initialize_supabase
from .pdf_processor import PDFProcessor
from .llamaindex_service import LlamaIndexService
from .ai_client_service import AIClientService, get_ai_client_service

__all__ = [
    "SupabaseClient",
    "get_supabase_client",
    "initialize_supabase",
    "PDFProcessor",
    "LlamaIndexService",
    "AIClientService",
    "get_ai_client_service"
]
