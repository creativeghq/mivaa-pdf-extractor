"""
Services Package for MIVAA PDF Extractor

This package contains all service classes and modules for the PDF processing application,
including Supabase integration, LlamaIndex RAG services, PDF processing, and external API integrations.
"""

# Import key services for convenience
from .supabase_client import SupabaseClient, get_supabase_client, initialize_supabase
from .pdf_processor import PDFProcessor
from .llamaindex_service import LlamaIndexService

__all__ = [
    "SupabaseClient",
    "get_supabase_client", 
    "initialize_supabase",
    "PDFProcessor",
    "LlamaIndexService"
]
