"""
Services Package for MIVAA PDF Extractor

This package contains all service classes and modules for the PDF processing application,
including Supabase integration, RAG services (Claude 4.5 + Direct Vector DB), PDF processing, and external API integrations.
"""

# Import key services for convenience
from .core.supabase_client import SupabaseClient, get_supabase_client, initialize_supabase
from .core.ai_client_service import AIClientService, get_ai_client_service

__all__ = [
    "SupabaseClient",
    "get_supabase_client",
    "initialize_supabase",
    "AIClientService",
    "get_ai_client_service"
]

