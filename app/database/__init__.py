"""
Database package for MIVAA PDF Extractor.

This package provides database connectivity, health checks, and connection management
for the PDF processing microservice.
"""

from .connection import (
    get_database_health,
    check_supabase_health,
    close_database_connections,
    test_database_performance,
    get_connection_manager,
    DatabaseConnectionManager
)

# Import Supabase client functions for convenience
from ..services.supabase_client import get_supabase_client, initialize_supabase

__all__ = [
    "get_database_health",
    "check_supabase_health",
    "close_database_connections",
    "test_database_performance",
    "get_connection_manager",
    "DatabaseConnectionManager",
    "get_supabase_client",
    "initialize_supabase"
]
