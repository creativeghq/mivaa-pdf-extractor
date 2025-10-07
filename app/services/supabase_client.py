"""
Supabase client initialization and configuration.

This module provides a centralized way to initialize and manage the Supabase client
for database operations and storage management.
"""

import logging
from typing import Optional
from supabase import create_client, Client
from app.config import Settings

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Singleton class for managing Supabase client instance."""
    
    _instance: Optional['SupabaseClient'] = None
    _client: Optional[Client] = None
    
    def __new__(cls) -> 'SupabaseClient':
        """Ensure only one instance of SupabaseClient exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the SupabaseClient (called only once due to singleton pattern)."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._settings: Optional[Settings] = None
    
    def initialize(self, settings: Settings) -> None:
        """
        Initialize the Supabase client with configuration settings.
        
        Args:
            settings: Application settings containing Supabase configuration
            
        Raises:
            ValueError: If required Supabase settings are missing
            Exception: If client initialization fails
        """
        try:
            self._settings = settings
            
            # Validate required settings
            if not settings.supabase_url:
                raise ValueError("SUPABASE_URL is required but not provided")
            
            if not settings.supabase_anon_key:
                raise ValueError("SUPABASE_ANON_KEY is required but not provided")
            
            # Create Supabase client
            # Use service role key if available, otherwise use anon key
            supabase_key = settings.supabase_service_role_key or settings.supabase_anon_key
            self._client = create_client(
                supabase_url=settings.supabase_url,
                supabase_key=supabase_key
            )
            
            logger.info("Supabase client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise
    
    @property
    def client(self) -> Client:
        """
        Get the Supabase client instance.
        
        Returns:
            Supabase client instance
            
        Raises:
            RuntimeError: If client is not initialized
        """
        if self._client is None:
            raise RuntimeError(
                "Supabase client not initialized. Call initialize() first."
            )
        return self._client
    
    @property
    def settings(self) -> Settings:
        """
        Get the application settings.
        
        Returns:
            Application settings instance
            
        Raises:
            RuntimeError: If settings are not available
        """
        if self._settings is None:
            raise RuntimeError(
                "Settings not available. Call initialize() first."
            )
        return self._settings
    
    def health_check(self) -> bool:
        """
        Perform a health check on the Supabase connection.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            # Simple query to test connection
            response = self._client.table('processed_documents').select('id').limit(1).execute()
            return True
        except Exception as e:
            logger.warning(f"Supabase health check failed: {str(e)}")
            return False

    async def list_documents(self, limit: int = 100, status_filter: str = None) -> dict:
        """
        List documents from the processed_documents table.

        Args:
            limit: Maximum number of documents to return
            status_filter: Filter by document status (e.g., "completed")

        Returns:
            Dictionary containing documents list
        """
        try:
            query = self._client.table('processed_documents').select('*')

            if status_filter:
                query = query.eq('status', status_filter)

            query = query.limit(limit)
            response = query.execute()

            return {
                "documents": response.data,
                "count": len(response.data)
            }
        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            return {"documents": [], "count": 0}
    
    def close(self) -> None:
        """Close the Supabase client connection."""
        if self._client:
            # Supabase client doesn't require explicit closing
            # but we can reset the instance for cleanup
            self._client = None
            logger.info("Supabase client connection closed")


# Global instance
supabase_client = SupabaseClient()


def get_supabase_client() -> SupabaseClient:
    """
    Get the global Supabase client instance.
    
    Returns:
        SupabaseClient instance
    """
    return supabase_client


def initialize_supabase(settings: Settings) -> None:
    """
    Initialize the global Supabase client.
    
    Args:
        settings: Application settings containing Supabase configuration
    """
    supabase_client.initialize(settings)