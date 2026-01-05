from __future__ import annotations

"""
Centralized AI Client Service

Provides singleton instances of AI API clients (Anthropic, OpenAI, TogetherAI)
with proper connection pooling, configuration management, and logging.

This service eliminates the need to create new clients in every function,
improving performance through connection reuse and providing consistent
configuration across the application.
"""

import logging
from typing import Optional, TYPE_CHECKING
from anthropic import Anthropic, AsyncAnthropic
import openai
import httpx

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI

from app.config import get_settings
from .ai_call_logger import AICallLogger

logger = logging.getLogger(__name__)


class AIClientService:
    """
    Centralized service for managing AI API clients.
    
    Provides singleton instances with proper configuration and connection pooling.
    All clients are initialized lazily on first access.
    """
    
    _instance: Optional['AIClientService'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Ensure only one instance exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the service (only runs once due to singleton)."""
        if self._initialized:
            return
            
        self.settings = get_settings()
        self.ai_logger = AICallLogger()
        
        # Client instances (initialized lazily)
        self._anthropic_client: Optional[Anthropic] = None
        self._anthropic_async_client: Optional[AsyncAnthropic] = None
        self._openai_client: Optional[openai.OpenAI] = None
        self._openai_async_client: Optional[openai.AsyncOpenAI] = None
        self._httpx_client: Optional[httpx.AsyncClient] = None
        
        AIClientService._initialized = True
        logger.info("✅ AIClientService initialized (singleton)")
    
    @property
    def anthropic(self) -> Anthropic:
        """Get synchronous Anthropic client (lazy initialization)."""
        if self._anthropic_client is None:
            if not self.settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not configured")
            
            self._anthropic_client = Anthropic(
                api_key=self.settings.anthropic_api_key,
                timeout=self.settings.anthropic_timeout,
                max_retries=3
            )
            logger.info("✅ Anthropic sync client initialized")
        
        return self._anthropic_client
    
    @property
    def anthropic_async(self) -> AsyncAnthropic:
        """Get asynchronous Anthropic client (lazy initialization)."""
        if self._anthropic_async_client is None:
            if not self.settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not configured")
            
            self._anthropic_async_client = AsyncAnthropic(
                api_key=self.settings.anthropic_api_key,
                timeout=self.settings.anthropic_timeout,
                max_retries=3
            )
            logger.info("✅ Anthropic async client initialized")
        
        return self._anthropic_async_client
    
    @property
    def openai(self) -> "OpenAI":
        """Get synchronous OpenAI client (lazy initialization)."""
        if self._openai_client is None:
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not configured")

            self._openai_client = openai.OpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.settings.openai_timeout,
                max_retries=3
            )
            logger.info("✅ OpenAI sync client initialized")

        return self._openai_client

    @property
    def openai_async(self) -> "AsyncOpenAI":
        """Get asynchronous OpenAI client (lazy initialization)."""
        if self._openai_async_client is None:
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not configured")
            
            self._openai_async_client = openai.AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.settings.openai_timeout,
                max_retries=3
            )
            logger.info("✅ OpenAI async client initialized")
        
        return self._openai_async_client
    
    @property
    def httpx(self) -> httpx.AsyncClient:
        """Get shared httpx async client for TogetherAI and other HTTP APIs.

        CRITICAL FIX: Timeout aligned with application timeout guard (30s for Qwen Vision)
        to prevent HTTP 499 (Client Closed Request) errors.

        Previous: 120s timeout caused conflicts when app timeout guard (30s) killed requests.
        Result: httpx client still waiting → connection closed → HTTP 499
        """
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(
                # CRITICAL FIX: Reduced from 120s to 35s to align with app timeout guard (30s)
                # This prevents HTTP 499 errors when timeout guard cancels the operation
                timeout=httpx.Timeout(35.0),  # Slightly higher than app guard (30s) for graceful handling
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
            )
            logger.info("✅ HTTPX async client initialized (timeout: 35s, aligned with app guards)")

        return self._httpx_client
    
    async def close(self):
        """Close all async clients (call on application shutdown)."""
        if self._httpx_client:
            await self._httpx_client.aclose()
            logger.info("✅ HTTPX client closed")
        
        # Anthropic and OpenAI async clients don't need explicit closing
        # but we can reset them
        self._anthropic_async_client = None
        self._openai_async_client = None


# Factory function for easy access
_ai_client_service: Optional[AIClientService] = None

def get_ai_client_service() -> AIClientService:
    """
    Get the singleton AIClientService instance.
    
    Returns:
        AIClientService: Singleton instance
    """
    global _ai_client_service
    if _ai_client_service is None:
        _ai_client_service = AIClientService()
    return _ai_client_service


