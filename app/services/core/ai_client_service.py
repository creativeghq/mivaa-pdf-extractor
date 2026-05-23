from __future__ import annotations

"""
Centralized AI Client Service

Provides singleton instances of AI API clients (Anthropic, OpenAI, HuggingFace)
with proper connection pooling, configuration management, and logging.

This service eliminates the need to create new clients in every function,
improving performance through connection reuse and providing consistent
configuration across the application.
"""

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING
import openai
import httpx

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI

from app.config import get_settings
from .ai_call_logger import AICallLogger

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic SDK shims (2026-05-23 — SDK removed, standardized on httpx)
#
# The `anthropic-sdk-python` package was removed as a dependency. Older call
# sites use `client.messages.create(**kwargs)`; these shims preserve that API
# by proxying to `claude_helper._call_anthropic_async / _sync`. Returns a
# `ClaudeResponse` shaped identically to the SDK's Message object, so
# `.content[i].type/.text/.input`, `.usage.input_tokens`, `.model`, etc.
# work unchanged.
#
# New code should call `tracked_claude_call_async` directly. The shims exist
# to make this migration zero-touch for the dozen-ish existing call sites.
# ─────────────────────────────────────────────────────────────────────────────


class _AnthropicMessagesAsync:
    """Async `.messages.create(...)` proxy."""
    async def create(self, **kwargs: Any) -> Any:
        from app.services.core.claude_helper import _call_anthropic_async
        # SDK accepts `model`, `max_tokens`, `messages`, `temperature`,
        # `system`, `tools`, `tool_choice` as kwargs. _call_anthropic_async
        # has the same signature with `**extra` for arbitrary additions.
        model = kwargs.pop("model")
        messages = kwargs.pop("messages")
        max_tokens = kwargs.pop("max_tokens", 4096)
        temperature = kwargs.pop("temperature", 0.0)
        system = kwargs.pop("system", None)
        return await _call_anthropic_async(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            **kwargs,
        )


class _AnthropicMessagesSync:
    """Sync `.messages.create(...)` proxy."""
    def create(self, **kwargs: Any) -> Any:
        from app.services.core.claude_helper import _call_anthropic_sync
        model = kwargs.pop("model")
        messages = kwargs.pop("messages")
        max_tokens = kwargs.pop("max_tokens", 4096)
        temperature = kwargs.pop("temperature", 0.0)
        system = kwargs.pop("system", None)
        return _call_anthropic_sync(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            **kwargs,
        )


class _AnthropicShimAsync:
    """Async shim mirroring `anthropic.AsyncAnthropic` surface."""
    def __init__(self, api_key: Optional[str] = None, **_ignored: Any):
        # api_key is read from settings by claude_helper at call time, so we
        # accept (and ignore) the constructor arg for SDK API parity.
        self.messages = _AnthropicMessagesAsync()


class _AnthropicShimSync:
    """Sync shim mirroring `anthropic.Anthropic` surface."""
    def __init__(self, api_key: Optional[str] = None, **_ignored: Any):
        self.messages = _AnthropicMessagesSync()


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
        
        # Client instances (initialized lazily). Anthropic uses shim classes
        # backed by httpx — no SDK dependency. See module docstring.
        self._anthropic_client: Optional[_AnthropicShimSync] = None
        self._anthropic_async_client: Optional[_AnthropicShimAsync] = None
        self._openai_client: Optional[openai.OpenAI] = None
        self._openai_async_client: Optional[openai.AsyncOpenAI] = None
        self._httpx_client: Optional[httpx.AsyncClient] = None
        
        AIClientService._initialized = True
        logger.info("✅ AIClientService initialized (singleton)")
    
    @property
    def anthropic(self) -> _AnthropicShimSync:
        """Get sync Anthropic shim. `.messages.create(...)` proxies to httpx."""
        if self._anthropic_client is None:
            if not self.settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not configured")
            self._anthropic_client = _AnthropicShimSync(api_key=self.settings.anthropic_api_key)
            logger.info("✅ Anthropic sync shim initialized (httpx-backed)")
        return self._anthropic_client

    @property
    def anthropic_async(self) -> _AnthropicShimAsync:
        """Get async Anthropic shim. `.messages.create(...)` proxies to httpx."""
        if self._anthropic_async_client is None:
            if not self.settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not configured")
            self._anthropic_async_client = _AnthropicShimAsync(api_key=self.settings.anthropic_api_key)
            logger.info("✅ Anthropic async shim initialized (httpx-backed)")
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
        """Get shared httpx async client for HuggingFace and other HTTP APIs.

        Timeout is 200s — sized for the slowest Qwen vision call we make.
        Previous default of 35s caused ReadTimeout on first inference of a
        cold-started endpoint; large multimodal requests (image + spec
        prompt) routinely take 60-120s end-to-end on the current Qwen
        endpoint, so 200s leaves headroom without masking real hangs.
        """
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(
                # 200s sized for the slowest Qwen vision call. See class
                # docstring for the latency profile that drove this.
                timeout=httpx.Timeout(200.0),  # Generous timeout for large vision models
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
            )
            logger.info("✅ HTTPX async client initialized (timeout: 200s for large vision models)")

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


