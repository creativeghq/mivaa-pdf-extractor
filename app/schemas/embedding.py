"""
Embedding service schemas and models.

This module provides Pydantic models for embedding service configuration
and request/response handling.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class EmbeddingConfig(BaseModel):
    """Configuration for embedding service."""

    model_name: str = Field(
        default="voyage-3.5",
        description="Embedding model name (voyage-3.5 or text-embedding-3-small)"
    )
    model_type: str = Field(
        default="voyage",
        description="Model type: 'voyage', 'openai', or 'huggingface'"
    )
    dimension: int = Field(
        default=1024,
        description="Embedding dimension (1024 for Voyage AI, 1536 for OpenAI)"
    )
    input_type: str = Field(
        default="document",
        description="Input type: 'document' for indexing, 'query' for search (Voyage AI only)"
    )
    api_key: str = Field(
        default="",
        description="API key (Voyage AI or OpenAI)"
    )
    max_tokens: int = Field(
        default=8000,
        description="Maximum tokens per request (8000 for Voyage AI, 8191 for OpenAI)"
    )
    batch_size: int = Field(
        default=100,
        description="Batch size for processing"
    )
    rate_limit_rpm: int = Field(
        default=3000,
        description="Rate limit requests per minute"
    )
    rate_limit_tpm: int = Field(
        default=1000000,
        description="Rate limit tokens per minute"
    )
    cache_ttl: int = Field(
        default=3600,
        description="Cache TTL in seconds"
    )
    enable_cache: bool = Field(
        default=True,
        description="Enable caching"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "model_name": "voyage-3.5",
                "model_type": "voyage",
                "dimension": 1024,
                "input_type": "document",
                "api_key": "pa-...",
                "max_tokens": 8000,
                "batch_size": 100,
                "rate_limit_rpm": 3000,
                "rate_limit_tpm": 1000000,
                "cache_ttl": 3600,
                "enable_cache": True
            }
        }


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""

    text: str = Field(
        description="Text to generate embedding for"
    )
    model: Optional[str] = Field(
        default=None,
        description="Override model name (voyage-3.5 or text-embedding-3-small)"
    )
    input_type: Optional[str] = Field(
        default="document",
        description="Input type: 'document' for indexing, 'query' for search"
    )
    dimensions: Optional[int] = Field(
        default=1024,
        description="Embedding dimensions (1024 for Voyage AI, 1536 for OpenAI)"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "text": "This is a sample text for embedding generation",
                "model": "voyage-3.5",
                "input_type": "document",
                "dimensions": 1024
            }
        }


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    
    embedding: List[float] = Field(
        description="Generated embedding vector"
    )
    model: str = Field(
        description="Model used for generation"
    )
    usage: Dict[str, Any] = Field(
        description="Usage statistics"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "embedding": [0.1, 0.2, 0.3],
                "model": "voyage-3.5",
                "usage": {
                    "prompt_tokens": 10,
                    "total_tokens": 10
                }
            }
        }


class BatchEmbeddingRequest(BaseModel):
    """Request model for batch embedding generation."""

    texts: List[str] = Field(
        description="List of texts to generate embeddings for"
    )
    model: Optional[str] = Field(
        default=None,
        description="Override model name (voyage-3.5 or text-embedding-3-small)"
    )
    input_type: Optional[str] = Field(
        default="document",
        description="Input type: 'document' for indexing, 'query' for search"
    )
    dimensions: Optional[int] = Field(
        default=1024,
        description="Embedding dimensions (1024 for Voyage AI, 1536 for OpenAI)"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "texts": [
                    "First text for embedding",
                    "Second text for embedding"
                ],
                "model": "voyage-3.5",
                "input_type": "document",
                "dimensions": 1024
            }
        }


class BatchEmbeddingResponse(BaseModel):
    """Response model for batch embedding generation."""
    
    results: List[EmbeddingResponse] = Field(
        description="List of embedding results"
    )
    total_usage: Dict[str, Any] = Field(
        description="Total usage statistics"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "embedding": [0.1, 0.2, 0.3],
                        "model": "voyage-3.5",
                        "usage": {"prompt_tokens": 5, "total_tokens": 5}
                    }
                ],
                "total_usage": {
                    "prompt_tokens": 10,
                    "total_tokens": 10
                }
            }
        }

