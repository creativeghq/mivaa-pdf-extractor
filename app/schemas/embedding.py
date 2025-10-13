"""
Embedding service schemas and models.

This module provides Pydantic models for embedding service configuration
and request/response handling.
"""

from typing import Optional, List, Dict, Any
try:
    # Try Pydantic v2 first
    from pydantic import BaseModel, Field, field_validator as validator
    PYDANTIC_V2 = True
except ImportError:
    # Fall back to Pydantic v1
    from pydantic import BaseModel, Field, validator
    PYDANTIC_V2 = False


class EmbeddingConfig(BaseModel):
    """Configuration for embedding service."""
    
    model_name: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name"
    )
    model_type: str = Field(
        default="openai",
        description="Model type: 'openai' or 'huggingface'"
    )
    dimension: int = Field(
        default=1536,
        description="Embedding dimension"
    )
    api_key: str = Field(
        default="",
        description="OpenAI API key"
    )
    max_tokens: int = Field(
        default=8191,
        description="Maximum tokens per request"
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
                "model_name": "text-embedding-3-small",
                "api_key": "sk-...",
                "max_tokens": 8191,
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
        description="Override model name"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "text": "This is a sample text for embedding generation",
                "model": "text-embedding-3-small"
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
                "model": "text-embedding-3-small",
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
        description="Override model name"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "texts": [
                    "First text for embedding",
                    "Second text for embedding"
                ],
                "model": "text-embedding-3-small"
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
                        "model": "text-embedding-3-small",
                        "usage": {"prompt_tokens": 5, "total_tokens": 5}
                    }
                ],
                "total_usage": {
                    "prompt_tokens": 10,
                    "total_tokens": 10
                }
            }
        }
