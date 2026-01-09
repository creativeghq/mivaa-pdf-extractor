"""
AI Model Configuration

Defines configurable AI models for different pipeline stages with defaults and alternatives.
All parameters can be overridden via API requests.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class AIModelConfig(BaseModel):
    """Configuration for AI models used in PDF processing pipeline."""

    # Visual Embedding Model (SLIG Cloud Endpoint - basiliskan/siglip2)
    # All visual embeddings are generated via HuggingFace cloud endpoint (768D)
    visual_embedding_primary: str = Field(
        default="basiliskan/siglip2",
        description="Visual embedding model via SLIG cloud endpoint (768D)"
    )
    visual_embedding_fallback: str = Field(
        default=None,
        description="Fallback visual embedding model (not used - cloud-only)"
    )
    
    # Text Embedding Model (Voyage AI primary, OpenAI fallback)
    text_embedding_model: str = Field(
        default="voyage-3.5",
        description="Text embedding model (Voyage AI primary, OpenAI fallback)"
    )
    text_embedding_dimensions: int = Field(
        default=1024,
        description="Text embedding dimensions (1024 for Voyage AI, 1536 for OpenAI)"
    )
    text_embedding_input_type: str = Field(
        default="document",
        description="Input type for Voyage AI: 'document' for indexing, 'query' for search"
    )
    
    # Image Classification Models (Qwen 32B only - HuggingFace Endpoint)
    classification_primary_model: str = Field(
        default="Qwen/Qwen3-VL-32B-Instruct",
        description="Primary image classification model (Qwen Vision 32B via HF Endpoint)"
    )
    classification_validation_model: str = Field(
        default="Qwen/Qwen3-VL-32B-Instruct",
        description="Validation model for low-confidence classifications (same as primary - 32B only)"
    )
    classification_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for triggering validation"
    )

    # Product Discovery Model
    discovery_model: Literal["claude-sonnet-4-5-20250929", "gpt-5", "gpt-4o"] = Field(
        default="claude-sonnet-4-5-20250929",
        description="Model for product discovery (Claude Sonnet 4.5, GPT-5, or GPT-4o)"
    )
    
    # Metadata Extraction Model
    metadata_extraction_model: Literal["claude", "gpt"] = Field(
        default="claude",
        description="Model for metadata extraction (claude or gpt)"
    )
    
    # Chunking Model (for semantic chunking)
    chunking_model: str = Field(
        default="gpt-4o",
        description="Model for semantic chunking and text analysis"
    )
    
    # Temperature Settings
    discovery_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for product discovery"
    )
    classification_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for image classification"
    )
    metadata_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for metadata extraction"
    )
    
    # Max Tokens
    discovery_max_tokens: int = Field(
        default=4096,
        ge=512,
        le=16384,
        description="Max tokens for product discovery"
    )
    classification_max_tokens: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Max tokens for image classification"
    )
    metadata_max_tokens: int = Field(
        default=4096,
        ge=512,
        le=16384,
        description="Max tokens for metadata extraction"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "visual_embedding_primary": "basiliskan/siglip2",
                "visual_embedding_fallback": null,
                "text_embedding_model": "voyage-3.5",
                "text_embedding_dimensions": 1024,
                "text_embedding_input_type": "document",
                "classification_primary_model": "Qwen/Qwen3-VL-8B-Instruct",
                "classification_validation_model": "claude-sonnet-4-20250514",
                "classification_confidence_threshold": 0.7,
                "discovery_model": "claude-sonnet-4-20250514",
                "metadata_extraction_model": "claude",
                "chunking_model": "gpt-4o",
                "discovery_temperature": 0.1,
                "classification_temperature": 0.1,
                "metadata_temperature": 0.1,
                "discovery_max_tokens": 4096,
                "classification_max_tokens": 512,
                "metadata_max_tokens": 4096
            }
        }


# Default configuration instance
DEFAULT_AI_CONFIG = AIModelConfig()


# Alternative configurations for different use cases
FAST_CONFIG = AIModelConfig(
    discovery_model="gpt-4o",  # Faster than Claude
    classification_validation_model="claude-haiku-4-20250514",  # Faster validation
    metadata_extraction_model="gpt",  # Faster metadata extraction
    discovery_max_tokens=2048,  # Reduce tokens for speed
    metadata_max_tokens=2048
)

HIGH_ACCURACY_CONFIG = AIModelConfig(
    discovery_model="gpt-5",  # Most accurate
    classification_validation_model="claude-sonnet-4-20250514",  # Best validation
    metadata_extraction_model="claude",  # Most accurate metadata
    classification_confidence_threshold=0.8,  # Higher threshold
    discovery_max_tokens=8192,  # More context
    metadata_max_tokens=8192
)

COST_OPTIMIZED_CONFIG = AIModelConfig(
    discovery_model="gpt-4o",  # Good balance
    classification_validation_model="claude-haiku-4-20250514",  # Cheaper validation
    metadata_extraction_model="gpt",  # Cheaper metadata
    classification_confidence_threshold=0.6,  # Lower threshold to reduce validation calls
    discovery_max_tokens=2048,  # Reduce tokens
    metadata_max_tokens=2048
)


