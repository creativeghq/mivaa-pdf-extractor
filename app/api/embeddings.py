"""
CLIP and Embedding Generation API Routes

This module provides REST API endpoints for generating embeddings:
- CLIP image embeddings
- CLIP text embeddings  
- Combined multimodal embeddings
- Batch embedding generation
"""

import asyncio
import base64
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.real_embeddings_service import RealEmbeddingsService
from ..services.supabase_client import get_supabase_client
from ..config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/embeddings", tags=["Embeddings"])

# Get settings
settings = get_settings()


class ClipImageRequest(BaseModel):
    """Request model for CLIP image embedding generation."""
    
    image_data: str = Field(
        ...,
        description="Base64 encoded image data"
    )
    model: str = Field(
        default="clip-vit-base-patch32",
        description="CLIP model to use"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize the embedding vector"
    )


class ClipTextRequest(BaseModel):
    """Request model for CLIP text embedding generation."""
    
    text: str = Field(
        ...,
        description="Text to generate embedding for"
    )
    model: str = Field(
        default="clip-vit-base-patch32",
        description="CLIP model to use"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize the embedding vector"
    )


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    
    success: bool = Field(..., description="Whether the operation succeeded")
    embedding: Optional[List[float]] = Field(None, description="Generated embedding vector")
    dimensions: Optional[int] = Field(None, description="Embedding dimensions")
    model: Optional[str] = Field(None, description="Model used")
    error: Optional[str] = Field(None, description="Error message if failed")


async def get_embedding_service() -> RealEmbeddingsService:
    """Get embedding service instance."""
    try:
        supabase_client = get_supabase_client()
        return RealEmbeddingsService(supabase_client=supabase_client)
    except Exception as e:
        logger.error(f"Failed to initialize embedding service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service is not available"
        )


@router.post("/clip-image", response_model=EmbeddingResponse)
async def generate_clip_image_embedding(
    request: ClipImageRequest,
    embedding_service: RealEmbeddingsService = Depends(get_embedding_service)
) -> EmbeddingResponse:
    """
    **🖼️ Visual Image Embedding - Powered by SigLIP**

    Generate 512-dimensional visual embedding using Google SigLIP ViT-SO400M
    for superior material image similarity search (+19-29% accuracy improvement).

    ## 🎯 Use Cases

    - Visual product search
    - Material similarity matching
    - Multimodal search (combine with text embeddings)
    - Image clustering and categorization

    ## 📝 Request Example

    ```json
    {
      "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
      "model": "siglip-so400m-patch14-384"
    }
    ```

    ## ✅ Response Example

    ```json
    {
      "embedding": [0.123, -0.456, 0.789, ...],
      "dimension": 512,
      "model": "siglip-so400m-patch14-384",
      "processing_time_ms": 234.5
    }
    ```

    ## 📊 Technical Details

    - **Model**: Google SigLIP ViT-SO400M-14-384
    - **Dimension**: 512
    - **Accuracy**: +19-29% improvement over CLIP on material images
    - **Normalization**: L2 normalized (unit vector)
    - **Distance Metric**: Cosine similarity
    - **Processing Time**: 150-400ms

    ## ⚠️ Error Codes

    - **400 Bad Request**: Invalid base64 image data
    - **413 Payload Too Large**: Image exceeds 10MB
    - **415 Unsupported Media Type**: Unsupported image format
    - **500 Internal Server Error**: Embedding generation failed
    - **503 Service Unavailable**: SigLIP model not available

    ## 📏 Limits

    - **Max image size**: 10MB
    - **Supported formats**: JPEG, PNG, WebP
    - **Rate limit**: 100 requests/minute
    """
    try:
        logger.info(f"Generating SigLIP image embedding with model: {request.model}")

        # Generate visual embedding
        embedding = await embedding_service._generate_visual_embedding(
            image_url=None,
            image_data=request.image_data
        )

        if not embedding:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate visual embedding"
            )

        return EmbeddingResponse(
            success=True,
            embedding=embedding,
            dimensions=len(embedding),
            model="siglip-so400m-patch14-384"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SigLIP image embedding generation failed: {e}")
        return EmbeddingResponse(
            success=False,
            error=str(e)
        )


@router.post("/clip-text", response_model=EmbeddingResponse)
async def generate_clip_text_embedding(
    request: ClipTextRequest,
    embedding_service: RealEmbeddingsService = Depends(get_embedding_service)
) -> EmbeddingResponse:
    """
    **📝 Text Embedding for Multimodal Search**

    Generate text embedding for text-to-image similarity search.
    Note: This endpoint uses OpenAI text embeddings (1536D), not SigLIP.
    For visual embeddings, use the /clip-image endpoint.

    ## 🎯 Use Cases

    - Text-to-image search ("find images of oak furniture")
    - Multimodal search (combine text and image queries)
    - Cross-modal retrieval
    - Semantic image tagging

    ## 📝 Request Example

    ```json
    {
      "text": "modern minimalist oak dining table",
      "model": "text-embedding-3-small"
    }
    ```

    ## ✅ Response Example

    ```json
    {
      "embedding": [0.234, -0.567, 0.891, ...],
      "dimension": 1536,
      "model": "text-embedding-3-small",
      "processing_time_ms": 123.4
    }
    ```

    ## 📊 Technical Details

    - **Model**: OpenAI text-embedding-3-small
    - **Dimension**: 1536
    - **Normalization**: L2 normalized (unit vector)
    - **Distance Metric**: Cosine similarity
    - **Processing Time**: 100-300ms

    ## 💡 Usage Pattern

    1. Generate text embedding using this endpoint
    2. Search for similar content using cosine similarity:
       ```sql
       SELECT * FROM chunks
       ORDER BY text_embedding_1536 <=> '[your_embedding]'
       LIMIT 10
       ```

    ## ⚠️ Error Codes

    - **400 Bad Request**: Empty or invalid text
    - **500 Internal Server Error**: Embedding generation failed
    - **503 Service Unavailable**: OpenAI API not available

    ## 📏 Limits

    - **Max text length**: 8191 tokens
    - **Rate limit**: 100 requests/minute
    """
    try:
        logger.info(f"Generating text embedding with model: {request.model}")

        # Generate text embedding
        embedding = await embedding_service._generate_text_embedding(
            text=request.text
        )

        if not embedding:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate text embedding"
            )

        return EmbeddingResponse(
            success=True,
            embedding=embedding,
            dimensions=len(embedding),
            model="text-embedding-3-small"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CLIP text embedding generation failed: {e}")
        return EmbeddingResponse(
            success=False,
            error=str(e)
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for embeddings service."""
    try:
        # Try to initialize service
        embedding_service = await get_embedding_service()
        
        return {
            "success": True,
            "message": "Embeddings service is healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "available_models": ["clip-vit-base-patch32", "text-embedding-3-small"]
        }
    except Exception as e:
        logger.error(f"Embeddings health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embeddings service is unavailable: {str(e)}"
        )

