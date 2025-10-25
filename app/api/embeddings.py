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
    Generate CLIP embedding for an image.
    
    This endpoint:
    - Accepts base64 encoded image data
    - Generates 512-dimensional CLIP embedding
    - Returns normalized embedding vector
    """
    try:
        logger.info(f"Generating CLIP image embedding with model: {request.model}")
        
        # Generate visual embedding
        embedding = await embedding_service._generate_visual_embedding(
            image_url=None,
            image_data=request.image_data
        )
        
        if not embedding:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate CLIP embedding"
            )
        
        return EmbeddingResponse(
            success=True,
            embedding=embedding,
            dimensions=len(embedding),
            model=request.model
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CLIP image embedding generation failed: {e}")
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
    Generate CLIP embedding for text.
    
    This endpoint:
    - Accepts text input
    - Generates 512-dimensional CLIP text embedding
    - Returns normalized embedding vector
    """
    try:
        logger.info(f"Generating CLIP text embedding with model: {request.model}")
        
        # Generate text embedding
        embedding = await embedding_service._generate_text_embedding(
            text=request.text
        )
        
        if not embedding:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate CLIP text embedding"
            )
        
        return EmbeddingResponse(
            success=True,
            embedding=embedding,
            dimensions=len(embedding),
            model=request.model
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

