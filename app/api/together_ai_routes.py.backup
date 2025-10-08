"""
TogetherAI API routes for LLaMA Vision semantic analysis.

This module provides REST API endpoints for TogetherAI/LLaMA Vision integration,
specifically for material semantic analysis using the meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo model.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
try:
    # Try Pydantic v2 first
    from pydantic import BaseModel, Field, field_validator as validator
except ImportError:
    # Fall back to Pydantic v1
    from pydantic import BaseModel, Field, validator

from ..schemas.common import BaseResponse, ErrorResponse
from ..services.together_ai_service import (
    TogetherAIService,
    SemanticAnalysisRequest,
    SemanticAnalysisResult,
    get_together_ai_service
)
from ..dependencies import get_current_user, get_workspace_context
from ..schemas.auth import WorkspaceContext, User
from ..config import get_settings
from ..utils.exceptions import ServiceError, ExternalServiceError

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["AI Analysis"])

# Get settings
settings = get_settings()


class SemanticAnalysisAPIRequest(BaseModel):
    """API request model for semantic analysis endpoint."""
    
    image_data: str = Field(
        ..., 
        description="Base64 encoded image data or image URL",
        min_length=1
    )
    analysis_type: str = Field(
        default="material_identification",
        description="Type of analysis to perform"
    )
    prompt: Optional[str] = Field(
        None,
        description="Custom analysis prompt (optional)"
    )
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional analysis options"
    )
    
    @validator("analysis_type")
    def validate_analysis_type(cls, v):
        """Validate analysis type."""
        allowed_types = ["material_identification", "semantic_description", "general_analysis"]
        if v not in allowed_types:
            raise ValueError(f"analysis_type must be one of: {allowed_types}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
                "analysis_type": "material_identification",
                "prompt": "Analyze this material image for material properties",
                "options": {
                    "temperature": 0.1,
                    "max_tokens": 200
                }
            }
        }


class SemanticAnalysisAPIResponse(BaseResponse):
    """API response model for semantic analysis endpoint."""
    
    analysis: str = Field(..., description="Generated semantic description")
    confidence: float = Field(..., description="Analysis confidence score (0.0-1.0)")
    model_used: str = Field(..., description="AI model used for analysis")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional analysis metadata"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Semantic analysis completed successfully",
                "timestamp": "2025-08-31T06:57:00Z",
                "analysis": "This material appears to be polished granite with...",
                "confidence": 0.95,
                "model_used": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
                "processing_time_ms": 1500,
                "metadata": {
                    "cache_hit": False,
                    "request_id": "req_abc123"
                }
            }
        }


@router.post(
    "/semantic-analysis",
    response_model=SemanticAnalysisAPIResponse,
    status_code=status.HTTP_200_OK,
    summary="Perform semantic analysis on material images",
    description="Analyze material images using TogetherAI's LLaMA Vision model for semantic descriptions and material identification."
)
async def semantic_analysis(
    request: SemanticAnalysisAPIRequest,
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    together_ai_service: TogetherAIService = Depends(get_together_ai_service)
):
    """
    Perform semantic analysis on material images using TogetherAI/LLaMA Vision.
    
    This endpoint provides material identification and semantic description capabilities
    using the meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo model.
    
    Args:
        request: Semantic analysis request with image data and parameters
        current_user: Authenticated user information
        workspace_context: Current workspace context
        together_ai_service: TogetherAI service instance
    
    Returns:
        SemanticAnalysisAPIResponse: Analysis results with confidence and metadata
    
    Raises:
        HTTPException: For validation errors, service failures, or rate limiting
    """
    start_time = time.time()
    
    try:
        logger.info(
            f"Starting semantic analysis for user {current_user.id} "
            f"with analysis_type: {request.analysis_type}"
        )
        
        # Determine image source type
        image_url = None
        image_base64 = None
        
        if request.image_data.startswith(('http://', 'https://')):
            image_url = request.image_data
        elif request.image_data.startswith('data:image/'):
            image_base64 = request.image_data
        else:
            # Assume raw base64
            image_base64 = request.image_data
        
        # Create service request
        service_request = SemanticAnalysisRequest(
            image_url=image_url,
            image_base64=image_base64,
            analysis_type=request.analysis_type,
            context=request.prompt,
            temperature=request.options.get("temperature", 0.1),
            max_tokens=request.options.get("max_tokens", 200)
        )
        
        # Perform semantic analysis
        result = await together_ai_service.analyze_semantic_content(service_request)
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Prepare response
        response = SemanticAnalysisAPIResponse(
            success=True,
            message="Semantic analysis completed successfully",
            analysis=result.analysis,
            confidence=result.confidence,
            model_used=result.model_used,
            processing_time_ms=processing_time_ms,
            metadata={
                "cache_hit": result.metadata.get("cache_hit", False),
                "request_id": result.metadata.get("request_id"),
                "analysis_type": request.analysis_type,
                "user_id": current_user.id,
                "workspace_id": workspace_context.workspace_id
            }
        )
        
        logger.info(
            f"Semantic analysis completed successfully for user {current_user.id}. "
            f"Processing time: {processing_time_ms}ms, Confidence: {result.confidence}"
        )
        
        return response
        
    except ServiceError as e:
        logger.error(f"Service error during semantic analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "TOGETHER_AI_SERVICE_ERROR",
                "error_type": "ServiceError",
                "message": str(e)
            }
        )
    
    except ExternalServiceError as e:
        logger.error(f"External service error during semantic analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "TOGETHER_AI_EXTERNAL_ERROR",
                "error_type": "ExternalServiceError",
                "message": str(e)
            }
        )
    
    except Exception as e:
        logger.error(f"Unexpected error during semantic analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "SEMANTIC_ANALYSIS_ERROR",
                "error_type": "UnexpectedError",
                "message": "An unexpected error occurred during semantic analysis"
            }
        )


@router.get(
    "/health",
    response_model=BaseResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check for TogetherAI service",
    description="Check the health and availability of the TogetherAI service."
)
async def health_check(
    together_ai_service: TogetherAIService = Depends(get_together_ai_service)
):
    """
    Health check endpoint for TogetherAI service.
    
    Args:
        together_ai_service: TogetherAI service instance
    
    Returns:
        BaseResponse: Health check status
    """
    try:
        # Simple health check - verify service configuration
        is_healthy = await together_ai_service.health_check()
        
        if is_healthy:
            return BaseResponse(
                success=True,
                message="TogetherAI service is healthy and ready"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error_code": "TOGETHER_AI_UNHEALTHY",
                    "error_type": "ServiceUnavailable",
                    "message": "TogetherAI service health check failed"
                }
            )
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "HEALTH_CHECK_ERROR",
                "error_type": "ServiceError",
                "message": "Health check endpoint failed"
            }
        )


@router.get(
    "/models",
    response_model=BaseResponse,
    status_code=status.HTTP_200_OK,
    summary="Get available TogetherAI models",
    description="Retrieve information about available TogetherAI models and their capabilities."
)
async def get_models(
    together_ai_service: TogetherAIService = Depends(get_together_ai_service)
):
    """
    Get available TogetherAI models information.
    
    Args:
        together_ai_service: TogetherAI service instance
    
    Returns:
        BaseResponse: Available models information
    """
    try:
        models_info = await together_ai_service.get_models_info()
        
        return BaseResponse(
            success=True,
            message="Available TogetherAI models retrieved successfully",
            data=models_info
        )
    
    except Exception as e:
        logger.error(f"Failed to retrieve models information: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "MODELS_INFO_ERROR",
                "error_type": "ServiceError",
                "message": "Failed to retrieve models information"
            }
        )