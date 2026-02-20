"""
Spaceformer Spatial Analysis API Routes

Endpoints for AI-powered spatial reasoning, room layout optimization,
material placement, and accessibility analysis.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import logging

from app.services.integrations.spaceformer_service import SpaceformerService, get_spaceformer_service
from app.schemas.common import BaseResponse, ErrorResponse

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/spaceformer", tags=["Spatial Analysis"])


# ============================================================================
# Request/Response Models
# ============================================================================

class SpatialFeature(BaseModel):
    """Detected spatial feature in the room"""
    type: str = Field(..., description="Feature type (window, door, furniture, etc.)")
    position: Dict[str, float] = Field(..., description="3D position {x, y, z}")
    dimensions: Dict[str, float] = Field(..., description="Dimensions {width, height, depth}")
    importance: float = Field(..., ge=0.0, le=1.0, description="Feature importance score")
    accessibility_rating: float = Field(..., ge=0.0, le=1.0, description="Accessibility rating")


class LayoutSuggestion(BaseModel):
    """AI-generated layout suggestion"""
    item_type: str = Field(..., description="Type of item to place")
    position: Dict[str, float] = Field(..., description="Suggested position {x, y, z}")
    rotation: float = Field(..., description="Rotation in degrees")
    reasoning: str = Field(..., description="Why this placement is suggested")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    alternative_positions: Optional[List[Dict[str, float]]] = Field(None, description="Alternative placements")


class MaterialPlacement(BaseModel):
    """Material placement recommendation"""
    material_id: str = Field(..., description="Material ID from catalog")
    position: Dict[str, float] = Field(..., description="Position {x, y, z}")
    surface_area: float = Field(..., description="Surface area in square meters")
    application_method: str = Field(..., description="How to apply the material")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Why this material is recommended")


class AccessibilityAnalysis(BaseModel):
    """Accessibility compliance analysis"""
    compliance_score: float = Field(..., ge=0.0, le=1.0, description="Overall compliance score")
    accessibility_features: List[str] = Field(..., description="Detected accessibility features")
    recommendations: List[str] = Field(..., description="Improvement recommendations")
    barrier_free_paths: List[Dict[str, Any]] = Field(..., description="Barrier-free pathways")
    ada_compliance: bool = Field(..., description="ADA compliance status")


class FlowOptimization(BaseModel):
    """Traffic flow optimization analysis"""
    traffic_patterns: List[Dict[str, Any]] = Field(..., description="Detected traffic patterns")
    bottlenecks: List[Dict[str, Any]] = Field(..., description="Identified bottlenecks")
    efficiency_score: float = Field(..., ge=0.0, le=1.0, description="Flow efficiency score")
    suggested_improvements: List[str] = Field(..., description="Improvement suggestions")


class SpaceformerRequest(BaseModel):
    """Request for spatial analysis"""
    image_url: Optional[str] = Field(None, description="URL of room image")
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    room_type: str = Field(..., description="Type of room (bedroom, kitchen, bathroom, etc.)")
    room_dimensions: Optional[Dict[str, float]] = Field(None, description="Room dimensions {width, height, depth}")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences for analysis")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Analysis constraints")
    analysis_type: str = Field(default="full", description="Type of analysis (full, layout, materials, accessibility)")
    workspace_id: Optional[str] = Field(None, description="Workspace ID (passed by internal callers)")
    user_id: Optional[str] = Field(None, description="User ID (passed by internal callers)")

    @validator("analysis_type")
    def validate_analysis_type(cls, v):
        """Validate analysis type"""
        allowed_types = ["full", "layout", "materials", "accessibility"]
        if v not in allowed_types:
            raise ValueError(f"analysis_type must be one of: {allowed_types}")
        return v

    @validator("image_url", "image_data", always=True)
    def validate_image_input(cls, v, values):
        """Ensure at least one image input is provided"""
        if not values.get("image_url") and not values.get("image_data"):
            raise ValueError("Either image_url or image_data must be provided")
        return v


class SpaceformerResponse(BaseModel):
    """Response from spatial analysis"""
    success: bool = Field(..., description="Analysis success status")
    analysis_id: str = Field(..., description="Unique analysis ID")
    spatial_features: List[SpatialFeature] = Field(..., description="Detected spatial features")
    layout_suggestions: List[LayoutSuggestion] = Field(..., description="Layout suggestions")
    material_placements: List[MaterialPlacement] = Field(..., description="Material placement recommendations")
    accessibility_analysis: AccessibilityAnalysis = Field(..., description="Accessibility analysis")
    flow_optimization: FlowOptimization = Field(..., description="Flow optimization analysis")
    reasoning_explanation: str = Field(..., description="Detailed reasoning for recommendations")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/analyze", response_model=SpaceformerResponse, status_code=status.HTTP_200_OK)
async def analyze_spatial_context(
    request: SpaceformerRequest,
    spaceformer: SpaceformerService = Depends(get_spaceformer_service),
) -> SpaceformerResponse:
    """
    **🏠 Spatial Analysis - AI-Powered Room Understanding**

    Analyze room images using Claude Vision for comprehensive spatial reasoning,
    layout optimization, material placement, and accessibility analysis.

    ## 🎯 Analysis Types

    - **full**: Complete analysis (layout + materials + accessibility + flow)
    - **layout**: Layout suggestions and furniture placement only
    - **materials**: Material placement recommendations only
    - **accessibility**: Accessibility compliance analysis only

    ## 📝 Request Example

    ```json
    {
      "image_url": "https://example.com/room.jpg",
      "room_type": "living_room",
      "room_dimensions": {"width": 5.0, "height": 2.8, "depth": 4.0},
      "analysis_type": "full"
    }
    ```

    ## ✨ Features

    - **Spatial Feature Detection**: Identify windows, doors, furniture, fixtures
    - **Layout Optimization**: AI-generated furniture placement suggestions
    - **Material Recommendations**: Optimal material placement for surfaces
    - **Accessibility Analysis**: ADA compliance and barrier-free path detection
    - **Flow Optimization**: Traffic pattern analysis and bottleneck identification

    ## 🔒 Authentication

    Requires valid user authentication and workspace context.

    ## 📊 Response

    Returns comprehensive spatial analysis with:
    - Detected spatial features with 3D positions
    - Layout suggestions with confidence scores
    - Material placement recommendations
    - Accessibility compliance analysis
    - Traffic flow optimization
    - Detailed reasoning explanations
    """
    try:
        logger.info(f"Starting spatial analysis for workspace={request.workspace_id}, room_type: {request.room_type}")

        # Perform spatial analysis
        result = await spaceformer.analyze_space(
            image_url=request.image_url,
            image_data=request.image_data,
            room_type=request.room_type,
            room_dimensions=request.room_dimensions,
            user_preferences=request.user_preferences,
            constraints=request.constraints,
            analysis_type=request.analysis_type,
            user_id=request.user_id,
            workspace_id=request.workspace_id
        )

        logger.info(f"Spatial analysis completed: {result['analysis_id']}")
        return SpaceformerResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in spatial analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in spatial analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Spatial analysis failed: {str(e)}"
        )


