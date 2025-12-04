"""
User Feedback API

Endpoints for submitting and analyzing user feedback with sentiment analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..services.supabase_client import get_supabase_client, SupabaseClient
from ..services.sentiment_analysis_service import SentimentAnalysisService
from ..config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/feedback", tags=["User Feedback"])

# Initialize sentiment analysis service
settings = get_settings()
sentiment_service = SentimentAnalysisService(together_api_key=settings.together_api_key)


# Request/Response Models
class SubmitFeedbackRequest(BaseModel):
    workspace_id: str = Field(..., description="Workspace ID")
    user_id: str = Field(..., description="User ID")
    material_id: Optional[str] = Field(None, description="Material/Product ID")
    feedback_text: str = Field(..., min_length=10, description="Feedback text (min 10 characters)")
    feedback_type: str = Field("review", description="Type: review, comment, rating, suggestion")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating 1-5 stars")
    is_verified: bool = Field(False, description="Verified purchase/usage")
    is_public: bool = Field(True, description="Show in public reviews")
    context: Optional[dict] = Field(None, description="Additional context")


class FeedbackResponse(BaseModel):
    id: str
    workspace_id: str
    user_id: str
    material_id: Optional[str]
    feedback_text: str
    feedback_type: str
    rating: Optional[int]
    sentiment_analysis: dict
    is_verified: bool
    is_public: bool
    helpful_count: int
    created_at: str
    analyzed_at: Optional[str]


class SentimentTrendsResponse(BaseModel):
    material_id: Optional[str]
    time_window: str
    window_start: str
    window_end: str
    total_feedback_count: int
    positive_count: int
    neutral_count: int
    negative_count: int
    average_rating: Optional[float]
    average_confidence: Optional[float]
    aspect_scores: dict
    top_positive_phrases: List[str]
    top_negative_phrases: List[str]


@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(
    request: SubmitFeedbackRequest,
    supabase_client: SupabaseClient = Depends(get_supabase_client)
) -> FeedbackResponse:
    """
    Submit user feedback with automatic sentiment analysis
    
    This endpoint:
    1. Accepts user feedback text and optional rating
    2. Performs AI-powered sentiment analysis
    3. Stores feedback with sentiment results in database
    4. Returns complete feedback record with analysis
    """
    try:
        # Get material name for context (if material_id provided)
        material_name = None
        if request.material_id:
            material_response = supabase_client.client.from_("products").select("name").eq("id", request.material_id).single().execute()
            if material_response.data:
                material_name = material_response.data.get("name")
        
        # Perform sentiment analysis
        sentiment_result = await sentiment_service.analyze_feedback(
            feedback_text=request.feedback_text,
            material_name=material_name,
            rating=request.rating
        )
        
        # Insert feedback into database
        feedback_data = {
            "workspace_id": request.workspace_id,
            "user_id": request.user_id,
            "material_id": request.material_id,
            "feedback_text": request.feedback_text,
            "feedback_type": request.feedback_type,
            "rating": request.rating,
            "sentiment_analysis": sentiment_result,
            "is_verified": request.is_verified,
            "is_public": request.is_public,
            "context": request.context or {},
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
        response = supabase_client.client.from_("user_feedback").insert(feedback_data).execute()
        
        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to save feedback")
        
        feedback_record = response.data[0]
        
        return FeedbackResponse(
            id=feedback_record["id"],
            workspace_id=feedback_record["workspace_id"],
            user_id=feedback_record["user_id"],
            material_id=feedback_record.get("material_id"),
            feedback_text=feedback_record["feedback_text"],
            feedback_type=feedback_record["feedback_type"],
            rating=feedback_record.get("rating"),
            sentiment_analysis=feedback_record["sentiment_analysis"],
            is_verified=feedback_record["is_verified"],
            is_public=feedback_record["is_public"],
            helpful_count=feedback_record.get("helpful_count", 0),
            created_at=feedback_record["created_at"],
            analyzed_at=feedback_record.get("analyzed_at")
        )
        
    except Exception as e:
        logger.error(f"Submit feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/material/{material_id}", response_model=List[FeedbackResponse])
async def get_material_feedback(
    material_id: str,
    workspace_id: str = Query(..., description="Workspace ID"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    sentiment_filter: Optional[str] = Query(None, description="Filter by sentiment: positive, neutral, negative"),
    supabase_client: SupabaseClient = Depends(get_supabase_client)
) -> List[FeedbackResponse]:
    """
    Get all feedback for a specific material

    Returns public feedback with sentiment analysis, sorted by most recent.
    """
    try:
        query = supabase_client.client.from_("user_feedback").select("*").eq("workspace_id", workspace_id).eq("material_id", material_id).eq("is_public", True)

        # Apply sentiment filter if provided
        if sentiment_filter:
            query = query.eq("sentiment_analysis->>sentiment", sentiment_filter)

        response = query.order("created_at", desc=True).range(offset, offset + limit - 1).execute()

        return [
            FeedbackResponse(
                id=record["id"],
                workspace_id=record["workspace_id"],
                user_id=record["user_id"],
                material_id=record.get("material_id"),
                feedback_text=record["feedback_text"],
                feedback_type=record["feedback_type"],
                rating=record.get("rating"),
                sentiment_analysis=record.get("sentiment_analysis", {}),
                is_verified=record["is_verified"],
                is_public=record["is_public"],
                helpful_count=record.get("helpful_count", 0),
                created_at=record["created_at"],
                analyzed_at=record.get("analyzed_at")
            )
            for record in response.data
        ]

    except Exception as e:
        logger.error(f"Get material feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends", response_model=List[SentimentTrendsResponse])
async def get_sentiment_trends(
    workspace_id: str = Query(..., description="Workspace ID"),
    material_id: Optional[str] = Query(None, description="Material ID (optional, for all materials if not provided)"),
    time_window: str = Query("daily", description="Time window: hourly, daily, weekly, monthly"),
    days_back: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    supabase_client: SupabaseClient = Depends(get_supabase_client)
) -> List[SentimentTrendsResponse]:
    """
    Get sentiment trends over time

    Returns aggregated sentiment metrics by time window.
    """
    try:
        response = supabase_client.client.rpc(
            "calculate_sentiment_trends",
            {
                "p_workspace_id": workspace_id,
                "p_material_id": material_id,
                "p_time_window": time_window,
                "p_days_back": days_back
            }
        ).execute()

        return [
            SentimentTrendsResponse(
                material_id=record.get("material_id"),
                time_window=record["time_window"],
                window_start=record["window_start"],
                window_end=record["window_end"],
                total_feedback_count=record["total_feedback_count"],
                positive_count=record["positive_count"],
                neutral_count=record["neutral_count"],
                negative_count=record["negative_count"],
                average_rating=record.get("average_rating"),
                average_confidence=record.get("average_confidence"),
                aspect_scores=record.get("aspect_scores", {}),
                top_positive_phrases=record.get("top_positive_phrases", []),
                top_negative_phrases=record.get("top_negative_phrases", [])
            )
            for record in response.data
        ]

    except Exception as e:
        logger.error(f"Get sentiment trends error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{feedback_id}/helpful")
async def mark_feedback_helpful(
    feedback_id: str,
    supabase_client: SupabaseClient = Depends(get_supabase_client)
) -> dict:
    """
    Mark feedback as helpful (increment helpful_count)
    """
    try:
        # Get current helpful_count
        response = supabase_client.client.from_("user_feedback").select("helpful_count").eq("id", feedback_id).single().execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Feedback not found")

        current_count = response.data.get("helpful_count", 0)

        # Increment helpful_count
        update_response = supabase_client.client.from_("user_feedback").update({"helpful_count": current_count + 1}).eq("id", feedback_id).execute()

        return {
            "success": True,
            "feedback_id": feedback_id,
            "helpful_count": current_count + 1
        }

    except Exception as e:
        logger.error(f"Mark helpful error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def feedback_health_check() -> dict:
    """Health check endpoint for User Feedback API"""
    return {
        "status": "healthy",
        "service": "user-feedback-api",
        "version": "1.0.0",
        "features": {
            "sentiment_analysis": True,
            "aspect_based_analysis": True,
            "trend_tracking": True,
            "public_reviews": True
        },
        "endpoints": {
            "submit_feedback": "/api/feedback/submit",
            "get_material_feedback": "/api/feedback/material/{material_id}",
            "get_sentiment_trends": "/api/feedback/trends",
            "mark_helpful": "/api/feedback/{feedback_id}/helpful"
        }
    }

