"""
Search Suggestions and Auto-Complete Schemas

This module defines Pydantic models for search suggestions, auto-complete,
trending searches, typo correction, and query expansion features.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================================
# Search Suggestion Models
# ============================================================================

class SearchSuggestion(BaseModel):
    """Individual search suggestion."""
    
    id: str = Field(..., description="Unique suggestion ID")
    suggestion_text: str = Field(..., description="Suggested query text")
    suggestion_type: str = Field(..., description="Type: product, material, category, property, trending, recent")
    category: Optional[str] = Field(None, description="Category of the suggestion")
    popularity_score: float = Field(0.0, description="Popularity score (0-1)")
    click_count: int = Field(0, description="Number of times clicked")
    impression_count: int = Field(0, description="Number of times shown")
    ctr: float = Field(0.0, description="Click-through rate")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    confidence: Optional[float] = Field(None, description="Confidence score for AI-generated suggestions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "suggestion_text": "fire resistant tiles",
                "suggestion_type": "trending",
                "category": "ceramic",
                "popularity_score": 0.85,
                "click_count": 142,
                "impression_count": 1250,
                "ctr": 0.1136,
                "metadata": {"related_terms": ["fireproof", "heat resistant"]},
                "confidence": 0.92
            }
        }


class AutoCompleteRequest(BaseModel):
    """Request for auto-complete suggestions."""
    
    query: str = Field(..., min_length=1, max_length=200, description="Partial query text")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of suggestions")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    include_trending: bool = Field(True, description="Include trending searches")
    include_recent: bool = Field(True, description="Include recent searches")
    include_popular: bool = Field(True, description="Include popular searches")
    categories: Optional[List[str]] = Field(None, description="Filter by categories")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "fire res",
                "limit": 10,
                "user_id": "user-123",
                "session_id": "session-456",
                "include_trending": True,
                "include_recent": True,
                "include_popular": True,
                "categories": ["ceramic", "tiles"]
            }
        }


class AutoCompleteResponse(BaseModel):
    """Response with auto-complete suggestions."""
    
    success: bool = Field(True, description="Whether the request was successful")
    query: str = Field(..., description="Original query")
    suggestions: List[SearchSuggestion] = Field(default_factory=list, description="List of suggestions")
    total_suggestions: int = Field(0, description="Total number of suggestions")
    processing_time_ms: int = Field(0, description="Processing time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ============================================================================
# Trending Searches Models
# ============================================================================

class TrendingSearch(BaseModel):
    """Trending search query."""
    
    query_text: str = Field(..., description="Trending query text")
    search_count: int = Field(0, description="Number of searches in time window")
    unique_users_count: int = Field(0, description="Number of unique users")
    trend_score: float = Field(0.0, description="Calculated trend score")
    growth_rate: float = Field(0.0, description="Growth rate percentage")
    time_window: str = Field(..., description="Time window: hourly, daily, weekly")
    category: Optional[str] = Field(None, description="Category of the search")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TrendingSearchesRequest(BaseModel):
    """Request for trending searches."""
    
    time_window: str = Field("daily", description="Time window: hourly, daily, weekly, monthly")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of results")
    category: Optional[str] = Field(None, description="Filter by category")
    min_search_count: int = Field(2, ge=1, description="Minimum search count threshold")


class TrendingSearchesResponse(BaseModel):
    """Response with trending searches."""
    
    success: bool = Field(True, description="Whether the request was successful")
    trending_searches: List[TrendingSearch] = Field(default_factory=list, description="List of trending searches")
    total_results: int = Field(0, description="Total number of trending searches")
    time_window: str = Field(..., description="Time window used")
    window_start: datetime = Field(..., description="Start of time window")
    window_end: datetime = Field(..., description="End of time window")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ============================================================================
# Query Correction Models
# ============================================================================

class QueryCorrection(BaseModel):
    """Query typo correction."""
    
    original_query: str = Field(..., description="Original query with potential typo")
    corrected_query: str = Field(..., description="Corrected query")
    correction_type: str = Field(..., description="Type: spelling, synonym, expansion, abbreviation")
    confidence_score: float = Field(0.8, ge=0.0, le=1.0, description="Confidence in correction")
    auto_applied: bool = Field(False, description="Whether correction was auto-applied")
    acceptance_rate: float = Field(0.0, description="Historical acceptance rate")


class TypoCorrectionRequest(BaseModel):
    """Request for typo correction."""
    
    query: str = Field(..., min_length=1, max_length=200, description="Query to check for typos")
    auto_apply_threshold: float = Field(0.9, ge=0.0, le=1.0, description="Confidence threshold for auto-apply")
    max_suggestions: int = Field(3, ge=1, le=10, description="Maximum correction suggestions")


class TypoCorrectionResponse(BaseModel):
    """Response with typo corrections."""
    
    success: bool = Field(True, description="Whether the request was successful")
    original_query: str = Field(..., description="Original query")
    has_corrections: bool = Field(False, description="Whether corrections were found")
    corrections: List[QueryCorrection] = Field(default_factory=list, description="List of corrections")
    recommended_correction: Optional[QueryCorrection] = Field(None, description="Highest confidence correction")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ============================================================================
# Query Expansion Models
# ============================================================================

class ExpandedQuery(BaseModel):
    """Expanded query with synonyms and related terms."""
    
    original_query: str = Field(..., description="Original query")
    expanded_terms: List[str] = Field(default_factory=list, description="Expanded query terms")
    synonyms: Dict[str, List[str]] = Field(default_factory=dict, description="Synonyms for each term")
    related_concepts: List[str] = Field(default_factory=list, description="Related concepts")
    confidence_score: float = Field(0.8, description="Confidence in expansion")


class QueryExpansionRequest(BaseModel):
    """Request for query expansion."""
    
    query: str = Field(..., min_length=1, max_length=200, description="Query to expand")
    max_synonyms_per_term: int = Field(3, ge=1, le=10, description="Max synonyms per term")
    max_related_concepts: int = Field(5, ge=1, le=20, description="Max related concepts")
    use_ai: bool = Field(True, description="Use AI for expansion")


class QueryExpansionResponse(BaseModel):
    """Response with expanded query."""
    
    success: bool = Field(True, description="Whether the request was successful")
    expanded_query: ExpandedQuery = Field(..., description="Expanded query details")
    suggested_searches: List[str] = Field(default_factory=list, description="Suggested search variations")
    processing_time_ms: int = Field(0, description="Processing time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ============================================================================
# Popular Searches Models
# ============================================================================

class PopularSearch(BaseModel):
    """Popular search query."""
    
    query_text: str = Field(..., description="Popular query text")
    search_count: int = Field(0, description="Total search count")
    unique_users_count: int = Field(0, description="Number of unique users")
    avg_result_count: float = Field(0.0, description="Average number of results")
    avg_satisfaction: float = Field(0.0, description="Average satisfaction rating")
    category: Optional[str] = Field(None, description="Category of the search")
    last_searched_at: datetime = Field(..., description="Last time this was searched")


class PopularSearchesRequest(BaseModel):
    """Request for popular searches."""
    
    time_period: str = Field("30d", description="Time period: 7d, 30d, 90d, all")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of results")
    category: Optional[str] = Field(None, description="Filter by category")
    min_search_count: int = Field(5, ge=1, description="Minimum search count threshold")


class PopularSearchesResponse(BaseModel):
    """Response with popular searches."""
    
    success: bool = Field(True, description="Whether the request was successful")
    popular_searches: List[PopularSearch] = Field(default_factory=list, description="List of popular searches")
    total_results: int = Field(0, description="Total number of popular searches")
    time_period: str = Field(..., description="Time period used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ============================================================================
# Personalized Suggestions Models
# ============================================================================

class PersonalizedSuggestionsRequest(BaseModel):
    """Request for personalized search suggestions."""
    
    user_id: str = Field(..., description="User ID for personalization")
    query: Optional[str] = Field(None, description="Optional partial query")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of suggestions")
    include_history: bool = Field(True, description="Include user's search history")
    include_preferences: bool = Field(True, description="Include user preferences")


class PersonalizedSuggestionsResponse(BaseModel):
    """Response with personalized suggestions."""
    
    success: bool = Field(True, description="Whether the request was successful")
    user_id: str = Field(..., description="User ID")
    suggestions: List[SearchSuggestion] = Field(default_factory=list, description="Personalized suggestions")
    total_suggestions: int = Field(0, description="Total number of suggestions")
    personalization_factors: Dict[str, Any] = Field(default_factory=dict, description="Factors used for personalization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ============================================================================
# Analytics Models
# ============================================================================

class SuggestionClickRequest(BaseModel):
    """Request to track suggestion click."""
    
    suggestion_id: str = Field(..., description="Suggestion ID that was clicked")
    user_id: Optional[str] = Field(None, description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    original_query: str = Field(..., description="Original query before suggestion")
    suggestion_position: int = Field(..., ge=0, description="Position in suggestion list")
    action_type: str = Field(..., description="Action: clicked, dismissed, ignored, accepted")
    result_count: Optional[int] = Field(None, description="Number of results after clicking")
    user_satisfied: Optional[bool] = Field(None, description="Whether user was satisfied")


class SuggestionClickResponse(BaseModel):
    """Response after tracking suggestion click."""
    
    success: bool = Field(True, description="Whether tracking was successful")
    message: str = Field("Click tracked successfully", description="Response message")


