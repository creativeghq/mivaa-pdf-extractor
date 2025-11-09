"""
Search Suggestions API Endpoints

This module provides API endpoints for intelligent search suggestions, auto-complete,
trending searches, typo correction, and query expansion.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query as QueryParam
from datetime import datetime, timedelta

from ..schemas.suggestions import (
    AutoCompleteRequest,
    AutoCompleteResponse,
    TrendingSearchesRequest,
    TrendingSearchesResponse,
    TypoCorrectionRequest,
    TypoCorrectionResponse,
    QueryExpansionRequest,
    QueryExpansionResponse,
    PopularSearchesRequest,
    PopularSearchesResponse,
    PersonalizedSuggestionsRequest,
    PersonalizedSuggestionsResponse,
    SuggestionClickRequest,
    SuggestionClickResponse,
    SearchSuggestion,
    TrendingSearch,
    PopularSearch
)
from ..services.search_suggestions_service import SearchSuggestionsService
from ..services.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/search", tags=["Search Suggestions"])

# Initialize services
supabase_client = SupabaseClient()


async def get_suggestions_service() -> SearchSuggestionsService:
    """Dependency to get search suggestions service."""
    if not supabase_client.client:
        raise HTTPException(
            status_code=503,
            detail="Database service is not available"
        )
    return SearchSuggestionsService(supabase_client)


@router.post(
    "/autocomplete",
    response_model=AutoCompleteResponse,
    summary="Get auto-complete suggestions",
    description="Get intelligent auto-complete suggestions as user types"
)
async def autocomplete(
    request: AutoCompleteRequest,
    service: SearchSuggestionsService = Depends(get_suggestions_service)
) -> AutoCompleteResponse:
    """
    Get auto-complete suggestions for a partial query.
    
    This endpoint provides:
    - Real-time suggestions as user types
    - Trending searches
    - Recent user searches
    - Popular searches
    - Product/material name matches
    """
    try:
        start_time = datetime.now()
        
        suggestions, metadata = await service.get_autocomplete_suggestions(
            query=request.query,
            limit=request.limit,
            user_id=request.user_id,
            session_id=request.session_id,
            include_trending=request.include_trending,
            include_recent=request.include_recent,
            include_popular=request.include_popular,
            categories=request.categories
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AutoCompleteResponse(
            success=True,
            query=request.query,
            suggestions=suggestions,
            total_suggestions=len(suggestions),
            processing_time_ms=int(processing_time),
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Auto-complete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/trending",
    response_model=TrendingSearchesResponse,
    summary="Get trending searches",
    description="Get currently trending search queries"
)
async def get_trending_searches(
    time_window: str = QueryParam("daily", description="Time window: hourly, daily, weekly, monthly"),
    limit: int = QueryParam(20, ge=1, le=100, description="Maximum number of results"),
    category: Optional[str] = QueryParam(None, description="Filter by category"),
    min_search_count: int = QueryParam(2, ge=1, description="Minimum search count"),
    service: SearchSuggestionsService = Depends(get_suggestions_service)
) -> TrendingSearchesResponse:
    """
    Get trending search queries.
    
    This endpoint provides:
    - Trending searches by time window
    - Growth rate analysis
    - Category filtering
    - Unique user counts
    """
    try:
        # Calculate time window
        now = datetime.now()
        if time_window == "hourly":
            window_start = now - timedelta(hours=1)
        elif time_window == "daily":
            window_start = now - timedelta(days=1)
        elif time_window == "weekly":
            window_start = now - timedelta(weeks=1)
        elif time_window == "monthly":
            window_start = now - timedelta(days=30)
        else:
            window_start = now - timedelta(days=1)
        
        # Get trending searches from database function
        response = supabase_client.client.rpc(
            "calculate_trending_searches",
            {
                "p_time_window": time_window,
                "p_limit": limit
            }
        ).execute()
        
        trending_searches = []
        if response.data:
            for row in response.data:
                trending_searches.append(TrendingSearch(
                    query_text=row["query_text"],
                    search_count=row["search_count"],
                    unique_users_count=row["unique_users"],
                    trend_score=float(row["trend_score"]),
                    growth_rate=float(row["growth_rate"]),
                    time_window=time_window,
                    category=category,
                    metadata={}
                ))
        
        # Filter by category if specified
        if category:
            trending_searches = [
                ts for ts in trending_searches
                if ts.category == category
            ]
        
        # Filter by minimum search count
        trending_searches = [
            ts for ts in trending_searches
            if ts.search_count >= min_search_count
        ]
        
        return TrendingSearchesResponse(
            success=True,
            trending_searches=trending_searches,
            total_results=len(trending_searches),
            time_window=time_window,
            window_start=window_start,
            window_end=now,
            metadata={"calculation_method": "database_function"}
        )
        
    except Exception as e:
        logger.error(f"Trending searches error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/typo-correction",
    response_model=TypoCorrectionResponse,
    summary="Check for typos and suggest corrections",
    description="Detect typos and suggest corrected queries"
)
async def check_typo_correction(
    request: TypoCorrectionRequest,
    service: SearchSuggestionsService = Depends(get_suggestions_service)
) -> TypoCorrectionResponse:
    """
    Check query for typos and suggest corrections.
    
    This endpoint provides:
    - Spelling correction
    - Abbreviation expansion
    - Fuzzy matching against popular searches
    - Confidence scoring
    """
    try:
        has_corrections, corrections, recommended = await service.check_typos(
            query=request.query,
            auto_apply_threshold=request.auto_apply_threshold,
            max_suggestions=request.max_suggestions
        )
        
        return TypoCorrectionResponse(
            success=True,
            original_query=request.query,
            has_corrections=has_corrections,
            corrections=corrections,
            recommended_correction=recommended,
            metadata={
                "total_corrections": len(corrections),
                "auto_apply_threshold": request.auto_apply_threshold
            }
        )
        
    except Exception as e:
        logger.error(f"Typo correction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/query-expansion",
    response_model=QueryExpansionResponse,
    summary="Expand query with synonyms and related terms",
    description="Expand search query with synonyms and related concepts"
)
async def expand_query(
    request: QueryExpansionRequest,
    service: SearchSuggestionsService = Depends(get_suggestions_service)
) -> QueryExpansionResponse:
    """
    Expand query with synonyms and related terms.
    
    This endpoint provides:
    - Synonym expansion
    - Related concept discovery
    - AI-powered semantic expansion
    - Suggested search variations
    """
    try:
        start_time = datetime.now()
        
        expanded_query = await service.expand_query(
            query=request.query,
            max_synonyms_per_term=request.max_synonyms_per_term,
            max_related_concepts=request.max_related_concepts,
            use_ai=request.use_ai
        )
        
        # Generate suggested search variations
        suggested_searches = []
        for term in expanded_query.expanded_terms[:5]:
            suggested_searches.append(f"{request.query} {term}")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryExpansionResponse(
            success=True,
            expanded_query=expanded_query,
            suggested_searches=suggested_searches,
            processing_time_ms=int(processing_time),
            metadata={
                "use_ai": request.use_ai,
                "total_synonyms": sum(len(syns) for syns in expanded_query.synonyms.values()),
                "total_related": len(expanded_query.related_concepts)
            }
        )
        
    except Exception as e:
        logger.error(f"Query expansion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/track-click",
    response_model=SuggestionClickResponse,
    summary="Track suggestion click",
    description="Track when user clicks on a search suggestion"
)
async def track_suggestion_click(
    request: SuggestionClickRequest,
    service: SearchSuggestionsService = Depends(get_suggestions_service)
) -> SuggestionClickResponse:
    """
    Track suggestion click for analytics.
    
    This endpoint tracks:
    - Suggestion clicks
    - User satisfaction
    - Result counts
    - Position in suggestion list
    """
    try:
        success = await service.track_suggestion_click(
            suggestion_id=request.suggestion_id,
            user_id=request.user_id,
            session_id=request.session_id,
            original_query=request.original_query,
            suggestion_position=request.suggestion_position,
            action_type=request.action_type,
            result_count=request.result_count,
            user_satisfied=request.user_satisfied
        )
        
        if success:
            return SuggestionClickResponse(
                success=True,
                message="Click tracked successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to track click")
            
    except Exception as e:
        logger.error(f"Track click error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

