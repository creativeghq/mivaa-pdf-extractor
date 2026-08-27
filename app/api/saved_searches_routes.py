"""
Saved Searches API Endpoints

This module provides endpoints for managing user saved searches with
AI-powered deduplication to prevent database bloat while respecting
important contextual differences.

Features:
- CRUD operations for saved searches
- Smart deduplication with Claude Haiku 4.5
- Merge suggestions and execution
- Search execution tracking
- Similar search discovery

AUTH (audit #22 M9-4). Every route here took `user_id` from the request — a body field
or a query parameter — and no route had a gate of any kind. So the caller named whose
saved searches to read, edit, merge or delete, and `verify_user_access` faithfully
verified the id against itself.

It was invisible because the module was ALSO dead: `search_deduplication_service` had a
double `.client` unwrap, so /check-duplicates and /merge raised on every call. Repairing
that typo without binding the identity in the same change is what would have made the
BOLA live — the trap #27 named for this module by name. Both land together.

`user_id` now comes from `current_user_id(get_current_user(...))` on all eight routes and
appears in no request model. The response still carries it: that is a fact about the row,
not a claim by the caller.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ..services.search.search_deduplication_service import (
    get_deduplication_service,
    SearchDeduplicationService
)
from ..dependencies import current_user_id, get_current_user
from ..services.core.supabase_client import get_supabase_client
from ..utils.timestamp_utils import normalize_timestamp

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/saved-searches", tags=["Saved Searches"])


# ============================================================================
# Request/Response Models
# ============================================================================

class MaterialFilters(BaseModel):
    """Material-specific filters for search."""
    materialTypes: Optional[List[str]] = Field(default=None, description="Material types filter")
    colors: Optional[List[str]] = Field(default=None, description="Color filter")
    suppliers: Optional[List[str]] = Field(default=None, description="Supplier filter")
    applications: Optional[List[str]] = Field(default=None, description="Application filter")
    textures: Optional[List[str]] = Field(default=None, description="Texture filter")
    priceRange: Optional[List[float]] = Field(default=None, description="Price range [min, max]")


class CreateSavedSearchRequest(BaseModel):
    """Request to create a new saved search.

    No `user_id`: it comes from the token (#22 M9-4).
    """
    query: str = Field(..., description="Search query text")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="General filters")
    material_filters: Optional[MaterialFilters] = Field(default=None, description="Material-specific filters")
    integration_context: Optional[str] = Field(default=None, description="Integration context (chat, moodboard, 3d_generation)")
    integration_id: Optional[str] = Field(default=None, description="Related integration ID")
    name: Optional[str] = Field(default=None, description="Custom name for the search")
    description: Optional[str] = Field(default=None, description="Description of the search")
    check_for_duplicates: bool = Field(default=True, description="Enable smart deduplication")


class UpdateSavedSearchRequest(BaseModel):
    """Request to update a saved search."""
    name: Optional[str] = Field(default=None, description="New name")
    description: Optional[str] = Field(default=None, description="New description")
    query: Optional[str] = Field(default=None, description="New query")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="New filters")
    material_filters: Optional[MaterialFilters] = Field(default=None, description="New material filters")


class CheckDuplicatesRequest(BaseModel):
    """Request to check for duplicate searches. No `user_id` — see the module docstring."""
    query: str = Field(..., description="Search query")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="General filters")
    material_filters: Optional[MaterialFilters] = Field(default=None, description="Material filters")


class MergeSearchRequest(BaseModel):
    """Request to merge a search into existing one. No `user_id` — see the module docstring."""
    new_query: str = Field(..., description="New query to merge")
    new_filters: Optional[Dict[str, Any]] = Field(default=None, description="New filters")
    new_material_filters: Optional[MaterialFilters] = Field(default=None, description="New material filters")


class SavedSearchResponse(BaseModel):
    """Response model for saved search."""
    id: str
    user_id: str
    query: str
    name: Optional[str]
    description: Optional[str]
    filters: Optional[Dict[str, Any]]
    material_filters: Optional[Dict[str, Any]]
    integration_context: Optional[str]
    integration_id: Optional[str]
    use_count: int
    last_executed_at: Optional[datetime]
    relevance_score: Optional[float]
    core_material: Optional[str]
    material_attributes: Optional[Dict[str, Any]]
    application_context: Optional[str]
    merge_count: int
    created_at: datetime
    updated_at: datetime


class MergeSuggestion(BaseModel):
    """Merge suggestion response."""
    existing_search: SavedSearchResponse
    similarity_score: float
    reason: str
    new_query: str


class CheckDuplicatesResponse(BaseModel):
    """Response for duplicate check."""
    has_duplicate: bool
    should_auto_merge: bool
    merge_suggestion: Optional[MergeSuggestion]


# ============================================================================
# Helper Functions
# ============================================================================

def get_dedup_service() -> SearchDeduplicationService:
    """Dependency to get deduplication service."""
    return get_deduplication_service()


async def verify_user_access(user_id: str, search_id: str) -> Dict:
    """Verify user has access to the search."""
    supabase = get_supabase_client().client

    response = supabase.table("saved_searches").select("*").eq(
        "id", search_id
    ).eq(
        "user_id", user_id
    ).single().execute()

    if not response.data:
        raise HTTPException(
            status_code=404,
            detail=f"Saved search {search_id} not found or access denied"
        )

    return response.data


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/check-duplicates", response_model=CheckDuplicatesResponse)
async def check_for_duplicates(
    request: CheckDuplicatesRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    dedup_service: SearchDeduplicationService = Depends(get_dedup_service),
):
    """
    Check if a search query has duplicates and get merge suggestions.
    
    Uses AI-powered semantic analysis to find similar searches while
    respecting contextual differences (floor vs wall, indoor vs outdoor).
    
    Returns:
    - has_duplicate: Whether a similar search exists
    - should_auto_merge: Whether to auto-merge (95%+ similarity)
    - merge_suggestion: Details about the suggested merge (if applicable)
    """
    try:
        logger.info(f"Checking duplicates for user {current_user_id(user)}: {request.query}")
        
        # Convert material_filters to dict
        material_filters_dict = request.material_filters.dict() if request.material_filters else {}
        
        # Find or merge search
        existing_id, should_merge, merge_suggestion = await dedup_service.find_or_merge_search(
            user_id=current_user_id(user),
            query=request.query,
            filters=request.filters or {},
            material_filters=material_filters_dict
        )
        
        if not existing_id:
            return CheckDuplicatesResponse(
                has_duplicate=False,
                should_auto_merge=False,
                merge_suggestion=None
            )
        
        # Get existing search details
        existing_search = await verify_user_access(current_user_id(user), existing_id)
        
        if should_merge:
            # Auto-merge case (95%+ similarity)
            return CheckDuplicatesResponse(
                has_duplicate=True,
                should_auto_merge=True,
                merge_suggestion=MergeSuggestion(
                    existing_search=SavedSearchResponse(**existing_search),
                    similarity_score=0.95,  # Minimum for auto-merge
                    reason="Very high similarity - auto-merge recommended",
                    new_query=request.query
                )
            )
        
        if merge_suggestion:
            # Suggest merge to user (85-95% similarity)
            return CheckDuplicatesResponse(
                has_duplicate=True,
                should_auto_merge=False,
                merge_suggestion=MergeSuggestion(
                    existing_search=SavedSearchResponse(**merge_suggestion["existing_search"]),
                    similarity_score=merge_suggestion["similarity_score"],
                    reason=merge_suggestion["reason"],
                    new_query=merge_suggestion["new_query"]
                )
            )
        
        return CheckDuplicatesResponse(
            has_duplicate=False,
            should_auto_merge=False,
            merge_suggestion=None
        )
        
    except Exception as e:
        logger.error(f"Error checking duplicates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{search_id}/merge", response_model=SavedSearchResponse)
async def merge_into_existing(
    search_id: str,
    request: MergeSearchRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    dedup_service: SearchDeduplicationService = Depends(get_dedup_service),
):
    """
    Merge a new search into an existing saved search.
    
    Strategy:
    - Keeps most specific query
    - Merges attributes (union, no conflicts)
    - Updates filters to be more inclusive
    - Increments merge_count
    - Updates last_merged_at timestamp
    """
    try:
        logger.info(f"Merging search into {search_id} for user {current_user_id(user)}")
        
        # Verify user has access
        await verify_user_access(current_user_id(user), search_id)
        
        # Analyze new query
        analysis = await dedup_service.analyze_search_query(request.new_query)
        
        # Convert material_filters to dict
        material_filters_dict = request.new_material_filters.dict() if request.new_material_filters else {}
        
        # Execute merge
        merged_id = await dedup_service.merge_into_existing(
            existing_id=search_id,
            new_query=request.new_query,
            new_filters=request.new_filters or {},
            new_material_filters=material_filters_dict,
            analysis=analysis
        )
        
        # Get updated search
        updated_search = await verify_user_access(current_user_id(user), merged_id)
        
        return SavedSearchResponse(**updated_search)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error merging search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=SavedSearchResponse, status_code=201)
async def create_saved_search(
    request: CreateSavedSearchRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    dedup_service: SearchDeduplicationService = Depends(get_dedup_service),
):
    """
    Create a new saved search with optional deduplication.

    If check_for_duplicates=True (default), will check for similar searches
    and return existing one if found. Otherwise creates new search.
    """
    try:
        logger.info(f"Creating saved search for user {current_user_id(user)}: {request.query}")

        supabase = get_supabase_client().client

        # Convert material_filters to dict
        material_filters_dict = request.material_filters.dict() if request.material_filters else {}

        # Check for duplicates if enabled
        if request.check_for_duplicates:
            existing_id, should_merge, merge_suggestion = await dedup_service.find_or_merge_search(
                user_id=current_user_id(user),
                query=request.query,
                filters=request.filters or {},
                material_filters=material_filters_dict
            )

            if should_merge and existing_id:
                # Auto-merge: return existing search
                logger.info(f"Auto-merging into existing search {existing_id}")

                # Analyze query for merge
                analysis = await dedup_service.analyze_search_query(request.query)

                # Execute merge
                await dedup_service.merge_into_existing(
                    existing_id=existing_id,
                    new_query=request.query,
                    new_filters=request.filters or {},
                    new_material_filters=material_filters_dict,
                    analysis=analysis
                )

                # Return merged search
                existing_search = await verify_user_access(current_user_id(user), existing_id)
                return SavedSearchResponse(**existing_search)

        # Analyze query with AI
        analysis = await dedup_service.analyze_search_query(request.query)

        # Create new search
        search_data = {
            "user_id": current_user_id(user),
            "query": request.query,
            "name": request.name,
            "description": request.description,
            "filters": request.filters,
            "material_filters": material_filters_dict,
            "integration_context": request.integration_context,
            "integration_id": request.integration_id,
            "semantic_fingerprint": analysis.semantic_fingerprint,
            "normalized_query": analysis.normalized_query,
            "core_material": analysis.core_material,
            "material_attributes": analysis.attributes,
            "application_context": analysis.application_context,
            "intent_category": analysis.intent_category,
            "use_count": 0,
            "merge_count": 1,
            "relevance_score": 1.0
        }

        response = supabase.table("saved_searches").insert(search_data).execute()

        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to create saved search")

        return SavedSearchResponse(**response.data[0])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating saved search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[SavedSearchResponse])
async def get_user_saved_searches(
    user: Dict[str, Any] = Depends(get_current_user),
    integration_context: Optional[str] = Query(None, description="Filter by integration context"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """
    Get all saved searches for a user with optional filtering.

    Supports:
    - Filtering by integration context (chat, moodboard, 3d_generation)
    - Pagination with limit/offset
    - Sorted by last_executed_at (most recent first)
    """
    try:
        logger.info(f"Getting saved searches for user {current_user_id(user)}")

        supabase = get_supabase_client().client

        # Build query
        query = supabase.table("saved_searches").select("*").eq("user_id", current_user_id(user))

        if integration_context:
            query = query.eq("integration_context", integration_context)

        # Order by last executed (most recent first), then by created_at
        query = query.order("last_executed_at", desc=True, nullsfirst=False).order(
            "created_at", desc=True
        ).range(offset, offset + limit - 1)

        response = query.execute()

        searches = [SavedSearchResponse(**search) for search in response.data] if response.data else []

        return searches

    except Exception as e:
        logger.error(f"Error getting saved searches: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{search_id}", response_model=SavedSearchResponse)
async def get_saved_search(
    search_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Get a specific saved search by ID."""
    try:
        search = await verify_user_access(current_user_id(user), search_id)
        return SavedSearchResponse(**search)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting saved search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{search_id}", response_model=SavedSearchResponse)
async def update_saved_search(
    search_id: str,
    request: UpdateSavedSearchRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Update a saved search."""
    try:
        logger.info(f"Updating saved search {search_id} for user {current_user_id(user)}")

        # Verify access
        await verify_user_access(current_user_id(user), search_id)

        # Build update data (only include provided fields)
        update_data = {}
        if request.name is not None:
            update_data["name"] = request.name
        if request.description is not None:
            update_data["description"] = request.description
        if request.query is not None:
            update_data["query"] = request.query
        if request.filters is not None:
            update_data["filters"] = request.filters
        if request.material_filters is not None:
            update_data["material_filters"] = request.material_filters.dict()

        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")

        update_data["updated_at"] = datetime.utcnow().isoformat()

        supabase = get_supabase_client().client
        response = supabase.table("saved_searches").update(update_data).eq(
            "id", search_id
        ).eq("user_id", current_user_id(user)).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Saved search not found")

        return SavedSearchResponse(**response.data[0])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating saved search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{search_id}", status_code=204)
async def delete_saved_search(
    search_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Delete a saved search."""
    try:
        logger.info(f"Deleting saved search {search_id} for user {current_user_id(user)}")

        # Verify access
        await verify_user_access(current_user_id(user), search_id)

        supabase = get_supabase_client().client
        supabase.table("saved_searches").delete().eq("id", search_id).eq(
            "user_id", current_user_id(user)
        ).execute()

        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting saved search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{search_id}/execute", response_model=SavedSearchResponse)
async def execute_saved_search(
    search_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Execute a saved search and track usage.

    Updates:
    - use_count (incremented)
    - last_executed_at (current timestamp)
    - relevance_score (based on usage patterns)
    """
    try:
        logger.info(f"Executing saved search {search_id} for user {current_user_id(user)}")

        # Verify access
        search = await verify_user_access(current_user_id(user), search_id)

        # Update usage tracking
        supabase = get_supabase_client().client
        update_data = {
            "use_count": search["use_count"] + 1,
            "last_executed_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        # Calculate relevance score based on usage
        # More recent and frequent usage = higher relevance
        days_since_created = (datetime.utcnow() - datetime.fromisoformat(normalize_timestamp(search["created_at"]))).days
        usage_frequency = (search["use_count"] + 1) / max(days_since_created, 1)
        update_data["relevance_score"] = min(usage_frequency * 10, 10.0)  # Cap at 10.0

        response = supabase.table("saved_searches").update(update_data).eq(
            "id", search_id
        ).eq("user_id", current_user_id(user)).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Saved search not found")

        return SavedSearchResponse(**response.data[0])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing saved search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


