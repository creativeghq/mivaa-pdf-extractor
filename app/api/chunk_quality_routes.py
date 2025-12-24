"""
Chunk Quality Metrics API Routes

Provides endpoints for monitoring chunk quality, deduplication, and flagged content.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.services.supabase_client import get_supabase_client, SupabaseClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/chunk-quality", tags=["Chunk Quality"])


class ChunkQualityMetrics(BaseModel):
    """Chunk quality metrics response model"""
    total_chunks: int
    total_documents: int
    exact_duplicates_prevented: int
    semantic_duplicates_prevented: int
    low_quality_rejected: int
    borderline_quality_flagged: int
    average_quality_score: float
    quality_distribution: dict
    flagged_chunks_pending_review: int
    flagged_chunks_reviewed: int


class FlaggedChunk(BaseModel):
    """Flagged chunk model"""
    id: str
    chunk_id: str
    document_id: str
    flag_type: str
    flag_reason: str
    quality_score: float
    content_preview: str
    flagged_at: str
    reviewed: bool
    reviewed_at: Optional[str] = None
    reviewed_by: Optional[str] = None
    review_action: Optional[str] = None


@router.get("/metrics", response_model=ChunkQualityMetrics)
async def get_chunk_quality_metrics(
    days: int = Query(default=30, ge=1, le=365),
    workspace_id: Optional[str] = None,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Get chunk quality metrics for the specified time period.
    
    Args:
        days: Number of days to look back (default: 30)
        workspace_id: Optional workspace filter
    
    Returns:
        ChunkQualityMetrics with aggregated statistics
    """
    try:
        # Calculate time range
        start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        # Get total chunks
        chunks_query = supabase.client.table("document_chunks").select("id, quality_score", count="exact")
        if workspace_id:
            chunks_query = chunks_query.eq("workspace_id", workspace_id)
        chunks_query = chunks_query.gte("created_at", start_date)
        chunks_response = chunks_query.execute()
        
        total_chunks = chunks_response.count or 0
        chunks_data = chunks_response.data or []
        
        # Calculate average quality score
        quality_scores = [float(chunk.get("quality_score", 0) or 0) for chunk in chunks_data if chunk.get("quality_score")]
        average_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Get total documents
        docs_query = supabase.client.table("documents").select("id", count="exact")
        if workspace_id:
            docs_query = docs_query.eq("workspace_id", workspace_id)
        docs_query = docs_query.gte("created_at", start_date)
        docs_response = docs_query.execute()
        total_documents = docs_response.count or 0
        
        # Get flagged chunks
        flagged_query = supabase.client.table("chunk_quality_flags").select("*", count="exact")
        if workspace_id:
            flagged_query = flagged_query.eq("workspace_id", workspace_id)
        flagged_query = flagged_query.gte("flagged_at", start_date)
        flagged_response = flagged_query.execute()
        
        flagged_data = flagged_response.data or []
        flagged_chunks_pending_review = sum(1 for f in flagged_data if not f.get("reviewed", False))
        flagged_chunks_reviewed = sum(1 for f in flagged_data if f.get("reviewed", False))
        borderline_quality_flagged = sum(1 for f in flagged_data if f.get("flag_type") == "borderline_quality")
        
        # Calculate quality distribution
        quality_distribution = {
            "excellent": sum(1 for s in quality_scores if s >= 0.9),
            "good": sum(1 for s in quality_scores if 0.7 <= s < 0.9),
            "fair": sum(1 for s in quality_scores if 0.5 <= s < 0.7),
            "poor": sum(1 for s in quality_scores if s < 0.5),
        }
        
        return ChunkQualityMetrics(
            total_chunks=total_chunks,
            total_documents=total_documents,
            exact_duplicates_prevented=0,  # TODO: Track this in processing
            semantic_duplicates_prevented=0,  # TODO: Track this in processing
            low_quality_rejected=0,  # TODO: Track this in processing
            borderline_quality_flagged=borderline_quality_flagged,
            average_quality_score=round(average_quality_score, 3),
            quality_distribution=quality_distribution,
            flagged_chunks_pending_review=flagged_chunks_pending_review,
            flagged_chunks_reviewed=flagged_chunks_reviewed
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch chunk quality metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch metrics: {str(e)}")


@router.get("/flagged", response_model=List[FlaggedChunk])
async def get_flagged_chunks(
    workspace_id: Optional[str] = None,
    document_id: Optional[str] = None,
    reviewed: Optional[bool] = None,
    limit: int = Query(default=50, ge=1, le=100),
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Get list of flagged chunks for review.

    Filters:
    - workspace_id: Filter by workspace
    - document_id: Filter by document
    - reviewed: Filter by review status (None = all, True = reviewed, False = pending)
    - limit: Maximum number of results
    """
    try:
        query = supabase.client.table("chunk_quality_flags").select("*")

        if workspace_id:
            query = query.eq("workspace_id", workspace_id)
        if document_id:
            query = query.eq("document_id", document_id)
        if reviewed is not None:
            query = query.eq("reviewed", reviewed)

        query = query.order("flagged_at", desc=True).limit(limit)
        response = query.execute()

        return response.data or []

    except Exception as e:
        logger.error(f"Failed to fetch flagged chunks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch flagged chunks: {str(e)}")


@router.post("/flagged/{flag_id}/review")
async def review_flagged_chunk(
    flag_id: str,
    action: str,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Review a flagged chunk.

    Actions:
    - approve: Mark as reviewed and approved
    - reject: Mark as reviewed and rejected
    - delete_chunk: Delete the chunk entirely
    """
    try:
        if action not in ["approve", "reject", "delete_chunk"]:
            raise HTTPException(status_code=400, detail="Invalid action. Must be 'approve', 'reject', or 'delete_chunk'")

        # Update flag record
        update_data = {
            "reviewed": True,
            "reviewed_at": datetime.utcnow().isoformat(),
            "review_action": action
        }

        response = supabase.client.table("chunk_quality_flags").update(update_data).eq("id", flag_id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Flagged chunk not found")

        # If action is delete_chunk, also delete the actual chunk
        if action == "delete_chunk":
            flag_data = response.data[0]
            chunk_id = flag_data.get("chunk_id")
            if chunk_id:
                supabase.client.table("document_chunks").delete().eq("id", chunk_id).execute()
                logger.info(f"Deleted chunk {chunk_id} as part of review action")

        return {"success": True, "message": f"Chunk {action}d successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to review flagged chunk: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to review chunk: {str(e)}")

