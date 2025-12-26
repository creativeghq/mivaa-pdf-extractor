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


def _generate_chunk_recommendations(
    chunk_size_stats: dict,
    chunk_overlap_stats: dict,
    quality_distribution: dict,
    average_quality_score: float,
    very_small_chunks: int,
    very_large_chunks: int,
    total_chunks: int
) -> List[str]:
    """
    Generate actionable recommendations based on chunk quality metrics.

    Returns:
        List of recommendation strings with specific actions to improve chunk quality
    """
    recommendations = []

    # Check for very small chunks (< 100 chars)
    if very_small_chunks > 0:
        small_chunk_pct = (very_small_chunks / total_chunks * 100) if total_chunks > 0 else 0
        if small_chunk_pct > 5:  # More than 5% are very small
            recommendations.append(
                f"⚠️ HIGH: {very_small_chunks} chunks ({small_chunk_pct:.1f}%) are very small (< 100 chars). "
                f"Consider increasing minimum chunk size to 200 characters or filtering out small chunks."
            )
        elif small_chunk_pct > 2:
            recommendations.append(
                f"⚡ MEDIUM: {very_small_chunks} chunks ({small_chunk_pct:.1f}%) are very small (< 100 chars). "
                f"Monitor this metric - may affect retrieval quality."
            )

    # Check for very large chunks (> 2500 chars)
    if very_large_chunks > 0:
        large_chunk_pct = (very_large_chunks / total_chunks * 100) if total_chunks > 0 else 0
        if large_chunk_pct > 10:  # More than 10% are very large
            recommendations.append(
                f"⚠️ HIGH: {very_large_chunks} chunks ({large_chunk_pct:.1f}%) are very large (> 2500 chars). "
                f"Consider reducing max_chunk_size from current setting to 2000 characters for better retrieval precision."
            )

    # Check chunk size variance
    if chunk_size_stats.get("stddev", 0) > 800:
        recommendations.append(
            f"⚡ MEDIUM: High chunk size variance (stddev: {chunk_size_stats['stddev']:.0f}). "
            f"This is normal for semantic chunking but monitor for consistency. "
            f"Consider using fixed-size chunking if variance is problematic."
        )

    # Check overlap ratio
    overlap_ratio = chunk_overlap_stats.get("overlap_ratio", 0)
    if overlap_ratio > 25:
        recommendations.append(
            f"⚡ MEDIUM: Overlap ratio is {overlap_ratio:.1f}% (> 25%). "
            f"Consider reducing chunk_overlap from {chunk_overlap_stats.get('avg_overlap', 0):.0f} to "
            f"{chunk_overlap_stats.get('avg_configured_size', 1000) * 0.15:.0f} (15%) to reduce processing time by ~10-15%."
        )
    elif overlap_ratio < 10:
        recommendations.append(
            f"💡 INFO: Overlap ratio is {overlap_ratio:.1f}% (< 10%). "
            f"Consider increasing chunk_overlap to 15-20% for better context preservation."
        )

    # Check quality score distribution
    poor_quality_pct = (quality_distribution.get("poor", 0) / total_chunks * 100) if total_chunks > 0 else 0
    if poor_quality_pct > 15:
        recommendations.append(
            f"⚠️ HIGH: {poor_quality_pct:.1f}% of chunks have poor quality scores (< 0.5). "
            f"Review chunking strategy - may need better sentence boundary detection or content filtering."
        )

    # Check average quality score
    if average_quality_score < 0.6:
        recommendations.append(
            f"⚠️ HIGH: Average quality score is {average_quality_score:.2f} (< 0.6). "
            f"Immediate action needed: Review chunking configuration and consider implementing pre-processing filters."
        )
    elif average_quality_score < 0.7:
        recommendations.append(
            f"⚡ MEDIUM: Average quality score is {average_quality_score:.2f} (< 0.7). "
            f"Consider tuning chunk_size and overlap parameters for better quality."
        )
    elif average_quality_score >= 0.8:
        recommendations.append(
            f"✅ EXCELLENT: Average quality score is {average_quality_score:.2f}. "
            f"Current chunking configuration is performing well!"
        )

    # Check for good quality distribution
    excellent_pct = (quality_distribution.get("excellent", 0) / total_chunks * 100) if total_chunks > 0 else 0
    good_pct = (quality_distribution.get("good", 0) / total_chunks * 100) if total_chunks > 0 else 0

    if excellent_pct + good_pct > 80:
        recommendations.append(
            f"✅ EXCELLENT: {excellent_pct + good_pct:.1f}% of chunks have good/excellent quality. "
            f"No immediate changes needed."
        )

    # Provide specific configuration recommendations
    if not recommendations or all("✅" in r for r in recommendations):
        recommendations.append(
            f"💡 CURRENT CONFIG: chunk_size={chunk_overlap_stats.get('avg_configured_size', 1000):.0f}, "
            f"chunk_overlap={chunk_overlap_stats.get('avg_overlap', 200):.0f}. "
            f"System is performing optimally - maintain current settings."
        )

    return recommendations


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
    # Enhanced metrics
    chunk_size_stats: dict  # min, max, avg, stddev
    chunk_overlap_stats: dict  # overlap analysis
    very_small_chunks: int  # chunks < 100 chars
    very_large_chunks: int  # chunks > 2500 chars
    recommendations: List[str]  # actionable recommendations


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

        # Get detailed chunk size statistics
        chunk_sizes_query = supabase.client.table("document_chunks").select(
            "id, content, metadata"
        ).gte("created_at", start_date).limit(1000)  # Sample for performance
        chunk_sizes_response = chunk_sizes_query.execute()
        chunk_sizes_data = chunk_sizes_response.data or []

        # Calculate chunk size statistics
        chunk_lengths = [len(chunk.get("content", "")) for chunk in chunk_sizes_data]
        very_small_chunks = sum(1 for length in chunk_lengths if length < 100)
        very_large_chunks = sum(1 for length in chunk_lengths if length > 2500)

        import statistics
        chunk_size_stats = {
            "min": min(chunk_lengths) if chunk_lengths else 0,
            "max": max(chunk_lengths) if chunk_lengths else 0,
            "avg": round(statistics.mean(chunk_lengths), 1) if chunk_lengths else 0,
            "stddev": round(statistics.stdev(chunk_lengths), 1) if len(chunk_lengths) > 1 else 0,
            "median": round(statistics.median(chunk_lengths), 1) if chunk_lengths else 0,
        }

        # Analyze chunk overlap from metadata
        chunk_overlaps = []
        chunk_sizes_config = []
        for chunk in chunk_sizes_data:
            metadata = chunk.get("metadata", {})
            if isinstance(metadata, dict):
                overlap = metadata.get("chunk_overlap")
                size = metadata.get("chunk_size")
                if overlap:
                    chunk_overlaps.append(int(overlap))
                if size:
                    chunk_sizes_config.append(int(size))

        chunk_overlap_stats = {
            "avg_overlap": round(statistics.mean(chunk_overlaps), 1) if chunk_overlaps else 0,
            "avg_configured_size": round(statistics.mean(chunk_sizes_config), 1) if chunk_sizes_config else 0,
            "overlap_ratio": round(statistics.mean(chunk_overlaps) / statistics.mean(chunk_sizes_config) * 100, 1) if chunk_overlaps and chunk_sizes_config else 0,
        }

        # Generate actionable recommendations
        recommendations = _generate_chunk_recommendations(
            chunk_size_stats=chunk_size_stats,
            chunk_overlap_stats=chunk_overlap_stats,
            quality_distribution=quality_distribution,
            average_quality_score=average_quality_score,
            very_small_chunks=very_small_chunks,
            very_large_chunks=very_large_chunks,
            total_chunks=total_chunks
        )

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
            flagged_chunks_reviewed=flagged_chunks_reviewed,
            chunk_size_stats=chunk_size_stats,
            chunk_overlap_stats=chunk_overlap_stats,
            very_small_chunks=very_small_chunks,
            very_large_chunks=very_large_chunks,
            recommendations=recommendations
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

