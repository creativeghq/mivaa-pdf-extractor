"""
Admin API endpoints for chunk quality monitoring and management.

Provides endpoints to:
- View chunk quality metrics and statistics
- Review flagged low-quality chunks
- Approve/reject flagged chunks
- View quality distribution across documents
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel
import logging

from ...services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/chunk-quality", tags=["admin", "chunk-quality"])


class ChunkQualityMetrics(BaseModel):
    """Chunk quality metrics response model."""
    total_chunks: int
    total_documents: int
    exact_duplicates_prevented: int
    semantic_duplicates_prevented: int
    low_quality_rejected: int
    borderline_quality_flagged: int
    average_quality_score: float
    quality_distribution: Dict[str, int]  # e.g., {"0.0-0.3": 5, "0.3-0.5": 10, ...}
    flagged_chunks_pending_review: int
    flagged_chunks_reviewed: int


class FlaggedChunk(BaseModel):
    """Flagged chunk model."""
    id: str
    chunk_id: str
    document_id: str
    flag_type: str
    flag_reason: str
    quality_score: float
    content_preview: str
    flagged_at: datetime
    reviewed: bool
    reviewed_at: Optional[datetime]
    reviewed_by: Optional[str]
    review_action: Optional[str]


class ReviewAction(BaseModel):
    """Review action request model."""
    action: str  # 'approve', 'reject', 'delete_chunk'
    notes: Optional[str] = None


@router.get("/metrics", response_model=ChunkQualityMetrics)
async def get_chunk_quality_metrics(
    workspace_id: Optional[str] = Query(None),
    document_id: Optional[str] = Query(None),
    days: int = Query(30, description="Number of days to analyze")
):
    """
    Get comprehensive chunk quality metrics.
    
    Returns statistics about:
    - Total chunks created
    - Duplicates prevented (exact and semantic)
    - Quality rejections
    - Quality score distribution
    - Flagged chunks status
    """
    try:
        supabase = get_supabase_client()
        
        # Calculate date range
        start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        # Build base query
        chunks_query = supabase.client.table('document_chunks').select('*', count='exact')
        
        if workspace_id:
            chunks_query = chunks_query.eq('workspace_id', workspace_id)
        if document_id:
            chunks_query = chunks_query.eq('document_id', document_id)
        
        chunks_query = chunks_query.gte('created_at', start_date)
        
        # Get total chunks
        chunks_result = chunks_query.execute()
        total_chunks = chunks_result.count or 0
        
        # Get unique documents
        docs_query = supabase.client.table('document_chunks')\
            .select('document_id', count='exact')\
            .gte('created_at', start_date)
        
        if workspace_id:
            docs_query = docs_query.eq('workspace_id', workspace_id)
        
        docs_result = docs_query.execute()
        unique_docs = len(set([d['document_id'] for d in docs_result.data])) if docs_result.data else 0
        
        # Calculate quality distribution
        quality_distribution = {
            "0.0-0.3": 0,
            "0.3-0.5": 0,
            "0.5-0.7": 0,
            "0.7-0.85": 0,
            "0.85-1.0": 0
        }
        
        total_quality_score = 0.0
        chunks_with_scores = 0
        
        for chunk in chunks_result.data or []:
            metadata = chunk.get('metadata', {})
            quality_score = metadata.get('quality_score')
            
            if quality_score is not None:
                total_quality_score += quality_score
                chunks_with_scores += 1
                
                # Categorize into distribution
                if quality_score < 0.3:
                    quality_distribution["0.0-0.3"] += 1
                elif quality_score < 0.5:
                    quality_distribution["0.3-0.5"] += 1
                elif quality_score < 0.7:
                    quality_distribution["0.5-0.7"] += 1
                elif quality_score < 0.85:
                    quality_distribution["0.7-0.85"] += 1
                else:
                    quality_distribution["0.85-1.0"] += 1
        
        average_quality_score = total_quality_score / chunks_with_scores if chunks_with_scores > 0 else 0.0
        
        # Get flagged chunks statistics
        flags_query = supabase.client.table('chunk_quality_flags')\
            .select('*', count='exact')\
            .gte('created_at', start_date)
        
        if workspace_id:
            flags_query = flags_query.eq('workspace_id', workspace_id)
        if document_id:
            flags_query = flags_query.eq('document_id', document_id)
        
        flags_result = flags_query.execute()
        total_flags = flags_result.count or 0
        
        flagged_pending = len([f for f in (flags_result.data or []) if not f.get('reviewed', False)])
        flagged_reviewed = len([f for f in (flags_result.data or []) if f.get('reviewed', False)])
        
        # Note: Exact/semantic duplicates prevented and low quality rejected
        # would need to be tracked in a separate metrics table or logs
        # For now, we'll estimate based on flags
        
        return ChunkQualityMetrics(
            total_chunks=total_chunks,
            total_documents=unique_docs,
            exact_duplicates_prevented=0,  # TODO: Track in metrics table
            semantic_duplicates_prevented=0,  # TODO: Track in metrics table
            low_quality_rejected=0,  # TODO: Track in metrics table
            borderline_quality_flagged=total_flags,
            average_quality_score=round(average_quality_score, 3),
            quality_distribution=quality_distribution,
            flagged_chunks_pending_review=flagged_pending,
            flagged_chunks_reviewed=flagged_reviewed
        )
    except Exception as e:
        logger.error(f"Error getting chunk quality metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flagged", response_model=List[FlaggedChunk])
async def get_flagged_chunks(
    workspace_id: Optional[str] = Query(None),
    document_id: Optional[str] = Query(None),
    reviewed: Optional[bool] = Query(None),
    limit: int = Query(50, le=200)
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
        supabase = get_supabase_client()
        
        query = supabase.client.table('chunk_quality_flags')\
            .select('*')\
            .order('flagged_at', desc=True)\
            .limit(limit)
        
        if workspace_id:
            query = query.eq('workspace_id', workspace_id)
        if document_id:
            query = query.eq('document_id', document_id)
        if reviewed is not None:
            query = query.eq('reviewed', reviewed)
        
        result = query.execute()
        
        return [FlaggedChunk(**flag) for flag in (result.data or [])]
    except Exception as e:
        logger.error(f"Error getting flagged chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/flagged/{flag_id}/review")
async def review_flagged_chunk(
    flag_id: str,
    review: ReviewAction,
    reviewer_id: Optional[str] = Query(None)
):
    """
    Review a flagged chunk.
    
    Actions:
    - approve: Mark chunk as acceptable, clear flag
    - reject: Keep flag, mark as reviewed
    - delete_chunk: Delete the chunk and clear flag
    """
    try:
        supabase = get_supabase_client()
        
        # Get flag details
        flag_result = supabase.client.table('chunk_quality_flags')\
            .select('*')\
            .eq('id', flag_id)\
            .execute()
        
        if not flag_result.data:
            raise HTTPException(status_code=404, detail="Flag not found")
        
        flag = flag_result.data[0]
        
        # Handle delete action
        if review.action == 'delete_chunk':
            # Delete the chunk (cascade will delete flag)
            supabase.client.table('document_chunks')\
                .delete()\
                .eq('id', flag['chunk_id'])\
                .execute()
            
            return {"message": "Chunk deleted successfully"}
        
        # Update flag with review
        update_data = {
            'reviewed': True,
            'reviewed_at': datetime.utcnow().isoformat(),
            'reviewed_by': reviewer_id,
            'review_action': review.action
        }
        
        supabase.client.table('chunk_quality_flags')\
            .update(update_data)\
            .eq('id', flag_id)\
            .execute()
        
        return {"message": f"Flag reviewed with action: {review.action}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reviewing flagged chunk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/distribution")
async def get_quality_distribution(
    workspace_id: Optional[str] = Query(None),
    document_id: Optional[str] = Query(None)
):
    """
    Get detailed quality score distribution with histogram data.
    """
    try:
        supabase = get_supabase_client()
        
        query = supabase.client.table('document_chunks').select('metadata')
        
        if workspace_id:
            query = query.eq('workspace_id', workspace_id)
        if document_id:
            query = query.eq('document_id', document_id)
        
        result = query.execute()
        
        # Create histogram buckets (0.0-1.0 in 0.1 increments)
        histogram = {f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(10)}
        
        for chunk in (result.data or []):
            metadata = chunk.get('metadata', {})
            quality_score = metadata.get('quality_score')
            
            if quality_score is not None:
                bucket_index = min(int(quality_score * 10), 9)
                bucket_key = f"{bucket_index/10:.1f}-{(bucket_index+1)/10:.1f}"
                histogram[bucket_key] += 1
        
        return {
            "histogram": histogram,
            "total_chunks": len(result.data or [])
        }
    except Exception as e:
        logger.error(f"Error getting quality distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

