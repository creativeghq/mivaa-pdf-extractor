"""
Duplicate Detection and Product Merge API Routes

CRITICAL: Duplicates are ONLY detected when products are from the SAME factory/manufacturer.
Visual similarity alone does NOT constitute a duplicate.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.services.core.supabase_client import SupabaseClient
from app.services.search.duplicate_detection_service import DuplicateDetectionService
from app.services.products.product_merge_service import ProductMergeService
from app.dependencies import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/duplicates", tags=["Duplicate Detection"])


# ============================================================================
# Request/Response Models
# ============================================================================

class DetectDuplicatesRequest(BaseModel):
    """Request to detect duplicates for a specific product."""
    product_id: str = Field(..., description="Product ID to check for duplicates")
    workspace_id: str = Field(..., description="Workspace ID")
    similarity_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0.0-1.0)"
    )


class BatchDetectRequest(BaseModel):
    """Request to scan entire workspace for duplicates."""
    workspace_id: str = Field(..., description="Workspace ID to scan")
    similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score"
    )
    limit: Optional[int] = Field(
        default=None,
        description="Optional limit on number of products to check"
    )


class MergeProductsRequest(BaseModel):
    """Request to merge duplicate products."""
    target_product_id: str = Field(..., description="Product to keep (merge into)")
    source_product_ids: List[str] = Field(..., description="Products to merge (will be deleted)")
    workspace_id: str = Field(..., description="Workspace ID")
    user_id: str = Field(..., description="User performing the merge")
    merge_strategy: str = Field(
        default="manual",
        description="Merge strategy: 'manual', 'auto', or 'suggested'"
    )
    merge_reason: Optional[str] = Field(
        default=None,
        description="Optional reason for merge"
    )


class UndoMergeRequest(BaseModel):
    """Request to undo a product merge."""
    history_id: str = Field(..., description="Merge history ID to undo")
    user_id: str = Field(..., description="User performing the undo")


class UpdateDuplicateStatusRequest(BaseModel):
    """Request to update duplicate detection status."""
    cache_id: str = Field(..., description="Duplicate cache ID")
    status: str = Field(..., description="New status: 'pending', 'reviewed', 'merged', 'dismissed'")
    user_id: str = Field(..., description="User updating the status")


# ============================================================================
# Dependency Injection
# ============================================================================
# REMOVED: get_supabase_client - now using centralized dependency from app.dependencies

def get_duplicate_service(
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> DuplicateDetectionService:
    """Get duplicate detection service instance."""
    return DuplicateDetectionService(supabase)


def get_merge_service(
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> ProductMergeService:
    """Get product merge service instance."""
    return ProductMergeService(supabase)


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/detect")
async def detect_duplicates_for_product(
    request: DetectDuplicatesRequest,
    service: DuplicateDetectionService = Depends(get_duplicate_service)
):
    """
    Detect potential duplicates for a specific product.
    
    CRITICAL: Only finds duplicates from the SAME factory/manufacturer.
    Returns empty list if product has no factory metadata.
    
    Returns:
        List of potential duplicate products with similarity scores
    """
    try:
        logger.info(
            f"Detecting duplicates for product {request.product_id} "
            f"(threshold: {request.similarity_threshold})"
        )
        
        duplicates = await service.detect_duplicates_for_product(
            product_id=request.product_id,
            workspace_id=request.workspace_id,
            similarity_threshold=request.similarity_threshold
        )
        
        return {
            "success": True,
            "product_id": request.product_id,
            "duplicates_found": len(duplicates),
            "duplicates": duplicates,
            "note": "Only products from the same factory/manufacturer are considered duplicates"
        }
        
    except Exception as e:
        logger.error(f"Error detecting duplicates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-detect")
async def batch_detect_duplicates(
    request: BatchDetectRequest,
    service: DuplicateDetectionService = Depends(get_duplicate_service)
):
    """
    Scan entire workspace for duplicate products.
    
    CRITICAL: Only detects duplicates from the SAME factory/manufacturer.
    Products without factory metadata are skipped.
    
    This can be a long-running operation for large workspaces.
    
    Returns:
        List of duplicate pairs with similarity scores
    """
    try:
        logger.info(
            f"Starting batch duplicate detection for workspace {request.workspace_id} "
            f"(threshold: {request.similarity_threshold})"
        )
        
        duplicate_pairs = await service.batch_detect_duplicates(
            workspace_id=request.workspace_id,
            similarity_threshold=request.similarity_threshold,
            limit=request.limit
        )
        
        return {
            "success": True,
            "workspace_id": request.workspace_id,
            "duplicate_pairs_found": len(duplicate_pairs),
            "duplicate_pairs": duplicate_pairs,
            "note": "Only products from the same factory/manufacturer are considered duplicates"
        }
        
    except Exception as e:
        logger.error(f"Error in batch duplicate detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cached")
async def get_cached_duplicates(
    workspace_id: str,
    status: Optional[str] = None,
    min_similarity: float = 0.60,
    service: DuplicateDetectionService = Depends(get_duplicate_service)
):
    """
    Get cached duplicate detections.
    
    Args:
        workspace_id: Workspace to query
        status: Filter by status ('pending', 'reviewed', 'merged', 'dismissed')
        min_similarity: Minimum similarity score (default: 0.60)
    
    Returns:
        List of cached duplicate pairs
    """
    try:
        cached = await service.get_cached_duplicates(
            workspace_id=workspace_id,
            status=status,
            min_similarity=min_similarity
        )
        
        return {
            "success": True,
            "workspace_id": workspace_id,
            "cached_duplicates": len(cached),
            "duplicates": cached
        }
        
    except Exception as e:
        logger.error(f"Error getting cached duplicates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-status")
async def update_duplicate_status(
    request: UpdateDuplicateStatusRequest,
    service: DuplicateDetectionService = Depends(get_duplicate_service)
):
    """
    Update the status of a cached duplicate detection.
    
    Statuses:
    - 'pending': Not yet reviewed
    - 'reviewed': Admin has reviewed
    - 'merged': Products have been merged
    - 'dismissed': Not actually duplicates
    """
    try:
        success = await service.update_duplicate_status(
            cache_id=request.cache_id,
            status=request.status,
            user_id=request.user_id
        )
        
        if success:
            return {
                "success": True,
                "message": f"Status updated to '{request.status}'"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update status")
        
    except Exception as e:
        logger.error(f"Error updating duplicate status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/merge")
async def merge_products(
    request: MergeProductsRequest,
    service: ProductMergeService = Depends(get_merge_service)
):
    """
    Merge duplicate products into a single product.
    
    Process:
    1. Merges data from source products into target product
    2. Transfers all relationships (images, chunks, etc.)
    3. Deletes source products
    4. Records merge in history for undo capability
    
    Returns:
        Merge result with history ID and updated product
    """
    try:
        logger.info(
            f"Merging {len(request.source_product_ids)} products into {request.target_product_id}"
        )
        
        result = await service.merge_products(
            target_product_id=request.target_product_id,
            source_product_ids=request.source_product_ids,
            workspace_id=request.workspace_id,
            user_id=request.user_id,
            merge_strategy=request.merge_strategy,
            merge_reason=request.merge_reason
        )
        
        if result.get('success'):
            return result
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Merge failed')
            )
        
    except Exception as e:
        logger.error(f"Error merging products: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/undo-merge")
async def undo_merge(
    request: UndoMergeRequest,
    service: ProductMergeService = Depends(get_merge_service)
):
    """
    Undo a product merge operation.
    
    Restores source products and reverts target product to pre-merge state.
    """
    try:
        logger.info(f"Undoing merge {request.history_id}")
        
        result = await service.undo_merge(
            history_id=request.history_id,
            user_id=request.user_id
        )
        
        if result.get('success'):
            return result
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Undo failed')
            )
        
    except Exception as e:
        logger.error(f"Error undoing merge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/merge-history")
async def get_merge_history(
    workspace_id: str,
    limit: int = 50,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Get merge history for a workspace.
    
    Returns:
        List of merge operations with details
    """
    try:
        response = supabase.client.table('product_merge_history').select(
            '*'
        ).eq('workspace_id', workspace_id).order(
            'merged_at', desc=True
        ).limit(limit).execute()
        
        return {
            "success": True,
            "workspace_id": workspace_id,
            "merge_count": len(response.data) if response.data else 0,
            "merges": response.data if response.data else []
        }
        
    except Exception as e:
        logger.error(f"Error getting merge history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


