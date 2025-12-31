"""
Admin Linking API - Manual entity linking endpoints for debugging and fixing relationships.

These endpoints allow admins to manually trigger entity linking for documents
that may have failed during processing or need re-linking.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.entity_linking_service import EntityLinkingService
from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin/linking", tags=["admin-linking"])


class LinkChunksToProductsRequest(BaseModel):
    """Request model for linking chunks to products."""
    document_id: str


class LinkChunksToProductsResponse(BaseModel):
    """Response model for linking chunks to products."""
    success: bool
    document_id: str
    chunk_product_links: int
    chunks_found: int
    products_found: int
    error: Optional[str] = None


class LinkAllEntitiesRequest(BaseModel):
    """Request model for linking all entities."""
    document_id: str


class LinkAllEntitiesResponse(BaseModel):
    """Response model for linking all entities."""
    success: bool
    document_id: str
    image_product_links: int
    image_chunk_links: int
    chunk_product_links: int
    error: Optional[str] = None


@router.post("/link-chunks-to-products", response_model=LinkChunksToProductsResponse)
async def link_chunks_to_products(request: LinkChunksToProductsRequest):
    """
    Manually link chunks to products for a specific document.
    
    This endpoint is useful for:
    - Fixing documents where chunk-product linking failed
    - Re-linking after product updates
    - Debugging relationship issues
    
    Args:
        request: Document ID to link
        
    Returns:
        LinkChunksToProductsResponse with statistics
    """
    try:
        logger.info(f"🔗 [ADMIN] Manual chunk-to-product linking for document {request.document_id}")
        
        # Get document info
        supabase = get_supabase_client()
        
        # Check if document exists
        doc_response = supabase.client.table('documents')\
            .select('id, filename')\
            .eq('id', request.document_id)\
            .single()\
            .execute()
            
        if not doc_response.data:
            raise HTTPException(status_code=404, detail=f"Document {request.document_id} not found")
        
        # Get chunks count
        chunks_response = supabase.client.table('document_chunks')\
            .select('id', count='exact')\
            .eq('document_id', request.document_id)\
            .execute()
        chunks_count = chunks_response.count or 0
        
        # Get products count
        products_response = supabase.client.table('products')\
            .select('id', count='exact')\
            .eq('source_document_id', request.document_id)\
            .execute()
        products_count = products_response.count or 0
        
        logger.info(f"   Document: {doc_response.data.get('filename')}")
        logger.info(f"   Chunks found: {chunks_count}")
        logger.info(f"   Products found: {products_count}")
        
        if chunks_count == 0:
            return LinkChunksToProductsResponse(
                success=False,
                document_id=request.document_id,
                chunk_product_links=0,
                chunks_found=0,
                products_found=products_count,
                error="No chunks found for this document"
            )
        
        if products_count == 0:
            return LinkChunksToProductsResponse(
                success=False,
                document_id=request.document_id,
                chunk_product_links=0,
                chunks_found=chunks_count,
                products_found=0,
                error="No products found for this document"
            )
        
        # Initialize linking service
        linking_service = EntityLinkingService()
        
        # Link chunks to products
        links_created = await linking_service.link_chunks_to_products(
            document_id=request.document_id
        )
        
        logger.info(f"✅ [ADMIN] Created {links_created} chunk-product relationships")
        
        return LinkChunksToProductsResponse(
            success=True,
            document_id=request.document_id,
            chunk_product_links=links_created,
            chunks_found=chunks_count,
            products_found=products_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ADMIN] Failed to link chunks to products: {e}", exc_info=True)
        return LinkChunksToProductsResponse(
            success=False,
            document_id=request.document_id,
            chunk_product_links=0,
            chunks_found=0,
            products_found=0,
            error=str(e)
        )

