"""
System Routes - Health checks and statistics

Endpoints:
- GET /health - Health check
- GET /stats - RAG statistics  
- GET /workspace-stats - Workspace statistics
"""

import logging
from fastapi import APIRouter, Depends, Query

from app.services.llamaindex_service import LlamaIndexService
from app.services.supabase_client import get_supabase_client
from .shared import get_llamaindex_service
from .models import HealthCheckResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Endpoints
# ============================================================================
@router.get("/health", response_model=HealthCheckResponse)
async def rag_health_check(
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    Health check endpoint for RAG service.
    
    Returns service status and availability of key components.
    """
    try:
        is_available = llamaindex_service.available if llamaindex_service else False
        
        return HealthCheckResponse(
            status="healthy" if is_available else "degraded",
            llamaindex_available=is_available,
            message="RAG service is operational" if is_available else "LlamaIndex service unavailable"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            llamaindex_available=False,
            message=f"Health check error: {str(e)}"
        )


@router.get("/stats")
async def get_rag_statistics(
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    Get overall RAG system statistics.
    
    Returns counts of documents, chunks, and embeddings across all workspaces.
    """
    try:
        supabase = get_supabase_client()
        
        # Get document count
        docs_response = supabase.client.table('documents').select('id', count='exact').execute()
        total_documents = docs_response.count if docs_response.count else 0
        
        # Get chunk count
        chunks_response = supabase.client.table('chunks').select('id', count='exact').execute()
        total_chunks = chunks_response.count if chunks_response.count else 0
        
        # Get embedding count
        embeddings_response = supabase.client.table('embeddings').select('id', count='exact').execute()
        total_embeddings = embeddings_response.count if embeddings_response.count else 0
        
        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "total_embeddings": total_embeddings,
            "llamaindex_available": llamaindex_service.available if llamaindex_service else False
        }
    except Exception as e:
        logger.error(f"Failed to get RAG statistics: {e}")
        raise


@router.get("/workspace-stats")
async def get_workspace_statistics(
    workspace_id: str,
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    Get statistics for a specific workspace.
    
    Returns counts of documents, chunks, images, and products for the workspace.
    """
    try:
        supabase = get_supabase_client()
        
        # Get document count
        docs_response = supabase.client.table('documents')\
            .select('id', count='exact')\
            .eq('workspace_id', workspace_id)\
            .execute()
        total_documents = docs_response.count if docs_response.count else 0
        
        # Get chunk count
        chunks_response = supabase.client.table('chunks')\
            .select('id', count='exact')\
            .eq('workspace_id', workspace_id)\
            .execute()
        total_chunks = chunks_response.count if chunks_response.count else 0
        
        # Get image count
        images_response = supabase.client.table('images')\
            .select('id', count='exact')\
            .eq('workspace_id', workspace_id)\
            .execute()
        total_images = images_response.count if images_response.count else 0
        
        # Get product count
        products_response = supabase.client.table('products')\
            .select('id', count='exact')\
            .eq('workspace_id', workspace_id)\
            .execute()
        total_products = products_response.count if products_response.count else 0
        
        return {
            "workspace_id": workspace_id,
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "total_images": total_images,
            "total_products": total_products
        }
    except Exception as e:
        logger.error(f"Failed to get workspace statistics: {e}")
        raise

