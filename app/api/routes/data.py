"""
Data Routes - Retrieve chunks, images, products, embeddings, relevancies

Endpoints:
- GET /chunks - Get document chunks
- GET /images - Get document images
- GET /products - Get document products
- GET /embeddings - Get document embeddings
- GET /relevancies - Get chunk-image relevancies
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse

from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Endpoints
# ============================================================================
@router.get("/chunks")
async def get_chunks(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of chunks to return"),
    offset: int = Query(0, ge=0, description="Number of chunks to skip"),
    include_embeddings: bool = Query(True, description="Include embeddings in response")
):
    """
    Get chunks for a document with embeddings.

    Args:
        document_id: Document ID to filter chunks
        limit: Maximum number of chunks to return
        offset: Pagination offset
        include_embeddings: Whether to include embeddings (default: True)

    Returns:
        List of chunks with metadata and embeddings
    """
    try:
        supabase_client = get_supabase_client()

        if not document_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="document_id is required"
            )

        # Query chunks
        query = supabase_client.client.table('document_chunks').select('*').eq('document_id', document_id)
        query = query.range(offset, offset + limit - 1)
        result = query.execute()

        chunks = result.data if result.data else []

        # Add embeddings to each chunk if requested
        if include_embeddings and chunks:
            for chunk in chunks:
                embeddings_response = supabase_client.client.table('embeddings').select('*').eq('chunk_id', chunk['id']).execute()
                embeddings = embeddings_response.data or []
                # Add embedding field for backward compatibility
                chunk['embedding'] = embeddings[0]['embedding'] if embeddings else None
                chunk['embeddings'] = embeddings
                chunk['has_embedding'] = len(embeddings) > 0

        return JSONResponse(content={
            "document_id": document_id,
            "chunks": chunks,
            "count": len(chunks),
            "limit": limit,
            "offset": offset
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chunks for document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chunks: {str(e)}"
        )


@router.get("/images")
async def get_images(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of images to return"),
    offset: int = Query(0, ge=0, description="Number of images to skip")
):
    """
    Get images for a document.

    Args:
        document_id: Document ID to filter images
        limit: Maximum number of images to return
        offset: Pagination offset

    Returns:
        List of images with metadata
    """
    try:
        supabase_client = get_supabase_client()

        if not document_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="document_id is required"
            )

        # Query images
        query = supabase_client.client.table('document_images').select('*').eq('document_id', document_id)
        query = query.range(offset, offset + limit - 1)
        result = query.execute()

        images = result.data if result.data else []

        return JSONResponse(content={
            "document_id": document_id,
            "images": images,
            "count": len(images),
            "limit": limit,
            "offset": offset
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get images for document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve images: {str(e)}"
        )


@router.get("/products")
async def get_products(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of products to return"),
    offset: int = Query(0, ge=0, description="Number of products to skip")
):
    """
    Get products for a document.

    Args:
        document_id: Document ID to filter products
        limit: Maximum number of products to return
        offset: Pagination offset

    Returns:
        List of products with metadata
    """
    try:
        supabase_client = get_supabase_client()

        if not document_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="document_id is required"
            )

        # Query products
        query = supabase_client.client.table('products').select('*').eq('source_document_id', document_id)
        query = query.range(offset, offset + limit - 1)
        result = query.execute()

        products = result.data if result.data else []

        return JSONResponse(content={
            "document_id": document_id,
            "products": products,
            "count": len(products),
            "limit": limit,
            "offset": offset
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get products for document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve products: {str(e)}"
        )


@router.get("/embeddings")
async def get_embeddings(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    embedding_type: Optional[str] = Query(None, description="Filter by embedding type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of embeddings to return"),
    offset: int = Query(0, ge=0, description="Number of embeddings to skip")
):
    """
    Get embeddings for a document.

    Args:
        document_id: Document ID to filter embeddings
        embedding_type: Type of embedding (text, visual, multimodal, color, texture, application)
        limit: Maximum number of embeddings to return
        offset: Pagination offset

    Returns:
        List of embeddings with metadata
    """
    try:
        supabase_client = get_supabase_client()

        if not document_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="document_id is required"
            )

        # Query embeddings - JOIN through chunks to get document_id
        query = supabase_client.client.table('embeddings').select(
            '*, document_chunks!inner(document_id)'
        ).eq('document_chunks.document_id', document_id)

        # Filter by embedding type if specified
        if embedding_type:
            query = query.eq('embedding_type', embedding_type)

        query = query.range(offset, offset + limit - 1)
        result = query.execute()

        embeddings = result.data if result.data else []

        # Enrich embeddings with summary statistics
        embedding_stats = {}
        for emb in embeddings:
            emb_type = emb.get('embedding_type', 'unknown')
            if emb_type not in embedding_stats:
                embedding_stats[emb_type] = 0
            embedding_stats[emb_type] += 1

        return JSONResponse(content={
            "document_id": document_id,
            "embeddings": embeddings,
            "count": len(embeddings),
            "limit": limit,
            "offset": offset,
            "statistics": {
                "total_embeddings": len(embeddings),
                "by_type": embedding_stats
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get embeddings for document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve embeddings: {str(e)}"
        )


@router.get("/relevancies")
async def get_relevancies(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of relevancies to return"),
    offset: int = Query(0, ge=0, description="Number of relevancies to skip"),
    min_score: float = Query(0.0, ge=0.0, le=1.0, description="Minimum relevance score")
):
    """
    Get chunk-image relevancy relationships for a document.

    Args:
        document_id: Document ID to filter relevancies
        limit: Maximum number of relevancies to return
        offset: Pagination offset
        min_score: Minimum relevance score threshold

    Returns:
        List of chunk-image relationships with relevance scores
    """
    try:
        supabase_client = get_supabase_client()

        if not document_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="document_id is required"
            )

        # Query chunk-image relationships through chunks
        query = supabase_client.client.table('chunk_image_relationships').select(
            '*, document_chunks!inner(document_id, content), document_images(image_url, caption)'
        ).eq('document_chunks.document_id', document_id)

        if min_score > 0:
            query = query.gte('relevance_score', min_score)

        query = query.order('relevance_score', desc=True)
        query = query.range(offset, offset + limit - 1)
        result = query.execute()

        relevancies = result.data if result.data else []

        # Calculate statistics
        relationship_types = {}
        for rel in relevancies:
            rel_type = rel.get('relationship_type', 'unknown')
            if rel_type not in relationship_types:
                relationship_types[rel_type] = 0
            relationship_types[rel_type] += 1

        return JSONResponse(content={
            "document_id": document_id,
            "relevancies": relevancies,
            "count": len(relevancies),
            "limit": limit,
            "offset": offset,
            "statistics": {
                "total_relevancies": len(relevancies),
                "by_relationship_type": relationship_types,
                "min_score_filter": min_score
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get relevancies for document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve relevancies: {str(e)}"
        )

