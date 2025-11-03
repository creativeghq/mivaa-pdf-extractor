"""
Metadata API Routes

This module provides comprehensive API endpoints for metadata management including:
- Metadata listing and filtering
- Scope detection (product-specific vs catalog-general)
- Metadata application to products
- Override management
- Metadata statistics and analytics
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query, status
from pydantic import BaseModel, Field

from app.services.dynamic_metadata_extractor import MetadataScopeDetector
from app.services.metadata_application_service import MetadataApplicationService
from app.services.supabase_client import get_supabase_client, SupabaseClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag/metadata", tags=["RAG System"])


# ============================================================================
# Request/Response Models
# ============================================================================

class MetadataScope(BaseModel):
    """Metadata scope classification."""
    scope: str = Field(..., description="Scope type: product_specific, catalog_general_explicit, catalog_general_implicit, category_specific")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    reasoning: str = Field(..., description="Explanation for scope classification")
    applies_to: Any = Field(..., description="Products/categories this metadata applies to")
    extracted_metadata: Dict[str, Any] = Field(..., description="Extracted metadata fields")
    is_override: bool = Field(False, description="Whether this overrides catalog-general metadata")


class ScopeDetectionRequest(BaseModel):
    """Request model for scope detection."""
    chunk_content: str = Field(..., min_length=1, description="Text content to analyze")
    product_names: List[str] = Field(..., description="List of product names in catalog")
    document_context: Optional[str] = Field(None, description="Optional document context")

    class Config:
        schema_extra = {
            "example": {
                "chunk_content": "Available in 15×38",
                "product_names": ["NOVA", "HARMONY", "ESSENCE"],
                "document_context": "Tile catalog"
            }
        }


class ScopeDetectionResponse(BaseModel):
    """Response model for scope detection."""
    success: bool = Field(..., description="Whether detection was successful")
    scope_result: MetadataScope = Field(..., description="Detected scope information")
    processing_time: float = Field(..., description="Processing time in seconds")


class MetadataApplicationRequest(BaseModel):
    """Request model for applying metadata to products."""
    document_id: str = Field(..., description="Document ID")
    chunks_with_scope: List[Dict[str, Any]] = Field(..., description="Chunks with scope detection results")

    class Config:
        schema_extra = {
            "example": {
                "document_id": "69cba085-9c2d-405c-aff2-8a20caf0b568",
                "chunks_with_scope": [
                    {
                        "chunk_id": "chunk-123",
                        "content": "Available in 15×38",
                        "scope": "catalog_general_implicit",
                        "applies_to": "all",
                        "extracted_metadata": {"dimensions": "15×38"},
                        "is_override": False
                    }
                ]
            }
        }


class MetadataApplicationResponse(BaseModel):
    """Response model for metadata application."""
    success: bool = Field(..., description="Whether application was successful")
    products_updated: int = Field(..., description="Number of products updated")
    overrides_detected: int = Field(..., description="Number of overrides detected")
    metadata_applied: Dict[str, Dict] = Field(..., description="Metadata applied per product")
    processing_time: float = Field(..., description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")


class MetadataListRequest(BaseModel):
    """Request model for listing metadata."""
    document_id: Optional[str] = Field(None, description="Filter by document ID")
    product_id: Optional[str] = Field(None, description="Filter by product ID")
    scope: Optional[str] = Field(None, description="Filter by scope type")
    metadata_key: Optional[str] = Field(None, description="Filter by metadata key")
    limit: int = Field(100, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Offset for pagination")


class MetadataItem(BaseModel):
    """Individual metadata item."""
    product_id: str = Field(..., description="Product ID")
    product_name: str = Field(..., description="Product name")
    metadata_key: str = Field(..., description="Metadata key")
    metadata_value: Any = Field(..., description="Metadata value")
    scope: Optional[str] = Field(None, description="Scope type")
    is_override: bool = Field(False, description="Whether this is an override")
    source_chunk_id: Optional[str] = Field(None, description="Source chunk ID")


class MetadataListResponse(BaseModel):
    """Response model for metadata listing."""
    success: bool = Field(..., description="Whether request was successful")
    total_count: int = Field(..., description="Total number of metadata items")
    items: List[MetadataItem] = Field(..., description="Metadata items")
    limit: int = Field(..., description="Limit used")
    offset: int = Field(..., description="Offset used")


class MetadataStatistics(BaseModel):
    """Metadata statistics."""
    total_products: int = Field(..., description="Total products")
    total_metadata_fields: int = Field(..., description="Total metadata fields")
    catalog_general_count: int = Field(..., description="Catalog-general metadata count")
    product_specific_count: int = Field(..., description="Product-specific metadata count")
    override_count: int = Field(..., description="Override count")
    most_common_fields: List[Dict[str, Any]] = Field(..., description="Most common metadata fields")


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/detect-scope", response_model=ScopeDetectionResponse)
async def detect_metadata_scope(
    request: ScopeDetectionRequest
) -> ScopeDetectionResponse:
    """
    Detect metadata scope for a text chunk.
    
    Classifies metadata as:
    - **product_specific**: Mentions specific product name
    - **catalog_general_explicit**: Explicitly says "all products"
    - **catalog_general_implicit**: Metadata mentioned without product context
    - **category_specific**: Applies to product category
    
    Args:
        request: Scope detection request
        
    Returns:
        ScopeDetectionResponse: Detected scope information
        
    Raises:
        HTTPException: If detection fails
    """
    try:
        start_time = datetime.utcnow()
        logger.info(f"🔍 Detecting scope for chunk: {request.chunk_content[:100]}...")
        
        # Initialize scope detector
        scope_detector = MetadataScopeDetector()
        
        # Detect scope
        scope_result = await scope_detector.detect_scope(
            chunk_content=request.chunk_content,
            product_names=request.product_names,
            document_context=request.document_context
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"✅ Scope detected: {scope_result['scope']} (confidence: {scope_result['confidence']})")
        
        return ScopeDetectionResponse(
            success=True,
            scope_result=MetadataScope(**scope_result),
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"❌ Scope detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scope detection failed: {str(e)}"
        )


@router.post("/apply-to-products", response_model=MetadataApplicationResponse)
async def apply_metadata_to_products(
    request: MetadataApplicationRequest,
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> MetadataApplicationResponse:
    """
    Apply metadata to products with scope-aware override logic.
    
    Processing order:
    1. Catalog-general metadata (explicit + implicit) → All products
    2. Category-specific metadata → Matching products
    3. Product-specific metadata → Specific products (can override)
    
    Args:
        request: Metadata application request
        supabase: Supabase client dependency
        
    Returns:
        MetadataApplicationResponse: Application results
        
    Raises:
        HTTPException: If application fails
    """
    try:
        start_time = datetime.utcnow()
        logger.info(f"🔄 Applying metadata to products for document: {request.document_id}")
        
        # Initialize metadata application service
        metadata_service = MetadataApplicationService(supabase)
        
        # Apply metadata
        result = await metadata_service.apply_metadata_to_products(
            document_id=request.document_id,
            chunks_with_scope=request.chunks_with_scope
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"✅ Metadata applied: {result['products_updated']} products updated, {result['overrides_detected']} overrides")
        
        return MetadataApplicationResponse(
            success=True,
            products_updated=result['products_updated'],
            overrides_detected=result['overrides_detected'],
            metadata_applied=result['metadata_applied'],
            processing_time=processing_time,
            error=result.get('error')
        )
    
    except Exception as e:
        logger.error(f"❌ Metadata application failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metadata application failed: {str(e)}"
        )


@router.get("/list", response_model=MetadataListResponse)
async def list_metadata(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    product_id: Optional[str] = Query(None, description="Filter by product ID"),
    scope: Optional[str] = Query(None, description="Filter by scope type"),
    metadata_key: Optional[str] = Query(None, description="Filter by metadata key"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> MetadataListResponse:
    """
    List metadata with filtering and pagination.
    
    Supports filtering by:
    - Document ID
    - Product ID
    - Scope type (product_specific, catalog_general_explicit, etc.)
    - Metadata key (dimensions, slip_resistance, etc.)
    
    Args:
        document_id: Filter by document ID
        product_id: Filter by product ID
        scope: Filter by scope type
        metadata_key: Filter by metadata key
        limit: Maximum results to return
        offset: Offset for pagination
        supabase: Supabase client dependency
        
    Returns:
        MetadataListResponse: List of metadata items
        
    Raises:
        HTTPException: If listing fails
    """
    try:
        logger.info(f"📋 Listing metadata (document_id={document_id}, product_id={product_id}, scope={scope}, key={metadata_key})")
        
        # Build query
        query = supabase.table('products').select('id, name, metadata')
        
        if document_id:
            query = query.eq('document_id', document_id)
        if product_id:
            query = query.eq('id', product_id)
        
        # Execute query
        response = query.range(offset, offset + limit - 1).execute()
        
        # Process results
        items = []
        for product in response.data:
            metadata = product.get('metadata', {})
            overrides = set(metadata.get('_overrides', []))
            
            for key, value in metadata.items():
                if key == '_overrides':
                    continue
                
                # Filter by metadata_key if specified
                if metadata_key and key != metadata_key:
                    continue
                
                items.append(MetadataItem(
                    product_id=product['id'],
                    product_name=product['name'],
                    metadata_key=key,
                    metadata_value=value,
                    scope=None,  # TODO: Track scope in metadata
                    is_override=key in overrides,
                    source_chunk_id=None  # TODO: Track source chunk
                ))
        
        logger.info(f"✅ Found {len(items)} metadata items")
        
        return MetadataListResponse(
            success=True,
            total_count=len(items),
            items=items,
            limit=limit,
            offset=offset
        )
    
    except Exception as e:
        logger.error(f"❌ Metadata listing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metadata listing failed: {str(e)}"
        )


@router.get("/statistics", response_model=MetadataStatistics)
async def get_metadata_statistics(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> MetadataStatistics:
    """
    Get metadata statistics and analytics.
    
    Provides insights into:
    - Total products and metadata fields
    - Catalog-general vs product-specific counts
    - Override counts
    - Most common metadata fields
    
    Args:
        document_id: Filter by document ID
        supabase: Supabase client dependency
        
    Returns:
        MetadataStatistics: Metadata statistics
        
    Raises:
        HTTPException: If statistics retrieval fails
    """
    try:
        logger.info(f"📊 Getting metadata statistics (document_id={document_id})")
        
        # Get products
        query = supabase.table('products').select('id, name, metadata')
        if document_id:
            query = query.eq('document_id', document_id)
        
        response = query.execute()
        products = response.data
        
        # Calculate statistics
        total_products = len(products)
        total_metadata_fields = 0
        override_count = 0
        field_counts = {}
        
        for product in products:
            metadata = product.get('metadata', {})
            overrides = set(metadata.get('_overrides', []))
            override_count += len(overrides)
            
            for key, value in metadata.items():
                if key == '_overrides':
                    continue
                
                total_metadata_fields += 1
                field_counts[key] = field_counts.get(key, 0) + 1
        
        # Get most common fields
        most_common_fields = [
            {"field": key, "count": count}
            for key, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        logger.info(f"✅ Statistics: {total_products} products, {total_metadata_fields} fields, {override_count} overrides")
        
        return MetadataStatistics(
            total_products=total_products,
            total_metadata_fields=total_metadata_fields,
            catalog_general_count=0,  # TODO: Track scope in metadata
            product_specific_count=0,  # TODO: Track scope in metadata
            override_count=override_count,
            most_common_fields=most_common_fields
        )
    
    except Exception as e:
        logger.error(f"❌ Statistics retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Statistics retrieval failed: {str(e)}"
        )

