"""
Document Entities API Routes

This module provides API endpoints for managing document entities
(certificates, logos, specifications, marketing, bank statements, etc.)
and their relationships with products.

ARCHITECTURE:
- Document entities are stored separately from products
- Linked to products via product_document_relationships table
- Support factory/group filtering for agentic queries
- Managed in "Docs" admin page
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field

from app.services.discovery.document_entity_service import DocumentEntityService, DocumentEntity
from app.services.core.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/document-entities", tags=["document-entities"])


# ============================================================================
# Request/Response Models
# ============================================================================

class DocumentEntityResponse(BaseModel):
    """Response model for a document entity."""
    id: str
    entity_type: str
    name: str
    description: Optional[str] = None
    page_range: List[int]
    factory_name: Optional[str] = None
    factory_group: Optional[str] = None
    manufacturer: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: str

    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "entity_type": "certificate",
                "name": "ISO 9001:2015",
                "description": "Quality Management System certification",
                "page_range": [45, 46],
                "factory_name": "Castellón Factory",
                "factory_group": "Harmony Group",
                "manufacturer": "Harmony Materials",
                "metadata": {
                    "certificate_type": "quality_management",
                    "issue_date": "2024-01-15",
                    "expiry_date": "2027-01-15",
                    "certifying_body": "TÜV SÜD",
                    "standards": ["ISO 9001:2015"]
                },
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class ProductDocumentRelationshipResponse(BaseModel):
    """Response model for product-document relationship."""
    id: str
    product_id: str
    document_entity_id: str
    relationship_type: str
    relevance_score: float
    metadata: Dict[str, Any] = {}
    created_at: str

    class Config:
        schema_extra = {
            "example": {
                "id": "660e8400-e29b-41d4-a716-446655440000",
                "product_id": "770e8400-e29b-41d4-a716-446655440000",
                "document_entity_id": "550e8400-e29b-41d4-a716-446655440000",
                "relationship_type": "certification",
                "relevance_score": 0.95,
                "metadata": {
                    "linking_method": "ai_discovery",
                    "confidence": 0.95
                },
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


# ============================================================================
# Dependency Injection
# ============================================================================

def get_document_entity_service() -> DocumentEntityService:
    """Get DocumentEntityService instance."""
    supabase = SupabaseClient()
    return DocumentEntityService(supabase.client)


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/", response_model=List[DocumentEntityResponse])
async def get_document_entities(
    workspace_id: str = Query(..., description="Workspace ID"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type (certificate, logo, specification, etc.)"),
    factory_name: Optional[str] = Query(None, description="Filter by factory name"),
    factory_group: Optional[str] = Query(None, description="Filter by factory group"),
    limit: int = Query(100, description="Maximum number of entities to return"),
    offset: int = Query(0, description="Number of entities to skip"),
    service: DocumentEntityService = Depends(get_document_entity_service)
):
    """
    Get all document entities for a workspace.
    
    Supports filtering by:
    - entity_type: certificate, logo, specification, marketing, bank_statement
    - factory_name: Filter by specific factory
    - factory_group: Filter by factory group
    
    Example agentic query: "Get all certifications for Castellón Factory"
    → GET /api/document-entities?workspace_id=xxx&entity_type=certificate&factory_name=Castellón Factory
    """
    try:
        # Build query
        query = service.supabase.client.table("document_entities")\
            .select("*")\
            .eq("workspace_id", workspace_id)
        
        if entity_type:
            query = query.eq("entity_type", entity_type)
        
        if factory_name:
            query = query.eq("factory_name", factory_name)
        
        if factory_group:
            query = query.eq("factory_group", factory_group)
        
        # Apply pagination
        query = query.range(offset, offset + limit - 1)
        
        result = query.execute()
        
        if not result.data:
            return []
        
        return result.data
    
    except Exception as e:
        logger.error(f"Error getting document entities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document entities: {str(e)}"
        )


@router.get("/{entity_id}", response_model=DocumentEntityResponse)
async def get_document_entity(
    entity_id: str,
    service: DocumentEntityService = Depends(get_document_entity_service)
):
    """Get a specific document entity by ID."""
    try:
        result = service.supabase.client.table("document_entities")\
            .select("*")\
            .eq("id", entity_id)\
            .execute()
        
        if not result.data or len(result.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document entity {entity_id} not found"
            )
        
        return result.data[0]
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document entity: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document entity: {str(e)}"
        )


@router.get("/product/{product_id}", response_model=List[DocumentEntityResponse])
async def get_entities_for_product(
    product_id: str,
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    service: DocumentEntityService = Depends(get_document_entity_service)
):
    """
    Get all document entities linked to a specific product.
    
    Example agentic query: "Get certifications for product NOVA"
    → First get product ID for NOVA, then:
    → GET /api/document-entities/product/{product_id}?entity_type=certificate
    """
    try:
        entities = await service.get_entities_for_product(
            product_id=product_id,
            entity_type=entity_type
        )
        
        # Convert to response format
        response = []
        for entity in entities:
            response.append({
                "entity_type": entity.entity_type,
                "name": entity.name,
                "description": entity.description,
                "page_range": entity.page_range,
                "factory_name": entity.factory_name,
                "factory_group": entity.factory_group,
                "manufacturer": entity.manufacturer,
                "metadata": entity.metadata or {}
            })
        
        return response
    
    except Exception as e:
        logger.error(f"Error getting entities for product: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get entities for product: {str(e)}"
        )


@router.get("/factory/{factory_name}", response_model=List[DocumentEntityResponse])
async def get_entities_by_factory(
    factory_name: str,
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    service: DocumentEntityService = Depends(get_document_entity_service)
):
    """
    Get all document entities for a specific factory.
    
    Example agentic query: "Get all certifications for Castellón Factory"
    → GET /api/document-entities/factory/Castellón Factory?entity_type=certificate
    """
    try:
        entities = await service.get_entities_by_factory(
            factory_name=factory_name,
            entity_type=entity_type
        )
        
        # Convert to response format
        response = []
        for entity in entities:
            response.append({
                "entity_type": entity.entity_type,
                "name": entity.name,
                "description": entity.description,
                "page_range": entity.page_range,
                "factory_name": entity.factory_name,
                "factory_group": entity.factory_group,
                "manufacturer": entity.manufacturer,
                "metadata": entity.metadata or {}
            })
        
        return response
    
    except Exception as e:
        logger.error(f"Error getting entities by factory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get entities by factory: {str(e)}"
        )


@router.get("/relationships/product/{product_id}", response_model=List[ProductDocumentRelationshipResponse])
async def get_product_document_relationships(
    product_id: str,
    service: DocumentEntityService = Depends(get_document_entity_service)
):
    """Get all document entity relationships for a specific product."""
    try:
        result = service.supabase.client.table("product_document_relationships")\
            .select("*")\
            .eq("product_id", product_id)\
            .execute()
        
        if not result.data:
            return []
        
        return result.data
    
    except Exception as e:
        logger.error(f"Error getting product-document relationships: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get product-document relationships: {str(e)}"
        )


