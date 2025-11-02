"""
Document Entity Service

Handles document entities (certificates, logos, specifications) as separate knowledge base.
These entities are stored in document_entities table and linked to products via relationships.

ARCHITECTURE:
- Document entities are OPTIONAL (based on extract_categories parameter)
- Can be extracted DURING or AFTER product processing
- Stored separately from products in document_entities table
- Linked to products via product_document_relationships table
- Managed in "Docs" admin page

SUPPORTED ENTITY TYPES:
- certificates: ISO, CE, quality certifications, fire ratings
- logos: company logos, brand marks, certification logos
- specifications: technical specs, installation guides, maintenance instructions
- marketing: marketing content, brochures (future)
- bank_statement: bank statements (future)

EXTENSIBILITY:
This service is designed to support future document types through a plugin system.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from supabase import Client

logger = logging.getLogger(__name__)


@dataclass
class DocumentEntity:
    """
    Document entity (certificate, logo, specification, etc.)
    Stored separately from products in document_entities table.
    """
    entity_type: str  # 'certificate', 'logo', 'specification', 'marketing', 'bank_statement'
    name: str
    page_range: List[int]
    description: Optional[str] = None
    content: Optional[str] = None
    
    # Factory/Group identification (for agentic queries)
    factory_name: Optional[str] = None
    factory_group: Optional[str] = None
    manufacturer: Optional[str] = None
    
    # Entity-specific metadata
    metadata: Dict[str, Any] = None
    """
    Metadata structure (entity-specific):
    
    For certificates:
    {
        "certification_type": "quality_management",
        "issue_date": "2024-01-15",
        "expiry_date": "2027-01-15",
        "certifying_body": "TÜV SÜD",
        "certificate_number": "12345678",
        "scope": "Quality Management System",
        "standards": ["ISO 9001:2015"]
    }
    
    For logos:
    {
        "logo_type": "company",
        "brand_name": "Harmony",
        "color_scheme": ["blue", "white"],
        "usage_context": "header",
        "file_format": "vector"
    }
    
    For specifications:
    {
        "spec_type": "installation",
        "language": "en",
        "page_count": 3,
        "topics": ["preparation", "installation", "maintenance"],
        "target_audience": "professional_installers"
    }
    """
    
    confidence: float = 0.0


@dataclass
class DocumentEntityCatalog:
    """Catalog of discovered document entities."""
    entities: List[DocumentEntity]
    total_entities: int = 0
    by_type: Dict[str, int] = None  # Count by entity_type
    processing_time_ms: float = 0.0
    model_used: str = ""
    confidence_score: float = 0.0
    
    def __post_init__(self):
        """Calculate statistics"""
        if self.by_type is None:
            self.by_type = {}
            for entity in self.entities:
                self.by_type[entity.entity_type] = self.by_type.get(entity.entity_type, 0) + 1
        
        if self.total_entities == 0:
            self.total_entities = len(self.entities)


class DocumentEntityService:
    """
    Service for discovering and managing document entities.
    
    Document entities are stored separately from products and can be extracted
    DURING or AFTER product processing based on extract_categories parameter.
    """
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.logger = logging.getLogger(__name__)
    
    async def save_entities(
        self,
        entities: List[DocumentEntity],
        source_document_id: str,
        workspace_id: str
    ) -> List[str]:
        """
        Save document entities to database.
        
        Args:
            entities: List of document entities to save
            source_document_id: ID of the source PDF document
            workspace_id: Workspace ID
        
        Returns:
            List of created entity IDs
        """
        entity_ids = []
        
        for entity in entities:
            try:
                # Prepare entity data
                entity_data = {
                    "entity_type": entity.entity_type,
                    "name": entity.name,
                    "description": entity.description,
                    "source_document_id": source_document_id,
                    "workspace_id": workspace_id,
                    "page_range": entity.page_range,
                    "content": entity.content,
                    "metadata": entity.metadata or {},
                    "factory_name": entity.factory_name,
                    "factory_group": entity.factory_group,
                    "manufacturer": entity.manufacturer,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                
                # Insert into database
                result = self.supabase.table("document_entities").insert(entity_data).execute()
                
                if result.data and len(result.data) > 0:
                    entity_id = result.data[0]["id"]
                    entity_ids.append(entity_id)
                    self.logger.info(f"✅ Saved {entity.entity_type}: {entity.name} (ID: {entity_id})")
                else:
                    self.logger.error(f"❌ Failed to save {entity.entity_type}: {entity.name}")
            
            except Exception as e:
                self.logger.error(f"❌ Error saving entity {entity.name}: {e}")
                continue
        
        self.logger.info(f"✅ Saved {len(entity_ids)}/{len(entities)} document entities")
        return entity_ids
    
    async def link_entity_to_product(
        self,
        product_id: str,
        entity_id: str,
        relationship_type: str,
        relevance_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Link a document entity to a product.
        
        Args:
            product_id: Product ID
            entity_id: Document entity ID
            relationship_type: Type of relationship ('certification', 'specification', 'logo', 'marketing')
            relevance_score: Relevance score 0.0-1.0
            metadata: Optional relationship metadata
        
        Returns:
            True if successful, False otherwise
        """
        try:
            relationship_data = {
                "product_id": product_id,
                "document_entity_id": entity_id,
                "relationship_type": relationship_type,
                "relevance_score": relevance_score,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat()
            }
            
            result = self.supabase.table("product_document_relationships").insert(relationship_data).execute()
            
            if result.data and len(result.data) > 0:
                self.logger.info(f"✅ Linked entity {entity_id} to product {product_id} ({relationship_type})")
                return True
            else:
                self.logger.error(f"❌ Failed to link entity {entity_id} to product {product_id}")
                return False
        
        except Exception as e:
            self.logger.error(f"❌ Error linking entity to product: {e}")
            return False
    
    async def get_entities_for_product(
        self,
        product_id: str,
        entity_type: Optional[str] = None
    ) -> List[DocumentEntity]:
        """
        Get all document entities linked to a product.
        
        Args:
            product_id: Product ID
            entity_type: Optional filter by entity type
        
        Returns:
            List of document entities
        """
        try:
            query = self.supabase.table("product_document_relationships")\
                .select("document_entities(*)")\
                .eq("product_id", product_id)
            
            if entity_type:
                query = query.eq("document_entities.entity_type", entity_type)
            
            result = query.execute()
            
            entities = []
            if result.data:
                for row in result.data:
                    entity_data = row.get("document_entities", {})
                    entity = DocumentEntity(
                        entity_type=entity_data.get("entity_type"),
                        name=entity_data.get("name"),
                        page_range=entity_data.get("page_range", []),
                        description=entity_data.get("description"),
                        content=entity_data.get("content"),
                        factory_name=entity_data.get("factory_name"),
                        factory_group=entity_data.get("factory_group"),
                        manufacturer=entity_data.get("manufacturer"),
                        metadata=entity_data.get("metadata", {})
                    )
                    entities.append(entity)
            
            return entities
        
        except Exception as e:
            self.logger.error(f"❌ Error getting entities for product: {e}")
            return []
    
    async def get_entities_by_factory(
        self,
        factory_name: str,
        entity_type: Optional[str] = None
    ) -> List[DocumentEntity]:
        """
        Get all document entities for a specific factory.
        
        Args:
            factory_name: Factory name
            entity_type: Optional filter by entity type
        
        Returns:
            List of document entities
        """
        try:
            query = self.supabase.table("document_entities")\
                .select("*")\
                .eq("factory_name", factory_name)
            
            if entity_type:
                query = query.eq("entity_type", entity_type)
            
            result = query.execute()
            
            entities = []
            if result.data:
                for entity_data in result.data:
                    entity = DocumentEntity(
                        entity_type=entity_data.get("entity_type"),
                        name=entity_data.get("name"),
                        page_range=entity_data.get("page_range", []),
                        description=entity_data.get("description"),
                        content=entity_data.get("content"),
                        factory_name=entity_data.get("factory_name"),
                        factory_group=entity_data.get("factory_group"),
                        manufacturer=entity_data.get("manufacturer"),
                        metadata=entity_data.get("metadata", {})
                    )
                    entities.append(entity)
            
            return entities
        
        except Exception as e:
            self.logger.error(f"❌ Error getting entities by factory: {e}")
            return []

