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
                result = self.supabase.client.table("document_entities").insert(entity_data).execute()
                
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
            
            result = self.supabase.client.table("product_document_relationships").insert(relationship_data).execute()
            
            if result.data and len(result.data) > 0:
                self.logger.info(f"✅ Linked entity {entity_id} to product {product_id} ({relationship_type})")
                return True
            else:
                self.logger.error(f"❌ Failed to link entity {entity_id} to product {product_id}")
                return False
        
        except Exception as e:
            self.logger.error(f"❌ Error linking entity to product: {e}")
            return False

    async def match_entities_to_products(
        self,
        document_id: str,
        workspace_id: str
    ) -> Dict[str, Any]:
        """
        Match document entities to products based on page overlap, factory, and manufacturer.

        Matching criteria (in order of priority):
        1. Page overlap - entity and product share pages
        2. Factory/manufacturer match - same factory or manufacturer name
        3. Name similarity - entity name mentions product name

        Args:
            document_id: Document ID to match entities for
            workspace_id: Workspace ID

        Returns:
            Dictionary with matching statistics
        """
        try:
            self.logger.info(f"🔗 Matching entities to products for document {document_id}")

            # Fetch all entities for this document
            entities_response = self.supabase.client.table('document_entities').select('*').eq(
                'source_document_id', document_id
            ).eq('workspace_id', workspace_id).execute()

            if not entities_response.data:
                self.logger.info("No entities found to match")
                return {"relationships_created": 0, "entities_processed": 0}

            entities = entities_response.data

            # Fetch all products for this document
            products_response = self.supabase.client.table('products').select('*').eq(
                'source_document_id', document_id
            ).eq('workspace_id', workspace_id).execute()

            if not products_response.data:
                self.logger.info("No products found to match")
                return {"relationships_created": 0, "entities_processed": len(entities)}

            products = products_response.data

            relationships_created = 0
            entities_processed = 0

            for entity in entities:
                entity_id = entity['id']
                entity_type = entity['entity_type']
                entity_name = entity.get('name', '')
                entity_page_range = entity.get('page_range', [])
                entity_factory = entity.get('factory_name', '').lower() if entity.get('factory_name') else None
                entity_manufacturer = entity.get('manufacturer', '').lower() if entity.get('manufacturer') else None

                entities_processed += 1
                matched_products = []

                for product in products:
                    product_id = product['id']
                    product_name = product.get('name', '')
                    product_metadata = product.get('metadata', {})

                    # Extract product page range from metadata
                    product_page_range = product_metadata.get('page_range', [])
                    product_factory = product_metadata.get('factory_name', '').lower() if product_metadata.get('factory_name') else None
                    product_manufacturer = product_metadata.get('manufacturer', '').lower() if product_metadata.get('manufacturer') else None

                    # Calculate match score
                    match_score = 0.0
                    match_reasons = []

                    # 1. Page overlap (highest priority - 0.6)
                    if entity_page_range and product_page_range:
                        overlap = set(entity_page_range) & set(product_page_range)
                        if overlap:
                            overlap_ratio = len(overlap) / max(len(entity_page_range), len(product_page_range))
                            match_score += 0.6 * overlap_ratio
                            match_reasons.append(f"Page overlap: {len(overlap)} pages")

                    # 2. Factory match (medium priority - 0.3)
                    if entity_factory and product_factory and entity_factory == product_factory:
                        match_score += 0.3
                        match_reasons.append(f"Factory match: {entity_factory}")

                    # 3. Manufacturer match (medium priority - 0.3)
                    if entity_manufacturer and product_manufacturer and entity_manufacturer == product_manufacturer:
                        match_score += 0.3
                        match_reasons.append(f"Manufacturer match: {entity_manufacturer}")

                    # 4. Name similarity (low priority - 0.1)
                    if product_name.lower() in entity_name.lower() or entity_name.lower() in product_name.lower():
                        match_score += 0.1
                        match_reasons.append(f"Name similarity")

                    # Create relationship if match score >= 0.5
                    if match_score >= 0.5:
                        matched_products.append({
                            'product_id': product_id,
                            'product_name': product_name,
                            'match_score': match_score,
                            'match_reasons': match_reasons
                        })

                # Link entity to all matched products
                for match in matched_products:
                    success = await self.link_entity_to_product(
                        product_id=match['product_id'],
                        entity_id=entity_id,
                        relationship_type=entity_type,  # Use entity type as relationship type
                        relevance_score=match['match_score'],
                        metadata={
                            'match_reasons': match['match_reasons'],
                            'entity_name': entity_name,
                            'product_name': match['product_name']
                        }
                    )

                    if success:
                        relationships_created += 1
                        self.logger.info(
                            f"   ✅ Linked {entity_type} '{entity_name}' to product '{match['product_name']}' "
                            f"(score: {match['match_score']:.2f}, reasons: {', '.join(match['match_reasons'])})"
                        )

            self.logger.info(
                f"✅ Entity matching complete: {relationships_created} relationships created "
                f"from {entities_processed} entities"
            )

            return {
                "relationships_created": relationships_created,
                "entities_processed": entities_processed,
                "products_available": len(products)
            }

        except Exception as e:
            self.logger.error(f"❌ Error matching entities to products: {e}")
            return {"relationships_created": 0, "entities_processed": 0, "error": str(e)}

    async def generate_entity_embeddings(
        self,
        document_id: str,
        workspace_id: str
    ) -> Dict[str, Any]:
        """
        Generate embeddings for all document entities.

        Creates text embeddings for entity content to enable semantic search.
        Embeddings are stored in the embeddings table with entity_type='entity'.

        Args:
            document_id: Document ID to generate embeddings for
            workspace_id: Workspace ID

        Returns:
            Dictionary with embedding generation statistics
        """
        try:
            self.logger.info(f"🎨 Generating embeddings for document entities in {document_id}")

            # Fetch all entities for this document
            entities_response = self.supabase.client.table('document_entities').select('*').eq(
                'source_document_id', document_id
            ).eq('workspace_id', workspace_id).execute()

            if not entities_response.data:
                self.logger.info("No entities found to generate embeddings for")
                return {"embeddings_generated": 0, "entities_processed": 0}

            entities = entities_response.data

            # Import embedding service
            from app.services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()

            embeddings_generated = 0
            entities_processed = 0

            for entity in entities:
                entity_id = entity['id']
                entity_type = entity['entity_type']
                entity_name = entity.get('name', '')
                entity_description = entity.get('description', '')
                entity_content = entity.get('content', '')

                # Build text content for embedding
                # Combine name, description, and content for rich semantic representation
                text_parts = []
                if entity_name:
                    text_parts.append(f"Name: {entity_name}")
                if entity_description:
                    text_parts.append(f"Description: {entity_description}")
                if entity_content:
                    text_parts.append(f"Content: {entity_content}")

                text_content = "\n".join(text_parts)

                if not text_content.strip():
                    self.logger.warning(f"⚠️ Entity {entity_id} has no text content, skipping embedding")
                    continue

                entities_processed += 1

                try:
                    # Generate text embedding using embedding service
                    embedding_result = await embedding_service.generate_all_embeddings(
                        entity_id=entity_id,
                        entity_type="entity",  # Use 'entity' as entity_type for embeddings table
                        text_content=text_content,
                        image_data=None,
                        material_properties={}
                    )

                    if embedding_result and embedding_result.get('success'):
                        embeddings = embedding_result.get('embeddings', {})
                        text_embedding = embeddings.get('text_512')

                        if text_embedding:
                            embeddings_generated += 1
                            self.logger.info(
                                f"   ✅ Generated embedding for {entity_type} '{entity_name}' "
                                f"({len(text_content)} chars)"
                            )
                        else:
                            self.logger.warning(f"   ⚠️ No text embedding returned for entity {entity_id}")
                    else:
                        self.logger.warning(f"   ⚠️ Embedding generation failed for entity {entity_id}")

                except Exception as emb_error:
                    self.logger.error(f"   ❌ Failed to generate embedding for entity {entity_id}: {emb_error}")
                    continue

            self.logger.info(
                f"✅ Entity embedding generation complete: {embeddings_generated} embeddings generated "
                f"from {entities_processed} entities"
            )

            return {
                "embeddings_generated": embeddings_generated,
                "entities_processed": entities_processed,
                "total_entities": len(entities)
            }

        except Exception as e:
            self.logger.error(f"❌ Error generating entity embeddings: {e}")
            return {"embeddings_generated": 0, "entities_processed": 0, "error": str(e)}

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
            query = self.supabase.client.table("product_document_relationships")\
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
            query = self.supabase.client.table("document_entities")\
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


