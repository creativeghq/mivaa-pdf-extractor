"""
Stage 4: Product Creation & Linking

This module handles product creation in database and entity linking.
"""

import logging
from typing import Dict, Any
from app.schemas.jobs import ProcessingStage
from app.services.checkpoint_recovery_service import ProcessingStage as CheckpointStage

logger = logging.getLogger(__name__)


async def process_stage_4_products(
    document_id: str,
    workspace_id: str,
    job_id: str,
    catalog: Any,
    extract_categories: list,
    product_creation_model: str,
    tracker: Any,
    checkpoint_recovery_service: Any,
    supabase: Any,
    logger: Any
) -> Dict[str, Any]:
    """
    Stage 4: Product Creation & Linking
    
    Creates products and document entities in database, then links them to images and chunks.
    
    Args:
        document_id: Unique document identifier
        workspace_id: Workspace identifier
        job_id: Job identifier for tracking
        catalog: Product catalog from Stage 0
        extract_categories: Categories extracted
        tracker: Job progress tracker
        checkpoint_recovery_service: Checkpoint service
        supabase: Supabase client
        logger: Logger instance
        
    Returns:
        Dictionary containing:
        - products_created: Number of products created
        - entities_created: Number of document entities created
        - linking_results: Entity linking results
    """
    from app.services.entity_linking_service import EntityLinkingService
    from app.services.document_entity_service import DocumentEntityService
    
    logger.info("🏭 [STAGE 4] Product Creation & Linking - Starting...")
    await tracker.update_stage(ProcessingStage.FINALIZING, stage_name="product_creation")
    
    # Create products in database (with metadata)
    products_created = 0
    product_id_map = {}  # Map product name to product ID for entity linking
    
    # Helper to check if a value is a "not found" placeholder
    def is_not_found(val):
        if not val:
            return True
        if isinstance(val, str):
            normalized = val.lower().strip()
            return normalized in ['not found', 'not explicitly mentioned', 'not mentioned', 'n/a', 'unknown', '']
        return False

    for product in catalog.products:
        try:
            # Use product.metadata field (new architecture - products + metadata inseparable)
            metadata = product.metadata or {}

            # Ensure page_range, confidence, and image_indices are in metadata
            if 'page_range' not in metadata:
                metadata['page_range'] = product.page_range
            if 'confidence' not in metadata:
                metadata['confidence'] = product.confidence
            # ALWAYS save image_indices (even if None or empty list)
            if 'image_indices' not in metadata:
                metadata['image_indices'] = product.image_indices if product.image_indices is not None else []

            # Inherit catalog-level factory info if product doesn't have it
            if is_not_found(metadata.get('factory_name')) and catalog.catalog_factory:
                metadata['factory_name'] = catalog.catalog_factory
                logger.debug(f"   📋 Inherited factory_name '{catalog.catalog_factory}' for {product.name}")

            if is_not_found(metadata.get('factory_group_name')) and catalog.catalog_factory_group:
                metadata['factory_group_name'] = catalog.catalog_factory_group
                logger.debug(f"   📋 Inherited factory_group_name '{catalog.catalog_factory_group}' for {product.name}")

            # Clean up "not found" values - set to None instead
            for key in ['factory_name', 'factory_group_name', 'material_category']:
                if is_not_found(metadata.get(key)):
                    metadata[key] = None

            product_data = {
                'source_document_id': document_id,
                'workspace_id': workspace_id,
                'name': product.name,
                'description': product.description or '',
                'metadata': metadata  # ALL product metadata stored here
            }

            result = supabase.client.table('products').insert(product_data).execute()

            if result.data and len(result.data) > 0:
                product_id = result.data[0]['id']
                product_id_map[product.name] = product_id
                products_created += 1
                logger.info(f"   ✅ Created product: {product.name} (ID: {product_id})")

        except Exception as e:
            logger.error(f"Failed to create product {product.name}: {e}")
    
    tracker.products_created = products_created
    logger.info(f"   Products created: {products_created}")
    
    # Save document entities (certificates, logos, specifications) if discovered
    document_entity_service = DocumentEntityService(supabase.client)
    entities_created = 0
    entity_id_map = {}  # Map entity name to entity ID
    
    # Combine all entities from catalog
    all_entities = []
    
    if "certificates" in extract_categories and catalog.certificates:
        from app.services.document_entity_service import DocumentEntity
        for cert in catalog.certificates:
            entity = DocumentEntity(
                entity_type="certificate",
                name=cert.name,
                page_range=cert.page_range,
                description=f"{cert.certificate_type or 'Certificate'} issued by {cert.issuer or 'Unknown'}",
                metadata={
                    "certificate_type": cert.certificate_type,
                    "issuer": cert.issuer,
                    "issue_date": cert.issue_date,
                    "expiry_date": cert.expiry_date,
                    "standards": cert.standards or [],
                    "confidence": cert.confidence
                }
            )
            all_entities.append(entity)
    
    if "logos" in extract_categories and catalog.logos:
        from app.services.document_entity_service import DocumentEntity
        for logo in catalog.logos:
            entity = DocumentEntity(
                entity_type="logo",
                name=logo.name,
                page_range=logo.page_range,
                description=logo.description,
                metadata={
                    "logo_type": logo.logo_type,
                    "confidence": logo.confidence
                }
            )
            all_entities.append(entity)
    
    if "specifications" in extract_categories and catalog.specifications:
        from app.services.document_entity_service import DocumentEntity
        for spec in catalog.specifications:
            entity = DocumentEntity(
                entity_type="specification",
                name=spec.name,
                page_range=spec.page_range,
                description=spec.description,
                metadata={
                    "spec_type": spec.spec_type,
                    "confidence": spec.confidence
                }
            )
            all_entities.append(entity)

    # Save all entities to database
    if all_entities:
        entity_ids = await document_entity_service.save_entities(
            entities=all_entities,
            source_document_id=document_id,
            workspace_id=workspace_id
        )
        entities_created = len(entity_ids)

        # Map entity names to IDs
        for entity, entity_id in zip(all_entities, entity_ids):
            entity_id_map[entity.name] = entity_id

        logger.info(f"   Document entities created: {entities_created}")
        logger.info(f"     - Certificates: {len([e for e in all_entities if e.entity_type == 'certificate'])}")
        logger.info(f"     - Logos: {len([e for e in all_entities if e.entity_type == 'logo'])}")
        logger.info(f"     - Specifications: {len([e for e in all_entities if e.entity_type == 'specification'])}")

    # Link entities (images, chunks, products)
    linking_service = EntityLinkingService()
    linking_results = await linking_service.link_all_entities(
        document_id=document_id,
        catalog=catalog
    )

    logger.info(f"   Entity linking complete:")
    logger.info(f"     - Image-to-product links: {linking_results['image_product_links']}")
    logger.info(f"     - Image-to-chunk links: {linking_results['image_chunk_links']}")

    # ✅ NEW: Match document entities to products and generate embeddings
    entity_product_relationships = 0
    entity_embeddings_generated = 0
    if entities_created > 0:
        logger.info(f"🔗 Matching {entities_created} document entities to products...")
        from app.services.document_entity_service import DocumentEntityService
        entity_service = DocumentEntityService(supabase.client)

        # Match entities to products
        matching_results = await entity_service.match_entities_to_products(
            document_id=document_id,
            workspace_id=workspace_id
        )

        entity_product_relationships = matching_results.get('relationships_created', 0)
        logger.info(f"   ✅ Created {entity_product_relationships} entity-product relationships")

        # Generate embeddings for entities
        logger.info(f"🎨 Generating embeddings for {entities_created} document entities...")
        embedding_results = await entity_service.generate_entity_embeddings(
            document_id=document_id,
            workspace_id=workspace_id
        )

        entity_embeddings_generated = embedding_results.get('embeddings_generated', 0)
        logger.info(f"   ✅ Generated {entity_embeddings_generated} entity embeddings")

    await tracker._sync_to_database(stage="product_creation")

    logger.info(f"✅ [STAGE 4] Product Creation & Linking Complete")

    # Create PRODUCTS_CREATED checkpoint
    checkpoint_metadata = {
        "entity_links": linking_results,
        "product_names": [p.name for p in catalog.products]
    }

    # Add document entity info if any were created
    if entities_created > 0:
        checkpoint_metadata["document_entities_created"] = entities_created
        checkpoint_metadata["entity_types"] = {
            "certificates": len([e for e in all_entities if e.entity_type == 'certificate']),
            "logos": len([e for e in all_entities if e.entity_type == 'logo']),
            "specifications": len([e for e in all_entities if e.entity_type == 'specification'])
        }
        checkpoint_metadata["entity_product_relationships"] = entity_product_relationships
        checkpoint_metadata["entity_embeddings_generated"] = entity_embeddings_generated  # ✅ NEW

    await checkpoint_recovery_service.create_checkpoint(
        job_id=job_id,
        stage=CheckpointStage.PRODUCTS_CREATED,
        data={
            "document_id": document_id,
            "products_created": products_created,
            "document_entities_created": entities_created
        },
        metadata=checkpoint_metadata
    )
    logger.info(f"✅ Created PRODUCTS_CREATED checkpoint for job {job_id}")

    # Force garbage collection after product creation to free memory
    import gc
    gc.collect()
    logger.info("💾 Memory freed after Stage 4 (Product Creation)")

    return {
        "products_created": products_created,
        "entities_created": entities_created,
        "entity_product_relationships": entity_product_relationships,
        "entity_embeddings_generated": entity_embeddings_generated,  # ✅ NEW
        "linking_results": linking_results
    }
