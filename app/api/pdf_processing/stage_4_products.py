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
    metadata_consolidation_count = 0
    metadata_consolidation_failed = 0
    metadata_consolidation_ai_calls = 0

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

            # Consolidate metadata from multiple sources
            try:
                from app.services.metadata_consolidation_service import MetadataConsolidationService
                consolidation_service = MetadataConsolidationService(workspace_id=workspace_id)

                # Collect metadata from all sources
                metadata_sources = {
                    "ai_text_extraction": metadata,  # From product discovery
                    "factory_defaults": {}
                }

                # Add factory defaults if available
                if catalog.catalog_factory:
                    metadata_sources["factory_defaults"]["factory_name"] = catalog.catalog_factory
                if catalog.catalog_factory_group:
                    metadata_sources["factory_defaults"]["factory_group_name"] = catalog.catalog_factory_group

                # Get visual metadata from product images
                if product.image_indices:
                    visual_metadata_list = []
                    for img_idx in product.image_indices:
                        try:
                            # Fetch image visual_metadata
                            img_result = supabase.client.table('document_images') \
                                .select('visual_metadata') \
                                .eq('document_id', document_id) \
                                .eq('page_number', img_idx) \
                                .limit(1) \
                                .execute()

                            if img_result.data and img_result.data[0].get('visual_metadata'):
                                visual_metadata_list.append(img_result.data[0]['visual_metadata'])
                        except Exception as e:
                            logger.debug(f"   ⚠️ Could not fetch visual metadata for image {img_idx}: {e}")

                    # Merge visual metadata from all product images
                    if visual_metadata_list:
                        merged_visual = {}
                        for vm in visual_metadata_list:
                            for field, data in vm.items():
                                if field not in merged_visual or data.get('confidence', 0) > merged_visual[field].get('confidence', 0):
                                    merged_visual[field] = data

                        # Extract primary values for consolidation
                        visual_metadata_extracted = {}
                        for field, data in merged_visual.items():
                            if isinstance(data, dict) and 'primary' in data:
                                visual_metadata_extracted[field] = data['primary']

                        if visual_metadata_extracted:
                            metadata_sources["visual_embeddings"] = visual_metadata_extracted

                # Consolidate all metadata sources
                consolidated_metadata = await consolidation_service.consolidate_metadata(
                    product_id=product.name,  # Use name as temp ID
                    sources=metadata_sources,
                    existing_metadata=metadata
                )
                metadata_consolidation_ai_calls += 1

                # Use consolidated metadata
                metadata = consolidated_metadata
                metadata_consolidation_count += 1
                logger.info(f"   ✅ Consolidated metadata for {product.name} (confidence: {metadata.get('_overall_confidence', 0):.2f})")

            except Exception as e:
                metadata_consolidation_failed += 1
                logger.warning(f"   ⚠️ Metadata consolidation failed for {product.name}, using original metadata: {e}")
                # Send to Sentry
                try:
                    import sentry_sdk
                    sentry_sdk.capture_exception(e)
                except:
                    pass

            product_data = {
                'source_document_id': document_id,
                'workspace_id': workspace_id,
                'name': product.name,
                'description': product.description or '',
                'metadata': metadata,  # ALL product metadata stored here (now consolidated)
                'source_type': 'pdf_processing',  # ✅ NEW: Track source type
                'source_job_id': job_id  # ✅ NEW: Track source job
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

    # Stage 4 progress: 70% → 85% (fixed when complete)
    await tracker.update_stage(
        ProcessingStage.FINALIZING,
        stage_name="product_creation",
        progress_percentage=85
    )

    await tracker._sync_to_database(stage="product_creation")

    logger.info(f"✅ [STAGE 4] Product Creation & Linking Complete")
    logger.info(f"   Products Created: {products_created}")
    logger.info(f"   Metadata Consolidation: {metadata_consolidation_count} successful, {metadata_consolidation_failed} failed")
    logger.info(f"   AI Calls for Consolidation: {metadata_consolidation_ai_calls}")
    logger.info(f"📊 Progress updated: 85% (Stage 4 complete - {products_created} products created)")

    # Create PRODUCTS_CREATED checkpoint
    checkpoint_metadata = {
        "entity_links": linking_results,
        "product_names": [p.name for p in catalog.products],
        "metadata_consolidation_count": metadata_consolidation_count,  # ✅ NEW
        "metadata_consolidation_failed": metadata_consolidation_failed,  # ✅ NEW
        "metadata_consolidation_ai_calls": metadata_consolidation_ai_calls  # ✅ NEW
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

    # Create PRODUCTS_CREATED checkpoint
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

    # Create DOCUMENT_ENTITIES_CREATED checkpoint (if entities were created)
    if entities_created > 0:
        await checkpoint_recovery_service.create_checkpoint(
            job_id=job_id,
            stage=CheckpointStage.DOCUMENT_ENTITIES_CREATED,
            data={
                "document_id": document_id,
                "entities_created": entities_created,
                "entity_product_relationships": entity_product_relationships,
                "entity_embeddings_generated": entity_embeddings_generated
            },
            metadata={
                "entity_types": {
                    "certificates": len([e for e in all_entities if e.entity_type == 'certificate']),
                    "logos": len([e for e in all_entities if e.entity_type == 'logo']),
                    "specifications": len([e for e in all_entities if e.entity_type == 'specification'])
                },
                "extract_categories": extract_categories
            }
        )
        logger.info(f"✅ Created DOCUMENT_ENTITIES_CREATED checkpoint for job {job_id}")

    # Create METADATA_EXTRACTED checkpoint (metadata consolidation metrics)
    if metadata_consolidation_count > 0 or products_created > 0:
        # Get factory metadata from first product if available
        factory_metadata = {}
        if catalog.products and len(catalog.products) > 0:
            first_product = catalog.products[0]
            if hasattr(first_product, 'metadata') and first_product.metadata:
                factory_metadata = {
                    "factory_name": first_product.metadata.get('factory_name'),
                    "factory_group": first_product.metadata.get('factory_group'),
                    "manufacturer": first_product.metadata.get('manufacturer'),
                    "country_of_origin": first_product.metadata.get('country_of_origin')
                }

        await checkpoint_recovery_service.create_checkpoint(
            job_id=job_id,
            stage=CheckpointStage.METADATA_EXTRACTED,
            data={
                "document_id": document_id,
                "metadata_consolidation_count": metadata_consolidation_count,
                "metadata_consolidation_failed": metadata_consolidation_failed,
                "products_with_metadata": products_created
            },
            metadata={
                "factory_metadata": factory_metadata,
                "metadata_consolidation_ai_calls": metadata_consolidation_ai_calls,
                "avg_metadata_fields_per_product": metadata_consolidation_count / products_created if products_created > 0 else 0
            }
        )
        logger.info(f"✅ Created METADATA_EXTRACTED checkpoint for job {job_id}")

    # Force garbage collection after product creation to free memory
    import gc
    gc.collect()
    logger.info("💾 Memory freed after Stage 4 (Product Creation)")

    return {
        "products_created": products_created,
        "entities_created": entities_created,
        "entity_product_relationships": entity_product_relationships,
        "entity_embeddings_generated": entity_embeddings_generated,
        "linking_results": linking_results,
        "metadata_consolidation_count": metadata_consolidation_count,  # ✅ NEW
        "metadata_consolidation_failed": metadata_consolidation_failed,  # ✅ NEW
        "metadata_consolidation_ai_calls": metadata_consolidation_ai_calls  # ✅ NEW
    }
