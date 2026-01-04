"""
Stage 4: Product Creation

This module handles product creation in the database for the product-centric pipeline.
"""

import logging
from typing import Dict, Any


async def create_single_product(
    product: Any,
    document_id: str,
    workspace_id: str,
    job_id: str,
    catalog: Any,
    supabase: Any,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Create a single product in the database (product-centric pipeline).

    Args:
        product: Single product object from catalog
        document_id: Document identifier
        workspace_id: Workspace identifier
        job_id: Job identifier
        catalog: Full catalog (for factory info inheritance)
        supabase: Supabase client
        logger: Logger instance

    Returns:
        Dictionary with product_id
    """
    logger.info(f"🏭 Creating product in database: {product.name}")

    def is_not_found(val):
        if not val:
            return True
        if isinstance(val, str):
            normalized = val.lower().strip()
            return normalized in ['not found', 'not explicitly mentioned', 'not mentioned', 'n/a', 'unknown', '']
        return False

    metadata = product.metadata or {}

    if 'page_range' not in metadata:
        metadata['page_range'] = product.page_range
    if 'confidence' not in metadata:
        metadata['confidence'] = product.confidence
    if 'image_indices' not in metadata:
        metadata['image_indices'] = product.image_indices if product.image_indices is not None else []

    if is_not_found(metadata.get('factory_name')) and catalog.catalog_factory:
        metadata['factory_name'] = catalog.catalog_factory

    if is_not_found(metadata.get('factory_group_name')) and catalog.catalog_factory_group:
        metadata['factory_group_name'] = catalog.catalog_factory_group

    for key in ['factory_name', 'factory_group_name', 'material_category']:
        if is_not_found(metadata.get(key)):
            metadata[key] = None

    product_data = {
        'source_document_id': document_id,
        'workspace_id': workspace_id,
        'name': product.name,
        'description': product.description or '',
        'metadata': metadata,
        'source_type': 'pdf_processing',
        'source_job_id': job_id
    }

    result = supabase.client.table('products').insert(product_data).execute()

    if result.data and len(result.data) > 0:
        product_id = result.data[0]['id']
        logger.info(f"   ✅ Created product in DB: {product.name} (ID: {product_id})")
        return {'product_id': product_id}
    else:
        raise Exception(f"Failed to create product {product.name} in database")
