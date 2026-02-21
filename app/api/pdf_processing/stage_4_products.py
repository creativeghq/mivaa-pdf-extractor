"""
Stage 4: Product Creation

This module handles product creation in the database for the product-centric pipeline.
Includes metadata consolidation from AI text extraction, visual analysis, and factory defaults.
"""

import logging
from typing import Dict, Any, List, Optional


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

    Implements Stage 4 metadata consolidation:
    1. AI text extraction metadata (from product.metadata)
    2. Visual metadata (from product images)
    3. Factory defaults (from catalog)

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

    # Start with AI-extracted metadata
    ai_metadata = product.metadata or {}

    # Add product-specific metadata
    if 'page_range' not in ai_metadata:
        ai_metadata['page_range'] = product.page_range
    if 'confidence' not in ai_metadata:
        ai_metadata['confidence'] = product.confidence
    if 'image_indices' not in ai_metadata:
        ai_metadata['image_indices'] = product.image_indices if product.image_indices is not None else []

    # Prepare factory defaults
    factory_defaults = {}
    if catalog.catalog_factory:
        factory_defaults['factory_name'] = catalog.catalog_factory
    if catalog.catalog_factory_group:
        factory_defaults['factory_group_name'] = catalog.catalog_factory_group

    # ✨ NEW: Fetch visual metadata from product images
    visual_metadata = await _fetch_visual_metadata_for_product(
        document_id=document_id,
        product_name=product.name,
        image_indices=product.image_indices,
        supabase=supabase,
        logger=logger
    )

    # ✨ NEW: Consolidate metadata from all sources
    try:
        from app.services.metadata.metadata_consolidation_service import MetadataConsolidationService

        consolidation_service = MetadataConsolidationService()
        consolidated_metadata = consolidation_service.consolidate_metadata(
            ai_metadata=ai_metadata,
            visual_metadata=visual_metadata,
            factory_defaults=factory_defaults
        )

        logger.info(f"   ✅ Consolidated metadata from {len(consolidated_metadata.get('_extraction_metadata', {}))} sources")
        metadata = consolidated_metadata

    except Exception as e:
        logger.warning(f"   ⚠️ Metadata consolidation failed, using AI metadata only: {e}")
        metadata = ai_metadata

    # Clean up "not found" values
    for key in ['factory_name', 'factory_group_name', 'material_category']:
        if is_not_found(metadata.get(key)):
            metadata[key] = None

    # ✅ FIX: Extract description from multiple sources if product.description is empty
    description = product.description or ''
    if not description.strip():
        # Try metadata.design.philosophy.value first (most common source)
        design = metadata.get('design', {})
        if isinstance(design, dict):
            philosophy = design.get('philosophy', {})
            if isinstance(philosophy, dict) and philosophy.get('value'):
                description = philosophy['value']
                logger.info(f"   📝 Extracted description from design.philosophy: {description[:50]}...")
            elif design.get('inspiration', {}).get('value'):
                description = design['inspiration']['value']
                logger.info(f"   📝 Extracted description from design.inspiration: {description[:50]}...")

        # Try metadata.description directly
        if not description.strip() and metadata.get('description'):
            meta_desc = metadata.get('description')
            if isinstance(meta_desc, dict) and meta_desc.get('value'):
                description = meta_desc['value']
            elif isinstance(meta_desc, str):
                description = meta_desc
            if description:
                logger.info(f"   📝 Extracted description from metadata.description: {description[:50]}...")

    product_data = {
        'source_document_id': document_id,
        'workspace_id': workspace_id,
        'name': product.name,
        'description': description,
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


async def _fetch_visual_metadata_for_product(
    document_id: str,
    product_name: str,
    image_indices: Optional[List[int]],
    supabase: Any,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Fetch and aggregate visual metadata from product images.

    Args:
        document_id: Document identifier
        product_name: Product name
        image_indices: List of image indices for this product
        supabase: Supabase client
        logger: Logger instance

    Returns:
        Aggregated visual metadata from all product images
    """
    try:
        if not image_indices or len(image_indices) == 0:
            logger.debug(f"   ℹ️ No images for product {product_name}, skipping visual metadata")
            return {}

        # Fetch images for this product
        images_response = supabase.client.table('document_images') \
            .select('id, metadata') \
            .eq('document_id', document_id) \
            .execute()

        if not images_response.data:
            logger.debug(f"   ℹ️ No images found in database for product {product_name}")
            return {}

        # Aggregate visual metadata from all images
        aggregated_visual_metadata = {}
        images_with_visual_data = 0

        for image in images_response.data:
            image_metadata = image.get('metadata', {})
            visual_analysis = image_metadata.get('visual_analysis', {})

            if visual_analysis:
                images_with_visual_data += 1
                # Merge visual metadata (take highest confidence values)
                for key, value_data in visual_analysis.items():
                    if isinstance(value_data, dict) and 'primary' in value_data:
                        confidence = value_data.get('confidence', 0.0)
                        existing_confidence = aggregated_visual_metadata.get(key, {}).get('confidence', 0.0)

                        if confidence > existing_confidence:
                            aggregated_visual_metadata[key] = value_data

        if images_with_visual_data > 0:
            logger.info(f"   ✅ Aggregated visual metadata from {images_with_visual_data} images")
        else:
            logger.debug(f"   ℹ️ No visual metadata found in images for product {product_name}")

        return aggregated_visual_metadata

    except Exception as e:
        logger.warning(f"   ⚠️ Failed to fetch visual metadata: {e}")
        return {}


async def propagate_common_fields_to_products(
    document_id: str,
    supabase: Any,
    logger: logging.Logger,
    material_category_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Propagate common fields (factory, manufacturing, material_category) across all products
    from the same document. If one product has factory info and others don't, share it.

    Common fields to propagate:
    - factory_name
    - factory_group_name
    - country_of_origin / origin
    - material_category (from upload settings - ALWAYS applied if provided)

    Args:
        document_id: Document identifier
        supabase: Supabase client
        logger: Logger instance
        material_category_override: Material category from upload settings (highest priority)

    Returns:
        Stats about propagation
    """
    logger.info(f"🔄 Propagating common fields across products from document {document_id}...")

    stats = {
        'products_checked': 0,
        'products_updated': 0,
        'fields_propagated': []
    }

    try:
        # If material_category_override not provided, try to get from document/job metadata
        if not material_category_override:
            # Try to get from background_jobs metadata first
            job_result = supabase.client.table('background_jobs').select('metadata').eq('document_id', document_id).order('created_at', desc=True).limit(1).execute()
            if job_result.data and len(job_result.data) > 0:
                job_metadata = job_result.data[0].get('metadata', {})
                material_category_override = job_metadata.get('material_category')

            # Fallback: try to get from documents metadata
            if not material_category_override:
                doc_result = supabase.client.table('documents').select('metadata').eq('id', document_id).execute()
                if doc_result.data and len(doc_result.data) > 0:
                    doc_metadata = doc_result.data[0].get('metadata', {})
                    material_category_override = doc_metadata.get('material_category')

        if material_category_override:
            logger.info(f"   📦 Using material_category from upload: {material_category_override}")

        # Fetch all products from this document
        products_response = supabase.client.table('products') \
            .select('id, name, metadata') \
            .eq('source_document_id', document_id) \
            .execute()

        if not products_response.data or len(products_response.data) == 0:
            logger.info("   ℹ️ No products found for document")
            return stats

        products = products_response.data
        stats['products_checked'] = len(products)

        # Top-level fields to propagate (shared across all products from same document)
        common_fields = [
            'factory_name',
            'factory_group_name',
            'country_of_origin',
            'origin',
            'material_category',
            # Manufacturing details
            'manufacturing_location',
            'manufacturing_process',
            'manufacturing_country',
            # Dimensions — catalog siblings share the same size options
            'available_sizes',
        ]

        # Nested fields to propagate: (parent_key, child_key)
        # Tiles/stones from the same catalog series share these material-level properties.
        nested_fields = [
            ('material_properties', 'thickness'),
            ('material_properties', 'body_type'),
            ('material_properties', 'composition'),
        ]

        # Find the best value for each common field (first non-empty value)
        common_values = {}

        # ALWAYS use material_category from upload if provided
        if material_category_override and not _is_empty_value(material_category_override):
            common_values['material_category'] = material_category_override
            logger.info(f"   ✅ material_category set from upload: {material_category_override}")

        for field in common_fields:
            # Skip material_category if we already have it from upload
            if field == 'material_category' and 'material_category' in common_values:
                continue

            for product in products:
                metadata = product.get('metadata', {}) or {}
                value = metadata.get(field)

                # Skip empty/invalid values
                if value and not _is_empty_value(value):
                    common_values[field] = value
                    break  # Use first valid value found

        # Find the best nested field values (first non-empty across all products)
        nested_common_values = {}
        for parent_key, child_key in nested_fields:
            for product in products:
                metadata = product.get('metadata', {}) or {}
                parent = metadata.get(parent_key) or {}
                if not isinstance(parent, dict):
                    continue
                value = parent.get(child_key)
                if value and not _is_empty_value(value):
                    nested_common_values[(parent_key, child_key)] = value
                    break

        if not common_values and not nested_common_values:
            logger.info("   ℹ️ No common values to propagate")
            return stats

        all_found = list(common_values.keys()) + [f"{pk}.{ck}" for pk, ck in nested_common_values.keys()]
        logger.info(f"   📦 Found common values: {all_found}")

        # Update products that are missing these common fields (one DB write per product)
        for product in products:
            product_id = product['id']
            metadata = product.get('metadata', {}) or {}
            updates_needed = {}
            nested_updates = {}

            for field, common_value in common_values.items():
                current_value = metadata.get(field)
                if _is_empty_value(current_value) and not _is_empty_value(common_value):
                    updates_needed[field] = common_value

            for (parent_key, child_key), common_value in nested_common_values.items():
                parent = metadata.get(parent_key) or {}
                if not isinstance(parent, dict):
                    continue
                current_value = parent.get(child_key)
                if _is_empty_value(current_value) and not _is_empty_value(common_value):
                    nested_updates[(parent_key, child_key)] = common_value

            if updates_needed or nested_updates:
                updated_metadata = {**metadata, **updates_needed}

                # Apply one-level-deep nested updates
                for (parent_key, child_key), value in nested_updates.items():
                    parent = dict(updated_metadata.get(parent_key) or {})
                    parent[child_key] = value
                    updated_metadata[parent_key] = parent

                supabase.client.table('products') \
                    .update({'metadata': updated_metadata}) \
                    .eq('id', product_id) \
                    .execute()

                stats['products_updated'] += 1
                propagated = list(updates_needed.keys()) + [f"{pk}.{ck}" for pk, ck in nested_updates.keys()]
                stats['fields_propagated'].extend(propagated)
                logger.info(f"   ✅ Updated product {product['name']}: {propagated}")

        # Deduplicate fields_propagated
        stats['fields_propagated'] = list(set(stats['fields_propagated']))

        logger.info(f"✅ Propagation complete: {stats['products_updated']}/{stats['products_checked']} products updated")
        return stats

    except Exception as e:
        logger.error(f"❌ Failed to propagate common fields: {e}")
        stats['error'] = str(e)
        return stats


async def extract_dimensions_from_document_chunks(
    document_id: str,
    supabase: Any,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Stage 4.6: scan all text chunks belonging to this document for dimension patterns
    (tile sizes like "10×45 cm", thickness like "6.9 mm") and fill in any products
    that are still missing those fields after Stage 4.5 propagation.

    Runs entirely on already-extracted text — no extra AI calls needed.

    Returns:
        Stats dict: products_checked, products_updated, dimensions_found
    """
    import re

    stats: Dict[str, Any] = {
        'products_checked': 0,
        'products_updated': 0,
        'dimensions_found': [],
    }

    try:
        # ── 1. Fetch all text chunks for this document ──────────────────────
        chunks_resp = supabase.client.table('document_chunks') \
            .select('content') \
            .eq('document_id', document_id) \
            .execute()
        chunks = chunks_resp.data or []

        if not chunks:
            logger.info("   ℹ️ No text chunks found for dimension scan")
            return stats

        # Merge all chunk content for scanning
        combined_text = ' '.join(c.get('content', '') for c in chunks if c.get('content'))

        # ── 2. Extract size patterns: 10×45 cm, 15 x 38, 20X60 ────────────
        size_re = re.compile(r'(\d{1,4})\s*[×xX]\s*(\d{1,4})(?:\s*cm)?', re.IGNORECASE)
        found_sizes = []
        for m in size_re.finditer(combined_text):
            w, h = int(m.group(1)), int(m.group(2))
            if 5 <= w <= 300 and 5 <= h <= 300:
                found_sizes.append(f"{w}×{h} cm")
        found_sizes = list(dict.fromkeys(found_sizes))  # dedup, preserve order

        # ── 3. Extract thickness near keyword ─────────────────────────────
        thickness_value: Optional[str] = None
        thick_re = re.compile(
            r'(?:thickness|spessore|epaisseur|st[aä]rke|dikte)[^\n]{0,80}?(\d+[\.,]?\d*)\s*mm',
            re.IGNORECASE,
        )
        m = thick_re.search(combined_text)
        if m:
            thickness_value = f"{m.group(1).replace(',', '.')}mm"
        else:
            bare_mm = re.findall(r'\b(\d+[\.,]\d+)\s*mm\b', combined_text, re.IGNORECASE)
            if bare_mm:
                thickness_value = f"{bare_mm[0].replace(',', '.')}mm"

        if not found_sizes and not thickness_value:
            logger.info("   ℹ️ No dimension patterns found in text chunks")
            return stats

        logger.info(
            f"   📐 Dimension patterns found in text: sizes={found_sizes}, thickness={thickness_value}"
        )
        if found_sizes:
            stats['dimensions_found'].extend(found_sizes)
        if thickness_value:
            stats['dimensions_found'].append(thickness_value)

        # ── 4. Fetch products for this document ───────────────────────────
        products_resp = supabase.client.table('products') \
            .select('id, name, metadata') \
            .eq('source_document_id', document_id) \
            .execute()
        products = products_resp.data or []
        stats['products_checked'] = len(products)

        # ── 5. Fill products that are still missing the fields ────────────
        for product in products:
            product_id = product['id']
            existing_metadata = product.get('metadata') or {}
            updated_metadata = {**existing_metadata}
            changed = False

            if found_sizes and _is_empty_value(existing_metadata.get('available_sizes')):
                updated_metadata['available_sizes'] = found_sizes
                changed = True

            if thickness_value:
                mat_props = dict(existing_metadata.get('material_properties') or {})
                if _is_empty_value(mat_props.get('thickness')):
                    mat_props['thickness'] = {
                        'value': thickness_value,
                        'confidence': 0.65,
                        'source': 'document_text',
                    }
                    updated_metadata['material_properties'] = mat_props
                    changed = True

            if changed:
                supabase.client.table('products') \
                    .update({'metadata': updated_metadata}) \
                    .eq('id', product_id) \
                    .execute()
                stats['products_updated'] += 1
                logger.info(
                    f"   ✅ Filled dimensions for product '{product['name']}' from text chunks"
                )

        logger.info(
            f"✅ Dimension extraction complete: {stats['products_updated']}/{stats['products_checked']} products updated"
        )
        return stats

    except Exception as e:
        logger.error(f"❌ Dimension extraction from text chunks failed: {e}")
        stats['error'] = str(e)
        return stats


def _is_empty_value(value) -> bool:
    """Check if a value is empty or a placeholder."""
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.lower().strip()
        return normalized in ['', 'n/a', 'not found', 'not explicitly mentioned', 'not mentioned', 'unknown', 'none']
    if isinstance(value, list):
        return len(value) == 0
    if isinstance(value, dict):
        # Check if it's a {value, confidence} object with empty value
        if 'value' in value:
            return _is_empty_value(value['value'])
        return len(value) == 0
    return False
