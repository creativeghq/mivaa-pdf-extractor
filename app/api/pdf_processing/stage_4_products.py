"""
Stage 4: Product Creation

This module handles product creation in the database for the product-centric pipeline.
Includes metadata consolidation from AI text extraction, visual analysis, and factory defaults.
"""

import logging
import os
import httpx
from typing import Dict, Any, List, Optional

# ── Category → default unit mapping (mirrors material_categories.default_unit) ─
_CATEGORY_DEFAULT_UNITS: Dict[str, str] = {
    'tiles': 'sqm',
    'wood': 'sqm',
    'decor': 'pcs',
    'furniture': 'pcs',
    'general_materials': 'pcs',
    'paint_wall_decor': 'sqm',
    'heating': 'pcs',
    'sanitary': 'pcs',
    'kitchen': 'pcs',
    'lighting': 'pcs',
}


def _resolve_default_unit(material_category: Optional[str]) -> str:
    """Resolve default unit from material category. Falls back to 'pcs'."""
    if not material_category:
        return 'pcs'
    cat = material_category.lower().strip()
    # Direct match
    if cat in _CATEGORY_DEFAULT_UNITS:
        return _CATEGORY_DEFAULT_UNITS[cat]
    # Fuzzy match: check if category contains a known key
    for key, unit in _CATEGORY_DEFAULT_UNITS.items():
        if key in cat or cat in key:
            return unit
    return 'pcs'


# ── Factory field keys (canonical set) ───────────────────────────────────────
_FACTORY_FIELDS = [
    'factory_name', 'factory_group_name', 'address', 'city', 'country',
    'postal_code', 'phone', 'email', 'website', 'country_of_origin',
    'founded_year', 'company_type', 'linkedin_url', 'employee_count',
]


def _build_factory_object(metadata: Dict[str, Any], factory_defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assemble a canonical `factory` nested object from AI-extracted metadata
    and catalog-level factory defaults.

    Priority (highest first):
      1. metadata.factory (already nested — from previous run or direct extraction)
      2. flat metadata fields (factory_name, factory_group_name, …)
      3. factory_defaults (from catalog.catalog_factory / catalog_factory_group)
    """
    # Start from existing nested object if present
    existing = metadata.get('factory') or {}
    if not isinstance(existing, dict):
        existing = {}

    factory: Dict[str, Any] = {}

    # Layer 3: defaults
    for k, v in factory_defaults.items():
        if v and not _is_empty_value(v):
            factory[k] = v

    # Layer 2: flat metadata fields
    for field in _FACTORY_FIELDS:
        val = metadata.get(field)
        if val and not _is_empty_value(val):
            factory[field] = val

    # Layer 1: existing nested object wins for non-empty values
    for k, v in existing.items():
        if v and not _is_empty_value(v):
            factory[k] = v

    return factory


async def _trigger_factory_enrichment(
    workspace_id: str,
    product_ids: List[str],
    scope_column: str,
    scope_value: str,
    logger: logging.Logger,
) -> None:
    """
    Fire-and-forget call to the trigger-factory-enrichment edge function.
    Queues a background agent job if completeness is below threshold.
    Never raises — enrichment is best-effort.
    """
    supabase_url = os.getenv("SUPABASE_URL", "")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not supabase_url or not service_role_key:
        return

    url = f"{supabase_url}/functions/v1/trigger-factory-enrichment"
    payload = {
        "workspace_id": workspace_id,
        "product_ids": product_ids,
        "scope_column": scope_column,
        "scope_value": scope_value,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {service_role_key}",
                    "Content-Type": "application/json",
                },
            )
            data = resp.json()
            logger.info(
                f"   🏭 Factory enrichment trigger: propagated={data.get('propagated', 0)}, "
                f"queued_job={data.get('queued_job_id') or 'none'}"
            )
    except Exception as exc:
        logger.warning(f"   ⚠️ Factory enrichment trigger failed (non-blocking): {exc}")

# ── Controlled vocabulary ────────────────────────────────────────────────────
MATERIAL_CATEGORY_VOCAB = {
    "floor_tile", "wall_tile", "wood_flooring", "laminate", "vinyl_flooring", "carpet",
    "wall_paint", "wallpaper", "countertop", "kitchen_worktop", "bathroom_tile", "shower_tile",
    "sofa", "armchair", "dining_chair", "accent_chair", "rug", "curtain", "cushion",
    "dining_table", "coffee_table", "side_table", "cabinet", "shelving", "sideboard",
    "door", "window", "fabric_swatch", "leather_swatch", "stone_slab", "metal_panel",
    "glass_panel", "outdoor_furniture", "lighting",
}
ZONE_INTENT_VOCAB = {"surface", "full_object", "upholstery", "sub_element"}


async def _classify_product(name: str, description: str, existing_category: str) -> Dict[str, str]:
    """
    Call Claude Haiku to assign material_category + zone_intent from controlled vocabulary.
    Returns dict with keys 'material_category' and 'zone_intent', or empty dict on failure.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {}

    import httpx, json as _json

    prompt = f"""Classify this interior material/furniture product into the controlled vocabularies below.

Product name: {name}
Description: {description or 'N/A'}
Current category (may be wrong or missing): {existing_category or 'N/A'}

CONTROLLED VOCABULARY — respond ONLY with a JSON object, no explanation:

material_category (pick exactly one):
  floor_tile | wall_tile | wood_flooring | laminate | vinyl_flooring | carpet |
  wall_paint | wallpaper | countertop | kitchen_worktop | bathroom_tile | shower_tile |
  sofa | armchair | dining_chair | accent_chair | rug | curtain | cushion |
  dining_table | coffee_table | side_table | cabinet | shelving | sideboard |
  door | window | fabric_swatch | leather_swatch | stone_slab | metal_panel |
  glass_panel | outdoor_furniture | lighting

zone_intent (pick exactly one):
  surface     — floor/wall/ceiling tiles, paint, wallpaper, countertops, cladding
  full_object — sofa, chair, rug, curtain, table, cabinet, lamp
  upholstery  — fabric/leather swatches for covering furniture
  sub_element — hardware, handles, trims, brackets

Respond with exactly: {{"material_category": "...", "zone_intent": "..."}}"""

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            text = resp.json()["content"][0]["text"].strip()
            data = _json.loads(text)
            result = {}
            if data.get("material_category") in MATERIAL_CATEGORY_VOCAB:
                result["material_category"] = data["material_category"]
            if data.get("zone_intent") in ZONE_INTENT_VOCAB:
                result["zone_intent"] = data["zone_intent"]
            return result
    except Exception as e:
        logging.getLogger(__name__).warning(f"Product classification failed: {e}")
        return {}


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

    # ── Assemble canonical factory nested object ──────────────────────────
    factory_obj = _build_factory_object(metadata, factory_defaults)
    if factory_obj:
        metadata['factory'] = factory_obj
        # Always keep backward-compat flat fields in sync
        if factory_obj.get('factory_name'):
            metadata['factory_name'] = factory_obj['factory_name']
        if factory_obj.get('factory_group_name'):
            metadata['factory_group_name'] = factory_obj['factory_group_name']
        logger.info(f"   🏭 Factory object assembled: {list(factory_obj.keys())}")

    # ── Auto-classify material_category + zone_intent if missing / not in vocab ──
    raw_cat = metadata.get("material_category")
    # Flatten dict-valued critical field if AI stored it nested
    if isinstance(raw_cat, dict):
        raw_cat = raw_cat.get("value")
        metadata["material_category"] = raw_cat

    raw_intent = metadata.get("zone_intent")
    if isinstance(raw_intent, dict):
        raw_intent = raw_intent.get("value")
        metadata["zone_intent"] = raw_intent

    needs_category = not raw_cat or raw_cat not in MATERIAL_CATEGORY_VOCAB
    needs_intent = not raw_intent or raw_intent not in ZONE_INTENT_VOCAB
    if needs_category or needs_intent:
        try:
            classified = await _classify_product(
                name=product.name,
                description=metadata.get("description", "") or ai_metadata.get("description", ""),
                existing_category=raw_cat or "",
            )
            if needs_category and classified.get("material_category"):
                metadata["material_category"] = classified["material_category"]
                logger.info(f"   🏷️ Auto-classified material_category: {classified['material_category']}")
            if needs_intent and classified.get("zone_intent"):
                metadata["zone_intent"] = classified["zone_intent"]
                logger.info(f"   🏷️ Auto-classified zone_intent: {classified['zone_intent']}")
        except Exception as cls_err:
            logger.warning(f"   ⚠️ Auto-classification skipped: {cls_err}")

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

    # Set default unit from category if not already present
    if not metadata.get('unit'):
        metadata['unit'] = _resolve_default_unit(metadata.get('material_category'))

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

        # ── Find the most complete nested factory object across all products ──
        best_factory: Dict[str, Any] = {}
        best_factory_score = 0
        _completeness_fields = ['factory_name', 'city', 'country', 'address',
                                 'phone', 'email', 'website', 'country_of_origin', 'employee_count']
        for product in products:
            meta = product.get('metadata', {}) or {}
            fobj = meta.get('factory') or {}
            if not isinstance(fobj, dict):
                continue
            score = sum(1 for f in _completeness_fields if fobj.get(f) and not _is_empty_value(fobj[f]))
            if score > best_factory_score:
                best_factory_score = score
                best_factory = fobj

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

            # Propagate nested factory object if product is missing/incomplete
            factory_updated = False
            if best_factory:
                existing_fobj = metadata.get('factory') or {}
                existing_score = sum(
                    1 for f in _completeness_fields
                    if existing_fobj.get(f) and not _is_empty_value(existing_fobj[f])
                ) if isinstance(existing_fobj, dict) else 0
                if existing_score < best_factory_score:
                    # Merge: existing values win, best_factory fills gaps
                    merged_factory = {**best_factory, **{
                        k: v for k, v in existing_fobj.items()
                        if v and not _is_empty_value(v)
                    }}
                    updates_needed['factory'] = merged_factory
                    # Keep flat backward-compat fields in sync
                    if merged_factory.get('factory_name'):
                        updates_needed['factory_name'] = merged_factory['factory_name']
                    if merged_factory.get('factory_group_name'):
                        updates_needed['factory_group_name'] = merged_factory['factory_group_name']
                    factory_updated = True

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
