"""
Stage 4: Product Creation

This module handles product creation in the database for the product-centric pipeline.
Includes metadata consolidation from AI text extraction, visual analysis, and factory defaults.
"""

import logging
import os
import httpx
from typing import Dict, Any, List, Optional

from app.services.metadata.metadata_normalizer import normalize_factory_keys

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

    # ✨ NEW (2026-04): Fetch the canonical spec field taxonomy ONCE per
    # product creation. Used for both icon metadata rollup AND the embedding
    # text builder so they agree on what counts as a "spec field".
    known_spec_fields = await _fetch_known_spec_fields(supabase, logger)

    # ✨ NEW (2026-04): Roll up per-image icon_metadata into flat top-level
    # spec keys (e.g. metadata['slip_resistance'] = 'R10'). The icon
    # extraction pipeline writes per-image audit data to
    # document_images.metadata['icon_metadata']; here we promote the highest
    # -confidence value for each field onto the product itself, normalized
    # against material_metadata_fields.
    icon_rollup = await _merge_icon_metadata_into_product(
        document_id=document_id,
        image_indices=product.image_indices,
        known_spec_fields=known_spec_fields,
        supabase=supabase,
        logger=logger,
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

    # ✨ NEW (2026-04): Merge icon rollup into top-level metadata.
    # Icon-extracted spec values are typically the most authoritative source
    # for technical specs (R-rating, PEI, fire rating, frost resistance, etc)
    # because they come from canonical icons in the catalog. We give them
    # priority over AI text extraction guesses by writing them last.
    if icon_rollup:
        for spec_field, spec_value in icon_rollup.items():
            metadata[spec_field] = spec_value
        logger.info(
            f"   🔖 Merged {len(icon_rollup)} icon-extracted spec fields onto product"
        )

    # Clean up "not found" values
    for key in ['factory_name', 'factory_group_name', 'material_category']:
        if is_not_found(metadata.get(key)):
            metadata[key] = None

    # ── Normalize factory aliases to canonical keys ───────────────────────
    # Folds metadata.manufacturer / brand / supplier / factory_group → factory_name / factory_group_name
    # and removes the alias keys, so the rest of the platform only sees the canonical schema.
    normalize_factory_keys(metadata)

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

        # Generate text_embedding_1024 for product-level vector search.
        # Builds a rich text from name + description + key metadata fields,
        # then embeds with Voyage AI (1024D) — same model as document_chunks.
        try:
            embedding_text_parts = [product.name or '']
            if description:
                embedding_text_parts.append(description)
            # Include searchable metadata in the embedding text.
            # Use canonical factory_name / factory_group_name only — manufacturer
            # alias is normalized away upstream by normalize_factory_keys().
            for key in ('factory_name', 'factory_group_name', 'designer', 'material_category', 'zone_intent'):
                val = metadata.get(key)
                if val and isinstance(val, str) and val.lower() not in ('not specified', 'not found', 'unknown', 'n/a'):
                    embedding_text_parts.append(val.replace('_', ' '))
            colors = metadata.get('available_colors')
            if isinstance(colors, list):
                embedding_text_parts.extend(colors)

            # ✨ NEW (2026-04): Walk every known spec field from material_metadata_fields
            # and append non-null values to the embedding text. This means a search
            # like "porcelain tile R10 PEI IV frost resistant" matches a product whose
            # spec icons have been extracted, even though those values were never
            # written manually — they were rolled up from icon_metadata into top-level
            # keys by `_merge_icon_metadata_into_product` above.
            #
            # This loop runs over the canonical taxonomy (single source of truth),
            # not a hardcoded list, so adding a new spec to material_metadata_fields
            # automatically flows it into Voyage embeddings without code changes.
            _embedded_spec_fields_so_far = {
                'factory_name', 'factory_group_name', 'designer',
                'material_category', 'zone_intent',
            }
            for spec_field in known_spec_fields:
                if spec_field in _embedded_spec_fields_so_far:
                    continue  # already appended above
                val = metadata.get(spec_field)
                if val is None or val == '' or val == []:
                    continue
                # Render the value to text — handle scalars, lists, and bools.
                if isinstance(val, bool):
                    if val:
                        embedding_text_parts.append(spec_field.replace('_', ' '))
                elif isinstance(val, (str, int, float)):
                    text_val = str(val).strip()
                    if text_val and text_val.lower() not in ('not specified', 'not found', 'unknown', 'n/a'):
                        embedding_text_parts.append(f"{spec_field.replace('_', ' ')}: {text_val}")
                elif isinstance(val, list) and val:
                    items = [str(v).strip() for v in val if v not in (None, '', [])]
                    if items:
                        embedding_text_parts.append(
                            f"{spec_field.replace('_', ' ')}: {', '.join(items)}"
                        )

            embedding_text = ' | '.join(embedding_text_parts)

            from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
            embeddings_svc = RealEmbeddingsService()
            emb_result = await embeddings_svc.generate_text_embedding(embedding_text)
            text_emb = emb_result.get('embedding') if emb_result.get('success') else None

            if text_emb:
                # Store as vector string for pgvector
                embedding_str = '[' + ','.join(str(x) for x in text_emb) + ']'
                supabase.client.table('products').update(
                    {'text_embedding_1024': embedding_str}
                ).eq('id', product_id).execute()
                logger.info(f"   🧠 Generated text_embedding_1024 for {product.name}")
            else:
                logger.warning(f"   ⚠️ Embedding generation returned no text_1024 for {product.name}")

        except Exception as emb_err:
            logger.warning(f"   ⚠️ Product embedding generation failed (non-blocking): {emb_err}")

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


# ────────────────────────────────────────────────────────────────────────── #
# Icon metadata: rollup from per-image icon_metadata into flat product keys  #
# ────────────────────────────────────────────────────────────────────────── #

# Field-name normalization map. The Icon-Based Metadata Extraction prompt
# returns some field names that don't quite match `material_metadata_fields`
# — typically singular vs plural mismatches. Anything not in this map is
# used as-is.
ICON_FIELD_NAME_NORMALIZATION = {
    'certification': 'certifications',  # prompt returns singular, schema is plural
}


def _normalize_icon_field_name(field_name: str) -> str:
    """Normalize an icon prompt's field_name to match material_metadata_fields."""
    if not field_name:
        return field_name
    return ICON_FIELD_NAME_NORMALIZATION.get(field_name.strip(), field_name.strip())


async def _fetch_known_spec_fields(
    supabase: Any,
    logger: logging.Logger,
) -> List[str]:
    """
    Fetch the canonical spec field name list from `material_metadata_fields`.

    This is the source of truth for which top-level keys on `products.metadata`
    are valid spec fields. Used by:
      1. `_merge_icon_metadata_into_product` — to drop icon entries whose
         field_name doesn't match a known spec (defensive against the prompt
         inventing new field names).
      2. The product embedding text builder — to walk every spec field that
         is populated on a product and append it to the Voyage embedding text,
         so a search like "porcelain tile R10 frost resistant" matches.
    """
    try:
        result = supabase.client.table('material_metadata_fields') \
            .select('field_name') \
            .execute()
        if not result.data:
            logger.warning("   ⚠️ material_metadata_fields is empty — spec rollup will skip all icons")
            return []
        # Deduplicate (the table has a few rows with the same field_name across
        # different applies_to_categories — we only care about the unique set).
        return sorted({row['field_name'] for row in result.data if row.get('field_name')})
    except Exception as e:
        logger.warning(f"   ⚠️ Failed to fetch material_metadata_fields: {e}")
        return []


async def _merge_icon_metadata_into_product(
    document_id: str,
    image_indices: Optional[List[int]],
    known_spec_fields: List[str],
    supabase: Any,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Walk all `document_images` for a product, collect every icon_metadata
    entry, normalize field names against `material_metadata_fields`, and
    return a flat dict of `{field_name: value}` to merge into the product's
    top-level metadata.

    Conflict resolution: when two images contribute the same field, the
    higher-confidence value wins. Anything whose normalized field_name is
    not in `known_spec_fields` is logged as a warning and dropped — we don't
    want the icon prompt inventing new fields.

    Returns:
        Flat dict of {spec_field_name: value} ready to merge into
        `product.metadata`. Empty dict if there are no icons.
    """
    if not known_spec_fields:
        return {}

    try:
        # Fetch all images for this document — we'll filter to ones with
        # icon_metadata in their metadata blob. We don't filter by image
        # index because Stage 3 may have routed icons to image indices the
        # product object doesn't know about (icons are detected post-classification).
        images_response = supabase.client.table('document_images') \
            .select('id, page_number, metadata') \
            .eq('document_id', document_id) \
            .execute()
        if not images_response.data:
            return {}

        known_spec_set = set(known_spec_fields)
        # field_name → (value, confidence) — highest-confidence wins
        best_per_field: Dict[str, tuple] = {}
        unknown_fields: Dict[str, int] = {}
        total_items_seen = 0

        for image in images_response.data:
            image_meta = image.get('metadata') or {}
            if not isinstance(image_meta, dict):
                continue
            icon_items = image_meta.get('icon_metadata') or []
            if not isinstance(icon_items, list) or not icon_items:
                continue

            for item in icon_items:
                if not isinstance(item, dict):
                    continue
                total_items_seen += 1
                raw_field = item.get('field_name')
                if not raw_field:
                    continue
                field = _normalize_icon_field_name(raw_field)
                if field not in known_spec_set:
                    unknown_fields[field] = unknown_fields.get(field, 0) + 1
                    continue
                value = item.get('value')
                if value is None or value == '' or value == []:
                    continue
                confidence = float(item.get('confidence') or 0.0)

                existing = best_per_field.get(field)
                if existing is None or confidence > existing[1]:
                    best_per_field[field] = (value, confidence)

        rollup = {field: value for field, (value, _conf) in best_per_field.items()}

        if rollup:
            logger.info(
                f"   🔖 Icon metadata rollup: {len(rollup)} flat spec fields from "
                f"{total_items_seen} icon items "
                f"({', '.join(sorted(rollup.keys())[:8])}{'...' if len(rollup) > 8 else ''})"
            )
        elif total_items_seen > 0:
            logger.info(
                f"   🔖 Icon metadata rollup: {total_items_seen} items seen but none "
                f"matched known spec fields"
            )

        if unknown_fields:
            logger.warning(
                f"   ⚠️ Icon metadata: dropped {sum(unknown_fields.values())} items with unknown field names: "
                f"{dict(sorted(unknown_fields.items(), key=lambda x: -x[1])[:5])}"
            )

        return rollup

    except Exception as e:
        logger.warning(f"   ⚠️ Failed to merge icon metadata into product: {e}")
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


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4.7 — Deterministic product enrichment from chunks + vision_analysis
# ═══════════════════════════════════════════════════════════════════════════
#
# The AI extractor (Stage 0) is probabilistic. When it returns empty sections
# (which happens regularly on bilingual / narrative-heavy catalogs), this stage
# fills the gaps deterministically:
#
#   - Regex patterns over the product's document_chunks catch factory name
#     from narrative ("from Harmony", "collaboration with X"), designer full
#     name ("Stacy Garcia, a New York-based designer"), SKU codes, grout
#     suppliers with dose, pieces_per_box, patterns_count, body_type.
#
#   - Majority-vote over document_images.vision_analysis rolls up per-image
#     material_type and finish into product-level material_category and finish,
#     and unions all observed color palettes.
#
# Only null/empty fields are filled. Confident AI values are never overwritten.
# Each written field is tagged with _source so we can trace provenance later.

import re as _re

# Controlled vocabulary map for vision_analysis material_type → products.material_category
_MATERIAL_TYPE_TO_CATEGORY = {
    "ceramic tile": "ceramic_tile",
    "porcelain tile": "porcelain_tile",
    "stoneware": "stoneware_tile",
    "stoneware tile": "stoneware_tile",
    "mosaic": "mosaic_tile",
    "mosaic tile": "mosaic_tile",
    "natural stone": "natural_stone",
    "marble": "marble",
    "granite": "granite",
    "slate": "slate",
    "limestone": "limestone",
    "wood": "wood_flooring",
    "wood flooring": "wood_flooring",
    "laminate": "laminate",
    "vinyl": "vinyl_flooring",
    "wall tile": "wall_tile",
    "floor tile": "floor_tile",
    "bathroom tile": "bathroom_tile",
    "outdoor tile": "outdoor_tile",
}


def _normalize_material_category(raw: str) -> Optional[str]:
    """Map an AI-freeform material_type to our controlled vocab."""
    if not raw or not isinstance(raw, str):
        return None
    key = raw.strip().lower()
    if key in _MATERIAL_TYPE_TO_CATEGORY:
        return _MATERIAL_TYPE_TO_CATEGORY[key]
    # Partial matches
    for phrase, vocab in _MATERIAL_TYPE_TO_CATEGORY.items():
        if phrase in key:
            return vocab
    return None


def _extract_fields_from_chunk_text(text: str) -> Dict[str, Any]:
    """Pure function: run regex extractors over combined chunk text.

    Returns a dict of candidate field → value pairs. Caller decides whether
    to apply them (only if the existing value is empty).
    """
    if not text:
        return {}

    candidates: Dict[str, Any] = {}

    # ── Factory name from narrative text ──────────────────────────────────
    # "collaboration from X", "collaboration with X", "from X.", "by X."
    # X must be Capitalized (possibly multi-word, but we stop at common words)
    factory_patterns = [
        _re.compile(r"collaboration\s+(?:from|with|by)\s+([A-Z][A-Za-z][A-Za-z0-9&'\-]*(?:\s+[A-Z][A-Za-z0-9&'\-]+)?)", _re.IGNORECASE),
        _re.compile(r"\bproduced\s+by\s+([A-Z][A-Za-z][A-Za-z0-9&'\-]*(?:\s+[A-Z][A-Za-z0-9&'\-]+)?)"),
        _re.compile(r"\bmade\s+by\s+([A-Z][A-Za-z][A-Za-z0-9&'\-]*(?:\s+[A-Z][A-Za-z0-9&'\-]+)?)"),
        # "the new Signature collaboration from Harmony" — most common pattern
        _re.compile(r"Signature\s+collaboration\s+from\s+([A-Z][A-Za-z0-9&'\-]+)"),
    ]
    factory_candidates = []
    for pat in factory_patterns:
        for m in pat.finditer(text):
            name = m.group(1).strip()
            # Reject common false positives (stop words, generic terms)
            if name.lower() in {"the", "a", "an", "this", "that", "our", "new", "stacy",
                                 "york", "barcelona", "valencia", "milan", "paris"}:
                continue
            if 2 <= len(name) <= 30:
                factory_candidates.append(name)
    if factory_candidates:
        # Pick the most frequent
        from collections import Counter
        most_common = Counter(factory_candidates).most_common(1)[0][0]
        candidates["factory_name"] = most_common

    # ── Designer full name from narrative ─────────────────────────────────
    # "Stacy Garcia, a New York-based designer"
    # "designed by Stacy Garcia"
    # "by Stacy Garcia, a ... designer"
    designer_patterns = [
        _re.compile(r"([A-Z][a-z]+\s+[A-Z][a-z]+),?\s+(?:a|an)\s+[^,.]*?(?:designer|architect|creative)"),
        _re.compile(r"designed\s+by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)"),
        _re.compile(r"by\s+([A-Z][a-z]+\s+[A-Z][a-z]+),?\s+a\s+[^,.]*?(?:designer|architect)"),
    ]
    designer_candidates = []
    for pat in designer_patterns:
        for m in pat.finditer(text):
            name = m.group(1).strip()
            if 5 <= len(name) <= 40:
                designer_candidates.append(name)
    if designer_candidates:
        from collections import Counter
        most_common = Counter(designer_candidates).most_common(1)[0][0]
        candidates["designers"] = [most_common]

    # ── SKU codes: "39656   VALENOVA WHITE LT/11,8X11,8" ─────────────────
    # Pattern: 5-6 digit code, then a product name in caps (multi-word), then
    # optional variant indicator
    sku_pattern = _re.compile(
        r"\b(\d{5,6})\s+([A-Z][A-Z0-9]+(?:\s+[A-Z0-9]+){0,4})\s+(?:LT|[A-Z]{2,3})\s*/",
    )
    sku_codes: Dict[str, str] = {}
    for m in sku_pattern.finditer(text):
        code = m.group(1)
        name = m.group(2).strip()
        if name not in sku_codes.values():  # avoid duplicate names with different codes
            sku_codes[name] = code
    if sku_codes:
        candidates["sku_codes"] = sku_codes

    # ── Grout suppliers + dose: "100 Mapei", "43 Kerakoll" ──────────────
    grout_pattern = _re.compile(
        r"(\d{1,4})\s+(Mapei|Kerakoll|Isomat|Technica|Litokol)\b",
        _re.IGNORECASE,
    )
    grout_suppliers = set()
    grout_entries = []
    for m in grout_pattern.finditer(text):
        dose = int(m.group(1))
        supplier = m.group(2).capitalize()
        grout_suppliers.add(supplier.upper())
        grout_entries.append({"supplier": supplier, "dose": dose})
    if grout_suppliers:
        candidates["grout_suppliers"] = sorted(grout_suppliers)
    if grout_entries and sku_codes:
        # Try to correlate grout entries with SKU codes by order of appearance.
        # This is best-effort — the SKU and grout dose often appear on the same
        # line in catalog tables.
        codes_in_order = list(sku_codes.values())
        if len(codes_in_order) == len(grout_entries):
            candidates["grout_color_codes"] = {
                codes_in_order[i]: grout_entries[i] for i in range(len(codes_in_order))
            }

    # ── Pieces per box: "12 pieces" ───────────────────────────────────────
    pieces_match = _re.search(r"\b(\d{1,3})\s+pieces?\b", text, _re.IGNORECASE)
    if pieces_match:
        n = int(pieces_match.group(1))
        if 1 <= n <= 500:
            candidates["pieces_per_box"] = n

    # ── Patterns count: "12 patterns" ─────────────────────────────────────
    patterns_match = _re.search(r"\b(\d{1,3})\s+patterns?\b", text, _re.IGNORECASE)
    if patterns_match:
        n = int(patterns_match.group(1))
        if 1 <= n <= 100:
            candidates["patterns_count"] = n

    # ── Body type: "white body tile", "full body ceramics", "porcelain stoneware" ─
    body_type_pattern = _re.compile(
        r"\b(white body tile|full body(?:\s+ceramics?)?|porcelain stoneware|"
        r"red body|color(?:ed)?\s+body)\b",
        _re.IGNORECASE,
    )
    bt = body_type_pattern.search(text)
    if bt:
        candidates["body_type"] = bt.group(1).lower()

    # ── Dimensions: "11,8x11,8 cm  4.65x4.65"" ────────────────────────────
    # Look for tight pairs like "11,8x11,8" followed by cm or near imperial equivalent
    dim_pattern = _re.compile(
        r"(\d{1,3}(?:[,.]\d{1,2})?)\s*[xX×]\s*(\d{1,3}(?:[,.]\d{1,2})?)\s*cm",
    )
    dims = []
    seen_dims = set()
    for m in dim_pattern.finditer(text):
        w = m.group(1).replace(",", ".")
        h = m.group(2).replace(",", ".")
        try:
            wf, hf = float(w), float(h)
            if 0.5 <= wf <= 300 and 0.5 <= hf <= 300:
                key = f"{wf}x{hf}"
                if key not in seen_dims:
                    seen_dims.add(key)
                    dims.append({"metric_cm": f"{w}x{h}", "format_label": None})
        except ValueError:
            continue

    # ── Imperial dimensions: '4.65x4.65"' — attach to existing metric dims ─
    imperial_pattern = _re.compile(r'(\d{1,3}(?:[.,]\d{1,3})?)\s*[xX×]\s*(\d{1,3}(?:[.,]\d{1,3})?)\s*["”]')
    imperials = []
    for m in imperial_pattern.finditer(text):
        w = m.group(1).replace(",", ".")
        h = m.group(2).replace(",", ".")
        imperials.append(f"{w}x{h}")
    # Zip imperials onto dims by order (best-effort pairing)
    for i, d in enumerate(dims):
        if i < len(imperials):
            d["imperial_in"] = imperials[i]

    # ── Format label: "Q59 (11,8x11,8 cm − 45/8x45/8")" captures the Q59 tag ─
    q_label_pattern = _re.compile(r"\b(Q\d{1,3})\s*\(", _re.IGNORECASE)
    q_matches = q_label_pattern.findall(text)
    if q_matches and dims:
        dims[0]["format_label"] = q_matches[0].upper()

    if dims:
        candidates["dimensions"] = dims

    # ── Grout product names: "MAPEI | ULTRACOLOR PLUS" and "KERAKOLL | FUGABELLA" ─
    grout_product_pattern = _re.compile(
        r"(MAPEI|KERAKOLL|ISOMAT|TECHNICA|LITOKOL)\s*\|\s*([A-Z][A-Z0-9\s]{2,30})",
        _re.IGNORECASE,
    )
    grout_products: Dict[str, str] = {}
    for m in grout_product_pattern.finditer(text):
        supplier = m.group(1).upper()
        product = m.group(2).strip()
        # Clean trailing whitespace/newlines from the captured product name
        product = _re.sub(r"\s+", " ", product).strip()
        # Stop at common end-of-line markers
        product = _re.split(r"\*+|\|", product, maxsplit=1)[0].strip()
        if 3 <= len(product) <= 40:
            grout_products[f"grout_{supplier.lower()}_product"] = product
    if grout_products:
        candidates.update(grout_products)

    # ── MATT/GLOSS finish indicators appearing in packing tables ─────────
    # "MATT" and "GLOSS" on the spec page are options, not confirmed values for
    # this product unless they appear with a "✓" or bold marker — but their
    # presence on the page at least hints at what finishes exist.
    finish_patterns = {
        "matt":      _re.compile(r"\bMATT\b"),
        "gloss":     _re.compile(r"\bGLOSS\b"),
        "polished":  _re.compile(r"\bPOLISHED\b", _re.IGNORECASE),
        "satin":     _re.compile(r"\bSATIN\b", _re.IGNORECASE),
        "natural":   _re.compile(r"\bNATURAL\b", _re.IGNORECASE),
    }
    # Only set finish if the text makes a definitive claim; multiple options =
    # skip (don't overwrite vision_analysis which has the actual answer).
    # We intentionally do NOT write candidates["finish"] here — vision rollup
    # handles finish more reliably.

    # ── Collection name: "{PRODUCT} by {DESIGNER}" + "the new X collection" ─
    # Also handle: "VALENOVA by SG NY is the new Signature collaboration"
    collection_patterns = [
        _re.compile(r"\b([A-Z][A-Z0-9]+)\s+by\s+[A-Z]", _re.IGNORECASE),
        _re.compile(r"(?:the\s+new\s+|the\s+)?([A-Z][A-Z0-9]+)\s+collection\b", _re.IGNORECASE),
    ]
    collection_candidates: List[str] = []
    for pat in collection_patterns:
        for m in pat.finditer(text):
            name = m.group(1).strip()
            if 3 <= len(name) <= 20 and name.isupper():
                collection_candidates.append(name)
    if collection_candidates:
        from collections import Counter
        most_common = Counter(collection_candidates).most_common(1)[0][0]
        # Title case it — "VALENOVA" → "Valenova"
        candidates["collection"] = most_common.title()

    # ── Inspiration: "draws inspiration from X", "inspired by X" ─────────
    inspiration_patterns = [
        _re.compile(r"draws?\s+inspiration\s+from\s+(?:the\s+)?([a-zA-Z][\w\s\-]{3,60})", _re.IGNORECASE),
        _re.compile(r"inspired\s+by\s+(?:the\s+)?([a-zA-Z][\w\s\-]{3,60})", _re.IGNORECASE),
    ]
    for pat in inspiration_patterns:
        m = pat.search(text)
        if m:
            inspiration = m.group(1).strip()
            # Stop at sentence boundary
            inspiration = _re.split(r"[,.;]", inspiration, maxsplit=1)[0].strip()
            if 3 <= len(inspiration) <= 80:
                candidates["inspiration"] = inspiration
                break

    return candidates


def _rollup_vision_analysis(vision_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure function: majority-vote rollup of per-image vision_analysis into
    product-level fields.

    Args:
        vision_rows: List of dicts, each containing a vision_analysis JSON blob.

    Returns:
        Candidate fields: material_category, finish, appearance_colors.
    """
    if not vision_rows:
        return {}

    from collections import Counter

    material_types: List[str] = []
    material_subtypes: List[str] = []
    finishes: List[str] = []
    patterns: List[str] = []
    textures: List[str] = []
    design_styles: List[str] = []
    all_colors: List[str] = []
    all_applications: List[str] = []
    primary_hex_codes: List[str] = []

    for row in vision_rows:
        va = row.get("vision_analysis") or {}
        if not isinstance(va, dict):
            continue

        mt = va.get("material_type")
        if isinstance(mt, str) and mt.strip():
            material_types.append(mt.strip().lower())

        mst = va.get("material_subtype")
        if isinstance(mst, str) and mst.strip():
            material_subtypes.append(mst.strip().lower())

        finish = va.get("finish")
        if isinstance(finish, str) and finish.strip():
            finishes.append(finish.strip().lower())

        pattern = va.get("pattern")
        if isinstance(pattern, str) and pattern.strip():
            patterns.append(pattern.strip())

        texture = va.get("texture")
        if isinstance(texture, str) and texture.strip():
            textures.append(texture.strip())

        style = va.get("design_style")
        if isinstance(style, str) and style.strip():
            design_styles.append(style.strip().lower())

        palette = va.get("color_palette") or va.get("colors") or []
        if isinstance(palette, list):
            for c in palette:
                if isinstance(c, str) and c.strip():
                    all_colors.append(c.strip().lower())

        apps = va.get("applications") or []
        if isinstance(apps, list):
            for a in apps:
                if isinstance(a, str) and a.strip():
                    all_applications.append(a.strip().lower())

        hx = va.get("primary_color_hex")
        if isinstance(hx, str) and hx.strip().startswith("#"):
            primary_hex_codes.append(hx.strip().upper())

    candidates: Dict[str, Any] = {}

    if material_types:
        most_common_mt, _count = Counter(material_types).most_common(1)[0]
        normalized = _normalize_material_category(most_common_mt)
        if normalized:
            candidates["material_category"] = normalized

    if material_subtypes:
        # Pick the most specific one (longest description)
        unique_subtypes = list({s: True for s in material_subtypes}.keys())
        candidates["material_subtype"] = max(unique_subtypes, key=len)

    if finishes:
        most_common_finish, _count = Counter(finishes).most_common(1)[0]
        candidates["finish"] = most_common_finish

    if patterns:
        # Primary (singular) pattern — the most descriptive entry, kept for
        # backwards compatibility with older UI consumers.
        candidates["pattern"] = max(set(patterns), key=len)
        # Aggregated list of all unique patterns seen across variants —
        # rendered as a chip list in the product detail modal's Appearance
        # section. Dedupe case-insensitively, preserve order by frequency.
        pattern_counts = Counter(patterns)
        unique_patterns: List[str] = []
        seen_pattern_norms: set = set()
        for p, _ in pattern_counts.most_common():
            norm = p.strip().lower()
            if norm and norm not in seen_pattern_norms:
                seen_pattern_norms.add(norm)
                unique_patterns.append(p.strip())
        if unique_patterns:
            candidates["patterns"] = unique_patterns[:20]  # cap

    if textures:
        candidates["texture"] = max(set(textures), key=len)

    if design_styles:
        most_common_style, _count = Counter(design_styles).most_common(1)[0]
        candidates["design_style"] = most_common_style

    if all_colors:
        # Union, deduped, preserving most common order
        color_counts = Counter(all_colors)
        unique_colors = [c for c, _ in color_counts.most_common()]
        candidates["appearance_colors"] = unique_colors[:20]  # cap at 20

    if all_applications:
        app_counts = Counter(all_applications)
        candidates["applications"] = [a for a, _ in app_counts.most_common()][:10]

    if primary_hex_codes:
        # Pick the most frequent hex (product-level primary color)
        most_common_hex, _count = Counter(primary_hex_codes).most_common(1)[0]
        candidates["primary_color_hex"] = most_common_hex

    return candidates


def _merge_enriched_fields_into_metadata(
    existing_metadata: Dict[str, Any],
    chunk_candidates: Dict[str, Any],
    vision_candidates: Dict[str, Any],
) -> (Dict[str, Any], List[str]):
    """Merge chunk + vision candidates into a fresh metadata dict.

    Rules:
        - Only fill fields where existing value is empty (_is_empty_value).
        - Never overwrite confident AI values.
        - Record provenance under metadata['_extraction_metadata'][field]['source'].

    Returns:
        (new_metadata, list_of_filled_field_names)
    """
    new_metadata = dict(existing_metadata or {})
    filled: List[str] = []

    # Ensure _extraction_metadata exists
    extraction_meta = dict(new_metadata.get("_extraction_metadata") or {})

    def fill_if_empty(key: str, value: Any, source: str, container: Optional[str] = None):
        if _is_empty_value(value):
            return
        if container:
            parent = dict(new_metadata.get(container) or {})
            if _is_empty_value(parent.get(key)):
                parent[key] = value
                new_metadata[container] = parent
                extraction_meta[f"{container}.{key}"] = {"source": source, "confidence": 0.90}
                filled.append(f"{container}.{key}")
        else:
            if _is_empty_value(new_metadata.get(key)):
                new_metadata[key] = value
                extraction_meta[key] = {"source": source, "confidence": 0.90}
                filled.append(key)

    # ── Apply chunk candidates ────────────────────────────────────────────
    fill_if_empty("factory_name", chunk_candidates.get("factory_name"), "chunk_regex")
    fill_if_empty("designers", chunk_candidates.get("designers"), "chunk_regex")
    fill_if_empty("collection", chunk_candidates.get("collection"), "chunk_regex")
    fill_if_empty("inspiration", chunk_candidates.get("inspiration"), "chunk_regex", container="design")
    fill_if_empty("pieces_per_box", chunk_candidates.get("pieces_per_box"), "chunk_regex", container="packaging")
    fill_if_empty("patterns_count", chunk_candidates.get("patterns_count"), "chunk_regex", container="packaging")
    fill_if_empty("body_type", chunk_candidates.get("body_type"), "chunk_regex", container="material_properties")
    fill_if_empty("sku_codes", chunk_candidates.get("sku_codes"), "chunk_regex", container="commercial")
    fill_if_empty("grout_suppliers", chunk_candidates.get("grout_suppliers"), "chunk_regex", container="commercial")
    fill_if_empty("grout_color_codes", chunk_candidates.get("grout_color_codes"), "chunk_regex", container="commercial")
    # Grout product names (chunk 17 style): MAPEI | ULTRACOLOR PLUS, etc.
    fill_if_empty("grout_mapei", chunk_candidates.get("grout_mapei_product"), "chunk_regex", container="commercial")
    fill_if_empty("grout_kerakoll", chunk_candidates.get("grout_kerakoll_product"), "chunk_regex", container="commercial")
    fill_if_empty("grout_isomat", chunk_candidates.get("grout_isomat_product"), "chunk_regex", container="commercial")
    fill_if_empty("grout_technica", chunk_candidates.get("grout_technica_product"), "chunk_regex", container="commercial")

    # Dimensions special-case: only fill if currently empty/missing/hallucinated
    chunk_dims = chunk_candidates.get("dimensions")
    if chunk_dims and _is_empty_value(new_metadata.get("dimensions")):
        new_metadata["dimensions"] = chunk_dims
        extraction_meta["dimensions"] = {"source": "chunk_regex", "confidence": 0.95}
        filled.append("dimensions")
        # When we fill chunk dimensions, also drop the stale AI-hallucinated
        # `available_sizes` so the UI sidebar (which prefers available_sizes
        # over dimensions) shows the real value. Only drop it if we just
        # filled dimensions — don't touch it on unrelated enrichment runs.
        if "available_sizes" in new_metadata:
            new_metadata.pop("available_sizes", None)
            filled.append("(dropped stale available_sizes)")

    # If we filled `designers` (plural array, correct full name), drop any
    # stale scalar `designer` leftover from the old AI extractor. Both existing
    # at once confuses the UI, and the plural is the canonical form.
    if (
        chunk_candidates.get("designers")
        and isinstance(new_metadata.get("designers"), list)
        and new_metadata.get("designers")
        and "designer" in new_metadata
    ):
        new_metadata.pop("designer", None)
        filled.append("(dropped stale designer scalar)")

    # ── Apply vision candidates ───────────────────────────────────────────
    fill_if_empty("material_category", vision_candidates.get("material_category"), "vision_rollup")
    fill_if_empty("finish", vision_candidates.get("finish"), "vision_rollup", container="material_properties")
    fill_if_empty("material_subtype", vision_candidates.get("material_subtype"), "vision_rollup", container="material_properties")
    fill_if_empty("pattern", vision_candidates.get("pattern"), "vision_rollup", container="appearance")
    fill_if_empty("patterns", vision_candidates.get("patterns"), "vision_rollup", container="appearance")
    fill_if_empty("texture", vision_candidates.get("texture"), "vision_rollup", container="appearance")
    fill_if_empty("design_style", vision_candidates.get("design_style"), "vision_rollup", container="design")
    fill_if_empty("applications", vision_candidates.get("applications"), "vision_rollup")
    fill_if_empty("primary_color_hex", vision_candidates.get("primary_color_hex"), "vision_rollup", container="appearance")

    vision_colors = vision_candidates.get("appearance_colors")
    if vision_colors:
        appearance = dict(new_metadata.get("appearance") or {})
        # Store under appearance.colors_from_vision so it doesn't clobber the
        # text-extracted colors_from_chunks if both exist.
        if _is_empty_value(appearance.get("colors_from_vision")):
            appearance["colors_from_vision"] = vision_colors
            new_metadata["appearance"] = appearance
            extraction_meta["appearance.colors_from_vision"] = {"source": "vision_rollup", "confidence": 0.85}
            filled.append("appearance.colors_from_vision")

    if filled:
        new_metadata["_extraction_metadata"] = extraction_meta

    return new_metadata, filled


async def enrich_products_from_chunks_and_vision(
    document_id: str,
    supabase: Any,
    logger: logging.Logger,
    target_product_id: Optional[str] = None,
    enable_spec_vision: bool = True,
    enable_description_writer: bool = True,
) -> Dict[str, Any]:
    """Stage 4.7: fill null product.metadata fields from chunks and vision_analysis.

    Runs after Stage 0 AI extraction + Stage 4.5 propagation + Stage 4.6 dimension
    extraction. Only fills empty values. Never overwrites AI-extracted data.

    Now also runs (per product):
      1. product_spec_vision_extractor — Claude Vision on the product's PDF spec
         pages for packing data, slip/PEI/fire/shade icons, certifications, etc.
      2. product_description_writer — Claude Haiku on the product's chunks to
         generate a clean English description (written to products.description).

    Args:
        document_id: Document to enrich.
        supabase: Supabase client.
        logger: Logger.
        target_product_id: If provided, enrich only this product (used by the
                           backfill endpoint). Otherwise, enrich all products
                           in the document.
        enable_spec_vision: Toggle for the Claude Vision spec page pass.
        enable_description_writer: Toggle for the Claude Haiku description generator.

    Returns:
        Stats: products_checked, products_updated, fields_filled (set),
               chunk_candidates_found, vision_candidates_found,
               spec_vision_calls, description_writes
    """
    stats: Dict[str, Any] = {
        "products_checked": 0,
        "products_updated": 0,
        "fields_filled": set(),
        "spec_vision_calls": 0,
        "description_writes": 0,
        "kb_docs_created": 0,
        "kb_attachments_created": 0,
        "catalog_kb_docs_created": 0,
    }

    try:
        # ── Fetch products (include description column for writer check) ───
        q = supabase.client.table("products").select("id, name, description, metadata")
        if target_product_id:
            q = q.eq("id", target_product_id)
        else:
            q = q.eq("source_document_id", document_id)
        products_resp = q.execute()
        products = products_resp.data or []
        stats["products_checked"] = len(products)

        if not products:
            logger.info("   ℹ️ Enrichment: no products to enrich")
            return stats

        # ── Fetch document chunks for the whole document ──────────────────
        # document_chunks has a direct `product_id` column AND stores
        # `metadata.product_pages` (array of absolute page numbers). We scope
        # chunks by product_id (exact) and fall back to doc-wide if a product
        # has no directly-linked chunks.
        chunks_resp = supabase.client.table("document_chunks") \
            .select("content, product_id, metadata") \
            .eq("document_id", document_id) \
            .execute()
        all_chunks = chunks_resp.data or []

        # ── Fetch vision_analysis rows for the whole document ─────────────
        vision_resp = supabase.client.table("document_images") \
            .select("page_number, vision_analysis") \
            .eq("document_id", document_id) \
            .not_.is_("vision_analysis", "null") \
            .execute()
        all_vision = vision_resp.data or []

        # ── Fetch image→product associations for page resolution fallback ──
        # Used when a product has no explicit page_range on its metadata:
        # every image linked to the product via image_product_associations
        # gives us a concrete page number to target with the spec vision pass.
        # Cheap — one query per document — and then in-memory lookups per
        # product. Keeps pipeline stages decoupled (no dependency on the
        # YOLO pass populating metadata.page_range up front).
        try:
            assoc_resp = supabase.client.table("image_product_associations") \
                .select("product_id, document_images!inner(page_number, document_id)") \
                .execute()
            product_to_image_pages: Dict[str, set] = {}
            for row in (assoc_resp.data or []):
                di = row.get("document_images") or {}
                if di.get("document_id") != document_id:
                    continue
                pn = di.get("page_number")
                if pn is None:
                    continue
                pid = row.get("product_id")
                if not pid:
                    continue
                product_to_image_pages.setdefault(pid, set()).add(int(pn))
        except Exception as e:
            logger.warning(f"   ⚠️ image→product page fallback unavailable: {e}")
            product_to_image_pages = {}

        # ── Per-product enrichment ────────────────────────────────────────
        for product in products:
            product_id = product["id"]
            product_name = product.get("name") or "(unnamed)"
            metadata = product.get("metadata") or {}

            # ── Scope chunks to this product FIRST ───────────────────────
            # We resolve chunks before pages because chunks are the best
            # source of page_range when the product row doesn't have one.
            product_chunks = [c for c in all_chunks if c.get("product_id") == product_id]
            if not product_chunks:
                product_chunks = all_chunks  # doc-wide fallback

            # ── Resolve the product's PDF page range ─────────────────────
            # Priority order (each is tried in turn until we get pages):
            #   1. products.metadata.page_range  — set by explicit Stage 4.5
            #   2. Union of document_chunks.metadata.product_pages from the
            #      chunks linked directly to this product via product_id
            #   3. Pages from document_images linked through
            #      image_product_associations
            #   4. Scan the product's own chunk text for "Page NN" / "---
            #      # Page NN ---" markers that the MIVAA Markdown extractor
            #      emits inline
            # Any of these gives us a concrete list for the spec vision pass.
            page_range = metadata.get("page_range") or []
            if not isinstance(page_range, list):
                page_range = []
            page_set: set = set(int(p) for p in page_range if isinstance(p, (int, str)) and str(p).isdigit())

            if not page_set:
                for c in product_chunks:
                    cmd = c.get("metadata")
                    if isinstance(cmd, dict):
                        for p in (cmd.get("product_pages") or []):
                            try:
                                page_set.add(int(p))
                            except (TypeError, ValueError):
                                pass

            if not page_set:
                assoc_pages = product_to_image_pages.get(product_id) or set()
                page_set |= assoc_pages

            if not page_set and product_chunks:
                # Last resort: scan chunk text for page markers.
                # MIVAA emits `--- # Page 29 ---` and bare `Page 29` variants.
                _page_marker_re = _re.compile(
                    r"(?:---\s*#\s*Page\s+(\d{1,4})\s*---|(?:^|\s)Page\s+(\d{1,4})(?:\s|$))",
                    _re.IGNORECASE,
                )
                for c in product_chunks:
                    txt = c.get("content") or ""
                    for m in _page_marker_re.finditer(txt):
                        raw = m.group(1) or m.group(2)
                        if raw and raw.isdigit():
                            n = int(raw)
                            if 1 <= n <= 10_000:
                                page_set.add(n)

            combined_text = " ".join(
                c.get("content", "") for c in product_chunks if c.get("content")
            )

            # ── Scope vision rows by page_range ──────────────────────────
            if page_set:
                product_vision = [
                    v for v in all_vision if v.get("page_number") in page_set
                ]
            else:
                product_vision = all_vision

            chunk_candidates = _extract_fields_from_chunk_text(combined_text)
            vision_candidates = _rollup_vision_analysis(product_vision)

            # ── Stage 4.7.c: Claude Vision spec page extractor ─────────────
            # Runs the PDF source pages through Claude Haiku Vision to pull
            # packing, tech characteristics, and certifications that chunks
            # and image vision_analysis don't cover.
            spec_vision_candidates: Dict[str, Any] = {}
            if enable_spec_vision and page_set:
                try:
                    from app.services.products.product_spec_vision_extractor import (
                        extract_specs_from_pdf_pages,
                        map_vision_specs_to_product_metadata,
                        _get_source_pdf_path,
                    )
                    pdf_path = _get_source_pdf_path(document_id)
                    if pdf_path:
                        raw_specs = extract_specs_from_pdf_pages(
                            pdf_path=pdf_path,
                            product_page_range=sorted(page_set),
                            product_name=product_name,
                        )
                        if raw_specs:
                            spec_vision_candidates = map_vision_specs_to_product_metadata(raw_specs)
                            stats["spec_vision_calls"] += 1
                    else:
                        logger.info(
                            f"   ℹ️ Spec vision: PDF source not on disk for doc {document_id}, skipping"
                        )
                except Exception as e:
                    logger.warning(
                        f"   ⚠️ Spec vision extractor failed for '{product_name}': {e}"
                    )

            if not chunk_candidates and not vision_candidates and not spec_vision_candidates:
                logger.info(
                    f"   ℹ️ Enrichment: no candidates found for '{product_name}' "
                    f"(pages={sorted(page_set) if page_set else 'all'})"
                )
                # Still try description writer even if structured fields empty
            else:
                # Merge spec-vision candidates in first (lowest priority — fills
                # only what chunks + vision_analysis didn't provide)
                pass

            new_metadata, filled = _merge_enriched_fields_into_metadata(
                existing_metadata=metadata,
                chunk_candidates=chunk_candidates,
                vision_candidates=vision_candidates,
            )

            # Second merge pass: bring in the nested dicts from spec_vision
            # Each top-level key in spec_vision_candidates is a nested container
            # (material_properties, performance, application, packaging, commercial,
            # compliance, plus two promoted fields dimensions_cm_from_vision /
            # dimensions_inch_from_vision). Merge each one into metadata without
            # clobbering existing keys.
            if spec_vision_candidates:
                extraction_meta = dict(new_metadata.get("_extraction_metadata") or {})
                for key, value in spec_vision_candidates.items():
                    if isinstance(value, dict):
                        # nested container — merge each child if missing
                        container = dict(new_metadata.get(key) or {})
                        any_added = False
                        for child_key, child_value in value.items():
                            if _is_empty_value(container.get(child_key)) and not _is_empty_value(child_value):
                                container[child_key] = child_value
                                extraction_meta[f"{key}.{child_key}"] = {
                                    "source": "claude_spec_vision", "confidence": 0.85
                                }
                                filled.append(f"{key}.{child_key}")
                                any_added = True
                        if any_added:
                            new_metadata[key] = container
                    else:
                        # flat promoted field
                        if _is_empty_value(new_metadata.get(key)) and not _is_empty_value(value):
                            new_metadata[key] = value
                            extraction_meta[key] = {"source": "claude_spec_vision", "confidence": 0.85}
                            filled.append(key)
                if filled:
                    new_metadata["_extraction_metadata"] = extraction_meta

            # ── Stage 4.7.d: Description writer (writes to products.description) ─
            # Separate from metadata because description is a top-level column.
            new_description: Optional[str] = None
            existing_description = (product.get("description") or "").strip() if isinstance(product.get("description"), str) else ""
            if enable_description_writer and not existing_description:
                try:
                    from app.services.products.product_description_writer import (
                        write_product_description_from_chunks,
                    )
                    new_description = write_product_description_from_chunks(
                        product_name=product_name,
                        chunks=product_chunks,
                    )
                    if new_description:
                        stats["description_writes"] += 1
                        filled.append("description")
                except Exception as e:
                    logger.warning(
                        f"   ⚠️ Description writer failed for '{product_name}': {e}"
                    )

            if not filled:
                logger.info(
                    f"   ℹ️ Enrichment: '{product_name}' already has all fields — "
                    f"no empty slots to fill"
                )
                continue

            # Re-fetch product row to get id (in case supabase select changed)
            update_payload: Dict[str, Any] = {"metadata": new_metadata}
            if new_description:
                update_payload["description"] = new_description

            supabase.client.table("products") \
                .update(update_payload) \
                .eq("id", product_id) \
                .execute()

            stats["products_updated"] += 1
            stats["fields_filled"].update(filled)
            logger.info(
                f"   ✅ Enriched '{product_name}': filled {filled} "
                f"(chunk_candidates={list(chunk_candidates.keys())}, "
                f"vision_candidates={list(vision_candidates.keys())})"
            )

            # ── Stage 4.7.e: Re-run AutoKBDocumentService with newly-enriched metadata ──
            # The existing product_processor wired this service at product creation,
            # but at that point packaging/compliance/care were all empty. Now that
            # enrichment has populated them, re-invoke the service to produce the KB
            # docs that were skipped the first time.
            try:
                from app.services.knowledge.auto_kb_document_service import AutoKBDocumentService
                kb_service = AutoKBDocumentService()
                kb_stats = await kb_service.create_kb_documents_from_metadata(
                    product_id=product_id,
                    product_name=product_name,
                    workspace_id=new_metadata.get("workspace_id") or product.get("workspace_id") or "",
                    metadata=new_metadata,
                )
                if kb_stats.get("documents_created"):
                    stats["kb_docs_created"] += kb_stats["documents_created"]
                    logger.info(
                        f"   📚 AutoKBDocumentService created {kb_stats['documents_created']} "
                        f"KB docs for '{product_name}' from enriched metadata"
                    )
            except Exception as kb_err:
                logger.warning(
                    f"   ⚠️ AutoKBDocumentService failed for '{product_name}': {kb_err}"
                )

        # ── Stage 4.7.f: Catalog-wide knowledge extraction (ONCE per document) ──
        # Runs after all products have been enriched. Uses Claude Vision on the
        # last ~10 pages of the PDF to extract iconography legends, regulations,
        # installation guides, care instructions, sustainability claims, etc.
        # Creates catalog-wide kb_docs and links them to every product.
        #
        # Only runs when we're enriching the WHOLE document (target_product_id
        # is None) — single-product backfill doesn't need to re-scan the catalog.
        if target_product_id is None and products:
            try:
                from app.services.knowledge.catalog_knowledge_extractor import (
                    extract_catalog_knowledge_from_pdf,
                    _get_source_pdf_path as _get_kb_pdf_path,
                )
                pdf_for_kb = _get_kb_pdf_path(document_id)
                if pdf_for_kb:
                    all_product_ids = [p["id"] for p in products if p.get("id")]
                    # Pick workspace_id from first product (all share one document)
                    workspace_id_for_kb = (
                        (products[0].get("metadata") or {}).get("workspace_id")
                        or products[0].get("workspace_id")
                        or ""
                    )
                    if not workspace_id_for_kb:
                        # Last resort: fetch from documents table
                        try:
                            doc_row = supabase.client.table("documents") \
                                .select("workspace_id") \
                                .eq("id", document_id) \
                                .limit(1) \
                                .execute()
                            if doc_row.data:
                                workspace_id_for_kb = doc_row.data[0].get("workspace_id") or ""
                        except Exception:
                            pass

                    if workspace_id_for_kb:
                        kb_cat_stats = await extract_catalog_knowledge_from_pdf(
                            document_id=document_id,
                            workspace_id=workspace_id_for_kb,
                            pdf_path=pdf_for_kb,
                            product_ids=all_product_ids,
                            supabase=supabase,
                            logger_instance=logger,
                        )
                        stats["catalog_kb_docs_created"] = kb_cat_stats.get("docs_created", 0)
                        stats["kb_attachments_created"] += kb_cat_stats.get("attachments_created", 0)
                    else:
                        logger.info(
                            f"   ℹ️ Catalog KB: no workspace_id available for {document_id}, skipping"
                        )
                else:
                    logger.info(
                        f"   ℹ️ Catalog KB: PDF not on disk for {document_id}, skipping"
                    )
            except Exception as kb_err:
                logger.warning(f"   ⚠️ Catalog knowledge extractor failed: {kb_err}")

        # Convert fields_filled set to list for JSON serialization
        stats["fields_filled"] = sorted(stats["fields_filled"])

        logger.info(
            f"✅ Stage 4.7 enrichment complete: "
            f"{stats['products_updated']}/{stats['products_checked']} products updated, "
            f"fields_filled={stats['fields_filled']}"
        )
        return stats

    except Exception as e:
        logger.error(f"❌ Stage 4.7 enrichment failed: {e}", exc_info=True)
        stats["error"] = str(e)
        stats["fields_filled"] = sorted(stats["fields_filled"]) if isinstance(stats["fields_filled"], set) else stats["fields_filled"]
        return stats
