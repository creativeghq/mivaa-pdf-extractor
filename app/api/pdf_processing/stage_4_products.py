"""
Stage 4: Product Creation

This module handles product creation in the database for the product-centric pipeline.
Includes metadata consolidation from AI text extraction, visual analysis, and factory defaults.
"""

import asyncio
import json
import logging
import os
import httpx
import sentry_sdk
from typing import Dict, Any, List, Optional

from app.services.metadata.metadata_normalizer import normalize_factory_keys
from app.services.facets import canonicalize_product_attributes

# ── Category → default unit mapping (mirrors material_categories.default_unit) ─
#
# Two-layer resolution:
#   1. Coarse "upload categories" (10 buckets, matches the
#      material_categories table on the DB and the upload-form selector)
#   2. Fine-grained vocab values from MATERIAL_CATEGORY_VOCAB. Stage 4.7's
#      auto-classifier writes these (e.g. 'porcelain_tile', 'sofa') so the
#      coarse-only map silently fell through to 'pcs' for every product.
#      Adding the fine-grained map below restores correct sqm units for
#      tile/wood/wall-paint products.
_CATEGORY_DEFAULT_UNITS: Dict[str, str] = {
    # Coarse buckets
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

# Fine-grained vocab → unit mapping. Mirrors MATERIAL_CATEGORY_VOCAB.
_FINE_CATEGORY_DEFAULT_UNITS: Dict[str, str] = {
    # Tiles → sqm
    'floor_tile': 'sqm', 'wall_tile': 'sqm', 'bathroom_tile': 'sqm',
    'shower_tile': 'sqm', 'porcelain_tile': 'sqm', 'ceramic_tile': 'sqm',
    # Wood / flooring → sqm
    'wood_flooring': 'sqm', 'laminate': 'sqm', 'vinyl_flooring': 'sqm',
    'carpet': 'sqm', 'hardwood': 'sqm', 'engineered_wood': 'sqm',
    'parquet': 'sqm',
    # Paint / wall decor → sqm (paintable surface) or pcs (panel/wallpaper roll)
    'wall_paint': 'sqm', 'wallpaper': 'sqm', 'decorative_plaster': 'sqm',
    'wall_panel': 'pcs', 'wall_coating': 'sqm',
    # General materials (slabs/sheets) → sqm; specific stones in linear/sqm
    'countertop': 'sqm', 'kitchen_worktop': 'sqm', 'stone_slab': 'sqm',
    'metal_panel': 'sqm', 'glass_panel': 'sqm', 'concrete': 'sqm',
    'terrazzo': 'sqm', 'quartz': 'sqm',
    # Furniture / decor / sanitary / kitchen / heating / lighting → pcs
    # (everything not enumerated above defaults to 'pcs' via the fallback)
}


def _resolve_default_unit(material_category: Optional[str]) -> str:
    """Resolve default unit from material category.

    Resolution order:
      1. Exact match against the fine-grained MATERIAL_CATEGORY_VOCAB map
         (e.g. 'porcelain_tile' → 'sqm').
      2. Exact match against the coarse upload-bucket map ('tiles' → 'sqm').
      3. Fuzzy substring match against the coarse map (catches mid-pipeline
         normalizations that strip the suffix).
      4. Fallback to 'pcs'.
    """
    if not material_category:
        return 'pcs'
    cat = material_category.lower().strip()
    # 1. Fine-grained vocab match — most specific wins
    if cat in _FINE_CATEGORY_DEFAULT_UNITS:
        return _FINE_CATEGORY_DEFAULT_UNITS[cat]
    # 2. Coarse bucket exact match
    if cat in _CATEGORY_DEFAULT_UNITS:
        return _CATEGORY_DEFAULT_UNITS[cat]
    # 3. Fuzzy match against coarse buckets
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
# Fine-grained material_category values that the AI classifier can assign.
# These map to the 10 upload categories via resolveUploadCategory() on the frontend
# and get_category_config() on the backend.
MATERIAL_CATEGORY_VOCAB = {
    # Tiles
    "floor_tile", "wall_tile", "bathroom_tile", "shower_tile",
    "porcelain_tile", "ceramic_tile",
    # Wood / Flooring
    "wood_flooring", "laminate", "vinyl_flooring", "carpet",
    "hardwood", "engineered_wood", "parquet",
    # Paint / Wall Decor
    "wall_paint", "wallpaper", "decorative_plaster", "wall_panel", "wall_coating",
    # Furniture
    "sofa", "armchair", "dining_chair", "accent_chair",
    "dining_table", "coffee_table", "side_table",
    "cabinet", "shelving", "sideboard", "bed", "desk",
    "outdoor_furniture",
    # Decor
    "rug", "curtain", "cushion", "vase", "mirror",
    "wall_art", "sculpture", "candle_holder", "planter",
    # General Materials
    "countertop", "kitchen_worktop", "stone_slab", "metal_panel",
    "glass_panel", "concrete", "terrazzo", "quartz",
    # Sanitary
    "toilet", "basin", "bathtub", "shower_tray", "bidet", "urinal",
    "vanity_unit", "shower_enclosure", "tap", "faucet", "mixer", "shower_head",
    # Kitchen
    "kitchen_cabinet", "kitchen_sink", "kitchen_tap", "kitchen_hood",
    "kitchen_appliance", "kitchen_handle", "kitchen_organiser",
    # Heating
    "radiator", "towel_rail", "underfloor_heating",
    "heat_pump", "boiler", "fireplace", "convector",
    # Lighting
    "lighting", "pendant_light", "ceiling_light", "wall_light",
    "floor_lamp", "table_lamp", "spotlight", "track_light",
    "recessed_light", "outdoor_light", "chandelier",
    # Legacy / generic
    "door", "window", "fabric_swatch", "leather_swatch",
}
ZONE_INTENT_VOCAB = {"surface", "full_object", "upholstery", "sub_element"}


async def _classify_product(
    name: str,
    description: str,
    existing_category: str,
    *,
    job_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    product_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Call Claude Haiku to assign material_category + zone_intent from controlled vocabulary.
    Returns dict with keys 'material_category' and 'zone_intent', or empty dict on failure.

    Routes through tracked_claude_call_async so per-product Anthropic spend
    is logged to ai_usage_logs with the right job_id / product_id /
    workspace_id — previously bypassed via raw httpx.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {}

    import json as _json

    prompt = f"""Classify this interior material/furniture product into the controlled vocabularies below.

Product name: {name}
Description: {description or 'N/A'}
Current category (may be wrong or missing): {existing_category or 'N/A'}

CONTROLLED VOCABULARY — respond ONLY with a JSON object, no explanation:

material_category (pick exactly one):
  TILES: floor_tile | wall_tile | bathroom_tile | shower_tile | porcelain_tile | ceramic_tile
  WOOD: wood_flooring | laminate | vinyl_flooring | carpet | hardwood | engineered_wood | parquet
  PAINT/WALL: wall_paint | wallpaper | decorative_plaster | wall_panel
  FURNITURE: sofa | armchair | dining_chair | accent_chair | dining_table | coffee_table | side_table | cabinet | shelving | sideboard | bed | desk | outdoor_furniture
  DECOR: rug | curtain | cushion | vase | mirror | wall_art | sculpture | planter
  GENERAL: countertop | kitchen_worktop | stone_slab | metal_panel | glass_panel | concrete | terrazzo | quartz
  SANITARY: toilet | basin | bathtub | shower_tray | bidet | vanity_unit | shower_enclosure | tap | faucet | mixer | shower_head
  KITCHEN: kitchen_cabinet | kitchen_sink | kitchen_tap | kitchen_hood | kitchen_appliance
  HEATING: radiator | towel_rail | underfloor_heating | heat_pump | boiler | fireplace | convector
  LIGHTING: lighting | pendant_light | ceiling_light | wall_light | floor_lamp | table_lamp | spotlight | chandelier | recessed_light

zone_intent (pick exactly one):
  surface     — floor/wall/ceiling tiles, paint, wallpaper, countertops, cladding
  full_object — sofa, chair, rug, curtain, table, cabinet, lamp, radiator, toilet, basin
  upholstery  — fabric/leather swatches for covering furniture
  sub_element — hardware, handles, trims, brackets, taps, faucets

Respond with exactly: {{"material_category": "...", "zone_intent": "..."}}"""

    try:
        from app.services.core.claude_helper import tracked_claude_call_async
        resp = await tracked_claude_call_async(
            task="product_classification",
            model="claude-haiku-4-5",
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
            job_id=job_id,
            workspace_id=workspace_id,
            product_id=product_id,
        )
        raw = (resp.content[0].text if resp.content else "").strip()
        # The model sometimes wraps the JSON in a ```json … ``` fence — strip it
        # before parsing so a cosmetic wrapper doesn't fail the whole classify.
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw[:4].lower() == "json":
                raw = raw[4:].strip()
        data = _json.loads(raw)
        result = {}
        if data.get("material_category") in MATERIAL_CATEGORY_VOCAB:
            result["material_category"] = data["material_category"]
        if data.get("zone_intent") in ZONE_INTENT_VOCAB:
            result["zone_intent"] = data["zone_intent"]
        return result
    except Exception as e:
        # A classification failure leaves the product uncategorized (wrong
        # default unit + broken facets), so surface it loudly rather than at
        # warning — this is a wholesale failure (parse / API), not "the model
        # legitimately couldn't classify" (which returns a valid-but-empty dict).
        logging.getLogger(__name__).error(
            f"Product classification failed for product {product_id}: {e}"
        )
        try:
            import sentry_sdk
            sentry_sdk.capture_exception(e)
        except Exception:
            pass
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

    # Fail-closed default (audit gap C): if the embedding block never runs,
    # the product has no text_embedding_1024 and MUST surface as failed so
    # the orchestrator marks it for re-embedding. The previous
    # `'embedding_failed' in locals()` guard at the return silently mapped
    # "never attempted" to success.
    embedding_failed = True

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

    # Fetch the canonical spec field taxonomy ONCE per product creation.
    # Used for both icon metadata rollup AND the embedding text builder
    # so they agree on what counts as a "spec field".
    known_spec_fields = await _fetch_known_spec_fields(supabase, logger)

    # Roll up per-image icon_metadata into flat top-level spec keys
    # (e.g. metadata['slip_resistance'] = 'R10'). The icon extraction pipeline
    # writes per-image audit data to document_images.metadata['icon_metadata'];
    # here we promote the highest-confidence value for each field onto the
    # product itself, normalized against material_metadata_fields.
    icon_rollup = await _merge_icon_metadata_into_product(
        document_id=document_id,
        image_indices=product.image_indices,
        known_spec_fields=known_spec_fields,
        supabase=supabase,
        logger=logger,
    )

    # Consolidate metadata from all sources
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

    # Merge icon rollup into top-level metadata. Icon-extracted spec values
    # are the most authoritative source for technical specs (R-rating, PEI,
    # fire rating, frost resistance) because they come from canonical icons
    # in the catalog — write them last to override AI text guesses.
    #
    # `_unknown_field_counts` is an audit-only sentinel produced by
    # `_merge_icon_metadata_into_product` (see audit fix #42). It must NOT
    # leak onto the product row — it would be picked up by the embedding text
    # builder below (which iterates known_spec_fields rigorously, but the
    # row-level metadata dict is also scanned by the frontend / search /
    # propagation paths). Strip sentinel keys before merging and stash the
    # audit data under a private prefix on the product instead.
    if icon_rollup:
        unknown_counts = icon_rollup.pop('_unknown_field_counts', None)
        for spec_field, spec_value in icon_rollup.items():
            if spec_field.startswith('_'):
                # Defense-in-depth: never write any underscore-prefixed
                # sentinel from the icon rollup as a top-level spec field.
                continue
            metadata[spec_field] = spec_value
        if unknown_counts:
            audit_block = metadata.get('_icon_rollup_audit') or {}
            if not isinstance(audit_block, dict):
                audit_block = {}
            audit_block['unknown_field_counts'] = unknown_counts
            metadata['_icon_rollup_audit'] = audit_block
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
        # Mirror the two filter-relevant fields to top-level metadata so the
        # frontend's `.contains('metadata', { factory_name: X })` filter works.
        # Used by MyFactoryTab and MarketTrendsTab — these flat keys are
        # canonical for SQL-level metadata-jsonb filtering, not legacy mirrors.
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
                job_id=job_id,
                workspace_id=workspace_id,
            )
            if needs_category and classified.get("material_category"):
                metadata["material_category"] = classified["material_category"]
                logger.info(f"   🏷️ Auto-classified material_category: {classified['material_category']}")
            if needs_intent and classified.get("zone_intent"):
                metadata["zone_intent"] = classified["zone_intent"]
                logger.info(f"   🏷️ Auto-classified zone_intent: {classified['zone_intent']}")
        except Exception as cls_err:
            logger.warning(f"   ⚠️ Auto-classification skipped: {cls_err}")

    # Extract description from multiple sources if product.description is empty
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

    # ── Auto-canonicalize descriptive facets (multilingual auto-merge) ────
    # Whitelisted facet values (color, material, finish, style, …) flow
    # through L1 string-normalize + L2 Voyage cosine cluster vs
    # facet_canonical_values. Greek/German/Italian raw values auto-collapse
    # to canonical English. Identifiers, numerics, codes are skipped.
    # Failures degrade silently — product insert is never blocked by this.
    #
    # 30s timeout (2026-05-23): Voyage embedding calls can hang on rate-limit
    # or network blip. Without a wait_for, the product insert would block
    # indefinitely behind canonicalization. We degrade to empty attributes
    # (preserving attributes_raw) on timeout — same contract as a Voyage error.
    import asyncio as _asyncio
    try:
        canonical = await _asyncio.wait_for(
            canonicalize_product_attributes(supabase, metadata, source='pdf_stage_4'),
            timeout=30.0,
        )
    except _asyncio.TimeoutError:
        logger.error(
            f"   ⏱️ Facet canonicalization exceeded 30s for {product.name} — "
            f"degrading to empty canonical attributes (raw values preserved). "
            f"Voyage/HF outage suspected."
        )
        # Degrade to empty CANONICAL attributes but keep the lossless raw map:
        # attributes_raw is the replay contract — with it, a later
        # re-canonicalization pass can rebuild attributes without re-ingesting;
        # without it the product is permanently unfacetable.
        # collect_raw_attributes applies the same whitelist as the
        # canonicalizer and returns the proper Dict[str, List[str]] shape.
        from app.services.facets import CanonicalizedAttributes
        from app.services.facets.facet_canonicalizer import collect_raw_attributes
        canonical = CanonicalizedAttributes(
            attributes={},
            attributes_raw=collect_raw_attributes(metadata),
            resolutions=[],
        )
    if canonical.resolutions:
        logger.info(
            f"   🏷️  Canonicalized {len(canonical.resolutions)} facet values "
            f"across {len(canonical.attributes)} facets"
        )

    product_data = {
        'source_document_id': document_id,
        'workspace_id': workspace_id,
        'name': product.name,
        'description': description,
        'metadata': metadata,
        'attributes': canonical.attributes,
        'attributes_raw': canonical.attributes_raw,
        'source_type': 'pdf_processing',
        'source_job_id': job_id
    }

    # Audit fix #40: validate FK targets before insert. Previously a missing
    # source_document_id would silently insert with the row failing at DB
    # constraint level (or worse, succeeding without the relationship if
    # constraints aren't enforced). Verifying upstream lets us fail with a
    # clear error instead of relying on opaque DB constraint messages.
    if document_id:
        try:
            doc_check = supabase.client.table('documents')\
                .select('id').eq('id', document_id).limit(1).execute()
            if not doc_check.data:
                raise Exception(
                    f"FK validation failed: source_document_id={document_id} does not exist in documents. "
                    f"Refusing to insert orphan product '{product.name}'."
                )
        except Exception as fk_err:
            if 'FK validation' in str(fk_err):
                raise
            # Network glitch — log and proceed (DB constraint will catch it).
            logger.warning(f"   ⚠️ FK pre-check failed (network?), proceeding: {fk_err}")

    result = supabase.client.table('products').insert(product_data).execute()

    if result.data and len(result.data) > 0:
        product_id = result.data[0]['id']
        logger.info(f"   ✅ Created product in DB: {product.name} (ID: {product_id})")

        # Generate text_embedding_1024 for product-level vector search.
        # Builds a rich text from name + description + key metadata fields,
        # then embeds with Voyage AI (1024D) — same model as document_chunks.
        try:
            # Shared with the product-embedding backfill — keep the text
            # construction identical so backfilled vectors live in the same
            # semantic space as inline-generated ones.
            embedding_text = build_product_embedding_text(
                name=product.name,
                description=description,
                metadata=metadata,
                known_spec_fields=known_spec_fields,
            )

            from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
            embeddings_svc = RealEmbeddingsService()

            text_emb = None
            last_err: Optional[Exception] = None
            for attempt in range(3):
                try:
                    emb_result = await embeddings_svc.generate_text_embedding(embedding_text)
                    if emb_result.get('success') and emb_result.get('embedding'):
                        text_emb = emb_result['embedding']
                        break
                    last_err = Exception(emb_result.get('error') or 'no embedding returned')
                except Exception as e:
                    last_err = e
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2 ** attempt))

            if text_emb:
                # Audit fix #16: validate dimension before storage so we don't
                # silently store a wrong-dim string (the OpenAI fallback now
                # pins to 1024 but defense-in-depth here).
                if len(text_emb) != 1024:
                    logger.error(
                        f"   ❌ Product embedding for {product.name} returned wrong dim "
                        f"({len(text_emb)}D, expected 1024D). Refusing to store."
                    )
                    embedding_failed = True
                else:
                    # Persist provenance alongside the vector so the admin UI
                    # can detect Voyage→OpenAI fallback drift and so the
                    # backfill cron can target stale-schema rows. Audit gap A.
                    embedding_str = '[' + ','.join(str(x) for x in text_emb) + ']'
                    embedding_model = emb_result.get('model') if isinstance(emb_result, dict) else None
                    update_payload: Dict[str, Any] = {
                        'text_embedding_1024': embedding_str,
                        'text_embedding_schema_version': 1,
                    }
                    if embedding_model:
                        update_payload['text_embedding_1024_model'] = embedding_model
                    supabase.client.table('products').update(
                        update_payload
                    ).eq('id', product_id).execute()
                    logger.info(f"   🧠 Generated text_embedding_1024 for {product.name} (model={embedding_model or 'voyage-4'})")
                    embedding_failed = False
            else:
                logger.error(
                    f"   ❌ Product embedding failed after 3 attempts for {product.name}: {last_err}"
                )
                embedding_failed = True

        except Exception as emb_err:
            logger.error(f"   ❌ Product embedding generation crashed for {product.name}: {emb_err}")
            embedding_failed = True

        # Audit fix #15: surface embedding failure in the return so the orchestrator
        # can mark the product as needing re-embedding rather than silently treating
        # it as a successful product creation. Previously embedding-failed products
        # had text_embedding_1024=NULL and were invisible to vector search forever.

        # Audit fix #41: write image_product_associations immediately when we
        # know the image set, instead of deferring to Stage 4.7. Closes the
        # window where products are searchable but not linked to their images.
        # Best-effort — failure here doesn't block the product creation;
        # Stage 4.7 still runs as a backstop.
        # Field name verified 2026-05-01: DiscoveredProduct uses `page_range`
        # (List[int]), NOT product_pages — see document_entity_service.py:42.
        try:
            page_range = (
                getattr(product, 'page_range', None)
                or getattr(product, 'product_pages', None)  # forward-compat
                or []
            )
            if page_range:
                img_resp = supabase.client.table('document_images')\
                    .select('id, page_number')\
                    .eq('document_id', document_id)\
                    .in_('page_number', page_range).execute()
                image_ids = [r['id'] for r in (img_resp.data or [])]
                if image_ids:
                    # Verified 2026-05-01: image_product_associations columns
                    # are (id, image_id, product_id, spatial_score, caption_score,
                    # clip_score, overall_score, confidence, reasoning, metadata,
                    # created_at, updated_at). Unique constraint is (image_id,
                    # product_id) — note the order matters for on_conflict.
                    # association_method has no dedicated column, store under metadata.
                    rows = [
                        {'product_id': product_id, 'image_id': iid,
                         'metadata': {'association_method': 'stage_4_immediate_by_page'}}
                        for iid in image_ids
                    ]
                    supabase.client.table('image_product_associations')\
                        .upsert(rows, on_conflict='image_id,product_id').execute()
                    logger.info(
                        f"   🔗 Linked {len(image_ids)} images to product {product_id} "
                        f"(immediate, audit fix #41, page_range={page_range})"
                    )
        except Exception as assoc_err:
            logger.warning(
                f"   ⚠️ Immediate image-product association failed (Stage 4.7 will retry): {assoc_err}"
            )

        return {'product_id': product_id, 'embedding_failed': embedding_failed}
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

        images_response = supabase.client.table('document_images') \
            .select('id, metadata') \
            .eq('document_id', document_id) \
            .in_('page_number', image_indices) \
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


def build_product_embedding_text(
    name: Optional[str],
    description: Optional[str],
    metadata: Dict[str, Any],
    known_spec_fields: List[str],
) -> str:
    """Build the canonical product embedding text from name + description +
    searchable metadata + spec fields.

    Single source of truth shared by Stage 4 inline generation AND the
    product-embedding backfill — both must produce byte-identical text for
    the same inputs so backfilled vectors live in the same semantic space.
    """
    embedding_text_parts = [name or '']
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

    # Walk every known spec field from material_metadata_fields and append
    # non-null values to the embedding text. A search like
    # "porcelain tile R10 PEI IV frost resistant" matches a product whose
    # spec icons have been extracted (rolled up from icon_metadata).
    # The loop runs over the canonical taxonomy so adding a new spec to
    # material_metadata_fields automatically flows it into embeddings.
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

    return ' | '.join(embedding_text_parts)


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
            # Audit fix #42: persist the dropped-field counts in the rollup
            # under a sentinel key so an admin can later widen the
            # known_spec_set / curate aliases. Was previously logged once
            # then forgotten — no signal for which typos to alias.
            rollup['_unknown_field_counts'] = dict(
                sorted(unknown_fields.items(), key=lambda x: -x[1])[:20]
            )

        return rollup

    except Exception as e:
        # Empty dict on exception is shape-identical to "no icons present" —
        # caller can't distinguish DB read failure from a clean zero. Bump
        # the log level and capture to Sentry so silent DB issues here
        # don't get filed as "no spec icons" in the rollup.
        logger.error(f"   ❌ Failed to merge icon metadata into product: {e}", exc_info=True)
        try:
            import sentry_sdk
            sentry_sdk.capture_exception(e)
        except Exception:
            pass
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
                    # Mirror to top-level metadata for jsonb-contains filtering
                    # by frontend (MyFactoryTab / MarketTrendsTab). Same
                    # rationale as the assembly path above.
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
# Maps AI-freeform material_type strings (from vision analysis) to
# controlled-vocab values that exist in MATERIAL_CATEGORY_VOCAB.
# Every value here MUST be present in the vocab set above.
_MATERIAL_TYPE_TO_CATEGORY = {
    # Tiles
    "ceramic tile": "ceramic_tile",
    "porcelain tile": "porcelain_tile",
    "stoneware": "floor_tile",       # normalize to floor_tile (in vocab)
    "stoneware tile": "floor_tile",
    "mosaic": "wall_tile",           # normalize to wall_tile (in vocab)
    "mosaic tile": "wall_tile",
    "outdoor tile": "floor_tile",    # normalize to floor_tile (in vocab)
    "wall tile": "wall_tile",
    "floor tile": "floor_tile",
    "bathroom tile": "bathroom_tile",
    "shower tile": "shower_tile",
    # Stone / general materials
    "natural stone": "stone_slab",
    "marble": "stone_slab",
    "granite": "stone_slab",
    "slate": "stone_slab",
    "limestone": "stone_slab",
    "travertine": "stone_slab",
    "quartz": "quartz",
    "terrazzo": "terrazzo",
    "concrete": "concrete",
    # Wood
    "wood": "wood_flooring",
    "wood flooring": "wood_flooring",
    "hardwood": "hardwood",
    "engineered wood": "engineered_wood",
    "parquet": "parquet",
    "laminate": "laminate",
    "vinyl": "vinyl_flooring",
    "bamboo": "wood_flooring",
    # Furniture
    "sofa": "sofa",
    "chair": "dining_chair",
    "table": "dining_table",
    "cabinet": "cabinet",
    # Sanitary
    "toilet": "toilet",
    "basin": "basin",
    "bathtub": "bathtub",
    "tap": "tap",
    "faucet": "faucet",
    # Heating
    "radiator": "radiator",
    "towel rail": "towel_rail",
    "boiler": "boiler",
    # Lighting
    "light": "lighting",
    "lamp": "lighting",
    "pendant": "pendant_light",
    "chandelier": "chandelier",
    # Paint / Wall
    "paint": "wall_paint",
    "wallpaper": "wallpaper",
    # Kitchen
    "worktop": "kitchen_worktop",
    "countertop": "countertop",
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


def _case_fold_key(s: str) -> str:
    """Normalization key for case-insensitive grouping in Counter()."""
    return s.strip().lower()


def _most_common_pretty(values: List[str]) -> Optional[str]:
    """Return the most-common value (case-insensitive), preserving the most
    common ORIGINAL case. Returns None on empty input.

    Resolves a long-standing bug where the rollup `.lower()`-ed everything
    before write, so frontend always saw "matte" / "glossy" never "Matte".
    """
    if not values:
        return None
    from collections import Counter
    # Group by case-folded key; among the values sharing each key, pick the
    # representative that appeared most often in original form.
    groups: Dict[str, List[str]] = {}
    for v in values:
        groups.setdefault(_case_fold_key(v), []).append(v.strip())
    fold_counts = Counter(_case_fold_key(v) for v in values)
    winning_fold, _ = fold_counts.most_common(1)[0]
    # Among the originals that shared the winning fold, pick the most-frequent.
    repr_counts = Counter(groups[winning_fold])
    winner, _ = repr_counts.most_common(1)[0]
    return winner


def _dedupe_pretty(values: List[str], cap: int) -> List[str]:
    """Case-insensitive union; preserve order by frequency, original case."""
    if not values:
        return []
    from collections import Counter
    fold_counts = Counter(_case_fold_key(v) for v in values)
    repr_by_fold: Dict[str, str] = {}
    for v in values:
        k = _case_fold_key(v)
        if k not in repr_by_fold:
            repr_by_fold[k] = v.strip()
    return [repr_by_fold[k] for k, _ in fold_counts.most_common(cap)]


def _rollup_vision_analysis(vision_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure function: majority-vote rollup of per-image vision_analysis into
    product-level fields.

    Reads the schema-locked field names from
    `app.models.vision_analysis.VisionAnalysis`. Older legacy keys
    (`pattern` / `texture` / `design_style` / `material_subtype` /
    `primary_color_hex` / `color_palette`) are still accepted as fallbacks
    for pre-2026-05-01 data still in the DB.

    Args:
        vision_rows: List of dicts, each containing a vision_analysis JSON blob.

    Returns:
        Candidate fields including: material_category, finish, design_style,
        pattern (+ patterns list), texture (+ textures list), appearance_colors,
        applications, category, subcategory, description, detected_text,
        vision_confidence.
    """
    if not vision_rows:
        return {}

    from collections import Counter

    material_types: List[str] = []
    finishes: List[str] = []
    surface_patterns: List[str] = []
    all_textures: List[str] = []
    styles: List[str] = []
    all_colors: List[str] = []
    all_applications: List[str] = []
    categories: List[str] = []
    subcategories: List[str] = []
    descriptions: List[str] = []
    all_detected_text: List[str] = []
    confidences: List[float] = []

    for row in vision_rows:
        va = row.get("vision_analysis") or {}
        if not isinstance(va, dict):
            continue

        # Required field
        mt = va.get("material_type")
        if isinstance(mt, str) and mt.strip():
            material_types.append(mt.strip())

        # Schema field: category
        cat = va.get("category")
        if isinstance(cat, str) and cat.strip():
            categories.append(cat.strip())

        # Schema field: subcategory (also accept legacy material_subtype)
        sub = va.get("subcategory") or va.get("material_subtype")
        if isinstance(sub, str) and sub.strip():
            subcategories.append(sub.strip())

        # Schema field: finish (string, single keyword)
        finish = va.get("finish")
        if isinstance(finish, str) and finish.strip():
            finishes.append(finish.strip())

        # Schema field: surface_pattern (singular). Legacy: pattern.
        sp = va.get("surface_pattern") or va.get("pattern")
        if isinstance(sp, str) and sp.strip():
            surface_patterns.append(sp.strip())

        # Schema field: textures (list). Legacy: texture (string).
        textures = va.get("textures")
        if isinstance(textures, list):
            for t in textures:
                if isinstance(t, str) and t.strip():
                    all_textures.append(t.strip())
        elif isinstance(textures, str) and textures.strip():
            all_textures.append(textures.strip())
        legacy_tex = va.get("texture")
        if isinstance(legacy_tex, str) and legacy_tex.strip():
            all_textures.append(legacy_tex.strip())

        # Schema field: style. Legacy: design_style.
        style = va.get("style") or va.get("design_style")
        if isinstance(style, str) and style.strip():
            styles.append(style.strip())

        # Schema field: colors. Legacy: color_palette.
        palette = va.get("colors") or va.get("color_palette") or []
        if isinstance(palette, list):
            for c in palette:
                if isinstance(c, str) and c.strip():
                    all_colors.append(c.strip())

        # Schema field: applications.
        apps = va.get("applications") or []
        if isinstance(apps, list):
            for a in apps:
                if isinstance(a, str) and a.strip():
                    all_applications.append(a.strip())

        # Schema field: description (one-sentence visual description).
        desc = va.get("description")
        if isinstance(desc, str) and desc.strip():
            descriptions.append(desc.strip())

        # Schema field: detected_text (OCR-style list).
        dt = va.get("detected_text")
        if isinstance(dt, list):
            for t in dt:
                if isinstance(t, str) and t.strip():
                    all_detected_text.append(t.strip())

        # Schema field: confidence (0.0-1.0).
        conf = va.get("confidence")
        if isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0:
            confidences.append(float(conf))

    candidates: Dict[str, Any] = {}

    # Material category — uses the controlled-vocab normalizer (lowercases
    # internally because the vocab is lowercase slugs like 'porcelain_tile').
    if material_types:
        most_common_mt, _count = Counter(m.lower() for m in material_types).most_common(1)[0]
        normalized = _normalize_material_category(most_common_mt)
        if normalized:
            candidates["material_category"] = normalized

    # High-level vision category ("flooring", "wall covering")
    cat_winner = _most_common_pretty(categories)
    if cat_winner:
        candidates["category"] = cat_winner

    # Subcategory — written as `material_subtype` for modal consumption
    # (existing UI consumer reads materialPropsData.material_subtype).
    sub_winner = _most_common_pretty(subcategories)
    if sub_winner:
        candidates["material_subtype"] = sub_winner
        candidates["subcategory"] = sub_winner

    # Finish — preserve original case ("Matte" stays "Matte").
    finish_winner = _most_common_pretty(finishes)
    if finish_winner:
        candidates["finish"] = finish_winner

    # Surface pattern — singular, plus dedupe-cap chip list.
    if surface_patterns:
        candidates["pattern"] = _most_common_pretty(surface_patterns)
        candidates["patterns"] = _dedupe_pretty(surface_patterns, cap=20)

    # Textures — singular best + plural list.
    if all_textures:
        candidates["texture"] = _most_common_pretty(all_textures)
        candidates["textures"] = _dedupe_pretty(all_textures, cap=10)

    # Style — schema field. Also write legacy `design_style` key so the
    # existing modal read (designData.design_style) keeps working.
    style_winner = _most_common_pretty(styles)
    if style_winner:
        candidates["design_style"] = style_winner
        candidates["style"] = style_winner

    if all_colors:
        candidates["appearance_colors"] = _dedupe_pretty(all_colors, cap=20)

    if all_applications:
        candidates["applications"] = _dedupe_pretty(all_applications, cap=10)

    if descriptions:
        # Pick the most-descriptive (longest) — descriptions are typically unique
        # per image; majority vote rarely converges. Length is the best signal.
        candidates["vision_description"] = max(descriptions, key=len)

    if all_detected_text:
        candidates["detected_text"] = _dedupe_pretty(all_detected_text, cap=20)

    if confidences:
        # Mean confidence across the product's images.
        candidates["vision_confidence"] = round(sum(confidences) / len(confidences), 3)

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
    fill_if_empty("textures", vision_candidates.get("textures"), "vision_rollup", container="appearance")
    fill_if_empty("design_style", vision_candidates.get("design_style"), "vision_rollup", container="design")
    fill_if_empty("style", vision_candidates.get("style"), "vision_rollup", container="design")
    fill_if_empty("applications", vision_candidates.get("applications"), "vision_rollup")
    # New schema-aligned propagation (post-2026-05-04 — was orphaned in the
    # legacy rollup that read fields the schema doesn't emit).
    fill_if_empty("category", vision_candidates.get("category"), "vision_rollup", container="appearance")
    fill_if_empty("subcategory", vision_candidates.get("subcategory"), "vision_rollup", container="appearance")
    fill_if_empty("vision_description", vision_candidates.get("vision_description"), "vision_rollup", container="appearance")
    fill_if_empty("detected_text", vision_candidates.get("detected_text"), "vision_rollup", container="appearance")
    fill_if_empty("vision_confidence", vision_candidates.get("vision_confidence"), "vision_rollup")

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
    enable_layout_analyzer: bool = True,
    enable_legend_extractor: bool = True,
    job_id: Optional[str] = None,
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
        # ── Fetch products (include description + workspace_id) ────────────
        q = supabase.client.table("products").select("id, name, description, workspace_id, metadata")
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

        # ── Layers 1 + 2: Catalog Layout + Catalog Legends (once/document) ──
        # These run BEFORE the per-product loop because every product shares
        # the same page classification and the same catalog-wide legends.
        # Both are idempotent — if they already ran for this document, the
        # calls return the cached result without doing work.
        catalog_layout: Dict[str, Any] = {}
        catalog_legends: Dict[str, Any] = {}

        # Resolve PDF path once; both layers need it.
        pdf_path = None
        try:
            from app.services.products.product_spec_vision_extractor import _get_source_pdf_path
            pdf_path = _get_source_pdf_path(document_id)
        except Exception as e:
            logger.warning(f"[{job_id or '-'}] 4.7.a: could not resolve pdf_path: {e}")

        if pdf_path and enable_layout_analyzer and not target_product_id:
            # Only run Layer 1/2 on whole-document passes, not single-product
            # backfills (those reuse the existing classification).
            try:
                from app.services.discovery.catalog_layout_analyzer import analyze_catalog_layout
                catalog_layout = await analyze_catalog_layout(
                    document_id=document_id,
                    pdf_path=pdf_path,
                    supabase=supabase,
                    job_id=job_id,
                    known_product_names=[p.get("name") for p in products if p.get("name")],
                )
                logger.info(
                    f"[{job_id or '-'}] 4.7.a: layout analyzer done — "
                    f"{catalog_layout.get('stats', {})}"
                )
            except Exception as layout_err:
                logger.warning(f"[{job_id or '-'}] 4.7.a layout analyzer failed: {layout_err}")
        else:
            # Single-product backfill path — load existing layout from DB so
            # Layer 3 can still use product_pages_by_name.
            try:
                row = supabase.client.table("documents") \
                    .select("metadata") \
                    .eq("id", document_id).limit(1).execute()
                if row.data:
                    catalog_layout = (row.data[0].get("metadata") or {}).get("catalog_layout") or {}
            except Exception:
                catalog_layout = {}

        if pdf_path and enable_legend_extractor and not target_product_id:
            try:
                from app.services.knowledge.catalog_legend_extractor_v2 import extract_catalog_legends
                workspace_id = products[0].get("workspace_id") if products else None
                if workspace_id:
                    legend_stats = await extract_catalog_legends(
                        document_id=document_id,
                        pdf_path=pdf_path,
                        workspace_id=workspace_id,
                        supabase=supabase,
                        job_id=job_id,
                    )
                    logger.info(f"[{job_id or '-'}] 4.7.b: legend extractor done — {legend_stats}")
                    # Reload doc metadata to pick up catalog_legends the service wrote
                    try:
                        row = supabase.client.table("documents") \
                            .select("metadata") \
                            .eq("id", document_id).limit(1).execute()
                        if row.data:
                            catalog_legends = (row.data[0].get("metadata") or {}).get("catalog_legends") or {}
                    except Exception:
                        pass
            except Exception as legend_err:
                logger.warning(f"[{job_id or '-'}] 4.7.b legend extractor failed: {legend_err}")
        else:
            # Load existing legends for single-product backfill
            try:
                row = supabase.client.table("documents") \
                    .select("metadata") \
                    .eq("id", document_id).limit(1).execute()
                if row.data:
                    catalog_legends = (row.data[0].get("metadata") or {}).get("catalog_legends") or {}
            except Exception:
                catalog_legends = {}

        # Make them available to the per-product loop via closure-local vars
        layout_product_pages = (catalog_layout or {}).get("product_pages_by_name") or {}

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
        # Each per-stage call is locally try/except-wrapped so a sub-stage
        # failure can't escape the loop. The merge + DB-write block at the
        # bottom is the one path still capable of raising — protected by an
        # outer try/except below so one bad product can't strand the rest.
        stats.setdefault("per_product_failures", 0)
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
            # We UNION four signals rather than short-circuiting on the first
            # one that produces output — earlier short-circuit logic silently
            # fell through to a noisy text-marker scan whenever the chunk
            # metadata fetch returned metadata as a string (Supabase JSONB
            # deserialization is not always dict-typed). The union approach
            # is both cheap and self-healing.
            #
            # Signals, in order of authority (all are tried):
            #   1. products.metadata.page_range      (explicit, if present)
            #   2. chunks.metadata.product_pages    (MIVAA Stage 2 output)
            #   3. image_product_associations       (YOLO layout result)
            #   4. '--- # Page NN ---' markers in chunk text (last resort)
            page_set: set = set()

            # Helper: parse a value that might be int, str, list, or serialized.
            def _to_int(val: Any) -> Optional[int]:
                try:
                    if isinstance(val, bool):
                        return None
                    if isinstance(val, int):
                        return val
                    s = str(val).strip()
                    return int(s) if s.isdigit() else None
                except Exception:
                    return None

            # 1. products.metadata.page_range
            for p in (metadata.get("page_range") or []):
                n = _to_int(p)
                if n is not None:
                    page_set.add(n)

            # 2. chunks.metadata.product_pages — handle both dict and JSON-string
            for c in product_chunks:
                cmd = c.get("metadata")
                if isinstance(cmd, str):
                    try:
                        cmd = json.loads(cmd)
                    except Exception:
                        cmd = None
                if not isinstance(cmd, dict):
                    continue
                for p in (cmd.get("product_pages") or []):
                    n = _to_int(p)
                    if n is not None:
                        page_set.add(n)

            # 3. image_product_associations
            for p in (product_to_image_pages.get(product_id) or set()):
                n = _to_int(p)
                if n is not None:
                    page_set.add(n)

            # 4. Text markers in chunk content (only as a supplement)
            if not page_set and product_chunks:
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

            if page_set:
                logger.info(
                    f"   🗺 '{product_name}' page_set resolved: {sorted(page_set)}"
                )

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

            # ── Stage 4.7.c: Product Spec Extractor v2 (3-tier hybrid) ─────
            # Tier A: PyMuPDF text-dict parser (free, deterministic)
            # Tier B: Claude Opus Vision (fallback when Tier A insufficient)
            # Tier C: Catalog legend inheritance (fills fields from Layer 2
            #         global legends, e.g. certifications, PEI defaults)
            #
            # Page resolution priority:
            #   1. Layer 1's authoritative text-based scan
            #      (catalog_layout.product_pages_by_name[product_name])
            #   2. page_set computed earlier from chunks + image associations
            spec_vision_candidates: Dict[str, Any] = {}
            resolved_page_indices: List[int] = []
            layout_pages = layout_product_pages.get(product_name) or []
            if layout_pages:
                # Layer 1 gives us 0-indexed PDF pages directly
                resolved_page_indices = sorted({int(p) for p in layout_pages})
            elif page_set:
                # Fallback to the chunk-derived page_set (1-indexed → 0-indexed)
                resolved_page_indices = sorted({int(p) - 1 for p in page_set if int(p) > 0})

            if enable_spec_vision and resolved_page_indices and pdf_path:
                try:
                    from app.services.products.product_spec_extractor_v2 import extract_product_spec
                    spec_vision_candidates = await extract_product_spec(
                        document_id=document_id,
                        product_id=product_id,
                        product_name=product_name,
                        pdf_path=pdf_path,
                        page_indices=resolved_page_indices,
                        catalog_legends=catalog_legends,
                        job_id=job_id,
                        enable_tier_b=True,
                    )
                    if spec_vision_candidates:
                        stats["spec_vision_calls"] += 1
                        tiers = spec_vision_candidates.pop("_source_tiers", [])
                        logger.info(
                            f"   ✅ spec_v2 '{product_name}': tiers={tiers}, "
                            f"pages={resolved_page_indices}"
                        )
                except Exception as e:
                    logger.warning(f"   ⚠️ spec_v2 failed for '{product_name}': {e}")
                    with sentry_sdk.configure_scope() as scope:
                        scope.set_tag("job_id", job_id)
                        scope.set_tag("document_id", document_id)
                        scope.set_tag("product_id", product_id)
                        scope.set_tag("stage", "stage_4.7.c_spec_v2")
                    sentry_sdk.capture_exception(e)
            elif not pdf_path:
                logger.info(
                    f"   ℹ️ spec_v2: no PDF on disk for doc {document_id}, skipping"
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
                    # Drop sentinel/audit keys (e.g. `_source_tiers` is popped
                    # earlier on the success path but `_error` stays on the
                    # failure path; both are extractor-internal, not product
                    # spec fields). Without this guard `_error` would be
                    # written as a top-level metadata key and surfaced on the
                    # frontend product detail view.
                    if isinstance(key, str) and key.startswith("_"):
                        continue
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
                        job_id=job_id,
                        product_id=product_id,
                        workspace_id=product.get("workspace_id"),
                    )
                    if new_description:
                        stats["description_writes"] += 1
                        filled.append("description")
                except Exception as e:
                    logger.warning(
                        f"   ⚠️ Description writer failed for '{product_name}': {e}"
                    )

            # ── Stage 4.7.e: Enhanced functional property extraction ─────
            # Single Claude Haiku call extracts 60+ structured properties
            # across 9 categories (slip safety, mechanical, thermal, etc.)
            # matching the frontend FunctionalMetadata interface.
            existing_functional = new_metadata.get("functional_metadata")
            if not existing_functional:
                try:
                    from app.services.products.enhanced_material_property_extractor import (
                        extract_functional_properties,
                    )
                    prop_result = await extract_functional_properties(
                        analysis_text=combined_text,
                        product_name=product_name,
                        job_id=job_id,
                        workspace_id=product.get("workspace_id"),
                        product_id=product_id,
                    )
                    if prop_result.coverage_pct > 0:
                        new_metadata["functional_metadata"] = prop_result.properties
                        new_metadata.setdefault("_extraction_metadata", {})["functional_metadata"] = {
                            "source": prop_result.method,
                            "confidence": prop_result.confidence,
                            "coverage_pct": prop_result.coverage_pct,
                            "processing_time_ms": prop_result.processing_time_ms,
                        }
                        filled.append("functional_metadata")
                        stats.setdefault("functional_property_extractions", 0)
                        stats["functional_property_extractions"] += 1
                except Exception as e:
                    logger.warning(
                        f"   ⚠️ Functional property extraction failed for '{product_name}': {e}"
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

            # Wrap the DB write so a transient Postgres / network error on
            # one product doesn't propagate out and silently strand every
            # subsequent product in this loop. The error is logged + counted
            # so an operator can see exactly how many products this affected.
            try:
                supabase.client.table("products") \
                    .update(update_payload) \
                    .eq("id", product_id) \
                    .execute()
            except Exception as db_err:
                stats["per_product_failures"] += 1
                logger.error(
                    f"   ❌ Enrichment DB write failed for '{product_name}' "
                    f"(product_id={product_id}): {db_err}"
                )
                try:
                    sentry_sdk.capture_exception(db_err)
                except Exception:
                    pass
                # Skip downstream KB regen for this product — its metadata
                # is in-memory but not persisted, so the AutoKBDocumentService
                # call below would write KB docs against stale data.
                continue

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
                # workspace_id lives on the product row itself — prefer that
                # over the metadata blob which is frequently missing the key.
                ws_id = (
                    product.get("workspace_id")
                    or new_metadata.get("workspace_id")
                    or None
                )
                if not ws_id:
                    logger.info(
                        f"   ℹ️ Skipping AutoKBDocumentService for '{product_name}' "
                        f"(no workspace_id resolvable — kb_docs requires a valid UUID)"
                    )
                else:
                    from app.services.knowledge.auto_kb_document_service import AutoKBDocumentService
                    kb_service = AutoKBDocumentService()
                    kb_stats = await kb_service.create_kb_documents_from_metadata(
                        product_id=product_id,
                        product_name=product_name,
                        workspace_id=ws_id,
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
                            job_id=job_id,
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
