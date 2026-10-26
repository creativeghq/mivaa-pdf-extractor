"""
Stage 4: Product Creation

This module handles product creation in the database for the product-centric pipeline.
Includes metadata consolidation from AI text extraction, visual analysis, and factory defaults.
"""

import asyncio
import json
import logging
import re
import os
import uuid
import httpx
import sentry_sdk
from typing import Any, Callable, Dict, List, Optional

from app.config import get_settings
from app.services.metadata.metadata_normalizer import normalize_factory_keys, scale_value
from app.services.facets import canonicalize_product_attributes
from app.services.metadata.field_validation import validate_metadata_against_registry
from app.services.metadata.field_registry import field_registry
from app.services.utilities.prompt_registry import load_prompt, prefetch, render
from app.services.products.category_units import (
    CategoryRegistryUnavailable, get_category_registry, resolve_default_unit,
)
from app.utils.postgrest_filters import escape_like
from app.utils.text_fold import fold_identity, fold_model_token


def _resolve_default_unit(material_category: Optional[str], supabase: Any) -> str:
    """Resolve the selling unit for a category key or a vocabulary value.

    Raises:
        CategoryRegistryUnavailable: `material_categories` could not be read.
    """
    units, vocab, _ids = get_category_registry(supabase)
    return resolve_default_unit(material_category, units, vocab)


def _resolve_category_id(material_category: Optional[str], supabase: Any) -> Optional[str]:
    """`material_categories.id` for the category the registry resolver picks; None if none.

    Raises:
        CategoryRegistryUnavailable: `material_categories` could not be read.
    """
    _units, _vocab, ids = get_category_registry(supabase)
    key = field_registry.resolve_category(material_category) if material_category else None
    return ids.get(key) if key else None


_UPLOAD_CATEGORY: Dict[str, Optional[str]] = {}


def _coarse_category(value: Any) -> Optional[str]:
    if not field_registry.is_loaded or not isinstance(value, str) or not value.strip():
        return None
    return field_registry.resolve_category(value)


def _upload_category_hint(supabase: Any, document_id: str) -> Optional[str]:
    """The category the operator declared on upload, when it is a specific one.

    It overrides the extractor's per-product guess (a wall tile came back as "sofa"); a
    catalogue that spans several categories is uploaded without one.
    """
    key = str(document_id)
    if key in _UPLOAD_CATEGORY:
        return _UPLOAD_CATEGORY[key]
    try:
        rows = supabase.client.table("documents").select("metadata").eq("id", key).limit(1).execute()
    except Exception as e:  # noqa: BLE001 — no hint is the old behaviour; not cached, so it retries
        logging.getLogger(__name__).warning(f"upload category hint unreadable for {key}: {e}")
        return None
    raw = (((rows.data or [{}])[0] or {}).get("metadata") or {}).get("material_category")
    coarse = _coarse_category(raw)
    _UPLOAD_CATEGORY[key] = coarse if coarse and coarse != "general_materials" else None
    return _UPLOAD_CATEGORY[key]


#: Registry keys with `destination='column'` that have a real products column. Order matters:
#: the first key carrying a value wins its column. An alias (`product_code`) resolves to its field.
_COLUMN_FOR_FIELD: Dict[str, str] = {
    'sku': 'sku',
    'mpn': 'mpn',
    'barcode': 'barcode',
    'warranty': 'warranty',
    'country_of_origin': 'country_of_origin',
}


def _route_column_fields(metadata: Dict[str, Any], category: Optional[str] = None,
                         country: Optional[Callable[[str], Optional[str]]] = None) -> Dict[str, str]:
    """Move registry `destination='column'` values out of metadata onto their products column.

    Keys are resolved through the registry (exact name, then aliases, scoped to `category` or
    the metadata's own), so `origin_country` reaches `country_of_origin`. A list or object
    value stays in metadata: a column holds one.
    """
    category = category or _category_of(metadata)
    order = list(_COLUMN_FOR_FIELD)
    candidates = []
    for raw_key in list(metadata.keys()):
        field = field_registry.resolve_field_key(raw_key, category)
        if field in _COLUMN_FOR_FIELD and field_registry.destination_of(field) == 'column':
            candidates.append((order.index(field), raw_key, field))

    columns: Dict[str, str] = {}
    for _rank, raw_key, field in sorted(candidates):
        column = _COLUMN_FOR_FIELD[field]
        if column in columns:
            continue
        value = metadata.get(raw_key)
        if isinstance(value, dict):
            value = value.get('value')
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        if not isinstance(value, str) or _is_empty_value(value):
            continue
        form = country if (column == 'country_of_origin' and country) else _COLUMN_FORM.get(column, str.strip)
        clean = form(value)
        if clean is None:
            continue  # stays in metadata as printed; the column's CHECK would refuse the whole product
        columns[column] = clean
        metadata.pop(raw_key, None)
    return columns


#: EU and UK conventions that are not ISO 3166-1 codes for the country they name.
_ISO_COUNTRY_EXCEPTIONS = {"EL": "GR", "UK": "GB"}


def _iso_country(value: str) -> Optional[str]:
    code = value.strip().upper()
    code = _ISO_COUNTRY_EXCEPTIONS.get(code, code)
    return code if re.fullmatch(r"[A-Z]{2}", code) else None


#: Columns with a CHECK on their form (`products_country_of_origin_check`: ISO 3166-1 alpha-2).
_COLUMN_FORM = {'country_of_origin': _iso_country}


def country_resolver(client: Any) -> Callable[[str], Optional[str]]:
    """`resolve_country_iso2` (names, native names, ISO-3, EL/UK) with a per-call cache.

    `_iso_country` is only the fallback when the vocabulary cannot be read: it cannot tell
    `EU` from a country.
    """
    cache: Dict[str, Optional[str]] = {}

    def resolve(value: str) -> Optional[str]:
        key = value.strip().lower()
        if key not in cache:
            try:
                found = client.rpc('resolve_country_iso2', {'p_value': value}).execute().data
            except Exception:  # noqa: BLE001
                return _iso_country(value)
            cache[key] = found if isinstance(found, str) and found else None
        return cache[key]

    return resolve


def _confidence_score(raw: Any) -> Optional[float]:
    """Extraction confidence as products.confidence_score (0..1), None when unknown."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value > 1.0 and value <= 100.0:
        value = value / 100.0
    if value < 0.0 or value > 1.0:
        return None
    return round(value, 4)


_CREDIT_KEYS = (
    'designer', 'designers', 'studio', 'design_studio', 'designer_studio', 'designer_name', 'designed_by',
)
_CREDIT_SPLIT = re.compile(r'\s*(?:/|&|,|;|\||\+|\band\b|\sx\s)\s*', re.IGNORECASE)


def _designer_names(metadata: Dict[str, Any]) -> set:
    """Folded names credited as designer or studio, each part of a "Studio / Person" credit included."""
    names = set()
    for key in _CREDIT_KEYS:
        value = metadata.get(key)
        for raw in (value if isinstance(value, list) else [value]):
            if isinstance(raw, str):
                for part in (raw, *_CREDIT_SPLIT.split(raw)):
                    folded = fold_identity(part)
                    if folded:
                        names.add(folded)
    return names


def _maker_is_a_designer_credit(name: Optional[str], metadata: Dict[str, Any]) -> bool:
    """True when `name` is a credited designer: a credit, not the company that makes the product."""
    folded = fold_identity(name)
    return bool(folded) and folded in _designer_names(metadata)


def _resolve_product_maker(
    maker: Optional[str], catalog_maker: Optional[str], metadata: Dict[str, Any], multi_brand: bool = False,
) -> Optional[str]:
    """A single-brand catalog's stated maker is every product's maker; a multi-brand catalog's is its distributor.

    A credited designer is never a maker; a multi-brand product keeps its own brand or has none.
    """
    if catalog_maker and not multi_brand:
        return catalog_maker
    return None if _maker_is_a_designer_credit(maker, metadata) else maker


def _job_is_multi_brand(supabase: Any, document_id: str) -> bool:
    """Whether the newest job for this document was uploaded as a multi-brand (distributor) catalog."""
    try:
        rows = (
            supabase.client.table('background_jobs').select('metadata')
            .eq('document_id', document_id).order('created_at', desc=True).limit(1).execute().data
        )
        return bool(rows and (rows[0].get('metadata') or {}).get('multi_brand'))
    except Exception:
        return False


def _find_workspace_duplicate(
    supabase: Any,
    workspace_id: str,
    document_id: str,
    name: str,
    factory_name: Optional[str],
    sku: Optional[str],
) -> Optional[Dict[str, Any]]:
    """An existing product of this workspace, from ANOTHER source, that is the same product.

    Same SKU, or same folded name AND same folded maker. A name alone is not an identity:
    two makers both sell a "Classic". The same document is left to Stage 0's own idempotency.
    """
    client = supabase.client
    if sku:
        rows = client.table('products') \
            .select('id, name, source_document_id, metadata') \
            .eq('workspace_id', workspace_id).eq('sku', sku).limit(5).execute().data or []
        for row in rows:
            if str(row.get('source_document_id')) != str(document_id):
                return row

    folded_name = fold_identity(name)
    folded_factory = fold_identity(factory_name)
    if not folded_name or not folded_factory:
        return None
    query = client.table('products') \
        .select('id, name, source_document_id, metadata') \
        .eq('workspace_id', workspace_id)
    query = query.ilike('name', escape_like(name.strip()))
    for row in query.limit(50).execute().data or []:
        if str(row.get('source_document_id')) == str(document_id):
            continue
        meta = row.get('metadata') or {}
        if fold_identity(row.get('name')) == folded_name and \
                fold_identity(meta.get('factory_name')) == folded_factory:
            return row
    return None


def _reuse_existing_product(supabase: Any, workspace_id: str, document_id: str,
                            existing: Dict[str, Any], name: str, logger: logging.Logger) -> Dict[str, Any]:
    """Record that `document_id` also carries an existing product, and return it."""
    existing_meta = dict(existing.get('metadata') or {})
    also_in = [d for d in (existing_meta.get('also_in_documents') or []) if d]
    if str(document_id) not in [str(d) for d in also_in] and \
            str(existing.get('source_document_id')) != str(document_id):
        also_in.append(str(document_id))
        existing_meta['also_in_documents'] = also_in
        supabase.client.table('products').update({'metadata': existing_meta}) \
            .eq('id', existing['id']).eq('workspace_id', workspace_id).execute()
    logger.info(
        f"   ♻️ '{name}' already exists in this workspace as {existing['id']} "
        f"(from document {existing.get('source_document_id')}) — reusing, not inserting"
    )
    return {
        'product_id': existing['id'],
        'embedding_failed': False,
        'deduplicated': True,
        'duplicate_of_document': existing.get('source_document_id'),
    }


def _category_of(metadata: Dict[str, Any]) -> Optional[str]:
    raw = (metadata or {}).get('material_category')
    if isinstance(raw, dict):
        raw = raw.get('value')
    return raw if isinstance(raw, str) and raw else None


def _flatten_for_canonicalization(metadata: Dict[str, Any], category: Optional[str] = None) -> Dict[str, Any]:
    """Top-level keys plus registry keys nested one level down, `{value: x}` unwrapped.

    The canonicalizer reads top-level field names only, so a section-nested field
    (`material_properties.finish`) or an alias (`available_colors`) is also placed under its
    registry name. An alias carrying a unit factor (`width_cm`) arrives converted, or not at all.
    """
    category = category or _category_of(metadata)
    flat: Dict[str, Any] = {}
    top: List[tuple] = []
    nested: List[tuple] = []
    for key, value in (metadata or {}).items():
        if not isinstance(key, str) or key.startswith('_'):
            continue
        if isinstance(value, dict) and 'value' in value:
            value = value['value']
        flat[key] = value
        if isinstance(value, dict):
            nested.extend((ck, cv, True) for ck, cv in value.items()
                          if isinstance(ck, str) and not ck.startswith('_'))
        else:
            top.append((key, value, False))
    if not field_registry.is_loaded:
        return flat
    for key, value, is_nested in top + nested:
        hit = field_registry.resolve_field(key, category)
        if hit is None or (hit[0] == key and not is_nested):
            continue
        resolved, factor = hit
        if not _is_empty_value(flat.get(resolved)):
            continue
        if isinstance(value, dict) and 'value' in value:
            value = value['value']
        if factor != 1.0:
            value = scale_value(value, factor)
            if value is None:
                continue
        flat[resolved] = value
    return flat


async def recanonicalize_product(
    supabase: Any,
    product_id: Optional[str],
    metadata: Dict[str, Any],
    workspace_id: Optional[str],
    source: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Re-derive attributes + attributes_raw from the product's final metadata.

    Returns the column payload to write WITH the metadata. Mutates `metadata` only to set
    or clear the `facet_canonicalization` degraded marker. A degraded run keeps the stored
    attributes rather than replacing them with an empty map.
    """
    from datetime import datetime as _dt
    from app.services.facets import CanonicalizedAttributes
    from app.services.facets.facet_canonicalizer import collect_raw_attributes

    await field_registry.ensure_loaded()
    flat = _flatten_for_canonicalization(metadata)
    try:
        canonical = await asyncio.wait_for(
            canonicalize_product_attributes(
                supabase, flat, source=source, product_id=product_id, workspace_id=workspace_id,
            ),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        canonical = CanonicalizedAttributes(
            attributes={}, attributes_raw=collect_raw_attributes(flat),
            resolutions=[], status='degraded',
        )

    if getattr(canonical, 'status', 'ok') != 'ok':
        metadata['facet_canonicalization'] = {
            'status': canonical.status, 'at': _dt.utcnow().isoformat(), 'stage': source,
        }
        logger.error(f"   ❌ Facet re-canonicalization DEGRADED for product {product_id} ({source})")
        return {}
    metadata.pop('facet_canonicalization', None)
    return {'attributes': canonical.attributes, 'attributes_raw': canonical.attributes_raw}


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
# The fine-grained material_category vocabulary lives in
# `material_categories.controlled_vocab` and is read through `field_registry` (#347 phase 3.2).
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


    # Offer EXACTLY the values the validator below accepts — same source, no second copy.
    await field_registry.ensure_loaded()
    _vocab_lines = "\n".join(
        f"  {group}: {' | '.join(values)}"
        for group, values in field_registry.controlled_vocab_by_category()
    )

    prompt = render(await load_prompt("classification", "product_classification", stage="entity_creation"),
        product_name=name, product_description=description or 'N/A',
        existing_category=existing_category or 'N/A', vocab_lines=_vocab_lines)

    # A FORCED tool call, with the vocabulary as the schema (#32).
    classify_tool = {
        "name": "emit_product_classification",
        "description": "Assign the material category and zone intent from the controlled vocabulary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "material_category": {
                    "type": "string",
                    "enum": sorted(field_registry.all_controlled_vocab()),
                },
                "zone_intent": {"type": "string", "enum": sorted(ZONE_INTENT_VOCAB)},
            },
            "required": ["material_category"],
        },
    }

    try:
        from app.services.core.claude_tool_call import ToolCallNotReturned, call_with_tool
        from app.config import get_settings

        call = await call_with_tool(
            task="product_classification",
            model=get_settings().anthropic_model_classification,
            # 64 was enough for a bare JSON object; a tool call needs headroom for the
            # block itself, and truncating it produces the missing-tool_use error rather
            # than a short answer.
            max_tokens=256,
            tool=classify_tool,
            messages=[{"role": "user", "content": prompt}],
            job_id=job_id,
            workspace_id=workspace_id,
            product_id=product_id,
        )
        data = call.data
        result = {}
        if data.get("material_category") in field_registry.all_controlled_vocab():
            result["material_category"] = data["material_category"]
        if data.get("zone_intent") in ZONE_INTENT_VOCAB:
            result["zone_intent"] = data["zone_intent"]
        return result
    except ToolCallNotReturned as e:
        # Distinct from the general failure below: a forced tool_choice that produced no
        # block is a broken API contract and is retryable, which "the model could not
        # classify this" is not.
        logging.getLogger(__name__).error(
            f"Product classification returned no tool call for product {product_id}: {e}"
        )
        return {}
    except Exception as e:
        # A classification failure leaves the product uncategorized (wrong
        # default unit + broken facets), so surface it loudly rather than at
        # warning — this is a wholesale failure (API / schema), not "the model
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
    logger: logging.Logger,
    multi_brand: bool = False,
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
    if catalog.catalog_factory and not multi_brand:
        factory_defaults['factory_name'] = catalog.catalog_factory
    if catalog.catalog_factory_group and not multi_brand:
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
        page_range=getattr(product, 'page_range', None),
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

    # Icon values fill gaps only: they are read off the product's own pages, but the
    # product's own extracted value is the more specific claim.
    if icon_rollup:
        unknown_counts = icon_rollup.pop('_unknown_field_counts', None)
        for spec_field, spec_value in icon_rollup.items():
            if spec_field.startswith('_'):
                continue
            if _is_empty_value(metadata.get(spec_field)):
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

    # The maker mints a CRM company, so a credited studio must never land here.
    _maker = _resolve_product_maker(metadata.get('factory_name'), catalog.catalog_factory, metadata, multi_brand)
    if _maker != metadata.get('factory_name'):
        logger.warning(
            f"   ⚠️ {product.name}: factory_name {metadata.get('factory_name')!r} replaced by {_maker!r}"
        )
        metadata['factory_name'] = _maker

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

    await field_registry.ensure_loaded()
    column_values = _route_column_fields(metadata, country=country_resolver(supabase.client))

    # One product per workspace identity, whichever upload it arrived in. Checked before any
    # paid call below; a lookup that cannot run fails the creation rather than duplicating.
    duplicate = _find_workspace_duplicate(
        supabase, workspace_id, document_id, product.name,
        metadata.get('factory_name'), column_values.get('sku'),
    )
    if duplicate:
        return _reuse_existing_product(supabase, workspace_id, document_id, duplicate, product.name, logger)

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

    await field_registry.ensure_loaded()
    needs_category = not raw_cat or raw_cat not in field_registry.all_controlled_vocab()
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

    upload_category = _upload_category_hint(supabase, document_id)
    if upload_category and metadata.get("material_category") != upload_category:
        logger.info(f"   🏷️ material_category {metadata.get('material_category')!r} -> upload's {upload_category!r}")
        metadata["material_category"] = upload_category
        metadata["_material_category_source"] = "upload"

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
    _category_id: Optional[str] = None
    try:
        if not metadata.get('unit'):
            metadata['unit'] = _resolve_default_unit(metadata.get('material_category'), supabase)
        _category_id = _resolve_category_id(metadata.get('material_category'), supabase)
    except CategoryRegistryUnavailable as _cat_err:
        # No unit and no category_id rather than `pcs` stated as fact.
        logger.error(f"   ⚠️ Category registry unavailable for {product.name}: {_cat_err}")
        metadata['_category_registry'] = {'status': 'unavailable', 'error': str(_cat_err)[:200]}

    # ── Validate extracted values against the field registry ─────────────
    # Runs BEFORE canonicalization on purpose: it rewrites enum values to the admin's curated
    # spelling ("MATT" -> "Matt", "R 11" -> "R11"), and four of the enum-constrained fields
    # (pei_rating, slip_resistance, wood_type, application_areas) are also canonicalizable, so
    # the clusterer should see the curated form.
    try:
        metadata, _validation_report = await validate_metadata_against_registry(
            metadata, category=metadata.get('material_category'),
        )
    except Exception as _val_err:
        # Validation is advisory. It must never be the reason a product fails to ingest —
        # but it also must not fail invisibly, so the product carries the marker.
        logger.error(f"   ⚠️ Registry validation failed for {product.name}: {_val_err}")
        metadata['_validation'] = {'status': 'validator_failed', 'error': str(_val_err)[:200]}

    # ── Auto-canonicalize descriptive facets (multilingual auto-merge) ────
    # Failures degrade with a marker; the product insert is never blocked by this.
    import asyncio as _asyncio
    _flat_metadata = _flatten_for_canonicalization(metadata)
    try:
        canonical = await _asyncio.wait_for(
            canonicalize_product_attributes(supabase, _flat_metadata, source='pdf_stage_4', workspace_id=workspace_id),
            timeout=30.0,
        )
    except _asyncio.TimeoutError:
        logger.error(
            f"   ⏱️ Facet canonicalization exceeded 30s for {product.name} — "
            f"degrading to empty canonical attributes (raw values preserved)."
        )
        # attributes_raw is the replay contract: it lets a later pass rebuild attributes.
        from app.services.facets import CanonicalizedAttributes
        from app.services.facets.facet_canonicalizer import collect_raw_attributes
        canonical = CanonicalizedAttributes(
            attributes={},
            attributes_raw=collect_raw_attributes(_flat_metadata),
            resolutions=[],
            status='degraded',
        )
    # The marker is how the re-canonicalization pass finds rows whose empty attributes
    # mean "a dependency failed", not "no facets".
    if getattr(canonical, 'status', 'ok') != 'ok':
        from datetime import datetime as _dt
        metadata['facet_canonicalization'] = {
            'status': canonical.status,
            'at': _dt.utcnow().isoformat(),
            'stage': 'pdf_stage_4',
        }
        logger.error(
            f"   ❌ Facet canonicalization DEGRADED for {product.name}; "
            f"flagged via metadata.facet_canonicalization"
        )
    if canonical.resolutions:
        logger.info(
            f"   🏷️  Canonicalized {len(canonical.resolutions)} facet values "
            f"across {len(canonical.attributes)} facets"
        )

    # Phase 2: resolve the verbatim maker (metadata.factory_name) to a
    # per-workspace crm_companies node so brand becomes a queryable entity.
    brand_company_id = None
    _factory_name = (metadata.get('factory_name') or '').strip()
    if _factory_name:
        try:
            _bc = supabase.client.rpc('resolve_brand_company', {
                'p_workspace_id': workspace_id,
                'p_name': _factory_name,
            }).execute()
            brand_company_id = _bc.data if (_bc and _bc.data) else None
        except Exception as _brand_err:
            logger.warning(f"   ⚠️ brand resolve failed for {_factory_name!r}: {_brand_err}")

    product_data = {
        'source_document_id': document_id,
        'workspace_id': workspace_id,
        'name': product.name,
        'description': description,
        'metadata': metadata,
        'attributes': canonical.attributes,
        'attributes_raw': canonical.attributes_raw,
        'brand_company_id': brand_company_id,
        'source_type': 'pdf_processing',
        'source_job_id': job_id,
        'status': 'draft',
        'category_id': _category_id,
        'confidence_score': _confidence_score(getattr(product, 'confidence', None)),
        **column_values,
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

    # Identity under the unique index the XML import uses, so two concurrent uploads of one
    # product cannot both insert: a SKU, or a name WITH its maker (an unbranded name is not one).
    from app.services.integrations.data_import_service import import_identity_for
    _sku = column_values.get('sku')
    identity = import_identity_for(
        _factory_name or None, fold_model_token(_sku) if _sku else None, product.name,
        allow_name_identity=bool(_factory_name),
    )
    if identity:
        product_data['import_identity'] = identity
    # Client-minted id + ignore-duplicates makes the insert safe to repeat after a lost reply.
    product_data['id'] = str(uuid.uuid4())
    try:
        result = supabase.client.table('products') \
            .upsert(product_data, on_conflict='id', ignore_duplicates=True).execute()
        if not result.data:
            result = supabase.client.table('products').select('id') \
                .eq('id', product_data['id']).limit(1).execute()
    except Exception as insert_err:
        if not (identity and '23505' in str(insert_err) and 'import_identity' in str(insert_err)):
            raise
        existing = supabase.client.table('products') \
            .select('id, name, source_document_id, metadata') \
            .eq('workspace_id', workspace_id).eq('import_identity', identity).limit(1).execute()
        if not existing.data:
            raise
        return _reuse_existing_product(supabase, workspace_id, document_id, existing.data[0],
                                       product.name, logger)

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
            page_body_text = await build_product_page_body_text(
                supabase,
                document_id,
                metadata.get('page_range'),
                logger,
            )
            embedding_text = build_product_embedding_text(
                name=product.name,
                description=description,
                metadata=metadata,
                known_spec_fields=known_spec_fields,
                page_body_text=page_body_text,
            )

            from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
            embeddings_svc = RealEmbeddingsService()

            text_emb = None
            last_err: Optional[Exception] = None
            for attempt in range(3):
                try:
                    emb_result = await embeddings_svc.generate_text_embedding(
                        embedding_text,
                        job_id=job_id,
                        product_id=product_id,
                    )
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
                    logger.info(f"   🧠 Generated text_embedding_1024 for {product.name} (model={embedding_model or get_settings().voyage_document_model})")
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

        # Images on pages this product does not share with another catalogue product are
        # its own; a shared page is left to the evidence-based linker (B8).
        image_link_status = 'skipped_no_pages'
        try:
            page_range = getattr(product, 'page_range', None) or []
            other_pages: set = set()
            for other in (getattr(catalog, 'products', None) or []):
                if other is product:
                    continue
                for p in (getattr(other, 'page_range', None) or []):
                    other_pages.add(p)
            own_pages = [p for p in page_range if p not in other_pages]
            if own_pages:
                img_resp = supabase.client.table('document_images')                    .select('id, page_number')                    .eq('document_id', document_id)                    .in_('page_number', own_pages).execute()
                image_ids = [r['id'] for r in (img_resp.data or [])]
                image_link_status = 'linked' if image_ids else 'no_images_yet'
                if image_ids:
                    rows = [
                        {'workspace_id': workspace_id, 'product_id': product_id, 'image_id': iid,
                         'metadata': {'association_method': 'stage_4_immediate_by_page'}}
                        for iid in image_ids
                    ]
                    supabase.client.table('image_product_associations').upsert(rows, on_conflict='image_id,product_id').execute()
                    logger.info(
                        f"   🔗 Linked {len(image_ids)} images to product {product_id} "
                        f"(immediate, own pages={own_pages})"
                    )
            elif page_range:
                image_link_status = 'skipped_shared_pages'
        except Exception as assoc_err:
            image_link_status = 'failed'
            logger.error(
                f"   ❌ Immediate image-product association failed for {product_id} "
                f"(the per-product linker retries): {assoc_err}"
            )

        return {
            'product_id': product_id,
            'embedding_failed': embedding_failed,
            'image_link_status': image_link_status,
        }
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


PRODUCT_BODY_TEXT_MAX_CHARS = 4000


def build_product_embedding_text(
    name: Optional[str],
    description: Optional[str],
    metadata: Dict[str, Any],
    known_spec_fields: List[str],
    page_body_text: Optional[str] = None,
) -> str:
    """Build the canonical product embedding text from name + description +
    searchable metadata + spec fields + (optionally) the product's reading-order
    page body text.
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
    # Discovery writes `color`; the vision rollup writes `appearance.colors_from_vision`.
    _color = metadata.get('color')
    if isinstance(_color, dict):
        _color = _color.get('value')
    _color_sources = [
        [_color] if isinstance(_color, str) else _color,
        metadata.get('appearance_colors'),
        (metadata.get('appearance') or {}).get('colors_from_vision') if isinstance(metadata.get('appearance'), dict) else None,
    ]
    _seen_colors: set = set()
    for _cs in _color_sources:
        if isinstance(_cs, list):
            for _c in _cs:
                _cv = str(_c).strip()
                if _cv and _cv.lower() not in _seen_colors:
                    _seen_colors.add(_cv.lower())
                    embedding_text_parts.append(_cv)

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

    # Append the product's reading-order page body text LAST (bounded upstream).
    if page_body_text and page_body_text.strip():
        embedding_text_parts.append(page_body_text.strip())

    return ' | '.join(embedding_text_parts)


async def build_product_page_body_text(
    supabase: Any,
    document_id: Optional[str],
    page_range: Optional[List[int]],
    logger: logging.Logger,
    max_chars: int = PRODUCT_BODY_TEXT_MAX_CHARS,
) -> str:
    """Reading-order page body text for a product, from the PaddleOCR-VL cache."""
    if not document_id or not page_range:
        return ""
    from app.api.pdf_processing.stage_1_layout_precompute import (
        load_page_texts_from_cache,
    )
    # Canonical cache reader (sync) off the event loop. Keyed by 1-based physical page.
    by_page = await asyncio.to_thread(
        load_page_texts_from_cache, supabase, document_id, page_range, logger,
    )
    if not by_page:
        return ""
    joined = "\n\n".join(
        by_page[p] for p in sorted(by_page) if by_page.get(p, "").strip()
    ).strip()
    if max_chars and len(joined) > max_chars:
        joined = joined[:max_chars].rstrip()
    return joined


async def reembed_product_text(
    supabase: Any,
    document_id: Optional[str],
    product_id: str,
    name: Optional[str],
    description: Optional[str],
    metadata: Dict[str, Any],
    known_spec_fields: List[str],
    logger: logging.Logger,
) -> bool:
    """Rebuild + persist a product's text_embedding_1024 from its FINAL fields."""
    try:
        page_body_text = await build_product_page_body_text(
            supabase, document_id, (metadata or {}).get("page_range"), logger,
        )
        embedding_text = build_product_embedding_text(
            name=name,
            description=description,
            metadata=metadata or {},
            known_spec_fields=known_spec_fields,
            page_body_text=page_body_text,
        )
        if not embedding_text.strip():
            return False
        from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
        emb_result = await RealEmbeddingsService().generate_text_embedding(
            embedding_text, product_id=product_id,
        )
        text_emb = emb_result.get("embedding") if emb_result.get("success") else None
        if not text_emb or len(text_emb) != 1024:
            logger.debug(
                f"   re-embed skipped for {product_id}: "
                f"{emb_result.get('error') or f'dim={len(text_emb) if text_emb else 0}'}"
            )
            return False
        payload: Dict[str, Any] = {
            "text_embedding_1024": "[" + ",".join(str(x) for x in text_emb) + "]",
            "text_embedding_schema_version": 1,
        }
        if emb_result.get("model"):
            payload["text_embedding_1024_model"] = emb_result["model"]
        await asyncio.to_thread(
            supabase.client.table("products").update(payload).eq("id", product_id).execute
        )
        return True
    except Exception as e:
        logger.debug(f"   re-embed after enrichment failed for {product_id} (non-fatal): {e}")
        return False


async def _fetch_known_spec_fields(
    supabase: Any,
    logger: logging.Logger,
) -> List[str]:
    """Fetch the canonical spec field name list from `material_metadata_fields`."""
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
    page_range: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Roll the icon_metadata of THIS product's pages up into flat `{field: value}`.

    Only images on the product's own pages are read: a catalogue-wide read let one
    product's R-rating or certification stamp every product in the document.
    """
    if not known_spec_fields:
        return {}
    pages = sorted({
        int(p) for p in (page_range or [])
        if isinstance(p, (int, float)) and not isinstance(p, bool)
    })
    if not pages:
        return {}

    try:
        images_response = supabase.client.table('document_images')             .select('id, page_number, metadata')             .eq('document_id', document_id)             .in_('page_number', pages)             .execute()
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


#: Facts about the MAKER, true of every product a catalogue describes.
_CATALOGUE_LEVEL_FIELDS = (
    'factory_name', 'factory_group_name', 'manufacturing_country',
    'manufacturing_location', 'origin',
)
#: Shared by products only when at least this many of them state the same value.
_COLLECTION_MIN_AGREEING = 2


def _unanimous_value(products: List[Dict[str, Any]], field: str) -> tuple:
    """(value, how many products state it) when every stated value agrees, else (None, 0)."""
    stated = []
    for product in products:
        value = (product.get('metadata') or {}).get(field)
        if isinstance(value, dict) and 'value' in value:
            value = value['value']
        if not _is_empty_value(value) and isinstance(value, (str, int, float)):
            stated.append(value)
    if not stated:
        return None, 0
    if len({fold_identity(str(v)) for v in stated}) != 1:
        return None, 0
    return stated[0], len(stated)


async def propagate_common_fields_to_products(
    document_id: str,
    supabase: Any,
    logger: logging.Logger,
    material_category_override: Optional[str] = None
) -> Dict[str, Any]:
    """Share VERIFIED catalogue-level facts across the products of one document.

    A maker fact is shared only when every product that states it agrees; a collection
    only when several agree. Sizes, thickness, body and composition are per-product facts
    and are never copied from a sibling.
    """
    logger.info(f"🔄 Propagating common fields across products from document {document_id}...")

    stats = {
        'products_checked': 0,
        'products_updated': 0,
        'fields_propagated': [],
        'fields_withheld_disagreement': [],
    }

    try:
        if not material_category_override:
            job_result = supabase.client.table('background_jobs').select('metadata').eq('document_id', document_id).order('created_at', desc=True).limit(1).execute()
            if job_result.data and len(job_result.data) > 0:
                job_metadata = job_result.data[0].get('metadata', {})
                material_category_override = job_metadata.get('material_category')

            if not material_category_override:
                doc_result = supabase.client.table('documents').select('metadata').eq('id', document_id).execute()
                if doc_result.data and len(doc_result.data) > 0:
                    doc_metadata = doc_result.data[0].get('metadata', {})
                    material_category_override = doc_metadata.get('material_category')

        if material_category_override:
            logger.info(f"   📦 Using material_category from upload: {material_category_override}")

        products_response = supabase.client.table('products') \
            .select('id, name, metadata, workspace_id') \
            .eq('source_document_id', document_id) \
            .execute()

        if not products_response.data:
            logger.info("   ℹ️ No products found for document")
            return stats

        products = products_response.data
        stats['products_checked'] = len(products)

        common_values: Dict[str, Any] = {}
        if material_category_override and not _is_empty_value(material_category_override):
            # The operator's own upload choice, not an inference from a sibling.
            common_values['material_category'] = material_category_override

        for field in (() if _job_is_multi_brand(supabase, document_id) else _CATALOGUE_LEVEL_FIELDS):
            value, _count = _unanimous_value(products, field)
            if value is not None:
                common_values[field] = value
            elif any(not _is_empty_value((p.get('metadata') or {}).get(field)) for p in products):
                stats['fields_withheld_disagreement'].append(field)

        collection, agreeing = _unanimous_value(products, 'collection')
        if collection is not None and agreeing >= _COLLECTION_MIN_AGREEING:
            common_values['collection'] = collection

        # The most complete factory object, shared only with products of the SAME maker.
        best_factory: Dict[str, Any] = {}
        best_factory_score = 0
        _completeness_fields = ['factory_name', 'city', 'country', 'address',
                                'phone', 'email', 'website', 'country_of_origin', 'employee_count']
        for product in products:
            fobj = (product.get('metadata') or {}).get('factory') or {}
            if not isinstance(fobj, dict):
                continue
            score = sum(1 for f in _completeness_fields if fobj.get(f) and not _is_empty_value(fobj[f]))
            if score > best_factory_score:
                best_factory_score = score
                best_factory = fobj
        best_maker = fold_identity(best_factory.get('factory_name')) if best_factory else ''

        if not common_values and not best_factory:
            logger.info("   ℹ️ No verified common values to propagate")
            return stats

        logger.info(f"   📦 Verified common values: {list(common_values.keys())}")

        for product in products:
            product_id = product['id']
            metadata = product.get('metadata', {}) or {}
            updates_needed: Dict[str, Any] = {}

            for field, common_value in common_values.items():
                if _is_empty_value(metadata.get(field)):
                    updates_needed[field] = common_value

            if best_factory:
                existing_fobj = metadata.get('factory') or {}
                if not isinstance(existing_fobj, dict):
                    existing_fobj = {}
                own_maker = fold_identity(
                    existing_fobj.get('factory_name') or metadata.get('factory_name')
                    or updates_needed.get('factory_name')
                )
                existing_score = sum(
                    1 for f in _completeness_fields
                    if existing_fobj.get(f) and not _is_empty_value(existing_fobj[f])
                )
                if best_maker and own_maker == best_maker and existing_score < best_factory_score:
                    merged_factory = {**best_factory, **{
                        k: v for k, v in existing_fobj.items()
                        if v and not _is_empty_value(v)
                    }}
                    updates_needed['factory'] = merged_factory
                    if merged_factory.get('factory_group_name') and _is_empty_value(metadata.get('factory_group_name')):
                        updates_needed['factory_group_name'] = merged_factory['factory_group_name']

            if updates_needed:
                updated_metadata = {**metadata, **updates_needed}
                columns = await recanonicalize_product(
                    supabase, product_id, updated_metadata, product.get('workspace_id'),
                    'pdf_stage_4_5_propagation', logger,
                )
                supabase.client.table('products') \
                    .update({'metadata': updated_metadata, **columns}) \
                    .eq('id', product_id) \
                    .execute()

                stats['products_updated'] += 1
                propagated = list(updates_needed.keys())
                stats['fields_propagated'].extend(propagated)
                logger.info(f"   ✅ Updated product {product['name']}: {propagated}")

        stats['fields_propagated'] = sorted(set(stats['fields_propagated']))

        logger.info(f"✅ Propagation complete: {stats['products_updated']}/{stats['products_checked']} products updated")
        return stats

    except Exception as e:
        logger.error(f"❌ Failed to propagate common fields: {e}")
        stats['error'] = str(e)
        return stats


_SIZE_RE = None
_THICKNESS_RE = None


def _dimension_patterns():
    global _SIZE_RE, _THICKNESS_RE
    if _SIZE_RE is None:
        import re
        num = r'(\d{1,4}(?:[.,]\d{1,2})?)'
        _SIZE_RE = re.compile(
            num + r'\s*[×xX]\s*' + num + r'(?:\s*[×xX]\s*' + num + r')?\s*(mm|cm|in|")(?![a-zA-Z])',
            re.IGNORECASE,
        )
        _THICKNESS_RE = re.compile(
            r'(?:thickness|spessore|epaisseur|st[aä]rke|dikte|πάχος)[^\n]{0,40}?'
            r'(\d{1,3}(?:[.,]\d{1,2})?)\s*(mm|cm)(?![a-zA-Z])',
            re.IGNORECASE,
        )
    return _SIZE_RE, _THICKNESS_RE


def _scan_dimensions(text: str) -> Dict[str, Any]:
    """Sizes and thickness stated in ONE product's own text, in the unit the text states.

    A size without a unit is not read: "60x60" alone cannot be told from mm or cm.
    """
    size_re, thick_re = _dimension_patterns()
    sizes: List[str] = []
    for m in size_re.finditer(text or ''):
        unit = m.group(4).lower().replace('"', 'in')
        parts = [g.replace(',', '.') for g in m.groups()[:3] if g]
        sizes.append('×'.join(parts) + f' {unit}')
    found: Dict[str, Any] = {}
    if sizes:
        found['available_sizes'] = list(dict.fromkeys(sizes))
    m = thick_re.search(text or '')
    if m:
        value = float(m.group(1).replace(',', '.'))
        found['thickness_mm'] = round(value * 10, 2) if m.group(2).lower() == 'cm' else value
    return found


async def extract_dimensions_from_document_chunks(
    document_id: str,
    supabase: Any,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Stage 4.6: fill a product's missing sizes/thickness from ITS OWN chunks.

    Each product is scanned against the chunks linked to it, never the whole document,
    so one product cannot receive the catalogue's size list. No AI calls.
    """
    stats: Dict[str, Any] = {
        'products_checked': 0,
        'products_updated': 0,
        'products_without_own_chunks': 0,
        'dimensions_found': [],
    }

    try:
        chunks_resp = supabase.client.table('document_chunks') \
            .select('content, product_id') \
            .eq('document_id', document_id) \
            .execute()
        text_by_product: Dict[str, List[str]] = {}
        for chunk in chunks_resp.data or []:
            if chunk.get('product_id') and chunk.get('content'):
                text_by_product.setdefault(chunk['product_id'], []).append(chunk['content'])

        products_resp = supabase.client.table('products') \
            .select('id, name, metadata, workspace_id') \
            .eq('source_document_id', document_id) \
            .execute()
        products = products_resp.data or []
        stats['products_checked'] = len(products)

        for product in products:
            product_id = product['id']
            own_text = text_by_product.get(product_id)
            if not own_text:
                stats['products_without_own_chunks'] += 1
                continue
            found = _scan_dimensions(' '.join(own_text))
            if not found:
                continue

            existing_metadata = product.get('metadata') or {}
            updated_metadata = {**existing_metadata}
            changed = []
            if found.get('available_sizes') and _is_empty_value(existing_metadata.get('available_sizes')):
                updated_metadata['available_sizes'] = found['available_sizes']
                changed.append('available_sizes')
            legacy_thickness = (existing_metadata.get('material_properties') or {}).get('thickness') \
                if isinstance(existing_metadata.get('material_properties'), dict) else None
            if 'thickness_mm' in found and _is_empty_value(existing_metadata.get('thickness_mm')) \
                    and _is_empty_value(existing_metadata.get('thickness')) and _is_empty_value(legacy_thickness):
                updated_metadata['thickness_mm'] = found['thickness_mm']
                changed.append('thickness_mm')
            if not changed:
                continue

            extraction_meta = dict(updated_metadata.get('_extraction_metadata') or {})
            for key in changed:
                extraction_meta[key] = {'source': 'product_chunk_regex', 'confidence': 0.65}
            updated_metadata['_extraction_metadata'] = extraction_meta
            columns = await recanonicalize_product(
                supabase, product_id, updated_metadata, product.get('workspace_id'),
                'pdf_stage_4_6_dimensions', logger,
            )
            supabase.client.table('products') \
                .update({'metadata': updated_metadata, **columns}) \
                .eq('id', product_id) \
                .execute()
            stats['products_updated'] += 1
            stats['dimensions_found'].extend(found.get('available_sizes') or [])
            logger.info(f"   ✅ Filled {changed} for product '{product['name']}' from its own chunks")

        stats['dimensions_found'] = list(dict.fromkeys(stats['dimensions_found']))
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

import re as _re

def _normalize_material_category(raw: str) -> Optional[str]:
    """A vision `material_type` as a vocabulary value, else the registry's category, else None.

    "porcelain tile" -> porcelain_tile; "marble" -> general_materials (a vocab alias). No
    substring guessing: an unresolved value is left unset, and this only fills a gap.
    """
    if not raw or not isinstance(raw, str) or not field_registry.is_loaded:
        return None
    slug = _re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    if slug in field_registry.all_controlled_vocab():
        return slug
    return field_registry.resolve_category(slug)


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
    grout_brands = set()
    grout_entries = []
    for m in grout_pattern.finditer(text):
        dose = int(m.group(1))
        supplier = m.group(2).capitalize()
        grout_brands.add(supplier.upper())
        grout_entries.append({"supplier": supplier, "dose": dose})
    # ONE `grout_color_codes` map: {suppliers, by_code, <supplier>: product name}.
    grout: Dict[str, Any] = {}
    if grout_brands:
        grout["suppliers"] = sorted(grout_brands)
    if grout_entries and sku_codes:
        # Best-effort: SKU and grout dose usually share a line in catalog tables.
        codes_in_order = list(sku_codes.values())
        if len(codes_in_order) == len(grout_entries):
            grout["by_code"] = {
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
    for m in grout_product_pattern.finditer(text):
        supplier = m.group(1).upper()
        product = m.group(2).strip()
        # Clean trailing whitespace/newlines from the captured product name
        product = _re.sub(r"\s+", " ", product).strip()
        # Stop at common end-of-line markers
        product = _re.split(r"\*+|\|", product, maxsplit=1)[0].strip()
        if 3 <= len(product) <= 40:
            grout[supplier.lower()] = product
    if grout:
        candidates["grout_color_codes"] = grout

    # ── MATT/GLOSS finish indicators appearing in packing tables ─────────
    # "MATT" and "GLOSS" on the spec page are options, not confirmed values for
    # this product unless they appear with a "✓" or bold marker — but their
    # presence on the page at least hints at what finishes exist.
    {
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

    Args:
        vision_rows: List of dicts, each containing a vision_analysis JSON blob.

    Returns:
        Candidate fields including: material_category, finish, design_style,
        pattern (+ patterns list), texture (+ textures list), appearance_colors,
        application, category, subcategory, description, detected_text,
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
        candidates["application"] = _dedupe_pretty(all_applications, cap=10)

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


def _registry_key(key: str, category: Optional[str] = None) -> Optional[str]:
    """The registry field `key` names (exact or unconverted alias), where the facet reader looks."""
    return field_registry.resolve_field_key(key, category) if field_registry.is_loaded else None


def _merge_enriched_fields_into_metadata(
    existing_metadata: Dict[str, Any],
    chunk_candidates: Dict[str, Any],
    vision_candidates: Dict[str, Any],
    allow_maker_fill: bool = True,
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
        resolved = _registry_key(key, _category_of(new_metadata)) if container else None
        if resolved:
            key, container = resolved, None
        if container:
            current = new_metadata.get(container)
            if not isinstance(current, dict) and not _is_empty_value(current):
                skipped = extraction_meta.setdefault(f"{container}._not_nested", [])
                if key not in skipped:
                    skipped.append(key)
                return
            parent = dict(current or {})
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
    _chunk_maker = chunk_candidates.get("factory_name") if allow_maker_fill else None
    if _chunk_maker and _maker_is_a_designer_credit(
        _chunk_maker, {**new_metadata, "designers": chunk_candidates.get("designers")}
    ):
        _chunk_maker = None
    fill_if_empty("factory_name", _chunk_maker, "chunk_regex")
    fill_if_empty("designers", chunk_candidates.get("designers"), "chunk_regex")
    fill_if_empty("collection", chunk_candidates.get("collection"), "chunk_regex")
    fill_if_empty("inspiration", chunk_candidates.get("inspiration"), "chunk_regex", container="design")
    fill_if_empty("pieces_per_box", chunk_candidates.get("pieces_per_box"), "chunk_regex", container="packaging")
    fill_if_empty("patterns_count", chunk_candidates.get("patterns_count"), "chunk_regex", container="packaging")
    fill_if_empty("body_type", chunk_candidates.get("body_type"), "chunk_regex", container="material_properties")
    fill_if_empty("sku_codes", chunk_candidates.get("sku_codes"), "chunk_regex", container="commercial")
    fill_if_empty("grout_color_codes", chunk_candidates.get("grout_color_codes"), "chunk_regex", container="commercial")

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
    fill_if_empty("application", vision_candidates.get("application"), "vision_rollup")
    # New schema-aligned propagation (post-2026-05-04 — was orphaned in the
    # legacy rollup that read fields the schema doesn't emit).
    fill_if_empty("category", vision_candidates.get("category"), "vision_rollup", container="appearance")
    fill_if_empty("subcategory", vision_candidates.get("subcategory"), "vision_rollup", container="appearance")
    fill_if_empty("vision_description", vision_candidates.get("vision_description"), "vision_rollup", container="appearance")
    fill_if_empty("detected_text", vision_candidates.get("detected_text"), "vision_rollup", container="appearance")
    fill_if_empty("vision_confidence", vision_candidates.get("vision_confidence"), "vision_rollup")

    vision_colors = vision_candidates.get("appearance_colors")
    current_appearance = new_metadata.get("appearance")
    if vision_colors and not isinstance(current_appearance, dict) and not _is_empty_value(current_appearance):
        skipped = extraction_meta.setdefault("appearance._not_nested", [])
        if "colors_from_vision" not in skipped:
            skipped.append("colors_from_vision")
    elif vision_colors:
        appearance = dict(current_appearance or {})
        # Store under appearance.colors_from_vision so it doesn't clobber the
        # text-extracted colors_from_chunks if both exist.
        if _is_empty_value(appearance.get("colors_from_vision")):
            appearance["colors_from_vision"] = vision_colors
            new_metadata["appearance"] = appearance
            extraction_meta["appearance.colors_from_vision"] = {"source": "vision_rollup", "confidence": 0.85}
            filled.append("appearance.colors_from_vision")

    if filled or any(k.endswith("._not_nested") for k in extraction_meta):
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
    # write_product_description_from_chunks is sync and reads the cache.
    await prefetch(("generation", "product_description"))
    await field_registry.ensure_loaded()
    multi_brand = _job_is_multi_brand(supabase, document_id)
    stats: Dict[str, Any] = {
        "products_checked": 0,
        "products_updated": 0,
        "fields_filled": set(),
        "spec_vision_calls": 0,
        # M3-15 (#16): failures used to increment spec_vision_calls alongside
        # successes, so a tier that threw on every product looked like a tier
        # that ran fine and found nothing. Counted separately.
        "spec_vision_failures": 0,
        "description_writes": 0,
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
        # product.
        try:
            # S4-5: scope to THIS document at the DB (filter on the inner-joined
            # document_images.document_id) instead of pulling every association row
            # in the table and filtering in Python — that was an unbounded scan per
            # enrichment run on a large multi-tenant DB.
            assoc_resp = supabase.client.table("image_product_associations") \
                .select("product_id, document_images!inner(page_number, document_id)") \
                .eq("document_images.document_id", document_id) \
                .execute()
            product_to_image_pages: Dict[str, set] = {}
            for row in (assoc_resp.data or []):
                di = row.get("document_images") or {}
                if di.get("document_id") != document_id:  # defensive; join already filters
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
        # Fetched once for the whole document — the embedding-text builder used by
        # the S4-2 re-embed below walks these fields (same as inline Stage 4).
        _known_spec_fields = await _fetch_known_spec_fields(supabase, logger)
        for product in products:
            product_id = product["id"]
            product_name = product.get("name") or "(unnamed)"
            metadata = product.get("metadata") or {}

            # ── Scope chunks to this product FIRST ───────────────────────
            # We resolve chunks before pages because chunks are the best
            # source of page_range when the product row doesn't have one.
            product_chunks = [c for c in all_chunks if c.get("product_id") == product_id]
            if not product_chunks:
                # Chunks on the product's own pages, never the whole document: a doc-wide
                # fallback gave every unlinked product the catalogue's text.
                own_pages = {
                    int(p) for p in (metadata.get("page_range") or [])
                    if isinstance(p, (int, float)) and not isinstance(p, bool)
                }
                product_chunks = [
                    c for c in all_chunks
                    if isinstance(c.get("metadata"), dict)
                    and c["metadata"].get("page_number") in own_pages
                ] if own_pages else []

            # ── Resolve the product's PDF page range ─────────────────────
            # We UNION four signals rather than short-circuiting on the first
            # one that produces output — earlier short-circuit logic silently
            # fell through to a noisy text-marker scan whenever the chunk
            # metadata fetch returned metadata as a string (Supabase JSONB
            # deserialization is not always dict-typed).
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
            product_vision = [
                v for v in all_vision if v.get("page_number") in page_set
            ] if page_set else []

            chunk_candidates = _extract_fields_from_chunk_text(combined_text)
            vision_candidates = _rollup_vision_analysis(product_vision)

            # ── Stage 4.7.c: Product Spec Extractor v2 (3-tier hybrid) ─────
            # Tier A: PyMuPDF text-dict parser (free, deterministic)
            # Tier B: Claude Opus Vision (fallback when Tier A insufficient)
            # Tier C: Catalog legend inheritance (fills fields from Layer 2
            # global legends, e.g. certifications, PEI defaults)
            spec_vision_candidates: Dict[str, Any] = {}
            resolved_page_indices: List[int] = []
            layout_pages = layout_product_pages.get(product_name) or []
            if layout_pages:
                # Layer 1 gives us 0-indexed PDF pages directly
                resolved_page_indices = sorted({int(p) for p in layout_pages})
            elif page_set:
                # Fallback to the chunk-derived page_set (1-indexed → 0-indexed)
                resolved_page_indices = sorted({int(p) - 1 for p in page_set if int(p) > 0})

            spec_status: Optional[Dict[str, Any]] = None
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
                        category=metadata.get("material_category"),
                    )
                    if spec_vision_candidates:
                        tiers = spec_vision_candidates.pop("_source_tiers", [])
                        # The extractor reports its own failure in-band rather
                        # than raising, so a truthy dict is not automatically a
                        # successful extraction.
                        extract_error = spec_vision_candidates.get("_error")
                        tier_error = spec_vision_candidates.get("_tier_error")
                        degraded_tiers = spec_vision_candidates.get("_degraded_tiers") or []
                        usable = any(
                            not str(k).startswith("_") for k in spec_vision_candidates
                        )
                        if extract_error or (tier_error and not usable):
                            stats["spec_vision_failures"] += 1
                            logger.error(
                                f"   ❌ spec_v2 '{product_name}' FAILED: "
                                f"{extract_error or tier_error}, tiers={tiers}"
                            )
                            spec_status = {
                                "status": "failed",
                                "error": str(extract_error or tier_error)[:300],
                                "tiers": tiers,
                            }
                            spec_vision_candidates = {}
                        elif tier_error or degraded_tiers:
                            stats.setdefault("spec_vision_degraded", 0)
                            stats["spec_vision_degraded"] += 1
                            logger.warning(
                                f"   ⚠️ spec_v2 '{product_name}' DEGRADED: "
                                f"tier_error={tier_error}, degraded={degraded_tiers}, tiers={tiers}"
                            )
                            spec_status = {
                                "status": "degraded",
                                "tier_error": str(tier_error)[:300] if tier_error else None,
                                "degraded_tiers": degraded_tiers,
                                "tiers": tiers,
                            }
                        else:
                            stats["spec_vision_calls"] += 1
                            logger.info(
                                f"   ✅ spec_v2 '{product_name}': tiers={tiers}, "
                                f"pages={resolved_page_indices}"
                            )
                except Exception as e:
                    stats["spec_vision_failures"] += 1
                    spec_status = {"status": "failed", "error": f"{type(e).__name__}: {e}"[:300]}
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
                allow_maker_fill=not multi_brand,
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
                        current = new_metadata.get(key)
                        # A scalar already holds this key (e.g. `packaging` as text): it is kept,
                        # and children with no registry field of their own are named, not nested.
                        scalar_owner = not isinstance(current, dict) and not _is_empty_value(current)
                        container = dict(current) if isinstance(current, dict) else {}
                        any_added = False
                        for child_key, child_value in value.items():
                            top_key = _registry_key(child_key)
                            if top_key:
                                if _is_empty_value(new_metadata.get(top_key)) and not _is_empty_value(child_value):
                                    new_metadata[top_key] = child_value
                                    extraction_meta[top_key] = {
                                        "source": "claude_spec_vision", "confidence": 0.85
                                    }
                                    filled.append(top_key)
                                continue
                            if scalar_owner:
                                if not _is_empty_value(child_value):
                                    skipped = extraction_meta.setdefault(f"{key}._not_nested", [])
                                    if child_key not in skipped:
                                        skipped.append(child_key)
                                continue
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
                if filled or any(k.endswith("._not_nested") for k in extraction_meta):
                    new_metadata["_extraction_metadata"] = extraction_meta

            markers_changed = False
            if spec_status is not None:
                new_metadata["_spec_extraction"] = spec_status
                markers_changed = True
            elif "_spec_extraction" in new_metadata and enable_spec_vision and resolved_page_indices and pdf_path:
                new_metadata.pop("_spec_extraction")
                markers_changed = True

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
                    if prop_result.status == "failed":
                        stats.setdefault("functional_property_failures", 0)
                        stats["functional_property_failures"] += 1
                        new_metadata["_functional_extraction"] = {
                            "status": "failed", "error": prop_result.error,
                        }
                        markers_changed = True
                        logger.error(
                            f"   ❌ Functional property extraction FAILED for "
                            f"'{product_name}': {prop_result.error}"
                        )
                    elif prop_result.coverage_pct > 0:
                        new_metadata.pop("_functional_extraction", None)
                        new_metadata["functional_metadata"] = prop_result.properties
                        new_metadata.setdefault("_extraction_metadata", {})["functional_metadata"] = {
                            "source": prop_result.method,
                            "confidence": prop_result.confidence,
                            "category_confidence": prop_result.category_confidence,
                            "coverage_pct": prop_result.coverage_pct,
                            "processing_time_ms": prop_result.processing_time_ms,
                        }
                        filled.append("functional_metadata")
                        stats.setdefault("functional_property_extractions", 0)
                        stats["functional_property_extractions"] += 1
                except Exception as e:
                    new_metadata["_functional_extraction"] = {
                        "status": "failed", "error": f"{type(e).__name__}: {e}"[:300],
                    }
                    markers_changed = True
                    logger.warning(
                        f"   ⚠️ Functional property extraction failed for '{product_name}': {e}"
                    )

            if not filled and not markers_changed:
                logger.info(
                    f"   ℹ️ Enrichment: '{product_name}' already has all fields — "
                    f"no empty slots to fill"
                )
                continue

            # Re-fetch product row to get id (in case supabase select changed)
            update_payload: Dict[str, Any] = {"metadata": new_metadata}
            if new_description:
                update_payload["description"] = new_description
            update_payload.update(await recanonicalize_product(
                supabase, product_id, new_metadata, product.get("workspace_id"),
                "pdf_stage_4_7_enrichment", logger,
            ))

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
                # Its metadata is in-memory but not persisted: skip the re-embed below.
                continue

            stats["products_updated"] += 1
            stats["fields_filled"].update(filled)
            logger.info(
                f"   ✅ Enriched '{product_name}': filled {filled} "
                f"(chunk_candidates={list(chunk_candidates.keys())}, "
                f"vision_candidates={list(vision_candidates.keys())})"
            )

            # S4-2: when enrichment writes a NEW description, the inline Stage 4
            # text_embedding_1024 (built before the description existed) is stale
            # w.r.t. the persisted row. Re-embed from the FINAL fields via the
            # shared helper so the stored vector matches the row and doesn't drift
            # from a later backfill. Best-effort — a failed re-embed leaves the
            # (stale-but-present) vector rather than nulling it.
            if new_description:
                reembedded = await reembed_product_text(
                    supabase, document_id, product_id, product_name,
                    new_description, new_metadata, _known_spec_fields, logger,
                )
                if reembedded:
                    stats.setdefault("products_reembedded", 0)
                    stats["products_reembedded"] += 1

        # ── Stage 4.7.f: Catalog-wide knowledge extraction (ONCE per document) ──
        # Runs after all products have been enriched. Uses Claude Vision on the
        # last ~10 pages of the PDF to extract iconography legends, regulations,
        # installation guides, care instructions, sustainability claims, etc.
        # Creates catalog-wide kb_docs and links them to every product.
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
