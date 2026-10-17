"""Metadata Normalization Service"""

import logging
import re
import unicodedata
from typing import Dict, Any, Optional, Tuple


logger = logging.getLogger(__name__)


# `flatten_extracted_metadata` lives in `metadata_shape.py` — a leaf module with no app
# imports, so the guard test can load it directly without dragging in the DB client.
from app.services.metadata.metadata_shape import flatten_extracted_metadata  # noqa: E402,F401
from app.services.metadata.field_registry import field_registry  # noqa: E402


# ============================================================================
# FIELD NAME NORMALIZATION
# ============================================================================


def resolve_standard_field(field_name: str, product_category: Optional[str] = None) -> Optional[Tuple[str, float]]:
    """(registry field, factor) to rename an extracted key to, or None to keep it.

    A key the registry knows for this category is left alone. Matching is deterministic
    (exact name, then `material_metadata_fields.aliases`, over case/accent/unit/plural
    spellings), never similarity. `factor` != 1 means the value must be converted.
    """
    if not field_name or not field_registry.is_loaded:
        return None
    raw = str(field_name)
    spec = field_registry.spec_for(raw)
    if spec is not None:
        category = field_registry.resolve_category(product_category) if product_category else None
        if category is None or spec.applies_to(category):
            return None
    hit = field_registry.resolve_field(raw, product_category)
    return hit if hit and hit[0] != raw else None


def find_standard_field(field_name: str, category: str = None, threshold: float = None,
                        *, product_category: Optional[str] = None) -> Optional[str]:
    """The registry field name an extracted key renames to when no unit conversion is needed.

    Args:
        field_name: the key as extracted.
        category: the metadata section; accepted for call compatibility.
        threshold: accepted and ignored — fuzzy matching is what merged distinct fields.
        product_category: the product's category, which decides a category-scoped alias.

    Returns:
        The registry field name to rename to, or None to keep the key unchanged.
    """
    hit = resolve_standard_field(field_name, product_category)
    return hit[0] if hit and hit[1] == 1.0 else None


_NUMERIC_TEXT_RE = re.compile(r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*[a-zA-Z%°²³/.]*\s*$")


def scale_value(value: Any, factor: float) -> Any:
    """`value` x `factor`, or None when it is not a number (the caller keeps the raw key)."""
    if isinstance(value, dict) and "value" in value:
        inner = scale_value(value.get("value"), factor)
        return None if inner is None else {**value, "value": inner}
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = _NUMERIC_TEXT_RE.match(str(value or ""))
        if not match:
            return None
        number = float(match.group(1).replace(",", "."))
    result = round(number * factor, 4)
    return int(result) if result.is_integer() else result


_GROUT_BRAND_KEY_RE = re.compile(r"^grout_([a-z]+)$")


def _grout_entry(value: Any) -> Any:
    if isinstance(value, dict):
        if "product" in value and "value" in value:
            return f"{value['value']} {value['product']}"
        if "value" in value:
            return value["value"]
    return value


# ============================================================================
# NULL VALUE DETECTION
# ============================================================================

# All variations of "not found" that should be treated as null
NOT_FOUND_VALUES = {
    "not found",
    "not explicitly mentioned",
    "not mentioned",
    "not available",
    "not specified",
    "unknown",
    "n/a",
    "na",
    "none",
    "-",
    "",
}


def _normalize_for_match(s: str) -> str:
    """Accent-strip and uppercase a string for prefix comparison."""
    return unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('ascii').upper().strip()


def filter_codes_by_product(codes: Any, product_name: str) -> Any:
    """Filter a sku_codes / product_codes dict so only entries that belong to
    `product_name` are kept.
    """
    if not product_name or not isinstance(codes, dict):
        return codes

    product_norm = _normalize_for_match(product_name)
    filtered = {}

    for key, value in codes.items():
        key_str = str(key)

        # Detect whether the key carries a product-name prefix.
        # Pattern 1: "PRODUCT COLOR/SIZE"  → slash present + space before slash
        # Pattern 2: "product_color_size"  → underscore-separated, first segment ≥ 3 chars
        has_slash_prefix = "/" in key_str and " " in key_str.split("/")[0]
        has_underscore_prefix = "_" in key_str and len(key_str.split("_")[0]) >= 3

        if not has_slash_prefix and not has_underscore_prefix:
            # Plain colour/variant name — always keep
            filtered[key] = value
            continue

        # Extract the prefix and compare
        if has_slash_prefix:
            raw_prefix = key_str.split("/")[0].split(" ")[0]
        else:
            raw_prefix = key_str.split("_")[0]

        prefix_norm = _normalize_for_match(raw_prefix)
        if prefix_norm == product_norm:
            filtered[key] = value
        # else: belongs to a different product — drop it

    return filtered


def _is_empty_leaf(value: Any) -> bool:
    """Empty after unwrapping a `{"value": ...}` envelope."""
    if isinstance(value, dict) and "value" in value:
        value = value.get("value")
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return is_not_found_value(value)


def is_not_found_value(value: Any) -> bool:
    """Check if a value represents a 'not found' placeholder."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.lower().strip() in NOT_FOUND_VALUES
    return False


def normalize_null_value(value: Any) -> Any:
    """Normalize 'not found' variations to None (will be excluded from metadata)."""
    if is_not_found_value(value):
        return None
    return value


# ============================================================================
# FACTORY KEY NORMALIZATION (canonical: factory_name / factory_group_name)
# ============================================================================

# Aliases that should fold into `factory_name` (the maker / brand)
_FACTORY_NAME_ALIASES = ('manufacturer', 'brand', 'supplier')

# Aliases that should fold into `factory_group_name` (the parent group)
_FACTORY_GROUP_ALIASES = ('factory_group',)


def normalize_factory_keys(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize factory/manufacturer keys to the canonical schema:
    - factory_name        — the maker (e.g. "Harmony")
    - factory_group_name  — the parent group, when meaningfully different (e.g. "Peronda Group")
    """
    if not isinstance(metadata, dict):
        return metadata

    # ── factory_name ─────────────────────────────────────────────────────────
    if is_not_found_value(metadata.get('factory_name')):
        # Try aliases in priority order
        for alias in _FACTORY_NAME_ALIASES:
            val = metadata.get(alias)
            if not is_not_found_value(val):
                metadata['factory_name'] = val
                break
        else:
            # Last resort: AI sometimes writes `factory` as a flat string instead of a nested dict
            factory_val = metadata.get('factory')
            if isinstance(factory_val, str) and not is_not_found_value(factory_val):
                metadata['factory_name'] = factory_val

    # ── factory_group_name ───────────────────────────────────────────────────
    if is_not_found_value(metadata.get('factory_group_name')):
        for alias in _FACTORY_GROUP_ALIASES:
            val = metadata.get(alias)
            if not is_not_found_value(val):
                metadata['factory_group_name'] = val
                break

    # ── Drop the alias keys (single source of truth) ─────────────────────────
    for alias in _FACTORY_NAME_ALIASES + _FACTORY_GROUP_ALIASES:
        metadata.pop(alias, None)

    # If `factory` was a flat string (not a nested dict), drop it too — its value
    # has been folded into factory_name above. The nested-dict form is preserved.
    if 'factory' in metadata and not isinstance(metadata['factory'], dict):
        metadata.pop('factory', None)

    return metadata


# ============================================================================
# MATERIAL CATEGORY NORMALIZATION
# ============================================================================

# ALIASES, not a vocabulary. audit #17 M4-9.
MATERIAL_CATEGORY_MAPPING = {
    # Tiles — all tile variations keep their controlled-vocab slug
    "tile": "floor_tile",
    "tiles": "floor_tile",
    "ceramic": "ceramic_tile",
    "ceramic tile": "ceramic_tile",
    "ceramic tiles": "ceramic_tile",
    "porcelain": "porcelain_tile",
    "porcelain tile": "porcelain_tile",
    "porcelain tiles": "porcelain_tile",
    "stoneware": "floor_tile",
    "stoneware tile": "floor_tile",
    "floor tile": "floor_tile",
    "wall tile": "wall_tile",
    "bathroom tile": "bathroom_tile",
    "shower tile": "shower_tile",
    # Wood / Flooring
    "wood": "wood_flooring",
    "hardwood": "hardwood",
    "laminate": "laminate",
    "engineered wood": "engineered_wood",
    "parquet": "parquet",
    "vinyl": "vinyl_flooring",
    "vinyl flooring": "vinyl_flooring",
    "carpet": "carpet",
    # Stone / General materials
    "stone": "stone_slab",
    "natural stone": "stone_slab",
    "marble": "stone_slab",
    "granite": "stone_slab",
    "limestone": "stone_slab",
    "travertine": "stone_slab",
    "slate": "stone_slab",
    "quartz": "quartz",
    "terrazzo": "terrazzo",
    "concrete": "concrete",
    "countertop": "countertop",
    # Paint / Wall Decor
    "paint": "wall_paint",
    "wall paint": "wall_paint",
    "wallpaper": "wallpaper",
    "wall panel": "wall_panel",
    # Furniture
    "sofa": "sofa",
    "armchair": "armchair",
    "chair": "dining_chair",
    "table": "dining_table",
    "cabinet": "cabinet",
    "bed": "bed",
    "desk": "desk",
    "shelving": "shelving",
    "sideboard": "sideboard",
    # Decor
    "rug": "rug",
    "curtain": "curtain",
    "cushion": "cushion",
    "vase": "vase",
    "mirror": "mirror",
    # Heating
    "radiator": "radiator",
    "towel rail": "towel_rail",
    "boiler": "boiler",
    "fireplace": "fireplace",
    "convector": "convector",
    "heat pump": "heat_pump",
    # Sanitary
    "toilet": "toilet",
    "basin": "basin",
    "bathtub": "bathtub",
    "shower tray": "shower_tray",
    "bidet": "bidet",
    "tap": "tap",
    "faucet": "faucet",
    "mixer": "mixer",
    # Kitchen
    "kitchen cabinet": "kitchen_cabinet",
    "kitchen sink": "kitchen_sink",
    "kitchen hood": "kitchen_hood",
    "worktop": "kitchen_worktop",
    # Lighting
    "light": "lighting",
    "lamp": "lighting",
    "pendant": "pendant_light",
    "chandelier": "chandelier",
    "spotlight": "spotlight",
    # Other
    "glass": "glass_panel",
    "metal": "metal_panel",
    "resin": "countertop",
    "fabric": "fabric_swatch",
    "leather": "leather_swatch",
    "textile": "fabric_swatch",
}

# Composition types for tiles (extracted separately from material category)
TILE_COMPOSITION_MAPPING = {
    "ceramic": "ceramic",
    "porcelain": "porcelain",
    "stoneware": "stoneware",
    "terracotta": "terracotta",
    "glass": "glass",
    "cement": "cement",
}


def normalize_material_category(category: str) -> Tuple[str, Optional[str]]:
    """
    Normalize material category to standard format and extract composition.

    Args:
        category: Raw category string (e.g., "ceramic tile", "porcelain", "Tile")

    Returns:
        Tuple of (normalized_category, composition)
        - normalized_category: Standard material type (e.g., "Tile", "Wood", "Stone")
        - composition: Material composition if applicable (e.g., "ceramic", "porcelain", None)

    Examples:
        "ceramic tile" → ("Tile", "ceramic")
        "porcelain" → ("Tile", "porcelain")
        "tile" → ("Tile", None)
        "marble" → ("Marble", None)
    """
    if not category:
        return (None, None)

    normalized = category.lower().strip()
    slug = "_".join(normalized.replace("-", " ").split())
    base_category = MATERIAL_CATEGORY_MAPPING.get(normalized) or normalized

    # audit #17 M4-9. The registry decides what a category is; this function only spells
    # things. A value already in the controlled vocab is kept BEFORE any alias is applied,
    # so `mdf` stays a panel board instead of being rewritten to wood flooring.
    if field_registry.is_loaded:
        vocab = field_registry.all_controlled_vocab()
        if normalized in vocab:
            base_category = normalized
        elif slug in vocab:
            base_category = slug
        if base_category not in vocab:
            logging.getLogger(__name__).warning(
                "normalize_material_category: '%s' resolved to '%s', which is not in "
                "material_categories.controlled_vocab — dropping rather than inventing a "
                "category. Add the value, or add an alias for it.",
                category, base_category,
            )
            return (None, None)

    # Extract composition for tiles
    composition = None
    if "tile" in (base_category or ""):
        for comp_key, comp_value in TILE_COMPOSITION_MAPPING.items():
            if comp_key in normalized:
                composition = comp_value
                break

    return (base_category, composition)


# ============================================================================
# FIELD VALUE NORMALIZATION
# ============================================================================

def normalize_field_value(value: Any, field_name: str, metadata_dict: dict = None) -> Any:
    """
    Normalize field values to consistent formats.

    Examples:
        - Single designer string → array: "John Doe" → ["John Doe"]
        - Individual SKU fields → object: {"sku_white": "123"} → {"white": "123"}
        - Grout supplier object → string: {"value": "MAPEI", "product": "..."} → "MAPEI ULTRACOLOR PLUS"
        - "not found" variations → None
        - Material category variations → Standard format + composition extraction

    Args:
        value: The value to normalize
        field_name: The field name being normalized
        metadata_dict: Optional metadata dictionary to update with extracted composition
    """
    # First, check if it's a "not found" value
    if is_not_found_value(value):
        return None

    # Normalize material category and extract composition
    if field_name == "material_category":
        if isinstance(value, str):
            category, composition = normalize_material_category(value)
            # If composition was extracted and we have a metadata dict, add it
            if composition and metadata_dict is not None:
                metadata_dict['composition'] = composition
            return category
        return value

    # Normalize factory/manufacturer names - clean up "not found" values
    if field_name in ["factory_name", "factory_group_name", "manufacturer"]:
        if is_not_found_value(value):
            return None
        # Return as-is if it's a valid value
        return value

    # Handle designer normalization (single → array)
    if field_name == "designers":
        if isinstance(value, str):
            return [value]
        elif isinstance(value, dict) and "value" in value:
            val = value["value"]
            return [val] if isinstance(val, str) else val
        return value
    
    # Extract value from confidence objects
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    
    return value


# ============================================================================
# MAIN NORMALIZATION FUNCTION
# ============================================================================

def normalize_metadata(metadata: Dict[str, Any], product_name: Optional[str] = None,
                       product_category: Optional[str] = None) -> Dict[str, Any]:
    """
    Normalize metadata to standardized schema.

    Args:
        metadata: Raw metadata dictionary with inconsistent field names
        product_name: Current product name — used to filter cross-product SKU contamination
        product_category: the product's category; decides category-scoped aliases

    Returns:
        Normalized metadata with standardized field names
    """
    if not metadata:
        return metadata

    normalized = {}

    # Process each category
    for category, fields in metadata.items():
        if not isinstance(fields, dict):
            # Keep non-dict values as-is (e.g., confidence, page_range)
            normalized[category] = fields
            continue

        normalized_fields = {}

        # Step 1: rename spellings/aliases onto the registry key. When two keys land on one
        # field, the first non-empty value wins regardless of which spelling came first.
        for field_name, field_value in fields.items():
            hit = resolve_standard_field(field_name, product_category)
            standard_field, value = field_name, field_value
            if hit:
                target, factor = hit
                if factor != 1.0:
                    converted = scale_value(field_value, factor)
                    if converted is not None:
                        standard_field = target
                        value = normalize_field_value(converted, target, normalized_fields)
                else:
                    standard_field = target
                    value = normalize_field_value(field_value, target, normalized_fields)

            brand = _GROUT_BRAND_KEY_RE.match(str(field_name).lower())
            if standard_field == "grout_color_codes" and brand and field_name != standard_field:
                codes = normalized_fields.get("grout_color_codes")
                codes = dict(codes) if isinstance(codes, dict) else {}
                entry = _grout_entry(field_value)
                if not _is_empty_leaf(entry):
                    codes.setdefault(brand.group(1), entry)
                normalized_fields["grout_color_codes"] = codes
                continue

            if standard_field in normalized_fields:
                existing = normalized_fields[standard_field]
                if _is_empty_leaf(existing) and not _is_empty_leaf(value):
                    normalized_fields[standard_field] = value
            else:
                normalized_fields[standard_field] = value

        # Step 2: Handle special cases (individual SKU/color fields → objects)
        normalized_fields = consolidate_individual_fields(normalized_fields, category, product_name)

        normalized[category] = normalized_fields

    return normalized


_CODE_LIST_KEY_RE = re.compile(r"^(product|format|reference)_code")


def consolidate_individual_fields(fields: Dict[str, Any], category: str, product_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Consolidate individual fields into objects.

    Examples:
        - sku_white, sku_clay, sku_green → sku_codes: {"white": "...", "clay": "...", "green": "..."}
        - grout_color_white, grout_color_clay → grout_color_codes: {"white": "...", "clay": "..."}

    Args:
        fields: Commercial fields dict
        category: Metadata category (only "commercial" is processed)
        product_name: Current product name used to filter cross-product SKU entries (Option B)
    """
    if category != "commercial":
        return fields

    consolidated = {}
    sku_codes = {}
    grout_colors = {}
    product_codes = []

    for field_name, field_value in fields.items():
        # Consolidate SKU codes
        if field_name.startswith("sku_") and field_name not in ["sku_codes", "sku_variants"]:
            # Extract color from field name (e.g., "sku_white" → "white", "sku_fold_white" → "white")
            color = field_name.replace("sku_", "").replace("fold_", "").replace("tri_fold_", "").replace("ona_", "")
            value = field_value.get("value") if isinstance(field_value, dict) else field_value
            if value:
                sku_codes[color] = value

        # Pass-through already-formed sku_codes dict — will be filtered below
        elif field_name == "sku_codes" and isinstance(field_value, dict):
            for k, v in field_value.items():
                sku_codes[k] = v

        # Consolidate grout color codes
        elif field_name != "grout_color_codes" and field_name.startswith("grout_color_") and "code" in field_name:
            # Extract color (e.g., "grout_color_code_white_mapei" → "white")
            parts = field_name.replace("grout_color_", "").replace("_code", "").replace("_mapei", "").replace("_kerakoll", "")
            color = parts.split("_")[0] if "_" in parts else parts
            value = field_value.get("value") if isinstance(field_value, dict) else field_value
            if value:
                grout_colors[color] = value

        # Consolidate product codes
        elif _CODE_LIST_KEY_RE.match(field_name):
            value = field_value.get("value") if isinstance(field_value, dict) else field_value
            if value and value not in product_codes:
                product_codes.append(value)

        else:
            # Keep other fields
            consolidated[field_name] = field_value

    # [Option B] Filter assembled sku_codes by product name to remove cross-product contamination
    if sku_codes and product_name:
        before = len(sku_codes)
        sku_codes = filter_codes_by_product(sku_codes, product_name)
        removed = before - len(sku_codes)
        if removed > 0:
            logger.info(f"🔑 SKU filter: removed {removed} foreign SKU entries for product '{product_name}'")

    # Add consolidated objects
    if sku_codes:
        consolidated["sku_codes"] = sku_codes
    if grout_colors:
        existing = consolidated.get("grout_color_codes")
        consolidated["grout_color_codes"] = {**existing, **grout_colors} if isinstance(existing, dict) else grout_colors
    if product_codes:
        consolidated["product_codes"] = product_codes if len(product_codes) > 1 else product_codes[0]

    return consolidated


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_normalization_report(original: Dict[str, Any], normalized: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a report showing what was normalized.

    Returns:
        {
            "fields_normalized": 15,
            "fields_consolidated": 8,
            "changes": [
                {"from": "colour", "to": "color", "category": "appearance"},
                ...
            ]
        }
    """
    changes = []
    fields_normalized = 0

    for category in original.keys():
        if not isinstance(original.get(category), dict):
            continue

        original_fields = set(original[category].keys())
        normalized_fields = set(normalized.get(category, {}).keys())

        # Find renamed fields
        for orig_field in original_fields:
            if orig_field not in normalized_fields:
                # Field was renamed or consolidated
                standard = find_standard_field(orig_field, category)
                if standard and standard in normalized_fields:
                    changes.append({
                        "from": orig_field,
                        "to": standard,
                        "category": category,
                        "type": "renamed"
                    })
                    fields_normalized += 1

    return {
        "fields_normalized": fields_normalized,
        "fields_consolidated": len([c for c in changes if c.get("type") == "consolidated"]),
        "changes": changes
    }


def validate_and_normalize_product_category(product_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize product category to prevent duplicates.

    This function ensures:
    1. Material type is normalized (e.g., "Tile" not "ceramic tile")
    2. Composition is extracted separately (e.g., "ceramic", "porcelain")
    3. No duplicate category variations exist

    Args:
        product_data: Product data dictionary with 'category' or 'metadata.material_category'

    Returns:
        Updated product data with normalized category and composition

    Examples:
        Input: {"category": "ceramic tile"}
        Output: {"category": "Tile", "metadata": {"composition": "ceramic"}}

        Input: {"category": "porcelain"}
        Output: {"category": "Tile", "metadata": {"composition": "porcelain"}}
    """
    if not product_data:
        return product_data

    # Initialize metadata if not exists
    if 'metadata' not in product_data:
        product_data['metadata'] = {}

    # Check category field
    category = product_data.get('category')
    if category and isinstance(category, str):
        normalized_category, composition = normalize_material_category(category)
        product_data['category'] = normalized_category
        if composition:
            product_data['metadata']['composition'] = composition

    # Check metadata.material_category field
    if 'material_category' in product_data.get('metadata', {}):
        material_category = product_data['metadata']['material_category']
        if isinstance(material_category, str):
            normalized_category, composition = normalize_material_category(material_category)
            product_data['metadata']['material_category'] = normalized_category
            if composition:
                product_data['metadata']['composition'] = composition

    return product_data


