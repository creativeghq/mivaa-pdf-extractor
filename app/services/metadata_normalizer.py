"""
Metadata Normalization Service

Automatically normalizes metadata field names to a standardized schema.
Uses semantic similarity to detect variations without hardcoding every possible field name.

This allows the system to:
1. Handle new field name variations automatically
2. Standardize existing inconsistencies
3. Work for ANY metadata field (not just known ones)
4. Minimal performance impact (~50ms per product)
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


# ============================================================================
# STANDARDIZED METADATA SCHEMA
# ============================================================================

STANDARD_SCHEMA = {
    "commercial": {
        "grout_mapei": ["recommended_grout_mapei", "grout_product_mapei", "supplier_mapei", "grout_supplier_mapei"],
        "grout_kerakoll": ["recommended_grout_kerakoll", "grout_product_kerakoll", "supplier_kerakoll", "grout_supplier_kerakoll"],
        "grout_suppliers": ["recommended_grout_brands", "grout_brands"],
        "grout_color_codes": ["grout_color_codes_mapei", "grout_color_codes_kerakoll"],
        "sku_codes": ["sku_variants", "sku_list"],
        "product_codes": ["product_code", "product_code_prefix", "reference_code", "format_code"],
    },
    "design": {
        "designers": ["designer", "designer_members", "designer_name"],
        "studio": ["design_studio", "designer_studio", "design_studio_name"],
        "studio_founded": ["studio_founded_year", "design_year_founded", "design_studio_founded"],
        "collection": ["collection_name", "collection_series"],
        "philosophy": ["design_philosophy", "design_concept"],
        "inspiration": ["design_inspiration", "inspiration_source"],
    },
    "packaging": {
        "pieces_per_box": ["pieces_per_unit", "pcs_per_box"],
        "boxes_per_pallet": ["boxes_per_pallet_count"],
        "weight_kg": ["weight_per_box", "weight_per_box_kg", "box_weight_kg"],
        "weight_lb": ["weight_per_box_lb", "box_weight_lb"],
        "coverage_m2": ["sqm_per_box", "square_meters_per_box", "area_per_box"],
        "coverage_sqft": ["sqft_per_box", "square_feet_per_box", "area_per_box_sqft"],
    },
    "material_properties": {
        "finish": ["surface_finish", "finish_type"],
        "body_type": ["body", "tile_body", "body_composition"],
        "composition": ["material_composition", "material_type"],
        "texture": ["surface_texture", "texture_type"],
    },
    "appearance": {
        "colors": ["color_variants", "colors_available", "available_colors"],
        "shade_variation": ["shade_var", "variation"],
        "visual_effect": ["visual_effects", "effect"],
    },
    "application": {
        "recommended_use": ["use", "application_type", "recommended_application"],
        "installation": ["installation_method", "installation_type"],
        "traffic_level": ["traffic", "traffic_rating"],
    },
}


# ============================================================================
# SEMANTIC SIMILARITY FUNCTIONS
# ============================================================================

def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings (0.0 to 1.0)."""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def find_standard_field(field_name: str, category: str, threshold: float = 0.6) -> Optional[str]:
    """
    Find the standard field name for a given field using semantic similarity.
    
    Args:
        field_name: The field name to normalize (e.g., "recommended_grout_mapei")
        category: The metadata category (e.g., "commercial")
        threshold: Minimum similarity score to consider a match (default: 0.6)
    
    Returns:
        Standard field name if found, None otherwise
    """
    if category not in STANDARD_SCHEMA:
        return None
    
    best_match = None
    best_score = 0.0
    
    for standard_field, variations in STANDARD_SCHEMA[category].items():
        # Check exact match with standard field
        if field_name == standard_field:
            return standard_field
        
        # Check exact match with known variations
        if field_name in variations:
            return standard_field
        
        # Check semantic similarity with standard field
        score = calculate_similarity(field_name, standard_field)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = standard_field
        
        # Check semantic similarity with variations
        for variation in variations:
            score = calculate_similarity(field_name, variation)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = standard_field
    
    return best_match


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
# MATERIAL CATEGORY NORMALIZATION
# ============================================================================

# Standard material categories - normalize variations to these
MATERIAL_CATEGORY_MAPPING = {
    # Tiles
    "tile": "Tile",
    "tiles": "Tile",
    "ceramic": "Ceramic Tile",
    "ceramic tile": "Ceramic Tile",
    "ceramic tiles": "Ceramic Tile",
    "porcelain": "Porcelain Tile",
    "porcelain tile": "Porcelain Tile",
    "porcelain tiles": "Porcelain Tile",
    # Stone
    "stone": "Natural Stone",
    "natural stone": "Natural Stone",
    "marble": "Marble",
    "granite": "Granite",
    # Wood
    "wood": "Wood",
    "hardwood": "Hardwood",
    "laminate": "Laminate",
    "mdf": "MDF",
    # Other
    "glass": "Glass",
    "metal": "Metal",
    "composite": "Composite",
}


def normalize_material_category(category: str) -> str:
    """Normalize material category to standard format."""
    if not category:
        return None

    normalized = category.lower().strip()
    return MATERIAL_CATEGORY_MAPPING.get(normalized, category.title())


# ============================================================================
# FIELD VALUE NORMALIZATION
# ============================================================================

def normalize_field_value(value: Any, field_name: str) -> Any:
    """
    Normalize field values to consistent formats.

    Examples:
        - Single designer string → array: "John Doe" → ["John Doe"]
        - Individual SKU fields → object: {"sku_white": "123"} → {"white": "123"}
        - Grout supplier object → string: {"value": "MAPEI", "product": "..."} → "MAPEI ULTRACOLOR PLUS"
        - "not found" variations → None
        - Material category variations → Standard format
    """
    # First, check if it's a "not found" value
    if is_not_found_value(value):
        return None

    # Normalize material category
    if field_name == "material_category":
        return normalize_material_category(value) if isinstance(value, str) else value

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
    
    # Handle grout supplier normalization (object → string)
    if field_name in ["grout_mapei", "grout_kerakoll"]:
        if isinstance(value, dict):
            if "product" in value and "value" in value:
                return f"{value['value']} {value['product']}"
            elif "value" in value:
                return value["value"]
        return value
    
    # Handle array normalization (array → first element for single values)
    if field_name in ["grout_suppliers"] and isinstance(value, list) and len(value) == 2:
        # Keep as array for multiple suppliers
        return value
    
    # Extract value from confidence objects
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    
    return value


# ============================================================================
# MAIN NORMALIZATION FUNCTION
# ============================================================================

def normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize metadata to standardized schema.

    Args:
        metadata: Raw metadata dictionary with inconsistent field names

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
        processed_fields = set()

        # Step 1: Normalize known fields
        for field_name, field_value in fields.items():
            if field_name in processed_fields:
                continue

            # Find standard field name
            standard_field = find_standard_field(field_name, category)

            if standard_field:
                # Normalize the value
                normalized_value = normalize_field_value(field_value, standard_field)

                # Merge if standard field already exists
                if standard_field in normalized_fields:
                    # Handle merging logic (e.g., combine arrays, prefer non-null values)
                    existing = normalized_fields[standard_field]
                    if normalized_value and not existing:
                        normalized_fields[standard_field] = normalized_value
                else:
                    normalized_fields[standard_field] = normalized_value

                processed_fields.add(field_name)
            else:
                # Keep unknown fields as-is
                normalized_fields[field_name] = field_value
                processed_fields.add(field_name)

        # Step 2: Handle special cases (individual SKU/color fields → objects)
        normalized_fields = consolidate_individual_fields(normalized_fields, category)

        normalized[category] = normalized_fields

    return normalized


def consolidate_individual_fields(fields: Dict[str, Any], category: str) -> Dict[str, Any]:
    """
    Consolidate individual fields into objects.

    Examples:
        - sku_white, sku_clay, sku_green → sku_codes: {"white": "...", "clay": "...", "green": "..."}
        - grout_color_white, grout_color_clay → grout_color_codes: {"white": "...", "clay": "..."}
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

        # Consolidate grout color codes
        elif field_name.startswith("grout_color_") and "code" in field_name:
            # Extract color (e.g., "grout_color_code_white_mapei" → "white")
            parts = field_name.replace("grout_color_", "").replace("_code", "").replace("_mapei", "").replace("_kerakoll", "")
            color = parts.split("_")[0] if "_" in parts else parts
            value = field_value.get("value") if isinstance(field_value, dict) else field_value
            if value:
                grout_colors[color] = value

        # Consolidate product codes
        elif field_name.startswith("product_code") or field_name.startswith("format_code") or field_name.startswith("reference_code"):
            value = field_value.get("value") if isinstance(field_value, dict) else field_value
            if value and value not in product_codes:
                product_codes.append(value)

        else:
            # Keep other fields
            consolidated[field_name] = field_value

    # Add consolidated objects
    if sku_codes:
        consolidated["sku_codes"] = sku_codes
    if grout_colors:
        consolidated["grout_color_codes"] = grout_colors
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
                {"from": "recommended_grout_mapei", "to": "grout_mapei", "category": "commercial"},
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


