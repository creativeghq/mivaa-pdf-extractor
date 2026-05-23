"""
Which metadata keys flow through canonicalization, and which never do.

Conservative whitelist — only descriptive natural-language attributes go through
L1+L2. Identifiers, codes, numerics, and free-form prose stay verbatim. Adding a
key here changes filter behavior across every ingest path, so additions should be
deliberate.
"""

CANONICALIZABLE_FACETS: set[str] = {
    "color",
    "available_colors",
    "material",
    "material_type",
    "finish",
    "style",
    "application",
    "room",
    "zone_intent",
    "socket",
    "light_color",
    "mounting_type",
    "surface_pattern",
    "slip_resistance",
    "pei_rating",
    "frost_resistance",
    "wood_type",
    "bowl_shape",
    "flush_type",
    "faucet_type",
    "weave",
    "fiber",
    "upholstery",
    "ip_rating",
    # `tags` intentionally excluded — material-tagger-agent writes free-form tags
    # that should NOT be funneled through canonical clustering (would collapse
    # distinct stylistic descriptors like "vintage" / "retro" / "throwback" that
    # belong as separate filterable values). Add here only if tag de-duplication
    # becomes a real product need.
}

NON_CANONICAL_FACETS: set[str] = {
    "brand", "factory", "factory_name", "factory_group_name",
    "designer", "manufacturer", "supplier",
    "sku", "external_sku", "model_number", "series",
    "dimensions", "width", "height", "depth", "length",
    "weight", "wattage", "voltage", "flow_rate",
    "price", "currency", "unit",
    "name", "description", "long_description",
    "page_range", "confidence", "image_indices",
}


def is_canonicalizable(key: str) -> bool:
    if not key or key.startswith("_"):
        return False
    if key in NON_CANONICAL_FACETS:
        return False
    return key in CANONICALIZABLE_FACETS
