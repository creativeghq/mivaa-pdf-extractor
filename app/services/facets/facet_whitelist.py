"""Which metadata keys flow through canonicalization, and which never do."""
from __future__ import annotations

import logging
from typing import Any

from app.services.metadata.field_registry import field_registry

logger = logging.getLogger(__name__)

# : Structural keys that are never facets under ANY registry: identifiers, prose, money, raw
# : dimensions, pipeline bookkeeping.
# :
NON_CANONICAL_FACETS: set[str] = {
    "brand", "factory", "factory_name", "factory_group_name",
    "designer", "manufacturer", "supplier",
    "sku", "external_sku", "model_number", "series",
    "dimensions", "width", "height", "depth", "length",
    "weight", "wattage", "voltage", "flow_rate",
    "price", "currency", "unit",
    "name", "description", "long_description",
    "page_range", "confidence", "image_indices",
    # `tags` intentionally excluded from canonicalization — material-tagger-agent writes
    # free-form tags that should NOT be funneled through canonical clustering (it would
    # collapse distinct stylistic descriptors like "vintage" / "retro" / "throwback" that
    # belong as separate filterable values).
    "tags",
}


def is_canonicalizable(key: str) -> bool:
    """True when this metadata key should be canonicalized (L1 normalize + L2 cluster).

    Raises `FieldRegistryNotLoaded` if the registry has not been loaded. That is deliberate:
    answering `False` for everything would silently produce zero facets forever, with nothing
    failing anywhere. Callers on an ingest path must `await field_registry.ensure_loaded()`.
    """
    return field_registry.is_canonicalizable(key)


def capture_permissively(key: str, value: Any) -> bool:
    """Degraded-path rule for `attributes_raw`: capture unless it is structurally never a facet.

    `attributes_raw` is the LOSSLESS replay contract — with it a later re-canonicalization pass
    rebuilds `attributes` without re-ingesting; without it the product is permanently unfacetable.
    So when the registry is unavailable the safe direction is to over-capture: a spurious key in
    the raw map is recoverable, a missing one is not.
    """
    if not key or key.startswith("_") or value is None:
        return False
    return key not in NON_CANONICAL_FACETS
