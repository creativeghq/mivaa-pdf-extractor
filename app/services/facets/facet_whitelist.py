"""
Which metadata keys flow through canonicalization, and which never do.

**The allowlist now lives in the database** — `material_metadata_fields.canonicalize` (#347
phase 3.2). This module is the seam: `is_canonicalizable()` delegates to the loaded field
registry so the ~40 call sites do not have to care where the answer comes from.

The hardcoded `CANONICALIZABLE_FACETS` set that used to live here is gone. It was one of six
copies of "what fields exist" and it disagreed with the others: 16 of its 24 keys were never
requested by any extraction prompt, so they could only ever arrive if the model volunteered them
under exactly that name.

`NON_CANONICAL_FACETS` survives, for one specific job — see below. It is NOT a second allowlist.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.metadata.field_registry import field_registry

logger = logging.getLogger(__name__)

#: Structural keys that are never facets under ANY registry: identifiers, prose, money, raw
#: dimensions, pipeline bookkeeping.
#:
#: This is NOT a mirror of the DB allowlist and must never be maintained as one. It exists only
#: for the DEGRADED path in `collect_raw_attributes()`, which runs when the registry could not be
#: loaded. There, refusing to answer would throw away the lossless raw map — so that path
#: over-captures instead, and this list is the floor under the over-capture.
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
