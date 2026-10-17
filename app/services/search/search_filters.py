"""Search filters as product predicates, applied before candidates are cut to top_k."""

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: Real `products` columns a filter may name directly.
PRODUCT_FILTER_COLUMNS = frozenset({
    "category", "item_type", "country_of_origin", "wood_species_common",
    "sku", "mpn", "barcode", "warranty",
})

#: Keys that mean "the product's material category", from callers and query understanding.
CATEGORY_FILTER_KEYS = frozenset({
    "categories", "category", "material_category", "attributes.material_category",
})

_ATTRIBUTE_KEY = re.compile(r"^attributes\.([a-z0-9_]+)$")
# PostgREST logic-tree delimiters and wildcards. Stripped, never escaped: a value is
# a word to match, so losing a comma or a parenthesis cannot change what it means.
_UNSAFE = re.compile(r'[,()"*\\:]')
_MAX_VALUE_LEN = 100


def filter_values(raw: Any) -> List[str]:
    """The match terms in one filter value (scalar, list, or {contains|in|eq: ...})."""
    if isinstance(raw, dict):
        inner = raw.get("contains", raw.get("in", raw.get("eq")))
        raw = inner
    values: Iterable[Any] = raw if isinstance(raw, (list, tuple, set)) else [raw]
    out: List[str] = []
    for value in values:
        if value is None or isinstance(value, (dict, list, tuple, set)):
            continue
        term = re.sub(r"\s+", " ", _UNSAFE.sub(" ", str(value))).strip()[:_MAX_VALUE_LEN]
        if term and term not in out:
            out.append(term)
    return out


def build_filter_predicates(
    material_filters: Optional[Dict[str, Any]],
    category_ids_by_key: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, Any]]:
    """Translate filters into PostgREST predicates over `products`.

    Returns `(predicates, applied, ignored)`. Each predicate is the member list of
    one `or(...)` group; every group must hold. A key with no predicate form (free
    metadata, numeric operators) lands in `ignored` so the response can say so.
    """
    predicates: List[str] = []
    applied: Dict[str, List[str]] = {}
    ignored: Dict[str, Any] = {}
    category_ids_by_key = category_ids_by_key or {}

    for key, raw in (material_filters or {}).items():
        values = filter_values(raw)
        if not values:
            ignored[key] = raw
            continue

        if key in CATEGORY_FILTER_KEYS:
            members: List[str] = []
            for value in values:
                members.append(f"category.ilike.{value}")
                members.append(f"attributes->>material_category.ilike.{value}")
            ids = sorted({category_ids_by_key[v] for v in values if v in category_ids_by_key})
            if ids:
                members.append(f"category_id.in.({','.join(ids)})")
        elif _ATTRIBUTE_KEY.match(key):
            field = _ATTRIBUTE_KEY.match(key).group(1)
            members = [f"attributes->>{field}.ilike.*{value}*" for value in values]
        elif key in PRODUCT_FILTER_COLUMNS:
            members = [f"{key}.ilike.*{value}*" for value in values]
        else:
            ignored[key] = raw
            continue

        predicates.append(",".join(members))
        applied[key] = values

    return predicates, applied, ignored


def predicate_tree(predicates: List[str]) -> Optional[str]:
    """One `or=` argument requiring every group: `and(or(g1),or(g2),...)`."""
    if not predicates:
        return None
    return "and(" + ",".join(f"or({group})" for group in predicates) + ")"
