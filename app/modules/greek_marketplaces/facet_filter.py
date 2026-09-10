"""Facet-aware post-filter for marketplace adapter results."""

from __future__ import annotations

from typing import Optional
from urllib.parse import unquote, urlparse

from app.services.integrations.product_identity_service import (
    QueryFacets,
    normalize_model_token,
    normalize_text,
)


def adaptive_marketplace_query(*, query: str, facets: Optional[QueryFacets]) -> str:
    """Build the search-string we send to Skroutz/Bestprice/Shopflix."""
    if facets is None:
        return query
    sku = (facets.sku_tokens or [None])[0] if facets.sku_tokens else None
    brand = (facets.brand or "").strip().upper() or None
    model = (facets.model or "").strip().upper() or None

    # Brand + Model + SKU → tightest while still anchoring on the product
    # line. Bestprice in particular returns zero for "BRAND SKU" but works
    # for "BRAND MODEL SKU" (validated against ORABELLA PRECIOSA 10356).
    # Shopflix's fuzzy match also benefits from the model token: without it
    # we get adjacent SKUs like 10365 instead of 10356.
    if sku and brand and model:
        return f"{brand} {model} {sku}"
    if sku and brand:
        return f"{brand} {sku}"
    if sku and model:
        return f"{model} {sku}"
    if sku:
        return sku
    if brand and model:
        return f"{brand} {model}"
    return query


def matches_facets(
    *,
    facets: Optional[QueryFacets],
    candidate_url: str,
    candidate_name: Optional[str] = None,
) -> bool:
    """Return True if the candidate is consistent with the facets we know."""
    if facets is None:
        return True

    sku_tokens = [normalize_model_token(t) for t in (facets.sku_tokens or []) if t]
    product_type = (facets.product_type or "").strip()

    if not sku_tokens and not product_type:
        return True

    # Build the haystack: URL path tokens + product name. Decode percent
    # escapes so Greek shows as Greek not %CE%9C…
    parsed = urlparse(candidate_url or "")
    slug = unquote(parsed.path or "")
    haystack_text = f"{slug} {candidate_name or ''}"
    haystack_norm = normalize_model_token(haystack_text)
    haystack_loose = normalize_text(haystack_text)

    # SKU is the strongest signal — when present, it's required.
    if sku_tokens:
        return any(t in haystack_norm for t in sku_tokens)

    # Product-type signal. We map a few common normalized types to
    # synonym lists (Greek + English) so the filter works across
    # localized retailer pages.
    type_synonyms = _PRODUCT_TYPE_SYNONYMS.get(product_type.lower())
    if not type_synonyms:
        # We don't have a synonym mapping — be permissive (let the
        # classifier handle it).
        return True
    return any(s in haystack_loose for s in type_synonyms)


# Greek + English synonym families per product_type. Keep these
# additive — when in doubt, accept the candidate (the LLM classifier
# is the final word; this filter is only there to drop OBVIOUSLY
# wrong rows like "spout" when the user asked for "faucet").
_PRODUCT_TYPE_SYNONYMS = {
    "basin_faucet": [
        "basin faucet", "basin mixer", "basin tap", "lavatory faucet",
        "μπαταρια νιπτηρα", "μπαταρια νιπτηρος", "νιπτηρα",
    ],
    "shower_faucet": [
        "shower faucet", "shower mixer", "shower tap",
        "μπαταρια ντουζ", "μπαταρια ντουζιερας", "ντους",
    ],
    "bath_faucet": [
        "bath faucet", "bath mixer", "bath tap", "tub faucet",
        "μπαταρια λουτρου", "μπαταρια μπανιερας",
    ],
    "kitchen_faucet": [
        "kitchen faucet", "kitchen mixer", "sink mixer",
        "μπαταρια κουζινας", "μπαταρια νεροχυτη",
    ],
    "shower_column": [
        "shower column", "shower system", "shower set",
        "στηλη ντους", "συστημα ντους",
    ],
    "shower_outlet": [
        "shower outlet", "shower spout",
        "εκροη ντους",
    ],
    "tile": [
        "tile", "πλακακι", "πλακακια",
    ],
    "range_hood": [
        "range hood", "extractor hood", "cooker hood",
        "απορροφητηρας",
    ],
}
