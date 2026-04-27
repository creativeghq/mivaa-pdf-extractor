"""
Facet-aware post-filter for marketplace adapter results.

Marketplace search engines (Skroutz/Bestprice/Shopflix) return their
top result for a given query string. When the query is brand-only
(e.g. "ORABELLA PRECIOSA"), the cheapest item in that brand line wins —
which is usually a small accessory, not the SKU the user actually
wants.

When the caller's facets carry SKU anchors or product-type words, we
can drop wrong-SKU/wrong-type rows BEFORE returning them so the
classifier doesn't see them and the user doesn't see them under
"Similar Products".

Used by all three Greek adapters and by the Idealo adapter.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import unquote, urlparse

from app.services.integrations.product_identity_service import (
    QueryFacets,
    normalize_model_token,
    normalize_text,
)


def adaptive_marketplace_query(*, query: str, facets: Optional[QueryFacets]) -> str:
    """
    Build the search-string we send to Skroutz/Bestprice/Shopflix.

    The free-text user query (e.g. "ORABELLA PRECIOSA Brushed Nickel
    Mpataria Niptiros 10356") often has too many tokens for these sites'
    literal-match search engines — appending the SKU to the full string
    can drop matches from "many results" to "zero" (Bestprice/Shopflix).

    Strategy:
      * If facets carry SKU + brand: send "{brand} {sku}". Tightest
        possible identity anchor — every marketplace that lists the SKU
        will index brand+SKU together.
      * If SKU is present but brand isn't: send "{model} {sku}" or just
        "{sku}".
      * If only brand+model are known (no SKU): send "{brand} {model}".
      * Otherwise: send the original free-text query unchanged.

    Brand and model are normalized to UPPER for consistency.
    """
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
    """
    Return True if the candidate is consistent with the facets we know.

    Decision tree:
      * facets is None → True (no constraints to apply)
      * facets.sku_tokens is empty AND product_type is empty → True
      * facets.sku_tokens non-empty → at least ONE sku_token must appear in
          (URL slug ∪ candidate_name). Otherwise False.
      * facets.sku_tokens empty BUT product_type non-empty → product_type
          word(s) should match SOMETHING in the candidate. We're lenient
          here because product_type is normalized English while pages
          often carry Greek nouns; we check a few common Greek/English
          synonyms before giving up.

    Cheap, deterministic, runs on every adapter result before the row
    leaves the adapter. No LLM cost.
    """
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
