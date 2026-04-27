"""
Bestprice.gr adapter — Firecrawl scrape of the site search page.

Bestprice.gr does not publish a public search API for non-merchants. We
scrape the public search results page sorted by price ascending so the
top row is the cheapest offer for the query.

URL format the user verified works:
  https://www.bestprice.gr/cat/<category-id>/<slug>.html?q=<query>&from=cat&o=2

But we don't always know the right category id up front. The site also
honours a category-less search at:
  https://www.bestprice.gr/search?q=<query>&o=2

`o=2` is the "price ascending" sort. We use the simpler URL — the
extractor only needs to read the top row.

ToS caveat: bestprice.gr's terms prohibit automated scraping. Treat the
adapter the same way as the Skroutz one — fine at admin-triggered
volumes, not safe to scale without a commercial agreement.

One scrape per query → at most one PriceHit. Stricter extraction now:
result is dropped unless the matched product name plausibly shares
tokens with the query, to avoid the "fallback featured product"
false-positive class we saw on bestdeals.gr (which is a different,
unrelated site we accidentally targeted in a previous iteration).
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import List, Optional

from pydantic import BaseModel, Field

from app.modules.greek_marketplaces.facet_filter import adaptive_marketplace_query, matches_facets
from app.modules.greek_marketplaces.match_filter import is_plausible_match
from app.modules.greek_marketplaces.models import MarketplaceProduct
from app.services.integrations.firecrawl_client import FirecrawlClient
from app.services.integrations.perplexity_price_search_service import PriceHit
from app.services.integrations.product_identity_service import QueryFacets
from app.utils.price_parsing import parse_price

logger = logging.getLogger(__name__)

SEARCH_URL_TEMPLATE = "https://www.bestprice.gr/search?q={query}&o=2"
MODULE_SLUG = "greek-marketplaces"
SOURCE_TAG = "bestprice"

EXTRACTION_PROMPT = (
    "You are reading a Bestprice.gr search results page sorted by price "
    "ascending (the o=2 query parameter). Extract the FIRST organic product "
    "listing — since results are sorted by price asc, this is the cheapest "
    "matching offer. **Return `found`=false unless the visible product name "
    "is clearly the same product as the query.** If the page shows 'no "
    "results' / 'δεν βρέθηκαν αποτελέσματα' followed by suggested "
    "products, return `found`=false. Ignore banner ads, sponsored slots, "
    "and 'ίσως σας ενδιαφέρει' (you might like) blocks — those are not "
    "matches. Return the absolute product detail URL on bestprice.gr "
    "(not a redirect or search URL). Include the visible 'was/now' prices "
    "as strings — do not parse numerics."
)

PRODUCT_PAGE_PROMPT = (
    "You are reading a Bestprice.gr product detail page (URL like "
    "/to/<id>/<slug>.html or /item/<id>/<slug>.html). Bestprice lists every "
    "shop selling this exact SKU in a 'Καταστήματα' / 'Shops' section. "
    "Extract EVERY visible shop row. For each row return: shop name, the "
    "shop's product URL on its OWN domain (Bestprice's 'Επίσκεψη "
    "καταστήματος' / 'Visit' button — that target URL, NOT a bestprice.gr "
    "URL), the visible price including VAT and currency symbol, and the "
    "availability label as shown. Return found=true and the shops list in "
    "the page's natural order (Bestprice sorts by price ascending). If no "
    "shop rows are visible, return found=false."
)


class BestpriceShopOffer(BaseModel):
    """One shop row inside a Bestprice product detail page."""

    merchant_name: Optional[str] = Field(default=None, description="Shop name as shown.")
    merchant_url: Optional[str] = Field(
        default=None,
        description="Direct URL on the shop's own domain (NOT bestprice.gr).",
    )
    price: Optional[str] = Field(default=None, description="Visible price string with currency.")
    availability: Optional[str] = Field(default=None, description="Availability label as shown.")


class BestpriceProductPageResult(BaseModel):
    """All visible shop offers on a Bestprice product detail page."""

    found: bool = Field(default=False)
    product_name: Optional[str] = Field(default=None)
    shops: List[BestpriceShopOffer] = Field(default_factory=list)


class BestpriceAdapter:
    """One-call-one-hit adapter for bestprice.gr."""

    def __init__(self, firecrawl_client: Optional[FirecrawlClient] = None) -> None:
        self.firecrawl = firecrawl_client or FirecrawlClient()

    async def search(
        self,
        query: str,
        *,
        user_id: str,
        workspace_id: Optional[str] = None,
        facets: Optional[QueryFacets] = None,
    ) -> List[PriceHit]:
        if not self.firecrawl.api_key:
            logger.debug("Bestprice: Firecrawl not configured, skipping.")
            return []

        # Adaptive query: when facets carry SKU + brand/model, build a tight
        # "{BRAND} {SKU}" search string. The free-text user query often has
        # too many tokens for Bestprice's literal-match search; tightening
        # the query is the difference between zero results and the right SKU.
        adaptive_query = adaptive_marketplace_query(query=query, facets=facets)
        url = SEARCH_URL_TEMPLATE.format(query=urllib.parse.quote_plus(adaptive_query))
        result = await self.firecrawl.scrape(
            url=url, extraction_model=MarketplaceProduct,
            user_id=user_id, workspace_id=workspace_id,
            extraction_prompt=EXTRACTION_PROMPT,
            use_javascript_render=False, only_main_content=True,
            module_slug=MODULE_SLUG, source_tag=SOURCE_TAG,
        )

        if not result.success or not result.data or not result.data.found:
            return []

        product = result.data
        if not product.product_url:
            return []

        # Match-quality safeguard — kills the "Brenthaven notebook lock for
        # an Orabella faucet query" class of false positives.
        if not is_plausible_match(query, product.product_url, product.retailer_name):
            logger.info(
                "Bestprice: dropped likely false positive — query=%r, url=%s",
                query, product.product_url,
            )
            return []

        # Facet-aware filter: when the caller passed SKU anchors / product type,
        # drop rows whose URL slug carries a different SKU before they reach
        # the LLM classifier.
        if not matches_facets(facets=facets, candidate_url=product.product_url, candidate_name=product.retailer_name):
            logger.info(
                "Bestprice: facet mismatch (sku=%s, type=%s) — query=%r, url=%s",
                facets.sku_tokens if facets else None,
                facets.product_type if facets else None,
                query, product.product_url,
            )
            return []

        # Bestprice product pages aggregate every shop selling that SKU.
        # Fan out into per-shop hits when the URL is a /to/ or /item/ page.
        if "bestprice.gr/to/" in product.product_url or "bestprice.gr/item/" in product.product_url:
            shop_hits = await self._fanout_product_page(
                product.product_url, query=query,
                fallback_currency=product.currency or "EUR",
                user_id=user_id, workspace_id=workspace_id,
                facets=facets,
            )
            if shop_hits:
                return shop_hits

        # Fallback: single-row legacy emit when fanout isn't possible (search
        # returned a category URL, or product page extraction failed).
        price, currency = parse_price(product.price, hint_currency=product.currency or "EUR")
        original, _ = parse_price(product.original_price, hint_currency=product.currency or "EUR")
        return [
            PriceHit(
                retailer_name=product.retailer_name or "Bestprice.gr",
                product_url=product.product_url,
                price=float(price) if price is not None else None,
                original_price=float(original) if original is not None else None,
                currency=currency or "EUR",
                availability=product.availability,
                source=SOURCE_TAG,
                verified=False,
                notes="via Bestprice",
            )
        ]

    async def _fanout_product_page(
        self,
        product_url: str,
        *,
        query: str,
        fallback_currency: str,
        user_id: str,
        workspace_id: Optional[str],
        facets: Optional[QueryFacets] = None,
    ) -> List[PriceHit]:
        """Scrape the Bestprice product detail page and emit a hit per shop."""
        try:
            result = await self.firecrawl.scrape(
                url=product_url,
                extraction_model=BestpriceProductPageResult,
                user_id=user_id, workspace_id=workspace_id,
                extraction_prompt=PRODUCT_PAGE_PROMPT,
                use_javascript_render=False, only_main_content=True,
                module_slug=MODULE_SLUG, source_tag="bestprice_product_page",
            )
        except Exception as e:
            logger.warning("Bestprice product-page scrape failed (%s): %s", product_url, e)
            return []
        if not result.success or not result.data or not result.data.found:
            return []
        # Page-level identity safeguard.
        if not is_plausible_match(query, product_url, result.data.product_name):
            logger.info(
                "Bestprice: product page failed plausibility check — query=%r, url=%s",
                query, product_url,
            )
            return []
        # Facet-aware filter on the product page itself (catches the case
        # where the search-page URL passed but the product detail page
        # is actually a different SKU).
        if not matches_facets(facets=facets, candidate_url=product_url, candidate_name=result.data.product_name):
            logger.info(
                "Bestprice: product page facet mismatch — query=%r, url=%s",
                query, product_url,
            )
            return []

        hits: List[PriceHit] = []
        for shop in result.data.shops:
            if not shop.merchant_url or not shop.merchant_name:
                continue
            price, currency = parse_price(shop.price, hint_currency=fallback_currency)
            avail = (shop.availability or "").lower()
            if any(tok in avail for tok in ("εκτός", "out", "unavail")):
                availability = "out_of_stock"
            else:
                availability = "in_stock"
            hits.append(
                PriceHit(
                    retailer_name=shop.merchant_name,
                    product_url=shop.merchant_url,
                    price=float(price) if price is not None else None,
                    currency=currency or fallback_currency,
                    availability=availability,
                    source=SOURCE_TAG,
                    verified=False,
                    notes="via Bestprice",
                )
            )
        return hits


_singleton: Optional[BestpriceAdapter] = None


def get_bestprice_adapter() -> BestpriceAdapter:
    global _singleton
    if _singleton is None:
        _singleton = BestpriceAdapter()
    return _singleton
