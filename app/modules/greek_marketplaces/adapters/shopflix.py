"""
Shopflix.gr adapter — Firecrawl scrape of the Spryker-powered search page.

Shopflix is a marketplace of third-party Greek sellers, built on
Spryker / Algolia. The canonical search URL with price-asc sort:

  https://shopflix.gr/search
    ?prod_GR_spryker[query]=<query>
    &prod_GR_spryker[sortBy]=prod_GR_spryker_search-result-data.price_asc
    &k=<query>

`prod_GR_spryker[query]` and `k` are both required — the second is
the URL-bar canonical that the JS framework reads on hard reload.
`sortBy=...price_asc` puts the cheapest match at the top, matching
the price-asc convention used by the Skroutz and Bestprice adapters.
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import List, Optional

from app.modules.greek_marketplaces.match_filter import is_plausible_match
from app.modules.greek_marketplaces.models import MarketplaceProduct
from app.services.integrations.firecrawl_client import FirecrawlClient
from app.services.integrations.perplexity_price_search_service import PriceHit
from app.utils.price_parsing import parse_price

logger = logging.getLogger(__name__)

SHOPFLIX_BASE_URL = "https://shopflix.gr/search"
SHOPFLIX_SORT_PRICE_ASC = "prod_GR_spryker_search-result-data.price_asc"
MODULE_SLUG = "greek-marketplaces"


def _build_search_url(query: str) -> str:
    """Compose the Spryker-style search URL with price-asc sort.

    Spryker reads both `prod_GR_spryker[query]` (the search input) and
    `k` (the URL-bar canonical) — we pass the same value to both.
    """
    encoded = urllib.parse.quote(query, safe="")
    params = (
        f"prod_GR_spryker%5Bquery%5D={encoded}"
        f"&prod_GR_spryker%5BsortBy%5D={SHOPFLIX_SORT_PRICE_ASC}"
        f"&k={encoded}"
    )
    return f"{SHOPFLIX_BASE_URL}?{params}"

EXTRACTION_PROMPT = (
    "You are reading a Shopflix.gr search results page sorted by price "
    "ascending (sortBy=price_asc). Extract the FIRST organic product "
    "listing — since results are sorted by price asc, this is the "
    "cheapest matching offer. Return `found`=false unless the visible "
    "product name plausibly matches the query (same brand/model). If "
    "the page shows 'no results' or only suggested/promotional products, "
    "return `found`=false. Return the absolute product detail URL on "
    "shopflix.gr. Use `retailer_name` for the marketplace seller label "
    "when shown; fall back to 'Shopflix.gr' otherwise. Keep prices as "
    "strings with currency symbols intact."
)


class ShopflixAdapter:
    """One-call-one-hit adapter for shopflix.gr."""

    def __init__(self, firecrawl_client: Optional[FirecrawlClient] = None) -> None:
        self.firecrawl = firecrawl_client or FirecrawlClient()

    async def search(
        self,
        query: str,
        *,
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> List[PriceHit]:
        if not self.firecrawl.api_key:
            logger.debug("Shopflix: Firecrawl not configured, skipping.")
            return []

        url = _build_search_url(query)

        # Shopflix's results are rendered client-side (Algolia/Spryker SPA),
        # so a static HTML scrape returns nothing. JS render is required.
        # Spryker hydration is slow + occasionally finishes after Firecrawl
        # snapshots — retry once on empty extraction (one extra credit when
        # we miss, no extra cost on the happy path).
        result = await self.firecrawl.scrape(
            url=url,
            extraction_model=MarketplaceProduct,
            user_id=user_id,
            workspace_id=workspace_id,
            extraction_prompt=EXTRACTION_PROMPT,
            use_javascript_render=True,
            only_main_content=True,
            module_slug=MODULE_SLUG,
            source_tag="shopflix",
        )
        if result.success and (not result.data or not result.data.found):
            logger.info("Shopflix: empty hydration, retrying once.")
            result = await self.firecrawl.scrape(
                url=url,
                extraction_model=MarketplaceProduct,
                user_id=user_id,
                workspace_id=workspace_id,
                extraction_prompt=EXTRACTION_PROMPT,
                use_javascript_render=True,
                only_main_content=True,
                module_slug=MODULE_SLUG,
                source_tag="shopflix",
            )

        if not result.success or not result.data or not result.data.found:
            return []

        product = result.data
        if not product.product_url:
            return []

        if not is_plausible_match(query, product.product_url, product.retailer_name):
            logger.info(
                "Shopflix: dropped likely false positive — query=%r, url=%s",
                query,
                product.product_url,
            )
            return []

        price, currency = parse_price(product.price, hint_currency=product.currency or "EUR")
        original, _ = parse_price(product.original_price, hint_currency=product.currency or "EUR")

        return [
            PriceHit(
                retailer_name=product.retailer_name or "Shopflix.gr",
                product_url=product.product_url,
                price=float(price) if price is not None else None,
                original_price=float(original) if original is not None else None,
                currency=currency or "EUR",
                availability=product.availability,
                source="shopflix",
                verified=False,
                notes="via Shopflix",
            )
        ]


_singleton: Optional[ShopflixAdapter] = None


def get_shopflix_adapter() -> ShopflixAdapter:
    global _singleton
    if _singleton is None:
        _singleton = ShopflixAdapter()
    return _singleton
