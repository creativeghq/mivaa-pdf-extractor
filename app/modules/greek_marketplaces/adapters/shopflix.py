"""
Shopflix.gr adapter — Firecrawl scrape of the site search page.

Shopflix is a marketplace of third-party Greek sellers. Like Bestdeals,
no public API is available. We fetch the search page via Firecrawl and
extract the top matching listing as a single PriceHit.
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import List, Optional

from app.modules.greek_marketplaces.models import MarketplaceProduct
from app.services.integrations.firecrawl_client import FirecrawlClient
from app.services.integrations.perplexity_price_search_service import PriceHit
from app.utils.price_parsing import parse_price

logger = logging.getLogger(__name__)

SEARCH_URL_TEMPLATE = "https://www.shopflix.gr/search?search={query}"
MODULE_SLUG = "greek-marketplaces"

EXTRACTION_PROMPT = (
    "You are reading a Shopflix.gr search results page. Extract the FIRST "
    "matching product listing. Return `found`=false if none. Prefer "
    "organic (non-sponsored) results. Return the product detail page URL. "
    "Use `retailer_name` for the marketplace seller label when shown; fall "
    "back to 'Shopflix.gr' otherwise. Keep prices as strings with currency "
    "symbols intact."
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

        url = SEARCH_URL_TEMPLATE.format(query=urllib.parse.quote_plus(query))
        result = await self.firecrawl.scrape(
            url=url,
            extraction_model=MarketplaceProduct,
            user_id=user_id,
            workspace_id=workspace_id,
            extraction_prompt=EXTRACTION_PROMPT,
            use_javascript_render=False,
            only_main_content=True,
            module_slug=MODULE_SLUG,
            source_tag="shopflix",
        )

        if not result.success or not result.data or not result.data.found:
            return []

        product = result.data
        if not product.product_url:
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
            )
        ]


_singleton: Optional[ShopflixAdapter] = None


def get_shopflix_adapter() -> ShopflixAdapter:
    global _singleton
    if _singleton is None:
        _singleton = ShopflixAdapter()
    return _singleton
