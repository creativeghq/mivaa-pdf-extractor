"""
Bestdeals.gr adapter — Firecrawl scrape of the site search page.

Bestdeals does not publish an API. We ask Firecrawl to open the site's
search results page for the query and return the top matching product
row (URL + price + availability). One query → at most one PriceHit.
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

SEARCH_URL_TEMPLATE = "https://www.bestdeals.gr/search?q={query}"
MODULE_SLUG = "greek-marketplaces"

EXTRACTION_PROMPT = (
    "You are reading a Bestdeals.gr search results page. Extract the FIRST "
    "matching product listing. Return `found`=false if no products are shown. "
    "Prefer the sponsored/top row but ignore banner ads. Return the product "
    "detail page URL (not a redirect or search URL). Include the visible "
    "'was/now' prices as strings — do not parse numerics."
)


class BestdealsAdapter:
    """One-call-one-hit adapter for bestdeals.gr."""

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
            logger.debug("Bestdeals: Firecrawl not configured, skipping.")
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
                retailer_name=product.retailer_name or "Bestdeals.gr",
                product_url=product.product_url,
                price=float(price) if price is not None else None,
                original_price=float(original) if original is not None else None,
                currency=currency or "EUR",
                availability=product.availability,
                source="bestdeals",
                verified=False,
            )
        ]


_singleton: Optional[BestdealsAdapter] = None


def get_bestdeals_adapter() -> BestdealsAdapter:
    global _singleton
    if _singleton is None:
        _singleton = BestdealsAdapter()
    return _singleton
