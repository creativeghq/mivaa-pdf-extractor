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

from app.modules.greek_marketplaces.match_filter import is_plausible_match
from app.modules.greek_marketplaces.models import MarketplaceProduct
from app.services.integrations.firecrawl_client import FirecrawlClient
from app.services.integrations.perplexity_price_search_service import PriceHit
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
    ) -> List[PriceHit]:
        if not self.firecrawl.api_key:
            logger.debug("Bestprice: Firecrawl not configured, skipping.")
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
            source_tag=SOURCE_TAG,
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
                query,
                product.product_url,
            )
            return []

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
            )
        ]


_singleton: Optional[BestpriceAdapter] = None


def get_bestprice_adapter() -> BestpriceAdapter:
    global _singleton
    if _singleton is None:
        _singleton = BestpriceAdapter()
    return _singleton
