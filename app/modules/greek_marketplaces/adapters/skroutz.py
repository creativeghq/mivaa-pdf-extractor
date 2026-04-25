"""
Skroutz.gr adapter — Firecrawl scrape of the public search page.

Why not the official API: Skroutz's developer Products API is merchant-only
(requires a Skroutz merchant account to obtain an API token). Since this
platform isn't a Skroutz-registered shop, the API path isn't available —
we read the public website the same way a browser would.

ToS caveat: Skroutz's Terms of Service prohibit automated scraping. Low-volume
admin-triggered or per-product-refresh queries are usually tolerated;
high-volume automated scraping can lead to rate limiting or IP blocking.
Before scaling this adapter up, either (a) contract with Skroutz for
commercial data access or (b) confirm legal clearance.

Flow: one scrape of `skroutz.gr/search?keyphrase=<query>` → extract the
top matching product row → return a PriceHit pointing at that product's
Skroutz page (which aggregates every merchant). Same credit cost as the
other two Greek adapters (1 Firecrawl credit per call).
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import List, Optional

from pydantic import BaseModel, Field

from app.services.integrations.firecrawl_client import FirecrawlClient
from app.services.integrations.perplexity_price_search_service import PriceHit
from app.utils.price_parsing import parse_price

logger = logging.getLogger(__name__)

SEARCH_URL_TEMPLATE = "https://www.skroutz.gr/search?keyphrase={query}"
MODULE_SLUG = "greek-marketplaces"

EXTRACTION_PROMPT = (
    "You are reading a Skroutz.gr search results page. Extract the FIRST "
    "matching product listing. Return `found`=false if no products are shown. "
    "For `product_url`, return the ABSOLUTE URL of the product detail page "
    "on skroutz.gr (e.g. https://www.skroutz.gr/s/123/some-product.html). "
    "If the page shows a direct merchant link on the first row (some product "
    "cards expose the cheapest merchant inline), use it for "
    "`cheapest_merchant_name` and `cheapest_merchant_url`; otherwise leave "
    "those null. Keep prices as strings with currency symbols intact."
)


class SkroutzSearchResult(BaseModel):
    """Fields Firecrawl extracts from skroutz.gr/search."""

    found: bool = Field(default=False, description="True if at least one product row is shown.")
    product_name: Optional[str] = Field(default=None, description="Top matching product display name.")
    product_url: Optional[str] = Field(
        default=None,
        description="Absolute URL of the top product's Skroutz detail page.",
    )
    best_price: Optional[str] = Field(
        default=None,
        description="Lowest price shown on the row, e.g. '79,90 €' or 'από €79,90'.",
    )
    merchant_count: Optional[int] = Field(
        default=None,
        description="Number of shops selling it (Skroutz displays 'από N καταστήματα' or similar).",
    )
    cheapest_merchant_name: Optional[str] = Field(
        default=None,
        description="Direct merchant name if the search row exposes one inline.",
    )
    cheapest_merchant_url: Optional[str] = Field(
        default=None,
        description="Direct merchant URL if the search row exposes one inline.",
    )
    currency: Optional[str] = Field(default="EUR")


class SkroutzAdapter:
    """Firecrawl-backed adapter for skroutz.gr."""

    def __init__(self, firecrawl_client: Optional[FirecrawlClient] = None) -> None:
        self.firecrawl = firecrawl_client or FirecrawlClient()

    @property
    def is_configured(self) -> bool:
        """Only Firecrawl is required — no separate Skroutz credentials."""
        return bool(self.firecrawl.api_key)

    async def search(
        self,
        query: str,
        *,
        user_id: str,
        workspace_id: Optional[str] = None,
        limit: int = 15,
    ) -> List[PriceHit]:
        # `limit` currently unused — the public search page returns its own
        # ranking and we only keep the top match. Kept in signature for
        # parity with the other adapters.
        del limit

        if not self.firecrawl.api_key:
            logger.debug("Skroutz: Firecrawl not configured, skipping.")
            return []

        url = SEARCH_URL_TEMPLATE.format(query=urllib.parse.quote_plus(query))
        result = await self.firecrawl.scrape(
            url=url,
            extraction_model=SkroutzSearchResult,
            user_id=user_id,
            workspace_id=workspace_id,
            extraction_prompt=EXTRACTION_PROMPT,
            use_javascript_render=True,  # skroutz.gr is JS-heavy
            only_main_content=True,
            module_slug=MODULE_SLUG,
            source_tag="skroutz",
        )

        if not result.success or not result.data or not result.data.found:
            return []

        data = result.data
        # Prefer a direct merchant URL when Skroutz exposes one on the search
        # row; fall back to the Skroutz product page (which itself lists every
        # merchant — the user is one click from a direct merchant URL).
        retailer_name = data.cheapest_merchant_name or "Skroutz"
        product_url = data.cheapest_merchant_url or data.product_url
        if not product_url:
            return []

        price, currency = parse_price(data.best_price, hint_currency=data.currency or "EUR")

        notes_parts = ["via Skroutz"]
        if data.merchant_count:
            notes_parts.append(
                f"{data.merchant_count} shop{'s' if data.merchant_count != 1 else ''}"
            )
        if not data.cheapest_merchant_url:
            notes_parts.append("aggregator URL (click through for merchants)")
        notes = " · ".join(notes_parts)

        return [
            PriceHit(
                retailer_name=retailer_name,
                product_url=product_url,
                price=float(price) if price is not None else None,
                currency=currency or "EUR",
                availability="in_stock",
                source="skroutz",
                verified=False,  # scrape, not first-party feed
                notes=notes,
            )
        ]


_singleton: Optional[SkroutzAdapter] = None


def get_skroutz_adapter() -> SkroutzAdapter:
    global _singleton
    if _singleton is None:
        _singleton = SkroutzAdapter()
    return _singleton
