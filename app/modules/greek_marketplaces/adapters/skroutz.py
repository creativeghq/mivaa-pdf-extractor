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

Flow: one scrape of
`skroutz.gr/search?keyphrase=<query>&order_by=pricevat&order_dir=asc`
→ Skroutz returns the cheapest offer first, so the top row IS the
lowest price for the query. We extract that single row and return a
PriceHit pointing at the product's Skroutz page (which aggregates every
merchant). Same credit cost as the other two Greek adapters (1
Firecrawl credit per call).

Why sort by price asc:
* Lowest price is what price-monitoring is trying to track.
* Skroutz's "find the right product" step often returns multiple
  variants; sorting by price puts the same-product cheapest offer at
  the top, which is the canonical row we want to extract.
* Firecrawl only sees the first viewport in many JS-rendered pages —
  sorting ensures the cheapest offer is always in the first viewport,
  reducing the chance Skroutz's lazy-load defeats us.
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import List, Optional

from pydantic import BaseModel, Field

from app.modules.greek_marketplaces.facet_filter import adaptive_marketplace_query, matches_facets
from app.modules.greek_marketplaces.match_filter import is_plausible_match
from app.services.integrations.firecrawl_client import FirecrawlClient
from app.services.integrations.perplexity_price_search_service import PriceHit
from app.services.integrations.product_identity_service import QueryFacets
from app.utils.price_parsing import parse_price

logger = logging.getLogger(__name__)

SEARCH_URL_TEMPLATE = (
    "https://www.skroutz.gr/search?keyphrase={query}"
    "&order_by=pricevat&order_dir=asc"
)
MODULE_SLUG = "greek-marketplaces"

EXTRACTION_PROMPT = (
    "You are reading a Skroutz.gr search results page sorted by lowest "
    "price (order_by=pricevat). Extract the FIRST matching product listing "
    "— since results are sorted asc by price-with-VAT, this is the cheapest "
    "available offer for the query. Return `found`=false if no products are "
    "shown. For `product_url`, return the ABSOLUTE URL of the product detail "
    "page on skroutz.gr (e.g. https://www.skroutz.gr/s/123/some-product.html). "
    "If the page shows a direct merchant link on the first row (some product "
    "cards expose the cheapest merchant inline), use it for "
    "`cheapest_merchant_name` and `cheapest_merchant_url`; otherwise leave "
    "those null. Keep prices as strings with currency symbols intact."
)

PRODUCT_PAGE_PROMPT = (
    "You are reading a Skroutz.gr product detail page. Extract EVERY visible "
    "merchant offer (the 'Καταστήματα' / shop list section). For each row "
    "return: shop name, the DIRECT merchant URL on the shop's own domain "
    "(NOT a skroutz.gr URL — Skroutz shows a 'Επίσκεψη καταστήματος' / "
    "'Visit shop' button that links externally; that is the URL we want), "
    "the visible price including VAT and currency symbol, and the "
    "availability label as shown. Return found=true and the merchants list "
    "in their natural order (Skroutz sorts by price asc by default). If no "
    "merchant rows are visible (rare — happens on out-of-stock products), "
    "return found=false with an empty merchants array."
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


class SkroutzMerchantOffer(BaseModel):
    """A single merchant row inside a Skroutz product detail page."""

    merchant_name: Optional[str] = Field(default=None, description="Shop name.")
    merchant_url: Optional[str] = Field(
        default=None,
        description="Absolute URL pointing to the shop's product page (NOT skroutz.gr).",
    )
    price: Optional[str] = Field(default=None, description="Shown price string with currency, e.g. '50,00 €'.")
    availability: Optional[str] = Field(
        default=None,
        description="Availability label as shown ('Διαθέσιμο', 'Άμεση παράδοση', 'Σε αναμονή', etc.).",
    )


class SkroutzProductPageResult(BaseModel):
    """All visible merchant rows on a Skroutz product detail page."""

    found: bool = Field(default=False, description="True if at least one merchant row is rendered.")
    product_name: Optional[str] = Field(default=None, description="Canonical product title.")
    merchants: List[SkroutzMerchantOffer] = Field(
        default_factory=list,
        description="One entry per visible merchant offer, in the page's natural order (price asc).",
    )


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
        facets: Optional[QueryFacets] = None,
    ) -> List[PriceHit]:
        if not self.firecrawl.api_key:
            logger.debug("Skroutz: Firecrawl not configured, skipping.")
            return []

        # Adaptive query: tight "{BRAND} {SKU}" when facets carry both.
        adaptive_query = adaptive_marketplace_query(query=query, facets=facets)

        # Step 1 — find the matching product on the search results page.
        search_result = await self._scrape_search(adaptive_query, user_id=user_id, workspace_id=workspace_id)
        if not search_result:
            return []

        data, product_url = search_result

        # Facet-aware filter on the cheapest-product candidate. If the SKU
        # doesn't match what the user actually wants, skip the entire fanout
        # — we'd just produce family rows that the classifier drops anyway.
        if not matches_facets(facets=facets, candidate_url=product_url, candidate_name=data.product_name):
            logger.info(
                "Skroutz: facet mismatch (sku=%s, type=%s) — query=%r, url=%s",
                facets.sku_tokens if facets else None,
                facets.product_type if facets else None,
                query, product_url,
            )
            return []

        # Step 2 — when Skroutz reports more than 1 shop, fan out: scrape the
        # product page itself and emit one PriceHit per visible merchant. This
        # is the difference between "Skroutz says 50€" and "5 actual retailers
        # showing 50/55/60/65/70€ each with a direct URL". The extra Firecrawl
        # credit is well spent — it turns one row into N.
        product_page_url = data.product_url or product_url
        wants_fanout = (
            (data.merchant_count or 0) > 1
            and product_page_url
            and "skroutz.gr" in (product_page_url or "")
        )
        if wants_fanout:
            merchants = await self._scrape_product_page(
                product_page_url, query=query,
                user_id=user_id, workspace_id=workspace_id,
            )
            if merchants:
                return self._fanout_hits(
                    merchants=merchants,
                    fallback_currency=data.currency or "EUR",
                    merchant_count=data.merchant_count,
                    product_name=data.product_name,
                    query=query,
                    limit=limit,
                )

        # Fallback: emit a single hit (legacy behaviour). This is what runs
        # when merchant_count <= 1, when product_url isn't on skroutz.gr,
        # or when product-page extraction returned nothing usable.
        retailer_name = data.cheapest_merchant_name or "Skroutz"
        single_url = data.cheapest_merchant_url or product_url
        if not is_plausible_match(query, single_url, data.product_name):
            logger.info(
                "Skroutz: dropped likely false positive — query=%r, url=%s",
                query, single_url,
            )
            return []

        price, currency = parse_price(data.best_price, hint_currency=data.currency or "EUR")
        notes_parts = ["via Skroutz"]
        if data.merchant_count:
            notes_parts.append(
                f"{data.merchant_count} shop{'s' if data.merchant_count != 1 else ''}"
            )
        if not data.cheapest_merchant_url:
            notes_parts.append("aggregator URL (click through for merchants)")
        return [
            PriceHit(
                retailer_name=retailer_name,
                product_url=single_url,
                price=float(price) if price is not None else None,
                currency=currency or "EUR",
                availability="in_stock",
                source="skroutz",
                verified=False,
                notes=" · ".join(notes_parts),
            )
        ]

    async def _scrape_search(
        self,
        query: str,
        *,
        user_id: str,
        workspace_id: Optional[str],
    ) -> Optional[tuple[SkroutzSearchResult, str]]:
        """Scrape skroutz.gr/search and return (data, product_url) or None."""
        url = SEARCH_URL_TEMPLATE.format(query=urllib.parse.quote_plus(query))
        # skroutz.gr is JS-heavy. Hydration occasionally finishes after the
        # Firecrawl render snapshot — retry once on success-but-empty so a
        # transient miss doesn't cost us a real result.
        result = await self.firecrawl.scrape(
            url=url, extraction_model=SkroutzSearchResult,
            user_id=user_id, workspace_id=workspace_id,
            extraction_prompt=EXTRACTION_PROMPT,
            use_javascript_render=True, only_main_content=True,
            module_slug=MODULE_SLUG, source_tag="skroutz",
        )
        if result.success and (not result.data or not result.data.found):
            logger.info("Skroutz: empty hydration, retrying once.")
            result = await self.firecrawl.scrape(
                url=url, extraction_model=SkroutzSearchResult,
                user_id=user_id, workspace_id=workspace_id,
                extraction_prompt=EXTRACTION_PROMPT,
                use_javascript_render=True, only_main_content=True,
                module_slug=MODULE_SLUG, source_tag="skroutz",
            )
        if not result.success or not result.data or not result.data.found:
            return None
        product_url = result.data.cheapest_merchant_url or result.data.product_url
        if not product_url:
            return None
        return result.data, product_url

    async def _scrape_product_page(
        self,
        product_page_url: str,
        *,
        query: str,
        user_id: str,
        workspace_id: Optional[str],
    ) -> List[SkroutzMerchantOffer]:
        """Scrape the Skroutz product page to extract every merchant offer."""
        try:
            result = await self.firecrawl.scrape(
                url=product_page_url,
                extraction_model=SkroutzProductPageResult,
                user_id=user_id, workspace_id=workspace_id,
                extraction_prompt=PRODUCT_PAGE_PROMPT,
                use_javascript_render=True, only_main_content=True,
                module_slug=MODULE_SLUG, source_tag="skroutz_product_page",
            )
        except Exception as e:
            logger.warning("Skroutz product-page scrape failed (%s): %s", product_page_url, e)
            return []
        if not result.success or not result.data or not result.data.found:
            return []
        # Page-level identity safeguard — if Skroutz served us a featured
        # product instead of the queried one, the page title won't carry the
        # query tokens. Drops the whole fanout in that case.
        if not is_plausible_match(query, product_page_url, result.data.product_name):
            logger.info(
                "Skroutz: product page failed plausibility check — query=%r, url=%s",
                query, product_page_url,
            )
            return []
        return [m for m in result.data.merchants if m.merchant_url and m.merchant_name]

    @staticmethod
    def _fanout_hits(
        *,
        merchants: List[SkroutzMerchantOffer],
        fallback_currency: str,
        merchant_count: Optional[int],
        product_name: Optional[str],
        query: str,
        limit: int,
    ) -> List[PriceHit]:
        """Convert per-merchant offers into a list of PriceHit rows."""
        del product_name  # reserved for future use (e.g. attaching to notes)
        hits: List[PriceHit] = []
        for offer in merchants[: max(limit, 1)]:
            price, currency = parse_price(offer.price, hint_currency=fallback_currency)
            avail = (offer.availability or "").lower()
            if any(tok in avail for tok in ("εκτός", "out", "unavail", "not available")):
                availability = "out_of_stock"
            elif any(tok in avail for tok in ("διαθέσιμ", "available", "in stock", "άμεσ")):
                availability = "in_stock"
            else:
                availability = "in_stock"
            hits.append(
                PriceHit(
                    retailer_name=offer.merchant_name or "Skroutz merchant",
                    product_url=offer.merchant_url,
                    price=float(price) if price is not None else None,
                    currency=currency or fallback_currency,
                    availability=availability,
                    source="skroutz",
                    verified=False,
                    notes="via Skroutz",
                )
            )
        if merchant_count and merchant_count > len(hits):
            logger.info(
                "Skroutz fanout: extracted %d/%d merchants for %r — older Firecrawl snapshot may be missing late-hydrated rows.",
                len(hits), merchant_count, query,
            )
        return hits


_singleton: Optional[SkroutzAdapter] = None


def get_skroutz_adapter() -> SkroutzAdapter:
    global _singleton
    if _singleton is None:
        _singleton = SkroutzAdapter()
    return _singleton
