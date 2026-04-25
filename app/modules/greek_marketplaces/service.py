"""
Greek Marketplaces discovery service.

Invoked alongside Perplexity + DataForSEO by
`perplexity_price_search_service.search_prices()` when all of the
following are true:

  * country_code == "GR"
  * the `greek-marketplaces` module is enabled in the `modules` DB table

Returns `List[PriceHit]` merged + deduped by retailer domain so the
caller can slot them into its normal pipeline (URL prefilter, Firecrawl
verification, identity classifier).
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional
from urllib.parse import urlparse

from app.modules.greek_marketplaces.adapters import (
    BestpriceAdapter,
    ShopflixAdapter,
    SkroutzAdapter,
)
from app.modules.greek_marketplaces.adapters.bestprice import get_bestprice_adapter
from app.modules.greek_marketplaces.adapters.shopflix import get_shopflix_adapter
from app.modules.greek_marketplaces.adapters.skroutz import get_skroutz_adapter
from app.services.integrations.perplexity_price_search_service import PriceHit

logger = logging.getLogger(__name__)

MODULE_SLUG = "greek-marketplaces"
ADAPTER_NAMES = ("skroutz", "bestprice", "shopflix")


class GreekMarketplacesService:
    """Thin orchestrator that fans out across Skroutz + Bestprice + Shopflix."""

    def __init__(
        self,
        *,
        skroutz: Optional[SkroutzAdapter] = None,
        bestprice: Optional[BestpriceAdapter] = None,
        shopflix: Optional[ShopflixAdapter] = None,
    ) -> None:
        self.skroutz = skroutz or get_skroutz_adapter()
        self.bestprice = bestprice or get_bestprice_adapter()
        self.shopflix = shopflix or get_shopflix_adapter()

    async def search(
        self,
        query: str,
        *,
        country_code: Optional[str],
        user_id: str,
        workspace_id: Optional[str] = None,
        limit: int = 15,
    ) -> List[PriceHit]:
        if (country_code or "").upper() != "GR":
            return []

        tasks = [
            self.skroutz.search(query, user_id=user_id, workspace_id=workspace_id, limit=limit),
            self.bestprice.search(query, user_id=user_id, workspace_id=workspace_id),
            self.shopflix.search(query, user_id=user_id, workspace_id=workspace_id),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        hits: List[PriceHit] = []
        for idx, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning(
                    "greek-marketplaces: %s adapter failed (%s)",
                    ADAPTER_NAMES[idx],
                    result,
                )
                continue
            hits.extend(result or [])

        return self._dedupe_by_domain(hits)

    @staticmethod
    def _dedupe_by_domain(hits: List[PriceHit]) -> List[PriceHit]:
        """
        Keep the first hit per (retailer domain, source) pair. Skroutz runs
        first so its rows win over scraper fallbacks for the same retailer
        on the SAME source. But a merchant that appears on both Skroutz and
        Bestprice fanouts is kept twice with distinct source tags — the
        outer pipeline handles cross-source dedup with its own merge logic
        (greek-marketplaces > dataforseo > perplexity for the same domain).

        Without the source tie-breaker, a 30-merchant Skroutz fanout
        collapses to 1 row when Bestprice also surfaces the same merchant.
        """
        seen: set[tuple[str, str]] = set()
        deduped: List[PriceHit] = []
        for hit in hits:
            domain = urlparse(hit.product_url).netloc.lower().removeprefix("www.")
            if not domain:
                continue
            key = (domain, hit.source or "")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
        return deduped


_singleton: Optional[GreekMarketplacesService] = None


def get_greek_marketplaces_service() -> GreekMarketplacesService:
    global _singleton
    if _singleton is None:
        _singleton = GreekMarketplacesService()
    return _singleton
