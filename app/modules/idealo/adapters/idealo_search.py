"""
Idealo search-page adapter — Firecrawl scrape.

Idealo.* doesn't publish a public search API. We scrape the public results
page sorted by price ascending. Returns up to `limit` PriceHit rows.

URL format the public site uses:
    https://www.idealo.de/preisvergleich/MainSearchProductCategory.html?q=<query>

The result page has a sortBar with `?sortKey=Cheapest` (DE) — added below.
For non-DE locales the slug stays similar; scraping the rendered page bypasses
locale-specific deep links.

ToS caveat: Idealo's terms prohibit automated scraping at scale. Treat the
adapter the same as the Skroutz/Bestprice ones — fine at admin-triggered
volumes, not safe to scale without a commercial agreement.
"""

from __future__ import annotations

import logging
import urllib.parse
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field

from app.services.integrations.firecrawl_client import get_firecrawl_client
from app.services.integrations.perplexity_price_search_service import PriceHit

logger = logging.getLogger(__name__)


class IdealoSearchResult(BaseModel):
    """Pydantic schema Firecrawl extracts against the rendered page."""

    listings: List[dict] = Field(
        default_factory=list,
        description=(
            "Top product listings from the search results, sorted by price "
            "ascending. Each item has: product_name, retailer_name (the "
            "merchant — NOT 'idealo'), price (number), currency, product_url "
            "(the click-through to the merchant if visible, otherwise the "
            "Idealo product page)."
        ),
    )


_EXTRACTION_PROMPT = (
    "From the rendered Idealo search results page, return up to 5 product "
    "listings sorted by price ascending. For each listing extract:\n"
    "1. product_name — exactly as shown on the listing card.\n"
    "2. retailer_name — the merchant offering this price, NOT 'Idealo'. "
    "   Idealo is the aggregator; the merchant is what we actually want.\n"
    "3. price — current numeric price (no currency symbol).\n"
    "4. currency — ISO 4217 (EUR / GBP).\n"
    "5. product_url — direct merchant URL if visible on the card, otherwise "
    "   the Idealo product detail URL.\n"
    "Skip ad slots, sponsored placements, and listings without a visible price."
)


async def scrape_idealo_search(
    *,
    host: str,
    query: str,
    user_id: Optional[str],
    workspace_id: Optional[str],
    limit: int = 5,
) -> List[PriceHit]:
    """One Firecrawl scrape of the search page → up to `limit` PriceHits."""

    encoded_q = urllib.parse.quote(query)
    url = f"https://{host}/preisvergleich/MainSearchProductCategory.html?q={encoded_q}&sortKey=Cheapest"

    firecrawl = get_firecrawl_client()
    try:
        result = await firecrawl.scrape(
            url=url,
            extraction_model=IdealoSearchResult,
            user_id=user_id or "system",
            workspace_id=workspace_id,
            extraction_prompt=_EXTRACTION_PROMPT,
            use_javascript_render=True,  # Idealo result rendering is JS-driven
        )
    except Exception as e:
        logger.debug(f"idealo: Firecrawl scrape crashed for {url}: {e}")
        return []

    if not result.success or not result.data:
        return []

    listings = (result.data.listings or [])[:limit]
    today = datetime.now(timezone.utc).date().isoformat()
    hits: List[PriceHit] = []
    for raw in listings:
        try:
            price_val = float(raw.get("price")) if raw.get("price") is not None else None
            if price_val is None or price_val <= 0:
                continue
            retailer = (raw.get("retailer_name") or "").strip()
            if not retailer or retailer.lower() in ("idealo", "idealo.de", "idealo.it"):
                continue  # aggregator, not a merchant
            hits.append(PriceHit(
                retailer_name=retailer,
                product_url=raw.get("product_url") or url,
                price=price_val,
                currency=raw.get("currency") or "EUR",
                price_unit="piece",
                availability="in_stock",
                ships_from_abroad=False,
                is_quote_only=False,
                last_verified=today,
                notes=f"via Idealo ({host})",
                source="idealo",
                product_title=raw.get("product_name"),
            ))
        except Exception as e:
            logger.debug(f"idealo: row parse failed: {e}")
            continue
    return hits
