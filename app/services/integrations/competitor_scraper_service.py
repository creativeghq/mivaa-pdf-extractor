"""
Competitor Scraper Service

Thin wrapper over FirecrawlClient that adds price-parsing and the dict-shape
contract expected by price_monitoring_service.py. Keep logic minimal — the
generic scrape/retry/credit work lives in FirecrawlClient, numeric parsing
lives in utils.price_parsing.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from app.models.extraction import PriceExtraction
from app.services.integrations.firecrawl_client import get_firecrawl_client
from app.utils.price_parsing import parse_price

logger = logging.getLogger(__name__)


class CompetitorScraperService:
    """Scrape a single competitor product page and return a price dict."""

    def __init__(self) -> None:
        self.firecrawl = get_firecrawl_client()
        self.logger = logger

    async def scrape_competitor_price(
        self,
        url: str,
        product_name: str,
        user_id: str,
        workspace_id: Optional[str] = None,
        scraping_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Scrape competitor price from a URL.

        `scraping_config` accepts:
            use_javascript_render: bool  — wait for JS before extracting
            only_main_content: bool      — strip nav/footer/ads (default True)

        Returns dict with: success, price (Decimal|None), currency,
        availability, shipping_cost, product_name, raw_data, credits_used,
        latency_ms, scraped_at, error (if failed).
        """
        cfg = scraping_config or {}

        result = await self.firecrawl.scrape(
            url=url,
            extraction_model=PriceExtraction,
            user_id=user_id,
            workspace_id=workspace_id,
            extraction_prompt=(
                f"Extract the current price, currency, and availability status for the product "
                f"'{product_name}'. Use the main product price, not related items or strike-through prices."
            ),
            use_javascript_render=bool(cfg.get("use_javascript_render", False)),
            only_main_content=bool(cfg.get("only_main_content", True)),
        )

        if not result.success:
            return {
                "success": False,
                "error": result.error or "scrape failed",
                "credits_used": result.credits_used,
                "latency_ms": result.latency_ms,
            }

        extracted = result.data  # PriceExtraction | None
        hint_currency = extracted.currency if extracted else None
        raw_price = extracted.price if extracted else None
        amount, currency = parse_price(raw_price, hint_currency=hint_currency)

        if amount is None:
            self.logger.warning(f"Could not parse price from '{raw_price}' for {url}")

        return {
            "success": True,
            "price": amount,
            "currency": currency,
            "availability": (extracted.availability if extracted else None),
            "shipping_cost": (extracted.shipping_cost if extracted else None),
            "product_name": (extracted.product_name if extracted else None),
            "raw_data": result.raw_extract or {},
            "credits_used": result.credits_used,
            "latency_ms": result.latency_ms,
            "scraped_at": datetime.utcnow().isoformat(),
        }


_service: Optional[CompetitorScraperService] = None


def get_competitor_scraper_service() -> CompetitorScraperService:
    """Get singleton instance."""
    global _service
    if _service is None:
        _service = CompetitorScraperService()
    return _service
