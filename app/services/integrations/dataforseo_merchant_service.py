"""
DataForSEO Merchant API client — Google Shopping panel / merchant listings.

This closes the gap that Perplexity can't reach: Google Shopping's paid
placements that don't appear in organic web search. Each Shopping result
is a distinct merchant with their own price, all served via DataForSEO's
feed access.

Runs in parallel with Perplexity inside PerplexityPriceSearchService.
Results are merged into the unified response, each hit tagged with
`source='dataforseo'`.

Auth: Basic Auth (DATAFORSEO_LOGIN:DATAFORSEO_PASSWORD) — same credentials
used by the TypeScript `dataforseo-client.ts` SEO pipeline. Needs to be
added to MIVAA's systemd env via deploy.yml.

Pricing: ~$0.001 per 40 results (live mode). Negligible vs Perplexity.
"""

import base64
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


DATAFORSEO_BASE_URL = "https://api.dataforseo.com/v3"
PRODUCTS_ENDPOINT = "/merchant/google/products/live/advanced"
HTTP_TIMEOUT_S = 45.0
DATAFORSEO_COST_PER_40_RESULTS = 0.001  # $1 per 1M results, billed per 40


# ISO country → DataForSEO location_code. Covers the common markets.
# Full list at DataForSEO docs; this is just the hot set.
_LOCATION_CODES: Dict[str, int] = {
    "US": 2840,
    "GB": 2826, "UK": 2826,
    "DE": 2276,
    "FR": 2250,
    "IT": 2380,
    "ES": 2724,
    "NL": 2528,
    "BE": 2056,
    "PT": 2620,
    "GR": 2300,
    "BG": 2100,
    "RO": 2642,
    "CY": 2196,
    "PL": 2616,
    "CZ": 2203,
    "SK": 2703,
    "HU": 2348,
    "AT": 2040,
    "CH": 2756,
    "SE": 2752,
    "DK": 2208,
    "NO": 2578,
    "FI": 2246,
    "IE": 2372,
    "TR": 2792,
    "CA": 2124,
    "AU": 2036,
}


class MerchantHit(BaseModel):
    """One Google Shopping merchant listing."""
    retailer_name: str
    product_url: str
    price: float
    currency: str
    product_title: Optional[str] = None
    image_url: Optional[str] = None
    rating_value: Optional[float] = None
    rating_votes: Optional[int] = None
    shipping: Optional[str] = None


class MerchantSearchResult(BaseModel):
    success: bool
    hits: List[MerchantHit] = []
    raw_results_count: int = 0  # pre-dedupe count, for cost tracking
    credits_used: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None


class DataForSeoMerchantService:
    """Google Shopping via DataForSEO Merchant API."""

    def __init__(self) -> None:
        # Two accepted auth modes (DataForSEO docs support both):
        #   1. DATAFORSEO_BASE64 — pre-encoded "login:password" as base64. Preferred
        #      when the secret is already encoded in a vault; no extra encoding step.
        #   2. DATAFORSEO_LOGIN + DATAFORSEO_PASSWORD — plaintext, we base64 it.
        # The base64 form wins if both are set.
        pre_encoded = os.getenv("DATAFORSEO_BASE64") or ""
        login = os.getenv("DATAFORSEO_LOGIN") or ""
        password = os.getenv("DATAFORSEO_PASSWORD") or ""

        if pre_encoded:
            self.auth_header = f"Basic {pre_encoded}"
            self.configured = True
            self._auth_mode = "base64"
        elif login and password:
            self.auth_header = "Basic " + base64.b64encode(f"{login}:{password}".encode()).decode()
            self.configured = True
            self._auth_mode = "login+password"
        else:
            self.auth_header = ""
            self.configured = False
            self._auth_mode = "none"
            logger.warning(
                "⚠️ DataForSEO credentials not configured — set DATAFORSEO_BASE64 "
                "(preferred) or DATAFORSEO_LOGIN + DATAFORSEO_PASSWORD"
            )

    # ────────── Public API ──────────

    async def search_shopping(
        self,
        product_name: str,
        dimensions: Optional[str] = None,
        country_code: Optional[str] = None,
        limit: int = 20,
    ) -> MerchantSearchResult:
        """
        Query Google Shopping via DataForSEO's Merchant products endpoint.
        Returns merchants with visible prices only — DataForSEO doesn't
        surface 'quote-only' rows, so this is clean by construction.
        """
        if not self.configured:
            return MerchantSearchResult(success=False, error="DATAFORSEO credentials not configured")

        query = f"{product_name} {dimensions}".strip() if dimensions else product_name
        location_code = _LOCATION_CODES.get((country_code or "").upper(), 2840)  # default US
        language_code = "en"  # DataForSEO handles multilingual results internally

        body = [{
            "keyword": query,
            "location_code": location_code,
            "language_code": language_code,
            "depth": min(max(limit, 10), 40),  # DataForSEO returns up to 100; we cap at 40 for cost
        }]

        start = datetime.now(timezone.utc)
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
                resp = await client.post(
                    f"{DATAFORSEO_BASE_URL}{PRODUCTS_ENDPOINT}",
                    headers={"Authorization": self.auth_header, "Content-Type": "application/json"},
                    json=body,
                )
        except httpx.TimeoutException as e:
            return MerchantSearchResult(success=False, error=f"timeout: {e}")
        except Exception as e:
            return MerchantSearchResult(success=False, error=str(e))

        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        if resp.status_code != 200:
            return MerchantSearchResult(
                success=False,
                latency_ms=latency_ms,
                error=f"dataforseo HTTP {resp.status_code}: {resp.text[:300]}",
            )

        data = resp.json()
        hits, raw_count = self._parse_response(data, limit=limit)
        # Billed per 40 results returned (see DataForSEO pricing)
        cost_usd = (max(raw_count, 1) / 40) * DATAFORSEO_COST_PER_40_RESULTS
        platform_credits = max(1, int(round(cost_usd * 100)))

        return MerchantSearchResult(
            success=True,
            hits=hits,
            raw_results_count=raw_count,
            credits_used=platform_credits,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )

    # ────────── Internals ──────────

    def _parse_response(self, data: Dict[str, Any], limit: int) -> tuple[List[MerchantHit], int]:
        """
        DataForSEO response shape:
        {
          tasks: [{
            result: [{
              items: [{
                type, title, url, seller, price: {current, currency}, product_image_url, ...
              }]
            }]
          }]
        }
        """
        tasks = data.get("tasks") or []
        if not tasks:
            return [], 0
        result_container = (tasks[0].get("result") or [{}])[0]
        items = result_container.get("items") or []

        hits: List[MerchantHit] = []
        seen_sellers = set()
        raw_count = len(items)
        for item in items:
            # Only organic / paid product items have seller + price
            if item.get("type") not in ("shopping_serp", "shopping_paid", "shopping_organic"):
                continue

            seller = item.get("seller") or item.get("source") or item.get("domain")
            price_info = item.get("price") or {}
            price_value = price_info.get("current") or price_info.get("regular") or price_info.get("price")
            currency = price_info.get("currency") or "USD"
            url = item.get("url") or item.get("seller_url")
            title = item.get("title") or item.get("description")
            image_url = item.get("product_image_url") or item.get("image_url")

            if not seller or price_value is None or not url:
                continue

            # Dedupe by seller — one row per merchant, cheapest wins
            seller_key = str(seller).strip().lower()
            if seller_key in seen_sellers:
                continue
            seen_sellers.add(seller_key)

            try:
                price_f = float(price_value)
            except (TypeError, ValueError):
                continue

            hits.append(MerchantHit(
                retailer_name=str(seller),
                product_url=self._clean_url(str(url)),
                price=price_f,
                currency=str(currency),
                product_title=str(title) if title else None,
                image_url=str(image_url) if image_url else None,
                rating_value=(
                    float(item.get("rating", {}).get("value"))
                    if isinstance(item.get("rating"), dict) and item["rating"].get("value") is not None
                    else None
                ),
                rating_votes=(
                    int(item.get("rating", {}).get("votes_count"))
                    if isinstance(item.get("rating"), dict) and item["rating"].get("votes_count") is not None
                    else None
                ),
                shipping=(
                    item.get("shipping", {}).get("shipping_price", {}).get("current") if isinstance(item.get("shipping"), dict) else None
                ),
            ))
            if len(hits) >= limit:
                break

        hits.sort(key=lambda h: h.price)
        return hits, raw_count

    @staticmethod
    def _clean_url(url: str) -> str:
        # DataForSEO sometimes returns Google redirect URLs; strip the redirect wrapper.
        m = re.match(r"https?://(?:www\.)?google\.[a-z.]+/aclk\?.*?adurl=([^&]+)", url)
        if m:
            from urllib.parse import unquote
            return unquote(m.group(1))
        return url


_service: Optional[DataForSeoMerchantService] = None


def get_dataforseo_merchant_service() -> DataForSeoMerchantService:
    global _service
    if _service is None:
        _service = DataForSeoMerchantService()
    return _service
