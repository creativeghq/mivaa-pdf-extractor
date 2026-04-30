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

import asyncio
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
# DataForSEO Merchant products is async-only: task_post → poll task_get.
# No /live/advanced endpoint for this data set (returns HTTP 404).
TASK_POST_ENDPOINT = "/merchant/google/products/task_post"
TASK_GET_ENDPOINT = "/merchant/google/products/task_get/advanced/{task_id}"
# Priority 2 = "up to 60s turnaround", $0.002 per request. Priority 1 is cheaper
# ($0.001) but can take up to 45min — unusable for a live API call.
TASK_PRIORITY = 2
HTTP_TIMEOUT_S = 15.0      # per HTTP call; polling loop has its own budget
MAX_POLL_SECONDS = 45.0    # overall budget — if task not ready by then, we bail with empty
POLL_INTERVAL_S = 3.0      # gap between task_get polls


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
    original_price: Optional[float] = None  # "was" / strikethrough price on the Shopping card
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
        # Default country is GR (platform's primary market). The /discover route
        # resolves this from user_profiles.location_country_code first, then from
        # the free-text location field (e.g. "Greece" → "GR"), so by the time the
        # code reaches here the country_code is almost always set.
        location_code = _LOCATION_CODES.get((country_code or "").upper(), _LOCATION_CODES["GR"])
        language_code = "en"  # DataForSEO handles multilingual results internally

        start = datetime.now(timezone.utc)

        # Step 1: post task
        headers = {"Authorization": self.auth_header, "Content-Type": "application/json"}
        task_body = [{
            "keyword": query,
            "location_code": location_code,
            "language_code": language_code,
            "depth": min(max(limit, 10), 40),
            "priority": TASK_PRIORITY,
        }]

        task_id: Optional[str] = None
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
                post_resp = await client.post(
                    f"{DATAFORSEO_BASE_URL}{TASK_POST_ENDPOINT}",
                    headers=headers,
                    json=task_body,
                )
                if post_resp.status_code != 200:
                    return MerchantSearchResult(
                        success=False,
                        latency_ms=int((datetime.now(timezone.utc) - start).total_seconds() * 1000),
                        error=f"dataforseo task_post HTTP {post_resp.status_code}: {post_resp.text[:300]}",
                    )
                post_json = post_resp.json()
                task = (post_json.get("tasks") or [{}])[0]
                if task.get("status_code") and int(task["status_code"]) >= 40000:
                    return MerchantSearchResult(
                        success=False,
                        latency_ms=int((datetime.now(timezone.utc) - start).total_seconds() * 1000),
                        error=f"dataforseo task rejected: {task.get('status_message')}",
                    )
                task_id = task.get("id")
                if not task_id:
                    return MerchantSearchResult(
                        success=False,
                        error="dataforseo task_post returned no task id",
                    )

                # Step 2: poll task_get until ready or MAX_POLL_SECONDS budget exhausted
                deadline = datetime.now(timezone.utc).timestamp() + MAX_POLL_SECONDS
                get_url = f"{DATAFORSEO_BASE_URL}{TASK_GET_ENDPOINT.format(task_id=task_id)}"
                while datetime.now(timezone.utc).timestamp() < deadline:
                    await asyncio.sleep(POLL_INTERVAL_S)
                    get_resp = await client.get(get_url, headers=headers)
                    if get_resp.status_code != 200:
                        continue
                    get_json = get_resp.json()
                    got_task = (get_json.get("tasks") or [{}])[0]
                    status_code = int(got_task.get("status_code") or 0)
                    # 20000 = OK, completed. 40602 = not ready yet.
                    if status_code == 20000:
                        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
                        hits, raw_count = self._parse_response(get_json, limit=limit)
                        cost_usd = float(get_json.get("cost") or post_json.get("cost") or 0.002)
                        platform_credits = max(1, int(round(cost_usd * 100)))
                        return MerchantSearchResult(
                            success=True,
                            hits=hits,
                            raw_results_count=raw_count,
                            credits_used=platform_credits,
                            latency_ms=latency_ms,
                            cost_usd=cost_usd,
                        )
                    if status_code >= 40000 and status_code != 40602:
                        return MerchantSearchResult(
                            success=False,
                            error=f"dataforseo task failed: {got_task.get('status_message')} (status {status_code})",
                        )
                    # else 40602 "Task In Queue" — keep polling

                # Polling budget exhausted
                return MerchantSearchResult(
                    success=False,
                    latency_ms=int((datetime.now(timezone.utc) - start).total_seconds() * 1000),
                    error=f"dataforseo task not ready within {MAX_POLL_SECONDS}s (task_id={task_id})",
                )
        except httpx.TimeoutException as e:
            return MerchantSearchResult(success=False, error=f"timeout: {e}")
        except Exception as e:
            return MerchantSearchResult(success=False, error=str(e))

    # ────────── Internals ──────────

    def _parse_response(self, data: Dict[str, Any], limit: int) -> tuple[List[MerchantHit], int]:
        """
        DataForSEO Merchant /products/task_get/advanced response shape:
        {
          tasks: [{
            result: [{
              items: [
                {
                  "type": "google_shopping_serp",
                  "title": "IKEA Billy Bookcase",
                  "price": 79,                              # flat number, NOT nested
                  "currency": "USD",                        # flat, NOT under price.
                  "seller": null | "<merchant name>",       # often null on SERP items
                  "domain": null | "<domain>",              # often null on SERP items
                  "shopping_url": "https://google.com/search?ibp=oshop&...",  # Google redirect
                  "product_images": [{"image_url":"..."}, ...],
                  "product_rating": {"value": 4.5, "votes_count": 1000},
                  "shop_rating": {...},
                  ...
                },
                ...
              ]
            }]
          }]
        }

        The SERP is PRODUCT-centric — each item is a product variant with its best
        price surfaced. To get per-merchant prices we'd need a second call to
        /merchant/google/sellers with each product_id. For now we expose what SERP
        gives: the product itself, price, and link to Google's Shopping detail page.
        """
        tasks = data.get("tasks") or []
        if not tasks:
            return [], 0
        result_container = (tasks[0].get("result") or [{}])[0]
        items = result_container.get("items") or []

        hits: List[MerchantHit] = []
        seen_keys = set()
        raw_count = len(items)
        for item in items:
            if item.get("type") != "google_shopping_serp":
                continue

            price_value = item.get("price")
            currency = item.get("currency") or "USD"
            if price_value is None:
                continue

            # Retailer name preference: seller > domain > derived from URL > fallback
            seller = item.get("seller") or item.get("domain")
            shopping_url = item.get("shopping_url") or ""
            if not seller and shopping_url:
                # Fall back to "Google Shopping" — the shopping_url is a google.com
                # redirect anyway; the UI can label it as an aggregator entry.
                seller = "Google Shopping"
            if not seller:
                continue

            url = shopping_url or item.get("url") or ""
            if not url:
                continue

            # Dedupe by (seller, product title) so we don't collapse distinct products
            # from the same merchant when they legitimately have different prices.
            title = item.get("title") or ""
            key = f"{str(seller).strip().lower()}::{title[:80].lower()}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            try:
                price_f = float(price_value)
            except (TypeError, ValueError):
                continue

            # old_price: strikethrough "was" price on the Shopping card. Nullable.
            old_price_raw = item.get("old_price")
            try:
                original_price_f = float(old_price_raw) if old_price_raw is not None else None
            except (TypeError, ValueError):
                original_price_f = None
            # Sanity: only keep if it's actually higher than current
            if original_price_f is not None and original_price_f <= price_f:
                original_price_f = None

            # product_rating is a dict with value + votes_count
            rating = item.get("product_rating") or {}
            rating_val = rating.get("value") if isinstance(rating, dict) else None
            rating_votes = rating.get("votes_count") if isinstance(rating, dict) else None

            # product_images is a list of dicts; grab the first image_url
            imgs = item.get("product_images") or []
            image_url = imgs[0].get("image_url") if isinstance(imgs, list) and imgs and isinstance(imgs[0], dict) else None

            hits.append(MerchantHit(
                retailer_name=str(seller),
                product_url=self._clean_url(str(url)),
                price=price_f,
                original_price=original_price_f,
                currency=str(currency),
                product_title=str(title) if title else None,
                image_url=str(image_url) if image_url else None,
                rating_value=float(rating_val) if rating_val is not None else None,
                rating_votes=int(rating_votes) if rating_votes is not None else None,
                shipping=None,
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
