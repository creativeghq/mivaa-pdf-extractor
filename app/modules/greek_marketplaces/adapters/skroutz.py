"""
Skroutz API adapter.

Skroutz is the first-party merchant index for the Greek market. One query
yields N rows — one per retailer selling the best-matching SKU, with
direct product URLs, live prices, availability, and shipping details.

Auth: OAuth2 client-credentials flow. Register the app at
https://developer.skroutz.gr/oauth/applications and set:

  SKROUTZ_CLIENT_ID
  SKROUTZ_CLIENT_SECRET
  SKROUTZ_BASE_URL        (optional, defaults to https://api.skroutz.gr)

Rate limit: 100 req/min per app. The service caller is expected to cap
concurrency; this adapter does not queue internally.
"""

from __future__ import annotations

import logging
import os
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

import httpx

from app.services.integrations.perplexity_price_search_service import PriceHit

logger = logging.getLogger(__name__)


class SkroutzAdapter:
    """OAuth2 client-credentials adapter for api.skroutz.gr."""

    DEFAULT_BASE_URL = "https://api.skroutz.gr"
    TOKEN_PATH = "/oauth2/token"
    ACCEPT_HEADER = "application/vnd.skroutz+json; version=3.1"
    HTTP_TIMEOUT_S = 15.0

    def __init__(self) -> None:
        self.client_id = os.getenv("SKROUTZ_CLIENT_ID") or ""
        self.client_secret = os.getenv("SKROUTZ_CLIENT_SECRET") or ""
        self.base_url = os.getenv("SKROUTZ_BASE_URL", self.DEFAULT_BASE_URL).rstrip("/")
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    async def search(self, query: str, limit: int = 15) -> List[PriceHit]:
        """Return up to `limit` Skroutz merchant hits for the query."""
        if not self.is_configured:
            logger.debug("Skroutz: credentials missing, skipping.")
            return []

        token = await self._ensure_token()
        if not token:
            return []

        async with httpx.AsyncClient(timeout=self.HTTP_TIMEOUT_S) as client:
            sku = await self._pick_best_sku(client, token, query)
            if not sku:
                return []
            shops = await self._fetch_shops(client, token, sku["id"])
            if not shops:
                return []

        product_name = sku.get("display_name") or sku.get("name") or query
        return self._build_hits(shops, product_name)[:limit]

    # ── internals ──────────────────────────────────────────────────────────

    async def _ensure_token(self) -> Optional[str]:
        """OAuth2 client-credentials flow. Cache token in memory until expiry."""
        now = time.monotonic()
        if self._token and now < self._token_expires_at - 30:
            return self._token

        try:
            async with httpx.AsyncClient(timeout=self.HTTP_TIMEOUT_S) as client:
                response = await client.post(
                    f"{self.base_url}{self.TOKEN_PATH}",
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "scope": "public",
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skroutz: token fetch failed (%s)", exc)
            return None

        token = payload.get("access_token")
        ttl = float(payload.get("expires_in") or 3600)
        if not token:
            logger.warning("Skroutz: token response missing access_token")
            return None

        self._token = token
        self._token_expires_at = now + ttl
        return token

    async def _pick_best_sku(
        self,
        client: httpx.AsyncClient,
        token: str,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Try /skus.json directly (works for SKU-name queries) and fall back to
        /search.json → /products/:id/skus.json for broader text queries.
        """
        headers = {"Authorization": f"Bearer {token}", "Accept": self.ACCEPT_HEADER}

        try:
            response = await client.get(
                f"{self.base_url}/skus/search",
                headers=headers,
                params={"q": query, "per": 5},
            )
            if response.status_code == 200:
                data = response.json().get("skus") or []
                if data:
                    return data[0]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skroutz: direct SKU search failed (%s), falling back to /search", exc)

        try:
            search_response = await client.get(
                f"{self.base_url}/search",
                headers=headers,
                params={"q": query, "per": 1},
            )
            search_response.raise_for_status()
            products = search_response.json().get("products") or []
            if not products:
                return None
            product_id = products[0]["id"]

            skus_response = await client.get(
                f"{self.base_url}/products/{product_id}/skus",
                headers=headers,
                params={"per": 1},
            )
            skus_response.raise_for_status()
            skus = skus_response.json().get("skus") or []
            return skus[0] if skus else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skroutz: SKU lookup failed (%s)", exc)
            return None

    async def _fetch_shops(
        self,
        client: httpx.AsyncClient,
        token: str,
        sku_id: int,
    ) -> List[Dict[str, Any]]:
        headers = {"Authorization": f"Bearer {token}", "Accept": self.ACCEPT_HEADER}
        try:
            response = await client.get(
                f"{self.base_url}/skus/{sku_id}/shops",
                headers=headers,
                params={"per": 25, "include_meta": "sku_reviews,sku_rating_breakdown"},
            )
            response.raise_for_status()
            return response.json().get("shops") or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skroutz: shops fetch failed for sku %s (%s)", sku_id, exc)
            return []

    def _build_hits(self, shops: List[Dict[str, Any]], product_name: str) -> List[PriceHit]:
        hits: List[PriceHit] = []
        for shop in shops:
            link = shop.get("link") or shop.get("url")
            name = shop.get("name") or shop.get("display_name")
            if not link or not name:
                continue

            price = self._decimal_to_float(shop.get("final_price") or shop.get("price"))
            original = self._decimal_to_float(shop.get("price_before_discount"))

            availability_raw = (shop.get("availability") or "").lower()
            availability: Optional[str] = None
            if "out of stock" in availability_raw or "sold out" in availability_raw:
                availability = "out_of_stock"
            elif availability_raw:
                availability = "in_stock"

            hits.append(
                PriceHit(
                    retailer_name=name,
                    product_url=link,
                    price=price,
                    original_price=original,
                    currency="EUR",
                    availability=availability,
                    source="skroutz",
                    verified=True,  # first-party feed, not a snippet guess
                )
            )
        return hits

    @staticmethod
    def _decimal_to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(Decimal(str(value)))
        except (ValueError, ArithmeticError):
            return None


_singleton: Optional[SkroutzAdapter] = None


def get_skroutz_adapter() -> SkroutzAdapter:
    global _singleton
    if _singleton is None:
        _singleton = SkroutzAdapter()
    return _singleton
