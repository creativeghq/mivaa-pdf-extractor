"""
Claude-powered price discovery via Anthropic's built-in web_search tool.

Used for:
- First-run price discovery when a user enables monitoring for a product
- Periodic re-discovery (6h throttle) to surface new retailers + drift
- Public /api/v1/prices/lookup when the caller passes `search_query` instead of `url`

Why Claude web_search over Firecrawl for this:
- Firecrawl can't scrape Google/Bing (consent walls, bot detection)
- Claude's built-in web_search handles the blocking upstream + synthesizes results
- Same ANTHROPIC_API_KEY we already use platform-wide — no new vendor

Design:
- Model: claude-haiku-4-5 (fast, cheap for factual search)
- Tools: web_search_20250305 (Anthropic built-in) + a forced-output `submit_price_results` tool
  that Claude must call at the end to emit structured JSON — no regex parsing.
- Credit logging via AICallLogger.log_claude_call (handles tokens + per-token billing).
  Web search tool calls bill at ~$0.01/search; we count them via response.usage.server_tool_use.
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5"
WEB_SEARCH_BETA_HEADER = "web-search-2025-03-05"
WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
MAX_WEB_SEARCHES = 5
MAX_TOKENS = 4096
HTTP_TIMEOUT_S = 90.0
THROTTLE_HOURS = 6

# Haiku 4.5 pricing (from app/config/ai_pricing.py — ballpark for logging).
# Real debit happens via AICallLogger; these are just fallback estimates.
HAIKU_INPUT_PER_1K = 0.0008
HAIKU_OUTPUT_PER_1K = 0.004
WEB_SEARCH_PER_CALL = 0.010  # $10 / 1000 web searches


# ────────────────────────────────────────────────────────────────────────────
# Models
# ────────────────────────────────────────────────────────────────────────────


class PriceHit(BaseModel):
    """Single retailer result returned by Claude for a product."""
    retailer_name: str = Field(..., description="Retailer display name, e.g. 'Topps Tiles', 'Mandarin Stone'.")
    product_url: str = Field(..., description="Direct product page URL. Must be a product page, not a search results page.")
    price: float = Field(..., description="Numeric price.")
    currency: str = Field(..., description="ISO 4217 currency code (USD, EUR, GBP, etc.)")
    availability: Optional[str] = Field(default=None, description="in_stock | out_of_stock | limited | unknown")
    notes: Optional[str] = Field(default=None, description="Optional short note (e.g. 'per box', 'shipping included').")


class PriceSearchResult(BaseModel):
    """Wrapper for results + metadata."""
    success: bool
    hits: List[PriceHit] = []
    credits_used: int = 0
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    web_searches: int = 0
    cost_usd: float = 0.0
    throttled: bool = False
    throttle_until: Optional[datetime] = None
    error: Optional[str] = None


# ────────────────────────────────────────────────────────────────────────────
# Service
# ────────────────────────────────────────────────────────────────────────────


class ClaudePriceSearchService:
    """Anthropic web_search-powered price discovery."""

    def __init__(self) -> None:
        self.api_key = os.getenv("ANTHROPIC_API_KEY") or ""
        self.supabase = get_supabase_client()
        if not self.api_key:
            logger.warning("⚠️ ANTHROPIC_API_KEY not configured — Claude price search disabled")

    # ────────── Public API ──────────

    async def search_prices(
        self,
        product_name: str,
        dimensions: Optional[str] = None,
        country_code: Optional[str] = None,
        limit: int = 10,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> PriceSearchResult:
        """
        Run one web-search pass for the given product. Stateless — caller handles
        throttle check and DB persistence.

        Args:
            product_name: Canonical product name (e.g. "Ferrara Beige").
            dimensions: Optional size spec (e.g. "120x60 cm"). Appended to the query.
            country_code: ISO-3166 alpha-2 (e.g. "GB", "GR"). Biases results but does
                not restrict them — caller requested global coverage with local
                preference.
            limit: Max retailers to return (hard cap 10 unless caller knows better).
            user_id / workspace_id: For credit debiting via AICallLogger.

        Returns:
            PriceSearchResult with hits list + usage metadata.
        """
        if not self.api_key:
            return PriceSearchResult(success=False, error="ANTHROPIC_API_KEY not configured")

        limit = max(1, min(limit, 10))
        prompt = self._build_prompt(product_name, dimensions, country_code, limit)
        tools = self._build_tools(limit)

        start = datetime.now(timezone.utc)
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
                resp = await client.post(
                    ANTHROPIC_BASE_URL,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "anthropic-beta": WEB_SEARCH_BETA_HEADER,
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "max_tokens": MAX_TOKENS,
                        "tools": tools,
                        "tool_choice": {"type": "auto"},
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
        except httpx.TimeoutException as e:
            return PriceSearchResult(success=False, error=f"timeout: {e}")
        except Exception as e:
            return PriceSearchResult(success=False, error=str(e))

        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)

        if resp.status_code != 200:
            return PriceSearchResult(
                success=False,
                latency_ms=latency_ms,
                error=f"anthropic HTTP {resp.status_code}: {resp.text[:300]}",
            )

        data = resp.json()
        hits = self._extract_structured_hits(data)

        usage = data.get("usage", {}) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        server_tool_use = usage.get("server_tool_use") or {}
        web_searches = int(server_tool_use.get("web_search_requests", 0) or 0)

        cost_usd = (
            (input_tokens / 1000) * HAIKU_INPUT_PER_1K
            + (output_tokens / 1000) * HAIKU_OUTPUT_PER_1K
            + web_searches * WEB_SEARCH_PER_CALL
        )
        # Platform credits: 1 credit = $0.01 (same convention as Firecrawl path).
        platform_credits = int(round(cost_usd * 100))

        # Log + debit via the existing pipeline so this shows up in admin metrics
        # alongside every other Claude call. Non-blocking: log failure does not
        # invalidate the search result.
        await self._log_usage(
            user_id=user_id,
            workspace_id=workspace_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            web_searches=web_searches,
            cost_usd=cost_usd,
            platform_credits=platform_credits,
            latency_ms=latency_ms,
            product_name=product_name,
            hits_count=len(hits),
        )

        return PriceSearchResult(
            success=True,
            hits=hits,
            credits_used=platform_credits,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            web_searches=web_searches,
            cost_usd=cost_usd,
        )

    def check_throttle(
        self, last_claude_search_at: Optional[datetime], force_refresh: bool = False
    ) -> Tuple[bool, Optional[datetime]]:
        """
        Return (is_throttled, throttle_until). Caller decides whether to skip the call.
        Admin `force_refresh=True` always returns (False, None).
        """
        if force_refresh or last_claude_search_at is None:
            return False, None
        expires_at = last_claude_search_at + timedelta(hours=THROTTLE_HOURS)
        if datetime.now(timezone.utc) < expires_at:
            return True, expires_at
        return False, None

    # ────────── Internals ──────────

    def _build_prompt(
        self, product_name: str, dimensions: Optional[str], country_code: Optional[str], limit: int
    ) -> str:
        size_part = f" {dimensions}" if dimensions else ""
        local_hint = (
            f"The user is in {country_code}. PREFER retailers that ship to or list prices in {country_code}, "
            "but DO include international retailers if local coverage is thin — do not silently drop global results."
            if country_code
            else "Do not restrict to any specific country — include global retailer results."
        )

        return (
            f"I need current prices for the product: \"{product_name}{size_part}\".\n\n"
            f"Use web search to find up to {limit} distinct retailers with visible prices for this product. "
            "Prioritize in this order: (1) Google Shopping / Sponsored listings, (2) dedicated retailer "
            "sites (their own domain), (3) marketplaces (Amazon, eBay), (4) price-comparison aggregators "
            "ONLY as a last resort. Skip blog posts, forums, manufacturer PDFs.\n\n"
            f"{local_hint}\n\n"
            "REQUIREMENTS for each entry:\n"
            "  1. A URL that leads to the specific product (product page, listing page, or Google Shopping "
            "     result page is fine — as long as the price is tied to THAT product, not a category).\n"
            "  2. A visible numeric price with a clear currency. Skip 'price on request' / 'trade only' listings.\n"
            "  3. Prefer the exact size match when dimensions are specified; for close-but-not-exact sizes "
            "     (±20%), include them and note the size mismatch in the `notes` field.\n"
            "  4. If the same retailer shows up multiple times, keep only the cheapest matching variant.\n\n"
            "Be inclusive, not strict. It is better to return a marketplace listing with a real price than "
            "to return nothing. For niche B2B products (stone, tiles, fabric, trade materials), include "
            "trade-focused retailers even if prices are wholesale. Only return an empty array if you truly "
            "found zero prices anywhere on the web.\n\n"
            "After your web searches, call the `submit_price_results` tool EXACTLY ONCE with the final "
            "deduplicated list. Do NOT fabricate URLs or prices — only submit what you actually saw."
        )

    def _build_tools(self, limit: int) -> List[Dict[str, Any]]:
        """Anthropic web_search + our forced-output tool."""
        return [
            {"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search", "max_uses": MAX_WEB_SEARCHES},
            {
                "name": "submit_price_results",
                "description": (
                    "Submit the final structured list of retailers and prices. Call this exactly once "
                    "at the end of your research, after all web searches. Empty list is valid if nothing "
                    "qualifying was found."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "retailers": {
                            "type": "array",
                            "maxItems": limit,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "retailer_name": {"type": "string"},
                                    "product_url": {"type": "string"},
                                    "price": {"type": "number"},
                                    "currency": {"type": "string", "description": "ISO 4217 code."},
                                    "availability": {
                                        "type": "string",
                                        "enum": ["in_stock", "out_of_stock", "limited", "unknown"],
                                    },
                                    "notes": {"type": "string"},
                                },
                                "required": ["retailer_name", "product_url", "price", "currency"],
                            },
                        },
                    },
                    "required": ["retailers"],
                },
            },
        ]

    def _extract_structured_hits(self, response: Dict[str, Any]) -> List[PriceHit]:
        """Pull the submit_price_results tool_use block. Ignore any text commentary."""
        for block in response.get("content", []) or []:
            if block.get("type") == "tool_use" and block.get("name") == "submit_price_results":
                raw_list = (block.get("input") or {}).get("retailers") or []
                hits: List[PriceHit] = []
                seen_urls = set()
                for raw in raw_list:
                    try:
                        hit = PriceHit(**raw)
                    except Exception as e:
                        logger.debug(f"Skipped malformed retailer entry: {e}")
                        continue
                    url_key = self._normalize_url(hit.product_url)
                    if url_key in seen_urls:
                        continue
                    seen_urls.add(url_key)
                    hits.append(hit)
                return hits
        # Fallback: if the model forgot to call the tool but dumped JSON in text.
        return self._extract_from_text(response)

    @staticmethod
    def _normalize_url(url: str) -> str:
        return re.sub(r"[?#].*$", "", (url or "").strip().lower().rstrip("/"))

    def _extract_from_text(self, response: Dict[str, Any]) -> List[PriceHit]:
        for block in response.get("content", []) or []:
            if block.get("type") != "text":
                continue
            text = block.get("text") or ""
            # Look for a fenced JSON block as a last resort.
            m = re.search(r"```(?:json)?\s*(\{[\s\S]+?\}|\[[\s\S]+?\])\s*```", text)
            if not m:
                continue
            try:
                parsed = json.loads(m.group(1))
                if isinstance(parsed, dict):
                    parsed = parsed.get("retailers") or []
                return [PriceHit(**r) for r in parsed if isinstance(r, dict)]
            except Exception:
                return []
        return []

    async def _log_usage(
        self,
        *,
        user_id: Optional[str],
        workspace_id: Optional[str],
        input_tokens: int,
        output_tokens: int,
        web_searches: int,
        cost_usd: float,
        platform_credits: int,
        latency_ms: int,
        product_name: str,
        hits_count: int,
    ) -> None:
        """Insert into ai_usage_logs directly. Bypasses log_claude_call because
        that helper is wired for Claude *without* server tools — it doesn't
        know about web_search call counts."""
        try:
            self.supabase.client.table("ai_usage_logs").insert(
                {
                    "user_id": user_id,
                    "workspace_id": workspace_id,
                    "operation_type": "price_search",
                    "model_name": MODEL,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_cost_usd": round((input_tokens / 1000) * HAIKU_INPUT_PER_1K, 6),
                    "output_cost_usd": round((output_tokens / 1000) * HAIKU_OUTPUT_PER_1K, 6),
                    "total_cost_usd": round(cost_usd, 6),
                    "credits_debited": platform_credits,
                    "metadata": {
                        "api_provider": "anthropic",
                        "tool": "web_search_20250305",
                        "web_searches": web_searches,
                        "product_name": product_name,
                        "hits_count": hits_count,
                        "latency_ms": latency_ms,
                    },
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            ).execute()
        except Exception as e:
            logger.warning(f"Failed to log Claude price_search usage: {e}")


_service: Optional[ClaudePriceSearchService] = None


def get_claude_price_search_service() -> ClaudePriceSearchService:
    global _service
    if _service is None:
        _service = ClaudePriceSearchService()
    return _service
