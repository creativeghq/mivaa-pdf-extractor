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
- Model: claude-opus-4-7 — matches claude.ai's default backend for web_search
  quality. Haiku was tried first (cheaper) but returned empty for niche B2B
  products that Opus finds reliably.
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
# Opus — more thorough at web_search reasoning + extracting prices from page content.
# Haiku (prior choice) was missing prices that claude.ai's Opus backend finds.
MODEL = "claude-opus-4-7"
WEB_SEARCH_BETA_HEADER = "web-search-2025-03-05"
WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
MAX_WEB_SEARCHES = 10  # was 5 — more coverage for niche products
MAX_TOKENS = 4096
HTTP_TIMEOUT_S = 120.0  # Opus is slower
THROTTLE_HOURS = 6

# Opus 4.7 pricing (from app/config/ai_pricing.py — ballpark for logging).
# Real debit happens via AICallLogger; these are just fallback estimates.
OPUS_INPUT_PER_1K = 0.015
OPUS_OUTPUT_PER_1K = 0.075
WEB_SEARCH_PER_CALL = 0.010  # $10 / 1000 web searches


# ── Country-specific hints for Claude's web_search ──
# Anthropic's web_search runs from US infra and doesn't auto-localize, so we
# hand-feed Claude the local retailers and price-comparison portals per country.
# Keep this narrow: just enough to get Claude started. Claude will find more on
# its own once it's on the right TLD.
_LOCAL_TLD: Dict[str, str] = {
    "GR": ".gr", "BG": ".bg", "RO": ".ro", "CY": ".cy",
    "IT": ".it", "ES": ".es", "PT": ".pt", "FR": ".fr",
    "DE": ".de", "AT": ".at", "CH": ".ch", "NL": ".nl", "BE": ".be",
    "GB": ".co.uk", "IE": ".ie",
    "PL": ".pl", "CZ": ".cz", "SK": ".sk", "HU": ".hu",
    "LT": ".lt", "LV": ".lv", "EE": ".ee",
    "SE": ".se", "DK": ".dk", "NO": ".no", "FI": ".fi",
    "TR": ".com.tr",
}

# Generic source-type hints — same guidance for every country, no retailer names.
# Claude picks the actual retailers by itself from the local TLD + the category
# noun. This works across any material (tiles, fabric, paint, wood, stone, …)
# because the source types are universal, not product-specific.
_GENERIC_SOURCE_TYPES = (
    "Source types to probe (in priority order — stop once you have enough results):\n"
    "  a) Local price-comparison portals that list 'Retailer X: €N' directly on-page "
    "     (the major one per country, e.g. the Greek/Bulgarian/Romanian/German equivalent "
    "     of Google Shopping). Treat each retailer row as a separate result, not the portal.\n"
    "  b) Local-TLD organic retailer results for the product + 'price' in local language.\n"
    "  c) Local home-improvement / builders-merchant chains in that country.\n"
    "  d) Local category-specialist retailers (e.g. a tile-specialist shop for tiles, a "
    "     fabric house for textiles, a flooring specialist for wood — let the material "
    "     type in the query guide you).\n"
    "  e) Local marketplace listings (country-specific Amazon / eBay / equivalent).\n"
)


# ────────────────────────────────────────────────────────────────────────────
# Models
# ────────────────────────────────────────────────────────────────────────────


class PriceHit(BaseModel):
    """Single retailer result returned by Claude for a product."""
    retailer_name: str = Field(..., description="Retailer display name, e.g. 'Topps Tiles', 'Mandarin Stone'.")
    product_url: str = Field(..., description="Direct product page URL. Must be a product page, not a search results page.")
    price: Optional[float] = Field(
        default=None,
        description="Numeric price. None ONLY when is_quote_only=true (retailer carries product but shows 'quote on request').",
    )
    currency: Optional[str] = Field(default=None, description="ISO 4217 currency code (USD, EUR, GBP, etc.)")
    price_unit: Optional[str] = Field(
        default="m2",
        description="Unit the price refers to: 'm2', 'box', 'piece', 'linear_meter'. Default 'm2' for tiles.",
    )
    availability: Optional[str] = Field(default=None, description="in_stock | out_of_stock | limited | unknown")
    city: Optional[str] = Field(default=None, description="Retailer city/region if known.")
    ships_from_abroad: bool = Field(
        default=False,
        description="True if retailer is outside the requested country but will ship to it.",
    )
    is_quote_only: bool = Field(
        default=False,
        description="True if retailer carries product but shows 'Quote on request' / 'Contact for price' instead of a number.",
    )
    last_verified: Optional[str] = Field(
        default=None,
        description="ISO date when the retailer's page was last verified to have this product. Usually today's date.",
    )
    notes: Optional[str] = Field(default=None, description="Short context note (promo valid dates, per-box vs per-m², shipping, etc.)")


class PriceSearchResult(BaseModel):
    """Wrapper for results + metadata."""
    success: bool
    hits: List[PriceHit] = []
    summary: Optional[str] = Field(
        default=None,
        description="2-3 sentence human-readable summary from Claude: closest retailer, manufacturer "
                    "showroom presence, pricing anomalies. Rendered below the table in the UI.",
    )
    credits_used: int = 0
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    web_searches: int = 0
    cost_usd: float = 0.0
    throttled: bool = False
    throttle_until: Optional[datetime] = None
    error: Optional[str] = None
    debug_reasoning: Optional[str] = Field(
        default=None,
        description="When hits is empty, Claude's text reasoning — useful to tell "
                    "'Claude searched but found nothing' from 'Claude never tried'.",
    )


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

        limit = max(1, min(limit, 25))  # raised 10→25 to capture Google Shopping merchant listings
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
        hits, summary = self._extract_structured_hits(data)
        debug_reasoning: Optional[str] = None
        if not hits:
            # Collect any text content blocks so callers can see Claude's reasoning.
            texts = [
                b.get("text", "")
                for b in (data.get("content") or [])
                if b.get("type") == "text" and b.get("text")
            ]
            debug_reasoning = "\n\n".join(texts).strip() or None

        usage = data.get("usage", {}) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        server_tool_use = usage.get("server_tool_use") or {}
        web_searches = int(server_tool_use.get("web_search_requests", 0) or 0)

        cost_usd = (
            (input_tokens / 1000) * OPUS_INPUT_PER_1K
            + (output_tokens / 1000) * OPUS_OUTPUT_PER_1K
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
            summary=summary,
            credits_used=platform_credits,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            web_searches=web_searches,
            cost_usd=cost_usd,
            debug_reasoning=debug_reasoning,
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
        """
        LOCAL-FIRST price search. For materials (tiles, stone, flooring, fabric),
        sales happen locally — international retailers are almost always irrelevant
        unless local coverage is genuinely zero.
        """
        size_part = f" {dimensions}" if dimensions else ""
        product_spec = f"{product_name}{size_part}".strip()
        country = country_code or "any country (global search)"
        today = datetime.now(timezone.utc).date().isoformat()

        country_hint = ""
        if country_code:
            local_tld = _LOCAL_TLD.get(country_code.upper(), "." + country_code.lower())
            country_hint = (
                f"LOCATION MATTERS — for the materials this platform deals with (tiles, stone, "
                f"wood, fabric, paint, flooring, hardware, etc.), sales happen locally. Your "
                f"primary target is retailers in {country_code}.\n\n"
                "CRITICAL SEARCH TACTICS for local coverage:\n"
                f"- Run at least ONE search in the LOCAL LANGUAGE of {country_code}. "
                "  Translate the product category noun into the local language "
                "  (e.g. 'tile' → 'πλακάκι' in GR, 'Fliese' in DE, 'azulejo' in ES, "
                "  'plytelės' in LT; 'fabric' → 'τσόχα' / 'Stoff' / 'tela' / etc.). "
                "  Add the local word for 'price' (τιμή / Preis / precio / cena). "
                "  Keep the brand + SKU in the original spelling. "
                "  This surfaces native-language retailer pages that English-only queries miss.\n"
                f"- Use the `site:` operator for the local TLD at least once, "
                f"  e.g. `{product_spec} site:{local_tld}`.\n"
                f"- Identify the dominant local price-comparison portal for {country_code} "
                "  yourself (it's usually the biggest one returned by the first organic search). "
                "  Its pages often list 'Retailer X: €N' inline — each retailer row is its own "
                "  result.\n\n"
                f"{_GENERIC_SOURCE_TYPES}\n"
                f"Only fall back to international (ships_from_abroad=true) after you have "
                f"exhausted local options — and prefer neighboring / same-language-region "
                "countries first.\n\n"
            )

        return (
            f"Find current retail prices for {product_spec} in {country}. "
            f"Return up to {limit} results — only retailers with a VISIBLE NUMERIC PRICE. "
            "Do NOT return quote-only / 'price on request' / 'contact for quote' entries.\n\n"
            f"{country_hint}"
            "Search strategy — do ALL of these in parallel web_search calls:\n"
            f"1. Direct organic retailer results in {country_code or 'the target market'} — local "
            "   language, local domain (.gr, .bg, .ro, .it, .es, .de, .co.uk, .fr, .pl, etc.)\n"
            f"2. Google Shopping results filtered to {country_code or 'local'} — each merchant is a "
            "   separate retailer\n"
            "3. Local price-comparison sites that show retailer names AND prices on the same page "
            "   (Skroutz.gr, BestPrice.gr, Heureka.cz, Ceneo.pl, Idealo.de, Compari.ro, etc). "
            "   These ARE useful — if their page lists 'Retailer X: €25' explicitly, treat each listed "
            "   retailer as a row (not the aggregator itself). Set notes='Price surfaced via <aggregator>'.\n"
            "4. Marketplaces (Amazon, eBay) where the product has a visible price.\n"
            "5. Local tile / material specialist retailers and their category pages — if a category page "
            "   shows THIS SKU with a clear price next to its name, that's a valid row.\n\n"
            "Requirements:\n"
            "1. One row per unique retailer domain — never list the same website twice. Pick the "
            "   cheapest available variant per retailer.\n"
            f"2. PRIORITIZE {country_code or 'local'} retailers. International results (ships_from_abroad=true) "
            "   are acceptable but should NOT dominate the list when local options exist.\n"
            "3. EXCLUDE any row where price is unknown or shown as 'quote on request'. Skip retailers "
            "   that carry the product but don't publish a price.\n"
            "4. OK to return fewer than the limit if there genuinely aren't that many retailers with "
            "   published prices. Never invent data to fill the count.\n\n"
            "For each row include:\n"
            "- retailer_name\n"
            "- city (if known from the retailer page or address)\n"
            "- price per m² in local currency (REQUIRED — no null prices)\n"
            "- currency (ISO code)\n"
            "- product_url (direct product page, category page showing this SKU with price, "
            "  Google Shopping result, or price-comparison page showing the retailer's entry)\n"
            "- availability (in_stock | out_of_stock | limited | unknown)\n"
            f"- last_verified ({today} unless you have a more specific date)\n"
            f"- ships_from_abroad=true if retailer is outside {country_code or 'the target market'}\n"
            "- notes (promo valid dates, per-box vs per-m², 'via Skroutz', 'via Google Shopping', etc.)\n\n"
            "Sort by: price ascending (cheapest first).\n\n"
            "Do not include:\n"
            "- Aggregators that hide prices behind click-through redirects with no price on the page. "
            "  If the aggregator page DOES show 'Retailer X: €25', treat the retailer as a row.\n"
            "- The manufacturer's own website unless they sell direct-to-consumer with published prices.\n"
            "- Any row without a real visible price.\n\n"
            "After compiling the list, write 2-3 sentences as `summary`: closest local retailer, "
            "manufacturer showroom presence in-country, pricing anomalies worth questioning, and which "
            "local retailers you found that carry the product but don't publish prices (so the user "
            "can contact them for a quote).\n\n"
            "Call submit_price_results EXACTLY ONCE with {retailers: [...], summary: '...'}. "
            "Only submit prices you actually saw — do not fabricate."
        )

    def _build_tools(self, limit: int) -> List[Dict[str, Any]]:
        """Anthropic web_search + our forced-output tool."""
        return [
            {"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search", "max_uses": MAX_WEB_SEARCHES},
            {
                "name": "submit_price_results",
                "description": (
                    "Submit the final structured list of retailers. Call this exactly once at the end "
                    "of your research. Include rows with is_quote_only=true to reach the requested "
                    "count when published prices are scarce. Always include the summary text."
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
                                    "price": {"type": ["number", "null"], "description": "Price per unit (usually per m²). Null only when is_quote_only."},
                                    "currency": {"type": ["string", "null"], "description": "ISO 4217 code. Null when is_quote_only."},
                                    "price_unit": {"type": "string", "enum": ["m2", "box", "piece", "linear_meter"], "default": "m2"},
                                    "availability": {"type": "string", "enum": ["in_stock", "out_of_stock", "limited", "unknown"]},
                                    "city": {"type": "string"},
                                    "ships_from_abroad": {"type": "boolean", "default": False},
                                    "is_quote_only": {"type": "boolean", "default": False},
                                    "last_verified": {"type": "string", "description": "ISO date."},
                                    "notes": {"type": "string"},
                                },
                                "required": ["retailer_name", "product_url"],
                            },
                        },
                        "summary": {
                            "type": "string",
                            "description": "2-3 sentences: closest retailer, manufacturer showroom presence in-country, pricing anomalies.",
                        },
                    },
                    "required": ["retailers", "summary"],
                },
            },
        ]

    def _extract_structured_hits(
        self, response: Dict[str, Any]
    ) -> Tuple[List[PriceHit], Optional[str]]:
        """Pull the submit_price_results tool_use block. Returns (hits, summary)."""
        for block in response.get("content", []) or []:
            if block.get("type") == "tool_use" and block.get("name") == "submit_price_results":
                payload = block.get("input") or {}
                raw_list = payload.get("retailers") or []
                summary = payload.get("summary")
                hits: List[PriceHit] = []
                seen_domains = set()
                for raw in raw_list:
                    try:
                        hit = PriceHit(**raw)
                    except Exception as e:
                        logger.debug(f"Skipped malformed retailer entry: {e}")
                        continue
                    # Hard filter: no quote-only rows, no missing prices.
                    if hit.is_quote_only or hit.price is None:
                        continue
                    # Dedupe by retailer domain (per spec: "one row per unique retailer")
                    domain = self._domain_of(hit.product_url)
                    if not domain or domain in seen_domains:
                        continue
                    seen_domains.add(domain)
                    hits.append(hit)
                # Sort: cheapest first.
                hits = self._sort_hits(hits)
                return hits, summary
        # Fallback: if the model forgot to call the tool but dumped JSON in text.
        return self._extract_from_text(response), None

    @staticmethod
    def _domain_of(url: str) -> str:
        m = re.match(r"^https?://([^/]+)", (url or "").strip(), flags=re.IGNORECASE)
        return (m.group(1) if m else url or "").lower().removeprefix("www.")

    @staticmethod
    def _sort_hits(hits: List[PriceHit]) -> List[PriceHit]:
        """Sort by price ascending. Quote-only rows are filtered upstream."""
        return sorted(hits, key=lambda h: (h.price if h.price is not None else float("inf")))

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
                    "input_cost_usd": round((input_tokens / 1000) * OPUS_INPUT_PER_1K, 6),
                    "output_cost_usd": round((output_tokens / 1000) * OPUS_OUTPUT_PER_1K, 6),
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
