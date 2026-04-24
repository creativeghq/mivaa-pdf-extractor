"""
Perplexity-powered price discovery via Sonar.

Replaces the Claude + web_search_20250305 path. Claude's API search tool
(Brave-based, snippets only, no geo-location) was materially weaker than
claude.ai's internal search; Perplexity Sonar has deeper page reading and
real user_location support, which is the difference between seeing
"€25,00/m²" on youbath.gr and missing it entirely.

Design:
- Model: sonar-pro (best quality for retail product search)
- Structured output via response_format.json_schema — forces clean JSON
  per the PriceHit schema, no regex parsing
- user_location.country biases results to the user's market
- search_recency_filter="month" trims stale prices
- Credit logging via ai_usage_logs (same pattern as Claude usage)

Why REST + httpx instead of an SDK wrapper:
- Perplexity's API is simple chat-completions shape; an SDK adds little
- Keeps the service in the same Python backend as the rest of price
  monitoring (price_lookups, ai_usage_logs, credits) — no runtime split
- Matches the exact shape the Vercel AI SDK produces (it just POSTs here)

Kept API-compatible with ClaudePriceSearchService: same PriceHit /
PriceSearchResult types and `search_prices()` method signature so the
route code can swap one import without touching anything else.
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

PERPLEXITY_BASE_URL = "https://api.perplexity.ai/chat/completions"
# sonar-pro: designed for multi-step research with higher-quality citations
# + deeper page reading. ~$3/M in, $15/M out, plus $5/1k searches at high
# context. We use high-context because retail price pages need it.
MODEL = "sonar-pro"
MAX_TOKENS = 3000
HTTP_TIMEOUT_S = 90.0
THROTTLE_HOURS = 6

# sonar-pro pricing, ballpark — real debit is metered server-side.
SONAR_PRO_INPUT_PER_1K = 0.003   # $3 / 1M
SONAR_PRO_OUTPUT_PER_1K = 0.015  # $15 / 1M
SONAR_SEARCH_PER_CALL = 0.005    # $5 / 1K requests (high context)


# Country → local TLD (used only as a soft hint in the system prompt).
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


# ────────────────────────────────────────────────────────────────────────────
# Models (identical shape to the Claude service — callers depend on this)
# ────────────────────────────────────────────────────────────────────────────


class PriceHit(BaseModel):
    """Single retailer result returned by the search engine."""
    retailer_name: str = Field(..., description="Retailer display name.")
    product_url: str = Field(..., description="Direct product page URL.")
    price: Optional[float] = Field(
        default=None,
        description="Numeric price. None ONLY when is_quote_only=true.",
    )
    currency: Optional[str] = Field(default=None, description="ISO 4217 currency code.")
    price_unit: Optional[str] = Field(
        default="m2",
        description="'m2', 'box', 'piece', 'linear_meter'. Default m2 for tiles.",
    )
    availability: Optional[str] = Field(default=None, description="in_stock | out_of_stock | limited | unknown")
    city: Optional[str] = Field(default=None, description="Retailer city/region if known.")
    ships_from_abroad: bool = Field(default=False)
    is_quote_only: bool = Field(default=False, description="Retailer carries product but shows 'quote on request'.")
    last_verified: Optional[str] = Field(default=None, description="ISO date verified.")
    notes: Optional[str] = Field(default=None)


class PriceSearchResult(BaseModel):
    success: bool
    hits: List[PriceHit] = []
    summary: Optional[str] = None
    credits_used: int = 0
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    web_searches: int = 0
    cost_usd: float = 0.0
    throttled: bool = False
    throttle_until: Optional[datetime] = None
    error: Optional[str] = None
    debug_reasoning: Optional[str] = None


# ────────────────────────────────────────────────────────────────────────────
# Service
# ────────────────────────────────────────────────────────────────────────────


class PerplexityPriceSearchService:
    """Perplexity Sonar-powered price discovery."""

    def __init__(self) -> None:
        self.api_key = os.getenv("PERPLEXITY_API_KEY") or ""
        self.supabase = get_supabase_client()
        if not self.api_key:
            logger.warning("⚠️ PERPLEXITY_API_KEY not configured — Perplexity price search disabled")

    # ────────── Public API ──────────

    async def search_prices(
        self,
        product_name: str,
        dimensions: Optional[str] = None,
        country_code: Optional[str] = None,
        limit: int = 10,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        preferred_retailer_domains: Optional[List[str]] = None,
    ) -> PriceSearchResult:
        """
        Run one Sonar search for the given product. Returns PriceSearchResult
        with hits + usage metadata. Stateless — caller handles DB persistence.
        """
        if not self.api_key:
            return PriceSearchResult(success=False, error="PERPLEXITY_API_KEY not configured")

        limit = max(1, min(limit, 25))
        system_prompt, user_prompt = self._build_messages(product_name, dimensions, country_code, limit)
        schema = self._response_schema(limit)

        body: Dict[str, Any] = {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "price_results", "schema": schema, "strict": True},
            },
            "web_search_options": {
                "search_context_size": "high",
            },
        }
        if country_code:
            body["web_search_options"]["user_location"] = {"country": country_code.upper()}
        # Option 2: domain pinning — force Perplexity to ALSO probe these known retailers.
        # Perplexity allows up to 10 domains in search_domain_filter. We cap at 10.
        if preferred_retailer_domains:
            cleaned = [d.strip().lower().removeprefix("www.") for d in preferred_retailer_domains if d and isinstance(d, str)]
            cleaned = [d for d in cleaned if d][:10]
            if cleaned:
                body["web_search_options"]["search_domain_filter"] = cleaned

        start = datetime.now(timezone.utc)
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
                resp = await client.post(
                    PERPLEXITY_BASE_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
        except httpx.TimeoutException as e:
            return PriceSearchResult(success=False, error=f"timeout: {e}")
        except Exception as e:
            return PriceSearchResult(success=False, error=f"request failed: {e}")

        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)

        if resp.status_code != 200:
            return PriceSearchResult(
                success=False,
                latency_ms=latency_ms,
                error=f"perplexity HTTP {resp.status_code}: {resp.text[:400]}",
            )

        data = resp.json()
        hits, summary, debug_reasoning = self._extract(data)

        usage = data.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        # Perplexity counts search requests as part of usage meta in newer
        # responses; the API is still evolving so we fall back to a heuristic.
        search_requests = int(usage.get("num_search_queries") or usage.get("search_queries") or 1)

        cost_usd = (
            (input_tokens / 1000) * SONAR_PRO_INPUT_PER_1K
            + (output_tokens / 1000) * SONAR_PRO_OUTPUT_PER_1K
            + search_requests * SONAR_SEARCH_PER_CALL
        )
        platform_credits = int(round(cost_usd * 100))

        await self._log_usage(
            user_id=user_id,
            workspace_id=workspace_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            search_requests=search_requests,
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
            web_searches=search_requests,
            cost_usd=cost_usd,
            debug_reasoning=debug_reasoning if not hits else None,
        )

    def check_throttle(
        self, last_search_at: Optional[datetime], force_refresh: bool = False
    ) -> Tuple[bool, Optional[datetime]]:
        """6h throttle per product. Admin force_refresh bypasses."""
        if force_refresh or last_search_at is None:
            return False, None
        expires_at = last_search_at + timedelta(hours=THROTTLE_HOURS)
        if datetime.now(timezone.utc) < expires_at:
            return True, expires_at
        return False, None

    # ────────── Internals ──────────

    def _build_messages(
        self,
        product_name: str,
        dimensions: Optional[str],
        country_code: Optional[str],
        limit: int,
    ) -> Tuple[str, str]:
        """Returns (system_prompt, user_prompt)."""
        size_part = f" {dimensions}" if dimensions else ""
        product_spec = f"{product_name}{size_part}".strip()
        today = datetime.now(timezone.utc).date().isoformat()

        local_directive = ""
        if country_code:
            local_tld = _LOCAL_TLD.get(country_code.upper(), "." + country_code.lower())
            local_directive = (
                f"\n\nLocation matters: for materials, sales happen locally. Prioritize retailers in "
                f"{country_code.upper()} (look for {local_tld} domains and local-language pages). "
                "Include international retailers only if local coverage is thin, and mark them with "
                "ships_from_abroad=true."
            )

        system_prompt = (
            "You are a price-research tool for building materials (tiles, stone, wood, fabric, "
            "paint, hardware, etc). Your job is to return a clean, deduplicated list of retailers "
            "with VISIBLE numeric prices for the product the user asks about.\n\n"
            "CRITICAL — Include out-of-stock listings as long as a numeric price is printed on the page.\n"
            "An out-of-stock listing with a posted price is valuable market data and MUST be included.\n"
            "Concrete example: a page showing 'Keros Ferrara Beige 60x120 — €25.00/m² — Out of stock' "
            "(or the local-language equivalent such as 'Εκτός διαθεσιμότητας' in Greek, 'Nicht auf Lager' "
            "in German, 'Agotado' in Spanish, 'Rupture de stock' in French) MUST be returned with "
            "price=25.00, currency=EUR, availability=out_of_stock. Do NOT exclude it just because it "
            "isn't currently buyable — the price tells the user what the market reference is.\n\n"
            "Other rules:\n"
            "- One row per unique retailer domain. Pick the cheapest variant if a retailer lists multiple.\n"
            "- EXCLUDE only when the retailer truly has NO price on the page: 'quote only', 'contact for "
            "  price', 'price on request', 'login for pricing', or a price that's completely missing. "
            "  Do NOT fabricate prices to fill the list.\n"
            "- EXCLUDE the manufacturer's own site unless they publish retail prices.\n"
            "- Sort by price ascending.\n"
            "- Include the product's real page URL — not a search page or homepage.\n"
            "- For aggregator / price-comparison portals that list 'Retailer X: €N' inline, treat each "
            "  listed retailer as its own row and note the aggregator in `notes`.\n"
            f"- Use today's date ({today}) for last_verified unless you have a more specific one.\n"
            f"{local_directive}"
        )

        user_prompt = (
            f"Find current published retail prices for: {product_spec}. "
            f"Return up to {limit} retailers. After the list, write a 2-3 sentence `summary` noting: "
            "the closest retailer to the user (if country is known), any manufacturer showroom "
            "presence in-country, and any pricing outliers worth questioning."
        )
        return system_prompt, user_prompt

    def _response_schema(self, limit: int) -> Dict[str, Any]:
        """JSON schema forced on the model output — mirrors PriceHit."""
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["retailers", "summary"],
            "properties": {
                "retailers": {
                    "type": "array",
                    "maxItems": limit,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["retailer_name", "product_url", "price", "currency"],
                        "properties": {
                            "retailer_name": {"type": "string"},
                            "product_url": {"type": "string"},
                            "price": {"type": "number"},
                            "currency": {"type": "string"},
                            "price_unit": {"type": "string", "enum": ["m2", "box", "piece", "linear_meter"]},
                            "availability": {"type": "string", "enum": ["in_stock", "out_of_stock", "limited", "unknown"]},
                            "city": {"type": "string"},
                            "ships_from_abroad": {"type": "boolean"},
                            "last_verified": {"type": "string"},
                            "notes": {"type": "string"},
                        },
                    },
                },
                "summary": {"type": "string"},
            },
        }

    def _extract(
        self, response: Dict[str, Any]
    ) -> Tuple[List[PriceHit], Optional[str], Optional[str]]:
        """Parse the Perplexity response into (hits, summary, debug_reasoning)."""
        choices = response.get("choices") or []
        if not choices:
            return [], None, None
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        debug = content if content else None

        # content is JSON per our response_format. Parse, fall back to text scan.
        payload: Dict[str, Any] = {}
        try:
            payload = json.loads(content) if content.strip().startswith("{") else {}
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]+\}", content)
            if m:
                try:
                    payload = json.loads(m.group(0))
                except Exception:
                    payload = {}

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
            if hit.is_quote_only or hit.price is None:
                continue
            domain = self._domain_of(hit.product_url)
            if not domain or domain in seen_domains:
                continue
            seen_domains.add(domain)
            hits.append(hit)
        hits.sort(key=lambda h: (h.price if h.price is not None else float("inf")))
        return hits, summary, debug

    @staticmethod
    def _domain_of(url: str) -> str:
        m = re.match(r"^https?://([^/]+)", (url or "").strip(), flags=re.IGNORECASE)
        return (m.group(1) if m else url or "").lower().removeprefix("www.")

    async def _log_usage(
        self,
        *,
        user_id: Optional[str],
        workspace_id: Optional[str],
        input_tokens: int,
        output_tokens: int,
        search_requests: int,
        cost_usd: float,
        platform_credits: int,
        latency_ms: int,
        product_name: str,
        hits_count: int,
    ) -> None:
        """Insert into ai_usage_logs. Same columns as the Claude path."""
        try:
            self.supabase.client.table("ai_usage_logs").insert(
                {
                    "user_id": user_id,
                    "workspace_id": workspace_id,
                    "operation_type": "price_search",
                    "model_name": MODEL,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_cost_usd": round((input_tokens / 1000) * SONAR_PRO_INPUT_PER_1K, 6),
                    "output_cost_usd": round((output_tokens / 1000) * SONAR_PRO_OUTPUT_PER_1K, 6),
                    "total_cost_usd": round(cost_usd, 6),
                    "credits_debited": platform_credits,
                    "metadata": {
                        "api_provider": "perplexity",
                        "tool": "sonar-pro",
                        "search_requests": search_requests,
                        "product_name": product_name,
                        "hits_count": hits_count,
                        "latency_ms": latency_ms,
                    },
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            ).execute()
        except Exception as e:
            logger.warning(f"Failed to log Perplexity price_search usage: {e}")


_service: Optional[PerplexityPriceSearchService] = None


def get_perplexity_price_search_service() -> PerplexityPriceSearchService:
    global _service
    if _service is None:
        _service = PerplexityPriceSearchService()
    return _service
