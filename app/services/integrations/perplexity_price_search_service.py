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

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.dataforseo_merchant_service import (
    get_dataforseo_merchant_service,
    MerchantHit,
)
from app.services.integrations.firecrawl_client import get_firecrawl_client
from app.models.extraction import PriceExtraction
from app.utils.price_parsing import parse_price

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
    """Single retailer result — may come from Perplexity web search or DataForSEO Shopping feed."""
    retailer_name: str = Field(..., description="Retailer display name.")
    product_url: str = Field(..., description="Direct product page URL.")
    price: Optional[float] = Field(
        default=None,
        description="Current numeric price. None ONLY when is_quote_only=true.",
    )
    original_price: Optional[float] = Field(
        default=None,
        description=(
            "On-page 'was' price when the retailer displays a promo (was €89, now €79). "
            "Distinct from observed historical price changes stored in price_history / "
            "tracked_query_price_history — this field reflects what the retailer advertises "
            "right now on the page."
        ),
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
    source: str = Field(
        default="perplexity",
        description="'perplexity' = web search (organic retailers), 'dataforseo' = Google Shopping merchants.",
    )
    verified: bool = Field(
        default=False,
        description=(
            "True when Firecrawl actually fetched the product page and confirmed the price. "
            "False means the price came only from Perplexity/DataForSEO snippet data and may "
            "be stale / hallucinated — callers should treat as indicative, not authoritative."
        ),
    )
    image_url: Optional[str] = Field(default=None, description="DataForSEO only: product thumbnail URL from the Shopping feed.")
    rating_value: Optional[float] = Field(default=None, description="DataForSEO only: merchant star rating.")
    rating_votes: Optional[int] = Field(default=None, description="DataForSEO only: number of rating votes.")


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
        verify_prices: bool = True,
    ) -> PriceSearchResult:
        """
        Run one Sonar search for the given product. Returns PriceSearchResult
        with hits + usage metadata. Stateless — caller handles DB persistence.
        """
        if not self.api_key:
            return PriceSearchResult(success=False, error="PERPLEXITY_API_KEY not configured")

        limit = max(1, min(limit, 25))

        # Run Perplexity + DataForSEO Merchant in parallel. Each returns up
        # to `limit` hits; we merge + dedupe by domain, keeping the cheapest
        # per domain regardless of source. Source field on each hit lets
        # the UI split them into "Discovered retailers" (perplexity) and
        # "Merchants" (dataforseo) sections.
        perplexity_task = asyncio.create_task(
            self._perplexity_call(
                product_name, dimensions, country_code, limit,
                preferred_retailer_domains, user_id, workspace_id,
            )
        )
        dataforseo_task = asyncio.create_task(
            get_dataforseo_merchant_service().search_shopping(
                product_name=product_name,
                dimensions=dimensions,
                country_code=country_code,
                limit=limit,
            )
        )
        perplexity_result, dataforseo_result = await asyncio.gather(
            perplexity_task, dataforseo_task, return_exceptions=True
        )

        # Unpack Perplexity result (may have raised)
        if isinstance(perplexity_result, BaseException):
            logger.warning(f"Perplexity call raised: {perplexity_result}")
            return PriceSearchResult(success=False, error=f"perplexity: {perplexity_result}")

        # Merge DataForSEO hits into the result (if it succeeded)
        if not isinstance(dataforseo_result, BaseException) and dataforseo_result.success:
            perplexity_result.hits = self._merge_with_dataforseo(
                perplexity_hits=perplexity_result.hits,
                dataforseo_hits=dataforseo_result.hits,
                country_code=country_code,
            )
            perplexity_result.credits_used += dataforseo_result.credits_used
            perplexity_result.cost_usd = (perplexity_result.cost_usd or 0.0) + dataforseo_result.cost_usd
        elif isinstance(dataforseo_result, BaseException):
            logger.warning(f"DataForSEO call raised (non-fatal): {dataforseo_result}")

        # Stage B: verify prices via Firecrawl per URL (parallel asyncio.gather).
        # Fetches each retailer's actual product page and extracts the real price
        # from the rendered HTML. Fixes Perplexity hallucinations + stale snippets.
        # Opt-out via verify_prices=False for callers who value latency over accuracy.
        if verify_prices and perplexity_result.hits:
            verify_credits = await self._verify_hits_with_firecrawl(
                perplexity_result.hits, user_id=user_id, workspace_id=workspace_id,
            )
            perplexity_result.credits_used += verify_credits

        return perplexity_result

    # The original Perplexity-only flow, extracted so we can run it alongside
    # DataForSEO in parallel without tangling the orchestration logic.
    async def _perplexity_call(
        self,
        product_name: str,
        dimensions: Optional[str],
        country_code: Optional[str],
        limit: int,
        preferred_retailer_domains: Optional[List[str]],
        user_id: Optional[str],
        workspace_id: Optional[str],
    ) -> "PriceSearchResult":
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
            "- When the page displays a was/now promo (e.g. 'Was €89, Now €79', '€89 €79', "
            "  strikethrough on the old price), populate BOTH: `price` = current, `original_price` = was. "
            "  Only set `original_price` when the previous price is actually visible on the page — never invent it.\n"
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
                            "original_price": {
                                "type": ["number", "null"],
                                "description": "On-page 'was' / 'original' price if the retailer displays a promo (was €89, now €79). Null if no markdown is shown. Do NOT invent — only populate when the was-price is clearly visible on the page.",
                            },
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

    async def _verify_hits_with_firecrawl(
        self,
        hits: List[PriceHit],
        *,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> int:
        """
        Second-stage verification: fetch each retailer's actual product page via
        Firecrawl, extract price + original_price via PriceExtraction schema,
        and rewrite the hit in place. Runs all URLs in parallel via asyncio.gather.

        Mutates `hits` list in place. Returns total Firecrawl credits consumed.

        Semantics per row:
          - Firecrawl finds a price on the page → replace `price`, fill
            `original_price` if was/now visible, set `verified=True`.
          - Firecrawl returns no price (404, blocked, truly missing) → leave
            the row as-is with `verified=False`.
          - Firecrawl price differs by >20% from the Perplexity/DataForSEO
            price → trust Firecrawl (it actually read the page) + append a
            note flagging the discrepancy so downstream consumers can see it.
        """
        firecrawl = get_firecrawl_client()

        async def verify_one(hit: PriceHit) -> int:
            """Returns credits consumed for this one verification (0 on failure)."""
            try:
                result = await firecrawl.scrape(
                    url=hit.product_url,
                    extraction_model=PriceExtraction,
                    user_id=user_id or "system",
                    workspace_id=workspace_id,
                    extraction_prompt=(
                        f"Extract the current price and on-page 'was' price (if any) for "
                        f"{hit.retailer_name}'s product page. Prefer the main product price, "
                        "not related / bundle / strike-through prices."
                    ),
                    use_javascript_render=False,
                )
            except Exception as e:
                logger.debug(f"Firecrawl verify crashed for {hit.product_url}: {e}")
                return 0

            if not result.success or not result.data:
                return result.credits_used or 1

            extracted = result.data
            hint_currency = extracted.currency or hit.currency
            amount, currency = parse_price(extracted.price, hint_currency=hint_currency)
            if amount is None:
                # Page loaded but no extractable price — keep the original data,
                # don't mark as verified.
                return result.credits_used or 1

            verified_price = float(amount)
            original_amount, _ = parse_price(extracted.original_price, hint_currency=hint_currency)
            verified_original = float(original_amount) if original_amount is not None else None

            # Discrepancy check: if Firecrawl's price is materially different from
            # Perplexity/DataForSEO's, flag it. Trust Firecrawl (read the page).
            prior_price = hit.price
            diff_note = None
            if prior_price is not None and prior_price > 0:
                diff_ratio = abs(verified_price - prior_price) / prior_price
                if diff_ratio > 0.20:
                    diff_note = (
                        f"verify: was {hit.source}=€{prior_price:.2f}, "
                        f"actual on page=€{verified_price:.2f}"
                    )

            hit.price = verified_price
            if verified_original is not None and verified_original > verified_price:
                hit.original_price = verified_original
            if currency and not hit.currency:
                hit.currency = currency
            if extracted.availability and extracted.availability in (
                "in_stock", "out_of_stock", "limited", "unknown"
            ):
                hit.availability = extracted.availability
            hit.verified = True
            hit.last_verified = datetime.now(timezone.utc).date().isoformat()
            if diff_note:
                hit.notes = f"{hit.notes} | {diff_note}" if hit.notes else diff_note

            return result.credits_used or 1

        # Parallel verify. Firecrawl's client already has its own retry/backoff
        # + per-call credit logging; gather-return-exceptions keeps one bad URL
        # from killing the whole batch.
        credit_results = await asyncio.gather(
            *(verify_one(h) for h in hits),
            return_exceptions=True,
        )
        total_credits = 0
        for c in credit_results:
            if isinstance(c, int):
                total_credits += c
        # Re-sort after verification — prices may have shifted.
        hits.sort(key=lambda h: (h.price if h.price is not None else float("inf")))
        return total_credits

    @classmethod
    def _merge_with_dataforseo(
        cls,
        perplexity_hits: List[PriceHit],
        dataforseo_hits: List[MerchantHit],
        country_code: Optional[str],
    ) -> List[PriceHit]:
        """
        Merge Perplexity + DataForSEO hits, deduped by retailer domain. When
        both sources return the same retailer, keep Perplexity's version
        (richer fields: availability, city, notes). DataForSEO-only retailers
        get added with source='dataforseo' and their Shopping-feed metadata
        (image_url, rating). Sort by price ascending.
        """
        by_domain: Dict[str, PriceHit] = {}
        for h in perplexity_hits:
            d = cls._domain_of(h.product_url)
            if d:
                h.source = h.source or "perplexity"
                by_domain[d] = h

        for m in dataforseo_hits:
            d = cls._domain_of(m.product_url)
            if not d or d in by_domain:
                continue
            by_domain[d] = PriceHit(
                retailer_name=m.retailer_name,
                product_url=m.product_url,
                price=m.price,
                original_price=m.original_price,
                currency=m.currency,
                price_unit="piece",  # Shopping feed is per-unit, not per-m²
                availability="in_stock",  # DataForSEO only surfaces buyable items
                city=None,
                ships_from_abroad=False,
                is_quote_only=False,
                last_verified=datetime.now(timezone.utc).date().isoformat(),
                notes="via Google Shopping (DataForSEO)",
                source="dataforseo",
                image_url=m.image_url,
                rating_value=m.rating_value,
                rating_votes=m.rating_votes,
            )

        merged = sorted(by_domain.values(), key=lambda h: (h.price if h.price is not None else float("inf")))
        return merged

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
