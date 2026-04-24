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
from app.services.integrations.product_identity_service import (
    get_product_identity_service,
    QueryFacets,
    url_prefilter,
    url_slug_tokens,
)

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
    # ── Product-identity verification (2026-04-25) ──
    # Populated by product_identity_service after Firecrawl runs. Tells the
    # caller whether the page we scraped is the asked product, a variant
    # (different color/finish but same model), a family sibling (same brand,
    # different model — dropped before reaching here), or couldn't be judged.
    match_kind: Optional[str] = Field(
        default=None,
        description="'exact' | 'variant' | 'family' | 'mismatch' | 'unverifiable'. None means identity wasn't checked (pre-2026-04-25 row or identity service disabled).",
    )
    match_score: Optional[int] = Field(
        default=None,
        description="0-100 identity-match confidence. 90+ exact, 70-89 variant, 50-69 family, <50 mismatch.",
    )
    match_note: Optional[str] = Field(
        default=None,
        description="Human-readable note surfacing the facet diff (e.g. 'Color differs: asked BLACK MATT, page shows WHITE MATT'). Null on exact matches.",
    )


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
        query_facets: Optional[QueryFacets] = None,
        manufacturer_hint: Optional[str] = None,
    ) -> PriceSearchResult:
        """
        Run one Sonar search for the given product. Returns PriceSearchResult
        with hits + usage metadata. Stateless — caller handles DB persistence.

        Pipeline:
          1. Facet extraction (Haiku) — decompose the query into brand/model/
             type/variants. Skipped when caller passes pre-cached facets.
          2. Perplexity + DataForSEO in parallel, merged + deduped.
          3. URL pre-filter — drop homepage/SERP/aggregator URLs before
             spending Firecrawl credits on them.
          4. Firecrawl verification — fetch each remaining product page
             and extract price + product_name + breadcrumb + attributes.
          5. Identity classification (batched Haiku) — decide per hit whether
             the page is an exact/variant/family/mismatch/unverifiable
             match for the query. Mismatches are dropped. Variants are kept
             with a human-readable match_note.

        Pass `query_facets` to skip step 1 (caller cached them on the tracked
        query). Pass `manufacturer_hint` to override Haiku's brand guess.
        """
        if not self.api_key:
            return PriceSearchResult(success=False, error="PERPLEXITY_API_KEY not configured")

        limit = max(1, min(limit, 25))

        # Step 1: facets. Prefer caller-supplied cache (tracked_queries or
        # catalog metadata) — only call Haiku when we have nothing.
        identity_svc = get_product_identity_service()
        facets = query_facets
        if facets is None:
            facet_query = product_name
            if dimensions:
                facet_query = f"{product_name} {dimensions}"
            facets = await identity_svc.extract_query_facets(
                facet_query, manufacturer_hint=manufacturer_hint
            )

        # Step 2: Perplexity + DataForSEO + Greek Marketplaces in parallel.
        # Greek Marketplaces is gated on country_code=GR AND the `greek-marketplaces`
        # module being enabled in the modules DB table. Lazy import avoids the
        # circular dependency (greek_marketplaces.service imports PriceHit from here).
        perplexity_task = asyncio.create_task(
            self._perplexity_call(
                product_name, dimensions, country_code, limit,
                preferred_retailer_domains, user_id, workspace_id,
            )
        )
        # Over-fetch DataForSEO (up to 30 merchants) — the Shopping feed
        # routinely has 20-30 retailers per product and the classifier +
        # dedupe trim them down later. Cheap: ~$0.002 flat per task.
        dataforseo_task = asyncio.create_task(
            get_dataforseo_merchant_service().search_shopping(
                product_name=product_name,
                dimensions=dimensions,
                country_code=country_code,
                limit=max(limit, 30),
            )
        )
        greek_task = asyncio.create_task(
            self._greek_marketplaces_call(
                query=f"{product_name} {dimensions}".strip() if dimensions else product_name,
                country_code=country_code,
                user_id=user_id,
                workspace_id=workspace_id,
                limit=limit,
            )
        )
        perplexity_result, dataforseo_result, greek_hits = await asyncio.gather(
            perplexity_task, dataforseo_task, greek_task, return_exceptions=True
        )

        if isinstance(perplexity_result, BaseException):
            logger.warning(f"Perplexity call raised: {perplexity_result}")
            return PriceSearchResult(success=False, error=f"perplexity: {perplexity_result}")

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

        # Merge Greek Marketplaces hits last — they're first-party retailer
        # data and override both Perplexity snippets and DataForSEO feed rows
        # for the same domain.
        if isinstance(greek_hits, BaseException):
            logger.warning(f"Greek marketplaces call raised (non-fatal): {greek_hits}")
        elif greek_hits:
            perplexity_result.hits = self._merge_with_greek_marketplaces(
                existing=perplexity_result.hits,
                greek_hits=greek_hits,
            )

        # Step 3: URL pre-filter. Pure rules, no network. Drops obvious
        # non-product URLs (homepages, SERPs, aggregator masquerades) so
        # Firecrawl budget is spent only on URLs that could actually match.
        # DataForSEO hits are waved through — their URL is SERP-shaped by
        # design, but the Shopping-feed payload is authoritative.
        if perplexity_result.hits:
            kept: List[PriceHit] = []
            for h in perplexity_result.hits:
                verdict = url_prefilter(
                    h.product_url, retailer_name=h.retailer_name, source=h.source
                )
                if verdict.keep:
                    kept.append(h)
                else:
                    logger.debug(f"URL prefilter dropped {h.retailer_name} ({verdict.reason})")
            perplexity_result.hits = kept

        # Extraction details keyed by product_url. Feeds the identity
        # classifier in step 5. Pre-populate with DataForSEO titles so the
        # classifier has product-name signal for merchants that we'll skip
        # Firecrawl-verifying (the Shopping-feed URL isn't scrapable).
        extractions: Dict[str, Dict[str, Any]] = {}
        if not isinstance(dataforseo_result, BaseException) and dataforseo_result.success:
            for m in dataforseo_result.hits:
                if m.product_title:
                    extractions[m.product_url] = {
                        "product_name": m.product_title,
                        "product_breadcrumb": None,
                        "visible_attributes": None,
                    }

        # Step 4: Firecrawl verification. DataForSEO hits skip this — we
        # already have trustworthy price/title/image/rating from the feed,
        # and their URL is typically a Google Shopping redirect that
        # Firecrawl can't scrape meaningfully.
        if verify_prices and perplexity_result.hits:
            perplexity_only = [h for h in perplexity_result.hits if h.source != "dataforseo"]
            if perplexity_only:
                verify_credits = await self._verify_hits_with_firecrawl(
                    perplexity_only,
                    extractions_out=extractions,
                    user_id=user_id,
                    workspace_id=workspace_id,
                )
                perplexity_result.credits_used += verify_credits
            # Mark DataForSEO hits as verified via feed — not Firecrawl, but
            # trusted for price purposes. Keeps the green "Verified" badge
            # honest: the price genuinely came from an authoritative source.
            for h in perplexity_result.hits:
                if h.source == "dataforseo" and not h.verified:
                    h.verified = True
                    h.last_verified = datetime.now(timezone.utc).date().isoformat()

        # Step 5: identity classification. Runs whether or not verification
        # ran — if Firecrawl was skipped, we still classify using URL slug
        # tokens + retailer name as weak signals, and the classifier will
        # mostly return 'unverifiable' verdicts which we keep.
        if perplexity_result.hits and facets:
            perplexity_result.hits = await self._classify_and_filter(
                hits=perplexity_result.hits,
                facets=facets,
                extractions=extractions,
                user_id=user_id,
                workspace_id=workspace_id,
            )

        return perplexity_result

    async def _classify_and_filter(
        self,
        *,
        hits: List[PriceHit],
        facets: QueryFacets,
        extractions: Dict[str, Dict[str, Any]],
        user_id: Optional[str],
        workspace_id: Optional[str],
    ) -> List[PriceHit]:
        """
        Ask the identity classifier which hits are exact/variant/family/
        mismatch/unverifiable, stamp match_kind/score/note onto each hit,
        and drop mismatches. Family matches are dropped per the 2026-04-25
        product decision (same brand ≠ useful pricing data).
        """
        identity_svc = get_product_identity_service()

        candidates: List[Dict[str, Any]] = []
        for h in hits:
            ext = extractions.get(h.product_url) or {}
            candidates.append({
                "retailer": h.retailer_name,
                "url": h.product_url,
                "product_name": ext.get("product_name"),
                "breadcrumb": ext.get("product_breadcrumb"),
                "visible_attributes": ext.get("visible_attributes") or {},
                "url_slug_tokens": url_slug_tokens(h.product_url),
            })

        verdicts = await identity_svc.classify_hits(
            facets, candidates, user_id=user_id, workspace_id=workspace_id,
        )

        kept: List[PriceHit] = []
        for hit, verdict in zip(hits, verdicts):
            kind = verdict.get("match_kind") or "unverifiable"
            hit.match_kind = kind
            hit.match_score = int(verdict.get("match_score") or 0)
            hit.match_note = verdict.get("match_note")
            # Drop mismatches + family per the 2026-04-25 product policy:
            # only exact / variant / unverifiable rows survive.
            if kind in ("mismatch", "family"):
                logger.debug(
                    f"Identity classifier dropped {hit.retailer_name}: {kind} "
                    f"(score={hit.match_score}, note={hit.match_note})"
                )
                continue
            kept.append(hit)
        return kept

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
        extractions_out: Optional[Dict[str, Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> int:
        """
        Second-stage verification: fetch each retailer's actual product page via
        Firecrawl, extract price + product_name + breadcrumb + visible_attributes,
        and rewrite the hit in place. Runs all URLs in parallel via asyncio.gather.

        Mutates `hits` list in place. Also writes extraction details keyed by
        `product_url` into `extractions_out` (when provided) so the identity
        classifier can use them without re-scraping.

        Semantics per row:
          - Firecrawl finds a price on the page → replace `price`, fill
            `original_price` if was/now visible and sane (discarded if
            original < price, or original/price > 5 — that's a SKU/ID, not
            a promo). Set `verified=True`.
          - Firecrawl returns no price (404, blocked, truly missing) → leave
            the row as-is with `verified=False`.
          - Firecrawl price differs by >20% from the Perplexity/DataForSEO
            price → trust Firecrawl + append a discrepancy note.
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
                        f"Extract the following from this {hit.retailer_name} product page:\n"
                        "1. The current main product price (the NOW price if there's a promo, "
                        "   not the related/bundle/strike-through).\n"
                        "2. The on-page 'was' price if a promo is displayed.\n"
                        "3. The product name exactly as shown in the main H1 or og:title.\n"
                        "4. The breadcrumb trail (e.g. 'Home > Bath > Faucets > Basin Faucets').\n"
                        "5. Any visible color/finish/size/material attributes — small dict, lowercase values."
                    ),
                    use_javascript_render=False,
                )
            except Exception as e:
                logger.debug(f"Firecrawl verify crashed for {hit.product_url}: {e}")
                return 0

            if not result.success or not result.data:
                return result.credits_used or 1

            extracted = result.data

            # Save extraction details for the downstream identity classifier
            # even when we can't verify the price (page loaded but no price
            # visible). The classifier still wants product_name to judge
            # identity.
            if extractions_out is not None:
                extractions_out[hit.product_url] = {
                    "product_name": extracted.product_name,
                    "product_breadcrumb": extracted.product_breadcrumb,
                    "visible_attributes": extracted.visible_attributes,
                }

            hint_currency = extracted.currency or hit.currency
            amount, currency = parse_price(extracted.price, hint_currency=hint_currency)
            if amount is None:
                # Page loaded but no extractable price — keep the original data,
                # don't mark as verified. Classifier still gets the extraction.
                return result.credits_used or 1

            verified_price = float(amount)
            original_amount, _ = parse_price(extracted.original_price, hint_currency=hint_currency)
            verified_original = float(original_amount) if original_amount is not None else None

            # original_price sanity bounds:
            #   - must be > current price (otherwise not a promo)
            #   - must not be more than 5× current price (that's a SKU or catalog
            #     number the extractor mistook for a price — Flobali €11,900 case)
            if verified_original is not None:
                if (
                    verified_price <= 0
                    or verified_original <= verified_price
                    or verified_original / verified_price > 5
                ):
                    verified_original = None

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
            if verified_original is not None:
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
        Merge Perplexity + DataForSEO hits.

        Dedupe policy:
          - Perplexity hits keyed by retailer DOMAIN (their URL is a real
            product page — host is unique per retailer).
          - DataForSEO hits keyed by (retailer_name, product_title). Their
            URL is a google.gr/search Shopping redirect, so domain-dedup
            would collapse all 20 merchants into 1. The Shopping feed
            already gives us a distinct product_title per listing, so use
            that as the primary discriminator.
          - When a retailer appears in BOTH sources (Perplexity found the
            direct product page + DataForSEO has it in the feed), keep the
            Perplexity row — it has richer fields (availability, city,
            notes) and a directly-scrapable URL.
          - DataForSEO-only retailers come through with source='dataforseo'
            and their Shopping-feed metadata (image, rating).
        """
        merged: List[PriceHit] = []
        perplexity_domains: set[str] = set()

        for h in perplexity_hits:
            d = cls._domain_of(h.product_url)
            if d:
                perplexity_domains.add(d)
            h.source = h.source or "perplexity"
            merged.append(h)

        # DataForSEO dedup: if we can normalize a retailer domain out of the
        # merchant name it's enough of a signal to dedup against Perplexity.
        # Otherwise, each distinct (merchant, product_title) survives.
        seen_dataforseo: set[Tuple[str, str]] = set()
        for m in dataforseo_hits:
            retailer_key = (m.retailer_name or "").strip().lower()
            title_key = (m.product_title or "")[:80].strip().lower()
            dedup_key = (retailer_key, title_key)
            if dedup_key in seen_dataforseo:
                continue
            seen_dataforseo.add(dedup_key)

            # Skip when the retailer has already been covered by Perplexity.
            # Match either by exact domain (e.g. "youbath" ↔ "youbath.gr") or
            # by retailer name-as-slug (handles "Casa Solutions" → casasolutions).
            retailer_slug = retailer_key.replace(" ", "").replace(".", "")
            if any(retailer_slug and retailer_slug in d.replace(".", "") for d in perplexity_domains):
                continue

            merged.append(PriceHit(
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
                notes=(f"via Google Shopping · {m.product_title}" if m.product_title else "via Google Shopping (DataForSEO)"),
                source="dataforseo",
                image_url=m.image_url,
                rating_value=m.rating_value,
                rating_votes=m.rating_votes,
            ))

        merged.sort(key=lambda h: (h.price if h.price is not None else float("inf")))
        return merged

    @classmethod
    def _merge_with_greek_marketplaces(
        cls,
        existing: List[PriceHit],
        greek_hits: List[PriceHit],
    ) -> List[PriceHit]:
        """
        Merge Greek Marketplaces hits over the existing (Perplexity+DataForSEO)
        set. Greek sources are first-party retailer data, so they OVERRIDE any
        earlier entry for the same retailer domain. New domains are appended.
        """
        by_domain: Dict[str, PriceHit] = {}
        for h in existing:
            d = cls._domain_of(h.product_url)
            if d:
                by_domain[d] = h

        for g in greek_hits:
            d = cls._domain_of(g.product_url)
            if d:
                by_domain[d] = g

        return sorted(
            by_domain.values(),
            key=lambda h: (h.price if h.price is not None else float("inf")),
        )

    async def _greek_marketplaces_call(
        self,
        *,
        query: str,
        country_code: Optional[str],
        user_id: Optional[str],
        workspace_id: Optional[str],
        limit: int,
    ) -> List[PriceHit]:
        """
        Call the Greek Marketplaces module (Skroutz + Bestdeals + Shopflix).
        Gated on country=GR and the DB module toggle. Returns [] when either
        guard fails so the parallel gather doesn't need special handling.
        """
        if (country_code or "").upper() != "GR":
            return []

        from app.modules import is_module_enabled  # lazy: avoid startup circular import
        if not is_module_enabled("greek-marketplaces"):
            return []

        from app.modules.greek_marketplaces.service import get_greek_marketplaces_service
        service = get_greek_marketplaces_service()
        return await service.search(
            query=query,
            country_code=country_code,
            user_id=user_id or "",
            workspace_id=workspace_id,
            limit=limit,
        )

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
