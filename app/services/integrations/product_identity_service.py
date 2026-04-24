"""
Product identity verification for price discovery.

Problem this solves
-------------------
The old pipeline verified that the retailer page had a readable price, but
not that the page had the *right* product. Real-world validation (ORABELLA
PRECIOSA basin faucet, 2026-04-25) surfaced rows like Casasolutionsgekas
pointing to an Orabella shower column — same brand, totally different SKU,
yet flagged "verified" because Firecrawl read a price off the page.

The fix, end-to-end:

  1. Decompose the query into facets ONCE per query (brand, model, type,
     variants). Cached on tracked_queries.query_facets so repeated refreshes
     don't re-pay for the Haiku call.
  2. Pre-filter URLs that can't possibly be product pages — homepages,
     `/search?q=`, `/catalog`, aggregator-wrapped links. Saves Firecrawl
     credits on hits we'd drop later anyway.
  3. Firecrawl scrapes the remaining URLs and now extracts product_name +
     breadcrumb + visible_attributes alongside the price (expanded
     PriceExtraction).
  4. Haiku 4.5 classifies each scraped page against the query facets in a
     SINGLE batched call. Returns match_kind + match_score + match_note
     per hit.
  5. Rows whose match_kind is 'mismatch' get dropped. 'variant' rows stay
     but are labeled so the admin UI can render a "Color differs" badge
     and exclude them from price statistics.

Variant policy (per user decision 2026-04-25)
---------------------------------------------
Same model, different color/finish/size → KEEP with variant annotation.
Different model under the same brand → DROP (family matches are not useful
for pricing). The classifier is soft on finish descriptors (MATT ≈ BLACK
MATT ≈ MATTE BLACK) because retailers don't use consistent language —
hard on brand and model identity.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Greek ↔ Latin normalization (pure rules, no LLM)
# ────────────────────────────────────────────────────────────────────────────
# Covers the visually-identical lookalikes we keep seeing in product codes.
# "7012ΜΤ" (Greek Μ + Τ) and "7012MT" (Latin M + T) must compare equal.
_GREEK_TO_LATIN: Dict[str, str] = {
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "α": "a", "β": "b", "ε": "e", "ζ": "z", "η": "h", "ι": "i", "κ": "k",
    "μ": "m", "ν": "n", "ο": "o", "ρ": "p", "τ": "t", "υ": "y", "χ": "x",
}

# Separators that routinely drift between versions of the same model number:
#   "7012-MT" / "7012 MT" / "7012_MT" / "7012.MT" → normalized as "7012MT"
_MODEL_SEP_RE = re.compile(r"[\s\-_./]+")


def _strip_accents(text: str) -> str:
    """Accent-insensitive compare: 'Νιπτήρα' ≡ 'Νιπτηρα'."""
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def normalize_text(text: Optional[str]) -> str:
    """
    Lowercase, accent-strip, collapse whitespace. Used for general
    facet matching (product type, variants, freeform strings).
    """
    if not text:
        return ""
    return " ".join(_strip_accents(text).lower().split())


def normalize_model_token(token: Optional[str]) -> str:
    """
    Aggressive normalization for model/SKU tokens where we want strict
    equality across Greek/Latin lookalikes + separator drift.

      "7012ΜΤ"   → "7012MT"
      "7012 MT"  → "7012MT"
      "preciosa-01" → "PRECIOSA01"
    """
    if not token:
        return ""
    # Greek → Latin (preserve case for readability, uppercase after)
    mapped = "".join(_GREEK_TO_LATIN.get(ch, ch) for ch in token)
    # Strip accents + uppercase
    mapped = _strip_accents(mapped).upper()
    # Remove separators
    return _MODEL_SEP_RE.sub("", mapped)


# ────────────────────────────────────────────────────────────────────────────
# URL pre-filter — rule-based, no LLM
# ────────────────────────────────────────────────────────────────────────────

# Hosts we know are aggregators. If Perplexity names a retailer but hands us
# an aggregator URL, the hit is worthless — Skroutz/Bestdeals/Shopflix get
# their own first-class adapters.
_AGGREGATOR_HOSTS = {
    "bestprice.gr", "skroutz.gr", "prekmobile.gr",
    "google.com", "google.gr", "google.de", "google.co.uk",
    "shopping.google.com", "idealo.de", "idealo.gr", "idealo.it",
    "pricerunner.com", "kelkoo.com",
}

# Path fragments that betray a listing / search / SERP page rather than a
# specific product page.
_NON_PRODUCT_PATH_MARKERS = (
    "/search", "/catalog", "/category", "/categories",
    "/brand/", "/brands/", "/shop", "/products?",
    "/tag/", "/tags/", "/collection/", "/collections/",
)


@dataclass
class UrlVerdict:
    """Result of pre-filter. `keep=False` → drop without spending Firecrawl."""
    keep: bool
    reason: Optional[str] = None


def url_prefilter(
    url: str,
    retailer_name: Optional[str] = None,
    *,
    source: Optional[str] = None,
) -> UrlVerdict:
    """
    Decide whether a URL is worth Firecrawl-scraping.

    `source` changes what counts as "bad" — DataForSEO hits come with a
    Google Shopping SERP-style URL by design, and their price/title/image/
    rating data is authoritative from the feed. Filtering them would strip
    every DataForSEO merchant from the list, which is the opposite of what
    we want.

    Drops (returns keep=False):
      - Empty / malformed URLs
      - Paths of "/" (bare homepage)
      - SERP / search / catalog / brand index pages
        (skipped when source == 'dataforseo' — those URLs ARE SERP-shaped
        but carry trusted Shopping-feed data)
      - Aggregator-hosted URLs when the row claims to be a different retailer
        (also skipped when source == 'dataforseo')
      - Very short slugs (<8 chars after the last "/") — homepages in disguise
    """
    if not url or not url.strip():
        return UrlVerdict(False, "empty URL")

    try:
        parsed = urlparse(url.strip())
    except Exception:
        return UrlVerdict(False, "unparseable URL")

    host = (parsed.hostname or "").lower().lstrip(".")
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path or "/"
    query = parsed.query or ""

    if not host:
        return UrlVerdict(False, "no host")

    # DataForSEO Shopping-feed hits come with their own trustworthy payload
    # (title, price, image, rating from the feed). Let them through without
    # any path/SERP checks — the classifier + UI will handle them differently
    # from free-web Perplexity hits.
    is_dataforseo = source == "dataforseo"

    if not is_dataforseo:
        if path in ("", "/"):
            return UrlVerdict(False, "homepage URL")

        path_lower = path.lower()
        for marker in _NON_PRODUCT_PATH_MARKERS:
            if marker in path_lower:
                return UrlVerdict(False, f"listing/search path ({marker})")

        if "ibp=oshop" in query or "tbm=shop" in query:
            return UrlVerdict(False, "Google Shopping SERP URL")

        if retailer_name:
            retailer_host_guess = normalize_text(retailer_name).replace(" ", "").replace(".", "")
            for agg_host in _AGGREGATOR_HOSTS:
                agg_key = agg_host.split(".")[0]
                if agg_key in host and agg_key not in retailer_host_guess:
                    return UrlVerdict(False, f"aggregator host {agg_host} claimed as {retailer_name}")

        last_segment = [p for p in path.split("/") if p]
        if last_segment and len(last_segment[-1]) < 4:
            return UrlVerdict(False, "URL slug too short")

    return UrlVerdict(True)


def url_slug_tokens(url: str) -> List[str]:
    """
    Extract identity-bearing tokens from a URL slug as a fallback signal when
    Firecrawl can't extract product_name. 'apothema.gr/maidtec-by-pyramis-7012mt-105584p'
    → ['MAIDTEC', 'BY', 'PYRAMIS', '7012MT', '105584P'].
    """
    if not url:
        return []
    try:
        path = urlparse(url).path or ""
    except Exception:
        return []
    segments = [p for p in path.split("/") if p]
    slug = segments[-1] if segments else ""
    # Strip common page suffixes
    slug = re.sub(r"\.(html?|php|aspx?)$", "", slug, flags=re.I)
    # Split on hyphens/underscores/dots
    tokens = re.split(r"[-_.]+", slug)
    return [normalize_model_token(t) for t in tokens if t and len(t) >= 2]


# ────────────────────────────────────────────────────────────────────────────
# Query facets — what the user is actually asking for
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class QueryFacets:
    """
    Decomposition of a user query into identity-bearing facets. Extracted
    once per query by Haiku (or populated directly from catalog metadata
    when available) and cached on tracked_queries.query_facets.
    """
    brand: Optional[str] = None              # "ORABELLA"
    model: Optional[str] = None              # "PRECIOSA"
    product_type: Optional[str] = None       # "basin_faucet"
    variants: Dict[str, str] = field(default_factory=dict)  # {"color":"BLACK","finish":"MATT"}
    required_tokens: List[str] = field(default_factory=list)   # must appear in page (brand + model)
    variant_tokens: List[str] = field(default_factory=list)    # matching boosts, mismatch → note
    raw_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brand": self.brand,
            "model": self.model,
            "product_type": self.product_type,
            "variants": self.variants,
            "required_tokens": self.required_tokens,
            "variant_tokens": self.variant_tokens,
            "raw_query": self.raw_query,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional["QueryFacets"]:
        if not d:
            return None
        return cls(
            brand=d.get("brand"),
            model=d.get("model"),
            product_type=d.get("product_type"),
            variants=d.get("variants") or {},
            required_tokens=d.get("required_tokens") or [],
            variant_tokens=d.get("variant_tokens") or [],
            raw_query=d.get("raw_query") or "",
        )


def facets_from_catalog(product_row: Optional[Dict[str, Any]]) -> Optional[QueryFacets]:
    """
    When /discover runs for a catalog product we already have structured
    metadata — use it as the reference facets rather than re-parsing the
    free-text name. Much cleaner signal than Haiku guessing from a name.
    """
    if not product_row:
        return None
    meta = product_row.get("metadata") or {}
    name = product_row.get("name") or ""
    brand = meta.get("manufacturer") or meta.get("brand")
    model = meta.get("model") or meta.get("model_name") or meta.get("sku")

    variants: Dict[str, str] = {}
    for k in ("color", "finish", "size", "dimensions"):
        v = meta.get(k)
        if v:
            variants[k] = str(v)

    required_tokens: List[str] = []
    if brand:
        required_tokens.append(normalize_model_token(brand))
    if model:
        required_tokens.append(normalize_model_token(model))

    variant_tokens = [normalize_text(v) for v in variants.values() if v]

    if not required_tokens and not name:
        return None

    return QueryFacets(
        brand=brand,
        model=model,
        product_type=meta.get("product_type") or meta.get("category"),
        variants=variants,
        required_tokens=required_tokens,
        variant_tokens=variant_tokens,
        raw_query=name,
    )


# ────────────────────────────────────────────────────────────────────────────
# Anthropic Haiku client — query decomposition + batched classifier
# ────────────────────────────────────────────────────────────────────────────

_ANTHROPIC_BASE = "https://api.anthropic.com/v1/messages"
_MODEL = "claude-haiku-4-5-20251001"
_ANTHROPIC_VERSION = "2023-06-01"


def _anthropic_headers() -> Dict[str, str]:
    key = os.getenv("ANTHROPIC_API_KEY") or ""
    return {
        "x-api-key": key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


_FACET_EXTRACTION_SYSTEM = (
    "You decompose product search queries into identity facets for a price-monitoring system. "
    "Return ONLY a JSON object matching this schema exactly:\n"
    "{\n"
    "  \"brand\": string | null,            // manufacturer brand, e.g. 'ORABELLA', 'MAIDTEC'\n"
    "  \"model\": string | null,            // model name/code/series, e.g. 'PRECIOSA', '7012MT'\n"
    "  \"product_type\": string | null,     // normalized category, e.g. 'basin_faucet', 'tile', 'range_hood'\n"
    "  \"variants\": {                      // visible product variant attributes\n"
    "    \"color\": string | null,\n"
    "    \"finish\": string | null,         // matt, gloss, satin, brushed, polished\n"
    "    \"size\": string | null\n"
    "  },\n"
    "  \"required_tokens\": [string],       // brand + model, MUST be on the page\n"
    "  \"variant_tokens\": [string]         // color/finish/size, soft match\n"
    "}\n\n"
    "Rules:\n"
    "- brand and model are the identity-bearing tokens. If the query only has a brand, model is null.\n"
    "- variants are SOFT descriptors. A product with different finish is a VARIANT, not a different product.\n"
    "- MATT and MATTE are the same. MATT BLACK = BLACK MATT = MATTE BLACK (same color+finish). Keep them unified.\n"
    "- product_type uses snake_case with common English category names, even if the query is in Greek.\n"
    "  Examples: 'Μπαταρία Νιπτήρα'→'basin_faucet', 'Απορροφητήρας'→'range_hood', 'Πλακάκι'→'tile'.\n"
    "- If a manufacturer is explicitly provided as a separate hint, honor it as the brand.\n"
    "- Return NULL for fields you can't confidently extract. Never guess."
)


_CLASSIFIER_SYSTEM = (
    "You classify whether scraped retailer pages match the product the user asked for. "
    "This is for a price-monitoring system — mismatches waste money and mislead admins.\n\n"
    "For EACH candidate page, compare its extracted facets against the query facets and "
    "return a classification. Reply ONLY with a JSON object:\n"
    "{ \"verdicts\": [ { \"match_kind\": \"exact\"|\"variant\"|\"family\"|\"mismatch\"|\"unverifiable\", "
    "\"match_score\": integer 0-100, \"variant_diffs\": [{\"facet\":string,\"asked\":string,\"found\":string}], "
    "\"match_note\": string|null } ] }\n\n"
    "Classification rules (strict order):\n"
    "- unverifiable: page_name is empty AND url_slug_tokens is empty. Can't judge.\n"
    "- mismatch (<50): brand differs, OR model differs (even if brand matches — that's a different SKU), "
    "  OR product_type differs (a shower column is NOT a basin faucet even with same brand).\n"
    "- family (50-69): brand matches, model differs but is in the same product family. RARELY used — "
    "  when in doubt between family and mismatch, prefer mismatch.\n"
    "- variant (70-89): brand + model + product_type all match, but color/finish/size differs.\n"
    "  Also use for bundles/sets that CONTAIN the asked product but include other items.\n"
    "- exact (90+): brand + model + product_type match, and any visible variants are consistent.\n\n"
    "Soft matching rules (be generous here):\n"
    "- MATT / MATTE / MAT are the same finish. BLACK MATT ≡ MATT BLACK ≡ MATTE BLACK.\n"
    "- GLOSS / GLOSSY / SHINY are the same finish. Missing finish in page name is OK if the asked finish "
    "  is a common default (GLOSS often unstated).\n"
    "- Greek/Latin lookalikes (Μ/M, Τ/T, Α/A, etc.) in model codes are always equivalent.\n"
    "- Model-code separators are noise: 7012-MT = 7012 MT = 7012MT.\n"
    "- Accent differences in Greek are noise: Νιπτήρα = Νιπτηρα.\n"
    "- Extra descriptive words in the page title are fine when required tokens are present "
    "  ('series PRECIOSA collection basin faucet' still matches query 'ORABELLA PRECIOSA').\n\n"
    "match_note guidance:\n"
    "- NULL for exact matches.\n"
    "- For variant: one-line English describing the facet diff, e.g. 'Color differs: asked BLACK MATT, page shows WHITE MATT'.\n"
    "- For bundle/set: 'Bundled with X' (where X is the other product visible on the page).\n"
    "- For family: 'Same brand, different model — asked PRECIOSA, page shows BELLA'.\n"
    "- For mismatch: short explanation — 'Different product type: shower column, not basin faucet'.\n"
    "- For unverifiable: 'Could not extract product identity from page'."
)


class ProductIdentityService:
    """
    Query facet extraction + batched identity classification via Haiku.

    Callers:
        svc = get_product_identity_service()
        facets = await svc.extract_query_facets("ORABELLA PRECIOSA Μπαταρία Νιπτήρα BLACK MATT")
        verdicts = await svc.classify_hits(facets, [page_facets, ...])
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("ANTHROPIC_API_KEY") or ""
        self.supabase = get_supabase_client()
        self._http_timeout = httpx.Timeout(60.0, connect=10.0)
        if not self.api_key:
            logger.warning("⚠️ ANTHROPIC_API_KEY not configured — identity classification disabled")

    # ── Query decomposition ──

    async def extract_query_facets(
        self,
        query: str,
        *,
        manufacturer_hint: Optional[str] = None,
    ) -> Optional[QueryFacets]:
        """
        Decompose a free-text search query into structured facets. Returns
        None on failure — callers should fall back to a minimal facets object
        (required_tokens = capitalized tokens from the query).
        """
        if not self.api_key or not query:
            return None

        user_prompt_parts = [f"Query: {query}"]
        if manufacturer_hint:
            user_prompt_parts.append(f"Explicit manufacturer hint: {manufacturer_hint}")
        user_prompt = "\n".join(user_prompt_parts)

        try:
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                resp = await client.post(
                    _ANTHROPIC_BASE,
                    headers=_anthropic_headers(),
                    json={
                        "model": _MODEL,
                        "max_tokens": 500,
                        "system": [
                            {
                                "type": "text",
                                "text": _FACET_EXTRACTION_SYSTEM,
                                "cache_control": {"type": "ephemeral"},
                            },
                        ],
                        "messages": [{"role": "user", "content": user_prompt}],
                    },
                )
                resp.raise_for_status()
                body = resp.json()
        except Exception as e:
            logger.warning(f"Haiku facet extraction failed for '{query}': {e}")
            return None

        parsed = _extract_json_content(body)
        if not parsed:
            return None

        # Build QueryFacets with normalized tokens. Honor explicit
        # manufacturer_hint — it overrides Haiku's guess because the caller
        # knows better.
        brand = manufacturer_hint or parsed.get("brand") or None
        model = parsed.get("model") or None
        variants = {k: v for k, v in (parsed.get("variants") or {}).items() if v}

        required_tokens: List[str] = []
        if brand:
            required_tokens.append(normalize_model_token(brand))
        if model:
            required_tokens.append(normalize_model_token(model))

        variant_tokens: List[str] = []
        for v in variants.values():
            variant_tokens.append(normalize_text(v))

        return QueryFacets(
            brand=brand,
            model=model,
            product_type=parsed.get("product_type"),
            variants=variants,
            required_tokens=[t for t in required_tokens if t],
            variant_tokens=[t for t in variant_tokens if t],
            raw_query=query,
        )

    # ── Batched identity classification ──

    async def classify_hits(
        self,
        facets: QueryFacets,
        candidates: List[Dict[str, Any]],
        *,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Classify a batch of scraped pages against the query facets.

        `candidates` is a list of dicts shaped like:
            {
              "retailer": "<retailer_name>",
              "url": "<product_url>",
              "product_name": str | None,
              "breadcrumb": str | None,
              "visible_attributes": dict | None,
              "url_slug_tokens": [str, ...],
            }

        Returns a list aligned 1:1 with candidates; each verdict is:
            {"match_kind": str, "match_score": int, "variant_diffs": list, "match_note": str | None}

        When the API key is missing or the call fails, falls back to a
        rule-based classifier so the pipeline degrades gracefully rather
        than dropping every row.
        """
        if not candidates:
            return []

        if not self.api_key:
            return [self._rule_based_verdict(facets, c) for c in candidates]

        user_payload = {
            "query_facets": facets.to_dict(),
            "pages": candidates,
        }

        try:
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                resp = await client.post(
                    _ANTHROPIC_BASE,
                    headers=_anthropic_headers(),
                    json={
                        "model": _MODEL,
                        "max_tokens": 2000,
                        "system": [
                            {
                                "type": "text",
                                "text": _CLASSIFIER_SYSTEM,
                                "cache_control": {"type": "ephemeral"},
                            },
                        ],
                        "messages": [
                            {
                                "role": "user",
                                "content": json.dumps(user_payload, ensure_ascii=False),
                            }
                        ],
                    },
                )
                resp.raise_for_status()
                body = resp.json()
        except Exception as e:
            logger.warning(f"Haiku identity classifier failed: {e}. Falling back to rule-based.")
            return [self._rule_based_verdict(facets, c) for c in candidates]

        parsed = _extract_json_content(body)
        verdicts = (parsed or {}).get("verdicts") if isinstance(parsed, dict) else None
        if not isinstance(verdicts, list) or len(verdicts) != len(candidates):
            logger.warning(
                f"Haiku classifier returned misshapen response: expected {len(candidates)} verdicts, got "
                f"{len(verdicts) if isinstance(verdicts, list) else 'non-list'}. Falling back to rule-based."
            )
            return [self._rule_based_verdict(facets, c) for c in candidates]

        # Audit-log the classifier decision for traceability.
        self._log_classifier_call(
            facets=facets,
            candidates=candidates,
            verdicts=verdicts,
            user_id=user_id,
            workspace_id=workspace_id,
        )

        # Sanity: clamp score to 0-100, enforce valid match_kind.
        cleaned: List[Dict[str, Any]] = []
        for v in verdicts:
            kind = v.get("match_kind") if isinstance(v, dict) else None
            if kind not in ("exact", "variant", "family", "mismatch", "unverifiable"):
                kind = "unverifiable"
            try:
                score = int(v.get("match_score", 0))
            except Exception:
                score = 0
            score = max(0, min(100, score))
            cleaned.append({
                "match_kind": kind,
                "match_score": score,
                "variant_diffs": v.get("variant_diffs") or [],
                "match_note": v.get("match_note"),
            })
        return cleaned

    # ── Fallback rule-based classifier ──

    def _rule_based_verdict(
        self, facets: QueryFacets, candidate: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Minimal fallback when Haiku is unavailable. Uses required_tokens ∩
        (product_name ∪ url_slug_tokens). Conservative: biases toward
        'unverifiable' rather than false-exact.
        """
        required = {normalize_model_token(t) for t in (facets.required_tokens or []) if t}
        if not required:
            # No required tokens → can't classify. Mark unverifiable and keep.
            return {
                "match_kind": "unverifiable",
                "match_score": 50,
                "variant_diffs": [],
                "match_note": "No required tokens from query",
            }

        name = candidate.get("product_name") or ""
        slug = " ".join(candidate.get("url_slug_tokens") or [])
        haystack = normalize_model_token(f"{name} {slug}")

        matches = [t for t in required if t in haystack]
        if len(matches) == len(required):
            return {
                "match_kind": "exact",
                "match_score": 90,
                "variant_diffs": [],
                "match_note": None,
            }
        if matches:
            return {
                "match_kind": "family",
                "match_score": 55,
                "variant_diffs": [],
                "match_note": f"Partial token match ({len(matches)}/{len(required)})",
            }
        if not name and not candidate.get("url_slug_tokens"):
            return {
                "match_kind": "unverifiable",
                "match_score": 40,
                "variant_diffs": [],
                "match_note": "Could not extract product identity from page",
            }
        return {
            "match_kind": "mismatch",
            "match_score": 20,
            "variant_diffs": [],
            "match_note": f"Required tokens {list(required)} not found on page",
        }

    # ── Usage logging ──

    def _log_classifier_call(
        self,
        *,
        facets: QueryFacets,
        candidates: List[Dict[str, Any]],
        verdicts: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> None:
        """
        Persist a single row to ai_usage_logs for auditability. Non-critical —
        never raises; logs on failure only.
        """
        try:
            self.supabase.client.table("ai_usage_logs").insert({
                "user_id": user_id,
                "workspace_id": workspace_id,
                "provider": "anthropic",
                "model": _MODEL,
                "operation_type": "product_match_classifier",
                "metadata": {
                    "query_facets": facets.to_dict(),
                    "candidates": candidates,
                    "verdicts": verdicts,
                },
            }).execute()
        except Exception as e:
            logger.debug(f"classifier usage log skipped: {e}")


def _extract_json_content(anthropic_body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Pull the JSON object out of an Anthropic messages API response.
    Haiku usually returns clean JSON; tolerate the occasional code fence.
    """
    try:
        blocks = anthropic_body.get("content") or []
        text = ""
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                text += b.get("text") or ""
        if not text:
            return None
        # Strip markdown fences if the model wrapped the JSON
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        else:
            # Grab the first balanced JSON object
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]
        return json.loads(text)
    except Exception as e:
        logger.warning(f"Failed to parse Haiku JSON response: {e}")
        return None


# Module-level singleton
_service: Optional[ProductIdentityService] = None


def get_product_identity_service() -> ProductIdentityService:
    global _service
    if _service is None:
        _service = ProductIdentityService()
    return _service
