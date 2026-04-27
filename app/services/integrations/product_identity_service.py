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

import asyncio
import json
import logging
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
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
# an aggregator URL, the hit is worthless — Skroutz/Bestprice/Shopflix get
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

    # Trusted sources bypass path/SERP/aggregator checks:
    #   - "dataforseo": Shopping-feed payload is authoritative; URL is SERP-shaped by design.
    #   - "skroutz" / "bestprice" / "shopflix": Greek marketplace adapters legitimately
    #     return deep-link URLs on the marketplace host (e.g. bestprice.gr/to/<id>) with
    #     the underlying merchant in retailer_name. Blocking these as "aggregator
    #     masquerading as merchant" loses every marketplace hit.
    is_trusted_source = source in {"dataforseo", "skroutz", "bestprice", "shopflix"}

    if not is_trusted_source:
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
    sku_tokens: List[str] = field(default_factory=list)        # digit-anchor SKU codes (catalog or query)
    raw_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brand": self.brand,
            "model": self.model,
            "product_type": self.product_type,
            "variants": self.variants,
            "required_tokens": self.required_tokens,
            "variant_tokens": self.variant_tokens,
            "sku_tokens": self.sku_tokens,
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
            sku_tokens=d.get("sku_tokens") or [],
            raw_query=d.get("raw_query") or "",
        )


def facets_from_catalog(product_row: Optional[Dict[str, Any]]) -> Optional[QueryFacets]:
    """
    When /discover runs for a catalog product we already have structured
    metadata — use it as the reference facets rather than re-parsing the
    free-text name. Much cleaner signal than Haiku guessing from a name.

    Reads the real metadata layout produced by the PDF ingestion pipeline:
      - metadata.factory_name      → brand
      - metadata.collection         → model/series name
      - metadata.commercial.sku_codes (dict of variant_name → sku)
        + metadata.commercial.vision_variants[].sku → digit-anchor SKU tokens
      - metadata.material_category  → product_type
      - metadata.material_properties.{finish}, available_colors, dimensions → variants

    Forward-compatible: still honors flat metadata.manufacturer/brand/model/sku
    keys if a future ingestion path writes them at the top level.
    """
    if not product_row:
        return None
    meta = product_row.get("metadata") or {}
    name = product_row.get("name") or ""

    brand = (
        meta.get("factory_name")
        or meta.get("manufacturer")
        or meta.get("brand")
    )
    model = (
        meta.get("collection")
        or meta.get("model")
        or meta.get("model_name")
    )

    # Pull every catalog SKU we know about. These are the strongest identity
    # anchors — when a retailer page carries a SKU code that's NOT in this
    # set, the page is a different SKU within the same collection.
    sku_tokens: List[str] = []
    commercial = meta.get("commercial") or {}
    sku_codes = commercial.get("sku_codes") or {}
    if isinstance(sku_codes, dict):
        for v in sku_codes.values():
            if v:
                sku_tokens.append(normalize_model_token(str(v)))
    vision_variants = commercial.get("vision_variants") or []
    if isinstance(vision_variants, list):
        for vv in vision_variants:
            if isinstance(vv, dict) and vv.get("sku"):
                sku_tokens.append(normalize_model_token(str(vv["sku"])))
    # Top-level sku field (forward-compat for non-PDF ingestion paths)
    if meta.get("sku"):
        sku_tokens.append(normalize_model_token(str(meta["sku"])))
    # Dedup, preserve order, drop empties / pure-name tokens (require ≥1 digit)
    seen: set = set()
    deduped: List[str] = []
    for t in sku_tokens:
        if t and any(c.isdigit() for c in t) and t not in seen:
            seen.add(t)
            deduped.append(t)
    sku_tokens = deduped

    variants: Dict[str, str] = {}
    mp = meta.get("material_properties") or {}
    if mp.get("finish"):
        variants["finish"] = str(mp["finish"])
    # Single-value color/size only when unambiguous — multi-color products
    # leave variants empty so the classifier doesn't false-mismatch.
    available_colors = meta.get("available_colors") or []
    if isinstance(available_colors, list) and len(available_colors) == 1:
        variants["color"] = str(available_colors[0])
    dims = meta.get("dimensions") or []
    if isinstance(dims, list) and dims:
        first = dims[0] if isinstance(dims[0], dict) else None
        if first and first.get("metric_cm"):
            variants["size"] = str(first["metric_cm"])
    # Forward-compat flat keys
    for k in ("color", "size"):
        v = meta.get(k)
        if v and k not in variants:
            variants[k] = str(v)

    required_tokens: List[str] = []
    if brand:
        required_tokens.append(normalize_model_token(brand))
    if model:
        required_tokens.append(normalize_model_token(model))

    variant_tokens = [normalize_text(v) for v in variants.values() if v]

    if not required_tokens and not name and not sku_tokens:
        return None

    return QueryFacets(
        brand=brand,
        model=model,
        product_type=(
            meta.get("product_type")
            or meta.get("material_category")
            or meta.get("category")
        ),
        variants=variants,
        required_tokens=required_tokens,
        variant_tokens=variant_tokens,
        sku_tokens=sku_tokens,
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
    "  \"model\": string | null,            // model name/series, e.g. 'PRECIOSA', 'VALENOVA'\n"
    "  \"product_type\": string | null,     // normalized category, e.g. 'basin_faucet', 'tile', 'range_hood'\n"
    "  \"variants\": {                      // visible product variant attributes\n"
    "    \"color\": string | null,\n"
    "    \"finish\": string | null,         // matt, gloss, satin, brushed, polished\n"
    "    \"size\": string | null\n"
    "  },\n"
    "  \"required_tokens\": [string],       // brand + model, MUST be on the page\n"
    "  \"variant_tokens\": [string],        // color/finish/size, soft match\n"
    "  \"sku_tokens\": [string]             // exact SKU/article codes — digit-bearing identity anchors\n"
    "}\n\n"
    "Rules:\n"
    "- brand and model are identity-bearing. If the query only has a brand, model is null.\n"
    "- sku_tokens contains every distinct SKU/model/article code (alphanumeric, USUALLY contains digits)\n"
    "  visible in the query, e.g. '10202', '7012MT', '39659', 'PRECIOSA-10259'. Strip separators on output\n"
    "  ('7012-MT' → '7012MT'). When the query carries a SKU, it is THE identity anchor; brand+model alone\n"
    "  are not enough. If no SKU is visible, return [].\n"
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
    "- mismatch (<50): brand differs, OR product_type differs from the query's primary product_type "
    "  (a shower outlet/column is NOT a basin faucet even with same brand+series; an εκροή/spout is "
    "  NOT a μπαταρία/mixer faucet — those are different SKUs sold separately).\n"
    "- family (50-69): brand+series match but the page is a DIFFERENT SKU within that series. Use this "
    "  WHENEVER the page name carries an explicit numeric SKU code (e.g. 'PRECIOSA 10259', '10159') and "
    "  either (a) the query has no SKU, or (b) the page SKU differs from the query SKU. Same series ≠ same product.\n"
    "  Also use when product_type words differ (Νιπτήρα/basin vs Ντουζιέρα/shower vs Λουτρού/bath, "
    "  Μπαταρία/mixer vs Εκροή/spout-outlet vs Στήλη/column).\n"
    "- variant (70-89): brand + model + product_type all match, only color/finish/size differs.\n"
    "  Reserved for genuine same-SKU variants (chrome vs black-matt of the SAME mixer).\n"
    "  Also use for bundles/sets that CONTAIN the asked product but include other items.\n"
    "- exact (90+): brand + model + product_type match, and any visible variants are consistent.\n\n"
    "Tie-breakers (apply BEFORE picking variant):\n"
    "- query_facets.sku_tokens is the strongest identity anchor. When non-empty, the page MUST contain at "
    "  least one of those SKU tokens (in product_name, breadcrumb, visible_attributes, or url_slug_tokens, "
    "  ignoring case and Greek/Latin lookalikes and separators) — otherwise the verdict is `family` if the "
    "  brand+series still match, else `mismatch`. Page SKUs that look like sku_tokens but aren't equal "
    "  (e.g. asked '10202', page shows '10259') always force `family` (or `mismatch` if product_type also "
    "  differs).\n"
    "- If sku_tokens is EMPTY but the page name contains a SKU/model code (any digit-bearing alphanumeric "
    "  near the brand/series, e.g. 'PRECIOSA 10259', '#39661'), treat the page as a different SKU. Default "
    "  to `family` unless product_type also differs (then `mismatch`).\n"
    "- 'Same product type' means the same Greek/English noun for the device on the page: "
    "  Μπαταρία≈Faucet/Mixer/Tap; Εκροή≈Spout/Outlet (a separate SKU, NOT a faucet); "
    "  Στήλη≈Column; Σύστημα Ντους≈Shower System. Different noun → different product_type.\n"
    "- Never label `variant` purely on brand+series match. The match_note must not contradict the verdict — "
    "  if the note says 'different product type' or 'different SKU', the verdict MUST be family or mismatch.\n\n"
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
    "- For family: 'Same series, different SKU — asked PRECIOSA (no SKU), page shows PRECIOSA 10259 shower outlet'.\n"
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

        # SKU tokens — keep only digit-bearing codes (model names like 'PRECIOSA'
        # already live in required_tokens; we don't want them double-counted as
        # SKU anchors). Dedup, preserve order.
        sku_tokens: List[str] = []
        seen: set = set()
        for raw in (parsed.get("sku_tokens") or []):
            t = normalize_model_token(str(raw))
            if t and any(c.isdigit() for c in t) and t not in seen:
                seen.add(t)
                sku_tokens.append(t)

        return QueryFacets(
            brand=brand,
            model=model,
            product_type=parsed.get("product_type"),
            variants=variants,
            required_tokens=[t for t in required_tokens if t],
            variant_tokens=[t for t in variant_tokens if t],
            sku_tokens=sku_tokens,
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

        # PR 4b: pull recent admin corrections as few-shot examples that get
        # prepended to the system prompt. Closes the loop without retraining
        # a model — admin clicks "this is wrong, should be `mismatch`" in the
        # UI, the classifier sees that example on the next refresh.
        few_shot_block = self._build_few_shot_block()

        # 2026-04-27: chunk large batches to avoid Haiku JSON truncation. We
        # observed misshapen responses for ≥25 candidates because the JSON
        # output sometimes exceeded max_tokens or the model trailed a stray
        # token past the closing brace. 12 per batch is a comfortable ceiling
        # — produces ≤ ~1.2K output tokens per call, leaves headroom on
        # max_tokens=3000, and runs the chunks in parallel so end-to-end
        # latency is unchanged.
        BATCH_SIZE = 12
        if len(candidates) > BATCH_SIZE:
            chunks = [candidates[i:i + BATCH_SIZE] for i in range(0, len(candidates), BATCH_SIZE)]
            sub_results = await asyncio.gather(
                *(self._classify_chunk(facets, chunk, few_shot_block) for chunk in chunks),
                return_exceptions=True,
            )
            verdicts: List[Dict[str, Any]] = []
            for i, sub in enumerate(sub_results):
                if isinstance(sub, BaseException):
                    logger.warning(f"Haiku chunk {i} crashed: {sub} — rule-based fallback for this chunk.")
                    verdicts.extend(self._rule_based_verdict(facets, c) for c in chunks[i])
                else:
                    verdicts.extend(sub)
            self._log_classifier_call(
                facets=facets, candidates=candidates, verdicts=verdicts,
                user_id=user_id, workspace_id=workspace_id,
            )
            return self._sanitize_verdicts(verdicts)

        verdicts = await self._classify_chunk(facets, candidates, few_shot_block)

        # Audit-log the classifier decision for traceability.
        self._log_classifier_call(
            facets=facets,
            candidates=candidates,
            verdicts=verdicts,
            user_id=user_id,
            workspace_id=workspace_id,
        )

        return self._sanitize_verdicts(verdicts)

    @staticmethod
    def _sanitize_verdicts(verdicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clamp score to 0-100, enforce valid match_kind."""
        cleaned: List[Dict[str, Any]] = []
        for v in verdicts:
            if not isinstance(v, dict):
                cleaned.append({"match_kind": "unverifiable", "match_score": 50, "variant_diffs": [], "match_note": None})
                continue
            kind = v.get("match_kind")
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

    async def _classify_chunk(
        self,
        facets: QueryFacets,
        candidates: List[Dict[str, Any]],
        few_shot_block: str,
    ) -> List[Dict[str, Any]]:
        """
        One Haiku call for a (small) batch. Returns sanitized verdicts aligned
        with `candidates`. Falls back to rule-based when JSON parse / shape
        fails. Output is NOT yet log-audited — caller logs once after merging
        all chunks.
        """
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
                        # 3000 leaves comfortable headroom for ~12 verdicts of
                        # ~80 tokens each. Even the chunked path keeps this
                        # cap so a single chunk never truncates.
                        "max_tokens": 3000,
                        "system": [
                            {
                                "type": "text",
                                "text": _CLASSIFIER_SYSTEM + few_shot_block,
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
            logger.warning(f"Haiku classifier chunk failed: {e}. Falling back to rule-based.")
            return [self._rule_based_verdict(facets, c) for c in candidates]

        parsed = _extract_json_content(body)
        verdicts = (parsed or {}).get("verdicts") if isinstance(parsed, dict) else None
        # Strict shape check: list of N verdicts.
        if not isinstance(verdicts, list) or len(verdicts) != len(candidates):
            logger.warning(
                f"Haiku chunk returned misshapen response: expected {len(candidates)} verdicts, got "
                f"{len(verdicts) if isinstance(verdicts, list) else 'non-list'}. Falling back to rule-based for chunk."
            )
            return [self._rule_based_verdict(facets, c) for c in candidates]

        return self._sanitize_verdicts(verdicts)

    # ── Verdict cache (PR-C #4) ──
    # 7-day TTL per (URL, facets_hash). Skips Haiku when we've already
    # classified the same URL against the same identity facets recently.

    def classifier_cache_lookup(
        self,
        *,
        product_urls: List[str],
        facets_hash: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Batch lookup. Returns {url: verdict_dict} for unexpired entries."""
        if not product_urls or not facets_hash:
            return {}
        try:
            resp = (
                self.supabase.client.table("classifier_verdict_cache")
                .select("product_url, match_kind, match_score, match_note, variant_diffs, expires_at")
                .in_("product_url", product_urls)
                .eq("facets_hash", facets_hash)
                .gt("expires_at", datetime.now(timezone.utc).isoformat())
                .execute()
            )
        except Exception as e:
            logger.debug(f"classifier_cache_lookup failed (non-fatal): {e}")
            return {}
        result: Dict[str, Dict[str, Any]] = {}
        for r in (resp.data or []):
            url = r.get("product_url")
            if not url:
                continue
            result[url] = {
                "match_kind": r.get("match_kind") or "unverifiable",
                "match_score": int(r.get("match_score") or 0),
                "match_note": r.get("match_note"),
                "variant_diffs": r.get("variant_diffs") or [],
            }
        return result

    def classifier_cache_upsert(
        self,
        *,
        product_urls: List[str],
        facets_hash: str,
        verdicts: List[Dict[str, Any]],
    ) -> None:
        """Best-effort persistence of fresh verdicts for 7 days."""
        if not product_urls or not facets_hash or not verdicts:
            return
        if len(product_urls) != len(verdicts):
            logger.warning("classifier_cache_upsert length mismatch — skipping persist")
            return
        expires = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        rows = []
        for url, v in zip(product_urls, verdicts):
            if not url:
                continue
            rows.append({
                "product_url": url,
                "facets_hash": facets_hash,
                "match_kind": v.get("match_kind") or "unverifiable",
                "match_score": int(v.get("match_score") or 0),
                "match_note": v.get("match_note"),
                "variant_diffs": v.get("variant_diffs") or [],
                "expires_at": expires,
            })
        if not rows:
            return
        try:
            self.supabase.client.table("classifier_verdict_cache").upsert(
                rows, on_conflict="product_url,facets_hash"
            ).execute()
        except Exception as e:
            logger.debug(f"classifier_cache_upsert failed (non-fatal): {e}")

    # ── Fallback rule-based classifier ──

    def _build_few_shot_block(self) -> str:
        """
        PR 4b: pull the most recent admin corrections from `match_corrections`
        and format them as a few-shot block to prepend to the classifier
        system prompt. Capped at 5 examples to keep token cost bounded.

        Cached in-process for 5 minutes so we don't query the DB on every
        classify call.
        """
        cached = getattr(self, "_few_shot_cache", None)
        if cached is not None:
            ts, block = cached
            if (datetime.now(timezone.utc) - ts).total_seconds() < 300:
                return block

        try:
            resp = (
                self.supabase.client.table("match_corrections")
                .select("query_facets, page_facets, original_match_kind, corrected_match_kind, correction_note")
                .order("created_at", desc=True)
                .limit(5)
                .execute()
            )
        except Exception as e:
            logger.debug(f"few-shot fetch skipped: {e}")
            block = ""
            self._few_shot_cache = (datetime.now(timezone.utc), block)
            return block

        rows = resp.data or []
        if not rows:
            block = ""
            self._few_shot_cache = (datetime.now(timezone.utc), block)
            return block

        examples: List[str] = []
        for r in rows:
            qf = r.get("query_facets") or {}
            pf = r.get("page_facets") or {}
            note = r.get("correction_note") or ""
            examples.append(
                f"- Query: {qf.get('brand') or ''} {qf.get('model') or ''} "
                f"(SKUs: {qf.get('sku_tokens') or []}); "
                f"Page: {pf.get('product_name') or ''} ({pf.get('breadcrumb') or ''}); "
                f"Wrong verdict: {r.get('original_match_kind') or '?'}; "
                f"Correct verdict: {r.get('corrected_match_kind')}"
                + (f"; Reason: {note}" if note else "")
            )
        block = (
            "\n\nLearning from prior admin corrections (apply the same reasoning):\n"
            + "\n".join(examples)
        )
        self._few_shot_cache = (datetime.now(timezone.utc), block)
        return block

    def _rule_based_verdict(
        self, facets: QueryFacets, candidate: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Minimal fallback when Haiku is unavailable. Uses required_tokens ∩
        (product_name ∪ url_slug_tokens). Conservative: biases toward
        'unverifiable' rather than false-exact. When facets.sku_tokens is
        non-empty, SKU equality is the deciding signal.
        """
        required = {normalize_model_token(t) for t in (facets.required_tokens or []) if t}
        sku_anchors = {normalize_model_token(t) for t in (facets.sku_tokens or []) if t}

        if not required and not sku_anchors:
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

        if not name and not candidate.get("url_slug_tokens"):
            return {
                "match_kind": "unverifiable",
                "match_score": 40,
                "variant_diffs": [],
                "match_note": "Could not extract product identity from page",
            }

        # SKU-anchor path: when the query carries a SKU, it dominates.
        if sku_anchors:
            sku_hits = [t for t in sku_anchors if t in haystack]
            required_hits = [t for t in required if t in haystack]
            if sku_hits:
                # Page carries a known SKU from the catalog → exact.
                return {
                    "match_kind": "exact",
                    "match_score": 95,
                    "variant_diffs": [],
                    "match_note": None,
                }
            # No SKU match. If brand+series still match, it's a sibling SKU
            # in the same series → family. Otherwise mismatch.
            if required and len(required_hits) == len(required):
                return {
                    "match_kind": "family",
                    "match_score": 55,
                    "variant_diffs": [],
                    "match_note": (
                        f"Same series, different SKU — query SKUs {sorted(sku_anchors)} not found on page"
                    ),
                }
            return {
                "match_kind": "mismatch",
                "match_score": 20,
                "variant_diffs": [],
                "match_note": f"Query SKUs {sorted(sku_anchors)} not found on page",
            }

        # Brand/series-only path (no SKU anchor).
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
    Haiku usually returns clean JSON; tolerate the occasional code fence
    AND tolerate JSON that the model truncated mid-array by walking the
    string left-to-right and parsing the largest valid prefix.
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
        fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        else:
            # Grab the first balanced JSON object
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]

        # Fast path: clean JSON parses on the first try.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Salvage path: model truncated or trailed garbage. Walk back from
        # the end character-by-character looking for a position where the
        # accumulated brace/bracket depth drops to zero with a valid parse.
        depth = 0
        in_string = False
        escape = False
        last_balanced_idx = -1
        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    last_balanced_idx = i
        if last_balanced_idx > 0:
            try:
                return json.loads(text[: last_balanced_idx + 1])
            except json.JSONDecodeError:
                pass
        logger.warning("Failed to parse Haiku JSON response: malformed even after salvage")
        return None
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
