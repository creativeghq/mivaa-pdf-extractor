"""
Self-healing recipe discovery for retailer price scrapes.

Goal: every Firecrawl scrape against a retailer page is treated as
training data. Once we've seen a domain enough times, future scrapes
on URLs matching the same pattern try a cheap httpx + selectolax fetch
first, falling back to Firecrawl on selector miss.

The orchestration:
  1. First time we ever scrape a URL on `flobali.gr`: record the URL pattern
     + the price text Firecrawl returned. After ~3 successful scrapes on the
     same pattern, derive CSS selectors heuristically (find the unique CSS
     path to the text containing the extracted price).
  2. On future scrapes: if the recipe has confidence >= 0.8 and selectors
     are non-empty, run httpx + selectolax. Cross-check with Firecrawl in
     "shadow mode" until we've seen 5+ matches in a row, then httpx-only.
  3. Selector drift detection: any 3-mismatch streak disables the recipe
     and flips back to Firecrawl. Manual re-validation re-enables.

This file ships the data model + lookup/upsert helpers. The heuristic
selector-derivation step lives in a separate worker (TODO) — for now the
recipe table is populated only when an admin (or a future migration)
hand-seeds it. The infra is wired so seeding immediately reduces cost.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# URL → pattern derivation
# ────────────────────────────────────────────────────────────────────

_NUMERIC_PATH_SEG = re.compile(r"^\d+$")
_LONG_HASH = re.compile(r"^[a-z0-9]{12,}$", re.IGNORECASE)


def url_pattern(url: str) -> str:
    """
    Reduce a URL to a stable pattern for recipe lookup.
      https://www.flobali.gr/en/product/some-name → flobali.gr|/en/product/*
      https://bestprice.gr/to/151938286/orabella-...html → bestprice.gr|/to/*/*
    Path segments that look like opaque IDs (long digits, long hashes) become *.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return ""
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if not host:
        return ""
    segments = []
    for part in (parsed.path or "").split("/"):
        if not part:
            continue
        if _NUMERIC_PATH_SEG.match(part) or _LONG_HASH.match(part) or len(part) > 30:
            segments.append("*")
        else:
            segments.append(part)
    return f"{host}|/{'/'.join(segments)}" if segments else f"{host}|/"


def host_of(url: str) -> str:
    try:
        h = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return h.removeprefix("www.")


# ────────────────────────────────────────────────────────────────────
# Recipe DB layer
# ────────────────────────────────────────────────────────────────────


def find_recipe(url: str) -> Optional[Dict[str, Any]]:
    """Return the active recipe for this URL's pattern (highest confidence)."""
    sb = get_supabase_client().client
    domain = host_of(url)
    if not domain:
        return None
    try:
        resp = (
            sb.table("retailer_extraction_recipes")
            .select("*")
            .eq("domain", domain)
            .eq("disabled", False)
            .order("confidence", desc=True)
            .limit(5)
            .execute()
        )
    except Exception as e:
        logger.debug(f"recipe lookup failed (non-fatal): {e}")
        return None
    rows = resp.data or []
    if not rows:
        return None
    # Pick the row whose url_pattern matches this URL's pattern most closely.
    candidate_pattern = url_pattern(url).split("|", 1)[-1]
    best = None
    for row in rows:
        if row.get("url_pattern") == candidate_pattern:
            return row
        if best is None or (row.get("confidence") or 0) > (best.get("confidence") or 0):
            best = row
    return best


def record_success(recipe_id: str) -> None:
    sb = get_supabase_client().client
    try:
        sb.rpc(
            "exec_sql",
            {"sql": (
                "UPDATE retailer_extraction_recipes "
                "SET success_count = success_count + 1, "
                "    confidence = LEAST(1.0, success_count::float / GREATEST(success_count + failure_count, 1)), "
                "    last_validated_at = now(), "
                "    updated_at = now() "
                f"WHERE id = '{recipe_id}'"
            )},
        ).execute()
    except Exception:
        # exec_sql RPC may not exist; fall back to direct update
        try:
            row = (
                sb.table("retailer_extraction_recipes").select("success_count, failure_count")
                .eq("id", recipe_id).maybe_single().execute()
            )
            r = (row.data if row else None) or {}
            sc = int(r.get("success_count") or 0) + 1
            fc = int(r.get("failure_count") or 0)
            conf = sc / max(sc + fc, 1)
            sb.table("retailer_extraction_recipes").update({
                "success_count": sc,
                "confidence": conf,
                "last_validated_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", recipe_id).execute()
        except Exception as e:
            logger.debug(f"recipe success record failed: {e}")


def record_failure(recipe_id: str, reason: str) -> None:
    sb = get_supabase_client().client
    try:
        row = (
            sb.table("retailer_extraction_recipes").select("success_count, failure_count")
            .eq("id", recipe_id).maybe_single().execute()
        )
        r = (row.data if row else None) or {}
        sc = int(r.get("success_count") or 0)
        fc = int(r.get("failure_count") or 0) + 1
        conf = sc / max(sc + fc, 1)
        # 3 consecutive failures + low confidence → auto-disable.
        update = {
            "failure_count": fc,
            "confidence": conf,
            "last_failure_at": datetime.now(timezone.utc).isoformat(),
            "last_failure_reason": reason[:500],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if conf < 0.5 and fc >= 3:
            update["disabled"] = True
        sb.table("retailer_extraction_recipes").update(update).eq("id", recipe_id).execute()
    except Exception as e:
        logger.debug(f"recipe failure record failed: {e}")


# ────────────────────────────────────────────────────────────────────
# httpx + selectolax cheap path
# ────────────────────────────────────────────────────────────────────


_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


async def fetch_via_recipe(url: str, recipe: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Try to extract price + product_name via httpx + selectolax. Returns None
    on any failure (which the caller treats as a signal to fall back to
    Firecrawl). Best-effort — never raises.
    """
    try:
        import selectolax.parser  # type: ignore
    except Exception:
        return None  # selectolax not installed yet — caller falls back

    if recipe.get("requires_js"):
        return None

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
        if resp.status_code != 200:
            return None
        if "captcha" in resp.text.lower() or "cloudflare" in resp.text.lower()[:2000]:
            return None
    except Exception:
        return None

    try:
        tree = selectolax.parser.HTMLParser(resp.text)
    except Exception:
        return None

    def pick(selector: Optional[str]) -> Optional[str]:
        if not selector:
            return None
        node = tree.css_first(selector)
        if not node:
            return None
        text = (node.text() or "").strip()
        return text or None

    price_text = pick(recipe.get("price_selector"))
    if not price_text:
        return None

    return {
        "price_text": price_text,
        "original_price_text": pick(recipe.get("original_price_selector")),
        "product_name": pick(recipe.get("product_name_selector")),
        "availability_text": pick(recipe.get("availability_selector")),
        "currency_default": recipe.get("currency_default") or "EUR",
        "via": "httpx_recipe",
    }
