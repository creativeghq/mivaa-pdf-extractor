"""
Mention Search Service — multi-source discovery + verification.

Cost-optimized parallel discovery over enabled sources (DataForSEO News +
Perplexity Sonar + RSS + YouTube), then dedupe + classify + persist.

Reddit was dropped 2026-05-03 — the API onboarding (Responsible Builder
Policy review) wasn't worth the friction for marginal coverage gain. Sonar
+ News pick up Reddit threads that get cited by news/blogs anyway.

Cost discipline (mirrors price v3):
  - Every source is opt-in per-subject via tracked_mentions.sources_enabled
  - Sonar (cheap) by default; Sonar-pro only on first refresh / forced refresh
  - Verdict cache hits cost zero credits
  - Rule pre-filter drops obvious mismatches before Haiku
  - Body fetch via Firecrawl is conditional — only when title/excerpt aren't
    enough for the classifier (out of scope for v1; we send what we have)

Returns one parallel list of MentionHit, sorted by published_at DESC.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.mention_identity_service import (
    SubjectFacets, alias_present, content_hash, normalize_text,
)

logger = logging.getLogger(__name__)


PERPLEXITY_API = "https://api.perplexity.ai/chat/completions"
DATAFORSEO_NEWS_API = "https://api.dataforseo.com/v3/serp/google/news/live/advanced"
YOUTUBE_API = "https://www.googleapis.com/youtube/v3"

# Cost caps per refresh (avoid runaway spend on a single subject)
MAX_RESULTS_PER_SOURCE = 30
MAX_TOTAL_RESULTS = 80
PERPLEXITY_RECENCY_DAYS = 14
RSS_RECENCY_DAYS = 14
NEWS_RECENCY_DAYS = 14


# ────────────────────────────────────────────────────────────────────────────
# DTOs
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class MentionHit:
    url: str
    title: Optional[str] = None
    excerpt: Optional[str] = None
    body_md: Optional[str] = None
    outlet_domain: Optional[str] = None
    outlet_name: Optional[str] = None
    outlet_type: str = "other"
    language_code: Optional[str] = None
    country_code: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[str] = None  # ISO
    source: str = ""  # 'dataforseo_news' | 'perplexity_sonar' | 'rss' | 'youtube'
    engagement: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None

    def canonical_url(self) -> str:
        return canonicalize_url(self.url)


@dataclass
class MentionSearchResult:
    hits: List[MentionHit]
    credits_used: int
    latency_ms: int
    by_source: Dict[str, int] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────────────
# URL canonicalization + outlet classification
# ────────────────────────────────────────────────────────────────────────────

_TRACKING_PARAMS = re.compile(r"^(utm_|fbclid|gclid|igshid|mc_cid|mc_eid|ref|share)", re.IGNORECASE)


def canonicalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        p = urlparse(url.strip())
        host = (p.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        path = p.path.rstrip("/") or "/"
        query_pairs = []
        if p.query:
            for kv in p.query.split("&"):
                if not kv or "=" not in kv:
                    continue
                k, _ = kv.split("=", 1)
                if not _TRACKING_PARAMS.match(k):
                    query_pairs.append(kv)
        rebuilt = f"{p.scheme or 'https'}://{host}{path}"
        if query_pairs:
            rebuilt += "?" + "&".join(query_pairs)
        return rebuilt
    except Exception:
        return url


def domain_of(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        host = (urlparse(url).hostname or "").lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return None


_YOUTUBE_HOSTS = {"youtube.com", "youtu.be", "m.youtube.com"}
_AGGREGATOR_HOSTS = {"news.google.com", "flipboard.com"}


def classify_outlet_type(host: Optional[str]) -> str:
    if not host:
        return "other"
    if host in _YOUTUBE_HOSTS:
        return "youtube"
    if host in _AGGREGATOR_HOSTS:
        return "aggregator"
    return "news"  # default for now; refined by domain reputation later


# ────────────────────────────────────────────────────────────────────────────
# Service
# ────────────────────────────────────────────────────────────────────────────

class MentionSearchService:
    def __init__(self) -> None:
        self.supabase = get_supabase_client()
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY") or ""
        self.dataforseo_b64 = os.getenv("DATAFORSEO_BASE64") or ""
        if not self.dataforseo_b64:
            login = os.getenv("DATAFORSEO_LOGIN") or ""
            pwd = os.getenv("DATAFORSEO_PASSWORD") or ""
            if login and pwd:
                self.dataforseo_b64 = base64.b64encode(f"{login}:{pwd}".encode()).decode()
        self.youtube_key = os.getenv("YOUTUBE_DATA_API_KEY") or ""

    # ───── Top-level ─────

    async def search(
        self,
        *,
        facets: SubjectFacets,
        sources_enabled: Dict[str, bool],
        source_config: Optional[Dict[str, Any]] = None,
        country_codes: Optional[List[str]] = None,
        recency_days: int = 14,
        force_full_discovery: bool = False,
    ) -> MentionSearchResult:
        """Run all enabled sources in parallel. Returns deduped hits."""
        start = time.time()
        cfg = source_config or {}
        country = (country_codes or [""])[0].upper() or None

        tasks: List[Tuple[str, asyncio.Task]] = []
        if sources_enabled.get("news", True):
            tasks.append(("dataforseo_news", asyncio.create_task(
                self._search_dataforseo_news(facets, country=country, recency_days=recency_days)
            )))
        if sources_enabled.get("blogs", True):
            # Blog discovery via Perplexity Sonar (web mode)
            tasks.append(("perplexity_sonar", asyncio.create_task(
                self._search_perplexity(facets, country=country, recency_days=recency_days,
                                       force_full_discovery=force_full_discovery)
            )))
        if sources_enabled.get("rss", True):
            tasks.append(("rss", asyncio.create_task(
                self._search_rss(facets, feeds=cfg.get("rss_feeds") or [])
            )))
        if sources_enabled.get("youtube", False):
            tasks.append(("youtube", asyncio.create_task(
                self._search_youtube(facets)
            )))

        all_hits: List[MentionHit] = []
        by_source: Dict[str, int] = {}
        errors: Dict[str, str] = {}
        credits_used = 0

        results = await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)
        for (name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                errors[name] = str(result)[:200]
                logger.warning(f"mention-search: source {name} failed: {result}")
                continue
            if not isinstance(result, dict):
                continue
            hits: List[MentionHit] = result.get("hits") or []
            credits_used += int(result.get("credits") or 0)
            by_source[name] = len(hits)
            all_hits.extend(hits)

        # Cross-source dedupe — canonical URL or content hash
        deduped = self._dedupe(all_hits)

        # Apply alias presence filter (cheap rule check; full classifier runs upstream)
        kept: List[MentionHit] = []
        for h in deduped:
            text = " ".join(filter(None, [h.title, h.excerpt, (h.body_md or "")[:600]]))
            if alias_present(text, facets):
                kept.append(h)

        # Cap total
        kept = kept[:MAX_TOTAL_RESULTS]

        latency_ms = int((time.time() - start) * 1000)
        return MentionSearchResult(
            hits=kept,
            credits_used=credits_used,
            latency_ms=latency_ms,
            by_source=by_source,
            errors=errors,
        )

    # ───── Helpers ─────

    def _fanout_queries(self, facets: SubjectFacets, *, max_queries: int = 3) -> List[str]:
        """Pick distinctive aliases to fan discovery queries across.

        Strategy: full label first, then single-word aliases ranked by
        distinctiveness (length + uppercase letters as a cheap signal).
        Skip aliases that are very short (<3 chars) or pure numeric, since
        those return mostly noise.

        The first query is always the most specific (full label). Subsequent
        queries broaden coverage when the niche subject has zero exact-phrase
        mentions globally.
        """
        candidates: List[str] = []
        seen: set = set()
        for a in facets.all_aliases():
            if not a:
                continue
            stripped = a.strip()
            if not stripped:
                continue
            # Skip ultra-short or pure-numeric aliases (poor signal)
            if len(stripped) < 3:
                continue
            if all(ch.isdigit() or ch in "._" for ch in stripped):
                continue
            key = normalize_text(stripped)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(stripped)
        if not candidates:
            return [facets.label] if facets.label else []
        # Order: full label first, then by distinctiveness (length desc)
        primary = candidates[0]
        rest = sorted(candidates[1:], key=lambda s: -len(s))
        return [primary, *rest][:max_queries]

    # ───── DataForSEO News ─────

    async def _search_dataforseo_news(
        self, facets: SubjectFacets, *, country: Optional[str], recency_days: int,
    ) -> Dict[str, Any]:
        if not self.dataforseo_b64:
            return {"hits": [], "credits": 0}

        # Multi-query fan-out: search the primary alias first, and if it
        # returns 0 results, try the next 1-2 distinctive aliases. Each
        # DataForSEO call is ~$0.0006 — fanning to 2-3 queries is still
        # cheap and dramatically improves coverage when the full label is
        # too niche (e.g. "ORABELLA PRECIOSA" alone returns 0 globally,
        # but "ORABELLA" or "PRECIOSA" find plenty).
        queries = self._fanout_queries(facets, max_queries=3)
        if not queries:
            return {"hits": [], "credits": 0}

        cutoff = datetime.now(timezone.utc) - timedelta(days=recency_days)
        all_hits: List[MentionHit] = []
        seen_urls: set = set()
        credits = 0

        for query in queries:
            body = [{
                "keyword": query,
                "depth": MAX_RESULTS_PER_SOURCE,
                "location_code": _country_to_dfs_location(country) if country else 2840,
                "language_code": (facets.language_codes or ["en"])[0].lower(),
            }]
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.post(
                        DATAFORSEO_NEWS_API,
                        headers={"Authorization": f"Basic {self.dataforseo_b64}",
                                 "Content-Type": "application/json"},
                        json=body,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                credits += 1
            except Exception as e:
                logger.warning(f"dataforseo_news: request '{query}' failed: {e}")
                continue

            query_hits = 0
            for task in (data.get("tasks") or []):
                for r in (task.get("result") or []):
                    for item in (r.get("items") or []):
                        if (item.get("type") or "").lower() not in ("news_search", "news"):
                            continue
                        url = item.get("url") or ""
                        if not url or url in seen_urls:
                            continue
                        pub_at = item.get("timestamp") or item.get("date_posted")
                        pub_dt = _parse_iso(pub_at)
                        if pub_dt and pub_dt < cutoff:
                            continue
                        host = domain_of(url)
                        seen_urls.add(url)
                        query_hits += 1
                        all_hits.append(MentionHit(
                            url=url,
                            title=item.get("title"),
                            excerpt=item.get("snippet") or item.get("description"),
                            outlet_domain=host,
                            outlet_name=item.get("source") or item.get("source_url"),
                            outlet_type=classify_outlet_type(host),
                            language_code=(facets.language_codes or ["en"])[0],
                            country_code=country,
                            published_at=pub_dt.isoformat() if pub_dt else None,
                            source="dataforseo_news",
                            raw={"rank": item.get("rank_absolute"), "query": query},
                        ))

            # Stop early if the primary alias already returned plenty
            if query_hits >= MAX_RESULTS_PER_SOURCE // 2:
                break

        return {"hits": all_hits[:MAX_RESULTS_PER_SOURCE], "credits": credits}

    # ───── Perplexity Sonar (blogs + general web) ─────

    async def _search_perplexity(
        self, facets: SubjectFacets, *, country: Optional[str], recency_days: int,
        force_full_discovery: bool,
    ) -> Dict[str, Any]:
        if not self.perplexity_key:
            return {"hits": [], "credits": 0}

        # Cost: sonar (cheap) when stable; sonar-pro on first/forced
        model = "sonar-pro" if force_full_discovery else "sonar"
        recency = "week" if recency_days <= 7 else "month"
        # Build a disjunctive query — Sonar handles "X OR Y OR Z" naturally.
        # Use up to 3 distinctive aliases for coverage when the full phrase
        # is too niche.
        queries = self._fanout_queries(facets, max_queries=3)
        if not queries:
            return {"hits": [], "credits": 0}
        primary = queries[0]
        alts = queries[1:]
        alt_clause = ""
        if alts:
            alt_str = " or ".join(f'"{a}"' for a in alts)
            alt_clause = f" (also accept mentions of {alt_str} when those refer to the same subject)"
        # Phrase the prompt as a recent-mentions sweep
        question = (
            f"Find blog posts and articles from the last {recency_days} days that mention "
            f"\"{primary}\"{alt_clause}"
        )
        if facets.brand and facets.brand != primary:
            question += f" (brand: {facets.brand})"
        question += (
            ". For each mention return: title, url, outlet name, publication date, "
            "and a 2-sentence summary. Skip retailer product pages and pure listings."
        )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    PERPLEXITY_API,
                    headers={"Authorization": f"Bearer {self.perplexity_key}",
                             "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "Return JSON only, no prose."},
                            {"role": "user", "content": question},
                        ],
                        "search_recency_filter": recency,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "mentions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "title": {"type": "string"},
                                                    "url": {"type": "string"},
                                                    "outlet": {"type": "string"},
                                                    "published": {"type": "string"},
                                                    "summary": {"type": "string"},
                                                },
                                                "required": ["title", "url"],
                                            },
                                        },
                                    },
                                    "required": ["mentions"],
                                },
                            },
                        },
                    },
                )
                resp.raise_for_status()
                payload = resp.json()
        except Exception as e:
            logger.warning(f"perplexity_sonar: request failed: {e}")
            return {"hits": [], "credits": 0}

        text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        try:
            import json as _json
            data = _json.loads(text)
        except Exception:
            return {"hits": [], "credits": 1}

        hits: List[MentionHit] = []
        for m in data.get("mentions") or []:
            url = m.get("url") or ""
            if not url:
                continue
            host = domain_of(url)
            hits.append(MentionHit(
                url=url,
                title=m.get("title"),
                excerpt=m.get("summary"),
                outlet_domain=host,
                outlet_name=m.get("outlet") or host,
                outlet_type=classify_outlet_type(host),
                language_code=(facets.language_codes or ["en"])[0],
                country_code=country,
                published_at=_parse_iso_str(m.get("published")),
                source="perplexity_sonar",
            ))
        # ~1 sonar credit; sonar-pro is more — bill 2 vs 1 as a cost proxy
        return {"hits": hits[:MAX_RESULTS_PER_SOURCE], "credits": 2 if force_full_discovery else 1}

    # ───── RSS ─────

    async def _search_rss(
        self, facets: SubjectFacets, *, feeds: List[str],
    ) -> Dict[str, Any]:
        if not feeds:
            return {"hits": [], "credits": 0}
        aliases = [normalize_text(a) for a in facets.all_aliases()]
        cutoff = datetime.now(timezone.utc) - timedelta(days=RSS_RECENCY_DAYS)

        async def _fetch(feed_url: str) -> List[MentionHit]:
            try:
                async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                    resp = await client.get(feed_url)
                    resp.raise_for_status()
                    body = resp.text
            except Exception as e:
                logger.debug(f"rss: fetch {feed_url} failed: {e}")
                return []
            try:
                root = ET.fromstring(body)
            except Exception:
                return []
            out: List[MentionHit] = []
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            host = domain_of(feed_url)
            # RSS 2.0
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                desc = (item.findtext("description") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                pub_dt = _parse_iso(pub)
                if pub_dt and pub_dt < cutoff:
                    continue
                text = normalize_text(f"{title} {desc}")
                if not any(a in text for a in aliases if a):
                    continue
                out.append(MentionHit(
                    url=link,
                    title=title,
                    excerpt=desc[:400],
                    outlet_domain=host,
                    outlet_name=host,
                    outlet_type="rss",
                    published_at=pub_dt.isoformat() if pub_dt else None,
                    source="rss",
                ))
            # Atom
            for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
                title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
                link_el = entry.find("atom:link", namespaces=ns)
                link = (link_el.attrib.get("href") if link_el is not None else "") or ""
                summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
                pub = (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip()
                pub_dt = _parse_iso(pub)
                if pub_dt and pub_dt < cutoff:
                    continue
                text = normalize_text(f"{title} {summary}")
                if not any(a in text for a in aliases if a):
                    continue
                out.append(MentionHit(
                    url=link,
                    title=title,
                    excerpt=summary[:400],
                    outlet_domain=host,
                    outlet_name=host,
                    outlet_type="rss",
                    published_at=pub_dt.isoformat() if pub_dt else None,
                    source="rss",
                ))
            return out

        results = await asyncio.gather(*[_fetch(f) for f in feeds[:20]], return_exceptions=True)
        hits: List[MentionHit] = []
        for r in results:
            if isinstance(r, list):
                hits.extend(r)
        return {"hits": hits[:MAX_RESULTS_PER_SOURCE], "credits": 0}  # RSS free

    # ───── YouTube ─────

    async def _search_youtube(self, facets: SubjectFacets) -> Dict[str, Any]:
        if not self.youtube_key:
            return {"hits": [], "credits": 0}
        aliases = facets.all_aliases()
        query = aliases[0] if aliases else facets.label
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{YOUTUBE_API}/search",
                    params={
                        "part": "snippet",
                        "q": query,
                        "type": "video",
                        "order": "date",
                        "maxResults": 25,
                        "key": self.youtube_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning(f"youtube: search failed: {e}")
            return {"hits": [], "credits": 0}

        hits: List[MentionHit] = []
        for item in data.get("items") or []:
            sn = item.get("snippet") or {}
            vid = (item.get("id") or {}).get("videoId")
            if not vid:
                continue
            url = f"https://www.youtube.com/watch?v={vid}"
            hits.append(MentionHit(
                url=url,
                title=sn.get("title"),
                excerpt=sn.get("description"),
                outlet_domain="youtube.com",
                outlet_name=sn.get("channelTitle"),
                outlet_type="youtube",
                author=sn.get("channelTitle"),
                published_at=sn.get("publishedAt"),
                source="youtube",
                raw={"video_id": vid},
            ))
        return {"hits": hits[:MAX_RESULTS_PER_SOURCE], "credits": 0}

    # ───── Dedupe ─────

    def _dedupe(self, hits: List[MentionHit]) -> List[MentionHit]:
        seen_urls: Dict[str, int] = {}
        seen_hashes: Dict[str, int] = {}
        out: List[MentionHit] = []
        for h in hits:
            ch = content_hash(url=h.url, title=h.title, body=h.body_md or h.excerpt)
            curl = h.canonical_url()
            if curl in seen_urls:
                continue
            if ch and ch in seen_hashes:
                continue
            seen_urls[curl] = 1
            if ch:
                seen_hashes[ch] = 1
            out.append(h)
        return out


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).strip()
    # Try a small set of common formats
    fmts = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%d",
    ]
    for f in fmts:
        try:
            d = datetime.strptime(s, f)
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _parse_iso_str(value: Any) -> Optional[str]:
    d = _parse_iso(value)
    return d.isoformat() if d else None


# Minimal country code → DataForSEO location_code mapping
def _country_to_dfs_location(code: Optional[str]) -> int:
    if not code:
        return 2840
    return {
        "US": 2840, "GB": 2826, "DE": 2276, "FR": 2250, "IT": 2380, "ES": 2724,
        "GR": 2300, "NL": 2528, "BE": 2056, "AT": 2040, "CH": 2756, "PT": 2620,
        "IE": 2372, "CA": 2124, "AU": 2036, "PL": 2616, "SE": 2752, "DK": 2208,
        "NO": 2578, "FI": 2246, "TR": 2792, "BG": 2100, "RO": 2642, "CY": 2196,
    }.get(code.upper(), 2840)


_search_service: Optional[MentionSearchService] = None


def get_mention_search_service() -> MentionSearchService:
    global _search_service
    if _search_service is None:
        _search_service = MentionSearchService()
    return _search_service
