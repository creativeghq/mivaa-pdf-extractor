"""
DataForSEO Unified Client — single async wrapper covering EVERY major endpoint
across all 13 product families:

  1. SERP API           — Google + Bing + YouTube + Baidu + Yahoo + Naver + Seznam
  2. AI Optimization    — LLM Mentions + LLM Responses (ChatGPT/Claude/Gemini/Perplexity)
  3. Keywords Data      — Google Ads + Bing Ads + Trends + DataForSEO Trends + Clickstream
  4. DataForSEO Labs    — Google + Amazon + App Store + Google Play
  5. Backlinks          — summary, anchors, referring domains, competitors, intersection
  6. OnPage             — full site crawler + lighthouse + duplicates + redirects
  7. Content Analysis   — sentiment, phrase trends, citation mining
  8. Domain Analytics   — technologies, whois
  9. Merchant           — Google Shopping + Amazon Products
 10. App Data           — Google Play + Apple App Store listings + reviews
 11. Business Data      — GMB, Trustpilot, Tripadvisor, Pinterest, Reddit
 12. Databases          — bulk dataset downloads (not exposed here)
 13. Appendix           — utilities (status, locations)

Cost discipline:
  - All callers go through `_call()` which logs to ai_usage_logs via
    `mention_cost_logger.log_dataforseo_*` with a per-call attribution.
  - Live endpoints used wherever available (Labs is Live-only; SERP and
    Merchant fall back to Task GET polling with a budget).
  - Sandbox mode supported via DATAFORSEO_SANDBOX=1 env var (returns
    dummy data, zero charge — useful in dev).

Auth: HTTP Basic via DATAFORSEO_BASE64 (preferred) or
DATAFORSEO_LOGIN:DATAFORSEO_PASSWORD. Same credential set used by
the TypeScript dataforseo-client and the existing merchant service.

This client is the foundation for the SEO agent toolkit. Every agent tool
should go through it — no direct httpx calls to DataForSEO from anywhere
else, so cost logging stays consistent.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.services.integrations.mention_cost_logger import (
    CostAttribution, log_dataforseo_labs_call, log_dataforseo_serp_call,
)

logger = logging.getLogger(__name__)


_BASE = "https://api.dataforseo.com/v3"
_SANDBOX_BASE = "https://sandbox.dataforseo.com/v3"
_HTTP_TIMEOUT = 30.0


# ─────────────────────────────────────────────────────────────────────────────
# Country / language code maps (full coverage from DataForSEO Appendix)
# ─────────────────────────────────────────────────────────────────────────────

# ISO-3166 alpha-2 → DataForSEO numeric location_code. Covers ~70 markets.
COUNTRY_TO_LOCATION: Dict[str, int] = {
    "US": 2840, "GB": 2826, "UK": 2826, "CA": 2124, "AU": 2036, "NZ": 2554,
    "IE": 2372, "ZA": 2710, "IN": 2356, "SG": 2702, "HK": 2344, "JP": 2392,
    "DE": 2276, "FR": 2250, "IT": 2380, "ES": 2724, "PT": 2620, "NL": 2528,
    "BE": 2056, "AT": 2040, "CH": 2756, "PL": 2616, "CZ": 2203, "SK": 2703,
    "HU": 2348, "GR": 2300, "BG": 2100, "RO": 2642, "CY": 2196, "MT": 2470,
    "DK": 2208, "SE": 2752, "NO": 2578, "FI": 2246, "IS": 2352, "EE": 2233,
    "LV": 2428, "LT": 2440, "TR": 2792, "RU": 2643, "UA": 2804, "BY": 2112,
    "BR": 2076, "MX": 2484, "AR": 2032, "CL": 2152, "CO": 2170, "PE": 2604,
    "VE": 2862, "EC": 2218, "UY": 2858, "PY": 2600, "BO": 2068, "DO": 2214,
    "GT": 2320, "CR": 2188, "PA": 2591, "PR": 2630, "CU": 2192,
    "AE": 2784, "SA": 2682, "EG": 2818, "MA": 2504, "DZ": 2012, "TN": 2788,
    "LB": 2422, "JO": 2400, "IL": 2376, "QA": 2634, "KW": 2414, "BH": 2048,
    "OM": 2512, "IQ": 2368, "KE": 2404, "NG": 2566, "GH": 2288, "TZ": 2834,
    "UG": 2800, "ET": 2231, "RW": 2646, "SN": 2686,
    "MY": 2458, "TH": 2764, "VN": 2704, "ID": 2360, "PH": 2608, "KR": 2410,
    "TW": 2158, "PK": 2586, "BD": 2050, "LK": 2144, "NP": 2524,
}


def country_to_location(code: Optional[str], default: int = 2840) -> int:
    if not code:
        return default
    return COUNTRY_TO_LOCATION.get(code.upper(), default)


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataForSEOResult:
    """Normalised result envelope for every call."""
    ok: bool
    items: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
    status_code: int = 200
    cost_usd: float = 0.0
    error: Optional[str] = None
    latency_ms: int = 0


class DataForSEOUnifiedClient:
    """Thin async wrapper around the full DataForSEO API surface.

    Every method returns `DataForSEOResult`. The client handles auth, retry
    on transient 5xx, sandbox routing, cost logging, and uniform error
    surfacing — callers get a stable shape no matter which endpoint they
    hit.
    """

    def __init__(self, *, sandbox: Optional[bool] = None):
        self.sandbox = bool(int(os.getenv("DATAFORSEO_SANDBOX", "0"))) if sandbox is None else sandbox
        self.base_url = _SANDBOX_BASE if self.sandbox else _BASE
        b64 = os.getenv("DATAFORSEO_BASE64") or ""
        if not b64:
            login = os.getenv("DATAFORSEO_LOGIN") or ""
            password = os.getenv("DATAFORSEO_PASSWORD") or ""
            if login and password:
                b64 = base64.b64encode(f"{login}:{password}".encode()).decode()
        self.b64 = b64
        if not self.b64:
            logger.warning("DataForSEO credentials missing — every call will fail with 401.")

    # ────────── Internal core ──────────

    async def _call(
        self,
        path: str,
        body: Optional[List[Dict[str, Any]]] = None,
        *,
        method: str = "POST",
        attribution: Optional[CostAttribution] = None,
        log_kind: str = "labs",
        operation: str = "",
    ) -> DataForSEOResult:
        """Single HTTP call with cost logging.

        log_kind: 'labs' or 'serp' — picks the right cost-logger fn so every
                   row goes to ai_usage_logs with the proper provider field.
        operation: free-text label appearing in the cost log row (e.g.
                   'serp.organic.live.advanced').
        """
        if not self.b64:
            return DataForSEOResult(ok=False, error="DataForSEO credentials not configured", status_code=401)
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Basic {self.b64}",
            "Content-Type": "application/json",
        }
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                if method == "GET":
                    resp = await client.get(url, headers=headers)
                else:
                    resp = await client.post(url, headers=headers, json=body or [])
        except httpx.RequestError as e:
            elapsed = int((time.time() - start) * 1000)
            self._log_cost(log_kind, attribution, operation, items=0, latency_ms=elapsed, success=False, error=str(e))
            return DataForSEOResult(ok=False, error=f"network: {e}", latency_ms=elapsed)

        elapsed = int((time.time() - start) * 1000)
        if resp.status_code >= 500:
            # 1 retry for 5xx
            await asyncio.sleep(0.5)
            try:
                async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                    resp = await client.post(url, headers=headers, json=body or [])
            except httpx.RequestError as e:
                self._log_cost(log_kind, attribution, operation, items=0, latency_ms=elapsed, success=False, error=str(e))
                return DataForSEOResult(ok=False, error=f"retry network: {e}", latency_ms=elapsed)

        if resp.status_code >= 400:
            err_text = resp.text[:200]
            self._log_cost(log_kind, attribution, operation, items=0, latency_ms=elapsed, success=False, error=err_text)
            return DataForSEOResult(ok=False, status_code=resp.status_code, error=f"{resp.status_code}: {err_text}", latency_ms=elapsed)

        try:
            data = resp.json()
        except Exception as e:
            return DataForSEOResult(ok=False, error=f"json parse: {e}", latency_ms=elapsed)

        items: List[Dict[str, Any]] = []
        cost = 0.0
        for task in (data.get("tasks") or []):
            cost += float(task.get("cost") or 0.0)
            for r in (task.get("result") or []):
                # Some endpoints return items[] inside result; others put data inline.
                inner = r.get("items")
                if inner:
                    items.extend(inner)
                else:
                    items.append(r)

        self._log_cost(log_kind, attribution, operation, items=len(items), latency_ms=elapsed, success=True)
        return DataForSEOResult(ok=True, items=items, raw=data, status_code=resp.status_code,
                                cost_usd=cost, latency_ms=elapsed)

    def _log_cost(
        self, kind: str, attribution: Optional[CostAttribution], operation: str,
        *, items: int, latency_ms: int, success: bool, error: Optional[str] = None,
    ) -> None:
        try:
            if kind == "serp":
                log_dataforseo_serp_call(
                    attribution=attribution, operation=operation or "serp", query=operation,
                    items_returned=items, latency_ms=latency_ms,
                    success=success, error_message=error,
                )
            else:
                log_dataforseo_labs_call(
                    attribution=attribution, seed_keyword=operation or "labs",
                    items_returned=items, latency_ms=latency_ms,
                    success=success, error_message=error,
                )
        except Exception as e:
            logger.debug(f"DataForSEO cost log failed: {e}")

    # ────────── 1. SERP API ──────────

    async def serp_google_organic(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        depth: int = 30, paa_depth: int = 1, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        """Google organic SERP (live/advanced) — full block surface (PAA + AI Overview
        + featured snippet + related searches + organic + videos + news + KG +
        paid + shopping)."""
        body = [{
            "keyword": keyword,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "depth": depth,
            "people_also_ask_click_depth": paa_depth,
        }]
        return await self._call("/serp/google/organic/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.google.organic:{keyword}")

    async def serp_google_maps(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "location_code": country_to_location(country_code), "language_code": language_code}]
        return await self._call("/serp/google/maps/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.google.maps:{keyword}")

    async def serp_google_local_finder(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "location_code": country_to_location(country_code), "language_code": language_code}]
        return await self._call("/serp/google/local_finder/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.google.local_finder:{keyword}")

    async def serp_google_news(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "location_code": country_to_location(country_code), "language_code": language_code}]
        return await self._call("/serp/google/news/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.google.news:{keyword}")

    async def serp_google_images(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "location_code": country_to_location(country_code), "language_code": language_code}]
        return await self._call("/serp/google/images/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.google.images:{keyword}")

    async def serp_google_jobs(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "location_code": country_to_location(country_code), "language_code": language_code}]
        return await self._call("/serp/google/jobs/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.google.jobs:{keyword}")

    async def serp_google_autocomplete(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "location_code": country_to_location(country_code), "language_code": language_code}]
        return await self._call("/serp/google/autocomplete/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.google.autocomplete:{keyword}")

    async def serp_google_finance(
        self, *, keyword: str, language_code: str = "en", endpoint: str = "explore",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        """endpoint: explore | markets | quote | ticker_search"""
        body = [{"keyword": keyword, "language_code": language_code}]
        return await self._call(f"/serp/google/finance/{endpoint}/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.google.finance.{endpoint}:{keyword}")

    async def serp_google_ai_summary(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "location_code": country_to_location(country_code), "language_code": language_code}]
        return await self._call("/serp/google/ai_summary/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.google.ai_summary:{keyword}")

    async def serp_bing_organic(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "location_code": country_to_location(country_code), "language_code": language_code}]
        return await self._call("/serp/bing/organic/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.bing.organic:{keyword}")

    async def serp_youtube_organic(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "location_code": country_to_location(country_code), "language_code": language_code}]
        return await self._call("/serp/youtube/organic/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.youtube.organic:{keyword}")

    async def serp_youtube_video_info(
        self, *, video_id: str, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"video_id": video_id, "language_code": language_code}]
        return await self._call("/serp/youtube/video_info/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.youtube.video_info:{video_id}")

    async def serp_youtube_video_subtitles(
        self, *, video_id: str, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"video_id": video_id, "language_code": language_code}]
        return await self._call("/serp/youtube/video_subtitles/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.youtube.video_subtitles:{video_id}")

    async def serp_youtube_video_comments(
        self, *, video_id: str, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"video_id": video_id, "language_code": language_code}]
        return await self._call("/serp/youtube/video_comments/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.youtube.video_comments:{video_id}")

    async def serp_baidu_organic(
        self, *, keyword: str, language_code: str = "zh",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "language_code": language_code}]
        return await self._call("/serp/baidu/organic/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.baidu.organic:{keyword}")

    async def serp_yahoo_organic(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "location_code": country_to_location(country_code), "language_code": language_code}]
        return await self._call("/serp/yahoo/organic/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.yahoo.organic:{keyword}")

    async def serp_naver_organic(
        self, *, keyword: str, language_code: str = "ko",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "language_code": language_code}]
        return await self._call("/serp/naver/organic/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.naver.organic:{keyword}")

    async def serp_seznam_organic(
        self, *, keyword: str, language_code: str = "cs",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "language_code": language_code}]
        return await self._call("/serp/seznam/organic/live/advanced", body,
                                attribution=attribution, log_kind="serp",
                                operation=f"serp.seznam.organic:{keyword}")

    # ────────── 2. AI Optimization API ──────────

    async def ai_keyword_search_volume(
        self, *, keywords: List[str], language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        """LLM-search-specific keyword volume (queries actually run against AI engines)."""
        body = [{"keywords": keywords[:1000], "language_code": language_code}]
        return await self._call("/ai_optimization/ai_keyword_data/keywords_search_volume/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="ai.keyword_search_volume")

    async def ai_llm_mentions_search(
        self, *, keyword: str, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "language_code": language_code}]
        return await self._call("/ai_optimization/llm_mentions/search/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"ai.llm_mentions.search:{keyword}")

    async def ai_llm_mentions_top_pages(
        self, *, keyword: str, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "language_code": language_code}]
        return await self._call("/ai_optimization/llm_mentions/top_pages/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"ai.llm_mentions.top_pages:{keyword}")

    async def ai_llm_mentions_top_domains(
        self, *, keyword: str, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "language_code": language_code}]
        return await self._call("/ai_optimization/llm_mentions/top_domains/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"ai.llm_mentions.top_domains:{keyword}")

    async def ai_llm_mentions_aggregated_metrics(
        self, *, keyword: str, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "language_code": language_code}]
        return await self._call("/ai_optimization/llm_mentions/aggregated_metrics/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"ai.llm_mentions.aggregated:{keyword}")

    async def ai_llm_response(
        self, *, model_family: str, prompt: str, model: Optional[str] = None,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        """model_family: chat_gpt | claude | gemini | perplexity"""
        body = [{"prompt": prompt, "model_name": model} if model else {"prompt": prompt}]
        path_map = {
            "chat_gpt": "/ai_optimization/chat_gpt/llm_responses/live",
            "claude": "/ai_optimization/claude/llm_responses/live",
            "gemini": "/ai_optimization/gemini/llm_responses/live",
            "perplexity": "/ai_optimization/perplexity/llm_responses/live",
        }
        path = path_map.get(model_family.lower())
        if not path:
            return DataForSEOResult(ok=False, error=f"unknown model_family: {model_family}")
        return await self._call(path, body, attribution=attribution, log_kind="labs",
                                operation=f"ai.{model_family}.llm_response")

    async def ai_llm_models(
        self, *, model_family: str, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        path_map = {
            "chat_gpt": "/ai_optimization/chat_gpt/llm_responses/models",
            "claude": "/ai_optimization/claude/llm_responses/models",
            "gemini": "/ai_optimization/gemini/llm_responses/models",
            "perplexity": "/ai_optimization/perplexity/llm_responses/models",
        }
        path = path_map.get(model_family.lower())
        if not path:
            return DataForSEOResult(ok=False, error=f"unknown model_family: {model_family}")
        return await self._call(path, method="GET", attribution=attribution, log_kind="labs",
                                operation=f"ai.{model_family}.models")

    # ────────── 3. Keywords Data API ──────────

    async def kw_google_ads_search_volume(
        self, *, keywords: List[str], country_code: Optional[str] = None,
        language_code: str = "en", attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keywords": keywords[:1000],
            "location_code": country_to_location(country_code),
            "language_code": language_code,
        }]
        return await self._call("/keywords_data/google_ads/search_volume/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="kw.google_ads.search_volume")

    async def kw_google_ads_keywords_for_site(
        self, *, target: str, country_code: Optional[str] = None,
        language_code: str = "en", attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"target": target, "location_code": country_to_location(country_code), "language_code": language_code}]
        return await self._call("/keywords_data/google_ads/keywords_for_site/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"kw.google_ads.keywords_for_site:{target}")

    async def kw_google_ads_keywords_for_keywords(
        self, *, keywords: List[str], country_code: Optional[str] = None,
        language_code: str = "en", attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keywords": keywords[:200],
            "location_code": country_to_location(country_code),
            "language_code": language_code,
        }]
        return await self._call("/keywords_data/google_ads/keywords_for_keywords/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="kw.google_ads.keywords_for_keywords")

    async def kw_google_ads_traffic_by_keywords(
        self, *, keywords: List[str], country_code: Optional[str] = None,
        bid: float = 1.0, match: str = "exact", attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keywords": keywords[:1000],
            "location_code": country_to_location(country_code),
            "bid": bid, "match": match,
        }]
        return await self._call("/keywords_data/google_ads/ad_traffic_by_keywords/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="kw.google_ads.ad_traffic")

    async def kw_bing_ads_search_volume(
        self, *, keywords: List[str], country_code: Optional[str] = None,
        language_code: str = "en", attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keywords": keywords[:1000],
            "location_code": country_to_location(country_code),
            "language_code": language_code,
        }]
        return await self._call("/keywords_data/bing/search_volume/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="kw.bing.search_volume")

    async def kw_google_trends_explore(
        self, *, keywords: List[str], country_code: Optional[str] = None,
        time_range: str = "past_12_months", attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keywords": keywords[:5],
            "location_code": country_to_location(country_code),
            "time_range": time_range,
        }]
        return await self._call("/keywords_data/google_trends/explore/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="kw.google_trends.explore")

    async def kw_dataforseo_trends_explore(
        self, *, keywords: List[str], country_code: Optional[str] = None,
        time_range: str = "past_12_months", attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keywords": keywords[:5],
            "location_code": country_to_location(country_code),
            "time_range": time_range,
        }]
        return await self._call("/keywords_data/dataforseo_trends/explore/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="kw.dataforseo_trends.explore")

    async def kw_clickstream_search_volume(
        self, *, keywords: List[str], country_code: Optional[str] = None,
        language_code: str = "en", attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keywords": keywords[:1000],
            "location_code": country_to_location(country_code),
            "language_code": language_code,
        }]
        return await self._call("/keywords_data/clickstream_data/dataforseo_search_volume/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="kw.clickstream.search_volume")

    # ────────── 4. DataForSEO Labs API ──────────

    async def labs_related_keywords(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        limit: int = 100, depth: int = 2, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keyword": keyword,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "limit": min(limit, 1000),
            "depth": depth,
        }]
        return await self._call("/dataforseo_labs/google/related_keywords/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"labs.related_keywords:{keyword}")

    async def labs_keyword_suggestions(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        limit: int = 100, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keyword": keyword,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "limit": min(limit, 1000),
        }]
        return await self._call("/dataforseo_labs/google/keyword_suggestions/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"labs.keyword_suggestions:{keyword}")

    async def labs_keyword_ideas(
        self, *, keywords: List[str], country_code: Optional[str] = None, language_code: str = "en",
        limit: int = 100, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keywords": keywords[:200],
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "limit": min(limit, 1000),
        }]
        return await self._call("/dataforseo_labs/google/keyword_ideas/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="labs.keyword_ideas")

    async def labs_bulk_keyword_difficulty(
        self, *, keywords: List[str], country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keywords": keywords[:1000],
            "location_code": country_to_location(country_code),
            "language_code": language_code,
        }]
        return await self._call("/dataforseo_labs/google/bulk_keyword_difficulty/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="labs.bulk_keyword_difficulty")

    async def labs_search_intent(
        self, *, keywords: List[str], language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keywords": keywords[:1000], "language_code": language_code}]
        return await self._call("/dataforseo_labs/google/search_intent/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="labs.search_intent")

    async def labs_keyword_overview(
        self, *, keywords: List[str], country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keywords": keywords[:700],
            "location_code": country_to_location(country_code),
            "language_code": language_code,
        }]
        return await self._call("/dataforseo_labs/google/keyword_overview/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="labs.keyword_overview")

    async def labs_historical_keyword_data(
        self, *, keywords: List[str], country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keywords": keywords[:700],
            "location_code": country_to_location(country_code),
            "language_code": language_code,
        }]
        return await self._call("/dataforseo_labs/google/historical_keyword_data/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="labs.historical_keyword_data")

    async def labs_serp_competitors(
        self, *, keywords: List[str], country_code: Optional[str] = None, language_code: str = "en",
        limit: int = 30, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keywords": keywords[:200],
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "limit": limit,
        }]
        return await self._call("/dataforseo_labs/google/serp_competitors/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="labs.serp_competitors")

    async def labs_ranked_keywords(
        self, *, target: str, country_code: Optional[str] = None, language_code: str = "en",
        limit: int = 100, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "target": target,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "limit": min(limit, 1000),
        }]
        return await self._call("/dataforseo_labs/google/ranked_keywords/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"labs.ranked_keywords:{target}")

    async def labs_competitors_domain(
        self, *, target: str, country_code: Optional[str] = None, language_code: str = "en",
        limit: int = 30, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "target": target,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "limit": limit,
        }]
        return await self._call("/dataforseo_labs/google/competitors_domain/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"labs.competitors_domain:{target}")

    async def labs_domain_intersection(
        self, *, targets: List[str], country_code: Optional[str] = None, language_code: str = "en",
        limit: int = 100, intersections: bool = True,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        if len(targets) < 2:
            return DataForSEOResult(ok=False, error="domain_intersection requires at least 2 targets")
        body = [{
            "target1": targets[0], "target2": targets[1],
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "limit": min(limit, 1000),
            "intersections": intersections,
        }]
        return await self._call("/dataforseo_labs/google/domain_intersection/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"labs.domain_intersection:{targets[0]}/{targets[1]}")

    async def labs_subdomains(
        self, *, target: str, country_code: Optional[str] = None, language_code: str = "en",
        limit: int = 100, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "target": target,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "limit": min(limit, 1000),
        }]
        return await self._call("/dataforseo_labs/google/subdomains/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"labs.subdomains:{target}")

    async def labs_relevant_pages(
        self, *, target: str, country_code: Optional[str] = None, language_code: str = "en",
        limit: int = 100, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "target": target,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "limit": min(limit, 1000),
        }]
        return await self._call("/dataforseo_labs/google/relevant_pages/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"labs.relevant_pages:{target}")

    async def labs_domain_rank_overview(
        self, *, target: str, country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "target": target,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
        }]
        return await self._call("/dataforseo_labs/google/domain_rank_overview/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"labs.domain_rank_overview:{target}")

    async def labs_historical_serps(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        date_from: Optional[str] = None, date_to: Optional[str] = None,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keyword": keyword,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "date_from": date_from, "date_to": date_to,
        }]
        return await self._call("/dataforseo_labs/google/historical_serps/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"labs.historical_serps:{keyword}")

    async def labs_historical_rank_overview(
        self, *, target: str, country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "target": target,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
        }]
        return await self._call("/dataforseo_labs/google/historical_rank_overview/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"labs.historical_rank_overview:{target}")

    async def labs_page_intersection(
        self, *, pages: List[str], country_code: Optional[str] = None, language_code: str = "en",
        limit: int = 100, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        if len(pages) < 2:
            return DataForSEOResult(ok=False, error="page_intersection requires at least 2 page URLs")
        body = [{
            "pages": pages[:20],
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "limit": min(limit, 1000),
        }]
        return await self._call("/dataforseo_labs/google/page_intersection/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="labs.page_intersection")

    async def labs_bulk_traffic_estimation(
        self, *, targets: List[str], country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "targets": targets[:1000],
            "location_code": country_to_location(country_code),
            "language_code": language_code,
        }]
        return await self._call("/dataforseo_labs/google/bulk_traffic_estimation/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="labs.bulk_traffic_estimation")

    async def labs_categories_for_domain(
        self, *, target: str, country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "target": target,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
        }]
        return await self._call("/dataforseo_labs/google/categories_for_domain/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"labs.categories_for_domain:{target}")

    async def labs_keywords_for_site(
        self, *, target: str, country_code: Optional[str] = None, language_code: str = "en",
        limit: int = 100, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "target": target,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "limit": min(limit, 1000),
        }]
        return await self._call("/dataforseo_labs/google/keywords_for_site/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"labs.keywords_for_site:{target}")

    # Amazon Labs
    async def labs_amazon_related_keywords(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        limit: int = 100, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keyword": keyword,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "limit": min(limit, 1000),
        }]
        return await self._call("/dataforseo_labs/amazon/related_keywords/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"labs.amazon.related_keywords:{keyword}")

    async def labs_amazon_ranked_keywords(
        self, *, asin: str, country_code: Optional[str] = None, language_code: str = "en",
        limit: int = 100, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "asin": asin,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "limit": min(limit, 1000),
        }]
        return await self._call("/dataforseo_labs/amazon/ranked_keywords/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"labs.amazon.ranked_keywords:{asin}")

    async def labs_app_keywords(
        self, *, store: str, app_id: str, country_code: Optional[str] = None, language_code: str = "en",
        limit: int = 100, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        """store: google_play | app_store"""
        path = f"/dataforseo_labs/{store}/keywords_for_app/live"
        body = [{
            "app_id": app_id,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "limit": min(limit, 1000),
        }]
        return await self._call(path, body, attribution=attribution, log_kind="labs",
                                operation=f"labs.{store}.keywords_for_app:{app_id}")

    # ────────── 5. Backlinks API ──────────

    async def backlinks_summary(
        self, *, target: str, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"target": target}]
        return await self._call("/backlinks/summary/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"backlinks.summary:{target}")

    async def backlinks_history(
        self, *, target: str, date_from: Optional[str] = None, date_to: Optional[str] = None,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"target": target, "date_from": date_from, "date_to": date_to}]
        return await self._call("/backlinks/history/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"backlinks.history:{target}")

    async def backlinks_backlinks(
        self, *, target: str, mode: str = "as_is", limit: int = 100,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        """mode: as_is | one_per_domain | one_per_anchor"""
        body = [{"target": target, "mode": mode, "limit": min(limit, 1000)}]
        return await self._call("/backlinks/backlinks/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"backlinks.backlinks:{target}")

    async def backlinks_anchors(
        self, *, target: str, limit: int = 100,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"target": target, "limit": min(limit, 1000)}]
        return await self._call("/backlinks/anchors/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"backlinks.anchors:{target}")

    async def backlinks_referring_domains(
        self, *, target: str, limit: int = 100,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"target": target, "limit": min(limit, 1000)}]
        return await self._call("/backlinks/referring_domains/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"backlinks.referring_domains:{target}")

    async def backlinks_competitors(
        self, *, target: str, limit: int = 100,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"target": target, "limit": min(limit, 1000)}]
        return await self._call("/backlinks/competitors/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"backlinks.competitors:{target}")

    async def backlinks_domain_intersection(
        self, *, targets: List[str], limit: int = 100,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        if len(targets) < 2:
            return DataForSEOResult(ok=False, error="backlinks domain_intersection requires 2 targets")
        body = [{"target1": targets[0], "target2": targets[1], "limit": min(limit, 1000)}]
        return await self._call("/backlinks/domain_intersection/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"backlinks.domain_intersection:{targets[0]}/{targets[1]}")

    async def backlinks_bulk_spam_score(
        self, *, targets: List[str], attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"targets": targets[:1000]}]
        return await self._call("/backlinks/bulk_spam_score/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="backlinks.bulk_spam_score")

    async def backlinks_bulk_ranks(
        self, *, targets: List[str], attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"targets": targets[:1000]}]
        return await self._call("/backlinks/bulk_ranks/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="backlinks.bulk_ranks")

    async def backlinks_bulk_referring_domains(
        self, *, targets: List[str], attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"targets": targets[:1000]}]
        return await self._call("/backlinks/bulk_referring_domains/live", body,
                                attribution=attribution, log_kind="labs",
                                operation="backlinks.bulk_referring_domains")

    async def backlinks_timeseries_summary(
        self, *, target: str, date_from: Optional[str] = None, date_to: Optional[str] = None,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"target": target, "date_from": date_from, "date_to": date_to}]
        return await self._call("/backlinks/timeseries_summary/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"backlinks.timeseries_summary:{target}")

    # ────────── 6. OnPage API ──────────

    async def onpage_task_post(
        self, *, target: str, max_crawl_pages: int = 100, load_resources: bool = True,
        enable_javascript: bool = True, custom_js: Optional[str] = None,
        pingback_url: Optional[str] = None,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "target": target,
            "max_crawl_pages": max_crawl_pages,
            "load_resources": load_resources,
            "enable_javascript": enable_javascript,
            "custom_js": custom_js,
            "pingback_url": pingback_url,
        }]
        return await self._call("/on_page/task_post", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"onpage.task_post:{target}")

    async def onpage_summary(
        self, *, task_id: str, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        return await self._call(f"/on_page/summary/{task_id}", method="GET",
                                attribution=attribution, log_kind="labs",
                                operation=f"onpage.summary:{task_id}")

    async def onpage_pages(
        self, *, task_id: str, limit: int = 100, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"id": task_id, "limit": min(limit, 1000)}]
        return await self._call("/on_page/pages", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"onpage.pages:{task_id}")

    async def onpage_duplicate_tags(
        self, *, task_id: str, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"id": task_id}]
        return await self._call("/on_page/duplicate_tags", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"onpage.duplicate_tags:{task_id}")

    async def onpage_duplicate_content(
        self, *, task_id: str, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"id": task_id}]
        return await self._call("/on_page/duplicate_content", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"onpage.duplicate_content:{task_id}")

    async def onpage_redirect_chains(
        self, *, task_id: str, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"id": task_id}]
        return await self._call("/on_page/redirect_chains", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"onpage.redirect_chains:{task_id}")

    async def onpage_non_indexable(
        self, *, task_id: str, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"id": task_id}]
        return await self._call("/on_page/non_indexable", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"onpage.non_indexable:{task_id}")

    async def onpage_links(
        self, *, task_id: str, limit: int = 100, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"id": task_id, "limit": min(limit, 1000)}]
        return await self._call("/on_page/links", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"onpage.links:{task_id}")

    async def onpage_keyword_density(
        self, *, task_id: str, keyword_length: int = 1,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"id": task_id, "keyword_length": keyword_length}]
        return await self._call("/on_page/keyword_density", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"onpage.keyword_density:{task_id}")

    async def onpage_microdata(
        self, *, task_id: str, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"id": task_id}]
        return await self._call("/on_page/microdata", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"onpage.microdata:{task_id}")

    async def onpage_lighthouse(
        self, *, url: str, for_mobile: bool = False, categories: Optional[List[str]] = None,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "url": url, "for_mobile": for_mobile,
            "categories": categories or ["performance", "accessibility", "best-practices", "seo"],
        }]
        return await self._call("/on_page/lighthouse/live/json", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"onpage.lighthouse:{url}")

    async def onpage_instant_pages(
        self, *, url: str, enable_javascript: bool = True, load_resources: bool = True,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"url": url, "enable_javascript": enable_javascript, "load_resources": load_resources}]
        return await self._call("/on_page/instant_pages", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"onpage.instant_pages:{url}")

    async def onpage_content_parsing(
        self, *, url: str, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"url": url}]
        return await self._call("/on_page/content_parsing/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"onpage.content_parsing:{url}")

    # ────────── 7. Content Analysis API ──────────

    async def content_search(
        self, *, keyword: str, limit: int = 50,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "limit": min(limit, 1000)}]
        return await self._call("/content_analysis/search/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"content.search:{keyword}")

    async def content_summary(
        self, *, keyword: str, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword}]
        return await self._call("/content_analysis/summary/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"content.summary:{keyword}")

    async def content_sentiment_analysis(
        self, *, keyword: str, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword}]
        return await self._call("/content_analysis/sentiment_analysis/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"content.sentiment:{keyword}")

    async def content_phrase_trends(
        self, *, keyword: str, date_from: Optional[str] = None, date_to: Optional[str] = None,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "date_from": date_from, "date_to": date_to}]
        return await self._call("/content_analysis/phrase_trends/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"content.phrase_trends:{keyword}")

    async def content_rating_distribution(
        self, *, keyword: str, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword}]
        return await self._call("/content_analysis/rating_distribution/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"content.rating_distribution:{keyword}")

    # ────────── 8. Domain Analytics ──────────

    async def domain_technologies(
        self, *, target: str, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"target": target}]
        return await self._call("/domain_analytics/technologies/technologies/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"domain.technologies:{target}")

    async def domain_whois(
        self, *, target: str, attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"target": target}]
        return await self._call("/domain_analytics/whois/overview/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"domain.whois:{target}")

    async def domain_domains_by_technology(
        self, *, technology: str, limit: int = 100,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"technology": technology, "limit": min(limit, 1000)}]
        return await self._call("/domain_analytics/technologies/domains_by_technology/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"domain.domains_by_technology:{technology}")

    # ────────── 9. Merchant API ──────────

    async def merchant_google_shopping_products(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        """Note: returns task_post immediately. Caller polls task_get separately,
        or uses the existing dataforseo_merchant_service.py for full polling."""
        body = [{
            "keyword": keyword,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "priority": 2,
        }]
        return await self._call("/merchant/google/products/task_post", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"merchant.google.products:{keyword}")

    async def merchant_amazon_products(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keyword": keyword,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "priority": 2,
        }]
        return await self._call("/merchant/amazon/products/task_post", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"merchant.amazon.products:{keyword}")

    async def merchant_amazon_asin(
        self, *, asin: str, country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "asin": asin,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "priority": 2,
        }]
        return await self._call("/merchant/amazon/asin/task_post", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"merchant.amazon.asin:{asin}")

    # ────────── 10. App Data ──────────

    async def app_data_search(
        self, *, store: str, keyword: str, country_code: Optional[str] = None,
        language_code: str = "en", attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        """store: google_play | apple"""
        body = [{
            "keyword": keyword,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "priority": 2,
        }]
        return await self._call(f"/app_data/{store}/app_searches/task_post", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"app_data.{store}.search:{keyword}")

    async def app_data_app_info(
        self, *, store: str, app_id: str, country_code: Optional[str] = None,
        language_code: str = "en", attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "app_id": app_id,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "priority": 2,
        }]
        return await self._call(f"/app_data/{store}/app_info/task_post", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"app_data.{store}.app_info:{app_id}")

    async def app_data_app_reviews(
        self, *, store: str, app_id: str, country_code: Optional[str] = None,
        language_code: str = "en", attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "app_id": app_id,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "priority": 2,
        }]
        return await self._call(f"/app_data/{store}/app_reviews/task_post", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"app_data.{store}.app_reviews:{app_id}")

    # ────────── 11. Business Data ──────────

    async def business_google_my_business_info(
        self, *, keyword: str, country_code: Optional[str] = None, language_code: str = "en",
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keyword": keyword,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
        }]
        return await self._call("/business_data/google/my_business_info/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"business.gmb.info:{keyword}")

    async def business_google_reviews(
        self, *, place_id: str, country_code: Optional[str] = None,
        language_code: str = "en", attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "place_id": place_id,
            "location_code": country_to_location(country_code),
            "language_code": language_code,
            "priority": 2,
        }]
        return await self._call("/business_data/google/reviews/task_post", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"business.gmb.reviews:{place_id}")

    async def business_trustpilot_search(
        self, *, keyword: str, country_code: Optional[str] = None,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keyword": keyword,
            "location_code": country_to_location(country_code),
            "priority": 2,
        }]
        return await self._call("/business_data/trustpilot/search/task_post", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"business.trustpilot.search:{keyword}")

    async def business_trustpilot_reviews(
        self, *, domain: str, country_code: Optional[str] = None,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "domain": domain,
            "location_code": country_to_location(country_code),
            "priority": 2,
        }]
        return await self._call("/business_data/trustpilot/reviews/task_post", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"business.trustpilot.reviews:{domain}")

    async def business_tripadvisor_search(
        self, *, keyword: str, country_code: Optional[str] = None,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{
            "keyword": keyword,
            "location_code": country_to_location(country_code),
            "priority": 2,
        }]
        return await self._call("/business_data/tripadvisor/search/task_post", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"business.tripadvisor.search:{keyword}")

    async def business_pinterest(
        self, *, keyword: str, country_code: Optional[str] = None,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "location_code": country_to_location(country_code)}]
        return await self._call("/business_data/social_media/pinterest/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"business.pinterest:{keyword}")

    async def business_reddit(
        self, *, keyword: str, country_code: Optional[str] = None,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "location_code": country_to_location(country_code)}]
        return await self._call("/business_data/social_media/reddit/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"business.reddit:{keyword}")

    async def business_listings_search(
        self, *, keyword: str, country_code: Optional[str] = None,
        attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        body = [{"keyword": keyword, "location_code": country_to_location(country_code)}]
        return await self._call("/business_data/business_listings/search/live", body,
                                attribution=attribution, log_kind="labs",
                                operation=f"business.listings.search:{keyword}")

    # ────────── Generic escape hatch ──────────

    async def call_arbitrary(
        self, *, path: str, body: Optional[List[Dict[str, Any]]] = None,
        method: str = "POST", attribution: Optional[CostAttribution] = None,
    ) -> DataForSEOResult:
        """Last-resort: hit any DataForSEO endpoint by raw path. Used by the
        `seo_dataforseo_call` agent escape-hatch tool when no specific method
        fits. Caller is responsible for path correctness."""
        if not path.startswith("/"):
            path = "/" + path
        # Strip /v3 prefix if accidentally included
        if path.startswith("/v3/"):
            path = path[3:]
        return await self._call(path, body, method=method,
                                attribution=attribution, log_kind="labs",
                                operation=f"arbitrary:{path}")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton accessor (mirror of the pattern used elsewhere in
# this codebase: each integration service has a get_<name>_service() factory).
# ─────────────────────────────────────────────────────────────────────────────

_client: Optional[DataForSEOUnifiedClient] = None


def get_dataforseo_client() -> DataForSEOUnifiedClient:
    global _client
    if _client is None:
        _client = DataForSEOUnifiedClient()
    return _client
