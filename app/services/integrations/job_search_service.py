"""
Job Search Service — discovery across DataForSEO Google Jobs, Perplexity Sonar,
and Firecrawl careers-page scraping.

Three sources, each implemented as a separate async method:

  1. DataForSEO Google Jobs (`/serp/google_jobs/live/advanced`)
     Cheapest, broadest coverage (Google for Jobs aggregates Indeed, LinkedIn,
     Glassdoor, ZipRecruiter etc.). Flat ~$0.0006/req.

  2. Perplexity Sonar with `search_domain_filter=[linkedin.com/jobs, indeed.com,
     glassdoor.com, weworkremotely.com, ...]` and a JSON-schema response. Reads
     job pages directly when DataForSEO misses them. ~$0.005/sweep.

  3. Firecrawl scrape of user-pinned career-page URLs with a `JobListing[]`
     extraction schema. Direct from companies' own pages — highest signal,
     pays Firecrawl credits per scrape.

Each adapter returns `List[JobHit]`. The orchestrator (`job_research_service`)
canonicalizes URLs, dedupes by content_hash, then hands the survivors to the
classifier.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import httpx
from pydantic import BaseModel, Field

from app.services.integrations import job_cost_logger as costs

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Output shape — what each source returns to the orchestrator
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class JobHit:
    url: str
    canonical_url: str
    content_hash: str
    title: Optional[str] = None
    company: Optional[str] = None
    company_domain: Optional[str] = None
    location: Optional[str] = None
    is_remote: Optional[bool] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    salary_currency: Optional[str] = None
    salary_period: Optional[str] = None
    employment_type: Optional[str] = None
    seniority: Optional[str] = None
    description_excerpt: Optional[str] = None
    posted_at: Optional[str] = None  # ISO string, parsed by orchestrator
    source: str = "google_jobs"
    raw_payload: Dict[str, Any] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────────────
# URL canonicalization + dedupe key
# ────────────────────────────────────────────────────────────────────────────

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "ref", "src", "trk", "trackingId",
}


def canonicalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        p = urlparse(url.strip())
        scheme = p.scheme.lower() or "https"
        netloc = p.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = p.path.rstrip("/")
        # strip tracking params
        if p.query:
            from urllib.parse import parse_qsl, urlencode
            kept = [(k, v) for k, v in parse_qsl(p.query) if k not in _TRACKING_PARAMS]
            query = urlencode(kept)
        else:
            query = ""
        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return url.strip()


def domain_of(url: str) -> str:
    try:
        n = urlparse(url).netloc.lower()
        return n[4:] if n.startswith("www.") else n
    except Exception:
        return ""


# v0.3.6: SERP / aggregator URL patterns. Belt-and-suspenders against Perplexity
# occasionally returning category pages instead of individual postings.
_SERP_URL_PATTERNS = [
    # Indeed search results: /q-<keyword>-jobs.html, /jobs?q=…
    re.compile(r"indeed\.[a-z.]+/(q-|jobs\?|cmp/|companies/)", re.I),
    # Glassdoor SERP / index pages
    re.compile(r"glassdoor\.[a-z.]+/Job/", re.I),
    re.compile(r"glassdoor\.[a-z.]+/Search/", re.I),
    # LinkedIn job SEARCH (real postings live under /jobs/view/<id> instead)
    re.compile(r"linkedin\.com/jobs/search", re.I),
    re.compile(r"linkedin\.com/jobs/?$", re.I),
    # WeWorkRemotely category landing pages (postings live under /remote-jobs/<id>-…)
    re.compile(r"weworkremotely\.com/categories/", re.I),
    re.compile(r"weworkremotely\.com/remote-jobs/?$", re.I),
    # v0.4.3: ZipRecruiter category pages (price-prefixed slug + city suffix)
    re.compile(r"ziprecruiter\.com/Jobs/[^/]+/-in-", re.I),
    re.compile(r"ziprecruiter\.com/c/[^/]+/Jobs", re.I),
    # v0.4.3: Dice search URLs (same q- pattern as Indeed)
    re.compile(r"dice\.com/(jobs/q-|jobs/?\?)", re.I),
    # v0.4.3: Monster search URLs
    re.compile(r"monster\.[a-z.]+/jobs/(search|q-)", re.I),
    # Generic "search results" hints
    re.compile(r"[?&]q=", re.I),
    re.compile(r"/search[?/]", re.I),
    re.compile(r"-SRCH_", re.I),
]

_AGGREGATOR_COMPANY_NAMES = {
    "indeed", "glassdoor", "linkedin", "monster", "ziprecruiter", "dice",
    "wellfound", "angellist", "stack overflow", "stackoverflow",
    "weworkremotely", "we work remotely", "remoteok", "remote ok",
    "google", "google jobs",
}


def _is_aggregator_serp_url(url: str) -> bool:
    """Returns True if the URL is obviously a search-results / category landing /
    aggregator-index page rather than an individual job posting."""
    if not url:
        return False
    for pat in _SERP_URL_PATTERNS:
        if pat.search(url):
            return True
    return False


# v0.4.1 — broader category-page detection. Catches the long-tail of niche
# aggregators (Arc, Built-In, Turing, Crossover, RemoteRocketship, etc.) that
# the explicit-domain patterns above don't enumerate.
def _is_category_page_url(url: str) -> bool:
    """Heuristic: True if the URL is a category / topic-landing / job-board
    INDEX page (lists many jobs) vs an individual job posting (one specific
    role with apply CTA). Strategy:
      1. POSITIVE signal first — if the path has a 4+ digit number ANYWHERE,
         it's almost certainly a job ID → NOT a category. Same for ?jk=<hash>
         and known patterns like /viewjob, /job-listing.
      2. Otherwise check for explicit category indicators (/jobs/category/,
         path ending in /jobs or /jobs/?, /X-jobs path suffix).
      3. Otherwise: short slug-only last segment with no digits and looks like
         a topic word (e.g. /python, /remote-senior-python-developer, /jobs/python)
         → category.
    """
    if not url:
        return False
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        path = (p.path or "").rstrip("/").lower()
        # ── Strong positives — real job IDs / known posting patterns
        if re.search(r"/\d{4,}(/|$|-)", path):
            return False
        if "?jk=" in url.lower() or "viewjob" in path or "job-listing" in path:
            return False
        if "/jobs/view/" in path or "/job/view/" in path:
            return False
        # ── Strong negatives — explicit category / index page indicators
        if re.search(r"/jobs?/(category|categories|search|board)/", path):
            return True
        if re.fullmatch(r"/jobs?", path):
            return True
        # path ends in /<word>-jobs or /<word>-job (e.g. /python-jobs, /remote-python-jobs)
        if re.search(r"/[a-z][a-z0-9-]*-jobs?/?$", path):
            return True
        # ── Last-segment heuristic for short topic slugs
        last_seg = path.rsplit("/", 1)[-1] if "/" in path else path
        if last_seg and len(last_seg) < 35 and not re.search(r"\d", last_seg):
            # Looks like a topic/category slug (e.g. "python", "developer-engineer",
            # "remote-senior-python-developer") — short, no digits, only [a-z0-9-]
            if re.fullmatch(r"[a-z][a-z0-9-]*", last_seg):
                return True
        return False
    except Exception:
        return False


_PLACEHOLDER_COMPANY_PATTERNS = re.compile(
    r"^\s*(acme|companyxyz|example|sample|placeholder|fictional|fake|"
    r"company\s*[a-z]?|your\s+company|test\s*co|demo\s*co|"
    r"\[?company\s*name\]?|\[?employer\]?|n/?a|tbd|unknown|undisclosed)"
    r"(\s|\.|,|$|inc|llc|co|ltd)",
    re.I,
)
_SEQUENTIAL_ID_RE = re.compile(r"(\d{4,})")
_PALINDROMIC_ID_RE = re.compile(r"\b(\d{6,})\b")


def _is_placeholder_company(name: Optional[str]) -> bool:
    """v0.4.4: Reject Sonar-fabricated placeholder company names."""
    if not name:
        return False
    return bool(_PLACEHOLDER_COMPANY_PATTERNS.match(name.strip()))


def _looks_hallucinated_url(url: str) -> bool:
    """v0.4.4: Reject Sonar-fabricated URLs whose numeric IDs are obviously
    not real: sequential (1234567890, 0987654321), palindromic, or very short
    consecutive runs (123, 12345). Real job-board IDs are large random ints."""
    if not url:
        return False
    for m in _PALINDROMIC_ID_RE.finditer(url):
        digits = m.group(1)
        # Sequential ascending: 1234567890, 123456789
        if digits == "".join(str((int(digits[0]) + i) % 10) for i in range(len(digits))):
            return True
        # Sequential descending: 0987654321
        if digits == "".join(str((int(digits[0]) - i) % 10) for i in range(len(digits))):
            return True
        # Palindrome
        if len(digits) >= 6 and digits == digits[::-1]:
            return True
        # All same digit
        if len(set(digits)) == 1:
            return True
    return False


def _looks_like_category_title(title: Optional[str]) -> bool:
    """Title-shape heuristic for aggregator/category pages. Catches:
      - "25 Python jobs in Developer / Engineer"
      - "Python Job Board"
      - "Best Remote Python Jobs in NYC, NY 2026"
      - "Top Remote Python Jobs in San Francisco Bay Area, CA"
      - "Remote Python Jobs (May 2026)"
      - "Python Jobs" (bare plural)
    Does NOT catch real job titles like:
      - "Senior Software Engineer - Backend/Python - USA Only (100% Remote)"
      - "Principal Backend Engineer AI (Python) in Remote"
      - "Drupal with Python Developer (Senior)"
    """
    if not title:
        return False
    t = title.strip()
    # 1. Starts with a digit count: "25 Python jobs", "754 senior python developer Jobs"
    if re.match(r"^\d+\s+.{1,60}\bjobs?\b", t, re.I):
        return True
    # 2. "Job Board" / "Jobs Board"
    if re.search(r"\bjobs?\s+board\b", t, re.I):
        return True
    # 3. "Best/Top ... Jobs ..." (Built-In / Crossover style category pages)
    if re.match(r"^(best|top)\s+.{1,80}\bjobs?\b", t, re.I):
        return True
    # 4. Short plural-jobs title (Remote Python Jobs / Python Jobs / Remote Python Developer Jobs)
    cleaned = re.sub(r"\([^)]*\)", "", t).strip()  # strip parentheticals like "(May 2026)"
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split()
    # Ends in plural "Jobs" and the title is short enough to be a category landing
    if len(words) <= 6 and re.search(r"\bjobs?\b\s*$", cleaned, re.I):
        return True
    # 5. "Apply Now" CTA at end — usually a category page
    if re.search(r"\bapply\s+now\b\s*$", t, re.I):
        return True
    # 6. v0.4.3: ZipRecruiter-shaped titles: "$<price> <Role> Jobs in <Place>"
    #    e.g. "$116k-$175k Senior Python Developer Jobs in Houston, TX"
    if re.search(r"\bjobs?\s+in\s+[A-Z][a-zA-Z]+(\s*,\s*[A-Z]{2,})?", t):
        return True
    # 7. v0.4.3: Salary-band-prefixed titles ("$Xk-$Yk ...") almost always category
    if re.match(r"^\s*\$\d", t):
        return True
    return False


def content_hash(canonical_url: str, title: Optional[str], company: Optional[str]) -> str:
    h = hashlib.sha1()
    h.update((canonical_url or "").encode("utf-8"))
    h.update(b"|")
    h.update(((title or "").lower()[:200]).encode("utf-8"))
    h.update(b"|")
    h.update(((company or "").lower()[:80]).encode("utf-8"))
    return h.hexdigest()


# ────────────────────────────────────────────────────────────────────────────
# Source 1 — DataForSEO Google Jobs
# ────────────────────────────────────────────────────────────────────────────

_DATAFORSEO_BASE = "https://api.dataforseo.com/v3"


def _dfs_auth_header() -> Optional[str]:
    b64 = os.getenv("DATAFORSEO_BASE64") or ""
    if not b64:
        login = os.getenv("DATAFORSEO_LOGIN") or ""
        pw = os.getenv("DATAFORSEO_PASSWORD") or ""
        if not (login and pw):
            return None
        import base64
        b64 = base64.b64encode(f"{login}:{pw}".encode()).decode()
    return f"Basic {b64}"


async def search_via_dataforseo_jobs(
    *,
    keywords: List[str],
    location: Optional[str],
    country_code: Optional[str],
    remote_only: bool,
    employment_type: Optional[List[str]],
    attribution: costs.CostAttribution,
    limit: int = 30,
) -> List[JobHit]:
    """Hit DataForSEO Google Jobs SERP."""
    auth = _dfs_auth_header()
    if not auth:
        logger.info("job-search: DATAFORSEO_BASE64 not configured — skipping google_jobs")
        return []

    # v0.3.7: DataForSEO Google Jobs interprets the keyword field as a literal
    # search phrase, so joining 13 expanded keywords with spaces produces a
    # 100+ char query that matches nothing. Use ONLY the user's primary
    # keyword (first in the list); rely on Google's own synonym matching for
    # nearby titles.
    primary_keyword = (keywords[0] if keywords else "").strip()
    keyword_str = primary_keyword
    if remote_only and "remote" not in keyword_str.lower():
        keyword_str = f"{keyword_str} remote"

    # v0.3.6: DataForSEO requires a real geographic location_name (city / region /
    # country); "remote" or empty silently returns 0 hits. Map the user's location
    # to a real fallback. Prefer country_code as a city-resolution hint.
    _COUNTRY_DEFAULT_LOCATION = {
        "US": "United States", "GB": "London,England,United Kingdom", "UK": "London,England,United Kingdom",
        "DE": "Germany", "FR": "France", "ES": "Spain", "IT": "Italy",
        "NL": "Netherlands", "CA": "Canada", "AU": "Australia",
        "GR": "Greece", "PL": "Poland", "SE": "Sweden", "DK": "Denmark",
    }
    raw_loc = (location or "").strip().lower()
    if not location or raw_loc in {"remote", "anywhere", "worldwide", "global", "any"}:
        resolved_location = _COUNTRY_DEFAULT_LOCATION.get((country_code or "US").upper(), "United States")
    else:
        resolved_location = location

    body = [{
        "keyword": keyword_str,
        "language_code": "en",
        "location_name": resolved_location,
        "depth": min(max(limit, 10), 100),
    }]
    if country_code:
        body[0]["location_country_code"] = country_code.upper()
    if employment_type:
        body[0]["employment_type"] = ",".join(employment_type)

    started = time.time()
    success = True
    err: Optional[str] = None
    hits: List[JobHit] = []
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=8.0)) as client:
            # DataForSEO Google Jobs endpoint: /v3/serp/google/jobs/live/advanced
            # (the path uses /google/jobs/, NOT /google_jobs/ — common confusion)
            resp = await client.post(
                f"{_DATAFORSEO_BASE}/serp/google/jobs/live/advanced",
                headers={"Authorization": auth, "Content-Type": "application/json"},
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        tasks = ((data or {}).get("tasks") or [])
        for task in tasks:
            for result in (task.get("result") or []):
                for item in (result.get("items") or []):
                    # DataForSEO returns `type` either as 'google_jobs_serp' (older) or
                    # 'jobs_element' (current). Accept both for forward compat.
                    item_type = (item.get("type") or "").lower()
                    if item_type not in ("google_jobs_serp", "jobs_element"):
                        continue
                    url = (item.get("apply_link") or {}).get("link") or item.get("url") or ""
                    if not url:
                        continue
                    canonical = canonicalize_url(url)
                    title = item.get("title")
                    company = item.get("company_name")
                    salary = item.get("salary") or {}
                    contract = item.get("contract_type") or item.get("schedule_type")
                    hits.append(JobHit(
                        url=url,
                        canonical_url=canonical,
                        content_hash=content_hash(canonical, title, company),
                        title=title,
                        company=company,
                        company_domain=domain_of(url),
                        location=item.get("location"),
                        is_remote=("remote" in (item.get("location") or "").lower()) or None,
                        salary_min=_to_int(salary.get("min_value")),
                        salary_max=_to_int(salary.get("max_value")),
                        salary_currency=salary.get("currency"),
                        salary_period=salary.get("type"),
                        employment_type=contract,
                        description_excerpt=(item.get("description") or "")[:600] or None,
                        posted_at=item.get("date_posted"),
                        source="google_jobs",
                        raw_payload={"thumbnail": item.get("thumbnail"), "via": item.get("via")},
                    ))
    except Exception as e:
        success = False
        err = str(e)[:200]
        logger.warning(f"job-search dataforseo: {err}")

    costs.log_dataforseo_jobs_call(
        attribution=attribution,
        query=keyword_str,
        location=location or "",
        hits_returned=len(hits),
        latency_ms=int((time.time() - started) * 1000),
        success=success,
        error_message=err,
    )
    return hits


def _to_int(v: Any) -> Optional[int]:
    try:
        return int(float(v)) if v is not None else None
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────────────
# Source 5 (v0.4) — DataForSEO general Google web SERP
#
# Why this exists: `search_via_dataforseo_jobs` only hits Google's STRUCTURED
# Jobs index (pages with Schema.org JobPosting markup). For niche queries,
# Google Jobs is sparse. The general web SERP catches Lever / Greenhouse /
# Workable / SmartRecruiters / company-blog career announcements that Google
# indexes as regular pages but doesn't always promote into Google Jobs.
#
# Same DataForSEO auth, similar cost (~$0.0006/req).
# ────────────────────────────────────────────────────────────────────────────

async def search_via_dataforseo_serp(
    *,
    queries: List[str],
    country_code: Optional[str],
    attribution: costs.CostAttribution,
    limit_per_query: int = 10,
) -> List[JobHit]:
    """Fan out general Google web search across query variations. Returns JobHits.
    The SERP-URL filter still drops obvious aggregator pages; classifier filters
    the rest downstream."""
    auth = _dfs_auth_header()
    if not auth or not queries:
        return []

    _COUNTRY_DEFAULT_LOCATION = {
        "US": "United States", "GB": "London,England,United Kingdom",
        "DE": "Germany", "FR": "France", "ES": "Spain", "IT": "Italy",
        "NL": "Netherlands", "CA": "Canada", "AU": "Australia",
        "GR": "Greece", "PL": "Poland", "SE": "Sweden",
    }
    location_name = _COUNTRY_DEFAULT_LOCATION.get((country_code or "US").upper(), "United States")

    async def _one(query: str) -> List[JobHit]:
        started = time.time()
        out: List[JobHit] = []
        err: Optional[str] = None
        success = True
        try:
            body = [{
                "keyword": query,
                "language_code": "en",
                "location_name": location_name,
                "depth": min(max(limit_per_query, 10), 30),
            }]
            if country_code:
                body[0]["location_country_code"] = country_code.upper()
            async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=8.0)) as client:
                resp = await client.post(
                    f"{_DATAFORSEO_BASE}/serp/google/organic/live/advanced",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
            for task in ((data or {}).get("tasks") or []):
                for result in (task.get("result") or []):
                    for item in (result.get("items") or []):
                        if (item.get("type") or "").lower() not in ("organic", "organic_serp", "featured_snippet"):
                            continue
                        url = item.get("url") or ""
                        if not url or not url.startswith(("http://", "https://")):
                            continue
                        title = item.get("title")
                        # v0.4.1: aggressive category-page filtering. SERP results come
                        # from anywhere on the web, so most niche aggregator landing
                        # pages will slip through unless we filter URL-shape AND
                        # title-shape. Anything that looks like a topic-index page,
                        # "Best X Jobs in Y", "N python jobs", "Python Job Board",
                        # etc. gets dropped here BEFORE persistence.
                        if _is_aggregator_serp_url(url) or _is_category_page_url(url):
                            continue
                        if _looks_like_category_title(title):
                            continue
                        canonical = canonicalize_url(url)
                        host = domain_of(url)
                        # v0.4.1: NEVER set company from the host for google_serp hits.
                        # The host is the aggregator (arc.dev, weworkremotely.com,
                        # careers-cotiviti.icims.com); the actual employer must be
                        # extracted from the title or description by the classifier
                        # downstream. Leaving company=None forces honest attribution
                        # rather than misleading "Arc" / "Cotiviti" / "Weworkremotely"
                        # labels.
                        out.append(JobHit(
                            url=url,
                            canonical_url=canonical,
                            content_hash=content_hash(canonical, title, None),
                            title=title,
                            company=None,
                            company_domain=host,
                            description_excerpt=(item.get("description") or "")[:500] or None,
                            source="google_serp",
                            raw_payload={"serp_query": query[:120]},
                        ))
        except Exception as e:
            success = False
            err = str(e)[:200]
            logger.warning(f"job-search dataforseo-serp ({query[:60]}): {err}")

        costs.log_external_call(
            operation_type="job_research.discovery.dataforseo_serp",
            model_name="dataforseo-google-organic",
            raw_cost_usd=costs.DATAFORSEO_JOBS_PER_CALL,  # same rate as the Jobs endpoint
            attribution=attribution,
            latency_ms=int((time.time() - started) * 1000),
            extra_metadata={"query": query[:120], "location": location_name, "hits_returned": len(out)},
            success=success,
            error_message=err,
        )
        return out

    # Run all variations in parallel; flatten + return.
    batches = await asyncio.gather(*[_one(q) for q in queries[:5]], return_exceptions=False)
    flat: List[JobHit] = []
    for b in batches:
        flat.extend(b)
    return flat


def build_query_variations(primary_keyword: str, location: Optional[str], remote_only: bool) -> List[str]:
    """Return a list of query-shape variations for the user's primary keyword.
    Used by the SERP source and (optionally) Perplexity to broaden recall.

    These are SEARCH PHRASE templates, not title variants. Title variants live
    in tracked_jobs.expanded_keywords (used by the classifier). Query phrasings
    optionally come from Haiku via the keyword-expansion service and merge with
    these defaults."""
    base = (primary_keyword or "").strip()
    if not base:
        return []
    where = (location or "").strip()
    where_part = ""
    if where and where.lower() not in {"remote", "anywhere", "worldwide", "global", "any"}:
        where_part = f" {where}"
    remote_suffix = " remote" if remote_only else ""
    return [
        f"{base}{remote_suffix} jobs{where_part}",
        f"{base} careers page{where_part}",
        f"{base}{remote_suffix} hiring{where_part}",
        f"{base}{remote_suffix} job opening{where_part}",
        f"{base}{remote_suffix} apply{where_part}",
    ]


# ────────────────────────────────────────────────────────────────────────────
# Source 2 — Perplexity Sonar with job-board domain filter + JSON schema
# ────────────────────────────────────────────────────────────────────────────

_PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

# Hardcoded fallback used only if the DB-side `job_research_sites` table is
# unreachable or empty. The authoritative list lives in the DB and is editable
# at /admin/knowledge-base/job-sources.
_DEFAULT_JOB_DOMAINS = [
    "linkedin.com",
    "indeed.com",
    "glassdoor.com",
    "weworkremotely.com",
    "remoteok.com",
    "stackoverflow.com",
    "wellfound.com",
    "ycombinator.com",
    "dice.com",
    "monster.com",
]


def _load_perplexity_domains_from_db() -> List[str]:
    """Load the operator-curated Perplexity domain filter from job_research_sites.
    Returns the hardcoded fallback if the DB read fails or returns nothing."""
    try:
        from app.services.core.supabase_client import get_supabase_client
        sb = get_supabase_client().client
        res = (
            sb.table("job_research_sites")
            .select("url_or_domain")
            .eq("site_type", "perplexity_domain")
            .eq("is_enabled", True)
            .execute()
        )
        rows = res.data or []
        domains = [(r.get("url_or_domain") or "").strip().lower() for r in rows if r.get("url_or_domain")]
        domains = [d for d in domains if d]
        if domains:
            return domains
    except Exception as e:
        logger.warning(f"job-search: load DB perplexity domains failed (using fallback): {e}")
    return list(_DEFAULT_JOB_DOMAINS)


def load_site_defaults_from_db(site_type: str) -> List[str]:
    """Load the operator-curated URL list for a given site_type from job_research_sites.
    Used by the refresh pipeline to UNION per-tracked_job URLs with global defaults.
    Returns [] if the DB read fails or the list is empty (so the engine cleanly skips)."""
    try:
        from app.services.core.supabase_client import get_supabase_client
        sb = get_supabase_client().client
        res = (
            sb.table("job_research_sites")
            .select("url_or_domain")
            .eq("site_type", site_type)
            .eq("is_enabled", True)
            .execute()
        )
        rows = res.data or []
        urls = [(r.get("url_or_domain") or "").strip() for r in rows if r.get("url_or_domain")]
        return [u for u in urls if u]
    except Exception as e:
        logger.warning(f"job-search: load DB defaults for {site_type} failed: {e}")
        return []

_JOB_LISTING_SCHEMA = {
    "type": "object",
    "properties": {
        "listings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "title": {"type": "string"},
                    "company": {"type": "string"},
                    "location": {"type": "string"},
                    "is_remote": {"type": "boolean"},
                    "salary_min": {"type": ["integer", "null"]},
                    "salary_max": {"type": ["integer", "null"]},
                    "salary_currency": {"type": ["string", "null"]},
                    "employment_type": {"type": ["string", "null"]},
                    "seniority": {"type": ["string", "null"]},
                    "description_excerpt": {"type": "string"},
                    "posted_at": {"type": ["string", "null"]},
                },
                "required": ["url", "title", "company"],
            },
        },
    },
    "required": ["listings"],
}


async def search_via_perplexity(
    *,
    keywords: List[str],
    location: Optional[str],
    remote_only: bool,
    seniority: Optional[str],
    excluded_keywords: Optional[List[str]],
    excluded_companies: Optional[List[str]],
    attribution: costs.CostAttribution,
    model: str = "sonar",
    extra_domains: Optional[List[str]] = None,
    limit: int = 15,
) -> List[JobHit]:
    api_key = os.getenv("PERPLEXITY_API_KEY") or ""
    if not api_key:
        logger.info("job-search: PERPLEXITY_API_KEY not configured — skipping perplexity")
        return []

    # v0.3.7: Sonar handles long OR-lists poorly. Cap at the user's primary 3
    # keywords (originals + a couple expansions). The classifier downstream
    # accepts the broader keyword set, but the DISCOVERY query stays tight.
    top_keywords = keywords[:3] if len(keywords) > 3 else keywords
    keyword_str = " OR ".join(f'"{k}"' for k in top_keywords)
    excl_kw = " ".join(f"NOT {k}" for k in (excluded_keywords or []))
    excl_co = " ".join(f"NOT {c}" for c in (excluded_companies or []))
    # v0.3.7: don't emit "in remote" — Sonar tries to parse it as a place. The
    # remote_clause below carries the remote constraint separately.
    raw_loc = (location or "").strip().lower()
    location_clause = (
        f"in {location}" if location and raw_loc not in {"remote", "anywhere", "worldwide", "global", "any"}
        else ""
    )
    remote_clause = "Remote-only roles (work-from-anywhere)." if remote_only else ""
    seniority_clause = f"Seniority: {seniority}." if seniority and seniority != "any" else ""

    # v0.4.4 — anti-hallucination overhaul. Sonar-pro will FABRICATE postings to
    # fill a structured JSON array when it can't find enough real ones (e.g.
    # company='Acme Inc.', palindromic Glassdoor IDs, sequential WeWorkRemotely
    # IDs 12345/12346/12347). Three counter-measures applied:
    #   1. Cap limit aggressively (5 not 15) — less pressure to invent
    #   2. Explicit anti-hallucination directive + permission for empty results
    #   3. Post-call hallucination detector (see _looks_like_hallucinated_hit)
    # v0.4.5: bump cap 5→7. The anti-hallucination guards (placeholder companies,
    # sequential/palindromic IDs) catch fabricated entries; more headroom lets
    # Sonar surface 2-3 real listings per call instead of stopping at 1-2.
    capped_limit = min(limit, 7)
    user_prompt = (
        f"Find INDIVIDUAL JOB POSTINGS for: {keyword_str} {location_clause}. "
        f"{remote_clause} {seniority_clause} "
        f"Exclude: {excl_kw} {excl_co}. "
        f"\n\n"
        f"⚠️ CRITICAL ANTI-HALLUCINATION RULES — DO NOT VIOLATE:\n"
        f"  - ONLY return URLs you have ACTUALLY found through search. NEVER guess, infer,\n"
        f"    or invent a URL/job-ID/company-name that you have not verified.\n"
        f"  - If you find fewer than {capped_limit} verifiable postings, RETURN FEWER.\n"
        f"    An empty `listings` array IS ACCEPTABLE. Quality over quantity.\n"
        f"  - NEVER use placeholder company names like 'Acme Inc.', 'CompanyXYZ',\n"
        f"    'Example Co.', 'Sample Corp', '[Company Name]', or any obvious placeholder.\n"
        f"    If you don't know the company, leave the field blank rather than inventing one.\n"
        f"  - NEVER fabricate sequential or palindromic numeric IDs in URLs. Real IDs are\n"
        f"    typically large random-looking integers or hashes.\n"
        f"\n"
        f"URL FILTER — each result must be a SPECIFIC JOB AD, not a listing page:\n"
        f"  ✅ ACCEPT: URLs that resolve to ONE position with title + company + apply CTA.\n"
        f"     Examples: linkedin.com/jobs/view/<id>, indeed.com/viewjob?jk=<hash>,\n"
        f"     glassdoor.com/job-listing/<slug>-JV_<id>.htm, <co>.greenhouse.io/jobs/<id>,\n"
        f"     <co>.lever.co/<job-id>, weworkremotely.com/remote-jobs/<id>-<slug>.\n"
        f"  ❌ REJECT: search-results pages, /q- paths, /search paths, /SRCH paths,\n"
        f"     category landing pages, 'X jobs in Y' aggregator pages.\n"
        f"\n"
        f"Return UP TO {capped_limit} verified postings as JSON, posted within the last 30 days. "
        f"Set `company` to the actual hiring employer (NEVER 'Indeed'/'Glassdoor' or a placeholder); "
        f"leave blank if unknown."
    )

    # v0.4: load the operator-curated list from job_research_sites (editable in the
    # hidden admin page at /admin/knowledge-base/job-sources). Falls back to the
    # hardcoded constant if the DB read fails or returns nothing.
    domains = _load_perplexity_domains_from_db()
    if extra_domains:
        for d in extra_domains:
            if d and d not in domains:
                domains.append(d)
    domains = domains[:10]  # Perplexity caps at 10

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a job-search assistant. Return ONLY JSON matching the requested schema."},
            {"role": "user", "content": user_prompt},
        ],
        "search_domain_filter": domains,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": _JOB_LISTING_SCHEMA},
        },
        "max_tokens": 3000,
        "temperature": 0.0,
    }

    started = time.time()
    success = True
    err: Optional[str] = None
    hits: List[JobHit] = []
    in_tokens = 0
    out_tokens = 0
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=10.0)) as client:
            resp = await client.post(
                _PERPLEXITY_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        usage = data.get("usage") or {}
        in_tokens = int(usage.get("prompt_tokens") or 0)
        out_tokens = int(usage.get("completion_tokens") or 0)

        choice = (data.get("choices") or [{}])[0]
        msg = (choice.get("message") or {}).get("content") or ""
        # Parse JSON; Perplexity sometimes wraps in code fences.
        import json
        try:
            parsed = json.loads(msg)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", msg)
            parsed = json.loads(m.group(0)) if m else {"listings": []}

        for item in (parsed.get("listings") or []):
            url = (item.get("url") or "").strip()
            if not url or not url.startswith(("http://", "https://")):
                continue
            title = item.get("title")
            # v0.3.6 + v0.4.1: post-filter for SERP / aggregator / category pages.
            if _is_aggregator_serp_url(url) or _is_category_page_url(url):
                logger.info(f"job-search perplexity: dropping category/SERP url: {url[:120]}")
                continue
            if _looks_like_category_title(title):
                logger.info(f"job-search perplexity: dropping category-shaped title: {(title or '')[:80]}")
                continue
            # v0.4.4: hallucination check — Sonar fabricates URLs with sequential
            # or palindromic IDs when asked to fill a structured listings array.
            if _looks_hallucinated_url(url):
                logger.warning(f"job-search perplexity: dropping likely-hallucinated URL (sequential/palindromic ID): {url[:120]}")
                continue
            canonical = canonicalize_url(url)
            company = item.get("company")
            # v0.3.6: also reject when the "company" is one of the aggregators
            # (Indeed / Glassdoor / LinkedIn / Monster as the company means it's a SERP)
            if company and company.strip().lower() in _AGGREGATOR_COMPANY_NAMES:
                logger.info(f"job-search perplexity: dropping result where company='{company}' (aggregator masquerading as employer)")
                continue
            # v0.4.4: drop fabricated placeholder companies ('Acme Inc.', 'CompanyXYZ', etc.)
            if _is_placeholder_company(company):
                logger.warning(f"job-search perplexity: dropping placeholder company '{company}' (likely hallucination)")
                continue
            hits.append(JobHit(
                url=url,
                canonical_url=canonical,
                content_hash=content_hash(canonical, title, company),
                title=title,
                company=company,
                company_domain=domain_of(url),
                location=item.get("location"),
                is_remote=item.get("is_remote"),
                salary_min=_to_int(item.get("salary_min")),
                salary_max=_to_int(item.get("salary_max")),
                salary_currency=item.get("salary_currency"),
                employment_type=item.get("employment_type"),
                seniority=item.get("seniority"),
                description_excerpt=(item.get("description_excerpt") or "")[:600] or None,
                posted_at=item.get("posted_at"),
                source="perplexity_sonar",
            ))
    except Exception as e:
        success = False
        err = str(e)[:200]
        logger.warning(f"job-search perplexity: {err}")

    costs.log_perplexity_call(
        attribution=attribution,
        model=model,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        hits_returned=len(hits),
        latency_ms=int((time.time() - started) * 1000),
        success=success,
        error_message=err,
    )
    return hits


# ────────────────────────────────────────────────────────────────────────────
# Source 3 — Firecrawl scrape of user-pinned careers pages
# ────────────────────────────────────────────────────────────────────────────

class _FirecrawlJobListing(BaseModel):
    url: str = Field(..., description="Direct application link or job posting URL")
    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    employment_type: Optional[str] = None
    seniority: Optional[str] = None
    description_excerpt: Optional[str] = Field(None, description="Short description, max 600 chars")
    posted_at: Optional[str] = None
    is_remote: Optional[bool] = None


class _FirecrawlCareersPage(BaseModel):
    listings: List[_FirecrawlJobListing] = Field(default_factory=list)


async def search_via_firecrawl_careers(
    *,
    careers_urls: List[str],
    company_hint: Optional[str],
    attribution: costs.CostAttribution,
) -> List[JobHit]:
    if not careers_urls:
        return []
    api_key = os.getenv("FIRECRAWL_API_KEY") or ""
    if not api_key:
        logger.info("job-search: FIRECRAWL_API_KEY not configured — skipping careers_pages")
        return []

    schema = _FirecrawlCareersPage.model_json_schema()

    async def _scrape_one(url: str) -> List[JobHit]:
        started = time.time()
        success = True
        err: Optional[str] = None
        out: List[JobHit] = []
        credits = 1
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                resp = await client.post(
                    "https://api.firecrawl.dev/v2/scrape",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "url": url,
                        "formats": [{"type": "json", "schema": schema}],
                        "onlyMainContent": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            payload = (data.get("data") or {})
            credits = int(data.get("creditsUsed") or 1)
            extracted = (payload.get("json") or {}) or {}
            page_company = company_hint or domain_of(url).split(".")[0].title()

            for item in (extracted.get("listings") or []):
                listing_url = (item.get("url") or "").strip()
                if not listing_url:
                    continue
                if not listing_url.startswith(("http://", "https://")):
                    # relative link — resolve against careers page
                    base = urlparse(url)
                    listing_url = f"{base.scheme}://{base.netloc}{listing_url if listing_url.startswith('/') else '/' + listing_url}"
                canonical = canonicalize_url(listing_url)
                title = item.get("title")
                co = item.get("company") or page_company
                out.append(JobHit(
                    url=listing_url,
                    canonical_url=canonical,
                    content_hash=content_hash(canonical, title, co),
                    title=title,
                    company=co,
                    company_domain=domain_of(listing_url) or domain_of(url),
                    location=item.get("location"),
                    is_remote=item.get("is_remote"),
                    employment_type=item.get("employment_type"),
                    seniority=item.get("seniority"),
                    description_excerpt=(item.get("description_excerpt") or "")[:600] or None,
                    posted_at=item.get("posted_at"),
                    source="firecrawl_careers",
                    raw_payload={"careers_page": url},
                ))
        except Exception as e:
            success = False
            err = str(e)[:200]
            logger.warning(f"job-search firecrawl ({url}): {err}")

        costs.log_firecrawl_call(
            attribution=attribution,
            url=url,
            credits_used=credits,
            listings_extracted=len(out),
            latency_ms=int((time.time() - started) * 1000),
            success=success,
            error_message=err,
        )
        return out

    results = await asyncio.gather(*[_scrape_one(u) for u in careers_urls[:10]], return_exceptions=False)
    flat: List[JobHit] = []
    for batch in results:
        flat.extend(batch)
    return flat


# ────────────────────────────────────────────────────────────────────────────
# Source 4 — RSS feeds (v0.3)
#
# Users pin RSS URLs (their target companies' careers feeds, WeWorkRemotely,
# remoteok.com/remote-jobs.rss, HN "Who's Hiring" archives, etc.). We poll each
# feed, extract item title + link + description, and emit JobHits. No LLM here
# — the classifier downstream decides relevance like for the other sources.
#
# Implementation note: we parse RSS/Atom by hand via xml.etree to avoid a
# `feedparser` dependency. Feeds are small (typically <50 items × <2KB each),
# so the parse is microseconds even on the slow path.
# ────────────────────────────────────────────────────────────────────────────

import xml.etree.ElementTree as _ET
from datetime import datetime, timezone


def _strip_html(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"<[^>]+>", " ", s).strip()


def _parse_rss_date(s: Optional[str]) -> Optional[str]:
    """RSS pubDate is RFC-822 ('Mon, 14 May 2026 09:30:00 GMT'); Atom is ISO-8601.
    Return ISO-8601 or the raw string if we can't parse."""
    if not s:
        return None
    s = s.strip()
    # Try RFC-822 first
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(s).astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    # Atom ISO
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except Exception:
        return s


async def search_via_rss_feeds(
    *,
    feed_urls: List[str],
    attribution: costs.CostAttribution,
    max_items_per_feed: int = 25,
) -> List[JobHit]:
    """Fetch each RSS/Atom URL, extract items, emit JobHits."""
    if not feed_urls:
        return []

    async def _fetch_one(url: str) -> List[JobHit]:
        started = time.time()
        out: List[JobHit] = []
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
                resp = await client.get(url, headers={"User-Agent": "MaterialKAI-JobBot/1.0"})
                resp.raise_for_status()
                body = resp.text

            try:
                root = _ET.fromstring(body)
            except _ET.ParseError as pe:
                logger.warning(f"job-search rss parse failed for {url}: {pe}")
                return []

            # RSS 2.0: root.channel.item
            # Atom 1.0: root.entry
            # Strip namespace prefix from tags for lookup
            def _local(tag: str) -> str:
                return tag.split("}", 1)[-1] if "}" in tag else tag

            items = []
            # Try Atom first
            for entry in root.iter():
                if _local(entry.tag).lower() == "entry":
                    items.append(("atom", entry))
                elif _local(entry.tag).lower() == "item":
                    items.append(("rss", entry))

            feed_host = domain_of(url)

            for kind, item in items[:max_items_per_feed]:
                title = None
                link = None
                desc = None
                pub = None
                author = None
                for child in item:
                    tag = _local(child.tag).lower()
                    if tag == "title":
                        title = (child.text or "").strip()
                    elif tag == "link":
                        if kind == "rss":
                            link = (child.text or "").strip()
                        else:
                            # Atom: link is in href attr; prefer rel="alternate"
                            href = child.attrib.get("href")
                            rel = (child.attrib.get("rel") or "alternate").lower()
                            if href and (rel == "alternate" or not link):
                                link = href
                    elif tag in ("description", "summary", "content"):
                        desc = _strip_html((child.text or ""))[:600] or desc
                    elif tag in ("pubdate", "published", "updated"):
                        pub = _parse_rss_date(child.text)
                    elif tag in ("author", "creator"):
                        # Atom <author><name/></author> or RSS <author>
                        if child.text:
                            author = child.text.strip()
                        else:
                            for inner in child:
                                if _local(inner.tag).lower() == "name" and inner.text:
                                    author = inner.text.strip()
                                    break

                if not link or not title:
                    continue
                if not link.startswith(("http://", "https://")):
                    continue
                canonical = canonicalize_url(link)
                # In RSS, the company is often the feed's host (e.g. stripe.com/jobs.rss → Stripe)
                # Fallback to the author if present.
                company = author or feed_host.split(".")[0].title() if feed_host else None
                out.append(JobHit(
                    url=link,
                    canonical_url=canonical,
                    content_hash=content_hash(canonical, title, company),
                    title=title,
                    company=company,
                    company_domain=domain_of(link) or feed_host,
                    description_excerpt=desc,
                    posted_at=pub,
                    source="rss_feed",
                    raw_payload={"feed_url": url},
                ))
        except Exception as e:
            logger.warning(f"job-search rss ({url}): {str(e)[:200]}")
        finally:
            # Free (no per-call billing for RSS); still emit a log row for visibility.
            costs.log_external_call(
                operation_type="job_research.discovery.rss_feed",
                model_name="rss-poll",
                raw_cost_usd=0.0,
                attribution=attribution,
                latency_ms=int((time.time() - started) * 1000),
                extra_metadata={"url": url[:200], "items_extracted": len(out)},
                success=True,
                markup_multiplier=1.0,
            )
        return out

    results = await asyncio.gather(*[_fetch_one(u) for u in feed_urls[:10]], return_exceptions=False)
    flat: List[JobHit] = []
    for batch in results:
        flat.extend(batch)
    return flat


# ────────────────────────────────────────────────────────────────────────────
# Cross-source dedupe (in-memory, before classifier)
# ────────────────────────────────────────────────────────────────────────────

def dedupe_hits(hits: List[JobHit]) -> List[JobHit]:
    seen: Dict[str, JobHit] = {}
    # source priority: firecrawl_careers (direct from company) > rss_feed (also direct, often
    # the same data) > perplexity (read page) > google_jobs (feed of feeds)
    priority = {"firecrawl_careers": 4, "rss_feed": 3, "perplexity_sonar": 2, "google_jobs": 1}
    for h in hits:
        existing = seen.get(h.content_hash)
        if not existing or priority.get(h.source, 0) > priority.get(existing.source, 0):
            seen[h.content_hash] = h
    return list(seen.values())
