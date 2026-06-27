"""
Job Research Service — orchestration for the job-research module.

Single chokepoint that:
  - Provides CRUD for tracked_jobs (internal flow + external api_key flow)
  - Runs the refresh pipeline: discovery → dedupe → classify → persist → cadence
  - Reads listings + summary + history for the UI
  - Marks user actions on individual listings (saved/applied/dismissed)

The refresh pipeline is the heart of the module. It:
  1. Loads the tracked_job + builds JobFacets + (optionally) expands keywords via Haiku.
  2. Fans out across enabled sources in parallel (asyncio.gather):
       - DataForSEO Google Jobs   (cheap, broad)
       - Perplexity Sonar         (deep page reading on big boards)
       - Firecrawl careers pages  (direct from companies the user pinned)
  3. Cross-source dedupe by content_hash.
  4. Drops URLs/companies/domains in `job_excluded_urls` (user blocklist).
  5. Drops dupes already in the DB (UNIQUE (tracked_job_id, content_hash) catches
     the rest, but pre-filtering avoids wasted classifier credits).
  6. Classifier batch (rule shortcut → 7d cache → Haiku).
  7. Persists `match` + `tangential` + `unverifiable` rows; drops `mismatch`.
  8. Updates denormalized cache (`current_*`) on tracked_jobs.
  9. Stamps refresh cost via `stamp_job_refresh_cost` RPC.
 10. Updates next_check_at via `update_tracked_job_cadence` RPC.

The digest dispatcher (in app/modules/job_research_notifications/service.py)
is the *consumer* of `job_listings` rows that haven't been included in a
digest yet. It runs on a separate hourly cron tick at :05 (this refresh runs
at :45) so a digest can pick up rows that were just refreshed an hour ago.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from app.services.core.supabase_client import get_supabase_client
from app.services.integrations import job_cost_logger as costs
from app.services.integrations import job_agent_runs as bookkeeping
from app.services.integrations.job_classifier_service import (
    JobFacets, classify_batch,
)
from app.services.integrations.job_keyword_expansion_service import expand_keywords
from app.services.integrations.job_salary_normalizer import normalize_listing_in_place
from app.services.integrations.job_search_service import (
    JobHit, build_query_variations, dedupe_hits, discover_local_job_boards,
    search_via_dataforseo_jobs, search_via_dataforseo_serp,
    search_via_firecrawl_careers, search_via_perplexity, search_via_rss_feeds,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(d: Optional[datetime]) -> Optional[str]:
    return d.isoformat() if d else None


def _parse_dt(v: Any) -> Optional[datetime]:
    if not v:
        return None
    if isinstance(v, datetime):
        return v
    try:
        s = str(v)
        # Accept ISO with Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


_REL_AGO_RE = re.compile(r"(\d+)\s*\+?\s*(hour|hr|day|week|month|year)s?\s*ago", re.I)


def normalize_posted_at(v: Any) -> Optional[str]:
    """Normalize a source-reported posted date into an ISO-8601 timestamp string
    (or None) so it never crashes the `posted_at timestamptz` insert.

    Job boards render the posted date as human text — "New", "Just posted",
    "2 days ago", "1 month ago", "30+ days ago", "Today", "Yesterday" — which
    Postgres rejects (22007). Before this normalizer every careers-page /
    job-board listing carrying a relative date was silently dropped at insert
    (audit 2026-06-26: 47/56 matched listings lost in one refresh).

    Strategy: already-ISO/datetime → pass through; relative phrases → compute
    from now(); anything unparseable → None (drop the field, keep the listing).
    """
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    s = str(v).strip()
    if not s:
        return None

    # 1) Already a real timestamp? Trust _parse_dt (handles ISO + trailing Z).
    parsed = _parse_dt(s)
    if parsed:
        return parsed.isoformat()

    low = s.lower()
    now = _utcnow()

    # 2) "New" / "just posted" / "today" / "active today" → now
    if any(tok in low for tok in ("new", "just posted", "just now", "today", "hours ago", "hour ago", "minutes ago", "minute ago")):
        # "N hours ago" handled below too; treat sub-day as ~now
        m = _REL_AGO_RE.search(low)
        if m and m.group(2).lower() in ("hour", "hr"):
            return (now - timedelta(hours=int(m.group(1)))).isoformat()
        return now.isoformat()

    if "yesterday" in low:
        return (now - timedelta(days=1)).isoformat()

    # 3) "N days/weeks/months/years ago" (also "30+ days ago")
    m = _REL_AGO_RE.search(low)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if unit in ("day",):
            delta = timedelta(days=n)
        elif unit in ("week",):
            delta = timedelta(weeks=n)
        elif unit in ("month",):
            delta = timedelta(days=30 * n)
        elif unit in ("year",):
            delta = timedelta(days=365 * n)
        elif unit in ("hour", "hr"):
            delta = timedelta(hours=n)
        else:
            delta = timedelta(0)
        return (now - delta).isoformat()

    # 4) Bare date like "2026-06-24" already handled by _parse_dt above; give up.
    return None


class JobResearchService:
    def __init__(self) -> None:
        self.sb = get_supabase_client().client

    # ────────────────────────────────────────────────────────────────────
    # CRUD
    # ────────────────────────────────────────────────────────────────────

    async def create(
        self,
        *,
        owner_user_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        label: str,
        keywords: List[str],
        excluded_keywords: Optional[List[str]] = None,
        location: Optional[str] = None,
        country_code: Optional[str] = None,
        remote_only: bool = False,
        seniority: Optional[str] = None,
        employment_type: Optional[List[str]] = None,
        salary_min: Optional[int] = None,
        salary_currency: Optional[str] = None,
        excluded_companies: Optional[List[str]] = None,
        preferred_companies: Optional[List[str]] = None,
        sources_enabled: Optional[Dict[str, bool]] = None,
        careers_page_urls: Optional[List[str]] = None,
        rss_feed_urls: Optional[List[str]] = None,
        digest_hour_utc: int = 7,
        digest_day_of_week: Optional[int] = None,
        alert_channels: Optional[List[str]] = None,
        alert_webhook_url: Optional[str] = None,
        refresh_interval_hours: int = 24,
        source_conversation_id: Optional[str] = None,
        run_first_refresh: bool = True,
        # External (api_key) flow: the tracked_job itself is api_key-owned
        # (user_id NULL), but we still attribute a background_agents row to the
        # api_key's OWNER so the search surfaces in /admin/background-agents and
        # its JobResearchSavedJobsPanel for that user. Not written onto tracked_jobs.
        api_key_owner_user_id: Optional[str] = None,
        # Legacy/no-op (retained for backwards compat with the old route shape)
        auto_expand_keywords: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Create a tracked_job. v0.2 changes vs v0.1.0:
          - Synchronously calls Haiku to expand keywords (was deferred behind an opt-in flag)
          - Creates the background_agents bookkeeping row
          - Optionally runs the first refresh inline so the agent's reply contains real findings
          - Accepts source_conversation_id so daily digests can chat-post into the conversation
        """
        if not keywords:
            raise ValueError("At least one keyword is required")
        if (owner_user_id is None) == (api_key_id is None):
            raise ValueError("Provide exactly one of owner_user_id or api_key_id")

        clean_keywords = [k.strip() for k in keywords if k and k.strip()]
        clean_excluded = [k.strip() for k in (excluded_keywords or []) if k and k.strip()]

        # Auto-enable rss_feeds / careers_pages source toggles when the operator
        # has curated global defaults — otherwise the global lists are dead
        # weight (the engine reads them but the toggle gates the call).
        # User-supplied sources_enabled still wins outright.
        from app.services.integrations.job_search_service import load_site_defaults_from_db
        rss_defaults_present = bool(load_site_defaults_from_db("rss_feed_default"))
        careers_defaults_present = bool(load_site_defaults_from_db("careers_page_default"))
        effective_sources = sources_enabled or {
            "google_jobs": True,
            "google_serp": True,
            "perplexity": True,
            "careers_pages": careers_defaults_present,
            "rss_feeds": rss_defaults_present,
        }

        row = {
            "user_id": owner_user_id,
            "api_key_id": api_key_id,
            "workspace_id": workspace_id,
            "label": label.strip()[:200],
            "keywords": clean_keywords,
            "excluded_keywords": clean_excluded,
            "location": (location or "").strip() or None,
            "country_code": (country_code or "").strip().upper() or None,
            "remote_only": bool(remote_only),
            "seniority": seniority,
            "employment_type": employment_type or [],
            "salary_min": salary_min,
            "salary_currency": salary_currency or "USD",
            "excluded_companies": excluded_companies or [],
            "preferred_companies": preferred_companies or [],
            "sources_enabled": effective_sources,
            "careers_page_urls": careers_page_urls or [],
            "rss_feed_urls": rss_feed_urls or [],
            "digest_hour_utc": max(0, min(23, int(digest_hour_utc))),
            "digest_day_of_week": digest_day_of_week,
            "alert_channels": alert_channels or ["bell", "email"],
            "alert_webhook_url": alert_webhook_url,
            # Daily floor (2026-06-27): never schedule a sub-daily refresh. A
            # job search runs at most once per day; the cadence RPC also floors
            # this at runtime as a backstop.
            "refresh_interval_hours": max(24, int(refresh_interval_hours)),
            "source_conversation_id": source_conversation_id,
            "next_check_at": _iso(_utcnow()),  # eligible immediately
        }
        res = self.sb.table("tracked_jobs").insert(row).execute()
        if not res.data:
            raise RuntimeError("Failed to create tracked_job")
        created = res.data[0]
        tracked_job_id = created["id"]

        # ── v0.2: keyword expansion (Haiku tool-use, default-on) ───────────
        attribution = costs.CostAttribution(
            user_id=owner_user_id, workspace_id=workspace_id,
            tracked_job_id=tracked_job_id, api_key_id=api_key_id,
        )
        try:
            expansion = await expand_keywords(
                label=label, keywords=clean_keywords, location=location,
                seniority=seniority, excluded_keywords=clean_excluded,
                attribution=attribution,
            )
            expanded = expansion.get("expanded") or []
            # v0.4: also persist Haiku-generated query_phrasings (used by refresh fan-out)
            query_phrasings = expansion.get("query_phrasings") or []
            self.sb.table("tracked_jobs").update({
                "expanded_keywords": expanded,
                "query_phrasings": query_phrasings,
                "last_keywords_expanded_at": _iso(_utcnow()),
            }).eq("id", tracked_job_id).execute()
            created["expanded_keywords"] = expanded
            created["query_phrasings"] = query_phrasings
            created["_keyword_expansion"] = expansion
        except Exception as e:
            logger.warning(f"job-research create: keyword expansion failed (non-fatal): {e}")
            created["expanded_keywords"] = []

        # ── v0.2: background_agents bookkeeping row ────────────────────────
        # Internal flow → owned by owner_user_id. External (api_key) flow →
        # owned by the api_key's owner (api_key_owner_user_id) so it still shows
        # up in that user's /admin/background-agents with the saved-jobs panel.
        bg_owner = owner_user_id or api_key_owner_user_id
        if bg_owner:
            try:
                bg_id = bookkeeping.create_background_agent_for_tracked_job(
                    tracked_job_id=tracked_job_id,
                    user_id=bg_owner,
                    workspace_id=workspace_id,
                    label=label,
                    keywords=clean_keywords,
                    refresh_interval_hours=row["refresh_interval_hours"],
                )
                if bg_id:
                    created["background_agent_id"] = bg_id
            except Exception as e:
                logger.warning(f"job-research create: background_agents row failed (non-fatal): {e}")

        # ── v0.2: synchronous first refresh so the agent's tool reply has real data
        if run_first_refresh:
            try:
                first_outcome = await self.refresh(tracked_job_id, force_full_discovery=True)
                created["first_refresh"] = first_outcome
                # Re-fetch the row to surface the updated denormalized counts
                fresh = self.get(tracked_job_id, owner_user_id=owner_user_id, api_key_id=api_key_id)
                if fresh:
                    created.update(fresh)
            except Exception as e:
                logger.warning(f"job-research create: first refresh failed (non-fatal): {e}")
                created["first_refresh"] = {"error": str(e)[:200]}

        return created

    async def regenerate_keywords(
        self, tracked_job_id: str, *, owner_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Re-run Haiku keyword expansion on demand. Returns the new expanded list + the rejected suggestions."""
        tj = self.get(tracked_job_id, owner_user_id=owner_user_id)
        if not tj:
            raise RuntimeError("tracked_job not found")
        attribution = costs.CostAttribution(
            user_id=tj.get("user_id"), workspace_id=tj.get("workspace_id"),
            tracked_job_id=tracked_job_id, api_key_id=tj.get("api_key_id"),
        )
        expansion = await expand_keywords(
            label=tj.get("label") or "",
            keywords=list(tj.get("keywords") or []),
            location=tj.get("location"),
            seniority=tj.get("seniority"),
            excluded_keywords=list(tj.get("excluded_keywords") or []),
            attribution=attribution,
        )
        expanded = expansion.get("expanded") or []
        query_phrasings = expansion.get("query_phrasings") or []
        self.sb.table("tracked_jobs").update({
            "expanded_keywords": expanded,
            "query_phrasings": query_phrasings,
            "last_keywords_expanded_at": _iso(_utcnow()),
        }).eq("id", tracked_job_id).execute()
        return {
            "expanded": expanded,
            "query_phrasings": query_phrasings,
            "rejected": expansion.get("rejected") or [],
            "raw": expansion.get("raw") or {},
        }

    def get(self, tracked_job_id: str, *, owner_user_id: Optional[str] = None,
            api_key_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        q = self.sb.table("tracked_jobs").select("*").eq("id", tracked_job_id)
        if owner_user_id:
            q = q.eq("user_id", owner_user_id)
        if api_key_id:
            q = q.eq("api_key_id", api_key_id)
        res = q.maybe_single().execute()
        return (res.data if res else None) or None

    def get_readable(self, tracked_job_id: str, reader_user_id: str) -> Optional[Dict[str, Any]]:
        """Read-authorization for the internal (session-JWT) UI. A user may READ a
        tracked_job if they own it directly (internal flow) OR if it is owned by an
        api_key that THEY own (external flow surfaced back into the premade pages).

        Used by the read endpoints (get / listings / summary / exclusions /
        correct-match owner-check). Write endpoints (refresh / update / delete)
        intentionally stay on the strict `get(owner_user_id=...)` path — api_key
        jobs are mutated through the external /api/v1/jobs/track API, not the UI.
        """
        row = self.get(tracked_job_id)
        if not row:
            return None
        if reader_user_id and row.get("user_id") == reader_user_id:
            return row
        api_key_id = row.get("api_key_id")
        if api_key_id and reader_user_id:
            try:
                ak = (
                    self.sb.table("api_keys").select("user_id")
                    .eq("id", api_key_id).maybe_single().execute()
                )
                if ak and ak.data and ak.data.get("user_id") == reader_user_id:
                    return row
            except Exception as e:
                logger.debug(f"get_readable api_key owner check failed: {e}")
        return None

    def list_for_user(self, user_id: str, *, only_active: bool = True) -> List[Dict[str, Any]]:
        q = self.sb.table("tracked_jobs").select("*").eq("user_id", user_id).order("created_at", desc=True)
        if only_active:
            q = q.eq("is_active", True)
        return (q.execute().data or [])

    def list_for_api_key(self, api_key_id: str, *, only_active: bool = True) -> List[Dict[str, Any]]:
        q = self.sb.table("tracked_jobs").select("*").eq("api_key_id", api_key_id).order("created_at", desc=True)
        if only_active:
            q = q.eq("is_active", True)
        return (q.execute().data or [])

    def update(self, tracked_job_id: str, owner_user_id: Optional[str], patch: Dict[str, Any]) -> Dict[str, Any]:
        ALLOWED = {
            "label", "keywords", "excluded_keywords", "location", "country_code",
            "remote_only", "seniority", "employment_type", "salary_min", "salary_currency",
            "excluded_companies", "preferred_companies", "sources_enabled", "careers_page_urls", "rss_feed_urls",
            "digest_enabled", "digest_hour_utc", "alert_channels", "alert_webhook_url",
            "refresh_interval_hours", "auto_expand_keywords", "is_active",
        }
        clean = {k: v for k, v in patch.items() if k in ALLOWED and v is not None}
        # Daily floor (2026-06-27): sub-daily refresh is not allowed.
        if "refresh_interval_hours" in clean:
            try:
                clean["refresh_interval_hours"] = max(24, int(clean["refresh_interval_hours"]))
            except (TypeError, ValueError):
                clean.pop("refresh_interval_hours", None)
        if not clean:
            return self.get(tracked_job_id, owner_user_id=owner_user_id) or {}
        q = self.sb.table("tracked_jobs").update(clean).eq("id", tracked_job_id)
        if owner_user_id:
            q = q.eq("user_id", owner_user_id)
        res = q.execute()
        if not res.data:
            raise RuntimeError("Update failed (row not found or no permission)")
        return res.data[0]

    def deactivate(self, tracked_job_id: str, owner_user_id: Optional[str]) -> bool:
        # Read background_agent_id first so we can mirror the deactivation
        existing = self.get(tracked_job_id, owner_user_id=owner_user_id)
        if not existing:
            return False
        q = self.sb.table("tracked_jobs").update({"is_active": False}).eq("id", tracked_job_id)
        if owner_user_id:
            q = q.eq("user_id", owner_user_id)
        ok = bool(q.execute().data)
        if ok:
            bookkeeping.deactivate_background_agent(existing.get("background_agent_id"))
        return ok

    # ────────────────────────────────────────────────────────────────────
    # Listings (history) read paths
    # ────────────────────────────────────────────────────────────────────

    def list_listings(
        self,
        tracked_job_id: str,
        *,
        relevance: Optional[str] = "match",
        days: Optional[int] = 30,
        only_actionable: bool = False,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        q = (
            self.sb.table("job_listings").select("*")
            .eq("tracked_job_id", tracked_job_id)
            .order("discovered_at", desc=True)
            .limit(min(limit, 500))
        )
        if relevance and relevance != "all":
            q = q.eq("relevance", relevance)
        if days:
            since = _iso(_utcnow() - timedelta(days=days))
            q = q.gte("discovered_at", since)
        if only_actionable:
            q = q.is_("user_action", "null")
        return (q.execute().data or [])

    def mark_listing(
        self,
        listing_id: str,
        *,
        action: str,
        user_id: str,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        if action not in ("saved", "applied", "dismissed", "interested"):
            raise ValueError("Invalid action")
        # RLS enforces ownership via the parent tracked_job
        res = (
            self.sb.table("job_listings")
            .update({
                "user_action": action,
                "user_action_at": _iso(_utcnow()),
                "user_notes": (notes or None),
            })
            .eq("id", listing_id)
            .execute()
        )
        if not res.data:
            raise RuntimeError("Listing not found or not owned by user")
        return res.data[0]

    def summary(self, tracked_job_id: str, *, days: int = 30) -> Dict[str, Any]:
        since_iso = _iso(_utcnow() - timedelta(days=days))
        rows = (
            self.sb.table("job_listings")
            .select("relevance, source, company, salary_min, posted_at, discovered_at, user_action")
            .eq("tracked_job_id", tracked_job_id)
            .gte("discovered_at", since_iso)
            .execute()
            .data
            or []
        )
        match_count = sum(1 for r in rows if r.get("relevance") == "match")
        applied_count = sum(1 for r in rows if r.get("user_action") == "applied")
        saved_count = sum(1 for r in rows if r.get("user_action") == "saved")
        sources: Dict[str, int] = {}
        companies: Dict[str, int] = {}
        for r in rows:
            sources[r.get("source") or "?"] = sources.get(r.get("source") or "?", 0) + 1
            co = r.get("company") or ""
            if co:
                companies[co] = companies.get(co, 0) + 1
        top_companies = sorted(companies.items(), key=lambda x: -x[1])[:10]
        return {
            "days": days,
            "total": len(rows),
            "matches": match_count,
            "applied": applied_count,
            "saved": saved_count,
            "by_source": sources,
            "top_companies": [{"company": c, "count": n} for c, n in top_companies],
        }

    # ────────────────────────────────────────────────────────────────────
    # Refresh — the heart of the module
    # ────────────────────────────────────────────────────────────────────

    async def refresh(
        self,
        tracked_job_id: str,
        *,
        force: bool = False,
        force_full_discovery: bool = False,
    ) -> Dict[str, Any]:
        tj = self.get(tracked_job_id)
        if not tj:
            raise RuntimeError(f"tracked_job {tracked_job_id} not found")
        if not tj.get("is_active") and not force:
            return {"skipped": True, "reason": "inactive"}

        run_id = str(uuid.uuid4())
        attribution = costs.CostAttribution(
            user_id=tj.get("user_id"),
            workspace_id=tj.get("workspace_id"),
            tracked_job_id=tracked_job_id,
            refresh_run_id=run_id,
            api_key_id=tj.get("api_key_id"),
        )

        # v0.2: write an agent_runs row so /admin/background-agents shows this run
        agent_run_id = bookkeeping.start_run(
            background_agent_id=tj.get("background_agent_id"),
            workspace_id=tj.get("workspace_id"),
            user_id=tj.get("user_id"),
            refresh_run_id=run_id,
            triggered_by="schedule" if not force else "manual",
        )

        started_at = _utcnow()

        sources_enabled = tj.get("sources_enabled") or {}
        # v0.2: discovery uses keywords ∪ expanded_keywords
        keywords = list(tj.get("keywords") or [])
        expanded = list(tj.get("expanded_keywords") or [])
        # Preserve order; dedupe case-insensitively
        seen: set = set()
        all_search_terms: List[str] = []
        for k in keywords + expanded:
            kl = (k or "").strip().lower()
            if kl and kl not in seen:
                seen.add(kl)
                all_search_terms.append(k.strip())

        excluded_keywords = list(tj.get("excluded_keywords") or [])
        location = tj.get("location")
        country_code = tj.get("country_code")
        remote_only = bool(tj.get("remote_only"))
        seniority = tj.get("seniority")
        excluded_companies = list(tj.get("excluded_companies") or [])
        preferred_companies = list(tj.get("preferred_companies") or [])

        # v0.4: query-shape variations — DB-persisted (from Haiku) ∪ default templates.
        primary_keyword = keywords[0] if keywords else (all_search_terms[0] if all_search_terms else "")
        haiku_phrasings = list(tj.get("query_phrasings") or [])
        default_phrasings = build_query_variations(primary_keyword, location, remote_only)
        # Dedupe (case-insensitive) — Haiku takes priority since the user paid for it.
        seen_q: set = set()
        all_query_variations: List[str] = []
        for q in haiku_phrasings + default_phrasings:
            ql = (q or "").strip().lower()
            if ql and ql not in seen_q:
                seen_q.add(ql)
                all_query_variations.append(q.strip())

        bookkeeping.append_log(
            run_id=agent_run_id, level="info",
            message=f"Refresh started — {len(all_search_terms)} search term(s), {len(all_query_variations)} query variations",
            data={"keywords": keywords, "expanded_count": len(expanded), "query_variations_count": len(all_query_variations), "force": force},
        )

        try:
            # No-board fallback (2026-06-27): for a location-scoped search, ask
            # Haiku once where these roles are posted locally and feed the
            # discovered domains into Perplexity's filter for this run. Stop-gap
            # until an operator curates real boards in job_research_sites.
            discovered_domains: List[str] = []
            if country_code or location:
                try:
                    discovered_domains = await discover_local_job_boards(
                        location=location, country_code=country_code,
                        keywords=all_search_terms, attribution=attribution,
                    )
                    if discovered_domains:
                        bookkeeping.append_log(
                            run_id=agent_run_id, level="info",
                            message=f"No-board fallback: discovered {len(discovered_domains)} local board(s): {', '.join(discovered_domains[:8])}",
                        )
                except Exception as e:
                    logger.warning(f"job-refresh: board discovery failed (non-fatal): {e}")

            tasks = []
            sources_called: List[str] = []
            if sources_enabled.get("google_jobs", True):
                sources_called.append("google_jobs")
                tasks.append(search_via_dataforseo_jobs(
                    keywords=all_search_terms,
                    location=location,
                    country_code=country_code,
                    remote_only=remote_only,
                    employment_type=tj.get("employment_type") or None,
                    attribution=attribution,
                    limit=30,
                ))
            # v0.4: google_serp — general Google web SERP, fanned out across query variations
            if sources_enabled.get("google_serp", True) and all_query_variations:
                sources_called.append("google_serp")
                tasks.append(search_via_dataforseo_serp(
                    queries=all_query_variations[:5],
                    country_code=country_code,
                    attribution=attribution,
                    limit_per_query=10,
                ))
            if sources_enabled.get("perplexity", True):
                # v0.4.5: fan out Perplexity across query phrasings instead of one call.
                # Primary call uses sonar-pro on the original keyword set (best for
                # the user's intent); up to 3 variation calls use cheap sonar with one
                # Haiku-generated phrasing each. anti-hallucination guard catches any
                # fakes; content_hash dedupes overlap.
                model_primary = "sonar-pro" if (force_full_discovery or not tj.get("last_refreshed_at")) else "sonar"
                sources_called.append("perplexity_primary")
                tasks.append(search_via_perplexity(
                    keywords=all_search_terms,
                    location=location,
                    remote_only=remote_only,
                    seniority=seniority,
                    excluded_keywords=excluded_keywords,
                    excluded_companies=excluded_companies,
                    attribution=attribution,
                    model=model_primary,
                    extra_domains=discovered_domains or None,
                    limit=7,
                ))
                for i, phrasing in enumerate(all_query_variations[:3] if all_query_variations else []):
                    sources_called.append(f"perplexity_var_{i+1}")
                    tasks.append(search_via_perplexity(
                        keywords=[phrasing],  # single-element → keyword_str becomes the phrasing
                        location=location,
                        remote_only=remote_only,
                        seniority=seniority,
                        excluded_keywords=excluded_keywords,
                        excluded_companies=excluded_companies,
                        attribution=attribution,
                        model="sonar",
                        extra_domains=discovered_domains or None,
                        limit=5,
                    ))
            if sources_enabled.get("careers_pages", False):
                # UNION per-tracked URLs with operator-curated global defaults from
                # job_research_sites (site_type='careers_page_default'). De-dup,
                # preserve per-tracked first so user overrides take precedence on tie.
                from app.services.integrations.job_search_service import load_site_defaults_from_db
                per_tracked = [u for u in (tj.get("careers_page_urls") or []) if u]
                globals_ = load_site_defaults_from_db("careers_page_default")
                merged_careers: List[str] = []
                seen = set()
                for u in per_tracked + globals_:
                    key = u.strip().lower()
                    if key and key not in seen:
                        seen.add(key)
                        merged_careers.append(u.strip())
                if merged_careers:
                    sources_called.append("careers_pages")
                    tasks.append(search_via_firecrawl_careers(
                        careers_urls=merged_careers,
                        company_hint=None,
                        attribution=attribution,
                    ))
            if sources_enabled.get("rss_feeds", False):
                from app.services.integrations.job_search_service import load_site_defaults_from_db
                per_tracked = [u for u in (tj.get("rss_feed_urls") or []) if u]
                globals_ = load_site_defaults_from_db("rss_feed_default")
                merged_rss: List[str] = []
                seen = set()
                for u in per_tracked + globals_:
                    key = u.strip().lower()
                    if key and key not in seen:
                        seen.add(key)
                        merged_rss.append(u.strip())
                if merged_rss:
                    sources_called.append("rss_feeds")
                    tasks.append(search_via_rss_feeds(
                        feed_urls=merged_rss,
                        attribution=attribution,
                    ))

            if not tasks:
                bookkeeping.append_log(run_id=agent_run_id, level="warning", message="No sources enabled — skipping")
                bookkeeping.complete_run(run_id=agent_run_id, output_data={"skipped": True, "reason": "no sources"},
                                         duration_ms=int((_utcnow() - started_at).total_seconds() * 1000))
                return {"skipped": True, "reason": "no sources enabled", "refresh_run_id": run_id}

            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            hits: List[JobHit] = []
            per_source_counts: Dict[str, int] = {}
            for source_name, r in zip(sources_called, all_results):
                if isinstance(r, Exception):
                    bookkeeping.append_log(
                        run_id=agent_run_id, level="warning",
                        message=f"Source {source_name} failed: {str(r)[:120]}",
                    )
                    per_source_counts[source_name] = -1
                    continue
                per_source_counts[source_name] = len(r)
                hits.extend(r)
                bookkeeping.append_log(
                    run_id=agent_run_id, level="info",
                    message=f"{source_name}: {len(r)} hits",
                )

            if not hits:
                self._update_after_refresh(tracked_job_id, run_id, persisted=0, new_matches=0)
                outcome = {"refresh_run_id": run_id, "discovered": 0, "persisted": 0, "matches": 0, "by_source": per_source_counts}
                bookkeeping.complete_run(run_id=agent_run_id, output_data=outcome,
                                         duration_ms=int((_utcnow() - started_at).total_seconds() * 1000))
                return outcome

            deduped = dedupe_hits(hits)
            # v0.4.6: URL-only dedup within the run. The content_hash dedup above
            # works on (canonical_url + title + company), so a single job returned
            # by both the primary Perplexity call and a variation call with
            # slightly different title/company can slip through (e.g. the Close
            # WeWorkRemotely job appearing twice with title "Senior Software
            # Engineer – Backend / Python" and "...USA Only (100% Remote)").
            # Force canonical_url uniqueness within a run.
            _seen_urls: set = set()
            url_unique: List[JobHit] = []
            for h in deduped:
                key = (h.canonical_url or h.url or "").lower()
                if key and key in _seen_urls:
                    continue
                if key:
                    _seen_urls.add(key)
                url_unique.append(h)
            deduped = url_unique

            # 2026-06-27: near-duplicate collapse. A job board sometimes exposes the
            # SAME role under two different URLs (different slugs/ids) — e.g. the two
            # Hays "SC Cleared Product Owner" listings. content_hash + canonical_url
            # both keyed on the URL, so both survived. Collapse on the role identity
            # itself: normalized (title + company), keeping the first occurrence
            # (deduped is already source-priority-ordered). Only fires when both
            # title and company are present so distinct-but-untitled rows aren't merged.
            def _norm_key(s: str) -> str:
                return re.sub(r"[\s\-_/|]+", " ", (s or "").strip().lower()).strip()
            _seen_role: set = set()
            role_unique: List[JobHit] = []
            for h in deduped:
                t = _norm_key(h.title or "")
                co = _norm_key(h.company or "")
                if t and co:
                    rkey = f"{t}|{co}"
                    if rkey in _seen_role:
                        continue
                    _seen_role.add(rkey)
                role_unique.append(h)
            deduped = role_unique

            excl = self._load_exclusions(tracked_job_id)
            deduped = [h for h in deduped if not _is_excluded(h, excl, excluded_companies)]
            existing_hashes = self._existing_content_hashes(tracked_job_id, [h.content_hash for h in deduped])
            candidates = [h for h in deduped if h.content_hash not in existing_hashes]
            # v0.4.6: also reject if the canonical_url already exists in DB for
            # this tracked_job (regardless of content_hash). Belt-and-suspenders.
            _existing_urls = self._existing_canonical_urls(tracked_job_id, [h.canonical_url for h in candidates])
            candidates = [h for h in candidates if (h.canonical_url or "").lower() not in _existing_urls]

            bookkeeping.append_log(
                run_id=agent_run_id, level="info",
                message=f"Cross-source dedupe → {len(deduped)}; new (not yet seen) → {len(candidates)}",
            )

            if not candidates:
                self._update_after_refresh(tracked_job_id, run_id, persisted=0, new_matches=0)
                outcome = {"refresh_run_id": run_id, "discovered": len(hits), "deduped": len(deduped),
                           "persisted": 0, "matches": 0, "by_source": per_source_counts}
                bookkeeping.complete_run(run_id=agent_run_id, output_data=outcome,
                                         duration_ms=int((_utcnow() - started_at).total_seconds() * 1000))
                return outcome

            facets = JobFacets(
                keywords=all_search_terms,
                excluded_keywords=excluded_keywords,
                location=location,
                remote_only=remote_only,
                seniority=seniority,
                excluded_companies=excluded_companies,
                preferred_companies=preferred_companies,
            )
            verdicts = await classify_batch(facets, candidates, attribution=attribution)

            rows_to_insert = []
            new_match_count = 0
            for hit, v in zip(candidates, verdicts):
                relevance = (v or {}).get("relevance") or "unverifiable"
                if relevance == "mismatch":
                    continue
                if relevance == "match":
                    new_match_count += 1
                row = {
                    "tracked_job_id": tracked_job_id,
                    "refresh_run_id": run_id,
                    "url": hit.url,
                    "canonical_url": hit.canonical_url,
                    "content_hash": hit.content_hash,
                    "title": hit.title,
                    "company": hit.company,
                    "company_domain": hit.company_domain,
                    "location": hit.location,
                    "is_remote": hit.is_remote,
                    "salary_min": hit.salary_min,
                    "salary_max": hit.salary_max,
                    "salary_currency": hit.salary_currency,
                    "salary_period": hit.salary_period,
                    "employment_type": hit.employment_type,
                    "seniority": hit.seniority,
                    "description_excerpt": hit.description_excerpt,
                    # Bug-fix 2026-06-26: normalize relative dates ("2 days ago",
                    # "New", "1 month ago") → ISO. A raw relative string used to
                    # 400 the insert (22007) and silently drop the listing — which
                    # cost us ~80% of careers/job-board matches every refresh.
                    "posted_at": normalize_posted_at(hit.posted_at),
                    "source": hit.source,
                    "relevance": relevance,
                    "relevance_score": (v or {}).get("relevance_score"),
                    "match_note": (v or {}).get("match_note"),
                    "classifier_cached": bool((v or {}).get("classifier_cached")),
                    "raw_payload": hit.raw_payload or None,
                }
                # v0.3: salary normalization — populate annual USD fields if a salary is present
                normalize_listing_in_place(row)
                rows_to_insert.append(row)

            persisted = 0
            if rows_to_insert:
                # v0.4.2: switched from upsert(ignore_duplicates=True) to plain insert().
                # supabase-py 2.3.0 with ignore_duplicates returns empty .data AND
                # silently doesn't insert (the Prefer header interaction is broken).
                # We've already pre-filtered against _existing_content_hashes(), so
                # the (tracked_job_id, content_hash) UNIQUE constraint is a safety
                # net. Insert one-at-a-time so a single dup doesn't kill the batch.
                for row in rows_to_insert:
                    try:
                        ins = self.sb.table("job_listings").insert(row).execute()
                        if ins.data:
                            persisted += len(ins.data)
                    except Exception as e:
                        # Unique-violation = a parallel/recent refresh already grabbed
                        # this content_hash. That's fine, skip silently.
                        emsg = str(e).lower()
                        if "duplicate" in emsg or "unique" in emsg or "23505" in emsg:
                            continue
                        logger.warning(f"job-refresh insert (one row): {str(e)[:200]}")
                        bookkeeping.append_log(run_id=agent_run_id, level="warning", message=f"Persist row failed: {str(e)[:120]}")

            bookkeeping.append_log(
                run_id=agent_run_id, level="info",
                message=f"Persisted {persisted} listings ({new_match_count} match, {len(rows_to_insert) - new_match_count} other)",
            )

            self._update_after_refresh(tracked_job_id, run_id, persisted=persisted, new_matches=new_match_count)
            costs.stamp_refresh_cost(tracked_job_id=tracked_job_id, refresh_run_id=run_id)

            # v0.3: real-time burst alert (between daily digest ticks). Fires only
            # when the tracked_job opted in via alert_on_burst=true AND new_matches
            # exceeds burst_threshold AND not within a 2h cooldown.
            burst_outcome: Dict[str, Any] = {}
            try:
                from app.modules.job_research_notifications.service import get_job_digest_dispatcher
                dispatcher = get_job_digest_dispatcher()
                burst_outcome = await dispatcher.dispatch_burst_if_warranted(
                    tracked_job_id=tracked_job_id, new_match_count=new_match_count,
                )
            except Exception as e:
                logger.warning(f"job-refresh: burst-alert dispatch failed (non-fatal): {e}")
                burst_outcome = {"error": str(e)[:200]}

            outcome = {
                "refresh_run_id": run_id,
                "discovered": len(hits),
                "deduped": len(deduped),
                "candidates_after_exclusions": len(candidates),
                "persisted": persisted,
                "matches": new_match_count,
                "by_source": per_source_counts,
                "burst_alert": burst_outcome,
            }
            bookkeeping.complete_run(
                run_id=agent_run_id, output_data=outcome,
                duration_ms=int((_utcnow() - started_at).total_seconds() * 1000),
            )
            return outcome
        except Exception as e:
            logger.exception(f"job-refresh failed for {tracked_job_id}: {e}")
            bookkeeping.fail_run(
                run_id=agent_run_id, error_message=str(e),
                duration_ms=int((_utcnow() - started_at).total_seconds() * 1000),
            )
            raise

    def _update_after_refresh(self, tracked_job_id: str, run_id: str, *, persisted: int, new_matches: int) -> None:
        # Update denormalized counters
        try:
            since_24h = _iso(_utcnow() - timedelta(hours=24))
            since_7d = _iso(_utcnow() - timedelta(days=7))
            count_24h = (
                self.sb.table("job_listings").select("id", count="exact", head=True)
                .eq("tracked_job_id", tracked_job_id).gte("discovered_at", since_24h)
                .execute().count or 0
            )
            count_7d = (
                self.sb.table("job_listings").select("id", count="exact", head=True)
                .eq("tracked_job_id", tracked_job_id).gte("discovered_at", since_7d)
                .execute().count or 0
            )
            self.sb.table("tracked_jobs").update({
                "current_listing_count_24h": int(count_24h),
                "current_listing_count_7d": int(count_7d),
                "current_snapshot_at": _iso(_utcnow()),
                "last_refreshed_at": _iso(_utcnow()),
            }).eq("id", tracked_job_id).execute()
        except Exception as e:
            logger.warning(f"job-refresh denorm update: {e}")

        # Adaptive cadence
        try:
            self.sb.rpc("update_tracked_job_cadence", {
                "p_id": tracked_job_id,
                "p_new_match_count": int(new_matches),
                "p_refresh_run_id": run_id,
            }).execute()
        except Exception as e:
            logger.warning(f"job-refresh cadence: {e}")

    def _load_exclusions(self, tracked_job_id: str) -> Dict[str, set]:
        try:
            rows = (
                self.sb.table("job_excluded_urls")
                .select("url, domain, company")
                .eq("tracked_job_id", tracked_job_id)
                .execute()
                .data
                or []
            )
            return {
                "urls": {(r.get("url") or "").strip().lower() for r in rows if r.get("url")},
                "domains": {(r.get("domain") or "").strip().lower() for r in rows if r.get("domain")},
                "companies": {(r.get("company") or "").strip().lower() for r in rows if r.get("company")},
            }
        except Exception:
            return {"urls": set(), "domains": set(), "companies": set()}

    def _existing_content_hashes(self, tracked_job_id: str, hashes: List[str]) -> set:
        if not hashes:
            return set()
        try:
            rows = (
                self.sb.table("job_listings")
                .select("content_hash")
                .eq("tracked_job_id", tracked_job_id)
                .in_("content_hash", hashes)
                .execute()
                .data
                or []
            )
            return {r["content_hash"] for r in rows}
        except Exception:
            return set()

    def _existing_canonical_urls(self, tracked_job_id: str, urls: List[str]) -> set:
        """v0.4.6: returns the set of canonical_url values already persisted for
        this tracked_job, lowercased for case-insensitive matching. Used as a
        secondary dedup pass beyond content_hash so a job that surfaces with
        slightly different title/company across runs doesn't re-persist."""
        urls = [u for u in (urls or []) if u]
        if not urls:
            return set()
        try:
            rows = (
                self.sb.table("job_listings")
                .select("canonical_url")
                .eq("tracked_job_id", tracked_job_id)
                .in_("canonical_url", urls)
                .execute()
                .data
                or []
            )
            return {(r.get("canonical_url") or "").lower() for r in rows if r.get("canonical_url")}
        except Exception:
            return set()

    # ────────────────────────────────────────────────────────────────────
    # Exclusion helpers (UI-driven)
    # ────────────────────────────────────────────────────────────────────

    def add_exclusion(
        self, tracked_job_id: str, *, url: Optional[str] = None,
        domain: Optional[str] = None, company: Optional[str] = None, reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not (url or domain or company):
            raise ValueError("Provide at least one of url, domain, company")
        res = self.sb.table("job_excluded_urls").insert({
            "tracked_job_id": tracked_job_id,
            "url": url, "domain": domain, "company": company, "reason": reason,
        }).execute()
        return (res.data or [{}])[0]

    def list_exclusions(self, tracked_job_id: str) -> List[Dict[str, Any]]:
        return (
            self.sb.table("job_excluded_urls").select("*")
            .eq("tracked_job_id", tracked_job_id)
            .order("created_at", desc=True)
            .execute().data or []
        )

    def remove_exclusion(self, exclusion_id: str) -> bool:
        return bool(
            self.sb.table("job_excluded_urls").delete().eq("id", exclusion_id).execute().data
        )


def _is_excluded(hit: JobHit, excl: Dict[str, set], excluded_companies: List[str]) -> bool:
    url_low = (hit.canonical_url or hit.url or "").lower()
    if any(u and u in url_low for u in excl.get("urls", set())):
        return True
    domain_low = (hit.company_domain or "").lower()
    if domain_low and domain_low in excl.get("domains", set()):
        return True
    co_low = (hit.company or "").lower()
    if co_low and (co_low in excl.get("companies", set())
                   or any(co_low == ex.lower() or ex.lower() in co_low for ex in (excluded_companies or []))):
        return True
    return False


# Module-level singleton (mirrors tracked_mentions_service)
@lru_cache(maxsize=1)
def get_job_research_service() -> JobResearchService:
    return JobResearchService()
