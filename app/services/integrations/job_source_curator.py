"""
Job-source self-curation (2026-06-30).

Runs at the END of every refresh. Closes the discovery loop:
  • tracks which board domains actually produced VERIFIED matches this run,
  • AUTO-LEARNS new boards that Perplexity/SERP surfaced — promoting a domain
    into the curated `perplexity_domain` list once it has yielded verified
    matches for >=3 distinct employers across >=2 separate refreshes (proof it's
    a real multi-employer board, not a one-off company page),
  • bumps yield stats on curated sources that delivered,
  • SELF-CLEANS: auto-disables domains WE auto-added that then go stale (>30d
    with no yield). Operator-seeded rows are only FLAGGED, never auto-removed.
  • writes a one-row audit per run to `job_source_review`.

Best-effort: never raises into the refresh path. No external API calls — pure
aggregation over data we already store.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Promotion guardrails
_MIN_EMPLOYERS = 3
_MIN_RUNS = 2
_STALE_AUTO_ADDED_DAYS = 30      # auto-disable our own additions after this with no yield
_MAX_SAMPLE_EMPLOYERS = 25

# Domains we never auto-promote as "boards": generic content/social (a verified
# match shouldn't come from these, but belt-and-braces) — NOT ATS/aggregators,
# which ARE valuable to add.
_NEVER_PROMOTE = {
    "reddit.com", "medium.com", "quora.com", "youtube.com", "substack.com",
    "twitter.com", "x.com", "facebook.com", "instagram.com", "wikipedia.org",
    "jooble.org", "google.com", "bing.com",
}

# Common 2-part public suffixes so registrable-domain collapse is correct.
_TWO_PART_TLDS = {
    "co.uk", "com.br", "co.jp", "com.au", "co.in", "com.mx", "co.za",
    "com.sg", "com.tr", "gr.com", "co.il",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _host(value: str) -> str:
    v = (value or "").strip().lower()
    if not v:
        return ""
    if "://" not in v and "/" not in v and " " not in v:
        host = v  # already a bare domain
    else:
        host = urlparse(v if "://" in v else "http://" + v).netloc or ""
    if host.startswith("www."):
        host = host[4:]
    return host.split(":")[0]


def _registrable(value: str) -> str:
    """Collapse a host/URL to its registrable domain (job-boards.greenhouse.io →
    greenhouse.io, ie.linkedin.com → linkedin.com, foo.co.uk → foo.co.uk)."""
    host = _host(value)
    if not host or "." not in host:
        return host
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _TWO_PART_TLDS:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _load_curated_domains(sb) -> Set[str]:
    try:
        rows = (
            sb.table("job_research_sites").select("url_or_domain")
            .eq("is_enabled", True).execute().data or []
        )
        return {_registrable(r.get("url_or_domain") or "") for r in rows if r.get("url_or_domain")}
    except Exception as e:
        logger.debug(f"curator: load curated domains failed: {e}")
        return set()


def _bump_curated_yield(sb, domain: str, matches: int) -> None:
    """Mark every enabled curated row whose domain == `domain` as having yielded."""
    try:
        rows = (
            sb.table("job_research_sites").select("id, url_or_domain, lifetime_verified")
            .eq("is_enabled", True).execute().data or []
        )
        for r in rows:
            if _registrable(r.get("url_or_domain") or "") == domain:
                sb.table("job_research_sites").update({
                    "last_yield_at": _now().isoformat(),
                    "lifetime_verified": int(r.get("lifetime_verified") or 0) + matches,
                }).eq("id", r["id"]).execute()
    except Exception as e:
        logger.debug(f"curator: bump curated yield failed ({domain}): {e}")


def _learn_or_promote(sb, domain: str, employers: Set[str], matches: int) -> Optional[str]:
    """Update job_board_index for a non-curated domain; promote it into
    perplexity_domain if guardrails clear. Returns the domain if promoted."""
    if domain in _NEVER_PROMOTE:
        return None
    try:
        existing = (
            sb.table("job_board_index").select("*")
            .eq("domain", domain).maybe_single().execute()
        )
        cur = (existing.data if existing else None) or None
        prior_emps = set((cur or {}).get("sample_employers") or [])
        merged = (prior_emps | {e for e in employers if e})
        merged_capped = sorted(merged)[:_MAX_SAMPLE_EMPLOYERS]
        rec = {
            "domain": domain,
            "verified_matches": int((cur or {}).get("verified_matches") or 0) + matches,
            "refresh_runs": int((cur or {}).get("refresh_runs") or 0) + 1,
            "distinct_employers": len(merged),
            "sample_employers": merged_capped,
            "status": (cur or {}).get("status") or "candidate",
            "last_seen_at": _now().isoformat(),
        }
        if not cur:
            rec["first_seen_at"] = _now().isoformat()
        sb.table("job_board_index").upsert(rec, on_conflict="domain").execute()

        if (rec["status"] == "candidate"
                and len(merged) >= _MIN_EMPLOYERS
                and rec["refresh_runs"] >= _MIN_RUNS):
            # Promote: add to the curated Perplexity domain list (idempotent).
            sb.table("job_research_sites").upsert({
                "site_type": "perplexity_domain",
                "url_or_domain": domain,
                "display_name": domain,
                "category": "auto-learned",
                "is_enabled": True,
                "auto_added": True,
                "discovered_via": "auto-yield",
                "last_yield_at": _now().isoformat(),
                "lifetime_verified": rec["verified_matches"],
                "notes": f"Auto-added: {len(merged)} employers across {rec['refresh_runs']} refreshes",
            }, on_conflict="site_type,url_or_domain").execute()
            sb.table("job_board_index").update({
                "status": "promoted", "promoted_at": _now().isoformat(),
            }).eq("domain", domain).execute()
            return domain
    except Exception as e:
        logger.debug(f"curator: learn/promote failed ({domain}): {e}")
    return None


def _auto_disable_stale_auto_added(sb) -> List[str]:
    """Disable ONLY domains we auto-added that have gone stale (no yield in
    30d). Operator-seeded rows are never auto-removed."""
    disabled: List[str] = []
    try:
        cutoff = (_now() - timedelta(days=_STALE_AUTO_ADDED_DAYS)).isoformat()
        rows = (
            sb.table("job_research_sites")
            .select("id, url_or_domain, last_yield_at, created_at")
            .eq("is_enabled", True).eq("auto_added", True).execute().data or []
        )
        for r in rows:
            ref = r.get("last_yield_at") or r.get("created_at")
            if ref and str(ref) < cutoff:
                sb.table("job_research_sites").update({
                    "is_enabled": False, "auto_disabled_at": _now().isoformat(),
                }).eq("id", r["id"]).execute()
                disabled.append(r.get("url_or_domain") or r["id"])
    except Exception as e:
        logger.debug(f"curator: auto-disable failed: {e}")
    return disabled


def review_and_learn_sources(sb, *, tracked_job_id: Optional[str], refresh_run_id: str) -> Dict[str, Any]:
    """Post-refresh entry point. Returns a small summary (also persisted)."""
    out: Dict[str, Any] = {"promoted": [], "disabled": [], "flagged": [], "domains": 0}
    try:
        rows = (
            sb.table("job_listings")
            .select("company_domain, company")
            .eq("refresh_run_id", refresh_run_id)
            .eq("relevance", "match")
            .execute().data or []
        )
        by_domain: Dict[str, Set[str]] = {}
        counts: Dict[str, int] = {}
        for r in rows:
            d = _registrable(r.get("company_domain") or "")
            if not d:
                continue
            emp = (r.get("company") or "").strip().lower()
            by_domain.setdefault(d, set()).add(emp or d)
            counts[d] = counts.get(d, 0) + 1

        curated = _load_curated_domains(sb)
        promoted: List[str] = []
        for d, emps in by_domain.items():
            if d in curated:
                _bump_curated_yield(sb, d, counts.get(d, 0))
            else:
                p = _learn_or_promote(sb, d, emps, counts.get(d, 0))
                if p:
                    promoted.append(p)

        disabled = _auto_disable_stale_auto_added(sb)

        per_domain = {d: {"matches": counts.get(d, 0), "employers": len(e)} for d, e in by_domain.items()}
        summary = (f"{len(by_domain)} board(s) yielded; "
                   f"promoted {len(promoted)}; auto-disabled {len(disabled)}")
        try:
            sb.table("job_source_review").insert({
                "tracked_job_id": tracked_job_id,
                "refresh_run_id": refresh_run_id,
                "per_domain": per_domain,
                "promoted": promoted or None,
                "disabled": disabled or None,
                "summary": summary,
            }).execute()
        except Exception as e:
            logger.debug(f"curator: review insert failed: {e}")

        out.update({"promoted": promoted, "disabled": disabled, "domains": len(by_domain), "summary": summary})
    except Exception as e:
        logger.warning(f"curator: review_and_learn_sources failed (non-fatal): {e}")
    return out
