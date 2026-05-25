"""
Auto-sync `job_research_sites` → ONE consolidated kb_doc in the
"Internal Configuration" category (access_level='agent'; agent reads it,
public KB hides it). Three Markdown sections in one doc, regenerated on
every CRUD.

The category is generic on purpose — future configurable subsystems
(mention outlets, price retailers, etc.) live as sibling docs in the same
category, sharing the same access_level model.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

INTERNAL_CONFIG_CATEGORY_NAME = "Internal Configuration"
DOC_KIND = "job_research_sites"
DOC_TITLE = "Job Research Sites"
DOC_SLUG = "job-research-sites"

SECTIONS = [
    ("perplexity_domain", "Perplexity domain filter",
     "Pinned via Sonar `search_domain_filter`. Capped at 10 by the upstream API; extras truncate alphabetically."),
    ("rss_feed_default", "Default RSS feeds",
     "Suggested to new tracked_jobs when `sources_enabled.rss_feeds: true`."),
    ("careers_page_default", "Default career pages",
     "Suggested to new tracked_jobs when `sources_enabled.careers_pages: true`."),
]


def _render_section(site_type: str, label: str, intro: str, sites: List[Dict[str, Any]], idx: int) -> str:
    enabled = [s for s in sites if s.get("is_enabled")]
    disabled = [s for s in sites if not s.get("is_enabled")]

    lines = [f"## {idx}. {label}", "", f"_{intro}_", ""]
    if not enabled and not disabled:
        lines.append("_(none configured)_")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"**{len(enabled)} enabled** · {len(disabled)} disabled.")
    lines.append("")

    if enabled:
        for s in sorted(enabled, key=lambda r: (r.get("country_code") or "ZZ", r.get("url_or_domain") or "")):
            url = s.get("url_or_domain") or ""
            name = s.get("display_name") or ""
            country = s.get("country_code")
            cat = s.get("category")
            display = f"`{url}`" if site_type == "perplexity_domain" else url
            extras = []
            if name:
                extras.append(name)
            if country:
                extras.append(f"[{country}]")
            if cat:
                extras.append(f"_{cat}_")
            lines.append(f"- {display}" + (" — " + " · ".join(extras) if extras else ""))
        lines.append("")
    if disabled:
        lines.append("<details><summary>Disabled</summary>")
        lines.append("")
        for s in sorted(disabled, key=lambda r: r.get("url_or_domain") or ""):
            lines.append(f"- ~~{s.get('url_or_domain') or ''}~~ — {s.get('display_name') or '(no name)'}")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


def _render_consolidated(all_sites: List[Dict[str, Any]]) -> str:
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for s in all_sites:
        by_type.setdefault(s["site_type"], []).append(s)

    en_counts = {st: sum(1 for s in by_type.get(st, []) if s.get("is_enabled")) for st, _, _ in SECTIONS}
    header = (
        f"# {DOC_TITLE}\n\n"
        f"_Operator-curated list of where the job-research engine looks._\n\n"
        f"**▸ To add / remove / toggle sites: open [/admin/knowledge-base/job-sources](/admin/knowledge-base/job-sources)** "
        f"— the dedicated management page (admin-only writes, hidden from public KB).\n\n"
        f"You can also manage via the KAI agent (`manage_job_sites` tool — say "
        f"\"add kariera.gr to the search\" / \"which job boards do you search?\"). "
        f"Per-tracked_job overrides (`tracked_jobs.careers_page_urls` / `rss_feed_urls`) take precedence over the defaults below.\n\n"
        f"Current totals: "
        f"**{en_counts['perplexity_domain']}** Perplexity domains · "
        f"**{en_counts['rss_feed_default']}** RSS feeds · "
        f"**{en_counts['careers_page_default']}** career pages enabled.\n\n---\n\n"
    )
    body_parts: List[str] = [header]
    for idx, (site_type, label, intro) in enumerate(SECTIONS, start=1):
        body_parts.append(_render_section(site_type, label, intro, by_type.get(site_type, []), idx))
    body_parts.append("---\n\n")
    body_parts.append(f"_Auto-synced at {datetime.now(timezone.utc).isoformat()} by `job_sites_kb_sync.py`._")
    return "".join(body_parts)


def sync_one_site_type(site_type: str) -> bool:
    """Backwards-compatible entry point — re-renders the entire consolidated doc
    regardless of which site_type changed. Kept named per-site_type so callers
    don't have to be updated."""
    return sync_all()


def sync_all() -> bool:
    try:
        sb = get_supabase_client().client

        # Resolve the category (handle both legacy 'Job Sources' name and new 'Internal Configuration')
        cat_res = (
            sb.table("kb_categories").select("id, workspace_id")
            .in_("name", [INTERNAL_CONFIG_CATEGORY_NAME, "Job Sources"])
            .limit(1)
            .execute()
        )
        cat_row = (cat_res.data or [None])[0]
        if not cat_row:
            logger.warning(f"job-kb-sync: '{INTERNAL_CONFIG_CATEGORY_NAME}' category not found; skipping")
            return False

        # Load all sites
        sites_res = sb.table("job_research_sites").select("*").execute()
        sites = sites_res.data or []
        content = _render_consolidated(sites)
        summary = f"{len([s for s in sites if s.get('is_enabled')])} enabled / {len(sites)} total job sites."

        # Upsert by category_id + metadata.doc_kind
        existing = (
            sb.table("kb_docs").select("id")
            .eq("category_id", cat_row["id"])
            .eq("metadata->>doc_kind", DOC_KIND)
            .maybe_single()
            .execute()
        )
        existing_row = (existing.data if existing else None) or None

        if existing_row:
            sb.table("kb_docs").update({
                "title": DOC_TITLE,
                "slug": DOC_SLUG,
                "content": content,
                "summary": summary,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", existing_row["id"]).execute()
        else:
            sb.table("kb_docs").insert({
                "workspace_id": cat_row["workspace_id"],
                "title": DOC_TITLE,
                "slug": DOC_SLUG,
                "category_id": cat_row["id"],
                "content": content,
                "summary": summary,
                "status": "published",
                "visibility": "workspace",
                "metadata": {"doc_kind": DOC_KIND, "auto_synced": True},
            }).execute()
        return True
    except Exception as e:
        logger.warning(f"job-kb-sync failed (non-fatal): {e}")
        return False
