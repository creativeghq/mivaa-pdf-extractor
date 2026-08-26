"""
Job Digest Dispatcher — consolidated daily email per user.

Cron sequence:
  job-research-digest-hourly (cron at :05) → MIVAA POST /cron-digest →
    JobDigestDispatcher.dispatch_due_users(current_hour_utc) →
      for each user whose tracked_job.digest_hour_utc == current_hour_utc
        and last_digest_sent_at < today:
          - load all of that user's tracked_jobs
          - for each tracked_job, fetch new match listings since last digest
          - if any tracked_job has new listings:
              build one consolidated email, send via email-api,
              also write bell notification, optionally POST webhook,
              call append_job_alert_log for each tracked_job (which also
                stamps last_digest_sent_at on tracked_jobs)
          - if NO tracked_job has new listings:
              still stamp last_digest_sent_at on each so we don't reprocess
              for another 24h ("no new listings today" is itself a state)

Module gate: every dispatch first checks
`is_module_enabled('job-research-notifications')`. Bell channel always
sends; email/webhook respect alert_channels config.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from functools import lru_cache
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings
from app.modules._core.registry import is_module_enabled
from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.cron_billing import charge_cron

logger = logging.getLogger(__name__)

MODULE_SLUG = "job-research-notifications"

CHANNEL_CREDIT_COST: Dict[str, int] = {
    "bell": 0,
    "email": 1,
    "webhook": 0,
}

# Per-user request (2026-08-17): NO per-email cap — every match found that day ships
# that day. Set to PostgREST's own page ceiling so it never truncates in practice (a
# 5-day window + daily purge keeps a real run to ~25-40). It stays a finite number
# purely as a runaway guard; the carry-forward path only engages past this, i.e. never.
MAX_LISTINGS_PER_SECTION = 1000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(d: datetime) -> str:
    return d.isoformat()


class JobDigestDispatcher:
    def __init__(self) -> None:
        self.sb = get_supabase_client().client
        # Source creds from `settings` (pydantic BaseSettings, env_file='.env'),
        # NOT raw os.getenv. On prod these live in MIVAA's .env and are NOT exported
        # to the process env, so os.getenv returned "" and _send_email bailed at the
        # guard below on EVERY digest — email had silently never delivered. `self.sb`
        # worked only because get_supabase_client() reads the same settings object.
        self._supabase_url = settings.supabase_url or os.getenv("SUPABASE_URL") or ""
        self._service_role_key = settings.supabase_service_role_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""
        self._http_timeout = httpx.Timeout(15.0, connect=5.0)

    def _module_active(self) -> bool:
        try:
            return is_module_enabled(MODULE_SLUG)
        except Exception as e:
            logger.warning(f"job-digest: module check failed (treating as disabled): {e}")
            return False

    # ────────────────────────────────────────────────────────────────────
    # Cron entry
    # ────────────────────────────────────────────────────────────────────

    async def dispatch_due_users(self, *, current_hour_utc: int) -> Dict[str, Any]:
        if not self._module_active():
            return {"skipped": True, "reason": "module disabled"}

        try:
            res = self.sb.rpc("get_tracked_jobs_due_for_digest", {
                "p_current_hour_utc": int(current_hour_utc),
                "p_limit": 200,
            }).execute()
            rows = res.data or []
        except Exception as e:
            logger.warning(f"job-digest: get_tracked_jobs_due_for_digest failed: {e}")
            return {"error": str(e)[:200]}

        # Group by user
        by_user: Dict[str, List[Dict[str, Any]]] = {}
        for tj in rows:
            uid = tj.get("user_id")
            if not uid:
                continue
            by_user.setdefault(uid, []).append(tj)

        sent = 0
        empty = 0
        errors = 0
        for user_id, tracked_jobs in by_user.items():
            try:
                outcome = await self._dispatch_for_user(user_id, tracked_jobs)
                if outcome.get("sent"):
                    sent += 1
                else:
                    empty += 1
            except Exception as e:
                errors += 1
                logger.warning(f"job-digest: dispatch user {user_id}: {e}")

        return {
            "current_hour_utc": current_hour_utc,
            "users_due": len(by_user),
            "tracked_jobs_due": len(rows),
            "sent": sent,
            "empty": empty,
            "errors": errors,
        }

    # ────────────────────────────────────────────────────────────────────
    # Per-user dispatch (the consolidated email + bell + webhook)
    # ────────────────────────────────────────────────────────────────────

    async def _dispatch_for_user(self, user_id: str, tracked_jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Fetch new listings (relevance='match', not yet digest-included) per tracked_job
        sections: List[Dict[str, Any]] = []
        total_listings = 0
        for tj in tracked_jobs:
            # Deliver every still-present match within the recency window, newest
            # first — NOT only those discovered since the last digest. Purge deletes
            # delivered rows, so "present" == "undelivered"; anchoring the fetch on
            # last_digest_sent_at silently dropped the overflow beyond the per-section
            # cap (a wide search finds >cap/day, and the window then advanced past
            # them). A rolling window carries the backlog forward until it drains.
            _window_days = max(1, min(60, int(tj.get("max_age_days") or 7)))
            since = _iso(_utcnow() - timedelta(days=_window_days))
            listings = (
                self.sb.table("job_listings")
                .select("id, content_hash, url, title, company, company_domain, location, is_remote, "
                        "salary_min, salary_max, salary_currency, employment_type, posted_at, source, "
                        "seniority, description_excerpt, relevance_score, match_note")
                .eq("tracked_job_id", tj["id"])
                .eq("relevance", "match")
                .is_("digest_included_at", "null")
                .gte("discovered_at", since)
                .order("discovered_at", desc=True)
                .limit(MAX_LISTINGS_PER_SECTION)
                .execute()
                .data or []
            )
            if listings:
                sections.append({"tracked_job": tj, "listings": listings})
                total_listings += len(listings)

        # Always stamp last_digest_sent_at via append_job_alert_log even for empty days,
        # so the same row isn't re-evaluated until tomorrow.
        if total_listings == 0:
            for tj in tracked_jobs:
                self._log_alert(tj, user_id, channels_attempted=[], channels_skipped=["bell", "email"], listing_count=0, payload={"reason": "no_new_matches"})
            return {"sent": False, "reason": "no_new_matches"}

        # Meter the digest BEFORE it goes out. Registered cron_key 'job-research-digest' (3 cr),
        # priced "per user consolidated job digest" — so it is charged per SEND, once for the
        # whole consolidated email, not per tracked_job in it.
        #
        # Deliberately placed AFTER the no_new_matches return: an empty day costs nothing.
        # Charging on the way in and refunding later is the shape the cron-billing docstring
        # warns about, and there is no cron_refund_user to make it symmetric anyway. This is the
        # last point before any outward work (email build + send), so it still satisfies
        # "debit before the upstream call".
        #
        # Fails OPEN on any metering error, False only when the payer is genuinely out of credits.
        digest_workspace_id = next((tj.get("workspace_id") for tj in tracked_jobs if tj.get("workspace_id")), None)
        if not charge_cron(
            self.sb, "job-research-digest",
            workspace_id=digest_workspace_id, user_id=user_id,
            description="Consolidated job digest",
            subject={"tracked_job_ids": [str(tj["id"]) for tj in tracked_jobs if tj.get("id")][:20]},
        ):
            for tj in tracked_jobs:
                self._log_alert(tj, user_id, channels_attempted=[], channels_skipped=["bell", "email"],
                                listing_count=total_listings, payload={"reason": "insufficient_credits"})
            return {"sent": False, "reason": "insufficient_credits"}

        # Build email payload
        user_profile = self._load_user_profile(user_id)
        title = self._build_title(sections, total_listings)
        body_html = self._build_body_html(sections, user_profile)
        body_text = self._build_body_text(sections)
        # Deep-link to the conversation when at least one tracked_job has one set
        first_convo = next((s["tracked_job"].get("source_conversation_id") for s in sections if s["tracked_job"].get("source_conversation_id")), None)
        action_path = self._build_action_path(sections[0]["tracked_job"]["id"], conversation_id=first_convo)
        action_url = self._absolute(action_path)   # email CTA; the bell gets the path

        # Channels: union of all tracked_jobs' alert_channels
        all_channels: set = set()
        for tj in tracked_jobs:
            for c in (tj.get("alert_channels") or ["bell", "email"]):
                all_channels.add(c)
        channels_attempted: List[str] = []
        channels_skipped: List[str] = []

        # 1. Bell (in-app notification) — always free, always send if requested.
        # Payload carries keyword-group counts (not the internal search label) so the
        # bell chips match the email's keyword headlines.
        if "bell" in all_channels:
            _kw_groups = self._group_by_keyword(
                [l for s in sections for l in s["listings"]], self._union_keywords(sections)
            )
            bell_payload = {"groups": [{"keyword": h, "count": len(ls)} for h, ls in _kw_groups if h],
                            "total": total_listings}
            ok = await self._send_bell(user_id, title=title, body=body_text[:300], action_url=action_path, payload=bell_payload)
            (channels_attempted if ok else channels_skipped).append("bell")

        # 1b. Chat post into the original conversation (v0.2) — primary user-facing surface.
        # Each tracked_job that has a source_conversation_id gets its own assistant
        # message inserted into that thread. The chunk metadata is rendered as a
        # rich card by AgentHub on conversation reload.
        chat_posted_count = 0
        for s in sections:
            tj = s["tracked_job"]
            convo_id = tj.get("source_conversation_id")
            if not convo_id:
                continue
            ok = await self._post_findings_to_chat(
                conversation_id=convo_id,
                tracked_job=tj,
                listings=s["listings"],
            )
            if ok:
                chat_posted_count += 1
        if chat_posted_count > 0:
            channels_attempted.append("chat")

        # 2. Email
        if "email" in all_channels:
            email_addr = (user_profile or {}).get("email")
            if email_addr:
                ok = await self._send_email(
                    to_email=email_addr,
                    to_name=(user_profile or {}).get("display_name") or "there",
                    title=title, body_html=body_html, action_url=action_url,
                    section_count=len(sections), total_listings=total_listings,
                    workspace_id=digest_workspace_id,
                )
                (channels_attempted if ok else channels_skipped).append("email")
            else:
                channels_skipped.append("email")

        # 3. Webhook(s) — per-tracked_job webhook, not user-level
        webhooks_to_call = [(tj, tj.get("alert_webhook_url")) for tj in tracked_jobs if tj.get("alert_webhook_url")]
        if webhooks_to_call:
            await asyncio.gather(*[self._send_webhook(url, {"tracked_job_id": tj["id"], "label": tj["label"], "listings": [s["listings"] for s in sections if s["tracked_job"]["id"] == tj["id"]]}) for tj, url in webhooks_to_call])
            channels_attempted.append("webhook")

        # Deliver-then-purge, never purge-then-lose. If EVERY channel failed
        # (nothing in channels_attempted), keep the listings and do NOT stamp
        # last_digest_sent_at — the next cron tick retries. Purging here would
        # silently delete jobs the user never saw; that is exactly how email+bell
        # being broken left no trace — the listings were deleted regardless.
        if not channels_attempted:
            logger.warning(
                f"job-digest: all channels failed for user {user_id} — "
                f"{total_listings} listing(s) kept, not stamping (will retry next run)"
            )
            return {"sent": False, "reason": "all_channels_failed", "total_listings": total_listings}

        # 4. Purge after send — "keep the search, not the data". Record each
        #    delivered job's content_hash in job_research_sent (compact tombstone
        #    so it's never re-delivered), then DELETE the full listing rows. The
        #    ledger is consulted by the refresh dedup, so no re-sends.
        listing_ids = [l["id"] for s in sections for l in s["listings"]]
        if listing_ids:
            ledger_rows = [
                {"tracked_job_id": s["tracked_job"]["id"], "content_hash": l.get("content_hash")}
                for s in sections for l in s["listings"] if l.get("content_hash")
            ]
            try:
                if ledger_rows:
                    self.sb.table("job_research_sent").upsert(
                        ledger_rows, on_conflict="tracked_job_id,content_hash"
                    ).execute()
            except Exception as e:
                logger.warning(f"job-digest: sent-ledger upsert failed (will not purge): {e}")
                ledger_rows = []
            # Only delete once the tombstones are safely recorded, so a failure
            # never loses dedup memory (worst case: rows linger, re-sent never).
            if ledger_rows:
                try:
                    self.sb.table("job_listings").delete().in_("id", listing_ids).execute()
                except Exception as e:
                    logger.warning(f"job-digest: post-send listing purge failed: {e}")

        # 5. Log + stamp last_digest_sent_at on each tracked_job
        for tj in tracked_jobs:
            section = next((s for s in sections if s["tracked_job"]["id"] == tj["id"]), None)
            count = len(section["listings"]) if section else 0
            self._log_alert(
                tj, user_id,
                channels_attempted=channels_attempted, channels_skipped=channels_skipped,
                listing_count=count,
                payload={
                    "title": title,
                    "section_listing_counts": {s["tracked_job"]["id"]: len(s["listings"]) for s in sections},
                    "total_listings": total_listings,
                },
            )

        return {"sent": True, "total_listings": total_listings, "section_count": len(sections), "channels": channels_attempted}

    # ────────────────────────────────────────────────────────────────────
    # Body composition
    # ────────────────────────────────────────────────────────────────────

    def _build_title(self, sections: List[Dict[str, Any]], total: int) -> str:
        # Recipient-agnostic — no internal search label or version. This is the same
        # email every user who opts into a digest receives.
        return f"{total} new job {'match' if total == 1 else 'matches'}"

    # ── Keyword grouping ────────────────────────────────────────────────
    def _union_keywords(self, sections: List[Dict[str, Any]]) -> List[str]:
        """Every search keyword across the user's tracked_jobs, in first-seen order."""
        seen: List[str] = []
        for s in sections:
            for k in (s["tracked_job"].get("keywords") or []):
                k = (k or "").strip()
                if k and k not in seen:
                    seen.append(k)
        return seen

    def _group_by_keyword(
        self, listings: List[Dict[str, Any]], keywords: List[str]
    ) -> List[tuple]:
        """Bucket each listing under the keyword its TITLE matches. Checked
        most-specific (longest) first so 'Senior Product Manager' wins over
        'Product Manager'. A role the classifier matched on FUNCTION but whose title
        carries no keyword phrase lands in 'More roles'. Returns (heading, listings)
        in the user's keyword order, 'More roles' last, empty buckets dropped."""
        if not keywords:
            return [("", listings)]
        check_order = sorted(keywords, key=len, reverse=True)
        buckets: Dict[str, List[Dict[str, Any]]] = {k: [] for k in keywords}
        more: List[Dict[str, Any]] = []
        for l in listings:
            title = (l.get("title") or "").lower()
            hit = next((k for k in check_order if k.lower() in title), None)
            (buckets[hit] if hit else more).append(l)
        groups: List[tuple] = [(k, buckets[k]) for k in keywords if buckets[k]]
        if more:
            groups.append(("More roles", more))
        return groups

    def _pretty_empty_sources(self, empty: List[str]) -> List[str]:
        """Human-readable empty-source list: collapse the internal perplexity_* fan-out
        into one line, keep board URLs as-is (they are browsable)."""
        out: List[str] = []
        perplexity = False
        for u in empty:
            if (u or "").startswith("perplexity"):
                perplexity = True
            elif u and u not in out:
                out.append(u)
        if perplexity:
            out.append("Perplexity (web search — check its API credits)")
        return out

    def _listing_card_html(self, l: Dict[str, Any]) -> str:
        salary = _fmt_salary(l)
        where = " · ".join(filter(None, [
            _html_escape(l.get("location") or ""),
            "Remote" if l.get("is_remote") else None,
            salary,
            _html_escape((l.get("employment_type") or "")),
        ]))
        return (
            f'<div style="margin:0 0 14px 0;padding:10px 12px;background:#1a1a1a;border-radius:6px;">'
            f'<a href="{_html_escape(l["url"])}" style="color:#d4a3bf;text-decoration:none;font-size:15px;font-weight:500;">'
            f'{_html_escape(l.get("title") or "(no title)")}</a><br>'
            f'<span style="color:#bbb;font-size:13px;">{_html_escape(l.get("company") or "")}</span><br>'
            f'<span style="color:#888;font-size:12px;">{where}</span>'
            f'</div>'
        )

    def _build_body_html(self, sections: List[Dict[str, Any]], user_profile: Optional[Dict[str, Any]]) -> str:
        parts: List[str] = []
        all_listings = [l for s in sections for l in s["listings"]]
        keywords = self._union_keywords(sections)
        groups = self._group_by_keyword(all_listings, keywords)
        # Only headline the groups when there is more than one — a single bucket needs
        # no divider.
        show_headlines = len(groups) > 1
        for heading, listings in groups:
            if show_headlines and heading:
                parts.append(
                    f'<h2 style="font-weight:400;font-size:16px;margin:24px 0 8px 0;color:#fff;'
                    f'border-bottom:1px solid #333;padding-bottom:6px;">'
                    f'{_html_escape(heading)} '
                    f'<span style="color:#888;font-size:13px;">({len(listings)})</span></h2>'
                )
            parts.append('<div style="display:block;">')
            for l in listings:
                parts.append(self._listing_card_html(l))
            parts.append('</div>')
        manual = self._manual_boards()
        if manual:
            parts.append(
                '<div style="margin-top:18px;padding-top:12px;border-top:1px solid #333;">'
                '<div style="color:#bbb;font-size:13px;margin-bottom:6px;">'
                '🔎 Browse these manually — great remote boards our scraper can\'t read:</div>'
            )
            for b in manual:
                parts.append(
                    f'<div style="margin:3px 0;">'
                    f'<a href="{_html_escape(b["url"])}" style="color:#d4a3bf;text-decoration:none;font-size:14px;">'
                    f'{_html_escape(b["name"])}</a></div>'
                )
            parts.append('</div>')
        # RULE: ALWAYS surface the sources that returned nothing — in the EMAIL too, not
        # just the bell — so a silently-dead board/API is visible, never hidden in a total.
        empty = self._pretty_empty_sources(self._empty_sources([s["tracked_job"]["id"] for s in sections]))
        if empty:
            parts.append(
                '<div style="margin-top:18px;padding-top:12px;border-top:1px solid #333;">'
                f'<div style="color:#bbb;font-size:13px;margin-bottom:6px;">Returned nothing this run '
                f'({len(empty)}) — worth a manual look:</div>'
            )
            for u in empty:
                parts.append(f'<div style="margin:3px 0;color:#888;font-size:12px;">{_html_escape(u)}</div>')
            parts.append('</div>')
        return "".join(parts)

    def _build_body_text(self, sections: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        all_listings = [l for s in sections for l in s["listings"]]
        groups = self._group_by_keyword(all_listings, self._union_keywords(sections))
        show_headlines = len(groups) > 1
        for heading, listings in groups:
            if show_headlines and heading:
                lines.append(f"\n=== {heading} ({len(listings)}) ===")
            for l in listings:
                lines.append(f"• {l.get('title') or '(no title)'} — {l.get('company') or ''}")
                lines.append(f"  {l['url']}")
        manual = self._manual_boards()
        if manual:
            lines.append("\n=== Browse these manually (great boards our scraper can't read) ===")
            for b in manual:
                lines.append(f"• {b['name']}: {b['url']}")
        # RULE: ALWAYS close with the sources that returned nothing this run, so a
        # silently-dead board/feed is visible instead of hidden in a total.
        empty = self._pretty_empty_sources(self._empty_sources([s["tracked_job"]["id"] for s in sections]))
        if empty:
            lines.append(f"\n=== Sources that returned NOTHING this run ({len(empty)}) ===")
            for u in empty:
                lines.append(f"• {u}")
        return "\n".join(lines)

    def _empty_sources(self, tracked_job_ids: List[str]) -> List[str]:
        """Sources (boards/feeds/APIs) that yielded zero on the latest refresh."""
        out: List[str] = []
        try:
            rows = (
                self.sb.table("tracked_jobs")
                .select("last_sources_empty")
                .in_("id", [i for i in tracked_job_ids if i])
                .execute().data or []
            )
            for r in rows:
                for u in (r.get("last_sources_empty") or []):
                    if u and u not in out:
                        out.append(u)
        except Exception as e:
            logger.debug(f"job-digest: empty-source lookup failed: {e}")
        return out

    def _manual_boards(self) -> List[Dict[str, str]]:
        try:
            from app.services.integrations.job_search_service import load_manual_review_boards
            return load_manual_review_boards()
        except Exception:
            return []

    def _build_action_path(self, tracked_job_id: str, *, conversation_id: Optional[str] = None) -> str:
        """The in-app destination, as a PATH.

        v0.2: deep-link to the conversation where the user set up the search. Falls back to a
        seeded agent-hub prompt if the tracked_job has no source_conversation_id (e.g. created
        via direct API call rather than the KAI agent).

        This is what goes on the bell notification, and it must NOT be absolute. The bell hands
        `user_notifications.action_url` to react-router's `navigate()`, which reads ANY string as
        a path — so an absolute URL became the path `/https://app.materialshub.gr/agent-hub` and
        every digest ever sent 404'd when clicked. `_absolute()` is for the email, where a path
        would be equally wrong.
        """
        if conversation_id:
            return "/agent-hub?" + urlencode({"agent": "kai", "conversation": conversation_id})
        return "/agent-hub?" + urlencode({
            "agent": "kai",
            "q": f"Show me today's findings for tracked_job_id {tracked_job_id}",
        })

    def _absolute(self, path: str) -> str:
        """Same destination for a recipient who is not in the app — an email CTA.

        PUBLIC_APP_URL is the platform-wide convention (also used by catalog-send-to-customers,
        catalog-tools.ts, etc.) — same env var, do not introduce a new one.
        """
        base = (os.getenv("PUBLIC_APP_URL") or "https://app.materialshub.gr").rstrip("/")
        return f"{base}{path}"

    # ────────────────────────────────────────────────────────────────────
    # Channel senders
    # ────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────
    # v0.3: real-time burst alert — between daily digest ticks
    # ────────────────────────────────────────────────────────────────────

    async def dispatch_burst_if_warranted(
        self,
        *,
        tracked_job_id: str,
        new_match_count: int,
    ) -> Dict[str, Any]:
        """Called from JobResearchService.refresh() after each refresh completes.
        Fires a single chat-post + bell notification if:
          - the tracked_job has alert_on_burst=true
          - new_match_count >= burst_threshold (default 10)
          - last_burst_alert_at was at least 2 hours ago (or NULL)
          - module is enabled
        Skips silently otherwise.
        """
        if not self._module_active() or new_match_count <= 0:
            return {"skipped": True, "reason": "module disabled or no matches"}

        try:
            res = (
                self.sb.table("tracked_jobs")
                .select("id, user_id, label, alert_on_burst, burst_threshold, last_burst_alert_at, "
                        "source_conversation_id, alert_channels, alert_webhook_url")
                .eq("id", tracked_job_id)
                .maybe_single()
                .execute()
            )
            tj = (res.data if res else None) or None
        except Exception as e:
            logger.warning(f"job-burst: load tracked_job failed: {e}")
            return {"error": str(e)[:200]}

        if not tj or not tj.get("alert_on_burst"):
            return {"skipped": True, "reason": "alert_on_burst not set"}

        threshold = int(tj.get("burst_threshold") or 10)
        if new_match_count < threshold:
            return {"skipped": True, "reason": f"below threshold ({new_match_count} < {threshold})"}

        # Cooldown: 2h between burst alerts for the same tracked_job
        last = tj.get("last_burst_alert_at")
        if last:
            try:
                last_dt = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
                if (_utcnow() - last_dt) < timedelta(hours=2):
                    return {"skipped": True, "reason": "within 2h cooldown"}
            except Exception:
                pass

        # Pull the just-discovered matches (since last_burst_alert_at OR last hour)
        since = last or _iso(_utcnow() - timedelta(hours=1))
        try:
            listings = (
                self.sb.table("job_listings")
                .select("id, url, title, company, company_domain, location, is_remote, "
                        "salary_min, salary_max, salary_currency, salary_annual_min_usd, salary_annual_max_usd, "
                        "employment_type, posted_at, source, seniority, description_excerpt, relevance_score, match_note")
                .eq("tracked_job_id", tracked_job_id)
                .eq("relevance", "match")
                .gte("discovered_at", since)
                .order("discovered_at", desc=True)
                .limit(20)
                .execute()
                .data or []
            )
        except Exception as e:
            logger.warning(f"job-burst: listings fetch failed: {e}")
            return {"error": str(e)[:200]}

        if not listings:
            return {"skipped": True, "reason": "no recent matches to surface"}

        user_id = tj["user_id"]
        label = tj.get("label") or "your job search"
        title = f"🚨 Burst: {new_match_count} new matches for {label}"
        body_text = "Hot batch just landed:\n" + "\n".join(
            f"• {l.get('title') or '(untitled)'} — {l.get('company') or ''}" for l in listings[:5]
        )
        # Path, not URL: the burst reaches the user through the bell and the chat thread, and
        # both are in-app. `_absolute()` is only for an email CTA, and this path sends no email.
        action_path = self._build_action_path(tracked_job_id, conversation_id=tj.get("source_conversation_id"))
        channels = set(tj.get("alert_channels") or ["bell", "email"])
        channels_attempted: List[str] = []
        channels_skipped: List[str] = []

        # Bell (always free)
        if "bell" in channels:
            ok = await self._send_bell(user_id, title=title, body=body_text[:300], action_url=action_path,
                                       payload={"new_match_count": new_match_count, "label": label})
            (channels_attempted if ok else channels_skipped).append("bell")

        # Chat post into the source conversation
        if tj.get("source_conversation_id"):
            ok = await self._post_findings_to_chat(
                conversation_id=tj["source_conversation_id"],
                tracked_job=tj,
                listings=listings,
            )
            if ok:
                channels_attempted.append("chat")

        # Stamp last_burst_alert_at + log
        try:
            self.sb.table("tracked_jobs").update({"last_burst_alert_at": _iso(_utcnow())}).eq("id", tracked_job_id).execute()
            self.sb.rpc("append_job_alert_log", {
                "p_tracked_job_id": tracked_job_id,
                "p_user_id": user_id,
                "p_alert_type": "high_match_burst",
                "p_channels": channels_attempted,
                "p_channels_skipped": channels_skipped,
                "p_listing_count": len(listings),
                "p_payload": {"title": title, "new_match_count": new_match_count, "threshold": threshold},
            }).execute()
        except Exception as e:
            logger.warning(f"job-burst: stamp+log failed: {e}")

        return {"fired": True, "new_match_count": new_match_count, "listings": len(listings), "channels": channels_attempted}

    async def _post_findings_to_chat(
        self, *, conversation_id: str, tracked_job: Dict[str, Any], listings: List[Dict[str, Any]],
    ) -> bool:
        """Insert an assistant message into the agent_chat_messages thread for the
        conversation where the user originally set up this tracked_job. The metadata
        column carries a structured `job_findings` chunk that AgentHub renders as a
        rich card with per-listing save/apply/dismiss buttons."""
        if not listings:
            return False
        try:
            label = tracked_job.get("label") or "your job search"
            count = len(listings)
            text_summary = (
                f"📬 Daily job digest for **{label}** — {count} new "
                f"{'match' if count == 1 else 'matches'} since yesterday."
            )
            payload = {
                "chunk_type": "job_findings",
                "tracked_job_id": tracked_job["id"],
                "tracked_job_label": label,
                "discovered_at_window": "last_24h",
                "listings": [
                    {
                        "id": l.get("id"),
                        "url": l.get("url"),
                        "title": l.get("title"),
                        "company": l.get("company"),
                        "location": l.get("location"),
                        "is_remote": l.get("is_remote"),
                        "salary_min": l.get("salary_min"),
                        "salary_max": l.get("salary_max"),
                        "salary_currency": l.get("salary_currency"),
                        "employment_type": l.get("employment_type"),
                        "posted_at": l.get("posted_at"),
                        "source": l.get("source"),
                        "seniority": l.get("seniority"),
                        "description_excerpt": l.get("description_excerpt"),
                        "relevance_score": l.get("relevance_score"),
                        "match_note": l.get("match_note"),
                    }
                    for l in listings
                ],
            }
            self.sb.table("agent_chat_messages").insert({
                "conversation_id": conversation_id,
                "role": "assistant",
                "content": text_summary,
                "metadata": payload,
            }).execute()
            # Bump the conversation's last_message_at + message_count so the UI surfaces it
            try:
                cur = self.sb.table("agent_chat_conversations").select("message_count").eq("id", conversation_id).maybe_single().execute()
                cnt = ((cur.data if cur else None) or {}).get("message_count") or 0
                self.sb.table("agent_chat_conversations").update({
                    "message_count": int(cnt) + 1,
                    "last_message_at": _iso(_utcnow()),
                    "updated_at": _iso(_utcnow()),
                }).eq("id", conversation_id).execute()
            except Exception as e:
                logger.debug(f"job-digest: bump convo metadata failed: {e}")
            return True
        except Exception as e:
            logger.warning(f"job-digest chat post failed (convo={conversation_id}): {e}")
            return False

    async def _send_bell(self, user_id: str, *, title: str, body: str, action_url: str, payload: Dict[str, Any]) -> bool:
        try:
            self.sb.table("user_notifications").insert({
                "user_id": user_id,
                "type": "job_digest",
                "title": title,
                "body": body,
                "action_url": action_url,
                "metadata": payload,
                # Column is `is_read` (NOT NULL, default false), not `read`. The old
                # key made PostgREST reject the whole insert as an unknown column, so
                # the bell threw every time and was swallowed — the in-app bell had
                # silently never fired for a job digest. Omitting it would also work
                # (default false); we set it explicitly to match the column.
                "is_read": False,
            }).execute()
            return True
        except Exception as e:
            logger.warning(f"job-digest bell send: {e}")
            return False

    async def _send_email(
        self,
        *, to_email: str, to_name: str, title: str, body_html: str, action_url: str,
        section_count: int, total_listings: int, workspace_id: str | None = None,
    ) -> bool:
        if not self._supabase_url or not self._service_role_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                resp = await client.post(
                    # email-api reads the action from the request BODY (or the last
                    # path segment) — NOT the query string. `?action=send` was ignored,
                    # so it resolved to "email-api" and returned 500 "Invalid endpoint".
                    # Put the action in the body AND the path so either resolver hits 'send'.
                    f"{self._supabase_url}/functions/v1/email-api/send",
                    headers={
                        "Authorization": f"Bearer {self._service_role_key}",
                        "apikey": self._service_role_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "action": "send",
                        "to": to_email,
                        "subject": title,
                        # WHOSE email this is, for the LOG ROW only.
                        #
                        # The workspace is resolved a hundred lines above to decide who pays for
                        # the digest, and it never reached the send — so every one of these landed
                        # in `email_logs` with no workspace. That table's member policy is
                        # `workspace_id IS NOT NULL AND is_workspace_member(workspace_id)`, so the
                        # tenant could not see mail going out in their own name.
                        #
                        # `attribution_workspace_id`, NOT `workspace_id`: the latter also picks the
                        # workspace's BYOK sender and meters its daily cap, and a platform-sent
                        # digest must keep taking neither.
                        **({"attribution_workspace_id": workspace_id} if workspace_id else {}),
                        # NO templateSlug: email-api's renderTemplateWithVariables()
                        # escapeHtml's every {{var}}, so the template's {{body}} turned
                        # our pre-built section HTML into literal <h2>…</h2> text in the
                        # inbox. We send the fully-rendered `html` below instead — same
                        # design as the template, but with body_html embedded raw (its
                        # dynamic fields are already _html_escape'd in _build_body_html,
                        # so this is XSS-safe).
                        "html": (
                            f'<!DOCTYPE html><html><body style="background:#0f0f0f;color:#e6e6e6;'
                            f'font-family:Helvetica,Arial,sans-serif;padding:24px;">'
                            f'<h1 style="font-weight:300;font-size:22px;color:#fff;">{_html_escape(title)}</h1>'
                            f'<p style="color:#9a9a9a;font-size:14px;">Hi {_html_escape(to_name)}, here are today\'s findings.</p>'
                            f'{body_html}'
                            f'<p style="margin-top:32px;font-size:12px;color:#777;">'
                            f'<a href="{_html_escape(action_url)}" style="color:#a76b8b;">Open Job Sources →</a>'
                            f'</p></body></html>'
                        ),
                    },
                )
                if resp.status_code >= 400:
                    logger.warning(f"job-digest email-api {resp.status_code}: {resp.text[:200]}")
                    return False
                return True
        except Exception as e:
            logger.warning(f"job-digest email send: {e}")
            return False

    async def _send_webhook(self, url: str, payload: Dict[str, Any]) -> bool:
        try:
            from app.utils.ssrf_guard import assert_safe_url
            assert_safe_url(url)  # pentest #250 E4/E6: block SSRF to internal hosts
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                resp = await client.post(url, json=payload, follow_redirects=False)
                return 200 <= resp.status_code < 300
        except Exception as e:
            logger.warning(f"job-digest webhook send: {e}")
            return False

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    def _load_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        profile: Dict[str, Any] = {}
        try:
            res = (
                self.sb.table("user_profiles")
                .select("user_id, email, display_name, full_name")
                .eq("user_id", user_id)
                .maybe_single()
                .execute()
            )
            profile = (res.data if res else None) or {}
        except Exception:
            profile = {}
        # user_profiles.email is frequently NULL — the address of record lives in
        # auth.users. Without this fallback _send_email got no address and skipped
        # email for every such user, so the digest quietly delivered bell-only
        # (and, before the purge-gate fix, deleted the jobs anyway). Resolve the
        # auth email via the service-role admin API.
        if not profile.get("email"):
            try:
                admin = self.sb.auth.admin.get_user_by_id(user_id)
                auth_user = getattr(admin, "user", None)
                if auth_user is None and isinstance(admin, dict):
                    auth_user = admin.get("user")
                auth_email = getattr(auth_user, "email", None)
                if not auth_email and isinstance(auth_user, dict):
                    auth_email = auth_user.get("email")
                if auth_email:
                    profile["email"] = auth_email
            except Exception as e:
                logger.warning(f"job-digest: auth email lookup failed for {user_id}: {e}")
        return profile or None

    def _log_alert(
        self, tj: Dict[str, Any], user_id: str, *,
        channels_attempted: List[str], channels_skipped: List[str], listing_count: int, payload: Dict[str, Any],
    ) -> None:
        try:
            self.sb.rpc("append_job_alert_log", {
                "p_tracked_job_id": tj["id"],
                "p_user_id": user_id,
                "p_alert_type": "daily_digest",
                "p_channels": channels_attempted,
                "p_channels_skipped": channels_skipped,
                "p_listing_count": int(listing_count),
                "p_payload": payload,
            }).execute()
        except Exception as e:
            logger.warning(f"job-digest append_job_alert_log: {e}")


def _html_escape(s: str) -> str:
    if not s:
        return ""
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&#39;"))


def _fmt_salary(l: Dict[str, Any]) -> Optional[str]:
    smin = l.get("salary_min")
    smax = l.get("salary_max")
    cur = l.get("salary_currency") or ""
    if not smin and not smax:
        return None
    if smin and smax:
        return f"{cur}{smin:,}–{smax:,}".strip()
    return f"{cur}{(smin or smax):,}+".strip()


@lru_cache(maxsize=1)
def get_job_digest_dispatcher() -> JobDigestDispatcher:
    return JobDigestDispatcher()
