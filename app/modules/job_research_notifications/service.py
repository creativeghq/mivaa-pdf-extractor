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
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.modules._core.registry import is_module_enabled
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

MODULE_SLUG = "job-research-notifications"

CHANNEL_CREDIT_COST: Dict[str, int] = {
    "bell": 0,
    "email": 1,
    "webhook": 0,
}

# Cap per tracked_job in the email so the body stays scannable
MAX_LISTINGS_PER_SECTION = 10


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(d: datetime) -> str:
    return d.isoformat()


class JobDigestDispatcher:
    def __init__(self) -> None:
        self.sb = get_supabase_client().client
        self._supabase_url = os.getenv("SUPABASE_URL") or ""
        self._service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""
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
            since = tj.get("last_digest_sent_at") or _iso(_utcnow() - timedelta(hours=26))
            listings = (
                self.sb.table("job_listings")
                .select("id, url, title, company, company_domain, location, is_remote, "
                        "salary_min, salary_max, salary_currency, employment_type, posted_at, source")
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

        # Build email payload
        user_profile = self._load_user_profile(user_id)
        title = self._build_title(sections, total_listings)
        body_html = self._build_body_html(sections, user_profile)
        body_text = self._build_body_text(sections)
        # Deep-link to the conversation when at least one tracked_job has one set
        first_convo = next((s["tracked_job"].get("source_conversation_id") for s in sections if s["tracked_job"].get("source_conversation_id")), None)
        action_url = self._build_action_url(sections[0]["tracked_job"]["id"], conversation_id=first_convo)

        # Channels: union of all tracked_jobs' alert_channels
        all_channels: set = set()
        for tj in tracked_jobs:
            for c in (tj.get("alert_channels") or ["bell", "email"]):
                all_channels.add(c)
        channels_attempted: List[str] = []
        channels_skipped: List[str] = []

        # 1. Bell (in-app notification) — always free, always send if requested
        if "bell" in all_channels:
            ok = await self._send_bell(user_id, title=title, body=body_text[:300], action_url=action_url, payload={"sections": [{"label": s["tracked_job"]["label"], "count": len(s["listings"])} for s in sections]})
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
                )
                (channels_attempted if ok else channels_skipped).append("email")
            else:
                channels_skipped.append("email")

        # 3. Webhook(s) — per-tracked_job webhook, not user-level
        webhooks_to_call = [(tj, tj.get("alert_webhook_url")) for tj in tracked_jobs if tj.get("alert_webhook_url")]
        if webhooks_to_call:
            await asyncio.gather(*[self._send_webhook(url, {"tracked_job_id": tj["id"], "label": tj["label"], "listings": [s["listings"] for s in sections if s["tracked_job"]["id"] == tj["id"]]}) for tj, url in webhooks_to_call])
            channels_attempted.append("webhook")

        # 4. Mark listings as digest-included
        listing_ids = [l["id"] for s in sections for l in s["listings"]]
        if listing_ids:
            try:
                self.sb.table("job_listings").update({"digest_included_at": _iso(_utcnow())}).in_("id", listing_ids).execute()
            except Exception as e:
                logger.warning(f"job-digest: mark digest_included_at failed: {e}")

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
        if len(sections) == 1:
            return f"{total} new {('match' if total == 1 else 'matches')} for {sections[0]['tracked_job']['label']}"
        return f"{total} new job matches across {len(sections)} of your searches"

    def _build_body_html(self, sections: List[Dict[str, Any]], user_profile: Optional[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for s in sections:
            tj = s["tracked_job"]
            parts.append(
                f'<h2 style="font-weight:400;font-size:16px;margin:24px 0 8px 0;color:#fff;border-bottom:1px solid #333;padding-bottom:6px;">'
                f'{_html_escape(tj["label"])} '
                f'<span style="color:#888;font-size:13px;">({len(s["listings"])} new)</span></h2>'
            )
            parts.append('<div style="display:block;">')
            for l in s["listings"]:
                salary = _fmt_salary(l)
                where = " · ".join(filter(None, [
                    _html_escape(l.get("location") or ""),
                    "Remote" if l.get("is_remote") else None,
                    salary,
                    _html_escape((l.get("employment_type") or "")),
                ]))
                parts.append(
                    f'<div style="margin:0 0 14px 0;padding:10px 12px;background:#1a1a1a;border-radius:6px;">'
                    f'<a href="{_html_escape(l["url"])}" style="color:#d4a3bf;text-decoration:none;font-size:15px;font-weight:500;">'
                    f'{_html_escape(l.get("title") or "(no title)")}</a><br>'
                    f'<span style="color:#bbb;font-size:13px;">{_html_escape(l.get("company") or "")}</span><br>'
                    f'<span style="color:#888;font-size:12px;">{where}</span>'
                    f'</div>'
                )
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
        return "".join(parts)

    def _build_body_text(self, sections: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for s in sections:
            lines.append(f"\n=== {s['tracked_job']['label']} ({len(s['listings'])} new) ===")
            for l in s["listings"]:
                lines.append(f"• {l.get('title') or '(no title)'} — {l.get('company') or ''}")
                lines.append(f"  {l['url']}")
        manual = self._manual_boards()
        if manual:
            lines.append("\n=== Browse these manually (great boards our scraper can't read) ===")
            for b in manual:
                lines.append(f"• {b['name']}: {b['url']}")
        return "\n".join(lines)

    def _manual_boards(self) -> List[Dict[str, str]]:
        try:
            from app.services.integrations.job_search_service import load_manual_review_boards
            return load_manual_review_boards()
        except Exception:
            return []

    def _build_action_url(self, tracked_job_id: str, *, conversation_id: Optional[str] = None) -> str:
        # v0.2: deep-link to the conversation where the user set up the search.
        # Falls back to the agent-hub root if the tracked_job has no source_conversation_id
        # (e.g. created via direct API call rather than KAI agent).
        # PUBLIC_APP_URL is the platform-wide convention (also used by catalog-send-to-customers,
        # catalog-tools.ts, etc.) — same env var, do not introduce a new one.
        base = (os.getenv("PUBLIC_APP_URL") or "https://app.materialshub.gr").rstrip("/")
        if conversation_id:
            return f"{base}/agent-hub?agent=kai&conversation={conversation_id}"
        return f"{base}/agent-hub?agent=kai&q=" + (
            f"Show me today%27s findings for tracked_job_id {tracked_job_id}"
        )

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
                        "employment_type, posted_at, source")
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
        action_url = self._build_action_url(tracked_job_id, conversation_id=tj.get("source_conversation_id"))
        channels = set(tj.get("alert_channels") or ["bell", "email"])
        channels_attempted: List[str] = []
        channels_skipped: List[str] = []

        # Bell (always free)
        if "bell" in channels:
            ok = await self._send_bell(user_id, title=title, body=body_text[:300], action_url=action_url,
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
                "read": False,
            }).execute()
            return True
        except Exception as e:
            logger.warning(f"job-digest bell send: {e}")
            return False

    async def _send_email(
        self,
        *, to_email: str, to_name: str, title: str, body_html: str, action_url: str,
        section_count: int, total_listings: int,
    ) -> bool:
        if not self._supabase_url or not self._service_role_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                resp = await client.post(
                    f"{self._supabase_url}/functions/v1/email-api?action=send",
                    headers={
                        "Authorization": f"Bearer {self._service_role_key}",
                        "apikey": self._service_role_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "to": to_email,
                        "subject": title,
                        "templateSlug": "job_alerts.daily_digest",
                        "variables": {
                            "name": to_name,
                            "title": title,
                            "body": body_html,
                            "action_url": action_url,
                            "tracked_job_count": section_count,
                            "total_new_listings": total_listings,
                        },
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
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                resp = await client.post(url, json=payload)
                return 200 <= resp.status_code < 300
        except Exception as e:
            logger.warning(f"job-digest webhook send: {e}")
            return False

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    def _load_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            res = (
                self.sb.table("user_profiles")
                .select("user_id, email, display_name, full_name")
                .eq("user_id", user_id)
                .maybe_single()
                .execute()
            )
            return (res.data if res else None) or None
        except Exception:
            return None

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
