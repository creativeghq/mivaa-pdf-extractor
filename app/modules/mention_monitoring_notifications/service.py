"""
Mention Alert Dispatcher — chokepoint for all mention-monitoring alerts.

Mirrors `price_monitoring_notifications.service.PriceAlertDispatcher`. Wraps
the four detector functions (mention spike, negative sentiment, new outlet,
LLM visibility change) and the multi-channel sender (bell / email / webhook)
with credit metering and 24h dedupe.

Module gate: every dispatch first checks
`is_module_enabled('mention-monitoring-notifications')`.

Channels (CHANNEL_CREDIT_COST):
  bell    = 0 cr
  email   = 1 cr
  webhook = 0 cr
"""

from __future__ import annotations

import logging
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.modules._core.registry import is_module_enabled
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

MODULE_SLUG = "mention-monitoring-notifications"

CHANNEL_CREDIT_COST: Dict[str, int] = {
    "bell": 0,
    "email": 1,
    "webhook": 0,
}

# Spike threshold: today vs trailing-7d daily-average
SPIKE_RATIO = 2.0
SPIKE_MIN_DAILY_BASELINE = 1.0  # avoid 0/0 noise

# Negative sentiment alert: only fires for outlets with DA >= this
NEGATIVE_SENTIMENT_DA_FLOOR = 30

# LLM visibility shift threshold: average position changed by this many ranks
LLM_POSITION_DELTA = 2.0

DEDUPE_WINDOW_HOURS = 24


@dataclass
class AlertCandidate:
    alert_type: str  # 'mention_spike' | 'negative_sentiment' | 'new_outlet' | 'llm_visibility_change'
    user_id: str
    tracked_mention_id: str
    title: str
    body: str
    action_url: str
    payload: Dict[str, Any]
    outlet_name: Optional[str] = None
    outlet_domain: Optional[str] = None
    product_id: Optional[str] = None


class MentionAlertDispatcher:
    def __init__(self) -> None:
        self.supabase = get_supabase_client()
        self._supabase_url = os.getenv("SUPABASE_URL") or ""
        self._service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""
        self._http_timeout = httpx.Timeout(10.0, connect=5.0)

    def _module_active(self) -> bool:
        try:
            return is_module_enabled(MODULE_SLUG)
        except Exception as e:
            logger.warning(f"mention-alerts: module check failed (treating as disabled): {e}")
            return False

    # ───── Detection ─────

    def detect_after_refresh(
        self, *, tracked_mention_id: str, new_rows: List[Dict[str, Any]],
    ) -> List[AlertCandidate]:
        """Examine newly-persisted rows + DB-side aggregates to produce candidates."""
        if not self._module_active() or not tracked_mention_id:
            return []

        prefs, user_id, product_id = self._load_subject_prefs(tracked_mention_id)
        if not user_id or not prefs:
            return []

        candidates: List[AlertCandidate] = []
        # Drop mismatch rows (defensive — caller already excluded them)
        rows = [r for r in new_rows if (r.get("relevance") or "").lower() != "mismatch"]

        # ── Spike detection: daily count vs trailing 7d daily avg
        if prefs.get("alert_on_spike"):
            spike = self._detect_spike(tracked_mention_id, todays_count=len(rows))
            if spike:
                candidates.append(AlertCandidate(
                    alert_type="mention_spike",
                    user_id=user_id,
                    tracked_mention_id=tracked_mention_id,
                    product_id=product_id,
                    title=f"Mention spike: {spike['today']} vs {spike['baseline']:.1f}/day baseline",
                    body=(
                        f"Today's count is {spike['ratio']:.1f}× the trailing 7d baseline. "
                        f"Top outlets: {', '.join(spike['top_outlets'][:3])}."
                    ),
                    action_url=self._build_action_url(product_id, tracked_mention_id),
                    payload=spike,
                ))

        # ── Negative sentiment from reputable outlets
        if prefs.get("alert_on_negative_sentiment"):
            for r in rows:
                if r.get("sentiment") != "negative":
                    continue
                domain = (r.get("outlet_domain") or "").lower()
                if not domain:
                    continue
                if not self._is_reputable_outlet(domain):
                    continue
                candidates.append(AlertCandidate(
                    alert_type="negative_sentiment",
                    user_id=user_id,
                    tracked_mention_id=tracked_mention_id,
                    product_id=product_id,
                    outlet_domain=domain,
                    outlet_name=r.get("outlet_name") or domain,
                    title=f"Negative mention from {r.get('outlet_name') or domain}",
                    body=(r.get("title") or r.get("excerpt") or "")[:200],
                    action_url=r.get("url") or self._build_action_url(product_id, tracked_mention_id),
                    payload={
                        "url": r.get("url"),
                        "title": r.get("title"),
                        "sentiment_score": r.get("sentiment_score"),
                        "match_note": r.get("match_note"),
                    },
                ))

        # ── New outlet (domain we've never seen)
        if prefs.get("alert_on_new_outlet"):
            seen_domains = self._known_outlet_domains(tracked_mention_id, exclude_recent=rows)
            announced: set = set()
            for r in rows:
                domain = (r.get("outlet_domain") or "").lower()
                if not domain or domain in seen_domains or domain in announced:
                    continue
                announced.add(domain)
                candidates.append(AlertCandidate(
                    alert_type="new_outlet",
                    user_id=user_id,
                    tracked_mention_id=tracked_mention_id,
                    product_id=product_id,
                    outlet_domain=domain,
                    outlet_name=r.get("outlet_name") or domain,
                    title=f"New outlet: {r.get('outlet_name') or domain}",
                    body=(r.get("title") or r.get("excerpt") or "")[:200],
                    action_url=r.get("url") or self._build_action_url(product_id, tracked_mention_id),
                    payload={"url": r.get("url"), "title": r.get("title")},
                ))

        return candidates

    def detect_after_llm_probe(
        self, *, tracked_mention_id: str, current_snapshot: Dict[str, Any],
    ) -> List[AlertCandidate]:
        """Compare latest LLM probe snapshot to the prior one. One alert if shifted."""
        if not self._module_active() or not tracked_mention_id:
            return []
        prefs, user_id, product_id = self._load_subject_prefs(tracked_mention_id)
        if not user_id or not prefs.get("alert_on_llm_visibility_change"):
            return []

        prev = self._prior_llm_snapshot(tracked_mention_id, exclude_run_id=current_snapshot.get("probe_run_id"))
        if not prev or not prev.get("avg_position") or not current_snapshot.get("avg_position"):
            return []

        delta = float(current_snapshot["avg_position"]) - float(prev["avg_position"])
        if abs(delta) < LLM_POSITION_DELTA:
            return []

        direction = "improved" if delta < 0 else "dropped"
        return [AlertCandidate(
            alert_type="llm_visibility_change",
            user_id=user_id,
            tracked_mention_id=tracked_mention_id,
            product_id=product_id,
            title=f"LLM visibility {direction} by {abs(delta):.1f} ranks",
            body=(
                f"Avg position across LLM probes: "
                f"{prev['avg_position']:.1f} → {current_snapshot['avg_position']:.1f}"
            ),
            action_url=self._build_action_url(product_id, tracked_mention_id),
            payload={
                "previous_avg_position": prev.get("avg_position"),
                "current_avg_position": current_snapshot.get("avg_position"),
                "share_of_voice": current_snapshot.get("share_of_voice"),
            },
        )]

    # ───── Dispatch ─────

    def dispatch(self, candidates: List[AlertCandidate]) -> int:
        if not candidates or not self._module_active():
            return 0

        sent_count = 0
        for cand in candidates:
            if self._is_duplicate(cand):
                logger.debug(f"mention-alerts: dedupe skip {cand.alert_type}/{cand.outlet_domain}")
                continue
            channels = self._channels_for(cand)
            fired: List[str] = []
            skipped: List[str] = []
            credits_total = 0
            for channel in channels:
                cost = CHANNEL_CREDIT_COST.get(channel, 0)
                if cost > 0:
                    if not self._charge_credits(
                        user_id=cand.user_id, amount=cost,
                        operation_type=f"mention_alert.{cand.alert_type}.{channel}",
                    ):
                        skipped.append(f"{channel}_no_credits")
                        continue
                    credits_total += cost
                ok = self._send_channel(channel, cand)
                if ok:
                    fired.append(channel)
                else:
                    skipped.append(f"{channel}_send_failed")
            if fired:
                sent_count += 1
            try:
                self.supabase.client.rpc("append_mention_alert_log", {
                    "p_user_id": cand.user_id,
                    "p_product_id": cand.product_id,
                    "p_tracked_mention_id": cand.tracked_mention_id,
                    "p_alert_type": cand.alert_type,
                    "p_outlet_name": cand.outlet_name,
                    "p_outlet_domain": cand.outlet_domain,
                    "p_payload": cand.payload,
                    "p_channels_fired": fired,
                    "p_channels_skipped": skipped,
                    "p_credits_charged": credits_total,
                }).execute()
            except Exception as e:
                logger.warning(f"mention-alerts: log insert failed: {e}")
        return sent_count

    # ───── Channels ─────

    def _send_channel(self, channel: str, cand: AlertCandidate) -> bool:
        if channel == "bell":
            return self._send_bell(cand)
        if channel == "email":
            return self._send_email(cand)
        if channel == "webhook":
            return self._send_webhook(cand)
        return False

    def _send_bell(self, cand: AlertCandidate) -> bool:
        try:
            self.supabase.client.table("user_notifications").insert({
                "user_id": cand.user_id,
                "type": cand.alert_type,
                "title": cand.title,
                "body": cand.body,
                "action_url": cand.action_url,
                "is_read": False,
                "metadata": {
                    "source_module": MODULE_SLUG,
                    "product_id": cand.product_id,
                    "tracked_mention_id": cand.tracked_mention_id,
                    "outlet_domain": cand.outlet_domain,
                    **cand.payload,
                },
            }).execute()
            return True
        except Exception as e:
            logger.warning(f"mention-alerts: bell send failed: {e}")
            return False

    def _send_email(self, cand: AlertCandidate) -> bool:
        if not self._supabase_url or not self._service_role_key:
            return False
        try:
            resp = (
                self.supabase.client.table("user_profiles")
                .select("email, full_name")
                .eq("user_id", cand.user_id)
                .maybe_single()
                .execute()
            )
            row = (resp.data if resp else None) or {}
            user_email = row.get("email")
            full_name = row.get("full_name") or "there"
        except Exception as e:
            logger.warning(f"mention-alerts: user email lookup failed: {e}")
            return False
        if not user_email:
            return False

        template_slug = f"mention_alert.{cand.alert_type}"
        variables = {
            "name": full_name,
            "title": cand.title,
            "body": cand.body,
            "outlet_name": cand.outlet_name or "",
            "outlet_domain": cand.outlet_domain or "",
            "action_url": cand.action_url,
            "alert_type": cand.alert_type,
            "tracked_mention_id": cand.tracked_mention_id,
            **{k: ("" if v is None else str(v)) for k, v in cand.payload.items()},
        }

        try:
            with httpx.Client(timeout=self._http_timeout) as client:
                resp = client.post(
                    f"{self._supabase_url}/functions/v1/email-api",
                    headers={
                        "Authorization": f"Bearer {self._service_role_key}",
                        "apikey": self._service_role_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "action": "send",
                        "to": user_email,
                        "subject": cand.title,
                        "templateSlug": template_slug,
                        "variables": variables,
                        "html": (
                            f"<h2>{cand.title}</h2><p>{cand.body}</p>"
                            f'<p><a href="{cand.action_url}">Open in Material KAI</a></p>'
                        ),
                        "text": f"{cand.title}\n\n{cand.body}\n\nOpen: {cand.action_url}",
                        "tags": [
                            {"name": "category", "value": "mention_alert"},
                            {"name": "alert_type", "value": cand.alert_type},
                        ],
                    },
                )
                if resp.status_code >= 400:
                    logger.info(f"mention-alerts: email-api {resp.status_code}: {resp.text[:200]}")
                    return False
            return True
        except Exception as e:
            logger.warning(f"mention-alerts: email send failed: {e}")
            return False

    def _send_webhook(self, cand: AlertCandidate) -> bool:
        url = self._user_webhook_url(cand)
        if not url:
            return False
        try:
            with httpx.Client(timeout=self._http_timeout) as client:
                resp = client.post(url, json={
                    "alert_type": cand.alert_type,
                    "title": cand.title,
                    "body": cand.body,
                    "product_id": cand.product_id,
                    "tracked_mention_id": cand.tracked_mention_id,
                    "outlet_name": cand.outlet_name,
                    "outlet_domain": cand.outlet_domain,
                    "payload": cand.payload,
                    "fired_at": datetime.now(timezone.utc).isoformat(),
                })
                resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"mention-alerts: webhook send failed for {url}: {e}")
            return False

    def _user_webhook_url(self, cand: AlertCandidate) -> Optional[str]:
        try:
            resp = (
                self.supabase.client.table("tracked_mentions")
                .select("alert_webhook_url")
                .eq("id", cand.tracked_mention_id)
                .maybe_single()
                .execute()
            )
            row = (resp.data if resp else None) or {}
            return row.get("alert_webhook_url")
        except Exception:
            return None

    # ───── Credits ─────

    def _charge_credits(self, *, user_id: str, amount: int, operation_type: str) -> bool:
        try:
            result = self.supabase.client.rpc("debit_user_credits", {
                "p_user_id": user_id,
                "p_amount": amount,
                "p_operation_type": operation_type,
            }).execute()
            return bool(result.data) if hasattr(result, "data") else True
        except Exception as e:
            logger.info(f"mention-alerts: credit charge skipped: {e}")
            return False

    # ───── Helpers ─────

    def _load_subject_prefs(
        self, tracked_mention_id: str,
    ) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
        try:
            resp = (
                self.supabase.client.table("tracked_mentions")
                .select(
                    "user_id, product_id, alert_on_spike, alert_on_negative_sentiment, "
                    "alert_on_new_outlet, alert_on_llm_visibility_change, alert_channels, alert_webhook_url"
                )
                .eq("id", tracked_mention_id)
                .maybe_single()
                .execute()
            )
            row = (resp.data if resp else None) or {}
            return row, row.get("user_id"), row.get("product_id")
        except Exception as e:
            logger.warning(f"mention-alerts: subject prefs lookup failed: {e}")
            return {}, None, None

    def _channels_for(self, cand: AlertCandidate) -> List[str]:
        prefs, _, _ = self._load_subject_prefs(cand.tracked_mention_id)
        chans = prefs.get("alert_channels") or ["bell"]
        return [c for c in chans if c in CHANNEL_CREDIT_COST]

    def _build_action_url(self, product_id: Optional[str], tracked_mention_id: Optional[str]) -> str:
        if product_id:
            return f"/admin/mention-monitoring/products/{product_id}"
        if tracked_mention_id:
            return f"/admin/mention-monitoring/tracked/{tracked_mention_id}"
        return "/admin/mention-monitoring"

    def _is_duplicate(self, cand: AlertCandidate) -> bool:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=DEDUPE_WINDOW_HOURS)).isoformat()
        try:
            q = (
                self.supabase.client.table("mention_alert_log")
                .select("id")
                .eq("alert_type", cand.alert_type)
                .gte("created_at", cutoff)
                .eq("tracked_mention_id", cand.tracked_mention_id)
            )
            if cand.outlet_domain:
                q = q.eq("outlet_domain", cand.outlet_domain)
            resp = q.limit(1).execute()
            return bool(resp.data)
        except Exception:
            return False

    def _detect_spike(self, tracked_mention_id: str, *, todays_count: int) -> Optional[Dict[str, Any]]:
        cutoff_7d = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        cutoff_today = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        try:
            r7 = (
                self.supabase.client.table("mention_history")
                .select("outlet_domain, discovered_at")
                .eq("tracked_mention_id", tracked_mention_id)
                .gte("discovered_at", cutoff_7d)
                .lt("discovered_at", cutoff_today)
                .execute()
            ).data or []
        except Exception:
            r7 = []
        baseline = max(SPIKE_MIN_DAILY_BASELINE, len(r7) / 7.0)
        ratio = todays_count / baseline if baseline > 0 else 0.0
        if ratio < SPIKE_RATIO:
            return None
        # Top outlets in today's window (best-effort)
        try:
            today_rows = (
                self.supabase.client.table("mention_history")
                .select("outlet_domain")
                .eq("tracked_mention_id", tracked_mention_id)
                .gte("discovered_at", cutoff_today)
                .execute()
            ).data or []
        except Exception:
            today_rows = []
        outlet_counts: Dict[str, int] = {}
        for row in today_rows:
            d = (row.get("outlet_domain") or "").lower()
            if d:
                outlet_counts[d] = outlet_counts.get(d, 0) + 1
        top_outlets = [d for d, _ in sorted(outlet_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]]
        return {
            "today": todays_count,
            "baseline": baseline,
            "ratio": ratio,
            "top_outlets": top_outlets,
        }

    def _is_reputable_outlet(self, domain: str) -> bool:
        try:
            r = (
                self.supabase.client.table("mention_outlets")
                .select("domain_authority")
                .eq("domain", domain)
                .maybe_single()
                .execute()
            )
            row = (r.data if r else None) or {}
            return int(row.get("domain_authority") or 0) >= NEGATIVE_SENTIMENT_DA_FLOOR
        except Exception:
            return False

    def _known_outlet_domains(
        self, tracked_mention_id: str, *, exclude_recent: List[Dict[str, Any]],
    ) -> set:
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        try:
            r = (
                self.supabase.client.table("mention_history")
                .select("outlet_domain, discovered_at")
                .eq("tracked_mention_id", tracked_mention_id)
                .lt("discovered_at", cutoff)
                .execute()
            )
            return {(x.get("outlet_domain") or "").lower() for x in (r.data or []) if x.get("outlet_domain")}
        except Exception:
            return set()

    def _prior_llm_snapshot(
        self, tracked_mention_id: str, *, exclude_run_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        try:
            q = (
                self.supabase.client.table("llm_mention_probes")
                .select("probe_run_id, run_at, position, mentioned")
                .eq("tracked_mention_id", tracked_mention_id)
                .order("run_at", desc=True)
                .limit(200)
            )
            rows = (q.execute()).data or []
        except Exception:
            return None
        # Group by run_id, find first run that is not the current one
        seen: List[str] = []
        for r in rows:
            rid = r.get("probe_run_id")
            if rid and rid != exclude_run_id and rid not in seen:
                seen.append(rid)
        if len(seen) < 1:
            return None
        prior_run_id = seen[0]
        prior_rows = [r for r in rows if r.get("probe_run_id") == prior_run_id and r.get("mentioned")]
        positions = [int(r.get("position")) for r in prior_rows if r.get("position")]
        if not positions:
            return None
        return {
            "probe_run_id": prior_run_id,
            "avg_position": sum(positions) / len(positions),
            "mentioned_count": len(prior_rows),
        }


_dispatcher: Optional[MentionAlertDispatcher] = None


def get_mention_alert_dispatcher() -> MentionAlertDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = MentionAlertDispatcher()
    return _dispatcher
