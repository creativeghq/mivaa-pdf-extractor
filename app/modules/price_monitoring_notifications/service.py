"""
Price Alert Dispatcher
======================

Single chokepoint for all price-monitoring-related alerts. Wraps the four
detector functions (sanity-band anomaly, price-drop, new-retailer,
promo-started) and the multi-channel sender (bell / email / webhook) with
credit metering and dedupe.

Design constraints:
  - Module-gated: every dispatch first checks `is_module_enabled('price-monitoring-notifications')`.
  - Per-channel credit cost: bell=0, email=1, webhook=0. Insufficient credits
    drop the channel silently and log to `price_alert_log.channels_skipped`.
  - 24h dedupe per (product, alert_type, retailer_domain) — re-firing the
    same alert wakes the user up at 3am for nothing.
  - Direct insert into `user_notifications` (the table the bell actually reads).
    The legacy `notifications` table is dead; do not write there.

The detection logic is intentionally plain SQL + Python — no LLM in the hot
path, because alerts must be cheap and deterministic. LLM is only invoked
upstream by the existing identity classifier.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.modules._core.registry import is_module_enabled
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

MODULE_SLUG = "price-monitoring-notifications"

# ── Per-channel credit cost. Bell and webhook are essentially free for us
#    (one DB row + a fan-out to consumer infra). Email is real money via
#    SES/Postmark, so it earns a credit charge. Tunable per platform policy.
CHANNEL_CREDIT_COST: Dict[str, int] = {
    "bell": 0,
    "email": 1,
    "webhook": 0,
}

# ── Sanity-band tunables. New reading outside [median/3, median*3] gets
#    flagged as anomaly. Window = trailing 7 days, min sample = 3 readings
#    (below that we trust the new reading; not enough history to judge).
SANITY_WINDOW_DAYS = 7
SANITY_MIN_SAMPLES = 3
SANITY_LOW_RATIO = 0.33   # < median * 0.33  => anomaly
SANITY_HIGH_RATIO = 3.0   # > median * 3.0   => anomaly

# ── Price-drop alert: trailing 7-day median vs prior 7-day median.
PRICE_DROP_THRESHOLD_PCT = 10.0  # require ≥ 10% drop W/W to fire

# ── Dedupe window. Same (alert_type, product, retailer_domain) won't re-fire
#    inside this window even if conditions remain true.
DEDUPE_WINDOW_HOURS = 24


# ────────────────────────────────────────────────────────────────────────────
# Detection result types
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SanityVerdict:
    """Outcome of the sanity band check on a single new price reading."""
    is_anomaly: bool
    rolling_median: Optional[float]
    sample_count: int
    reason: Optional[str] = None


@dataclass
class AlertCandidate:
    """Detected alert ready for dispatch."""
    alert_type: str        # 'price_drop' | 'new_retailer' | 'promo_started' | 'anomaly_detected'
    user_id: str
    tracked_query_id: str
    retailer_name: str
    retailer_domain: Optional[str]
    title: str
    body: str
    action_url: str
    payload: Dict[str, Any]
    # Optional product_id for downstream consumers (price_alert_log row,
    # bell-action URL). Resolved from tracked_queries.product_id at dispatch
    # time when the row is internal-flow.
    product_id: Optional[str] = None


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _domain_of(url: Optional[str]) -> Optional[str]:
    """Extract a normalized hostname from a URL (no www., lowercase)."""
    if not url:
        return None
    m = re.match(r"^https?://([^/]+)", url.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    host = m.group(1).lower()
    return host[4:] if host.startswith("www.") else host


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


# ────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ────────────────────────────────────────────────────────────────────────────

class PriceAlertDispatcher:
    """
    Stateless service. All state lives in DB tables. Methods are async-friendly
    but use the sync supabase client (the codebase convention).
    """

    def __init__(self) -> None:
        self.supabase = get_supabase_client()
        # Email is wired through the platform's existing `email-api` edge
        # function (Resend-backed). When SUPABASE_URL is set we dispatch via
        # functions.invoke; only when it's missing do we degrade to bell-only.
        self._supabase_url = os.getenv("SUPABASE_URL") or ""
        self._service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""
        self._http_timeout = httpx.Timeout(10.0, connect=5.0)

    # ───── Module gate ─────

    def _module_active(self) -> bool:
        try:
            return is_module_enabled(MODULE_SLUG)
        except Exception as e:
            logger.warning(f"price-alerts: module check failed (treating as disabled): {e}")
            return False

    # ───── PR 1a: rolling-median sanity band ─────

    def check_sanity(
        self,
        *,
        tracked_query_id: str,
        retailer_domain: str,
        new_price: float,
    ) -> SanityVerdict:
        """
        Compute the trailing-7d median for (tracked_query, retailer) and
        decide whether `new_price` is plausible. Returns a verdict the caller
        stamps onto the row before insert.
        """
        if not retailer_domain or new_price is None or new_price <= 0:
            return SanityVerdict(False, None, 0, None)
        if not tracked_query_id:
            return SanityVerdict(False, None, 0, "no subject")

        cutoff = (datetime.now(timezone.utc) - timedelta(days=SANITY_WINDOW_DAYS)).isoformat()

        try:
            resp = (
                self.supabase.client.table("tracked_query_price_history")
                .select("price, product_url, match_kind")
                .eq("tracked_query_id", tracked_query_id)
                .gte("scraped_at", cutoff)
                .eq("is_anomaly", False)
                .execute()
            )
            rows = resp.data or []
        except Exception as e:
            logger.warning(f"price-alerts: sanity history fetch failed: {e}")
            return SanityVerdict(False, None, 0, f"history fetch failed: {e}")

        retailer_prices: List[float] = []
        url_field = "product_url"
        for r in rows:
            if (r.get("match_kind") or "").lower() == "family":
                continue
            url = r.get(url_field) or ""
            if _domain_of(url) == retailer_domain and r.get("price") is not None:
                try:
                    retailer_prices.append(float(r["price"]))
                except (TypeError, ValueError):
                    continue

        if len(retailer_prices) < SANITY_MIN_SAMPLES:
            return SanityVerdict(False, None, len(retailer_prices), "insufficient history")

        med = _median(retailer_prices) or 0.0
        if med <= 0:
            return SanityVerdict(False, med, len(retailer_prices), "zero median")

        ratio = new_price / med
        if ratio < SANITY_LOW_RATIO:
            return SanityVerdict(
                True, med, len(retailer_prices),
                f"reading €{new_price:.2f} is {ratio:.2f}× the 7d median €{med:.2f} — too low",
            )
        if ratio > SANITY_HIGH_RATIO:
            return SanityVerdict(
                True, med, len(retailer_prices),
                f"reading €{new_price:.2f} is {ratio:.2f}× the 7d median €{med:.2f} — too high",
            )
        return SanityVerdict(False, med, len(retailer_prices), None)

    # ───── PR 1b: detection — runs after a refresh persists ─────

    def detect_after_refresh(
        self,
        *,
        tracked_query_id: str,
        new_rows: List[Dict[str, Any]],
    ) -> List[AlertCandidate]:
        """
        Examine the rows just written by the refresh and produce alert
        candidates. Caller fans them out via `dispatch()`.

        new_rows are the just-written dicts as inserted (with is_anomaly /
        original_price already stamped).
        """
        if not self._module_active() or not tracked_query_id:
            return []

        candidates: List[AlertCandidate] = []
        prefs, user_id, product_id = self._load_subject_prefs(tracked_query_id=tracked_query_id)
        if user_id is None or not prefs:
            return []

        # Family rows are inert — they never trigger ANY alert. The user
        # explicitly told us they're not the tracked product.
        new_rows = [r for r in new_rows if (r.get("match_kind") or "").lower() != "family"]

        # Anomaly alert (always opt-in by virtue of sanity band running).
        for r in new_rows:
            if not r.get("is_anomaly"):
                continue
            domain = _domain_of(r.get("product_url"))
            candidates.append(AlertCandidate(
                alert_type="anomaly_detected",
                user_id=user_id,
                tracked_query_id=tracked_query_id,
                product_id=product_id,
                retailer_name=r.get("retailer_name") or domain or "unknown",
                retailer_domain=domain,
                title=f"Anomalous price flagged: {r.get('retailer_name') or domain}",
                body=(r.get("anomaly_reason") or "")[:240],
                action_url=self._build_action_url(product_id, tracked_query_id),
                payload={
                    "rolling_median": r.get("rolling_median_at_check"),
                    "rejected_price": r.get("price"),
                    "url": r.get("product_url"),
                },
            ))

        # Promo started — original_price became non-null on a row that prior
        # had it null. We compute by joining the new row to the most recent
        # PRIOR row for the same (subject, retailer_domain).
        if prefs.get("alert_on_promo"):
            for r in new_rows:
                if r.get("original_price") is None:
                    continue
                domain = _domain_of(r.get("product_url"))
                if not domain:
                    continue
                if not self._previously_had_promo(tracked_query_id, domain):
                    candidates.append(AlertCandidate(
                        alert_type="promo_started",
                        user_id=user_id,
                        tracked_query_id=tracked_query_id,
                        product_id=product_id,
                        retailer_name=r.get("retailer_name") or domain,
                        retailer_domain=domain,
                        title=f"Promo started at {r.get('retailer_name') or domain}",
                        body=(
                            f"Now €{r.get('price')} (was €{r.get('original_price')})"
                        ),
                        action_url=self._build_action_url(product_id, tracked_query_id),
                        payload={
                            "current_price": r.get("price"),
                            "original_price": r.get("original_price"),
                            "url": r.get("product_url"),
                        },
                    ))

        # New retailer — domain we've never seen for this tracked_query.
        if prefs.get("alert_on_new_retailer"):
            seen_domains = self._known_retailer_domains(tracked_query_id, exclude_new_rows=new_rows)
            announced: set = set()
            for r in new_rows:
                if r.get("is_anomaly"):
                    continue
                domain = _domain_of(r.get("product_url"))
                if not domain or domain in seen_domains or domain in announced:
                    continue
                announced.add(domain)
                candidates.append(AlertCandidate(
                    alert_type="new_retailer",
                    user_id=user_id,
                    tracked_query_id=tracked_query_id,
                    product_id=product_id,
                    retailer_name=r.get("retailer_name") or domain,
                    retailer_domain=domain,
                    title=f"New retailer found: {r.get('retailer_name') or domain}",
                    body=f"Listed at €{r.get('price')}.",
                    action_url=self._build_action_url(product_id, tracked_query_id),
                    payload={
                        "current_price": r.get("price"),
                        "url": r.get("product_url"),
                    },
                ))

        # Price drop — compares week-over-week median per retailer.
        if prefs.get("alert_on_price_drop"):
            drops = self._detect_price_drops(tracked_query_id=tracked_query_id)
            for d in drops:
                candidates.append(AlertCandidate(
                    alert_type="price_drop",
                    user_id=user_id,
                    tracked_query_id=tracked_query_id,
                    product_id=product_id,
                    retailer_name=d["retailer_name"] or d["domain"],
                    retailer_domain=d["domain"],
                    title=f"Price drop at {d['retailer_name'] or d['domain']}",
                    body=f"Down {d['delta_pct']:.1f}% W/W: €{d['previous_median']:.2f} → €{d['current_median']:.2f}",
                    action_url=self._build_action_url(product_id, tracked_query_id),
                    payload=d,
                ))

        return candidates

    # ───── Multi-channel send + credit metering ─────

    def dispatch(self, candidates: List[AlertCandidate]) -> int:
        """
        Fan out each candidate across the user's preferred channels, charging
        credits per channel. Returns the count of alerts that fired (any
        channel succeeded). Records every attempt in `price_alert_log`.
        """
        if not candidates or not self._module_active():
            return 0

        sent_count = 0
        for cand in candidates:
            if self._is_duplicate(cand):
                logger.debug(f"price-alerts: dedupe skip {cand.alert_type}/{cand.retailer_domain}")
                continue

            channels = self._channels_for(cand)
            fired: List[str] = []
            skipped: List[str] = []
            credits_total = 0

            for channel in channels:
                cost = CHANNEL_CREDIT_COST.get(channel, 0)
                if cost > 0:
                    if not self._charge_credits(
                        user_id=cand.user_id,
                        amount=cost,
                        operation_type=f"price_alert.{cand.alert_type}.{channel}",
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
                self.supabase.client.table("price_alert_log").insert({
                    "user_id": cand.user_id,
                    "product_id": cand.product_id,
                    "tracked_query_id": cand.tracked_query_id,
                    "alert_type": cand.alert_type,
                    "retailer_name": cand.retailer_name,
                    "retailer_domain": cand.retailer_domain,
                    "payload": cand.payload,
                    "channels_fired": fired,
                    "channels_skipped": skipped,
                    "credits_charged": credits_total,
                }).execute()
            except Exception as e:
                logger.warning(f"price-alerts: alert log insert failed: {e}")

        return sent_count

    # ───── Channel senders ─────

    def _send_channel(self, channel: str, cand: AlertCandidate) -> bool:
        if channel == "bell":
            return self._send_bell(cand)
        if channel == "email":
            return self._send_email(cand)
        if channel == "webhook":
            return self._send_webhook(cand)
        logger.warning(f"price-alerts: unknown channel {channel}")
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
                    "tracked_query_id": cand.tracked_query_id,
                    "retailer_domain": cand.retailer_domain,
                    **cand.payload,
                },
            }).execute()
            return True
        except Exception as e:
            logger.warning(f"price-alerts: bell send failed: {e}")
            return False

    def _send_email(self, cand: AlertCandidate) -> bool:
        """
        Email goes through the platform's existing `email-api` edge function
        (Resend-backed, configured with RESEND_API_KEY). We invoke it with a
        templateSlug — the templates are seeded into `email_templates` and
        rendered with React Email server-side. If the user's email is missing
        or RESEND_API_KEY isn't configured, we degrade to bell-only and the
        skip is logged on the alert log row.
        """
        if not self._supabase_url or not self._service_role_key:
            return False

        # Resolve the user's email — required by Resend.
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
            logger.warning(f"price-alerts: user email lookup failed: {e}")
            return False
        if not user_email:
            return False

        template_slug = f"price_alert.{cand.alert_type}"
        variables = {
            "name": full_name,
            "title": cand.title,
            "body": cand.body,
            "retailer_name": cand.retailer_name or "",
            "retailer_domain": cand.retailer_domain or "",
            "action_url": cand.action_url,
            "alert_type": cand.alert_type,
            **{k: ("" if v is None else str(v)) for k, v in cand.payload.items()},
        }

        try:
            with httpx.Client(timeout=self._http_timeout) as client:
                resp = client.post(
                    f"{self._supabase_url}/functions/v1/email-api",
                    headers={
                        # Both Authorization AND apikey required — the email-api
                        # auth helper recognizes service-role only when sent as
                        # `apikey` (it gets level='secret' bypass). Sending
                        # service-role only on Authorization fails getUser.
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
                        # Fallback for the case where the template hasn't been
                        # seeded yet — email-api will use these fields directly.
                        "html": (
                            f"<h2>{cand.title}</h2><p>{cand.body}</p>"
                            f'<p><a href="{cand.action_url}">Open in Material KAI</a></p>'
                        ),
                        "text": f"{cand.title}\n\n{cand.body}\n\nOpen: {cand.action_url}",
                        "tags": [
                            {"name": "category", "value": "price_alert"},
                            {"name": "alert_type", "value": cand.alert_type},
                        ],
                    },
                )
                # email-api returns 4xx when RESEND_API_KEY is missing or template
                # is invalid; both legitimately mean "fall back to bell-only" so
                # we don't block dispatch.
                if resp.status_code >= 400:
                    logger.info(f"price-alerts: email-api {resp.status_code}: {resp.text[:200]}")
                    return False
            return True
        except Exception as e:
            logger.warning(f"price-alerts: email send failed: {e}")
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
                    "tracked_query_id": cand.tracked_query_id,
                    "retailer_name": cand.retailer_name,
                    "retailer_domain": cand.retailer_domain,
                    "payload": cand.payload,
                    "fired_at": datetime.now(timezone.utc).isoformat(),
                })
                resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"price-alerts: webhook send failed for {url}: {e}")
            return False

    def _user_webhook_url(self, cand: AlertCandidate) -> Optional[str]:
        """
        Resolution: external tracked_queries carry their own per-query webhook
        URL. Internal product monitoring has no per-product webhook today —
        future iteration could read user_profiles.alert_webhook_url.
        """
        if cand.tracked_query_id:
            try:
                resp = (
                    self.supabase.client.table("tracked_queries")
                    .select("alert_webhook_url")
                    .eq("id", cand.tracked_query_id)
                    .maybe_single()
                    .execute()
                )
                row = (resp.data if resp else None) or {}
                return row.get("alert_webhook_url")
            except Exception:
                return None
        return None

    # ───── Credit metering ─────

    def _charge_credits(self, *, user_id: str, amount: int, operation_type: str) -> bool:
        """
        Atomic debit via existing RPC. Returns True if charged successfully.
        On insufficient balance / failure returns False — caller skips the
        channel without raising.
        """
        try:
            result = self.supabase.client.rpc("debit_user_credits", {
                "p_user_id": user_id,
                "p_amount": amount,
                "p_operation_type": operation_type,
            }).execute()
            ok = bool(result.data) if hasattr(result, "data") else True
            return ok
        except Exception as e:
            logger.info(f"price-alerts: credit charge skipped (likely insufficient): {e}")
            return False

    # ───── Subject lookups ─────

    def _load_subject_prefs(
        self,
        *,
        tracked_query_id: str,
    ) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
        """Read alert prefs + owner user_id + product_id (when internal-flow).
        Returns (prefs_dict, user_id, product_id).
        """
        try:
            resp = (
                self.supabase.client.table("tracked_queries")
                .select(
                    "user_id, product_id, alert_on_price_drop, alert_on_new_retailer, "
                    "alert_on_promo, alert_channels, alert_webhook_url"
                )
                .eq("id", tracked_query_id)
                .maybe_single()
                .execute()
            )
            row = (resp.data if resp else None) or {}
            return row, row.get("user_id"), row.get("product_id")
        except Exception as e:
            logger.warning(f"price-alerts: subject prefs lookup failed: {e}")
        return {}, None, None

    def _channels_for(self, cand: AlertCandidate) -> List[str]:
        prefs, _, _ = self._load_subject_prefs(tracked_query_id=cand.tracked_query_id)
        chans = prefs.get("alert_channels") or ["bell"]
        # Dedupe + intersect with known channels
        return [c for c in chans if c in CHANNEL_CREDIT_COST]

    def _build_action_url(self, product_id: Optional[str], tracked_query_id: Optional[str]) -> str:
        if product_id:
            return f"/admin/price-monitoring/products/{product_id}"
        if tracked_query_id:
            return f"/admin/price-monitoring/tracked/{tracked_query_id}"
        return "/admin/price-monitoring"

    def _is_duplicate(self, cand: AlertCandidate) -> bool:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=DEDUPE_WINDOW_HOURS)).isoformat()
        try:
            q = (
                self.supabase.client.table("price_alert_log")
                .select("id")
                .eq("alert_type", cand.alert_type)
                .gte("created_at", cutoff)
                .eq("tracked_query_id", cand.tracked_query_id)
            )
            if cand.retailer_domain:
                q = q.eq("retailer_domain", cand.retailer_domain)
            resp = q.limit(1).execute()
            return bool(resp.data)
        except Exception as e:
            logger.debug(f"price-alerts: dedupe check failed (treating as not-dup): {e}")
            return False

    def _previously_had_promo(self, tracked_query_id: str, retailer_domain: str) -> bool:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        try:
            resp = (
                self.supabase.client.table("tracked_query_price_history")
                .select("original_price, product_url")
                .eq("tracked_query_id", tracked_query_id)
                .gte("scraped_at", cutoff)
                .not_.is_("original_price", "null")
                .limit(50)
                .execute()
            )
            for r in resp.data or []:
                if _domain_of(r.get("product_url")) == retailer_domain:
                    return True
        except Exception:
            pass
        return False

    def _known_retailer_domains(
        self,
        tracked_query_id: str,
        exclude_new_rows: List[Dict[str, Any]],
    ) -> set:
        seen: set = set()
        try:
            # Exclude rows scraped in the last 5 minutes — those are the
            # rows we just persisted. Without this filter every row is
            # "known" the moment we insert it, and new_retailer alerts
            # never fire.
            cutoff = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
            resp = (
                self.supabase.client.table("tracked_query_price_history")
                .select("product_url, scraped_at")
                .eq("tracked_query_id", tracked_query_id)
                .lt("scraped_at", cutoff)
                .execute()
            )
            for r in resp.data or []:
                domain = _domain_of(r.get("product_url"))
                if domain:
                    seen.add(domain)
        except Exception as e:
            logger.warning(f"price-alerts: known-retailer scan failed: {e}")
        return seen

    def _detect_price_drops(self, *, tracked_query_id: str) -> List[Dict[str, Any]]:
        """Trailing 7d median vs prior 7d median per retailer for this
        tracked_query. Returns one entry per retailer whose drop ≥
        PRICE_DROP_THRESHOLD_PCT.
        """
        now = datetime.now(timezone.utc)
        cur_lo = (now - timedelta(days=7)).isoformat()
        prev_lo = (now - timedelta(days=14)).isoformat()
        prev_hi = (now - timedelta(days=7)).isoformat()

        url_field = "product_url"
        name_field = "retailer_name"

        try:
            cur_rows = (
                self.supabase.client.table("tracked_query_price_history")
                .select(f"price, {url_field}, {name_field}, scraped_at")
                .eq("tracked_query_id", tracked_query_id)
                .eq("is_anomaly", False)
                .gte("scraped_at", cur_lo)
                .execute()
            ).data or []
            prev_rows = (
                self.supabase.client.table("tracked_query_price_history")
                .select(f"price, {url_field}, {name_field}, scraped_at")
                .eq("tracked_query_id", tracked_query_id)
                .eq("is_anomaly", False)
                .gte("scraped_at", prev_lo)
                .lt("scraped_at", prev_hi)
                .execute()
            ).data or []
        except Exception as e:
            logger.warning(f"price-alerts: drop detect query failed: {e}")
            return []

        def _by_domain(rows: List[Dict[str, Any]]) -> Dict[str, Tuple[List[float], str]]:
            out: Dict[str, Tuple[List[float], str]] = {}
            for r in rows:
                d = _domain_of(r.get(url_field))
                if not d or r.get("price") is None:
                    continue
                bucket = out.setdefault(d, ([], r.get(name_field) or d))
                try:
                    bucket[0].append(float(r["price"]))
                except (TypeError, ValueError):
                    continue
            return out

        cur_by = _by_domain(cur_rows)
        prev_by = _by_domain(prev_rows)
        results: List[Dict[str, Any]] = []
        for domain, (cur_prices, name) in cur_by.items():
            if domain not in prev_by:
                continue
            cm = _median(cur_prices)
            pm = _median(prev_by[domain][0])
            if not cm or not pm or pm <= 0:
                continue
            delta_pct = (pm - cm) / pm * 100.0
            if delta_pct >= PRICE_DROP_THRESHOLD_PCT:
                results.append({
                    "domain": domain,
                    "retailer_name": name,
                    "current_median": cm,
                    "previous_median": pm,
                    "delta_pct": delta_pct,
                })
        return results


# Module-level singleton
_dispatcher: Optional[PriceAlertDispatcher] = None


def get_price_alert_dispatcher() -> PriceAlertDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = PriceAlertDispatcher()
    return _dispatcher
