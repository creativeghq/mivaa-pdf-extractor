"""
Service layer for API-consumer price tracking.

Backs the public /api/v1/prices/track endpoints. External projects use
their api_keys Bearer token to register tracked queries (product name +
country + refresh cadence), our cron refreshes them on schedule via
Perplexity, and they poll or receive results. Deleting the api_key
cascades out every tracked_queries row and its price history.

Kept separate from the platform's internal price monitoring flow
(price_monitoring_products + competitor_sources) so the external data
model can evolve independently without touching catalog products.
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


def _domain_of(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = re.match(r"^https?://([^/]+)", url.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    host = m.group(1).lower()
    return host[4:] if host.startswith("www.") else host

from app.services.core.supabase_client import get_supabase_client
from app.modules.price_monitoring_notifications.service import (
    get_price_alert_dispatcher,
)
from app.services.integrations.perplexity_price_search_service import (
    get_perplexity_price_search_service,
    PriceHit,
    PriceSearchResult,
)
from app.services.integrations.product_identity_service import (
    get_product_identity_service,
    QueryFacets,
)

logger = logging.getLogger(__name__)


class TrackedQueriesService:
    """Owns create/read/update/delete + refresh for tracked_queries rows."""

    def __init__(self) -> None:
        self.supabase = get_supabase_client()
        self.search = get_perplexity_price_search_service()

    # ────────── CRUD ──────────

    async def create(
        self,
        *,
        api_key_id: str,
        user_id: Optional[str],
        workspace_id: Optional[str],
        search_query: str,
        dimensions: Optional[str] = None,
        country_code: Optional[str] = None,
        manufacturer: Optional[str] = None,
        preferred_retailer_domains: Optional[List[str]] = None,
        refresh_interval_hours: int = 24,
        verify_prices: bool = True,
        alert_channels: Optional[List[str]] = None,
        alert_on_price_drop: Optional[bool] = None,
        alert_on_new_retailer: Optional[bool] = None,
        alert_on_promo: Optional[bool] = None,
        alert_webhook_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert a new tracked query + run the first refresh synchronously so
        the caller gets initial results in the same POST response.

        Extracts query facets (Haiku) once at create time and caches them on
        the tracked_queries.query_facets column. Every subsequent refresh
        reuses the cached facets — we don't re-pay for decomposition on a
        query that already has a frozen facet signature.
        """
        # Facet extraction runs before the insert so the row stores them from
        # day one. Facet failures are non-fatal — the row is still created
        # and refresh() falls back to extracting on-demand.
        facet_query = search_query
        if dimensions:
            facet_query = f"{search_query} {dimensions}"
        facets: Optional[QueryFacets] = None
        try:
            identity_svc = get_product_identity_service()
            facets = await identity_svc.extract_query_facets(
                facet_query, manufacturer_hint=manufacturer
            )
        except Exception as e:
            logger.warning(f"Facet extraction failed on create (non-fatal): {e}")

        row: Dict[str, Any] = {
            "api_key_id": api_key_id,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "search_query": search_query,
            "dimensions": dimensions,
            "country_code": country_code,
            "manufacturer": manufacturer,
            "preferred_retailer_domains": preferred_retailer_domains,
            "refresh_interval_hours": max(1, min(refresh_interval_hours, 720)),
            "verify_prices": bool(verify_prices),
            "query_facets": facets.to_dict() if facets else None,
        }
        # Alert opt-ins are passed through only when the caller specified them
        # — None means "let the column default decide" so we don't accidentally
        # disable alerts for callers who didn't know they existed.
        if alert_channels is not None:
            row["alert_channels"] = alert_channels
        if alert_on_price_drop is not None:
            row["alert_on_price_drop"] = bool(alert_on_price_drop)
        if alert_on_new_retailer is not None:
            row["alert_on_new_retailer"] = bool(alert_on_new_retailer)
        if alert_on_promo is not None:
            row["alert_on_promo"] = bool(alert_on_promo)
        if alert_webhook_url is not None:
            row["alert_webhook_url"] = alert_webhook_url
        res = self.supabase.client.table("tracked_queries").insert(row).execute()
        created = (res.data or [{}])[0]
        tracking_id = created.get("id")
        if not tracking_id:
            raise RuntimeError("Failed to create tracked_queries row")

        # Fire the initial refresh synchronously so external caller gets data back.
        await self.refresh(tracking_id, force=True)
        # Re-read to include last_refreshed_at + counters
        return await self.get(tracking_id) or created

    async def get(self, tracking_id: str) -> Optional[Dict[str, Any]]:
        res = (
            self.supabase.client.table("tracked_queries")
            .select("*")
            .eq("id", tracking_id)
            .maybe_single()
            .execute()
        )
        return (res.data if res else None) or None

    async def list_for_api_key(
        self, api_key_id: str, include_inactive: bool = False, limit: int = 100
    ) -> List[Dict[str, Any]]:
        q = (
            self.supabase.client.table("tracked_queries")
            .select("*")
            .eq("api_key_id", api_key_id)
            .order("created_at", desc=True)
            .limit(max(1, min(limit, 500)))
        )
        if not include_inactive:
            q = q.eq("is_active", True)
        res = q.execute()
        return res.data or []

    async def update(
        self,
        tracking_id: str,
        *,
        refresh_interval_hours: Optional[int] = None,
        country_code: Optional[str] = None,
        preferred_retailer_domains: Optional[List[str]] = None,
        dimensions: Optional[str] = None,
        manufacturer: Optional[str] = None,
        verify_prices: Optional[bool] = None,
        alert_channels: Optional[List[str]] = None,
        alert_on_price_drop: Optional[bool] = None,
        alert_on_new_retailer: Optional[bool] = None,
        alert_on_promo: Optional[bool] = None,
        alert_webhook_url: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        updates: Dict[str, Any] = {"updated_at": datetime.now(timezone.utc).isoformat()}
        if refresh_interval_hours is not None:
            updates["refresh_interval_hours"] = max(1, min(refresh_interval_hours, 720))
        if country_code is not None:
            updates["country_code"] = country_code
        if preferred_retailer_domains is not None:
            updates["preferred_retailer_domains"] = preferred_retailer_domains
        if dimensions is not None:
            updates["dimensions"] = dimensions
        if manufacturer is not None:
            updates["manufacturer"] = manufacturer
        if verify_prices is not None:
            updates["verify_prices"] = bool(verify_prices)
        if alert_channels is not None:
            updates["alert_channels"] = alert_channels
        if alert_on_price_drop is not None:
            updates["alert_on_price_drop"] = bool(alert_on_price_drop)
        if alert_on_new_retailer is not None:
            updates["alert_on_new_retailer"] = bool(alert_on_new_retailer)
        if alert_on_promo is not None:
            updates["alert_on_promo"] = bool(alert_on_promo)
        if alert_webhook_url is not None:
            updates["alert_webhook_url"] = alert_webhook_url

        # Cached query_facets are derived from search_query + dimensions +
        # manufacturer. Invalidate when any of those change so the next
        # refresh rebuilds them from the new inputs.
        if dimensions is not None or manufacturer is not None:
            updates["query_facets"] = None

        if len(updates) == 1:  # only updated_at
            return await self.get(tracking_id)

        res = (
            self.supabase.client.table("tracked_queries")
            .update(updates)
            .eq("id", tracking_id)
            .execute()
        )
        return (res.data or [None])[0]

    async def deactivate(self, tracking_id: str) -> bool:
        """Soft delete — keep history, stop refreshing."""
        res = (
            self.supabase.client.table("tracked_queries")
            .update({"is_active": False, "updated_at": datetime.now(timezone.utc).isoformat()})
            .eq("id", tracking_id)
            .execute()
        )
        return bool(res.data)

    # ────────── Refresh flow ──────────

    async def refresh(self, tracking_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Run Perplexity for this tracked query and persist results.
        Respects refresh_interval_hours unless force=True.
        Returns {status, results, error?}.
        """
        tq = await self.get(tracking_id)
        if not tq:
            return {"status": "not_found", "error": "tracking_id not found"}
        if not tq.get("is_active"):
            return {"status": "inactive", "error": "tracking is deactivated"}

        # Cadence check
        if not force:
            last_at_raw = tq.get("last_refreshed_at")
            if last_at_raw:
                last_at = datetime.fromisoformat(last_at_raw.replace("Z", "+00:00"))
                interval = timedelta(hours=int(tq.get("refresh_interval_hours") or 24))
                next_at = last_at + interval
                if datetime.now(timezone.utc) < next_at:
                    return {
                        "status": "throttled",
                        "throttle_until": next_at.isoformat(),
                        "results": await self.latest_results(tracking_id),
                    }

        # Use the cached query_facets if we have them (created on first insert).
        # Rows that predate facet caching (or whose create-time extraction
        # failed) get a one-shot extraction here and the result is persisted
        # so subsequent refreshes don't re-pay for the Haiku call.
        cached_facets = QueryFacets.from_dict(tq.get("query_facets"))
        if cached_facets is None:
            facet_query = tq.get("search_query") or ""
            if tq.get("dimensions"):
                facet_query = f"{facet_query} {tq['dimensions']}"
            try:
                identity_svc = get_product_identity_service()
                cached_facets = await identity_svc.extract_query_facets(
                    facet_query, manufacturer_hint=tq.get("manufacturer"),
                )
                if cached_facets is not None:
                    self.supabase.client.table("tracked_queries").update({
                        "query_facets": cached_facets.to_dict(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }).eq("id", tracking_id).execute()
            except Exception as e:
                logger.warning(f"Backfill facet extraction failed (non-fatal): {e}")

        # Option 2: domain pinning. If the caller has saved preferred retailer
        # domains, Perplexity's search_domain_filter forces those to be probed.
        # verify_prices controls the Firecrawl verification pass (default True).
        # First refresh = double-read verification pass to catch transient /
        # A/B-tested prices. Subsequent refreshes single-read. We flip the
        # marker AFTER a successful refresh so a crash mid-run doesn't lose
        # the chance to double-read.
        is_first_refresh = not bool(tq.get("first_refresh_verified"))

        # Pull known retailer domains from history so Perplexity can prioritize
        # finding NEW retailers instead of cycling through the same set.
        known_domains: List[str] = []
        try:
            hist = (
                self.supabase.client.table("tracked_query_price_history")
                .select("product_url")
                .eq("tracked_query_id", tracking_id)
                .order("scraped_at", desc=True)
                .limit(200)
                .execute()
            )
            seen: set = set()
            for r in hist.data or []:
                d = _domain_of(r.get("product_url"))
                if d and d not in seen:
                    seen.add(d)
                    known_domains.append(d)
        except Exception as e:
            logger.debug(f"known retailer fetch failed: {e}")
        result: PriceSearchResult = await self.search.search_prices(
            product_name=tq.get("search_query") or "",
            dimensions=tq.get("dimensions"),
            country_code=tq.get("country_code"),
            limit=20,
            user_id=tq.get("user_id"),
            workspace_id=tq.get("workspace_id"),
            preferred_retailer_domains=tq.get("preferred_retailer_domains") or None,
            verify_prices=bool(tq.get("verify_prices", True)),
            query_facets=cached_facets,
            manufacturer_hint=tq.get("manufacturer"),
            double_read=is_first_refresh and bool(tq.get("verify_prices", True)),
            known_retailer_domains=known_domains,
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        if not result.success:
            self.supabase.client.table("tracked_queries").update({
                "last_error": result.error,
                "updated_at": now_iso,
            }).eq("id", tracking_id).execute()
            return {"status": "error", "error": result.error, "credits_used": result.credits_used}

        # Persist result rows — one per retailer, grouped by refresh_run_id.
        # Each row passes through the sanity band before insert: an outlier
        # reading is still persisted but stamped is_anomaly=true so the UI
        # can show it under a yellow banner without it polluting medians.
        refresh_run_id = str(uuid4())
        rows: List[Dict[str, Any]] = []
        if result.hits:
            dispatcher = get_price_alert_dispatcher()
            for h in result.hits:
                price_val = float(h.price) if h.price is not None else None
                is_anomaly = False
                anomaly_reason: Optional[str] = None
                rolling_med: Optional[float] = None
                if price_val is not None:
                    domain = _domain_of(h.product_url)
                    verdict = dispatcher.check_sanity(
                        tracked_query_id=tracking_id,
                        retailer_domain=domain or "",
                        new_price=price_val,
                    )
                    is_anomaly = verdict.is_anomaly
                    anomaly_reason = verdict.reason if is_anomaly else None
                    rolling_med = verdict.rolling_median
                rows.append({
                    "tracked_query_id": tracking_id,
                    "refresh_run_id": refresh_run_id,
                    "source": (
                        "dataforseo_shopping" if getattr(h, "source", "perplexity") == "dataforseo"
                        else "perplexity_web_search"
                    ),
                    "retailer_name": h.retailer_name,
                    "product_url": h.product_url,
                    "price": price_val,
                    "original_price": float(h.original_price) if h.original_price is not None else None,
                    "currency": h.currency,
                    "price_unit": h.price_unit or "m2",
                    "availability": h.availability,
                    "city": h.city,
                    "ships_from_abroad": bool(h.ships_from_abroad),
                    "verified": bool(h.verified),
                    "notes": h.notes,
                    "match_kind": h.match_kind,
                    "match_score": h.match_score,
                    "match_note": h.match_note,
                    "product_title": h.product_title,
                    "scraped_at": now_iso,
                    "is_anomaly": is_anomaly,
                    "anomaly_reason": anomaly_reason,
                    "rolling_median_at_check": rolling_med,
                })
            try:
                self.supabase.client.table("tracked_query_price_history").insert(rows).execute()
            except Exception as e:
                logger.warning(f"Failed to insert tracked_query_price_history rows: {e}")
                rows = []  # don't run alert detection on failed insert

        # Alert detection runs against the just-persisted rows. Module-gated +
        # credit-metered + dedupe-protected inside the dispatcher.
        if rows:
            try:
                dispatcher = get_price_alert_dispatcher()
                candidates = dispatcher.detect_after_refresh(
                    tracked_query_id=tracking_id,
                    new_rows=rows,
                )
                if candidates:
                    fired = dispatcher.dispatch(candidates)
                    logger.info(
                        f"price-alerts: tracked_query={tracking_id} fired {fired}/{len(candidates)} alerts"
                    )
            except Exception as e:
                logger.warning(f"price-alerts: dispatch failed (non-fatal): {e}")

        # Update counters on the tracked_query row + mark first-refresh done.
        prev_total = int(tq.get("total_credits_used") or 0)
        self.supabase.client.table("tracked_queries").update({
            "last_refreshed_at": now_iso,
            "last_refresh_credits_used": result.credits_used,
            "total_credits_used": prev_total + result.credits_used,
            "last_error": None,
            "updated_at": now_iso,
            "first_refresh_verified": True,
        }).eq("id", tracking_id).execute()

        return {
            "status": "refreshed",
            "refresh_run_id": refresh_run_id,
            "credits_used": result.credits_used,
            "latency_ms": result.latency_ms,
            "results": [h.model_dump() for h in result.hits],
            "summary": result.summary,
        }

    async def latest_results(self, tracking_id: str) -> List[Dict[str, Any]]:
        """Return the most recent refresh_run's retailer rows, cheapest first."""
        latest = (
            self.supabase.client.table("tracked_query_price_history")
            .select("refresh_run_id")
            .eq("tracked_query_id", tracking_id)
            .order("scraped_at", desc=True)
            .limit(1)
            .execute()
        )
        latest_rows = latest.data or []
        if not latest_rows:
            return []
        run_id = latest_rows[0]["refresh_run_id"]
        res = (
            self.supabase.client.table("tracked_query_price_history")
            .select("*")
            .eq("tracked_query_id", tracking_id)
            .eq("refresh_run_id", run_id)
            .order("price", desc=False)
            .execute()
        )
        return res.data or []

    async def history(
        self, tracking_id: str, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """All historical price points, newest first."""
        res = (
            self.supabase.client.table("tracked_query_price_history")
            .select(
                "scraped_at, refresh_run_id, retailer_name, product_url, price, "
                "original_price, currency, price_unit, availability, city, verified, "
                "match_kind, match_score, match_note, product_title, notes, "
                "is_anomaly, anomaly_reason, rolling_median_at_check"
            )
            .eq("tracked_query_id", tracking_id)
            .order("scraped_at", desc=True)
            .limit(max(1, min(limit, 2000)))
            .execute()
        )
        return res.data or []

    async def due_for_refresh(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Used by the cron. Returns active queries whose last_refreshed_at is
        older than their refresh_interval_hours."""
        # Postgres-side filter: last_refreshed_at IS NULL OR now() - last_refreshed_at > interval
        # Hardcoded max 30 days (720h) so worst-case we still re-probe even misconfigured rows.
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=720)).isoformat()
        # We pull candidates and check interval in Python to keep the query simple.
        # For scale, move this to a Postgres function later.
        candidates = (
            self.supabase.client.table("tracked_queries")
            .select("id, last_refreshed_at, refresh_interval_hours")
            .eq("is_active", True)
            .or_(f"last_refreshed_at.is.null,last_refreshed_at.lt.{cutoff}")
            .limit(max(1, min(limit, 500)))
            .execute()
        )
        rows = candidates.data or []
        due: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        for r in rows:
            last = r.get("last_refreshed_at")
            if not last:
                due.append(r)
                continue
            last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
            interval = timedelta(hours=int(r.get("refresh_interval_hours") or 24))
            if now - last_dt >= interval:
                due.append(r)
        return due


_service: Optional[TrackedQueriesService] = None


def get_tracked_queries_service() -> TrackedQueriesService:
    global _service
    if _service is None:
        _service = TrackedQueriesService()
    return _service
