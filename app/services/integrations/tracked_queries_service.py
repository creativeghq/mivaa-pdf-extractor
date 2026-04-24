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
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.perplexity_price_search_service import (
    get_perplexity_price_search_service,
    PriceHit,
    PriceSearchResult,
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
    ) -> Dict[str, Any]:
        """Insert a new tracked query + run the first refresh synchronously so
        the caller gets initial results in the same POST response."""
        row = {
            "api_key_id": api_key_id,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "search_query": search_query,
            "dimensions": dimensions,
            "country_code": country_code,
            "manufacturer": manufacturer,
            "preferred_retailer_domains": preferred_retailer_domains,
            "refresh_interval_hours": max(1, min(refresh_interval_hours, 720)),
        }
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

        # Option 2: domain pinning. If the caller has saved preferred retailer
        # domains, Perplexity's search_domain_filter forces those to be probed.
        result: PriceSearchResult = await self.search.search_prices(
            product_name=tq.get("search_query") or "",
            dimensions=tq.get("dimensions"),
            country_code=tq.get("country_code"),
            limit=20,
            user_id=tq.get("user_id"),
            workspace_id=tq.get("workspace_id"),
            preferred_retailer_domains=tq.get("preferred_retailer_domains") or None,
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        if not result.success:
            self.supabase.client.table("tracked_queries").update({
                "last_error": result.error,
                "updated_at": now_iso,
            }).eq("id", tracking_id).execute()
            return {"status": "error", "error": result.error, "credits_used": result.credits_used}

        # Persist result rows — one per retailer, grouped by refresh_run_id
        refresh_run_id = str(uuid4())
        if result.hits:
            rows = [
                {
                    "tracked_query_id": tracking_id,
                    "refresh_run_id": refresh_run_id,
                    # Map engine source (perplexity|dataforseo) to DB enum
                    "source": (
                        "dataforseo_shopping" if getattr(h, "source", "perplexity") == "dataforseo"
                        else "perplexity_web_search"
                    ),
                    "retailer_name": h.retailer_name,
                    "product_url": h.product_url,
                    "price": float(h.price) if h.price is not None else None,
                    "currency": h.currency,
                    "price_unit": h.price_unit or "m2",
                    "availability": h.availability,
                    "city": h.city,
                    "ships_from_abroad": bool(h.ships_from_abroad),
                    "notes": h.notes,
                    "scraped_at": now_iso,
                }
                for h in result.hits
            ]
            try:
                self.supabase.client.table("tracked_query_price_history").insert(rows).execute()
            except Exception as e:
                logger.warning(f"Failed to insert tracked_query_price_history rows: {e}")

        # Update counters on the tracked_query row
        prev_total = int(tq.get("total_credits_used") or 0)
        self.supabase.client.table("tracked_queries").update({
            "last_refreshed_at": now_iso,
            "last_refresh_credits_used": result.credits_used,
            "total_credits_used": prev_total + result.credits_used,
            "last_error": None,
            "updated_at": now_iso,
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
            .select("scraped_at, refresh_run_id, retailer_name, product_url, price, currency, price_unit, availability, city")
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
