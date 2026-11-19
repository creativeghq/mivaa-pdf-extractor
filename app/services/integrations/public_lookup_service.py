"""Public-tools quota + cache layer."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

ANONYMOUS_DAILY_QUOTA = 2
CACHE_TTL_HOURS = 24

#: Outcomes that spend a slot. `pending` is a reservation taken before the upstream call,
#: so two concurrent requests cannot both pass a count that neither has written to yet.
#: A failed upstream does not spend a slot: the reservation is moved off `pending`.
COUNTED_OUTCOMES = ("success", "pending")

#: The scans this budget meters. The table is shared with the edge `public-project-plan`
#: (`kitchen_lead`, `project_plan`), whose rows must not spend a visitor's scans here.
COUNTED_SCAN_TYPES = ("price", "mention", "product")


class QuotaUnavailable(RuntimeError):
    """The quota could not be read or reserved. An unanswerable count is not a zero."""


@dataclass
class QuotaStatus:
    allowed: bool
    used: int
    limit: int
    remaining: int
    reset_at: datetime  # UTC timestamp for when oldest scan ages out


@dataclass
class Reservation:
    id: str
    created_at: str


def normalize_query(text: str) -> str:
    """Lower-case + collapse whitespace + strip — so 'BLUM Cabinet ' and
    'blum cabinet' hash to the same cache key."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def query_hash(
    scan_type: str,
    query: str,
    country_code: Optional[str] = None,
    aliases: Optional[Iterable[str]] = None,
) -> str:
    """Stable hash for cache keys + log dedupe. Aliases are order- and case-insensitive."""
    parts = [scan_type, normalize_query(query), (country_code or "").upper()]
    alias_keys = sorted({normalize_query(a) for a in (aliases or []) if a and a.strip()})
    if alias_keys:
        parts.append(",".join(alias_keys))
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def check_quota(
    *,
    ip_address: Optional[str],
    user_id: Optional[str],
    created_up_to: Optional[str] = None,
) -> QuotaStatus:
    """Return current quota usage for this IP (or user_id). Raises QuotaUnavailable.

    `created_up_to` counts only rows created at or before that instant — a reservation
    re-checks against the rows that were ahead of it, so the earliest of two racing
    requests wins rather than both losing.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=24)

    if not user_id and not ip_address:
        return QuotaStatus(
            allowed=False,
            used=ANONYMOUS_DAILY_QUOTA,
            limit=ANONYMOUS_DAILY_QUOTA,
            remaining=0,
            reset_at=since + timedelta(hours=24),
        )

    try:
        sb = get_supabase_client().client
        # cache_hit rows are free (no upstream spend), so they never count.
        q = (
            sb.table("public_lookup_log")
            .select("created_at", count="exact")
            .in_("outcome", list(COUNTED_OUTCOMES))
            .in_("scan_type", list(COUNTED_SCAN_TYPES))
            .eq("cache_hit", False)
            .gte("created_at", since.isoformat())
        )
        if created_up_to:
            q = q.lte("created_at", created_up_to)
        q = q.eq("user_id", user_id) if user_id else q.eq("ip_address", ip_address)
        resp = q.order("created_at", desc=False).execute()
        rows = resp.data or []
        used = resp.count if resp.count is not None else len(rows)
    except Exception as e:
        logger.warning(f"quota check failed — refusing (fail closed): {e}")
        raise QuotaUnavailable(str(e)) from e

    limit = ANONYMOUS_DAILY_QUOTA
    remaining = max(0, limit - used)

    reset_at = datetime.now(timezone.utc) + timedelta(hours=24)
    if rows:
        try:
            oldest = rows[0]["created_at"]
            reset_at = datetime.fromisoformat(str(oldest).replace("Z", "+00:00")) + timedelta(hours=24)
        except Exception:
            pass

    return QuotaStatus(
        allowed=remaining > 0,
        used=used,
        limit=limit,
        remaining=remaining,
        reset_at=reset_at,
    )


def reserve_scan(
    *,
    scan_type: str,
    ip_address: Optional[str],
    user_id: Optional[str],
    qhash: str,
    query_text: str,
    user_agent: Optional[str] = None,
) -> Reservation:
    """Insert the `pending` row that holds one quota slot. Raises QuotaUnavailable."""
    try:
        sb = get_supabase_client().client
        resp = sb.table("public_lookup_log").insert({
            "scan_type": scan_type,
            "ip_address": ip_address,
            "user_id": user_id,
            "query_hash": qhash,
            "query_text": query_text[:500],
            "cache_hit": False,
            "upstream_cost_usd": 0,
            "latency_ms": 0,
            "outcome": "pending",
            "user_agent": (user_agent or "")[:500] or None,
        }).execute()
        row = (resp.data or [None])[0] if isinstance(resp.data, list) else resp.data
    except Exception as e:
        logger.warning(f"quota reservation insert failed — refusing (fail closed): {e}")
        raise QuotaUnavailable(str(e)) from e
    if not isinstance(row, dict) or not row.get("id") or not row.get("created_at"):
        logger.warning("quota reservation insert returned no row — refusing (fail closed)")
        raise QuotaUnavailable("reservation insert returned no row")
    return Reservation(id=str(row["id"]), created_at=str(row["created_at"]))


def finalize_scan(
    reservation: Reservation,
    *,
    outcome: str,
    upstream_cost_usd: float = 0,
    latency_ms: int = 0,
    error_message: Optional[str] = None,
) -> bool:
    """Move a reservation to its final outcome. On failure the row stays `pending`, which
    keeps counting — the visitor loses a slot rather than the platform losing the cap."""
    try:
        sb = get_supabase_client().client
        sb.table("public_lookup_log").update({
            "outcome": outcome,
            "upstream_cost_usd": float(upstream_cost_usd or 0),
            "latency_ms": int(latency_ms or 0),
            "error_message": (error_message or "")[:500] or None,
        }).eq("id", reservation.id).execute()
        return True
    except Exception as e:
        logger.warning(f"public_lookup_log reservation {reservation.id} not finalised ({outcome}): {e}")
        return False


def read_cache(*, scan_type: str, qhash: str) -> Optional[dict[str, Any]]:
    sb = get_supabase_client().client
    try:
        resp = (
            sb.table("public_lookup_cache")
            .select("result, expires_at, hit_count")
            .eq("query_hash", qhash)
            .eq("scan_type", scan_type)
            .maybe_single()
            .execute()
        )
        if not resp or not resp.data:
            return None
        expires_at = resp.data.get("expires_at")
        if expires_at:
            exp_dt = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
            if exp_dt < datetime.now(timezone.utc):
                return None
        # Bump hit count async (best-effort, don't block on this)
        try:
            sb.table("public_lookup_cache").update({
                "hit_count": (resp.data.get("hit_count") or 0) + 1,
            }).eq("query_hash", qhash).eq("scan_type", scan_type).execute()
        except Exception:
            pass
        return resp.data["result"]
    except Exception as e:
        logger.debug(f"cache read failed: {e}")
        return None


def write_cache(*, scan_type: str, qhash: str, result: dict[str, Any]) -> None:
    sb = get_supabase_client().client
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=CACHE_TTL_HOURS)).isoformat()
    try:
        sb.table("public_lookup_cache").upsert({
            "query_hash": qhash,
            "scan_type": scan_type,
            "result": result,
            "expires_at": expires_at,
            "hit_count": 0,
        }, on_conflict="query_hash,scan_type").execute()
    except Exception as e:
        logger.warning(f"cache write failed: {e}")


def log_scan(
    *,
    scan_type: str,
    ip_address: Optional[str],
    user_id: Optional[str],
    qhash: str,
    query_text: str,
    cache_hit: bool,
    upstream_cost_usd: float,
    latency_ms: int,
    outcome: str,
    error_message: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> bool:
    """Analytics row. Never the anonymous quota counter — that is `reserve_scan`."""
    sb = get_supabase_client().client
    try:
        sb.table("public_lookup_log").insert({
            "scan_type": scan_type,
            "ip_address": ip_address,
            "user_id": user_id,
            "query_hash": qhash,
            "query_text": query_text[:500],
            "cache_hit": cache_hit,
            "upstream_cost_usd": float(upstream_cost_usd or 0),
            "latency_ms": int(latency_ms or 0),
            "outcome": outcome,
            "error_message": (error_message or "")[:500] or None,
            "user_agent": (user_agent or "")[:500] or None,
        }).execute()
        return True
    except Exception as e:
        logger.warning(f"public_lookup_log insert failed: {e}")
        return False
