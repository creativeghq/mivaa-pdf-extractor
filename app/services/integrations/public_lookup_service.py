"""
Public-tools quota + cache layer.

Two responsibilities:
  1. Quota check — count rows in `public_lookup_log` for an IP (or user_id) in
     the last 24h. Returns remaining + limit + reset_at.
  2. Result cache — read/write `public_lookup_cache` keyed on (query_hash,
     scan_type) with a 24h TTL. Identical queries skip the upstream APIs.

The quota is COMBINED across scan types (2 total scans/day/IP).
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

ANONYMOUS_DAILY_QUOTA = 2
CACHE_TTL_HOURS = 24


@dataclass
class QuotaStatus:
    allowed: bool
    used: int
    limit: int
    remaining: int
    reset_at: datetime  # UTC timestamp for when oldest scan ages out


def normalize_query(text: str) -> str:
    """Lower-case + collapse whitespace + strip — so 'BLUM Cabinet ' and
    'blum cabinet' hash to the same cache key."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def query_hash(scan_type: str, query: str, country_code: Optional[str] = None) -> str:
    """Stable hash for cache keys + log dedupe."""
    parts = [scan_type, normalize_query(query), (country_code or "").upper()]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def check_quota(*, ip_address: Optional[str], user_id: Optional[str]) -> QuotaStatus:
    """Return current quota usage for this IP (or user_id).

    Counts ONLY successful scans — rate-limited / captcha-failed / errored
    attempts don't burn quota. A user who hits the limit then refreshes the
    page should not be punished for failed bot-check attempts.
    """
    sb = get_supabase_client().client
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    since_iso = since.isoformat()

    q = sb.table("public_lookup_log").select("created_at", count="exact").eq("outcome", "success").gte("created_at", since_iso)
    if user_id:
        q = q.eq("user_id", user_id)
    elif ip_address:
        q = q.eq("ip_address", ip_address)
    else:
        # No identity — treat as exhausted to avoid abuse
        return QuotaStatus(
            allowed=False,
            used=ANONYMOUS_DAILY_QUOTA,
            limit=ANONYMOUS_DAILY_QUOTA,
            remaining=0,
            reset_at=since + timedelta(hours=24),
        )

    try:
        resp = q.order("created_at", desc=False).execute()
        rows = resp.data or []
        used = resp.count if resp.count is not None else len(rows)
    except Exception as e:
        logger.warning(f"quota check failed (treat as 0 used): {e}")
        rows = []
        used = 0

    limit = ANONYMOUS_DAILY_QUOTA
    remaining = max(0, limit - used)
    allowed = remaining > 0

    # reset_at = when the oldest scan in the window ages out
    if rows:
        try:
            oldest = rows[0]["created_at"]
            oldest_dt = datetime.fromisoformat(str(oldest).replace("Z", "+00:00"))
            reset_at = oldest_dt + timedelta(hours=24)
        except Exception:
            reset_at = datetime.now(timezone.utc) + timedelta(hours=24)
    else:
        reset_at = datetime.now(timezone.utc) + timedelta(hours=24)

    return QuotaStatus(
        allowed=allowed,
        used=used,
        limit=limit,
        remaining=remaining,
        reset_at=reset_at,
    )


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
) -> None:
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
    except Exception as e:
        logger.warning(f"public_lookup_log insert failed: {e}")
