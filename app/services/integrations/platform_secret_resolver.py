"""Platform secret resolver — env-first, DB-fallback."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


@dataclass
class ResolvedSecret:
    key: str
    value: Optional[str]
    source: str  # 'env' | 'db' | 'default' | 'missing'


_CACHE_TTL_S = 30.0
_cache: dict[str, tuple[Optional[str], Optional[str], float]] = {}
_lock = threading.Lock()


def _load_row(key: str) -> tuple[Optional[str], Optional[str]]:
    """Return (value, default_value) from platform_secrets, cached for 30s."""
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and (now - hit[2]) < _CACHE_TTL_S:
            return hit[0], hit[1]

    value: Optional[str] = None
    default: Optional[str] = None
    try:
        sb = get_supabase_client().client
        resp = (
            sb.table("platform_secrets")
            .select("value, default_value")
            .eq("key", key)
            .maybe_single()
            .execute()
        )
        if resp and resp.data:
            value = resp.data.get("value")
            default = resp.data.get("default_value")
    except Exception as e:
        logger.debug(f"platform_secrets read failed for {key}: {e}")

    with _lock:
        _cache[key] = (value, default, now)
    return value, default


def resolve_secret(key: str) -> ResolvedSecret:
    """Env-first, DB-fallback. Returns the source so callers can log it."""
    env_val = os.getenv(key)
    if env_val and env_val.strip():
        return ResolvedSecret(key=key, value=env_val, source="env")

    db_val, default_val = _load_row(key)
    if db_val and db_val.strip():
        return ResolvedSecret(key=key, value=db_val, source="db")
    if default_val and default_val.strip():
        return ResolvedSecret(key=key, value=default_val, source="default")
    return ResolvedSecret(key=key, value=None, source="missing")


def require_secret(key: str) -> str:
    r = resolve_secret(key)
    if not r.value:
        raise RuntimeError(
            f'Secret "{key}" is not configured. Set it via env var or '
            f'/admin/operations → Keys.'
        )
    return r.value


def invalidate_cache(key: Optional[str] = None) -> None:
    with _lock:
        if key is None:
            _cache.clear()
        else:
            _cache.pop(key, None)
