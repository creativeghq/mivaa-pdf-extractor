"""Signals corpus cron (#462). x-cron-secret only; the corpus is operator-owned and global."""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from app.services.core.supabase_client import get_supabase_client
from app.services.integrations import signals_corpus_service as corpus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/signals", tags=["signals"])


def _verify_cron_secret(x_cron_secret: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("CRON_SECRET") or os.getenv("PRICE_MONITORING_CRON_SECRET") or ""
    if not expected:
        raise HTTPException(status_code=500, detail="CRON_SECRET not configured")
    if not hmac.compare_digest((x_cron_secret or "").encode(), expected.encode()):
        raise HTTPException(status_code=403, detail="Invalid x-cron-secret")


def _core_enabled() -> bool:
    """Fails closed: no row, or a read error, is not an enabled module."""
    try:
        res = get_supabase_client().client.table("modules").select("enabled").eq("slug", "signals-core").maybe_single().execute()
    except Exception as e:  # noqa: BLE001 — refusing to run is the safe answer
        logger.warning(f"signals cron: module read failed, skipping: {e}")
        return False
    row = res.data if res else None
    return bool(row and row.get("enabled"))


@router.post("/cron-corpus")
async def cron_corpus(
    resolve_limit: int = Query(8, ge=0, le=50),
    enumerate_limit: int = Query(10, ge=0, le=50),
    _: None = Depends(_verify_cron_secret),
):
    """Discover job boards for due domains, then read due boards. ~120s wall clock, inside the edge's 150s."""
    if not await asyncio.to_thread(_core_enabled):
        return {"success": False, "skipped": "module_disabled", "module": "signals-core"}
    resolution = await corpus.run_resolution(resolve_limit) if resolve_limit else {"checked": 0}
    enumeration = await corpus.run_enumeration(enumerate_limit) if enumerate_limit else {"due": 0}
    return {"success": True, "resolution": resolution, "enumeration": enumeration}
