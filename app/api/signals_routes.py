"""Signals corpus cron (#462). x-cron-secret only; the corpus is operator-owned and global."""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app.dependencies import get_workspace_context, require_workspace_capability
from app.schemas.auth import WorkspaceContext
from app.services.core.claude_tool_call import ToolCallNotReturned
from app.services.core.supabase_client import get_supabase_client
from app.services.integrations import described_signals_service as described
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


# ── Described signals (#499) ───────────────────────────────────────────────

_DESCRIBED_MODULE = "signals-described"
_RATE_WINDOW_S = 3600
_RATE_MAX = 40
_recent: Dict[str, Deque[float]] = defaultdict(deque)


class DescribeRequest(BaseModel):
    description: str = Field(..., min_length=10, max_length=1000)


class BacktestRequest(BaseModel):
    description: str = Field(..., min_length=10, max_length=1000)
    spec: Dict[str, Any]
    crm_only: bool = False


async def _described_caller(ctx: WorkspaceContext) -> str:
    """crm.view in the workspace AND the module entitled — the same two gates signals_save applies."""
    ws = str(ctx.workspace_id)
    await require_workspace_capability(ctx.user_id, ws, "crm.view")
    entitled = await asyncio.to_thread(lambda: get_supabase_client().client.rpc(
        "is_workspace_entitled", {"p_workspace_id": ws, "p_module_slug": _DESCRIBED_MODULE}).execute().data)
    if entitled is not True:
        raise HTTPException(status_code=402, detail="Described Signals is not enabled for this workspace.")
    window = _recent[ws]
    now = time.monotonic()
    while window and now - window[0] > _RATE_WINDOW_S:
        window.popleft()
    if len(window) >= _RATE_MAX:
        raise HTTPException(status_code=429, detail="Too many drafts and test runs this hour. Try again later.")
    window.append(now)
    return ws


@router.post("/described/compile")
async def described_compile(body: DescribeRequest, ctx: WorkspaceContext = Depends(get_workspace_context)):
    """Draft matching rules from a plain-language description. Free; nothing is saved."""
    ws = await _described_caller(ctx)
    try:
        return {"success": True, **await described.compile_description(body.description, workspace_id=ws, user_id=str(ctx.user_id))}
    except (ToolCallNotReturned, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"Could not draft rules from that description: {str(e)[:200]}")


@router.post("/described/backtest")
async def described_backtest(body: BacktestRequest, ctx: WorkspaceContext = Depends(get_workspace_context)):
    """Run drafted rules over the last 30 days: candidate count, a judged sample and a weekly cost estimate."""
    ws = await _described_caller(ctx)
    try:
        result = await described.backtest(body.description, body.spec, crm_only=body.crm_only,
                                          workspace_id=ws, user_id=str(ctx.user_id))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)[:200])
    return {"success": True, **result}


@router.post("/cron-described")
async def cron_described(
    limit: int = Query(20, ge=1, le=100),
    _: None = Depends(_verify_cron_secret),
):
    """Embed the observation queue, then check every due described signal. ~120s, inside the edge's 150s."""
    return {"success": True, **await described.run_described(limit)}
