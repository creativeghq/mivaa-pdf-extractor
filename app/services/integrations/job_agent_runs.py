"""
Job Research — background_agents bookkeeping helpers.

Wires the job-research module into the platform's existing background-agents
framework so each tracked_job appears in `/admin/background-agents` alongside
the other background agents (product enrichment, material tagger, etc.).

We do NOT use the `background-agent-runner` edge function as the executor
(our cron at :45 is the orchestrator). We just write the rows so the admin
UI can render runs + logs uniformly.

Mapping:
  background_agents row    ←→ tracked_jobs row (1:1, via tracked_jobs.background_agent_id)
  agent_runs row           ←→ one refresh of one tracked_job
  agent_run_logs rows      ←→ per-source progress messages (DataForSEO / Sonar / Firecrawl / Classifier / Persist)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_background_agent_for_tracked_job(
    *,
    tracked_job_id: str,
    user_id: str,
    workspace_id: Optional[str],
    label: str,
    keywords: List[str],
    refresh_interval_hours: int,
) -> Optional[str]:
    """Insert a background_agents row mirroring this tracked_job. Returns the new id, or None on failure."""
    try:
        sb = get_supabase_client().client
        res = sb.table("background_agents").insert({
            "name": f"Job Research — {label[:140]}",
            "description": f"Background job-discovery for keywords: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}",
            "agent_type": "job-research",
            "trigger_type": "schedule",
            "schedule": f"every {refresh_interval_hours}h",
            "model": "claude-haiku-4-5-20251001",
            "config": {
                "tracked_job_id": tracked_job_id,
                "label": label,
                "keywords": keywords,
                "refresh_interval_hours": refresh_interval_hours,
            },
            "enabled": True,
            "workspace_id": workspace_id,
            "created_by": user_id,
        }).execute()
        new_id = (res.data or [{}])[0].get("id")
        if new_id:
            sb.table("tracked_jobs").update({"background_agent_id": new_id}).eq("id", tracked_job_id).execute()
        return new_id
    except Exception as e:
        logger.warning(f"job-agent-runs: create_background_agent failed: {e}")
        return None


def deactivate_background_agent(agent_id: Optional[str]) -> None:
    if not agent_id:
        return
    try:
        sb = get_supabase_client().client
        sb.table("background_agents").update({"enabled": False}).eq("id", agent_id).execute()
    except Exception as e:
        logger.warning(f"job-agent-runs: deactivate failed: {e}")


def reactivate_background_agent(agent_id: Optional[str]) -> None:
    if not agent_id:
        return
    try:
        sb = get_supabase_client().client
        sb.table("background_agents").update({"enabled": True}).eq("id", agent_id).execute()
    except Exception as e:
        logger.warning(f"job-agent-runs: reactivate failed: {e}")


def start_run(
    *,
    background_agent_id: Optional[str],
    workspace_id: Optional[str],
    user_id: Optional[str],
    refresh_run_id: str,
    triggered_by: str = "schedule",
) -> Optional[str]:
    """Insert an agent_runs row in 'running' state. Returns the agent_runs.id."""
    if not background_agent_id:
        return None
    try:
        sb = get_supabase_client().client
        res = sb.table("agent_runs").insert({
            "agent_id": background_agent_id,
            "status": "running",
            "triggered_by": triggered_by,
            "input_data": {"refresh_run_id": refresh_run_id},
            "model_used": "claude-haiku-4-5-20251001",
            "started_at": _utc_iso(),
            "last_heartbeat": _utc_iso(),
            "workspace_id": workspace_id,
            "initiated_by_user": user_id,
        }).execute()
        return (res.data or [{}])[0].get("id")
    except Exception as e:
        logger.warning(f"job-agent-runs: start_run failed: {e}")
        return None


def append_log(
    *,
    run_id: Optional[str],
    level: str,                 # debug | info | warning | error
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    if not run_id:
        return
    try:
        sb = get_supabase_client().client
        sb.table("agent_run_logs").insert({
            "run_id": run_id,
            "level": level,
            "message": message[:480],
            "data": data or {},
        }).execute()
        # Lightweight heartbeat — every log update bumps it
        sb.table("agent_runs").update({"last_heartbeat": _utc_iso()}).eq("id", run_id).execute()
    except Exception as e:
        logger.debug(f"job-agent-runs: append_log failed: {e}")


def complete_run(
    *,
    run_id: Optional[str],
    output_data: Dict[str, Any],
    duration_ms: Optional[int] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    credits_debited: int = 0,
) -> None:
    if not run_id:
        return
    try:
        sb = get_supabase_client().client
        sb.table("agent_runs").update({
            "status": "completed",
            "output_data": output_data,
            "completed_at": _utc_iso(),
            "duration_ms": duration_ms,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "credits_debited": int(credits_debited or 0),
        }).eq("id", run_id).execute()
        # Update background_agents.last_run_*
        bg_res = sb.table("agent_runs").select("agent_id").eq("id", run_id).maybe_single().execute()
        bg_id = (bg_res.data if bg_res else None) or {}
        if bg_id.get("agent_id"):
            sb.table("background_agents").update({
                "last_run_at": _utc_iso(),
                "last_run_status": "completed",
                "run_count": _bump_run_count(bg_id["agent_id"]),
            }).eq("id", bg_id["agent_id"]).execute()
    except Exception as e:
        logger.warning(f"job-agent-runs: complete_run failed: {e}")


def fail_run(
    *,
    run_id: Optional[str],
    error_message: str,
    duration_ms: Optional[int] = None,
) -> None:
    if not run_id:
        return
    try:
        sb = get_supabase_client().client
        sb.table("agent_runs").update({
            "status": "failed",
            "error_message": (error_message or "")[:480],
            "completed_at": _utc_iso(),
            "duration_ms": duration_ms,
        }).eq("id", run_id).execute()
        bg_res = sb.table("agent_runs").select("agent_id").eq("id", run_id).maybe_single().execute()
        bg_id = (bg_res.data if bg_res else None) or {}
        if bg_id.get("agent_id"):
            sb.table("background_agents").update({
                "last_run_at": _utc_iso(),
                "last_run_status": "failed",
            }).eq("id", bg_id["agent_id"]).execute()
    except Exception as e:
        logger.warning(f"job-agent-runs: fail_run failed: {e}")


def _bump_run_count(agent_id: str) -> int:
    """Read current run_count and return +1. Best effort — not atomic."""
    try:
        sb = get_supabase_client().client
        res = sb.table("background_agents").select("run_count").eq("id", agent_id).maybe_single().execute()
        current = ((res.data if res else None) or {}).get("run_count") or 0
        return int(current) + 1
    except Exception:
        return 1
