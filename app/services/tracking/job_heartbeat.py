"""
Background heartbeat for in-flight PDF jobs.

Stage transitions update `background_jobs.last_heartbeat` opportunistically,
but a job can sit inside one stage (e.g. waiting on Stage 0 vision discovery)
for many minutes. Without a periodic heartbeat the auto-recovery cron cannot
distinguish "still working" from "process died." This helper writes a
heartbeat every `JOB_HEARTBEAT_INTERVAL_SECONDS` while the orchestrator is
running, regardless of stage progress.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class JobHeartbeat:
    """Periodic heartbeat writer for background_jobs.last_heartbeat.

    Use as an async context manager around the job orchestrator body:

        async with JobHeartbeat(job_id, supabase_client):
            await process_document_with_discovery(...)
    """

    def __init__(self, job_id: str, supabase_client, interval_seconds: Optional[int] = None):
        self.job_id = job_id
        self.supabase = supabase_client
        if interval_seconds is None:
            from app.config import get_settings
            interval_seconds = get_settings().job_heartbeat_interval_seconds
        self.interval_seconds = max(15, int(interval_seconds))
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def _write_once(self) -> None:
        try:
            now = datetime.now(timezone.utc).isoformat()
            await asyncio.to_thread(
                lambda: self.supabase.client.table('background_jobs').update({
                    'last_heartbeat': now,
                }).eq('id', self.job_id).execute()
            )
        except Exception as e:
            # Heartbeat failures are not fatal — the orchestrator continues.
            # Auto-recovery cron uses last_heartbeat ± stuck_threshold so a
            # short outage just looks like one missed heartbeat.
            logger.debug(f"heartbeat write failed for {self.job_id}: {e}")

    async def _loop(self) -> None:
        # Write one immediately so dashboards show "started" instantly.
        await self._write_once()
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                await self._write_once()

    async def __aenter__(self) -> "JobHeartbeat":
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name=f"heartbeat-{self.job_id}")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        # Final heartbeat so the cron sees a fresh timestamp on natural completion.
        await self._write_once()
