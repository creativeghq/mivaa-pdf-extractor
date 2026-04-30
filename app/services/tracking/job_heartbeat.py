"""
Background heartbeat for in-flight PDF jobs.

Stage transitions update `background_jobs.last_heartbeat` opportunistically,
but a job can sit inside one stage (e.g. waiting on Stage 0 vision discovery)
for many minutes. Without a periodic heartbeat the auto-recovery cron cannot
distinguish "still working" from "process died." This helper writes a
heartbeat every `JOB_HEARTBEAT_INTERVAL_SECONDS` while the orchestrator is
running, regardless of stage progress.

Heartbeat runs on a real OS thread (not asyncio.create_task) so even if
the orchestrator's event loop is blocked by a long synchronous call
(PyMuPDF page processing, sync HF SDK, big GC pause) the heartbeat keeps
firing. Otherwise a CPU-bound stage looks "stuck" to the auto-recovery
cron and triggers unnecessary recovery attempts.
"""

import asyncio
import logging
import threading
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
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _write_once(self) -> None:
        """Synchronous heartbeat write — runs on the heartbeat thread, not the asyncio loop."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            self.supabase.client.table('background_jobs').update({
                'last_heartbeat': now,
            }).eq('id', self.job_id).execute()
        except Exception as e:
            # Heartbeat failures are not fatal — the orchestrator continues.
            # Auto-recovery cron uses last_heartbeat ± stuck_threshold so a
            # short outage just looks like one missed heartbeat.
            logger.debug(f"heartbeat write failed for {self.job_id}: {e}")

    def _thread_loop(self) -> None:
        """Threaded heartbeat loop. Runs independently of the asyncio event loop."""
        # Write one immediately so dashboards show "started" instantly.
        self._write_once()
        while not self._stop_event.is_set():
            # Event.wait returns True if set, False if timeout — we use it
            # as a non-busy sleep that can be interrupted by stop().
            if self._stop_event.wait(timeout=self.interval_seconds):
                break
            self._write_once()
        # Final heartbeat so the cron sees a fresh timestamp on natural completion.
        self._write_once()

    async def __aenter__(self) -> "JobHeartbeat":
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._thread_loop,
            name=f"heartbeat-{self.job_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._thread is not None:
            # Wait briefly for the thread to flush its final write.
            # Don't block forever — daemon=True so it'll die with the process.
            await asyncio.to_thread(self._thread.join, 5.0)
