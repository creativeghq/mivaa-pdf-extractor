"""Background heartbeat for in-flight PDF jobs."""

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_KEEPALIVE_SECONDS = 240


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

    def _keep_models_warm(self) -> None:
        """Ping the PaddleOCR Modal app so its container outlives a quiet stretch of the job.

        A cold start can wedge for the whole warm-up deadline, and a job goes more than the
        container's idle window without OCR (18 min measured), so the next call paid for one.
        """
        try:
            import requests
            from app.config import get_settings
            cfg = get_settings()
            if not cfg.paddleocr_enabled or not cfg.paddleocr_modal_url:
                return
            headers = {"Authorization": f"Bearer {cfg.paddleocr_modal_api_key}"} if cfg.paddleocr_modal_api_key else {}
            requests.get(cfg.paddleocr_modal_url.rstrip("/") + "/health", headers=headers, timeout=15)
        except Exception as e:  # noqa: BLE001 - a missed ping must never touch the job
            logger.debug(f"model keep-alive ping failed for {self.job_id}: {e}")

    def _thread_loop(self) -> None:
        """Threaded heartbeat loop. Runs independently of the asyncio event loop.

        Audit fix #44: gates the per-tick write on a fresh status check. If the
        job has been marked terminal (completed/failed/cancelled) by another
        path, this loop stops writing — so a daemon thread that survives a
        hard kill can't keep refreshing last_heartbeat on a finished job and
        fool auto-recovery into thinking it's still alive.
        """
        # Write one immediately so dashboards show "started" instantly.
        self._write_once()
        last_ping = 0.0
        while not self._stop_event.is_set():
            # Event.wait returns True if set, False if timeout — we use it
            # as a non-busy sleep that can be interrupted by stop().
            if self._stop_event.wait(timeout=self.interval_seconds):
                break
            # Audit fix #44: gate write on terminal-status check.
            if self._is_job_terminal():
                logger.info(
                    f"heartbeat thread for {self.job_id} detected terminal status — stopping"
                )
                self._stop_event.set()
                break
            self._write_once()
            if time.monotonic() - last_ping >= MODEL_KEEPALIVE_SECONDS:
                last_ping = time.monotonic()
                self._keep_models_warm()
        # Final heartbeat so the cron sees a fresh timestamp on natural completion.
        # Skipped if we exited due to terminal-status detection (already final).
        if not self._is_job_terminal():
            self._write_once()

    def _is_job_terminal(self) -> bool:
        """Best-effort check for terminal job status. Defaults to False on error
        so a transient DB hiccup doesn't accidentally stop heartbeats."""
        try:
            if not self.supabase:
                return False
            resp = self.supabase.client.table("background_jobs")\
                .select("status")\
                .eq("id", self.job_id).single().execute()
            status = (resp.data or {}).get("status")
            return status in ("completed", "failed", "cancelled")
        except Exception:
            return False

    async def __aenter__(self) -> "JobHeartbeat":
        self._stop_event.clear()
        # Audit fix #44: non-daemon so a hard process kill can't leave a
        # thread alive writing to a finished job. Combined with terminal-
        # status check above, double safety.
        self._thread = threading.Thread(
            target=self._thread_loop,
            name=f"heartbeat-{self.job_id}",
            daemon=False,
        )
        self._thread.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._thread is not None:
            # Wait briefly for the thread to flush its final write.
            await asyncio.to_thread(self._thread.join, 5.0)
