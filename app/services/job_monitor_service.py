"""
Job Monitoring and Auto-Recovery Service

This service continuously monitors background jobs and automatically:
1. Detects stuck jobs (processing too long without progress)
2. Restarts stuck jobs from last checkpoint
3. Kills zombie processes
4. Cleans up orphaned data
5. Reports health metrics

Run as a background task in the FastAPI application.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from app.services.supabase_client import get_supabase_client
from app.services.checkpoint_recovery_service import (
    checkpoint_recovery_service,
    ProcessingStage
)
from app.utils.retry_utils import retry_async
from app.utils.query_metrics import track_query_performance, query_metrics

logger = logging.getLogger(__name__)

# Import Sentry for stuck job alerts
try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    logger.warning("Sentry SDK not available - stuck job alerts disabled")


class JobMonitorService:
    """
    Monitors background jobs and performs auto-recovery.

    Features:
    - Detect stuck jobs
    - Auto-restart from checkpoints
    - Kill zombie processes
    - Health reporting
    - Retry logic with exponential backoff
    - Circuit breaker for database protection
    - Query performance tracking
    """

    def __init__(
        self,
        check_interval_seconds: int = 60,
        stuck_job_timeout_minutes: int = 30,
        auto_restart_enabled: bool = True
    ):
        """
        Initialize job monitor service.

        Args:
            check_interval_seconds: How often to check for stuck jobs
            stuck_job_timeout_minutes: Consider job stuck if no update for this long
            auto_restart_enabled: Whether to automatically restart stuck jobs
        """
        self.supabase_client = get_supabase_client()
        self.check_interval = check_interval_seconds
        self.stuck_timeout = stuck_job_timeout_minutes
        self.auto_restart = auto_restart_enabled
        self.running = False
        self.stats = {
            "checks_performed": 0,
            "stuck_jobs_detected": 0,
            "jobs_restarted": 0,
            "jobs_failed": 0,
            "last_check": None,
            "errors": 0
        }

        logger.info(f"JobMonitorService initialized (check_interval={check_interval_seconds}s, timeout={stuck_job_timeout_minutes}min)")
    
    async def start(self):
        """Start the monitoring loop"""
        if self.running:
            logger.warning("Monitor already running")
            return
        
        self.running = True
        logger.info("🔍 Job monitor started")
        
        try:
            while self.running:
                await self._check_and_recover()
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            logger.info("🛑 Job monitor cancelled")
            self.running = False
        except Exception as e:
            logger.error(f"❌ Job monitor error: {e}", exc_info=True)
            self.running = False
    
    async def stop(self):
        """Stop the monitoring loop"""
        logger.info("🛑 Stopping job monitor...")
        self.running = False
    
    async def _check_and_recover(self):
        """Main monitoring and recovery logic for ALL job types"""
        try:
            self.stats["checks_performed"] += 1
            self.stats["last_check"] = datetime.utcnow().isoformat()

            # 1. Detect stuck PDF processing jobs (by heartbeat timeout - 15min detection)
            heartbeat_stuck_jobs = await self._detect_heartbeat_timeout_jobs()

            # 2. Detect stuck PDF processing jobs (by updated_at timeout - 5min detection)
            stuck_jobs = await checkpoint_recovery_service.detect_stuck_jobs(
                timeout_minutes=self.stuck_timeout
            )

            # 3. Detect stuck web scraping sessions
            stuck_scraping_sessions = await self._detect_stuck_scraping_sessions()

            # 4. Detect stuck XML import jobs
            stuck_import_jobs = await self._detect_stuck_import_jobs()

            # Combine all detection methods
            all_stuck_jobs = heartbeat_stuck_jobs + stuck_jobs

            if all_stuck_jobs:
                self.stats["stuck_jobs_detected"] += len(all_stuck_jobs)
                logger.warning(f"🛑 Found {len(all_stuck_jobs)} stuck PDF jobs ({len(heartbeat_stuck_jobs)} by heartbeat, {len(stuck_jobs)} by timeout)")

                # Try to recover each stuck PDF job
                for job in all_stuck_jobs:
                    await self._recover_stuck_job(job)

            # Handle stuck scraping sessions
            if stuck_scraping_sessions:
                logger.warning(f"🛑 Found {len(stuck_scraping_sessions)} stuck scraping sessions")
                for session in stuck_scraping_sessions:
                    await self._recover_stuck_scraping_session(session)

            # Handle stuck import jobs
            if stuck_import_jobs:
                logger.warning(f"🛑 Found {len(stuck_import_jobs)} stuck XML import jobs")
                for job in stuck_import_jobs:
                    await self._recover_stuck_import_job(job)

            # 5. Cleanup old checkpoints
            if self.stats["checks_performed"] % 60 == 0:  # Every hour
                await self._cleanup_old_data()

            # 6. Report health metrics
            if self.stats["checks_performed"] % 10 == 0:  # Every 10 checks
                await self._report_health()

        except Exception as e:
            logger.error(f"❌ Error in check_and_recover: {e}", exc_info=True)

    @retry_async(
        max_attempts=3,
        base_delay=2.0,
        exceptions=(Exception,)
    )
    @track_query_performance("background_jobs", "select_heartbeat_timeout")
    async def _detect_heartbeat_timeout_jobs(self, heartbeat_timeout_seconds: int = 900) -> List[Dict[str, Any]]:
        """
        Detect jobs that haven't sent a heartbeat in heartbeat_timeout_seconds.

        This provides crash detection compared to stuck_job_timeout.
        Increased to 900s (15min) to accommodate long-running PDF processing operations.

        Args:
            heartbeat_timeout_seconds: Consider job crashed if no heartbeat for this long (default: 900s = 15min)

        Returns:
            List of stuck jobs detected by heartbeat timeout
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(seconds=heartbeat_timeout_seconds)

            # Query jobs with heartbeat timeout
            result = self.supabase_client.client.table("background_jobs")\
                .select("*")\
                .eq("status", "processing")\
                .lt("last_heartbeat", cutoff_time.isoformat())\
                .execute()

            # Handle both dict and object response types
            stuck_jobs = []
            if isinstance(result, dict):
                stuck_jobs = result.get('data', []) or []
            elif hasattr(result, 'data'):
                stuck_jobs = result.data or []
            elif isinstance(result, str):
                # Handle case where result is a string (error case)
                logger.error(f"❌ Unexpected string result from Supabase: {result}")
                return []
            else:
                logger.error(f"❌ Unexpected result type from Supabase: {type(result)}")
                return []

            if stuck_jobs:
                logger.warning(f"🫀 Detected {len(stuck_jobs)} jobs with stale heartbeat (>{heartbeat_timeout_seconds}s)")

            return stuck_jobs

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"❌ Error detecting heartbeat timeout jobs: {e}")
            return []

    async def _detect_stuck_scraping_sessions(self, timeout_minutes: int = 30) -> List[Dict[str, Any]]:
        """
        Detect stuck web scraping sessions.

        A scraping session is considered stuck if:
        - Status is 'processing' or 'scraping'
        - No update in timeout_minutes (default: 30 minutes)

        Args:
            timeout_minutes: Consider session stuck if no update for this long

        Returns:
            List of stuck scraping sessions
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=timeout_minutes)

            result = self.supabase_client.client.table("scraping_sessions")\
                .select("*")\
                .in_("status", ["processing", "scraping"])\
                .lt("updated_at", cutoff_time.isoformat())\
                .execute()

            stuck_sessions = result.data or []

            if stuck_sessions:
                logger.warning(f"🕷️ Detected {len(stuck_sessions)} stuck scraping sessions (>{timeout_minutes}min)")

            return stuck_sessions

        except Exception as e:
            logger.error(f"❌ Error detecting stuck scraping sessions: {e}")
            return []

    async def _detect_stuck_import_jobs(self, timeout_minutes: int = 20) -> List[Dict[str, Any]]:
        """
        Detect stuck XML import jobs.

        An import job is considered stuck if:
        - Status is 'processing'
        - No update in timeout_minutes (default: 20 minutes)

        Args:
            timeout_minutes: Consider job stuck if no update for this long

        Returns:
            List of stuck import jobs
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=timeout_minutes)

            result = self.supabase_client.client.table("data_import_jobs")\
                .select("*")\
                .eq("status", "processing")\
                .lt("updated_at", cutoff_time.isoformat())\
                .execute()

            stuck_jobs = result.data or []

            if stuck_jobs:
                logger.warning(f"📦 Detected {len(stuck_jobs)} stuck XML import jobs (>{timeout_minutes}min)")

            return stuck_jobs

        except Exception as e:
            logger.error(f"❌ Error detecting stuck import jobs: {e}")
            return []

    async def _recover_stuck_job(self, job: Dict[str, Any]):
        """
        Attempt to recover a stuck job.
        
        Strategy:
        1. Check if job has valid checkpoint
        2. If yes, restart from checkpoint
        3. If no, mark as failed
        """
        job_id = job["id"]
        filename = job.get("filename", "unknown")
        
        logger.info(f"🔄 Attempting to recover stuck job: {job_id} ({filename})")
        
        try:
            # Check if we can resume from checkpoint
            can_resume, last_stage = await checkpoint_recovery_service.can_resume_from_checkpoint(job_id)
            
            if can_resume and self.auto_restart:
                # Verify checkpoint data is valid
                is_valid = await checkpoint_recovery_service.verify_checkpoint_data(job_id, last_stage)
                
                if is_valid:
                    # Restart from checkpoint
                    success = await checkpoint_recovery_service.auto_restart_stuck_job(job_id)
                    
                    if success:
                        self.stats["jobs_restarted"] += 1
                        logger.info(f"✅ Restarted {job_id} from {last_stage.value}")
                    else:
                        await self._mark_job_failed(job_id, "Failed to restart from checkpoint")
                        self.stats["jobs_failed"] += 1
                else:
                    # Checkpoint data is invalid - cleanup and fail
                    await checkpoint_recovery_service.cleanup_invalid_checkpoints(job_id)
                    await self._mark_job_failed(job_id, "Invalid checkpoint data")
                    self.stats["jobs_failed"] += 1
                    logger.warning(f"⚠️ Invalid checkpoint for {job_id} - marked as failed")
            else:
                # No valid checkpoint - mark as failed
                await self._mark_job_failed(job_id, "Stuck without valid checkpoint")
                self.stats["jobs_failed"] += 1
                logger.warning(f"⚠️ No valid checkpoint for {job_id} - marked as failed")
                
        except Exception as e:
            logger.error(f"❌ Failed to recover job {job_id}: {e}", exc_info=True)
            await self._mark_job_failed(job_id, f"Recovery error: {str(e)}")
            self.stats["jobs_failed"] += 1
    
    async def _mark_job_failed(self, job_id: str, reason: str):
        """
        Mark a job as failed and send Sentry alert.

        Args:
            job_id: Job ID to mark as failed
            reason: Reason for failure
        """
        try:
            # Get job details for Sentry context
            job_result = self.supabase_client.client.table("background_jobs")\
                .select("*")\
                .eq("id", job_id)\
                .execute()

            job = job_result.data[0] if job_result.data else {}

            # 🚨 SENTRY ALERT: Send stuck job alert
            if SENTRY_AVAILABLE:
                try:
                    with sentry_sdk.configure_scope() as scope:
                        # Add job context
                        scope.set_tag("job_id", job_id)
                        scope.set_tag("document_id", job.get("document_id", "unknown"))
                        scope.set_tag("job_type", job.get("job_type", "unknown"))
                        scope.set_tag("error_type", "stuck_job")
                        scope.set_tag("failure_reason", reason)

                        # Add job details
                        scope.set_context("stuck_job", {
                            "job_id": job_id,
                            "document_id": job.get("document_id"),
                            "job_type": job.get("job_type"),
                            "filename": job.get("filename"),
                            "progress": job.get("progress", 0),
                            "created_at": job.get("created_at"),
                            "started_at": job.get("started_at"),
                            "last_heartbeat": job.get("last_heartbeat"),
                            "updated_at": job.get("updated_at"),
                            "failure_reason": reason
                        })

                    # Capture message with context
                    sentry_sdk.capture_message(
                        f"Stuck job detected and failed: {job_id}",
                        level="warning",
                        fingerprint=["stuck-job", reason]
                    )

                    logger.info(f"📊 Sent stuck job alert to Sentry for job {job_id}")

                except Exception as sentry_error:
                    logger.warning(f"Failed to send Sentry alert: {sentry_error}")

            # Update job status in database
            self.supabase_client.client.table("background_jobs")\
                .update({
                    "status": "failed",
                    "error": reason,
                    "failed_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                })\
                .eq("id", job_id)\
                .execute()

            logger.info(f"❌ Marked job {job_id} as failed: {reason}")
        except Exception as e:
            logger.error(f"Failed to mark job as failed: {e}")

    async def _recover_stuck_scraping_session(self, session: Dict[str, Any]):
        """
        Attempt to recover a stuck scraping session.

        Strategy:
        1. Mark session as failed
        2. Send Sentry alert with session details
        3. Update background_jobs if linked
        """
        session_id = session["id"]
        source_url = session.get("source_url", "unknown")

        logger.info(f"🕷️ Recovering stuck scraping session: {session_id} ({source_url})")

        try:
            # Prepare failure reason
            stuck_duration = (
                datetime.utcnow() -
                datetime.fromisoformat(session["updated_at"].replace("Z", "+00:00")).replace(tzinfo=None)
            ).total_seconds() / 60

            reason = f"Scraping session stuck for {stuck_duration:.1f} minutes without progress"

            # 🚨 SENTRY ALERT: Send stuck scraping session alert
            if SENTRY_AVAILABLE:
                try:
                    with sentry_sdk.configure_scope() as scope:
                        scope.set_tag("session_id", session_id)
                        scope.set_tag("job_type", "web_scraping")
                        scope.set_tag("error_type", "stuck_scraping_session")
                        scope.set_tag("source_url", source_url)

                        scope.set_context("stuck_scraping_session", {
                            "session_id": session_id,
                            "source_url": source_url,
                            "status": session.get("status"),
                            "total_pages": session.get("total_pages", 0),
                            "completed_pages": session.get("completed_pages", 0),
                            "failed_pages": session.get("failed_pages", 0),
                            "created_at": session.get("created_at"),
                            "updated_at": session.get("updated_at"),
                            "stuck_duration_minutes": stuck_duration,
                            "failure_reason": reason
                        })

                    sentry_sdk.capture_message(
                        f"Stuck scraping session detected: {session_id}",
                        level="warning",
                        fingerprint=["stuck-scraping-session", source_url]
                    )

                    logger.info(f"📊 Sent stuck scraping session alert to Sentry")

                except Exception as sentry_error:
                    logger.warning(f"Failed to send Sentry alert: {sentry_error}")

            # Mark session as failed
            self.supabase_client.client.table("scraping_sessions")\
                .update({
                    "status": "failed",
                    "error_message": reason,
                    "updated_at": datetime.utcnow().isoformat()
                })\
                .eq("id", session_id)\
                .execute()

            # Update linked background_job if exists
            if session.get("background_job_id"):
                self.supabase_client.client.table("background_jobs")\
                    .update({
                        "status": "failed",
                        "error": reason,
                        "failed_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    })\
                    .eq("id", session["background_job_id"])\
                    .execute()

            logger.info(f"❌ Marked scraping session {session_id} as failed")

        except Exception as e:
            logger.error(f"Failed to recover stuck scraping session: {e}")

    async def _recover_stuck_import_job(self, job: Dict[str, Any]):
        """
        Attempt to recover a stuck XML import job.

        Strategy:
        1. Mark job as failed
        2. Send Sentry alert with job details
        3. Update background_jobs if linked
        """
        job_id = job["id"]
        source_name = job.get("source_name", "unknown")

        logger.info(f"📦 Recovering stuck XML import job: {job_id} ({source_name})")

        try:
            # Prepare failure reason
            stuck_duration = (
                datetime.utcnow() -
                datetime.fromisoformat(job["updated_at"].replace("Z", "+00:00")).replace(tzinfo=None)
            ).total_seconds() / 60

            reason = f"XML import job stuck for {stuck_duration:.1f} minutes without progress"

            # 🚨 SENTRY ALERT: Send stuck import job alert
            if SENTRY_AVAILABLE:
                try:
                    with sentry_sdk.configure_scope() as scope:
                        scope.set_tag("import_job_id", job_id)
                        scope.set_tag("job_type", "xml_import")
                        scope.set_tag("error_type", "stuck_import_job")
                        scope.set_tag("source_name", source_name)

                        scope.set_context("stuck_import_job", {
                            "job_id": job_id,
                            "source_name": source_name,
                            "import_type": job.get("import_type"),
                            "status": job.get("status"),
                            "total_products": job.get("total_products", 0),
                            "processed_products": job.get("processed_products", 0),
                            "failed_products": job.get("failed_products", 0),
                            "created_at": job.get("created_at"),
                            "updated_at": job.get("updated_at"),
                            "stuck_duration_minutes": stuck_duration,
                            "failure_reason": reason
                        })

                    sentry_sdk.capture_message(
                        f"Stuck XML import job detected: {job_id}",
                        level="warning",
                        fingerprint=["stuck-import-job", source_name]
                    )

                    logger.info(f"📊 Sent stuck import job alert to Sentry")

                except Exception as sentry_error:
                    logger.warning(f"Failed to send Sentry alert: {sentry_error}")

            # Mark import job as failed
            self.supabase_client.client.table("data_import_jobs")\
                .update({
                    "status": "failed",
                    "error_message": reason,
                    "updated_at": datetime.utcnow().isoformat()
                })\
                .eq("id", job_id)\
                .execute()

            # Update linked background_job if exists
            if job.get("background_job_id"):
                self.supabase_client.client.table("background_jobs")\
                    .update({
                        "status": "failed",
                        "error": reason,
                        "failed_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    })\
                    .eq("id", job["background_job_id"])\
                    .execute()

            logger.info(f"❌ Marked XML import job {job_id} as failed")

        except Exception as e:
            logger.error(f"Failed to recover stuck import job: {e}")

    async def _cleanup_old_data(self):
        """Cleanup old checkpoints and completed jobs"""
        try:
            # Cleanup checkpoints older than 7 days
            result = self.supabase_client.client.rpc("cleanup_old_checkpoints", {}).execute()
            deleted = result.data if result.data else 0

            if deleted > 0:
                logger.info(f"🧹 Cleaned up {deleted} old checkpoints")

            # Cleanup completed jobs older than 5 days (aligned with Edge Function)
            cutoff = (datetime.utcnow() - timedelta(days=5)).isoformat()
            result = self.supabase_client.client.table("background_jobs")\
                .delete()\
                .eq("status", "completed")\
                .lt("completed_at", cutoff)\
                .execute()

            deleted_jobs = len(result.data) if result.data else 0
            if deleted_jobs > 0:
                logger.info(f"🧹 Cleaned up {deleted_jobs} old completed jobs")

        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}", exc_info=True)
    
    async def _report_health(self):
        """Report health metrics"""
        try:
            # Get job statistics
            result = self.supabase_client.client.table("background_jobs")\
                .select("status")\
                .execute()
            
            jobs = result.data or []
            
            status_counts = {}
            for job in jobs:
                status = job["status"]
                status_counts[status] = status_counts.get(status, 0) + 1
            
            logger.info("📊 Job Monitor Health Report:")
            logger.info(f"   Checks performed: {self.stats['checks_performed']}")
            logger.info(f"   Stuck jobs detected: {self.stats['stuck_jobs_detected']}")
            logger.info(f"   Jobs restarted: {self.stats['jobs_restarted']}")
            logger.info(f"   Jobs failed: {self.stats['jobs_failed']}")
            logger.info(f"   Last check: {self.stats['last_check']}")
            logger.info(f"   Job status breakdown: {status_counts}")
            
        except Exception as e:
            logger.error(f"❌ Health report error: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        try:
            # Get job counts
            result = self.supabase_client.client.table("background_jobs")\
                .select("status")\
                .execute()
            
            jobs = result.data or []
            
            status_counts = {}
            for job in jobs:
                status = job["status"]
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Get stuck jobs
            stuck_jobs = await checkpoint_recovery_service.detect_stuck_jobs(
                timeout_minutes=self.stuck_timeout
            )
            
            return {
                "monitor_running": self.running,
                "stats": self.stats,
                "job_counts": status_counts,
                "stuck_jobs_count": len(stuck_jobs),
                "stuck_jobs": [
                    {
                        "job_id": job["id"],
                        "filename": job.get("filename"),
                        "stuck_duration_minutes": (
                            datetime.utcnow() - 
                            datetime.fromisoformat(job["updated_at"].replace("Z", "+00:00")).replace(tzinfo=None)
                        ).total_seconds() / 60
                    }
                    for job in stuck_jobs
                ],
                "health": "healthy" if len(stuck_jobs) == 0 else "degraded",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get health status: {e}")
            return {
                "monitor_running": self.running,
                "error": str(e),
                "health": "error",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def force_restart_job(self, job_id: str) -> Dict[str, Any]:
        """
        Manually force restart a job from last checkpoint.
        
        Returns:
            Status of restart operation
        """
        try:
            # Get job
            result = self.supabase_client.client.table("background_jobs")\
                .select("*")\
                .eq("id", job_id)\
                .single()\
                .execute()
            
            if not result.data:
                return {
                    "success": False,
                    "error": "Job not found"
                }
            
            job = result.data
            
            # Check if job can be restarted
            if job["status"] in ["completed"]:
                return {
                    "success": False,
                    "error": f"Job is {job['status']} - cannot restart"
                }
            
            # Get last checkpoint
            can_resume, last_stage = await checkpoint_recovery_service.can_resume_from_checkpoint(job_id)
            
            if not can_resume:
                return {
                    "success": False,
                    "error": "No valid checkpoint found"
                }
            
            # Verify checkpoint
            is_valid = await checkpoint_recovery_service.verify_checkpoint_data(job_id, last_stage)
            
            if not is_valid:
                return {
                    "success": False,
                    "error": "Checkpoint data is invalid"
                }
            
            # Restart
            success = await checkpoint_recovery_service.auto_restart_stuck_job(job_id)
            
            if success:
                return {
                    "success": True,
                    "message": f"Job restarted from {last_stage.value}",
                    "restart_stage": last_stage.value
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to restart job"
                }
                
        except Exception as e:
            logger.error(f"❌ Force restart error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }


# Global instance
job_monitor_service = JobMonitorService(
    check_interval_seconds=60,  # Check every 60 seconds (reasonable for monitoring)
    stuck_job_timeout_minutes=30,  # Consider stuck after 30 minutes (PDF processing can take 20+ minutes)
    auto_restart_enabled=True  # Auto-restart enabled
)

