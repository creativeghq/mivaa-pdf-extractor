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

logger = logging.getLogger(__name__)


class JobMonitorService:
    """
    Monitors background jobs and performs auto-recovery.
    
    Features:
    - Detect stuck jobs
    - Auto-restart from checkpoints
    - Kill zombie processes
    - Health reporting
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
            "last_check": None
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
        """Main monitoring and recovery logic"""
        try:
            self.stats["checks_performed"] += 1
            self.stats["last_check"] = datetime.utcnow().isoformat()
            
            # 1. Detect stuck jobs
            stuck_jobs = await checkpoint_recovery_service.detect_stuck_jobs(
                timeout_minutes=self.stuck_timeout
            )
            
            if stuck_jobs:
                self.stats["stuck_jobs_detected"] += len(stuck_jobs)
                logger.warning(f"🛑 Found {len(stuck_jobs)} stuck jobs")
                
                # 2. Try to recover each stuck job
                for job in stuck_jobs:
                    await self._recover_stuck_job(job)
            
            # 3. Cleanup old checkpoints
            if self.stats["checks_performed"] % 60 == 0:  # Every hour
                await self._cleanup_old_data()
            
            # 4. Report health metrics
            if self.stats["checks_performed"] % 10 == 0:  # Every 10 checks
                await self._report_health()
                
        except Exception as e:
            logger.error(f"❌ Error in check_and_recover: {e}", exc_info=True)
    
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
        """Mark a job as failed"""
        try:
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
    
    async def _cleanup_old_data(self):
        """Cleanup old checkpoints and completed jobs"""
        try:
            # Cleanup checkpoints older than 7 days
            result = self.supabase_client.client.rpc("cleanup_old_checkpoints").execute()
            deleted = result.data if result.data else 0
            
            if deleted > 0:
                logger.info(f"🧹 Cleaned up {deleted} old checkpoints")
            
            # Cleanup completed jobs older than 30 days
            cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
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
    check_interval_seconds=60,  # Check every minute
    stuck_job_timeout_minutes=30,  # Consider stuck after 30 minutes
    auto_restart_enabled=True  # Auto-restart enabled
)

