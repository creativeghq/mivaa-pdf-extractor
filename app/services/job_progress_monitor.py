"""
Job Progress Monitor Service

Monitors job progress and sends detailed status updates every minute to:
- Server logs (console)
- Sentry (for remote monitoring)

This service provides real-time visibility into long-running PDF processing jobs,
helping identify stuck jobs and troubleshoot issues.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from app.database.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class JobProgressMonitor:
    """
    Monitors job progress and reports detailed status every 60 seconds.
    
    Features:
    - Logs detailed progress to console every 60s
    - Sends status to Sentry for remote monitoring
    - Tracks stage transitions and time spent in each stage
    - Identifies stuck jobs (>10min in one stage)
    - Provides comprehensive status reports
    """
    
    def __init__(self, job_id: str, document_id: str, total_stages: int = 9):
        """
        Initialize the progress monitor.
        
        Args:
            job_id: The job ID to monitor
            document_id: The document ID being processed
            total_stages: Total number of processing stages (default: 9)
        """
        self.job_id = job_id
        self.document_id = document_id
        self.total_stages = total_stages
        self.current_stage = "initializing"
        self.current_stage_start = datetime.utcnow()
        self.stage_history: List[Dict[str, Any]] = []
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.start_time = datetime.utcnow()
        
        logger.info(f"📊 [MONITOR] Initialized for job {job_id}")
    
    async def start(self):
        """Start the monitoring task"""
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"📊 [MONITOR] Started monitoring for job {self.job_id}")
    
    async def stop(self):
        """Stop the monitoring task"""
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Send final report
        await self._report_status(final=True)
        logger.info(f"📊 [MONITOR] Stopped monitoring for job {self.job_id}")
    
    def update_stage(self, stage: str, details: Optional[Dict[str, Any]] = None):
        """
        Update current stage and log transition.
        
        Args:
            stage: The new stage name
            details: Optional details about the stage
        """
        # Record stage transition
        stage_duration = (datetime.utcnow() - self.current_stage_start).total_seconds()
        
        self.stage_history.append({
            "stage": self.current_stage,
            "duration_seconds": stage_duration,
            "completed_at": datetime.utcnow().isoformat()
        })
        
        logger.info(
            f"📊 [MONITOR] Stage transition: {self.current_stage} → {stage} "
            f"(spent {stage_duration:.1f}s in {self.current_stage})"
        )
        
        # Update to new stage
        self.current_stage = stage
        self.current_stage_start = datetime.utcnow()
        
        # Send to Sentry
        try:
            import sentry_sdk
            sentry_sdk.capture_message(
                f"Job {self.job_id}: Stage transition to {stage}",
                level="info",
                extras={
                    "job_id": self.job_id,
                    "document_id": self.document_id,
                    "stage": stage,
                    "previous_stage": self.stage_history[-1]["stage"] if self.stage_history else "none",
                    "stage_duration": stage_duration,
                    "details": details or {}
                }
            )
        except Exception as e:
            logger.warning(f"Failed to send stage transition to Sentry: {e}")
    
    async def _monitor_loop(self):
        """Monitor loop that reports status every minute"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Report every 60 seconds
                await self._report_status()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ [MONITOR] Error in monitoring loop: {e}", exc_info=True)
    
    async def _report_status(self, final: bool = False):
        """
        Report detailed status to logs and Sentry.
        
        Args:
            final: Whether this is the final report (job completed/failed)
        """
        try:
            # Get job status from database
            supabase = get_supabase_client()
            job_response = supabase.client.table('background_jobs').select('*').eq('id', self.job_id).execute()
            
            if not job_response.data:
                logger.warning(f"⚠️ [MONITOR] Job {self.job_id} not found in database")
                return
            
            job = job_response.data[0]
            
            # Calculate metrics
            total_duration = (datetime.utcnow() - self.start_time).total_seconds()
            stage_duration = (datetime.utcnow() - self.current_stage_start).total_seconds()
            
            # Check if stuck (>10min in one stage)
            is_stuck = stage_duration > 600  # 10 minutes
            
            # Build status report
            status_report = {
                "job_id": self.job_id,
                "document_id": self.document_id,
                "current_stage": self.current_stage,
                "stage_duration_seconds": stage_duration,
                "total_duration_seconds": total_duration,
                "progress_percent": job.get('progress', 0),
                "status": job.get('status', 'unknown'),
                "is_stuck": is_stuck,
                "stages_completed": len(self.stage_history),
                "total_stages": self.total_stages,
                "last_heartbeat": job.get('last_heartbeat'),
                "metadata": job.get('metadata', {})
            }
            
            # Log to console
            report_type = "FINAL REPORT" if final else "STATUS REPORT"
            logger.info("=" * 80)
            logger.info(f"📊 [MONITOR] {report_type} - Job {self.job_id}")
            logger.info("=" * 80)
            logger.info(f"📄 Document ID: {self.document_id}")
            logger.info(f"🎯 Current Stage: {self.current_stage}")
            logger.info(f"⏱️  Stage Duration: {stage_duration:.1f}s ({stage_duration/60:.1f}min)")
            logger.info(f"⏱️  Total Duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")
            logger.info(f"📊 Progress: {job.get('progress', 0)}%")
            logger.info(f"✅ Status: {job.get('status', 'unknown').upper()}")
            logger.info(f"🔄 Stages: {len(self.stage_history)}/{self.total_stages} completed")
            
            if is_stuck:
                logger.warning(f"⚠️  STUCK: Job has been in '{self.current_stage}' stage for {stage_duration/60:.1f} minutes!")
            
            # Show stage history
            if self.stage_history:
                logger.info("📜 Stage History:")
                for i, stage_info in enumerate(self.stage_history[-5:], 1):  # Last 5 stages
                    logger.info(f"   {i}. {stage_info['stage']}: {stage_info['duration_seconds']:.1f}s")
            
            # Show metadata if available
            metadata = job.get('metadata', {})
            if metadata:
                logger.info("📋 Metadata:")
                for key, value in list(metadata.items())[:10]:  # First 10 items
                    logger.info(f"   - {key}: {value}")
            
            logger.info("=" * 80)
            
            # Send to Sentry
            try:
                import sentry_sdk
                
                level = "warning" if is_stuck else "info"
                message = f"Job {self.job_id}: {report_type} - {self.current_stage}"
                
                if is_stuck:
                    message += f" (STUCK for {stage_duration/60:.1f}min)"
                
                sentry_sdk.capture_message(
                    message,
                    level=level,
                    extras=status_report
                )
            except Exception as e:
                logger.warning(f"Failed to send status to Sentry: {e}")
        
        except Exception as e:
            logger.error(f"❌ [MONITOR] Failed to report status: {e}", exc_info=True)
