"""
Job Progress Monitor Service

Monitors job progress and sends detailed status updates to logs and Sentry every minute.
This helps identify exactly where jobs get stuck and what's happening at each stage.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, Any
import sentry_sdk

logger = logging.getLogger(__name__)


# Stage-specific timeout thresholds (in seconds)
# These are realistic timeouts based on actual processing times for different stages
STAGE_TIMEOUTS = {
    "downloading": 120,              # 2 minutes - file download should be quick
    "extracting_text": 300,          # 5 minutes - text extraction is fast
    "extracting_images": 1200,       # 20 minutes - image extraction can be slow for large PDFs
    "generating_embeddings": 1500,   # 25 minutes - CLIP embeddings take time (4 specialized embeddings per image)
    "product_discovery": 900,        # 15 minutes - Claude API calls for product analysis
    "chunking": 600,                 # 10 minutes - text chunking and processing
    "storing_chunks": 600,           # 10 minutes - database operations
    "image_processing": 1200,        # 20 minutes - same as extracting_images
    "metadata_extraction": 900,      # 15 minutes - metadata processing
    "default": 600                   # 10 minutes - fallback for unknown stages
}


class JobProgressMonitor:
    """
    Monitors job progress and reports detailed status every minute.
    
    Features:
    - Logs detailed progress every 60 seconds
    - Sends status to Sentry for remote monitoring
    - Tracks stage transitions and time spent in each stage
    - Identifies stuck jobs and reports exact location
    """
    
    def __init__(self, job_id: str, document_id: str, total_stages: int = 9):
        self.job_id = job_id
        self.document_id = document_id
        self.total_stages = total_stages
        self.current_stage = "initializing"
        self.current_stage_start = datetime.utcnow()
        self.stage_history = []
        self.last_report_time = datetime.utcnow()
        self.monitoring_task = None
        self.is_running = False
        
    async def start(self):
        """Start the monitoring task"""
        if self.is_running:
            return

        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitor_loop())

        # Log to both console and Sentry
        logger.info(f"📊 Started progress monitoring for job {self.job_id}")
        sentry_sdk.capture_message(
            f"📊 Started progress monitoring for job {self.job_id}",
            level="info"
        )
        
    async def stop(self):
        """Stop the monitoring task"""
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        # CRITICAL FIX: Log only, don't send to Sentry (reduces noise)
        logger.info(f"📊 Stopped progress monitoring for job {self.job_id}")
        # REMOVED: sentry_sdk.capture_message() - was creating noise in Sentry
        
    def update_stage(self, stage: str, details: Optional[Dict[str, Any]] = None):
        """Update current stage and log transition"""
        time_in_previous_stage = (datetime.utcnow() - self.current_stage_start).total_seconds()

        self.stage_history.append({
            "stage": self.current_stage,
            "duration_seconds": time_in_previous_stage,
            "completed_at": datetime.utcnow().isoformat()
        })

        # Log to console
        logger.info(f"🔄 Job {self.job_id}: {self.current_stage} → {stage} (took {time_in_previous_stage:.1f}s)")

        self.current_stage = stage
        self.current_stage_start = datetime.utcnow()

        # CRITICAL FIX: Only send to Sentry if stage took unusually long (> 5 minutes)
        # Normal stage transitions create noise (MIVAA-39, 38, 35...)
        if time_in_previous_stage > 300:  # 5 minutes
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("job_id", self.job_id)
                scope.set_tag("document_id", self.document_id)
                scope.set_tag("stage", stage)
                scope.set_tag("previous_stage", self.stage_history[-1]["stage"] if self.stage_history else "start")
                scope.set_context("stage_transition", {
                    "from": self.stage_history[-1]["stage"] if self.stage_history else "start",
                    "to": stage,
                    "duration_seconds": time_in_previous_stage,
                    "duration_minutes": round(time_in_previous_stage / 60, 2),
                    "details": details or {},
                    "timestamp": datetime.utcnow().isoformat()
                })

                # Only alert on slow stage transitions
                sentry_sdk.capture_message(
                    f"⚠️ SLOW STAGE: {self.current_stage if self.stage_history else 'start'} → {stage} took {time_in_previous_stage/60:.1f} minutes",
                    level="warning"
                )
        # REMOVED: Normal stage transitions no longer sent to Sentry - reduces noise
    
    async def _monitor_loop(self):
        """Monitor loop that reports status every minute"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Report every 60 seconds
                await self._report_status()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error to both console and Sentry
                logger.error(f"❌ Error in monitoring loop: {e}", exc_info=True)
                sentry_sdk.capture_exception(e)
                
    async def _report_status(self):
        """Report detailed status to logs and Sentry"""
        time_in_current_stage = (datetime.utcnow() - self.current_stage_start).total_seconds()
        total_time = sum(s["duration_seconds"] for s in self.stage_history) + time_in_current_stage

        status_msg = (
            f"\n{'='*80}\n"
            f"📊 JOB PROGRESS REPORT - {datetime.utcnow().strftime('%H:%M:%S')}\n"
            f"{'='*80}\n"
            f"Job ID: {self.job_id}\n"
            f"Document ID: {self.document_id}\n"
            f"Current Stage: {self.current_stage}\n"
            f"Time in Current Stage: {time_in_current_stage:.1f}s ({time_in_current_stage/60:.1f}min)\n"
            f"Total Processing Time: {total_time:.1f}s ({total_time/60:.1f}min)\n"
            f"Stages Completed: {len(self.stage_history)}/{self.total_stages}\n"
            f"\nStage History:\n"
        )

        for i, stage in enumerate(self.stage_history[-5:], 1):  # Last 5 stages
            status_msg += f"  {i}. {stage['stage']}: {stage['duration_seconds']:.1f}s\n"

        status_msg += f"{'='*80}\n"

        # Log to console
        logger.info(status_msg)

        # Send comprehensive status to Sentry with full context
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("job_id", self.job_id)
            scope.set_tag("document_id", self.document_id)
            scope.set_tag("current_stage", self.current_stage)
            scope.set_tag("stages_completed", f"{len(self.stage_history)}/{self.total_stages}")
            scope.set_tag("total_time_minutes", round(total_time / 60, 2))

            scope.set_context("job_progress", {
                "current_stage": self.current_stage,
                "time_in_stage_seconds": time_in_current_stage,
                "time_in_stage_minutes": round(time_in_current_stage / 60, 2),
                "total_time_seconds": total_time,
                "total_time_minutes": round(total_time / 60, 2),
                "stages_completed": len(self.stage_history),
                "total_stages": self.total_stages,
                "progress_percentage": round((len(self.stage_history) / self.total_stages) * 100, 1),
                "stage_history": self.stage_history[-5:],
                "timestamp": datetime.utcnow().isoformat()
            })

            # CRITICAL FIX: Use stage-specific timeouts instead of global 10-minute threshold
            # This reduces false "stuck job" alerts for legitimately slow stages
            stage_timeout = STAGE_TIMEOUTS.get(self.current_stage, STAGE_TIMEOUTS["default"])

            if time_in_current_stage > stage_timeout:
                timeout_minutes = stage_timeout / 60
                actual_minutes = time_in_current_stage / 60
                logger.warning(
                    f"⚠️ Job {self.job_id} stuck in {self.current_stage} for {actual_minutes:.1f} minutes "
                    f"(threshold: {timeout_minutes:.1f} minutes)"
                )
                sentry_sdk.capture_message(
                    f"⚠️ STUCK JOB: {self.job_id} in {self.current_stage} for {actual_minutes:.1f} minutes "
                    f"(threshold: {timeout_minutes:.1f}min)",
                    level="warning"
                )
            else:
                # Regular progress update - LOG ONLY, don't send to Sentry (reduces noise)
                logger.debug(
                    f"📊 Progress Update: Job {self.job_id} - {self.current_stage} "
                    f"({len(self.stage_history)}/{self.total_stages} stages, {total_time/60:.1f}min total)"
                )
                # REMOVED: sentry_sdk.capture_message() for normal progress - was creating noise

