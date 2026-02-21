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
# Updated 2025-12-26: Increased thresholds to reduce false "stuck job" alerts (MIVAA-7W, MIVAA-7X)
STAGE_TIMEOUTS = {
    "downloading": 120,              # 2 minutes - file download should be quick
    "extracting_text": 300,          # 5 minutes - text extraction is fast
    "extracting_images": 1800,       # 30 minutes - image extraction can be slow for large PDFs (was 20min)
    "generating_embeddings": 2400,   # 40 minutes - CLIP embeddings take time (4 specialized embeddings per image) (was 25min)
    "product_discovery": 1200,       # 20 minutes - Claude API calls for product analysis (was 15min)
    "focused_extraction": 900,       # 15 minutes - focused text extraction for products (NEW - fixes MIVAA-7W)
    "chunking": 900,                 # 15 minutes - text chunking and processing (was 10min)
    "storing_chunks": 900,           # 15 minutes - database operations (was 10min)
    "image_processing": 1800,        # 30 minutes - same as extracting_images (was 20min - fixes MIVAA-7X)
    "metadata_extraction": 1200,     # 20 minutes - metadata processing (was 15min)
    "field_propagation": 120,        # 2 minutes  - sibling field propagation (DB reads/writes only)
    "dimension_extraction": 120,     # 2 minutes  - regex scan of text chunks for sizes/thickness
    "quality_enhancement": 1800,     # 30 minutes - Stage 5 quality validation
    "default": 900                   # 15 minutes - fallback for unknown stages (was 10min)
}

# Slow stage warning threshold (in seconds)
# Warn if a stage takes longer than this, but don't mark as stuck
SLOW_STAGE_THRESHOLD = 300  # 5 minutes


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

        # CRITICAL FIX: Only send to Sentry if stage took unusually long
        # Use stage-specific threshold or SLOW_STAGE_THRESHOLD (5 minutes)
        # This reduces false alerts for legitimately slow stages (MIVAA-7W)
        previous_stage_name = self.stage_history[-1]["stage"] if self.stage_history else "start"
        stage_threshold = STAGE_TIMEOUTS.get(previous_stage_name, SLOW_STAGE_THRESHOLD)

        if time_in_previous_stage > stage_threshold:
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("job_id", self.job_id)
                scope.set_tag("document_id", self.document_id)
                scope.set_tag("stage", stage)
                scope.set_tag("previous_stage", previous_stage_name)
                scope.set_context("stage_transition", {
                    "from": previous_stage_name,
                    "to": stage,
                    "duration_seconds": time_in_previous_stage,
                    "duration_minutes": round(time_in_previous_stage / 60, 2),
                    "threshold_minutes": round(stage_threshold / 60, 2),
                    "details": details or {},
                    "timestamp": datetime.utcnow().isoformat()
                })

                # Only alert on slow stage transitions that exceed threshold
                sentry_sdk.capture_message(
                    f"⚠️ SLOW STAGE: {previous_stage_name} → {stage} took {time_in_previous_stage/60:.1f} minutes (threshold: {stage_threshold/60:.1f}min)",
                    level="warning"
                )

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

                # ENHANCED LOGGING: Add detailed context to identify if job is truly stuck
                logger.warning(
                    f"⚠️ Job {self.job_id} in {self.current_stage} for {actual_minutes:.1f} minutes "
                    f"(threshold: {timeout_minutes:.1f} minutes)\n"
                    f"   📊 Progress: {len(self.stage_history)}/{self.total_stages} stages completed\n"
                    f"   ⏱️  Total time: {total_time/60:.1f} minutes\n"
                    f"   🔄 Last heartbeat: {datetime.utcnow().isoformat()}"
                )

                # Send to Sentry with enhanced context
                sentry_sdk.capture_message(
                    f"⚠️ POTENTIAL STUCK JOB: {self.job_id} in {self.current_stage} for {actual_minutes:.1f} minutes "
                    f"(threshold: {timeout_minutes:.1f}min) - Check if job is making progress or truly stuck",
                    level="warning"
                )
            else:
                # Regular progress update - LOG ONLY, don't send to Sentry (reduces noise)
                logger.debug(
                    f"📊 Progress Update: Job {self.job_id} - {self.current_stage} "
                    f"({len(self.stage_history)}/{self.total_stages} stages, {total_time/60:.1f}min total)"
                )


