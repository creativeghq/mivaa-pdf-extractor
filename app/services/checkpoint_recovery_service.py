"""
Checkpoint-Based Recovery Service

This service provides granular checkpoint-based recovery for PDF processing jobs.
It allows jobs to resume from the last successful checkpoint instead of restarting from scratch.

Features:
- Checkpoint creation at each processing stage
- Automatic recovery from last checkpoint
- Smart restart detection (stuck jobs, failed operations)
- Partial result preservation
- Idempotent operations (safe to retry)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum

from app.services.supabase_client import get_supabase_client
from app.utils.timestamp_utils import normalize_timestamp

logger = logging.getLogger(__name__)


class ProcessingStage(str, Enum):
    """Processing stages with checkpoint support"""
    INITIALIZED = "initialized"
    PDF_EXTRACTED = "pdf_extracted"
    CHUNKS_CREATED = "chunks_created"
    TEXT_EMBEDDINGS_GENERATED = "text_embeddings_generated"
    IMAGES_EXTRACTED = "images_extracted"
    IMAGE_EMBEDDINGS_GENERATED = "image_embeddings_generated"
    PRODUCTS_DETECTED = "products_detected"
    PRODUCTS_CREATED = "products_created"
    RELATIONSHIPS_CREATED = "relationships_created"
    DOCUMENT_ENTITIES_CREATED = "document_entities_created"
    METADATA_EXTRACTED = "metadata_extracted"
    COMPLETED = "completed"


class CheckpointRecoveryService:
    """
    Service for checkpoint-based recovery of PDF processing jobs.
    
    Allows jobs to resume from last successful checkpoint instead of restarting.
    """
    
    def __init__(self):
        self.supabase_client = get_supabase_client()
        self.checkpoints_table = "job_checkpoints"
        self.jobs_table = "background_jobs"
        logger.info("CheckpointRecoveryService initialized")
    
    async def create_checkpoint(
        self,
        job_id: str,
        stage: ProcessingStage,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a checkpoint for a job at a specific stage.
        
        Args:
            job_id: Job identifier
            stage: Processing stage
            data: Data to save (e.g., chunk IDs, image IDs, etc.)
            metadata: Additional metadata
            
        Returns:
            bool: True if checkpoint created successfully
        """
        try:
            checkpoint_data = {
                "job_id": job_id,
                "stage": stage.value,
                "checkpoint_data": data,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Upsert checkpoint (update if exists, insert if not)
            self.supabase_client.client.table(self.checkpoints_table)\
                .upsert(checkpoint_data, on_conflict="job_id,stage")\
                .execute()
            
            logger.info(f"✅ Checkpoint created: {job_id} @ {stage.value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create checkpoint for {job_id} @ {stage.value}: {e}")
            return False
    
    async def get_last_checkpoint(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the last successful checkpoint for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Checkpoint data or None if no checkpoint exists
        """
        try:
            result = self.supabase_client.client.table(self.checkpoints_table)\
                .select("*")\
                .eq("job_id", job_id)\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            
            if result.data and len(result.data) > 0:
                checkpoint = result.data[0]
                logger.info(f"📍 Last checkpoint for {job_id}: {checkpoint['stage']}")
                return checkpoint
            
            logger.info(f"📍 No checkpoint found for {job_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get checkpoint for {job_id}: {e}")
            return None
    
    async def get_all_checkpoints(self, job_id: str) -> List[Dict[str, Any]]:
        """Get all checkpoints for a job (ordered by creation time)"""
        try:
            result = self.supabase_client.client.table(self.checkpoints_table)\
                .select("*")\
                .eq("job_id", job_id)\
                .order("created_at", desc=False)\
                .execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"❌ Failed to get checkpoints for {job_id}: {e}")
            return []
    
    async def can_resume_from_checkpoint(self, job_id: str) -> tuple[bool, Optional[ProcessingStage]]:
        """
        Check if a job can be resumed from a checkpoint.
        
        Returns:
            (can_resume, last_stage)
        """
        checkpoint = await self.get_last_checkpoint(job_id)
        
        if not checkpoint:
            return False, None
        
        # Check if checkpoint is recent (within 24 hours)
        checkpoint_time = datetime.fromisoformat(normalize_timestamp(checkpoint["created_at"]))
        age = datetime.utcnow() - checkpoint_time.replace(tzinfo=None)
        
        if age > timedelta(hours=24):
            logger.warning(f"⚠️ Checkpoint for {job_id} is too old ({age.total_seconds() / 3600:.1f} hours)")
            return False, None
        
        stage = ProcessingStage(checkpoint["stage"])
        logger.info(f"✅ Job {job_id} can resume from {stage.value}")
        return True, stage
    
    async def detect_stuck_jobs(self, timeout_minutes: int = 30) -> List[Dict[str, Any]]:
        """
        Detect jobs that are stuck (processing for too long without progress).
        
        Args:
            timeout_minutes: Consider job stuck if processing longer than this
            
        Returns:
            List of stuck job records
        """
        try:
            cutoff_time = (datetime.utcnow() - timedelta(minutes=timeout_minutes)).isoformat()
            
            result = self.supabase_client.client.table(self.jobs_table)\
                .select("*")\
                .eq("status", "processing")\
                .lt("updated_at", cutoff_time)\
                .execute()
            
            stuck_jobs = result.data or []
            
            if stuck_jobs:
                logger.warning(f"🛑 Found {len(stuck_jobs)} stuck jobs (>{timeout_minutes}min without update)")
                for job in stuck_jobs:
                    logger.warning(f"   - {job['id']}: {job.get('filename', 'unknown')} (updated: {job['updated_at']})")
            
            return stuck_jobs
            
        except Exception as e:
            logger.error(f"❌ Failed to detect stuck jobs: {e}")
            return []
    
    async def auto_restart_stuck_job(self, job_id: str) -> bool:
        """
        Automatically restart a stuck job from last checkpoint.
        
        Args:
            job_id: Job identifier
            
        Returns:
            bool: True if restart initiated successfully
        """
        try:
            # Check if we can resume from checkpoint
            can_resume, last_stage = await self.can_resume_from_checkpoint(job_id)
            
            if not can_resume:
                logger.warning(f"⚠️ Cannot resume {job_id} from checkpoint - marking as failed")
                await self._mark_job_failed(job_id, "Stuck without valid checkpoint")
                return False
            
            # Mark job as pending for restart
            self.supabase_client.client.table(self.jobs_table)\
                .update({
                    "status": "pending",
                    "metadata": {
                        "restart_from_stage": last_stage.value,
                        "restart_reason": "auto_recovery_stuck_job",
                        "restart_at": datetime.utcnow().isoformat()
                    },
                    "updated_at": datetime.utcnow().isoformat()
                })\
                .eq("id", job_id)\
                .execute()
            
            logger.info(f"🔄 Marked {job_id} for restart from {last_stage.value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to restart stuck job {job_id}: {e}")
            return False
    
    async def verify_checkpoint_data(self, job_id: str, stage: ProcessingStage) -> bool:
        """
        Verify that checkpoint data is valid and database records exist.
        
        Args:
            job_id: Job identifier
            stage: Stage to verify
            
        Returns:
            bool: True if checkpoint data is valid
        """
        try:
            checkpoint = await self.get_last_checkpoint(job_id)
            
            if not checkpoint or checkpoint["stage"] != stage.value:
                return False
            
            data = checkpoint["checkpoint_data"]
            
            # Verify based on stage
            if stage == ProcessingStage.CHUNKS_CREATED:
                # Verify chunks exist in database
                chunk_ids = data.get("chunk_ids", [])
                chunks_created = data.get("chunks_created", 0)

                # Allow empty chunk_ids if chunks_created is 0 (focused extraction may skip chunking)
                if chunks_created == 0 and not chunk_ids:
                    logger.info(f"✅ Checkpoint valid: no chunks created (focused extraction)")
                    return True

                if not chunk_ids:
                    return False

                result = self.supabase_client.client.table("document_chunks")\
                    .select("id")\
                    .in_("id", chunk_ids)\
                    .execute()

                found_count = len(result.data or [])
                expected_count = len(chunk_ids)

                if found_count != expected_count:
                    logger.warning(f"⚠️ Checkpoint data mismatch: expected {expected_count} chunks, found {found_count}")
                    return False
            
            elif stage == ProcessingStage.TEXT_EMBEDDINGS_GENERATED:
                # Verify embeddings exist
                chunk_ids = data.get("chunk_ids", [])
                if not chunk_ids:
                    return False
                
                result = self.supabase_client.client.table("embeddings")\
                    .select("id")\
                    .in_("chunk_id", chunk_ids)\
                    .execute()
                
                found_count = len(result.data or [])
                expected_count = len(chunk_ids)
                
                if found_count < expected_count * 0.9:  # Allow 10% missing
                    logger.warning(f"⚠️ Too many missing embeddings: {found_count}/{expected_count}")
                    return False
            
            elif stage == ProcessingStage.IMAGES_EXTRACTED:
                # Verify images exist
                image_ids = data.get("image_ids", [])
                if not image_ids:
                    return True  # No images is valid
                
                result = self.supabase_client.client.table("document_images")\
                    .select("id")\
                    .in_("id", image_ids)\
                    .execute()
                
                found_count = len(result.data or [])
                expected_count = len(image_ids)
                
                if found_count != expected_count:
                    logger.warning(f"⚠️ Image data mismatch: expected {expected_count}, found {found_count}")
                    return False
            
            logger.info(f"✅ Checkpoint data verified for {job_id} @ {stage.value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to verify checkpoint data: {e}")
            return False
    
    async def cleanup_invalid_checkpoints(self, job_id: str) -> int:
        """
        Remove checkpoints with invalid data.
        
        Returns:
            Number of checkpoints removed
        """
        try:
            checkpoints = await self.get_all_checkpoints(job_id)
            removed = 0
            
            for checkpoint in checkpoints:
                stage = ProcessingStage(checkpoint["stage"])
                is_valid = await self.verify_checkpoint_data(job_id, stage)
                
                if not is_valid:
                    self.supabase_client.client.table(self.checkpoints_table)\
                        .delete()\
                        .eq("job_id", job_id)\
                        .eq("stage", stage.value)\
                        .execute()
                    
                    logger.info(f"🗑️ Removed invalid checkpoint: {job_id} @ {stage.value}")
                    removed += 1
            
            return removed
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup checkpoints: {e}")
            return 0
    
    async def _mark_job_failed(self, job_id: str, reason: str):
        """Mark a job as failed"""
        try:
            self.supabase_client.client.table(self.jobs_table)\
                .update({
                    "status": "failed",
                    "error": reason,
                    "failed_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                })\
                .eq("id", job_id)\
                .execute()
        except Exception as e:
            logger.error(f"Failed to mark job as failed: {e}")


# Global instance
checkpoint_recovery_service = CheckpointRecoveryService()

