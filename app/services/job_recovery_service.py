"""
Job Recovery Service

This service handles persistence and recovery of background processing jobs.
It ensures that interrupted jobs can be resumed after service restarts.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class JobRecoveryService:
    """
    Service for persisting and recovering background jobs.
    
    Features:
    - Persist job state to database
    - Recover interrupted jobs on startup
    - Track job lifecycle (pending, processing, completed, failed, interrupted)
    - Prevent duplicate job execution
    """
    
    def __init__(self, supabase_client):
        """
        Initialize job recovery service.
        
        Args:
            supabase_client: Supabase client instance
        """
        self.supabase = supabase_client
        self.table_name = "background_jobs"
        logger.info("JobRecoveryService initialized")
    
    async def persist_job(
        self,
        job_id: str,
        document_id: str,
        filename: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
        progress: int = 0,
        error: Optional[str] = None
    ) -> bool:
        """
        Persist job state to database.
        
        Args:
            job_id: Unique job identifier
            document_id: Document being processed
            filename: Original filename
            status: Job status (pending, processing, completed, failed, interrupted)
            metadata: Additional job metadata
            progress: Job progress percentage (0-100)
            error: Error message if job failed
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if job exists and get existing metadata
            existing = self.supabase.client.table(self.table_name).select("id, metadata").eq("id", job_id).execute()

            # Merge metadata with existing metadata instead of replacing
            merged_metadata = {}
            if existing.data and existing.data[0].get("metadata"):
                merged_metadata = existing.data[0]["metadata"].copy()
            if metadata:
                merged_metadata.update(metadata)

            job_data = {
                "id": job_id,
                "document_id": document_id,
                "filename": filename,
                "status": status,
                "progress": progress,
                "metadata": merged_metadata,
                "error": error,
                "updated_at": datetime.utcnow().isoformat()
            }

            if existing.data:
                # Update existing job
                self.supabase.client.table(self.table_name).update(job_data).eq("id", job_id).execute()
                logger.debug(f"Updated job {job_id} in database: status={status}, progress={progress}")
            else:
                # Insert new job
                job_data["created_at"] = datetime.utcnow().isoformat()
                self.supabase.client.table(self.table_name).insert(job_data).execute()
                logger.info(f"Persisted new job {job_id} to database: status={status}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to persist job {job_id}: {e}", exc_info=True)
            return False
    
    async def get_interrupted_jobs(self) -> List[Dict[str, Any]]:
        """
        Get all interrupted jobs that need recovery.
        
        Returns:
            List of interrupted job records
        """
        try:
            # Find jobs that were processing or pending when service stopped
            response = self.supabase.client.table(self.table_name).select("*").in_(
                "status", ["processing", "pending"]
            ).order("created_at").execute()
            
            interrupted_jobs = response.data or []
            
            if interrupted_jobs:
                logger.warning(f"🔄 Found {len(interrupted_jobs)} interrupted jobs that need recovery")
                for job in interrupted_jobs:
                    logger.warning(f"   - Job {job['id']}: Document {job['document_id']}, Status: {job['status']}")
            else:
                logger.info("✅ No interrupted jobs found")
            
            return interrupted_jobs
            
        except Exception as e:
            logger.error(f"Failed to get interrupted jobs: {e}", exc_info=True)
            return []
    
    async def mark_job_interrupted(self, job_id: str, reason: str = "Service shutdown") -> bool:
        """
        Mark a job as interrupted.
        
        Args:
            job_id: Job identifier
            reason: Reason for interruption
            
        Returns:
            bool: True if successful
        """
        try:
            self.supabase.client.table(self.table_name).update({
                "status": "interrupted",
                "error": reason,
                "interrupted_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", job_id).execute()
            
            logger.info(f"Marked job {job_id} as interrupted: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark job {job_id} as interrupted: {e}")
            return False
    
    async def mark_all_processing_as_interrupted(self, reason: str = "Service restart") -> int:
        """
        Mark all currently processing jobs as interrupted.
        This should be called on service startup.
        
        Args:
            reason: Reason for interruption
            
        Returns:
            int: Number of jobs marked as interrupted
        """
        try:
            # Get all processing jobs
            response = self.supabase.client.table(self.table_name).select("id").in_(
                "status", ["processing", "pending"]
            ).execute()
            
            jobs = response.data or []
            
            if not jobs:
                logger.info("✅ No processing jobs to mark as interrupted")
                return 0
            
            # Mark them all as interrupted
            job_ids = [job["id"] for job in jobs]
            
            self.supabase.client.table(self.table_name).update({
                "status": "interrupted",
                "error": reason,
                "interrupted_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).in_("id", job_ids).execute()
            
            logger.warning(f"🛑 Marked {len(job_ids)} jobs as interrupted due to: {reason}")
            for job_id in job_ids:
                logger.warning(f"   - Job {job_id}")
            
            return len(job_ids)
            
        except Exception as e:
            logger.error(f"Failed to mark processing jobs as interrupted: {e}", exc_info=True)
            return 0
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status from database.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job data or None if not found
        """
        try:
            response = self.supabase.client.table(self.table_name).select("*").eq("id", job_id).execute()
            
            if response.data:
                return response.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get job status for {job_id}: {e}")
            return None
    
    async def cleanup_old_jobs(self, days: int = 5) -> int:
        """
        Clean up completed/failed jobs older than specified days.

        Args:
            days: Number of days to keep jobs (default: 5 days)

        Returns:
            int: Number of jobs deleted
        """
        try:
            from datetime import timedelta

            cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

            # Delete old completed and failed jobs
            response = self.supabase.client.table(self.table_name).delete().in_(
                "status", ["completed", "failed"]
            ).lt("updated_at", cutoff_date).execute()

            deleted_count = len(response.data) if response.data else 0

            if deleted_count > 0:
                logger.info(f"🧹 Cleaned up {deleted_count} old jobs (older than {days} days)")

            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old jobs: {e}")
            return 0
    
    async def get_job_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about jobs in the system.
        
        Returns:
            Dictionary with job statistics
        """
        try:
            # Get counts by status
            response = self.supabase.client.table(self.table_name).select("status").execute()
            
            jobs = response.data or []
            
            stats = {
                "total": len(jobs),
                "pending": sum(1 for j in jobs if j["status"] == "pending"),
                "processing": sum(1 for j in jobs if j["status"] == "processing"),
                "completed": sum(1 for j in jobs if j["status"] == "completed"),
                "failed": sum(1 for j in jobs if j["status"] == "failed"),
                "interrupted": sum(1 for j in jobs if j["status"] == "interrupted"),
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get job statistics: {e}")
            return {
                "total": 0,
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "interrupted": 0,
                "error": str(e)
            }

