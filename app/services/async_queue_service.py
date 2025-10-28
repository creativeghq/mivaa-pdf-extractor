"""
Async Queue Service for PDF Processing

Manages async job queuing for image processing and AI analysis stages.
Uses Supabase tables for persistence and real-time monitoring.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class AsyncQueueService:
    """Service for managing async job queues"""

    def __init__(self):
        self.supabase = get_supabase_client()

    async def queue_image_processing_jobs(
        self,
        document_id: str,
        images: List[Dict[str, Any]],
        priority: int = 0
    ) -> int:
        """
        Queue image processing jobs for all extracted images.
        
        Args:
            document_id: Document ID
            images: List of image data dicts with 'id' and 'path'
            priority: Job priority (0 = normal)
            
        Returns:
            Number of jobs queued
        """
        try:
            jobs = []
            for image in images:
                job = {
                    'id': str(uuid.uuid4()),
                    'document_id': document_id,
                    'image_id': image.get('id'),
                    'status': 'pending',
                    'priority': priority,
                    'retry_count': 0,
                    'max_retries': 3,
                    'created_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat()
                }
                jobs.append(job)

            if jobs:
                self.supabase.table('image_processing_queue').insert(jobs).execute()
                logger.info(f"✅ Queued {len(jobs)} image processing jobs for document {document_id}")

            return len(jobs)

        except Exception as e:
            logger.error(f"❌ Failed to queue image processing jobs: {e}")
            raise

    async def queue_ai_analysis_jobs(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        analysis_type: str = 'classification',
        priority: int = 0
    ) -> int:
        """
        Queue AI analysis jobs for all chunks.
        
        Args:
            document_id: Document ID
            chunks: List of chunk data dicts with 'id'
            analysis_type: Type of analysis (classification, metadata, product_detection)
            priority: Job priority (0 = normal)
            
        Returns:
            Number of jobs queued
        """
        try:
            jobs = []
            for chunk in chunks:
                job = {
                    'id': str(uuid.uuid4()),
                    'document_id': document_id,
                    'chunk_id': chunk.get('id'),
                    'analysis_type': analysis_type,
                    'status': 'pending',
                    'priority': priority,
                    'retry_count': 0,
                    'created_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat()
                }
                jobs.append(job)

            if jobs:
                self.supabase.table('ai_analysis_queue').insert(jobs).execute()
                logger.info(f"✅ Queued {len(jobs)} AI analysis jobs for document {document_id}")

            return len(jobs)

        except Exception as e:
            logger.error(f"❌ Failed to queue AI analysis jobs: {e}")
            raise

    async def update_progress(
        self,
        document_id: str,
        stage: str,
        progress: int,
        total_items: Optional[int] = None,
        completed_items: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update progress for a document processing stage.
        
        Args:
            document_id: Document ID
            stage: Processing stage (extraction, image_processing, chunking, ai_analysis, product_creation)
            progress: Progress percentage (0-100)
            total_items: Total items to process
            completed_items: Items completed so far
            metadata: Additional metadata
        """
        try:
            data = {
                'document_id': document_id,
                'stage': stage,
                'progress': min(100, max(0, progress)),
                'updated_at': datetime.utcnow().isoformat()
            }

            if total_items is not None:
                data['total_items'] = total_items

            if completed_items is not None:
                data['completed_items'] = completed_items

            if metadata:
                data['metadata'] = metadata

            self.supabase.table('job_progress').upsert(
                data,
                on_conflict='document_id,stage'
            ).execute()

            logger.debug(f"📊 Updated progress for {document_id}/{stage}: {progress}%")

        except Exception as e:
            logger.error(f"❌ Failed to update progress: {e}")
            # Don't raise - progress updates shouldn't block processing

    async def get_queue_metrics(self) -> Dict[str, Any]:
        """
        Get current queue metrics for monitoring dashboard.
        
        Returns:
            Dictionary with queue statistics
        """
        try:
            # Get image queue metrics
            image_queue = self.supabase.table('image_processing_queue').select(
                'status, count'
            ).execute()

            # Get AI queue metrics
            ai_queue = self.supabase.table('ai_analysis_queue').select(
                'status, count'
            ).execute()

            # Get active documents
            progress = self.supabase.table('job_progress').select(
                'document_id'
            ).eq('progress', '<', 100).execute()

            return {
                'image_queue': image_queue.data if image_queue.data else [],
                'ai_queue': ai_queue.data if ai_queue.data else [],
                'active_documents': len(set(p['document_id'] for p in progress.data)) if progress.data else 0
            }

        except Exception as e:
            logger.error(f"❌ Failed to get queue metrics: {e}")
            return {
                'image_queue': [],
                'ai_queue': [],
                'active_documents': 0
            }

    async def mark_job_failed(
        self,
        job_id: str,
        queue_type: str,
        error_message: str,
        retry_count: int
    ) -> None:
        """
        Mark a job as failed and handle retry logic.
        
        Args:
            job_id: Job ID
            queue_type: 'image' or 'ai'
            error_message: Error message
            retry_count: Current retry count
        """
        try:
            table_name = 'image_processing_queue' if queue_type == 'image' else 'ai_analysis_queue'

            if retry_count < 3:
                # Re-queue for retry
                self.supabase.table(table_name).update({
                    'status': 'pending',
                    'retry_count': retry_count + 1,
                    'error_message': error_message,
                    'updated_at': datetime.utcnow().isoformat()
                }).eq('id', job_id).execute()

                logger.info(f"🔄 Re-queued job {job_id} for retry (attempt {retry_count + 1}/3)")
            else:
                # Mark as permanently failed
                self.supabase.table(table_name).update({
                    'status': 'failed',
                    'error_message': f"Max retries exceeded: {error_message}",
                    'updated_at': datetime.utcnow().isoformat()
                }).eq('id', job_id).execute()

                logger.error(f"❌ Job {job_id} failed permanently after 3 retries")

        except Exception as e:
            logger.error(f"❌ Failed to mark job as failed: {e}")


# Singleton instance
_async_queue_service: Optional[AsyncQueueService] = None


def get_async_queue_service() -> AsyncQueueService:
    """Get or create async queue service instance"""
    global _async_queue_service
    if _async_queue_service is None:
        _async_queue_service = AsyncQueueService()
    return _async_queue_service

