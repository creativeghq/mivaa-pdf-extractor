"""
Progress tracking service for PDF processing jobs.

This service provides real-time progress tracking, page-level status monitoring,
database integration status, and automatic persistence to Supabase.

Features:
- In-memory tracking for fast access
- Automatic database persistence (background_jobs + job_progress tables)
- Support for new 5-stage pipeline (discovery, extraction, chunking, images, products)
- Page-level status tracking
- Error and warning tracking
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from app.schemas.jobs import ProcessingStage, PageProcessingStatus, JobProgressDetail
from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


@dataclass
class ProgressTracker:
    """
    Thread-safe progress tracker for PDF processing jobs.

    Features:
    - In-memory tracking for fast access
    - Automatic database persistence to background_jobs and job_progress tables
    - Support for new 5-stage pipeline
    - Page-level status tracking
    """

    job_id: str
    document_id: str
    total_pages: int
    job_storage: Optional[Dict[str, Any]] = None  # Reference to global job_storage dict

    # Progress state
    current_stage: ProcessingStage = ProcessingStage.INITIALIZING
    pages_completed: int = 0
    pages_failed: int = 0
    pages_skipped: int = 0
    current_page: Optional[int] = None

    # Page-level tracking
    page_statuses: Dict[int, PageProcessingStatus] = field(default_factory=dict)

    # Timing
    processing_start_time: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None

    # Database tracking
    database_records_created: int = 0
    knowledge_base_entries: int = 0
    images_stored: int = 0
    chunks_created: int = 0
    products_created: int = 0

    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    # Performance metrics
    total_text_extracted: int = 0
    total_images_extracted: int = 0
    ocr_pages_processed: int = 0

    # Database client (initialized in __post_init__)
    _supabase: Any = field(default=None, init=False, repr=False)
    _db_sync_enabled: bool = field(default=True, init=False)

    def __post_init__(self):
        """Initialize page statuses and database client."""
        # Initialize Supabase client
        try:
            self._supabase = get_supabase_client()
            # Test if client is actually initialized by accessing the .client property
            _ = self._supabase.client
            logger.info(f"✅ Supabase client initialized for job {self.job_id}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize Supabase client: {e}")
            logger.warning(f"⚠️ Database sync will be disabled for job {self.job_id}")
            self._db_sync_enabled = False
            self._supabase = None

        # Initialize page statuses
        for page_num in range(1, self.total_pages + 1):
            self.page_statuses[page_num] = PageProcessingStatus(
                page_number=page_num,
                stage=ProcessingStage.INITIALIZING,
                status="pending"
            )

    async def _sync_to_database(self, stage: Optional[str] = None):
        """
        Sync progress to database (background_jobs + job_progress tables).

        Args:
            stage: Optional stage name for job_progress table
        """
        if not self._db_sync_enabled or not self._supabase:
            return

        try:
            # 1. Update background_jobs table
            progress_pct = self.calculate_progress_percentage()

            job_update = {
                'status': 'processing',
                'progress': int(progress_pct),
                'metadata': {
                    'current_stage': self.current_stage.value,
                    'pages_completed': self.pages_completed,
                    'pages_failed': self.pages_failed,
                    'pages_skipped': self.pages_skipped,
                    'database_records_created': self.database_records_created,
                    'knowledge_base_entries': self.knowledge_base_entries,
                    'images_stored': self.images_stored,
                    'chunks_created': self.chunks_created,
                    'products_created': self.products_created,
                    'errors_count': len(self.errors),
                    'warnings_count': len(self.warnings)
                },
                'updated_at': datetime.utcnow().isoformat()
            }

            self._supabase.client.table('background_jobs')\
                .update(job_update)\
                .eq('id', self.job_id)\
                .execute()

            # 2. Update job_progress table (if stage provided)
            if stage:
                progress_data = {
                    'document_id': self.document_id,
                    'stage': stage,
                    'progress': int(progress_pct),
                    'total_items': self.total_pages,
                    'completed_items': self.pages_completed,
                    'metadata': {
                        'current_page': self.current_page,
                        'images_extracted': self.total_images_extracted,
                        'chunks_created': self.chunks_created,
                        'products_created': self.products_created
                    },
                    'updated_at': datetime.utcnow().isoformat()
                }

                self._supabase.client.table('job_progress')\
                    .upsert(progress_data, on_conflict='document_id,stage')\
                    .execute()

            # 3. Update job_storage (in-memory)
            if self.job_storage and self.job_id in self.job_storage:
                self.job_storage[self.job_id]['status'] = 'processing'
                self.job_storage[self.job_id]['progress'] = int(progress_pct)
                self.job_storage[self.job_id]['current_stage'] = self.current_stage.value

                if 'metadata' not in self.job_storage[self.job_id]:
                    self.job_storage[self.job_id]['metadata'] = {}

                self.job_storage[self.job_id]['metadata'].update({
                    'pages_completed': self.pages_completed,
                    'pages_failed': self.pages_failed,
                    'images_stored': self.images_stored,
                    'chunks_created': self.chunks_created,
                    'products_created': self.products_created
                })

            logger.debug(f"📊 Synced progress to DB: {progress_pct:.0f}% - {self.current_stage.value}")

        except Exception as e:
            logger.error(f"❌ Failed to sync progress to database: {e}")
            # Don't raise - progress sync failures shouldn't block processing

    async def start_processing(self):
        """Mark processing as started."""
        self.processing_start_time = datetime.utcnow()
        self.current_stage = ProcessingStage.DOWNLOADING
        logger.info(f"Started processing job {self.job_id} with {self.total_pages} pages")

        # Sync to database
        await self._sync_to_database()

    async def update_stage(self, stage: ProcessingStage, stage_name: Optional[str] = None):
        """
        Update the current processing stage.

        Args:
            stage: ProcessingStage enum value
            stage_name: Optional stage name for job_progress table (e.g., 'product_discovery')
        """
        self.current_stage = stage
        logger.info(f"Job {self.job_id} moved to stage: {stage.value}")

        # Sync to database
        await self._sync_to_database(stage=stage_name)

    async def complete_page_processing(self, page_number: int,
                               text_extracted: bool = False,
                               images_extracted: int = 0,
                               ocr_applied: bool = False,
                               ocr_confidence: Optional[float] = None,
                               processing_time_ms: Optional[int] = None,
                               database_saved: bool = False,
                               sync_to_db: bool = True):
        """
        Mark a page as completed with detailed results.

        Args:
            sync_to_db: If True, sync progress to database after update
        """
        if page_number in self.page_statuses:
            page_status = self.page_statuses[page_number]
            page_status.status = "success"
            page_status.text_extracted = text_extracted
            page_status.images_extracted = images_extracted
            page_status.ocr_applied = ocr_applied
            page_status.ocr_confidence = ocr_confidence
            page_status.processing_time_ms = processing_time_ms
            page_status.database_saved = database_saved
            page_status.stage = ProcessingStage.COMPLETED

            self.pages_completed += 1
            if ocr_applied:
                self.ocr_pages_processed += 1
            self.total_images_extracted += images_extracted

            logger.info(f"Completed page {page_number} for job {self.job_id}")

            # Sync to database every 10 pages or if explicitly requested
            if sync_to_db and (self.pages_completed % 10 == 0 or sync_to_db):
                await self._sync_to_database()

    def fail_page_processing(self, page_number: int, error_message: str, stage: ProcessingStage):
        """Mark a page as failed."""
        if page_number in self.page_statuses:
            page_status = self.page_statuses[page_number]
            page_status.status = "failed"
            page_status.error_message = error_message
            page_status.stage = stage

            self.pages_failed += 1
            self.add_error(f"Page {page_number} Failed", error_message, {"page": page_number, "stage": stage.value})

            logger.error(f"Failed page {page_number} for job {self.job_id}: {error_message}")

    def skip_page_processing(self, page_number: int, reason: str):
        """Mark a page as skipped."""
        if page_number in self.page_statuses:
            page_status = self.page_statuses[page_number]
            page_status.status = "skipped"
            page_status.error_message = reason

            self.pages_skipped += 1
            self.add_warning(f"Page {page_number} Skipped", reason, {"page": page_number})

            logger.warning(f"Skipped page {page_number} for job {self.job_id}: {reason}")

    async def update_database_stats(
        self,
        records_created: int = 0,
        kb_entries: int = 0,
        images_stored: int = 0,
        chunks_created: int = 0,
        products_created: int = 0,
        sync_to_db: bool = True
    ):
        """
        Update database integration statistics.

        Args:
            sync_to_db: If True, sync progress to database after update
        """
        self.database_records_created += records_created
        self.knowledge_base_entries += kb_entries
        self.images_stored += images_stored
        self.chunks_created += chunks_created
        self.products_created += products_created

        logger.info(f"Updated database stats for job {self.job_id}: "
                   f"records={self.database_records_created}, kb={self.knowledge_base_entries}, "
                   f"images={self.images_stored}, chunks={self.chunks_created}, products={self.products_created}")

        # Sync to database
        if sync_to_db:
            await self._sync_to_database()

    def add_error(self, title: str, message: str, context: Dict[str, Any] = None):
        """Add an error to the tracking."""
        error = {
            "title": title,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "context": context or {}
        }
        self.errors.append(error)

    def add_warning(self, title: str, message: str, context: Dict[str, Any] = None):
        """Add a warning to the tracking."""
        warning = {
            "title": title,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "context": context or {}
        }
        self.warnings.append(warning)

    def calculate_progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_pages == 0:
            return 0.0

        completed_pages = self.pages_completed + self.pages_failed + self.pages_skipped
        return (completed_pages / self.total_pages) * 100.0

    def estimate_completion_time(self) -> Optional[datetime]:
        """Estimate completion time based on current progress."""
        if not self.processing_start_time or self.pages_completed == 0:
            return None

        elapsed = datetime.utcnow() - self.processing_start_time
        avg_time_per_page = elapsed.total_seconds() / self.pages_completed
        remaining_pages = self.total_pages - self.pages_completed - self.pages_failed - self.pages_skipped

        if remaining_pages <= 0:
            return datetime.utcnow()

        estimated_remaining = timedelta(seconds=avg_time_per_page * remaining_pages)
        return datetime.utcnow() + estimated_remaining

    def to_progress_detail(self) -> JobProgressDetail:
        """Convert to JobProgressDetail schema for API responses."""
        return JobProgressDetail(
            job_id=self.job_id,
            document_id=self.document_id,
            current_stage=self.current_stage,
            total_pages=self.total_pages,
            pages_completed=self.pages_completed,
            pages_failed=self.pages_failed,
            pages_skipped=self.pages_skipped,
            current_page=self.current_page,
            page_statuses=list(self.page_statuses.values()),
            progress_percentage=self.calculate_progress_percentage(),
            estimated_completion_time=self.estimate_completion_time(),
            processing_start_time=self.processing_start_time,
            database_records_created=self.database_records_created,
            knowledge_base_entries=self.knowledge_base_entries,
            images_stored=self.images_stored,
            errors=self.errors,
            warnings=self.warnings,
            average_page_processing_time=None,  # Calculate if needed
            ocr_pages_processed=self.ocr_pages_processed,
            total_text_extracted=self.total_text_extracted,
            total_images_extracted=self.total_images_extracted
        )

    async def complete_job(self, result: Dict[str, Any]):
        """
        Mark job as complete and sync final state to database.

        Args:
            result: Final result data
        """
        self.current_stage = ProcessingStage.COMPLETED

        try:
            # Update background_jobs table
            if self._db_sync_enabled and self._supabase:
                self._supabase.client.table('background_jobs')\
                    .update({
                        'status': 'completed',
                        'progress': 100,
                        'metadata': {
                            'pages_completed': self.pages_completed,
                            'pages_failed': self.pages_failed,
                            'pages_skipped': self.pages_skipped,
                            'database_records_created': self.database_records_created,
                            'knowledge_base_entries': self.knowledge_base_entries,
                            'images_stored': self.images_stored,
                            'chunks_created': self.chunks_created,
                            'products_created': self.products_created,
                            'errors_count': len(self.errors),
                            'warnings_count': len(self.warnings),
                            'result': result  # Store result in metadata instead
                        },
                        'completed_at': datetime.utcnow().isoformat(),
                        'updated_at': datetime.utcnow().isoformat()
                    })\
                    .eq('id', self.job_id)\
                    .execute()

            # Update job_storage
            if self.job_storage and self.job_id in self.job_storage:
                self.job_storage[self.job_id]['status'] = 'completed'
                self.job_storage[self.job_id]['progress'] = 100
                self.job_storage[self.job_id]['result'] = result
                self.job_storage[self.job_id]['completed_at'] = datetime.utcnow().isoformat()

            logger.info(f"✅ [JOB {self.job_id}] Completed successfully")

        except Exception as e:
            logger.error(f"❌ Failed to mark job as complete: {e}")

    async def fail_job(self, error: Exception):
        """
        Mark job as failed and sync final state to database.

        Args:
            error: Exception that caused failure
        """
        self.current_stage = ProcessingStage.FAILED
        error_message = str(error)

        try:
            # Update background_jobs table
            if self._db_sync_enabled and self._supabase:
                self._supabase.client.table('background_jobs')\
                    .update({
                        'status': 'failed',
                        'progress': int(self.calculate_progress_percentage()),
                        'error': error_message,
                        'metadata': {
                            'pages_completed': self.pages_completed,
                            'pages_failed': self.pages_failed,
                            'errors': self.errors,
                            'warnings': self.warnings
                        },
                        'failed_at': datetime.utcnow().isoformat(),
                        'updated_at': datetime.utcnow().isoformat()
                    })\
                    .eq('id', self.job_id)\
                    .execute()

            # Update job_storage
            if self.job_storage and self.job_id in self.job_storage:
                self.job_storage[self.job_id]['status'] = 'failed'
                self.job_storage[self.job_id]['error'] = error_message
                self.job_storage[self.job_id]['failed_at'] = datetime.utcnow().isoformat()

            logger.error(f"❌ [JOB {self.job_id}] Failed: {error_message}")

        except Exception as e:
            logger.error(f"❌ Failed to mark job as failed: {e}")


class ProgressTrackingService:
    """Service for managing multiple progress trackers."""

    def __init__(self):
        self.trackers: Dict[str, ProgressTracker] = {}
        self.logger = logging.getLogger(__name__)

    def create_tracker(self, job_id: str, document_id: str, total_pages: int) -> ProgressTracker:
        """Create a new progress tracker for a job."""
        tracker = ProgressTracker(job_id=job_id, document_id=document_id, total_pages=total_pages)
        self.trackers[job_id] = tracker
        self.logger.info(f"Created progress tracker for job {job_id}")
        return tracker

    def get_tracker(self, job_id: str) -> Optional[ProgressTracker]:
        """Get a progress tracker by job ID."""
        return self.trackers.get(job_id)

    def get_all_active_trackers(self) -> Dict[str, ProgressTracker]:
        """Get all active progress trackers."""
        active = {}
        for job_id, tracker in self.trackers.items():
            if tracker.current_stage not in [ProcessingStage.COMPLETED, ProcessingStage.FAILED]:
                active[job_id] = tracker
        return active

    def remove_tracker(self, job_id: str):
        """Remove a completed tracker."""
        if job_id in self.trackers:
            del self.trackers[job_id]
            self.logger.info(f"Removed progress tracker for job {job_id}")

    def cleanup_old_trackers(self, max_age_hours: int = 24):
        """Clean up old completed trackers."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        to_remove = []

        for job_id, tracker in self.trackers.items():
            if (tracker.current_stage in [ProcessingStage.COMPLETED, ProcessingStage.FAILED] and
                tracker.processing_start_time and tracker.processing_start_time < cutoff_time):
                to_remove.append(job_id)

        for job_id in to_remove:
            self.remove_tracker(job_id)

        if to_remove:
            self.logger.info(f"Cleaned up {len(to_remove)} old progress trackers")


# Global progress tracking service instance
progress_service = ProgressTrackingService()


def get_progress_service() -> ProgressTrackingService:
    """Get the global progress tracking service instance."""
    return progress_service
