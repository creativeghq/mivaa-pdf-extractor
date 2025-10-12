"""
Progress tracking service for PDF processing jobs.

This service provides real-time progress tracking, page-level status monitoring,
and database integration status for OCR and PDF processing operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from app.schemas.jobs import ProcessingStage, PageProcessingStatus, JobProgressDetail

logger = logging.getLogger(__name__)


@dataclass
class ProgressTracker:
    """Thread-safe progress tracker for PDF processing jobs."""

    job_id: str
    document_id: str
    total_pages: int

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

    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    # Performance metrics
    total_text_extracted: int = 0
    total_images_extracted: int = 0
    ocr_pages_processed: int = 0

    def __post_init__(self):
        """Initialize page statuses."""
        for page_num in range(1, self.total_pages + 1):
            self.page_statuses[page_num] = PageProcessingStatus(
                page_number=page_num,
                stage=ProcessingStage.INITIALIZING,
                status="pending"
            )

    def start_processing(self):
        """Mark processing as started."""
        self.processing_start_time = datetime.utcnow()
        self.current_stage = ProcessingStage.DOWNLOADING
        logger.info(f"Started processing job {self.job_id} with {self.total_pages} pages")

    def update_stage(self, stage: ProcessingStage):
        """Update the current processing stage."""
        self.current_stage = stage
        logger.info(f"Job {self.job_id} moved to stage: {stage.value}")

    def complete_page_processing(self, page_number: int,
                               text_extracted: bool = False,
                               images_extracted: int = 0,
                               ocr_applied: bool = False,
                               ocr_confidence: Optional[float] = None,
                               processing_time_ms: Optional[int] = None,
                               database_saved: bool = False):
        """Mark a page as completed with detailed results."""
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

    def update_database_stats(self, records_created: int = 0, kb_entries: int = 0, images_stored: int = 0):
        """Update database integration statistics."""
        self.database_records_created += records_created
        self.knowledge_base_entries += kb_entries
        self.images_stored += images_stored

        logger.info(f"Updated database stats for job {self.job_id}: "
                   f"records={self.database_records_created}, kb={self.knowledge_base_entries}, images={self.images_stored}")

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
    
    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    total_text_extracted: int = 0
    total_images_extracted: int = 0
    ocr_pages_processed: int = 0
    
    def __post_init__(self):
        """Initialize page statuses."""
        for page_num in range(1, self.total_pages + 1):
            self.page_statuses[page_num] = PageProcessingStatus(
                page_number=page_num,
                stage=ProcessingStage.PENDING,
                status="pending"
            )
    
    def start_processing(self):
        """Mark processing as started."""
        self.processing_start_time = datetime.utcnow()
        self.current_stage = ProcessingStage.DOWNLOADING
        logger.info(f"Started processing job {self.job_id} with {self.total_pages} pages")
    
    def update_stage(self, stage: ProcessingStage):
        """Update the current processing stage."""
        self.current_stage = stage
        logger.info(f"Job {self.job_id} moved to stage: {stage.value}")
    
    def start_page_processing(self, page_number: int, stage: ProcessingStage):
        """Start processing a specific page."""
        self.current_page = page_number
        if page_number in self.page_statuses:
            self.page_statuses[page_number].stage = stage
            self.page_statuses[page_number].status = "processing"
        logger.debug(f"Started processing page {page_number} at stage {stage.value}")
    
    def complete_page_processing(self, page_number: int, 
                               text_extracted: bool = False,
                               images_extracted: int = 0,
                               ocr_applied: bool = False,
                               ocr_confidence: Optional[float] = None,
                               processing_time_ms: Optional[int] = None,
                               database_saved: bool = False):
        """Mark a page as completed with detailed results."""
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
        if images_extracted > 0:
            self.total_images_extracted += images_extracted
        if ocr_applied:
            self.ocr_pages_processed += 1
        
        logger.info(f"Completed page {page_number}: text={text_extracted}, images={images_extracted}, ocr={ocr_applied}")
    
    def fail_page_processing(self, page_number: int, error_message: str, stage: ProcessingStage):
        """Mark a page as failed."""
        if page_number in self.page_statuses:
            page_status = self.page_statuses[page_number]
            page_status.status = "failed"
            page_status.error_message = error_message
            page_status.stage = stage
        
        self.pages_failed += 1
        self.add_error(f"Page {page_number} failed", error_message, {"page": page_number, "stage": stage.value})
        logger.error(f"Failed processing page {page_number}: {error_message}")
    
    def skip_page_processing(self, page_number: int, reason: str):
        """Mark a page as skipped."""
        if page_number in self.page_statuses:
            page_status = self.page_statuses[page_number]
            page_status.status = "skipped"
            page_status.error_message = reason
        
        self.pages_skipped += 1
        self.add_warning(f"Page {page_number} skipped", reason, {"page": page_number})
        logger.warning(f"Skipped page {page_number}: {reason}")
    
    def add_error(self, title: str, message: str, context: Optional[Dict[str, Any]] = None):
        """Add an error to the tracking."""
        self.errors.append({
            "timestamp": datetime.utcnow().isoformat(),
            "title": title,
            "message": message,
            "context": context or {}
        })
    
    def add_warning(self, title: str, message: str, context: Optional[Dict[str, Any]] = None):
        """Add a warning to the tracking."""
        self.warnings.append({
            "timestamp": datetime.utcnow().isoformat(),
            "title": title,
            "message": message,
            "context": context or {}
        })
    
    def update_database_stats(self, records_created: int = 0, kb_entries: int = 0, images_stored: int = 0):
        """Update database integration statistics."""
        self.database_records_created += records_created
        self.knowledge_base_entries += kb_entries
        self.images_stored += images_stored
        logger.info(f"Database stats updated: records={records_created}, kb_entries={kb_entries}, images={images_stored}")
    
    def update_text_stats(self, characters_extracted: int):
        """Update text extraction statistics."""
        self.total_text_extracted += characters_extracted
    
    def calculate_progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_pages == 0:
            return 0.0
        
        completed_weight = 0.8  # 80% for page completion
        db_weight = 0.2        # 20% for database operations
        
        page_progress = (self.pages_completed / self.total_pages) * completed_weight
        
        # Estimate database progress based on completed pages
        expected_db_operations = self.pages_completed * 2  # Rough estimate
        actual_db_operations = self.database_records_created + self.knowledge_base_entries
        db_progress = min(1.0, actual_db_operations / max(1, expected_db_operations)) * db_weight
        
        return min(100.0, (page_progress + db_progress) * 100)
    
    def estimate_completion_time(self) -> Optional[datetime]:
        """Estimate completion time based on current progress."""
        if not self.processing_start_time or self.pages_completed == 0:
            return None
        
        elapsed = datetime.utcnow() - self.processing_start_time
        avg_time_per_page = elapsed.total_seconds() / self.pages_completed
        remaining_pages = self.total_pages - self.pages_completed
        
        estimated_remaining_seconds = remaining_pages * avg_time_per_page
        return datetime.utcnow() + timedelta(seconds=estimated_remaining_seconds)
    
    def get_average_page_processing_time(self) -> Optional[float]:
        """Get average processing time per page in seconds."""
        if not self.processing_start_time or self.pages_completed == 0:
            return None
        
        elapsed = datetime.utcnow() - self.processing_start_time
        return elapsed.total_seconds() / self.pages_completed
    
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
            average_page_processing_time=self.get_average_page_processing_time(),
            ocr_pages_processed=self.ocr_pages_processed,
            total_text_extracted=self.total_text_extracted,
            total_images_extracted=self.total_images_extracted
        )


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
    
    def remove_tracker(self, job_id: str):
        """Remove a progress tracker (cleanup after job completion)."""
        if job_id in self.trackers:
            del self.trackers[job_id]
            self.logger.info(f"Removed progress tracker for job {job_id}")
    
    def get_all_active_trackers(self) -> Dict[str, ProgressTracker]:
        """Get all active progress trackers."""
        return self.trackers.copy()


# Global progress tracking service instance
progress_service = ProgressTrackingService()


def get_progress_service() -> ProgressTrackingService:
    """Get the global progress tracking service."""
    return progress_service
