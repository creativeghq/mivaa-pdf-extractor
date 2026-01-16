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
from app.services.core.supabase_client import get_supabase_client
from app.utils.retry_helper import async_retry_with_backoff

logger = logging.getLogger(__name__)

# Import Sentry for crash alerts
try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    logger.warning("Sentry SDK not available - crash alerts disabled")


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
    job_type: str = "pdf_processing"  # Job type for Sentry tracking

    # Progress state
    current_stage: ProcessingStage = ProcessingStage.INITIALIZING
    pages_completed: int = 0
    pages_failed: int = 0
    pages_skipped: int = 0
    current_page: Optional[int] = None
    manual_progress_override: Optional[int] = None  # Manual progress percentage (0-100)

    # Detailed progress tracking for UI
    current_step: Optional[str] = None  # e.g., "Processing chunks", "Generating embeddings"
    progress_current: int = 0  # Current item being processed
    progress_total: int = 0  # Total items to process

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
    text_embeddings_generated: int = 0  # NEW: Track text embeddings separately
    image_embeddings_generated: int = 0  # NEW: Track image embeddings separately
    clip_embeddings_generated: int = 0  # Track CLIP/SigLIP embeddings separately
    total_images_extracted: int = 0  # Total images found in PDF (including non-material)
    relations_created: int = 0  # Track entity relationships created

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

    # Heartbeat monitoring
    _heartbeat_task: Optional[Any] = field(default=None, init=False, repr=False)
    _heartbeat_running: bool = field(default=False, init=False)
    last_heartbeat: Optional[datetime] = None
    last_db_sync: Optional[datetime] = None
    MIN_SYNC_INTERVAL: float = 2.0  # Minimum seconds between database syncs (reduced from 5.0 for more responsive updates)

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

    @async_retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_multiplier=2.0, max_delay=10.0)
    async def _sync_to_database(self, stage: Optional[str] = None, force: bool = False):
        """
        Sync progress to database (background_jobs + job_progress tables).

        Args:
            force: Force sync even if within debounce interval (default: False)
            stage: Optional stage name for job_progress table
        """
        if not self._db_sync_enabled or not self._supabase:
            return
        # Debouncing: Skip sync if called too frequently (unless forced)
        if not force and self.last_db_sync:
            time_since_last_sync = (datetime.utcnow() - self.last_db_sync).total_seconds()
            if time_since_last_sync < self.MIN_SYNC_INTERVAL:
                logger.debug(f"⏭️  Skipping DB sync (debounced): {time_since_last_sync:.1f}s < {self.MIN_SYNC_INTERVAL}s")
                return


        try:
            # 1. Update background_jobs table
            # Use manual progress override if set, otherwise calculate from pages
            if self.manual_progress_override is not None:
                progress_pct = self.manual_progress_override
            else:
                progress_pct = self.calculate_progress_percentage()

            job_update = {
                'status': 'processing',
                'progress': int(progress_pct),
                'metadata': {
                    'current_stage': self.current_stage.value,
                    'stage': self.current_stage.value,  # Alias for UI compatibility
                    'current_step': self.current_step,  # Detailed step description for UI
                    'progress_current': self.progress_current,  # Current item number for UI
                    'progress_total': self.progress_total,  # Total items for UI
                    'total_pages': self.total_pages,
                    'pages_completed': self.pages_completed,
                    'pages_failed': self.pages_failed,
                    'pages_skipped': self.pages_skipped,
                    'database_records_created': self.database_records_created,
                    'knowledge_base_entries': self.knowledge_base_entries,
                    'images_extracted': self.images_stored,  # Material images saved to DB
                    'total_images_extracted': self.total_images_extracted,  # All images found in PDF
                    'chunks_created': self.chunks_created,
                    'products_created': self.products_created,
                    'relations_created': self.relations_created,  # Entity relationships created
                    'embeddings_generated': self.text_embeddings_generated + self.image_embeddings_generated,  # Total embeddings (backward compatibility)
                    'text_embeddings_generated': self.text_embeddings_generated,  # NEW: Separate text embeddings count
                    'image_embeddings_generated': self.image_embeddings_generated,  # NEW: Separate image embeddings count
                    'clip_embeddings_generated': self.image_embeddings_generated,  # Alias for backward compatibility
                    'clip_embeddings': self.clip_embeddings_generated,  # CLIP/SigLIP embeddings count
                    'ocr_pages_processed': self.ocr_pages_processed,
                    'total_text_extracted': self.total_text_extracted,
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
                        'total_pages': self.total_pages,
                        'images_extracted': self.images_stored,  # Material images saved to DB
                        'total_images_extracted': self.total_images_extracted,  # All images found
                        'chunks_created': self.chunks_created,
                        'products_created': self.products_created,
                        'embeddings_generated': self.text_embeddings_generated + self.image_embeddings_generated,  # Total (backward compatibility)
                        'text_embeddings_generated': self.text_embeddings_generated,  # NEW: Separate text embeddings
                        'image_embeddings_generated': self.image_embeddings_generated,  # NEW: Separate image embeddings
                        'clip_embeddings_generated': self.image_embeddings_generated,  # Alias for backward compatibility
                        'ocr_pages_processed': self.ocr_pages_processed
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
                    'images_extracted': self.images_stored,  # Use images_extracted for API compatibility
                    'chunks_created': self.chunks_created,
                    'products_created': self.products_created
                })

            logger.debug(f"📊 Synced progress to DB: {progress_pct:.0f}% - {self.current_stage.value}")

            # Update last sync timestamp
            self.last_db_sync = datetime.utcnow()
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

    async def update_stage(self, stage: ProcessingStage, stage_name: Optional[str] = None, progress_percentage: Optional[int] = None):
        """
        Update the current processing stage.

        Args:
            stage: ProcessingStage enum value
            stage_name: Optional stage name for job_progress table (e.g., 'product_discovery')
            progress_percentage: Optional manual progress percentage (0-100) to override calculated progress
        """
        self.current_stage = stage
        if progress_percentage is not None:
            self.manual_progress_override = progress_percentage
        logger.info(f"Job {self.job_id} moved to stage: {stage.value}" + (f" ({progress_percentage}%)" if progress_percentage is not None else ""))

        # Sync to database
        await self._sync_to_database(stage=stage_name)

    async def update_progress(
        self,
        progress_percentage: int,
        details: Optional[Dict[str, Any]] = None,
        sync_to_db: bool = True
    ):
        """
        Update progress with a manual percentage override and optional details.

        Args:
            progress_percentage: Progress percentage (0-100)
            details: Optional dictionary with progress details (e.g., current_step, products_completed)
            sync_to_db: Whether to sync to database
        """
        self.manual_progress_override = progress_percentage

        # Update detailed progress fields if provided
        if details:
            if "current_step" in details:
                self.current_step = details["current_step"]
            if "products_completed" in details:
                self.products_created = details.get("products_completed", 0)

        logger.debug(f"Job {self.job_id}: Progress {progress_percentage}% - {details}")

        # Sync to database
        if sync_to_db:
            await self._sync_to_database()

    async def update_detailed_progress(
        self,
        current_step: str,
        progress_current: int,
        progress_total: int,
        sync_to_db: bool = True
    ):
        """
        Update detailed progress for UI display (e.g., "Processing chunks 10/100").

        Args:
            current_step: Description of current step (e.g., "Processing chunks", "Generating embeddings")
            progress_current: Current item being processed
            progress_total: Total items to process
            sync_to_db: Whether to sync to database
        """
        self.current_step = current_step
        self.progress_current = progress_current
        self.progress_total = progress_total

        logger.debug(f"Job {self.job_id}: {current_step} ({progress_current}/{progress_total})")

        # Sync to database (with debouncing)
        if sync_to_db:
            await self._sync_to_database()

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
        clip_embeddings: int = 0,
        text_embeddings: int = 0,  # NEW: Separate text embeddings count
        image_embeddings: int = 0,  # NEW: Separate image embeddings count
        relations_created: int = 0,  # Entity relationships created
        total_images_extracted: int = 0,
        sync_to_db: bool = True
    ):
        """
        Update database integration statistics.

        Args:
            text_embeddings: Number of text embeddings generated (from chunks)
            image_embeddings: Number of image embeddings generated (CLIP/SigLIP)
            relations_created: Number of entity relationships created
            sync_to_db: If True, sync progress to database after update
        """
        self.database_records_created += records_created
        self.knowledge_base_entries += kb_entries
        self.images_stored += images_stored
        self.chunks_created += chunks_created
        self.products_created += products_created
        self.clip_embeddings_generated += clip_embeddings
        self.text_embeddings_generated += text_embeddings  # NEW: Track text embeddings
        self.image_embeddings_generated += image_embeddings  # NEW: Track image embeddings
        self.relations_created += relations_created  # Track relationships
        if total_images_extracted > 0:
            self.total_images_extracted = total_images_extracted

        logger.info(f"Updated database stats for job {self.job_id}: "
                   f"records={self.database_records_created}, kb={self.knowledge_base_entries}, "
                   f"images={self.images_stored}, chunks={self.chunks_created}, products={self.products_created}, "
                   f"text_emb={self.text_embeddings_generated}, image_emb={self.image_embeddings_generated}, "
                   f"relations={self.relations_created}")

        # Sync to database
        if sync_to_db:
            await self._sync_to_database()

    async def sync_counts_from_database(self):
        """
        Query database for actual counts and update tracker to match reality.

        This ensures job metadata reflects actual database state, not just in-memory tracking.
        Should be called after each major stage completes.
        """
        if not self._supabase:
            logger.warning("Cannot sync counts from database - Supabase client not initialized")
            return

        try:
            logger.info(f"🔄 Syncing counts from database for document {self.document_id}")

            # Query actual counts from database
            # 1. Count chunks
            chunks_result = self._supabase.client.table('document_chunks')\
                .select('id', count='exact')\
                .eq('document_id', self.document_id)\
                .execute()
            actual_chunks = chunks_result.count if chunks_result.count is not None else 0

            # 2. Count images
            images_result = self._supabase.client.table('document_images')\
                .select('id', count='exact')\
                .eq('document_id', self.document_id)\
                .execute()
            actual_images = images_result.count if images_result.count is not None else 0

            # 3. Count products
            products_result = self._supabase.client.table('products')\
                .select('id', count='exact')\
                .eq('source_document_id', self.document_id)\
                .execute()
            actual_products = products_result.count if products_result.count is not None else 0

            # 4. Count embeddings (stored directly in document_chunks.text_embedding)
            embeddings_result = self._supabase.client.table('document_chunks')\
                .select('id', count='exact')\
                .eq('document_id', self.document_id)\
                .not_.is_('text_embedding', 'null')\
                .execute()
            actual_embeddings = embeddings_result.count if embeddings_result.count is not None else 0

            # Log differences if any
            if actual_chunks != self.chunks_created:
                logger.warning(f"⚠️ Chunk count mismatch: tracker={self.chunks_created}, DB={actual_chunks}")
            if actual_images != self.images_stored:
                logger.warning(f"⚠️ Image count mismatch: tracker={self.images_stored}, DB={actual_images}")
            if actual_products != self.products_created:
                logger.warning(f"⚠️ Product count mismatch: tracker={self.products_created}, DB={actual_products}")

            # Update tracker with actual counts
            self.chunks_created = actual_chunks
            self.images_stored = actual_images
            self.products_created = actual_products

            logger.info(f"✅ Synced counts from database:")
            logger.info(f"   Chunks: {actual_chunks}")
            logger.info(f"   Images: {actual_images}")
            logger.info(f"   Products: {actual_products}")
            logger.info(f"   Embeddings: {actual_embeddings}")

            # Sync updated counts to database
            await self._sync_to_database()

        except Exception as e:
            logger.error(f"❌ Failed to sync counts from database: {e}")
            # Don't raise - count sync failures shouldn't block processing

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

        # Stop heartbeat when job completes
        await self.stop_heartbeat()

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
                            'images_extracted': self.images_stored,  # Use images_extracted for API compatibility
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
        Sends crash alert to Sentry for monitoring.

        Args:
            error: Exception that caused failure
        """
        self.current_stage = ProcessingStage.FAILED
        error_message = str(error)

        # Stop heartbeat when job fails
        await self.stop_heartbeat()

        # 🚨 SENTRY ALERT: Send crash alert for job failure
        if SENTRY_AVAILABLE:
            try:
                with sentry_sdk.configure_scope() as scope:
                    # Add job context
                    scope.set_tag("job_id", self.job_id)
                    scope.set_tag("document_id", self.document_id)
                    scope.set_tag("job_type", self.job_type)
                    scope.set_tag("current_stage", self.current_stage.value if hasattr(self.current_stage, 'value') else str(self.current_stage))
                    scope.set_tag("error_type", "job_crash")

                    # Add job progress context
                    scope.set_context("job_progress", {
                        "job_id": self.job_id,
                        "document_id": self.document_id,
                        "job_type": self.job_type,
                        "progress_percentage": self.calculate_progress_percentage(),
                        "pages_completed": self.pages_completed,
                        "pages_failed": self.pages_failed,
                        "total_pages": self.total_pages,
                        "errors_count": len(self.errors),
                        "warnings_count": len(self.warnings)
                    })

                    # Add error details
                    scope.set_context("error_details", {
                        "error_message": error_message,
                        "error_type": type(error).__name__,
                        "recent_errors": self.errors[-5:] if self.errors else [],
                        "recent_warnings": self.warnings[-5:] if self.warnings else []
                    })

                # Capture exception with full context
                sentry_sdk.capture_exception(
                    error,
                    level="error",
                    fingerprint=["job-crash", self.job_type, type(error).__name__]
                )

                logger.info(f"📊 Sent crash alert to Sentry for job {self.job_id}")

            except Exception as sentry_error:
                logger.warning(f"Failed to send Sentry alert: {sentry_error}")

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

    async def start_heartbeat(self, interval_seconds: int = 30):
        """
        Start sending heartbeat signals every interval_seconds to prove job is alive.

        This allows the job monitor to detect crashes within 2 missed heartbeats (60s)
        instead of waiting for the full stuck_job_timeout (5min).

        Args:
            interval_seconds: How often to send heartbeat (default: 30s)
        """
        if self._heartbeat_running:
            logger.warning(f"Heartbeat already running for job {self.job_id}")
            return

        self._heartbeat_running = True
        logger.info(f"🫀 Starting heartbeat for job {self.job_id} (interval: {interval_seconds}s)")

        async def heartbeat_loop():
            """Background task that sends heartbeat signals"""
            import asyncio

            while self._heartbeat_running:
                try:
                    await self.update_heartbeat()
                    await asyncio.sleep(interval_seconds)
                except asyncio.CancelledError:
                    logger.info(f"🫀 Heartbeat cancelled for job {self.job_id}")
                    break
                except Exception as e:
                    logger.error(f"❌ Heartbeat error for job {self.job_id}: {e}")
                    await asyncio.sleep(interval_seconds)  # Continue despite errors

        # Start heartbeat task
        import asyncio
        self._heartbeat_task = asyncio.create_task(heartbeat_loop())

    async def check_if_cancelled(self) -> bool:
        """
        Check if the job has been cancelled by checking the database.

        Returns:
            True if job is cancelled, False otherwise
        """
        try:
            if self._db_sync_enabled and self._supabase:
                response = self._supabase.client.table('background_jobs')\
                    .select('status')\
                    .eq('id', self.job_id)\
                    .execute()

                if response.data and len(response.data) > 0:
                    status = response.data[0].get('status')
                    if status == 'cancelled':
                        logger.warning(f"🛑 Job {self.job_id} has been cancelled")
                        return True

            return False

        except Exception as e:
            logger.error(f"❌ Failed to check cancellation status for job {self.job_id}: {e}")
            return False

    async def update_heartbeat(self):
        """
        Send a heartbeat signal to the database to prove job is alive.

        Updates the 'last_heartbeat' timestamp in background_jobs table.
        Job monitor uses this to detect silent crashes.

        Also checks if job has been cancelled.
        """
        try:
            self.last_heartbeat = datetime.utcnow()

            if self._db_sync_enabled and self._supabase:
                # Check for cancellation first
                response = self._supabase.client.table('background_jobs')\
                    .select('status')\
                    .eq('id', self.job_id)\
                    .execute()

                if response.data and len(response.data) > 0:
                    status = response.data[0].get('status')
                    if status == 'cancelled':
                        logger.warning(f"🛑 Job {self.job_id} has been cancelled - stopping heartbeat")
                        await self.stop_heartbeat()
                        raise asyncio.CancelledError(f"Job {self.job_id} was cancelled")

                # Update heartbeat
                self._supabase.client.table('background_jobs')\
                    .update({
                        'last_heartbeat': self.last_heartbeat.isoformat(),
                        'updated_at': self.last_heartbeat.isoformat()
                    })\
                    .eq('id', self.job_id)\
                    .execute()

                logger.debug(f"🫀 Heartbeat sent for job {self.job_id}")

        except asyncio.CancelledError:
            raise  # Re-raise cancellation
        except Exception as e:
            logger.error(f"❌ Failed to update heartbeat for job {self.job_id}: {e}")

    async def stop_heartbeat(self):
        """Stop the heartbeat monitoring task"""
        if not self._heartbeat_running:
            return

        self._heartbeat_running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except:
                pass  # Ignore cancellation errors
            self._heartbeat_task = None

        logger.info(f"🫀 Heartbeat stopped for job {self.job_id}")


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

