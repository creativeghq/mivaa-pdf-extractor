"""
Endpoint Registry - Singleton manager for HuggingFace endpoint clients.

This module provides a centralized registry for endpoint managers and clients
to ensure they are only initialized and warmed up once per job/process.

This prevents:
- Repeated endpoint warmups during product processing
- Multiple SLIGClient instances being created
- Unnecessary YOLO endpoint manager re-initializations

Usage:
    from app.services.embeddings.endpoint_registry import endpoint_registry

    # Get or create SLIG client (singleton)
    slig_client = endpoint_registry.get_slig_client()

    # Get or create YOLO manager (singleton)
    yolo_manager = endpoint_registry.get_yolo_manager()

    # Start processing (prevents auto-pause)
    endpoint_registry.start_processing(job_id)

    # End processing (allows auto-pause)
    endpoint_registry.end_processing(job_id)

    # Clear all endpoints (call at job completion)
    endpoint_registry.clear_all()
"""

import logging
import time
from typing import Optional, Dict, Any, List
from threading import Lock

logger = logging.getLogger(__name__)


class EndpointRegistry:
    """
    Singleton registry for HuggingFace endpoint managers and clients.

    Thread-safe implementation ensuring endpoints are only initialized once
    and shared across all processing stages within a job.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._slig_client = None
        self._slig_manager = None
        self._yolo_manager = None
        self._qwen_manager = None
        self._chandra_manager = None

        # Track warmup status
        self._slig_warmed_up = False
        self._yolo_warmed_up = False
        self._qwen_warmed_up = False
        self._chandra_warmed_up = False

        # Track health check status
        self._health_validated = False
        self._health_results: Dict[str, Any] = {}

        # Processing lock - prevents auto-pause during active jobs
        self._active_jobs: List[str] = []
        self._processing_start_time: Optional[float] = None

        # Lock for thread-safe client creation
        self._client_lock = Lock()

        self._initialized = True
        logger.info("✅ EndpointRegistry initialized (singleton)")

    def get_slig_client(self, force_new: bool = False) -> Optional[Any]:
        """
        Get or create the singleton SLIG client.

        Args:
            force_new: If True, create a new client even if one exists.

        Returns:
            SLIGClient instance or None if configuration is missing.
        """
        with self._client_lock:
            if self._slig_client is not None and not force_new:
                logger.debug("♻️ Reusing existing SLIG client (singleton)")
                return self._slig_client

            try:
                from app.config import get_settings
                from app.services.embeddings.slig_client import SLIGClient

                settings = get_settings()

                if not settings.slig_endpoint_url or not settings.slig_endpoint_token:
                    logger.warning("⚠️ SLIG endpoint not configured")
                    return None

                # ✅ FIX: Pass the pre-warmed manager from registry to the client
                # This allows the client to check endpoint status before inference
                # and prevents 400 errors when endpoint is not ready
                self._slig_client = SLIGClient(
                    endpoint_url=settings.slig_endpoint_url,
                    token=settings.slig_endpoint_token,
                    endpoint_name="mh-siglip2",
                    namespace="basiliskan",
                    auto_pause=False,  # Disable auto-pause to prevent re-warmups
                    endpoint_manager=self._slig_manager  # ✅ Pass pre-warmed manager!
                )

                if self._slig_manager:
                    logger.info("✅ SLIG client created with pre-warmed manager (singleton)")
                else:
                    logger.warning("⚠️ SLIG client created without manager - endpoint status checks disabled")
                return self._slig_client

            except Exception as e:
                logger.error(f"❌ Failed to create SLIG client: {e}")
                return None

    def get_slig_manager(self, force_new: bool = False) -> Optional[Any]:
        """
        Get or create the singleton SLIG endpoint manager.

        Returns:
            SLIGEndpointManager instance or None if configuration is missing.
        """
        with self._client_lock:
            if self._slig_manager is not None and not force_new:
                logger.debug("♻️ Reusing existing SLIG manager (singleton)")
                return self._slig_manager

            try:
                from app.config import get_settings
                from app.services.embeddings.slig_endpoint_manager import SLIGEndpointManager

                settings = get_settings()

                if not settings.slig_endpoint_url or not settings.slig_endpoint_token:
                    logger.warning("⚠️ SLIG endpoint not configured")
                    return None

                self._slig_manager = SLIGEndpointManager(
                    endpoint_url=settings.slig_endpoint_url,
                    hf_token=settings.slig_endpoint_token,
                    endpoint_name="mh-siglip2",
                    namespace="basiliskan",
                    auto_pause_timeout=60,
                    warmup_timeout=60
                )

                logger.info("✅ SLIG manager created (singleton)")
                return self._slig_manager

            except Exception as e:
                logger.error(f"❌ Failed to create SLIG manager: {e}")
                return None

    def get_yolo_manager(self, force_new: bool = False) -> Optional[Any]:
        """
        Get or create the singleton YOLO endpoint manager.

        Returns:
            YoloEndpointManager instance or None if configuration is missing.
        """
        with self._client_lock:
            if self._yolo_manager is not None and not force_new:
                logger.debug("♻️ Reusing existing YOLO manager (singleton)")
                return self._yolo_manager

            try:
                from app.config import get_settings
                from app.services.pdf.yolo_endpoint_manager import YoloEndpointManager

                settings = get_settings()
                yolo_config = settings.get_yolo_config()

                if not yolo_config.get("enabled", False):
                    logger.warning("⚠️ YOLO endpoint not enabled")
                    return None

                self._yolo_manager = YoloEndpointManager(
                    endpoint_url=yolo_config["endpoint_url"],
                    hf_token=yolo_config.get("hf_token", ""),
                    endpoint_name=yolo_config.get("endpoint_name"),
                    namespace=yolo_config.get("namespace"),
                    enabled=True
                )

                logger.info("✅ YOLO manager created (singleton)")
                return self._yolo_manager

            except Exception as e:
                logger.error(f"❌ Failed to create YOLO manager: {e}")
                return None

    def set_slig_client(self, client: Any):
        """Set a pre-created SLIG client."""
        with self._client_lock:
            self._slig_client = client
            logger.info("📌 Registered pre-created SLIG client")

    def set_slig_warmed_up(self, warmed_up: bool = True):
        """Mark SLIG endpoint as warmed up."""
        self._slig_warmed_up = warmed_up
        logger.info(f"📌 SLIG warmup status: {warmed_up}")

    def set_yolo_warmed_up(self, warmed_up: bool = True):
        """Mark YOLO endpoint as warmed up."""
        self._yolo_warmed_up = warmed_up
        logger.info(f"📌 YOLO warmup status: {warmed_up}")

    def is_slig_warmed_up(self) -> bool:
        """Check if SLIG endpoint is warmed up."""
        return self._slig_warmed_up

    def is_yolo_warmed_up(self) -> bool:
        """Check if YOLO endpoint is warmed up."""
        return self._yolo_warmed_up

    def register_endpoint_managers(self, endpoint_managers: Dict[str, Any]):
        """
        Register pre-created endpoint managers from warmup phase.

        This is called from rag_routes.py after initial warmup to share
        the warmed-up managers with the processing pipeline.

        Args:
            endpoint_managers: Dict with 'slig', 'yolo', 'qwen', 'chandra' keys
        """
        with self._client_lock:
            registered_count = 0

            if 'slig' in endpoint_managers:
                self._slig_manager = endpoint_managers['slig']
                self._slig_warmed_up = True
                registered_count += 1
                logger.info("📌 Registered pre-warmed SLIG manager")

            if 'yolo' in endpoint_managers:
                self._yolo_manager = endpoint_managers['yolo']
                self._yolo_warmed_up = True
                registered_count += 1
                logger.info("📌 Registered pre-warmed YOLO manager")

            if 'qwen' in endpoint_managers:
                self._qwen_manager = endpoint_managers['qwen']
                self._qwen_warmed_up = True
                registered_count += 1
                logger.info("📌 Registered pre-warmed Qwen manager")

            if 'chandra' in endpoint_managers:
                self._chandra_manager = endpoint_managers['chandra']
                self._chandra_warmed_up = True
                registered_count += 1
                logger.info("📌 Registered pre-warmed Chandra manager")

            logger.info(f"📊 Total endpoint managers registered: {registered_count}")

    def clear_all(self):
        """
        Clear all endpoint clients and managers.

        Call this at job completion to release resources.
        """
        with self._client_lock:
            self._slig_client = None
            self._slig_manager = None
            self._yolo_manager = None
            self._qwen_manager = None
            self._chandra_manager = None
            self._slig_warmed_up = False
            self._yolo_warmed_up = False
            self._qwen_warmed_up = False
            self._chandra_warmed_up = False
            self._health_validated = False
            self._health_results = {}
            self._active_jobs = []
            self._processing_start_time = None
            logger.info("🗑️ Cleared all endpoint managers from registry")

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "slig_client_active": self._slig_client is not None,
            "slig_manager_active": self._slig_manager is not None,
            "slig_warmed_up": self._slig_warmed_up,
            "yolo_manager_active": self._yolo_manager is not None,
            "yolo_warmed_up": self._yolo_warmed_up,
            "qwen_manager_active": self._qwen_manager is not None,
            "qwen_warmed_up": self._qwen_warmed_up,
            "chandra_manager_active": self._chandra_manager is not None,
            "chandra_warmed_up": self._chandra_warmed_up,
            "health_validated": self._health_validated,
            "active_jobs": len(self._active_jobs),
            "is_processing": self.is_processing()
        }

    # =========================================================================
    # PROCESSING LOCK - Prevents auto-pause during active jobs
    # =========================================================================

    def start_processing(self, job_id: str):
        """
        Mark the start of processing for a job.
        This prevents auto-pause while jobs are actively using endpoints.

        Args:
            job_id: The job ID starting processing
        """
        with self._client_lock:
            if job_id not in self._active_jobs:
                self._active_jobs.append(job_id)
                if len(self._active_jobs) == 1:
                    self._processing_start_time = time.time()
                logger.info(f"🔒 Processing started for job {job_id} (active jobs: {len(self._active_jobs)})")

    def end_processing(self, job_id: str):
        """
        Mark the end of processing for a job.
        Auto-pause can resume once all jobs complete.

        Args:
            job_id: The job ID ending processing
        """
        with self._client_lock:
            if job_id in self._active_jobs:
                self._active_jobs.remove(job_id)
                logger.info(f"🔓 Processing ended for job {job_id} (active jobs: {len(self._active_jobs)})")

                if len(self._active_jobs) == 0:
                    if self._processing_start_time:
                        duration = time.time() - self._processing_start_time
                        logger.info(f"⏱️ Total processing time: {duration:.1f}s")
                    self._processing_start_time = None

    def is_processing(self) -> bool:
        """Check if any jobs are actively processing."""
        return len(self._active_jobs) > 0

    def can_auto_pause(self) -> bool:
        """Check if auto-pause is allowed (no active processing)."""
        return not self.is_processing()

    def get_active_jobs(self) -> List[str]:
        """Get list of currently active job IDs."""
        return self._active_jobs.copy()

    # =========================================================================
    # HEALTH VALIDATION - Ensures endpoints are truly ready
    # =========================================================================

    def set_health_validated(self, validated: bool = True, results: Optional[Dict[str, Any]] = None):
        """
        Mark health validation as complete.

        Args:
            validated: Whether health checks passed
            results: Health check results dictionary
        """
        self._health_validated = validated
        if results:
            self._health_results = results
        logger.info(f"🏥 Health validation status: {'PASSED' if validated else 'FAILED'}")

    def is_health_validated(self) -> bool:
        """Check if health validation has been performed and passed."""
        return self._health_validated

    def get_health_results(self) -> Dict[str, Any]:
        """Get health check results."""
        return self._health_results.copy()

    def is_ready_for_processing(self) -> bool:
        """
        Check if all conditions are met to start processing:
        1. Health validated
        2. Required managers registered
        """
        if not self._health_validated:
            logger.warning("⚠️ Not ready: Health validation not completed")
            return False

        # Check at least one manager is available
        has_manager = (
            self._slig_manager is not None or
            self._qwen_manager is not None or
            self._yolo_manager is not None
        )

        if not has_manager:
            logger.warning("⚠️ Not ready: No endpoint managers registered")
            return False

        return True

    def get_qwen_manager(self) -> Optional[Any]:
        """Get the registered Qwen manager."""
        return self._qwen_manager

    def get_chandra_manager(self) -> Optional[Any]:
        """Get the registered Chandra manager."""
        return self._chandra_manager


# Global singleton instance
endpoint_registry = EndpointRegistry()
