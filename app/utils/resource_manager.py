"""
Resource Manager - Event-Based Cleanup System

Replaces time-based cleanup with event-driven lifecycle management.
Ensures resources are only cleaned up when processing is COMPLETE, not based on timeouts.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ResourceState(Enum):
    """Resource lifecycle states"""
    CREATED = "created"
    IN_USE = "in_use"
    READY_FOR_CLEANUP = "ready_for_cleanup"
    CLEANED_UP = "cleaned_up"


@dataclass
class Resource:
    """Represents a managed resource (file, directory, PyMuPDF object, etc.)"""
    id: str
    type: str  # 'file', 'directory', 'pymupdf_doc', 'temp_file'
    path: Optional[str] = None
    state: ResourceState = ResourceState.CREATED
    created_at: datetime = field(default_factory=datetime.utcnow)
    in_use_by: Set[str] = field(default_factory=set)  # Set of job_ids using this resource
    metadata: Dict = field(default_factory=dict)


class ResourceManager:
    """
    Event-based resource lifecycle manager.
    
    Resources are tracked and only cleaned up when:
    1. All jobs using them have completed/failed
    2. Explicit cleanup is requested
    3. Service shutdown (graceful cleanup)
    
    NOT cleaned up based on time/timeouts.
    """
    
    def __init__(self):
        self.resources: Dict[str, Resource] = {}
        self._lock = asyncio.Lock()
        logger.info("🔧 ResourceManager initialized (event-based cleanup)")
    
    async def register_resource(
        self,
        resource_id: str,
        resource_type: str,
        path: Optional[str] = None,
        job_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Resource:
        """
        Register a new resource for lifecycle management.
        
        Args:
            resource_id: Unique identifier for the resource
            resource_type: Type of resource ('file', 'directory', 'pymupdf_doc', etc.)
            path: File system path (if applicable)
            job_id: Job ID that created/uses this resource
            metadata: Additional metadata
        
        Returns:
            Resource object
        """
        async with self._lock:
            if resource_id in self.resources:
                # Resource already exists, add job_id to in_use_by
                resource = self.resources[resource_id]
                if job_id:
                    resource.in_use_by.add(job_id)
                logger.debug(f"📌 Resource {resource_id} already registered, added job {job_id}")
                return resource
            
            resource = Resource(
                id=resource_id,
                type=resource_type,
                path=path,
                state=ResourceState.CREATED,
                metadata=metadata or {}
            )
            
            if job_id:
                resource.in_use_by.add(job_id)
            
            self.resources[resource_id] = resource
            logger.info(f"✅ Registered resource: {resource_id} (type={resource_type}, job={job_id})")
            
            return resource
    
    async def mark_in_use(self, resource_id: str, job_id: str):
        """Mark resource as actively in use by a job"""
        async with self._lock:
            if resource_id not in self.resources:
                logger.warning(f"⚠️ Cannot mark unknown resource {resource_id} as in use")
                return
            
            resource = self.resources[resource_id]
            resource.state = ResourceState.IN_USE
            resource.in_use_by.add(job_id)
            logger.debug(f"🔒 Resource {resource_id} marked IN_USE by job {job_id}")
    
    async def release_resource(self, resource_id: str, job_id: str):
        """
        Release a resource from a job.
        If no jobs are using it, mark ready for cleanup.
        """
        async with self._lock:
            if resource_id not in self.resources:
                logger.debug(f"⚠️ Resource {resource_id} not found (already cleaned up?)")
                return
            
            resource = self.resources[resource_id]
            resource.in_use_by.discard(job_id)
            
            logger.debug(f"🔓 Resource {resource_id} released by job {job_id} ({len(resource.in_use_by)} jobs still using)")
            
            # If no jobs are using it, mark ready for cleanup
            if len(resource.in_use_by) == 0:
                resource.state = ResourceState.READY_FOR_CLEANUP
                logger.info(f"✅ Resource {resource_id} ready for cleanup (no jobs using it)")
    
    async def cleanup_ready_resources(self) -> int:
        """
        Clean up all resources that are ready for cleanup.
        Returns number of resources cleaned up.
        """
        cleaned_count = 0

        async with self._lock:
            resources_to_cleanup = [
                r for r in self.resources.values()
                if r.state == ResourceState.READY_FOR_CLEANUP
            ]

        for resource in resources_to_cleanup:
            try:
                await self._cleanup_resource(resource)
                cleaned_count += 1
            except Exception as e:
                logger.error(f"❌ Failed to cleanup resource {resource.id}: {e}")

        return cleaned_count

    async def shutdown_cleanup_all(self) -> int:
        """Force-cleanup every tracked resource on graceful shutdown.

        Used by the FastAPI shutdown hook so a process restart mid-job does not
        leave temp PDFs in /tmp. Unlike cleanup_ready_resources, this does not
        check in_use_by — at shutdown there is no future job that will release
        the handle.
        """
        async with self._lock:
            resources = list(self.resources.values())
        cleaned = 0
        for resource in resources:
            try:
                await self._cleanup_resource(resource)
                cleaned += 1
            except Exception as e:
                logger.error(f"❌ Shutdown cleanup failed for {resource.id}: {e}")
        logger.info(f"🛑 Shutdown cleanup complete: {cleaned} resource(s) released")
        return cleaned
    
    async def _cleanup_resource(self, resource: Resource):
        """Actually perform the cleanup of a resource"""
        try:
            if resource.type == 'file' and resource.path:
                if os.path.exists(resource.path):
                    os.remove(resource.path)
                    logger.debug(f"🗑️  Deleted file: {resource.path}")
            
            elif resource.type == 'directory' and resource.path:
                if os.path.exists(resource.path):
                    import shutil
                    shutil.rmtree(resource.path)
                    logger.debug(f"🗑️  Deleted directory: {resource.path}")
            
            elif resource.type == 'pymupdf_doc':
                # PyMuPDF documents are handled by Python GC
                # Just mark as cleaned up
                pass
            
            # Mark as cleaned up
            async with self._lock:
                if resource.id in self.resources:
                    resource.state = ResourceState.CLEANED_UP
                    # Remove from tracking after cleanup
                    del self.resources[resource.id]
            
            logger.info(f"✅ Cleaned up resource: {resource.id}")
        
        except Exception as e:
            logger.error(f"❌ Cleanup failed for {resource.id}: {e}")
            raise


# Global instance
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get the global ResourceManager instance"""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


def sweep_orphan_temp_pdfs(
    tmp_dir: str = "/tmp",
    max_age_hours: int = 12,
) -> Dict[str, int]:
    """Disk-level janitor for orphan PDF temp files left by crashes.

    The ResourceManager only knows about resources from the currently-running
    process. On SIGKILL / OOM kill, the in-memory state is lost and any temp
    PDF that was registered orphans on the host's /tmp filesystem with no
    cleanup. Across many incidents this fills the partition.

    This sweep:
      1. Scans `tmp_dir` for files matching the PDF temp patterns we know
         the orchestrator creates (`tmp*.pdf` from NamedTemporaryFile +
         the per-document subdirs `pdf_processor_*`).
      2. Filters to files / dirs older than `max_age_hours` (default 12h —
         well past the per-product timeout, so we never touch live work).
      3. Deletes them. Honours errors silently (don't crash startup over a
         file we can't unlink).

    Called from `lifespan()` at startup so each fresh process inherits a
    clean /tmp. Safe to call multiple times.

    Returns counts: `{scanned, deleted, errors, skipped_recent}`.
    """
    import os as _os
    import shutil as _shutil
    import time as _time

    stats = {"scanned": 0, "deleted": 0, "errors": 0, "skipped_recent": 0}
    if not _os.path.isdir(tmp_dir):
        return stats
    cutoff = _time.time() - (max_age_hours * 3600)

    try:
        for name in _os.listdir(tmp_dir):
            full = _os.path.join(tmp_dir, name)
            # Match the patterns the orchestrator + NamedTemporaryFile produce.
            is_pdf_temp_file = name.startswith("tmp") and name.endswith(".pdf")
            is_pdf_temp_dir = name.startswith("pdf_processor_")
            if not (is_pdf_temp_file or is_pdf_temp_dir):
                continue
            stats["scanned"] += 1
            try:
                mtime = _os.path.getmtime(full)
            except OSError:
                stats["errors"] += 1
                continue
            if mtime > cutoff:
                stats["skipped_recent"] += 1
                continue
            try:
                if _os.path.isdir(full):
                    _shutil.rmtree(full, ignore_errors=True)
                else:
                    _os.unlink(full)
                stats["deleted"] += 1
                logger.info(f"🧹 [janitor] Removed orphan PDF temp: {full}")
            except Exception as e:
                stats["errors"] += 1
                logger.debug(f"   janitor: failed to remove {full}: {e}")
    except Exception as outer_err:
        logger.warning(f"   janitor: scan of {tmp_dir} failed: {outer_err}")

    if stats["deleted"] or stats["errors"]:
        logger.info(f"🧹 [janitor] /tmp PDF sweep: {stats}")
    return stats


