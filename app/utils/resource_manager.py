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


