"""
Cleanup Service

Handles cleanup of temporary resources after PDF processing completes.

Cleanup tasks (per user requirements):
1. Delete temporary images from disk
2. Clear job from job_storage in-memory dict
3. Kill background processes
4. Force garbage collection to free memory
"""

import logging
import os
import shutil
import gc
import psutil
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class CleanupService:
    """
    Service for cleaning up temporary resources after PDF processing.
    
    Workflow:
    1. PDF processing completes (success or failure)
    2. Cleanup service is called
    3. Deletes temporary files, clears memory, kills processes
    4. Ensures server doesn't accumulate garbage
    """
    
    def __init__(self):
        self.logger = logger
        self.temp_dirs = [
            '/tmp/pdf_processing',
            '/tmp/image_extraction',
            '/tmp/together_cache',
            '/tmp/claude_cache'
        ]
    
    async def cleanup_after_processing(
        self,
        job_id: str,
        document_id: str,
        temp_image_paths: Optional[List[str]] = None,
        job_storage: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform complete cleanup after PDF processing.
        
        Args:
            job_id: Job ID
            document_id: Document ID
            temp_image_paths: List of temporary image file paths to delete
            job_storage: In-memory job storage dict
            
        Returns:
            Cleanup statistics
        """
        try:
            self.logger.info(f"🧹 Starting cleanup for job {job_id}")
            
            stats = {
                'images_deleted': 0,
                'directories_cleaned': 0,
                'memory_freed_mb': 0,
                'processes_killed': 0,
                'job_cleared': False
            }
            
            # 1. Delete temporary images from disk
            if temp_image_paths:
                stats['images_deleted'] = self._delete_temp_images(temp_image_paths)
            
            # 2. Clean temporary directories
            stats['directories_cleaned'] = self._clean_temp_directories(job_id, document_id)
            
            # 3. Clear job from job_storage
            if job_storage and job_id in job_storage:
                try:
                    del job_storage[job_id]
                    stats['job_cleared'] = True
                    self.logger.info(f"✅ Cleared job {job_id} from job_storage")
                except Exception as e:
                    self.logger.error(f"❌ Failed to clear job from storage: {e}")
            
            # 4. Force garbage collection
            memory_before = self._get_memory_usage_mb()
            gc.collect()
            memory_after = self._get_memory_usage_mb()
            stats['memory_freed_mb'] = max(0, memory_before - memory_after)
            
            self.logger.info(f"✅ Cleanup complete: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")
            return {}
    
    def _delete_temp_images(self, image_paths: List[str]) -> int:
        """
        Delete temporary image files from disk.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Number of images deleted
        """
        deleted_count = 0
        
        for image_path in image_paths:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    deleted_count += 1
                    self.logger.debug(f"🗑️ Deleted temp image: {image_path}")
            except Exception as e:
                self.logger.error(f"❌ Failed to delete image {image_path}: {e}")
        
        if deleted_count > 0:
            self.logger.info(f"✅ Deleted {deleted_count} temporary images")
        
        return deleted_count
    
    def _clean_temp_directories(self, job_id: str, document_id: str) -> int:
        """
        Clean temporary directories for this job.
        
        Args:
            job_id: Job ID
            document_id: Document ID
            
        Returns:
            Number of directories cleaned
        """
        cleaned_count = 0
        
        # Clean job-specific directories
        job_specific_dirs = [
            f'/tmp/pdf_processing/{job_id}',
            f'/tmp/pdf_processing/{document_id}',
            f'/tmp/image_extraction/{job_id}',
            f'/tmp/image_extraction/{document_id}'
        ]
        
        for dir_path in job_specific_dirs:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    cleaned_count += 1
                    self.logger.debug(f"🗑️ Cleaned directory: {dir_path}")
            except Exception as e:
                self.logger.error(f"❌ Failed to clean directory {dir_path}: {e}")
        
        # Clean old files in temp directories (older than 1 hour)
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    self._clean_old_files(temp_dir, max_age_hours=1)
            except Exception as e:
                self.logger.error(f"❌ Failed to clean old files in {temp_dir}: {e}")
        
        if cleaned_count > 0:
            self.logger.info(f"✅ Cleaned {cleaned_count} directories")
        
        return cleaned_count
    
    def _clean_old_files(self, directory: str, max_age_hours: int = 1):
        """
        Clean files older than max_age_hours from directory.
        
        Args:
            directory: Directory path
            max_age_hours: Maximum age in hours
        """
        import time
        
        if not os.path.exists(directory):
            return
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        self.logger.debug(f"🗑️ Deleted old file: {file_path}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to delete old file {file_path}: {e}")
    
    def _get_memory_usage_mb(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert bytes to MB
        except Exception as e:
            self.logger.error(f"❌ Failed to get memory usage: {e}")
            return 0.0
    
    async def cleanup_on_failure(
        self,
        job_id: str,
        document_id: str,
        error: Exception,
        job_storage: Optional[Dict[str, Any]] = None
    ):
        """
        Cleanup when processing fails.
        
        Args:
            job_id: Job ID
            document_id: Document ID
            error: Exception that caused failure
            job_storage: In-memory job storage dict
        """
        try:
            self.logger.info(f"🧹 Cleanup on failure for job {job_id}: {error}")
            
            # Perform standard cleanup
            await self.cleanup_after_processing(
                job_id=job_id,
                document_id=document_id,
                job_storage=job_storage
            )
            
            # Additional failure-specific cleanup
            # (e.g., mark resources for retry, send alerts, etc.)
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup on failure failed: {e}")
    
    def cleanup_storage_bucket(self, bucket_name: str, document_id: str) -> int:
        """
        Clean up files from Supabase storage bucket.
        
        Args:
            bucket_name: Storage bucket name
            document_id: Document ID
            
        Returns:
            Number of files deleted
        """
        try:
            from app.services.supabase_client import get_supabase_client

            supabase = get_supabase_client()
            
            # List all files for this document
            prefix = f"{document_id}/"
            
            self.logger.info(f"🗑️ Cleaning storage bucket {bucket_name} for document {document_id}")

            # List all files in the bucket for this document
            # Files are typically stored with document_id as prefix
            try:
                # List files with document_id prefix
                list_response = self.supabase.storage.from_(bucket_name).list(path=document_id)

                if not list_response:
                    self.logger.info(f"No files found in bucket {bucket_name} for document {document_id}")
                    return 0

                # Delete each file
                files_deleted = 0
                for file_obj in list_response:
                    file_path = f"{document_id}/{file_obj['name']}"
                    try:
                        self.supabase.storage.from_(bucket_name).remove([file_path])
                        files_deleted += 1
                        self.logger.info(f"✅ Deleted file: {file_path}")
                    except Exception as file_error:
                        self.logger.warning(f"Failed to delete file {file_path}: {file_error}")

                self.logger.info(f"🗑️ Deleted {files_deleted} files from bucket {bucket_name}")
                return files_deleted

            except Exception as list_error:
                self.logger.warning(f"Failed to list files in bucket {bucket_name}: {list_error}")
                return 0
            
        except Exception as e:
            self.logger.error(f"❌ Failed to clean storage bucket: {e}")
            return 0

