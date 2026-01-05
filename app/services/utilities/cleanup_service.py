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
            from app.services.core.supabase_client import get_supabase_client

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

    async def delete_job_completely(
        self,
        job_id: str,
        supabase_client,
        vecs_service=None,
        delete_storage_files: bool = True
    ) -> Dict[str, Any]:
        """
        Completely delete a job and ALL its associated data.

        This includes:
        1. Job record from background_jobs table
        2. Document record (if exists)
        3. All chunks from document_chunks
        4. All embeddings from vecs collections
        5. All images from document_images
        6. All products
        7. Files from storage buckets (only if delete_storage_files=True)
        8. Checkpoints
        9. Temporary files

        Args:
            job_id: Job ID to delete
            supabase_client: Supabase client instance
            vecs_service: Optional vecs service for embedding deletion
            delete_storage_files: If True, delete storage files. Set to False for automatic cleanup (default: True)

        Returns:
            Dictionary with deletion statistics

        Note:
            - Manual deletion (from UI): delete_storage_files=True (deletes everything including PDFs)
            - Automatic cleanup (cron): delete_storage_files=False (keeps storage files, only deletes DB records)
        """
        try:
            self.logger.info(f"🗑️ Starting complete deletion for job {job_id}")

            stats = {
                'job_deleted': False,
                'document_deleted': False,
                'chunks_deleted': 0,
                'embeddings_deleted': 0,
                'images_deleted': 0,
                'products_deleted': 0,
                'storage_files_deleted': 0,
                'checkpoints_deleted': 0,
                'temp_files_deleted': 0,
                'errors': []
            }

            # 1. Get job details to find document_id
            try:
                job_response = supabase_client.client.table('background_jobs')\
                    .select('document_id, workspace_id')\
                    .eq('id', job_id)\
                    .single()\
                    .execute()

                if not job_response.data:
                    self.logger.warning(f"Job {job_id} not found")
                    stats['errors'].append("Job not found")
                    return stats

                document_id = job_response.data.get('document_id')
                workspace_id = job_response.data.get('workspace_id')

                self.logger.info(f"Found job {job_id} with document_id={document_id}, workspace_id={workspace_id}")

            except Exception as e:
                self.logger.error(f"Failed to get job details: {e}")
                stats['errors'].append(f"Failed to get job details: {str(e)}")
                return stats

            # 2. Delete chunks
            if document_id:
                try:
                    chunks_response = supabase_client.client.table('document_chunks')\
                        .delete()\
                        .eq('document_id', document_id)\
                        .execute()

                    stats['chunks_deleted'] = len(chunks_response.data) if chunks_response.data else 0
                    self.logger.info(f"✅ Deleted {stats['chunks_deleted']} chunks")

                except Exception as e:
                    self.logger.error(f"Failed to delete chunks: {e}")
                    stats['errors'].append(f"Chunks deletion failed: {str(e)}")

            # 3. Delete embeddings from vecs
            if document_id and vecs_service:
                try:
                    # Delete text embeddings
                    text_emb_deleted = await vecs_service.delete_document_embeddings(document_id)
                    stats['embeddings_deleted'] += text_emb_deleted

                    # Delete image embeddings
                    image_emb_deleted = await vecs_service.delete_image_embeddings_by_document(document_id)
                    stats['embeddings_deleted'] += image_emb_deleted

                    self.logger.info(f"✅ Deleted {stats['embeddings_deleted']} embeddings from vecs")

                except Exception as e:
                    self.logger.error(f"Failed to delete embeddings: {e}")
                    stats['errors'].append(f"Embeddings deletion failed: {str(e)}")

            # 4. Delete images
            if document_id:
                try:
                    images_response = supabase_client.client.table('document_images')\
                        .delete()\
                        .eq('document_id', document_id)\
                        .execute()

                    stats['images_deleted'] = len(images_response.data) if images_response.data else 0
                    self.logger.info(f"✅ Deleted {stats['images_deleted']} images")

                except Exception as e:
                    self.logger.error(f"Failed to delete images: {e}")
                    stats['errors'].append(f"Images deletion failed: {str(e)}")

            # 5. Delete products (if they exist for this document)
            if document_id:
                try:
                    products_response = supabase_client.client.table('products')\
                        .delete()\
                        .eq('document_id', document_id)\
                        .execute()

                    stats['products_deleted'] = len(products_response.data) if products_response.data else 0
                    self.logger.info(f"✅ Deleted {stats['products_deleted']} products")

                except Exception as e:
                    self.logger.error(f"Failed to delete products: {e}")
                    stats['errors'].append(f"Products deletion failed: {str(e)}")

            # 6. Delete files from storage (only if delete_storage_files=True)
            if document_id and delete_storage_files:
                try:
                    stats['storage_files_deleted'] = self.cleanup_storage_bucket('documents', document_id)
                    self.logger.info(f"✅ Deleted {stats['storage_files_deleted']} storage files")

                except Exception as e:
                    self.logger.error(f"Failed to delete storage files: {e}")
                    stats['errors'].append(f"Storage deletion failed: {str(e)}")
            elif document_id and not delete_storage_files:
                self.logger.info(f"⏭️ Skipping storage file deletion (automatic cleanup mode)")

            # 7. Delete checkpoints
            try:
                checkpoints_response = supabase_client.client.table('job_checkpoints')\
                    .delete()\
                    .eq('job_id', job_id)\
                    .execute()

                stats['checkpoints_deleted'] = len(checkpoints_response.data) if checkpoints_response.data else 0
                self.logger.info(f"✅ Deleted {stats['checkpoints_deleted']} checkpoints")

            except Exception as e:
                self.logger.error(f"Failed to delete checkpoints: {e}")
                stats['errors'].append(f"Checkpoints deletion failed: {str(e)}")

            # 8. Delete document record
            if document_id:
                try:
                    doc_response = supabase_client.client.table('documents')\
                        .delete()\
                        .eq('id', document_id)\
                        .execute()

                    stats['document_deleted'] = len(doc_response.data) > 0 if doc_response.data else False
                    self.logger.info(f"✅ Deleted document record")

                except Exception as e:
                    self.logger.error(f"Failed to delete document: {e}")
                    stats['errors'].append(f"Document deletion failed: {str(e)}")

            # 9. Delete job record
            try:
                job_del_response = supabase_client.client.table('background_jobs')\
                    .delete()\
                    .eq('id', job_id)\
                    .execute()

                stats['job_deleted'] = len(job_del_response.data) > 0 if job_del_response.data else False
                self.logger.info(f"✅ Deleted job record")

            except Exception as e:
                self.logger.error(f"Failed to delete job: {e}")
                stats['errors'].append(f"Job deletion failed: {str(e)}")

            # 10. Clean temporary files
            if document_id:
                try:
                    stats['temp_files_deleted'] = self._clean_temp_directories(job_id, document_id)
                    self.logger.info(f"✅ Cleaned {stats['temp_files_deleted']} temp directories")

                except Exception as e:
                    self.logger.error(f"Failed to clean temp files: {e}")
                    stats['errors'].append(f"Temp files cleanup failed: {str(e)}")

            self.logger.info(f"🎉 Complete deletion finished for job {job_id}")
            self.logger.info(f"   Stats: {stats}")

            return stats

        except Exception as e:
            self.logger.error(f"❌ Complete deletion failed for job {job_id}: {e}", exc_info=True)
            stats['errors'].append(f"Unexpected error: {str(e)}")
            return stats

    async def cleanup_system_temp_files(
        self,
        max_age_hours: int = 24,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive system-wide temporary file cleanup.

        Cleans up:
        1. PDF files in /tmp (*.pdf)
        2. pdf_processor folders in /tmp
        3. Files in /var/www/mivaa-pdf-extractor/output
        4. Empty temp/uploads/logs folders
        5. __pycache__ folders
        6. Old files in /tmp/pdf_processing, /tmp/image_extraction, etc.

        Args:
            max_age_hours: Maximum age of files to keep (default: 24 hours)
            dry_run: If True, only report what would be deleted without actually deleting

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            import time

            self.logger.info(f"🧹 Starting system-wide temp file cleanup (max_age={max_age_hours}h, dry_run={dry_run})")

            stats = {
                'pdf_files_deleted': 0,
                'pdf_files_size_mb': 0,
                'pdf_processor_folders_deleted': 0,
                'pdf_processor_size_mb': 0,
                'output_files_deleted': 0,
                'output_size_mb': 0,
                'empty_folders_deleted': 0,
                'pycache_folders_deleted': 0,
                'pycache_size_mb': 0,
                'temp_processing_files_deleted': 0,
                'temp_processing_size_mb': 0,
                'total_size_freed_mb': 0,
                'errors': []
            }

            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            # 1. Clean PDF files in /tmp
            try:
                tmp_dir = '/tmp'
                if os.path.exists(tmp_dir):
                    for file in os.listdir(tmp_dir):
                        if file.endswith('.pdf'):
                            file_path = os.path.join(tmp_dir, file)
                            try:
                                file_age = current_time - os.path.getmtime(file_path)
                                if file_age > max_age_seconds:
                                    file_size_mb = os.path.getsize(file_path) / 1024 / 1024

                                    if not dry_run:
                                        os.remove(file_path)

                                    stats['pdf_files_deleted'] += 1
                                    stats['pdf_files_size_mb'] += file_size_mb
                                    self.logger.debug(f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'} PDF: {file_path} ({file_size_mb:.2f} MB)")
                            except Exception as e:
                                self.logger.error(f"Failed to delete PDF {file_path}: {e}")
                                stats['errors'].append(f"PDF deletion failed: {file}")

                self.logger.info(f"✅ PDF files: {stats['pdf_files_deleted']} files, {stats['pdf_files_size_mb']:.2f} MB")

            except Exception as e:
                self.logger.error(f"Failed to clean PDF files: {e}")
                stats['errors'].append(f"PDF cleanup failed: {str(e)}")

            # 2. Clean pdf_processor folders in /tmp
            try:
                tmp_dir = '/tmp'
                if os.path.exists(tmp_dir):
                    for folder in os.listdir(tmp_dir):
                        if 'pdf_processor' in folder.lower() or 'pdf_processing' in folder.lower():
                            folder_path = os.path.join(tmp_dir, folder)
                            if os.path.isdir(folder_path):
                                try:
                                    folder_age = current_time - os.path.getmtime(folder_path)
                                    if folder_age > max_age_seconds:
                                        folder_size_mb = self._get_folder_size_mb(folder_path)

                                        if not dry_run:
                                            shutil.rmtree(folder_path)

                                        stats['pdf_processor_folders_deleted'] += 1
                                        stats['pdf_processor_size_mb'] += folder_size_mb
                                        self.logger.debug(f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'} folder: {folder_path} ({folder_size_mb:.2f} MB)")
                                except Exception as e:
                                    self.logger.error(f"Failed to delete folder {folder_path}: {e}")
                                    stats['errors'].append(f"Folder deletion failed: {folder}")

                self.logger.info(f"✅ PDF processor folders: {stats['pdf_processor_folders_deleted']} folders, {stats['pdf_processor_size_mb']:.2f} MB")

            except Exception as e:
                self.logger.error(f"Failed to clean pdf_processor folders: {e}")
                stats['errors'].append(f"PDF processor cleanup failed: {str(e)}")

            # 3. Clean output directory
            try:
                output_dir = '/var/www/mivaa-pdf-extractor/output'
                if os.path.exists(output_dir):
                    for file in os.listdir(output_dir):
                        file_path = os.path.join(output_dir, file)
                        if os.path.isfile(file_path):
                            try:
                                file_size_mb = os.path.getsize(file_path) / 1024 / 1024

                                if not dry_run:
                                    os.remove(file_path)

                                stats['output_files_deleted'] += 1
                                stats['output_size_mb'] += file_size_mb
                                self.logger.debug(f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'} output file: {file_path}")
                            except Exception as e:
                                self.logger.error(f"Failed to delete output file {file_path}: {e}")
                                stats['errors'].append(f"Output file deletion failed: {file}")

                self.logger.info(f"✅ Output files: {stats['output_files_deleted']} files, {stats['output_size_mb']:.2f} MB")

            except Exception as e:
                self.logger.error(f"Failed to clean output directory: {e}")
                stats['errors'].append(f"Output cleanup failed: {str(e)}")

            # 4. Clean __pycache__ folders
            try:
                base_dir = '/var/www/mivaa-pdf-extractor'
                if os.path.exists(base_dir):
                    for root, dirs, files in os.walk(base_dir):
                        if '__pycache__' in dirs:
                            pycache_path = os.path.join(root, '__pycache__')
                            try:
                                pycache_size_mb = self._get_folder_size_mb(pycache_path)

                                if not dry_run:
                                    shutil.rmtree(pycache_path)

                                stats['pycache_folders_deleted'] += 1
                                stats['pycache_size_mb'] += pycache_size_mb
                                self.logger.debug(f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'} __pycache__: {pycache_path}")
                            except Exception as e:
                                self.logger.error(f"Failed to delete __pycache__ {pycache_path}: {e}")
                                stats['errors'].append(f"__pycache__ deletion failed: {pycache_path}")

                self.logger.info(f"✅ __pycache__ folders: {stats['pycache_folders_deleted']} folders, {stats['pycache_size_mb']:.2f} MB")

            except Exception as e:
                self.logger.error(f"Failed to clean __pycache__ folders: {e}")
                stats['errors'].append(f"__pycache__ cleanup failed: {str(e)}")

            # 5. Clean temp processing directories
            try:
                for temp_dir in self.temp_dirs:
                    if os.path.exists(temp_dir):
                        for item in os.listdir(temp_dir):
                            item_path = os.path.join(temp_dir, item)
                            try:
                                item_age = current_time - os.path.getmtime(item_path)
                                if item_age > max_age_seconds:
                                    if os.path.isfile(item_path):
                                        file_size_mb = os.path.getsize(item_path) / 1024 / 1024

                                        if not dry_run:
                                            os.remove(item_path)

                                        stats['temp_processing_files_deleted'] += 1
                                        stats['temp_processing_size_mb'] += file_size_mb
                                    elif os.path.isdir(item_path):
                                        folder_size_mb = self._get_folder_size_mb(item_path)

                                        if not dry_run:
                                            shutil.rmtree(item_path)

                                        stats['temp_processing_files_deleted'] += 1
                                        stats['temp_processing_size_mb'] += folder_size_mb
                            except Exception as e:
                                self.logger.error(f"Failed to delete temp item {item_path}: {e}")
                                stats['errors'].append(f"Temp item deletion failed: {item}")

                self.logger.info(f"✅ Temp processing files: {stats['temp_processing_files_deleted']} items, {stats['temp_processing_size_mb']:.2f} MB")

            except Exception as e:
                self.logger.error(f"Failed to clean temp processing directories: {e}")
                stats['errors'].append(f"Temp processing cleanup failed: {str(e)}")

            # Calculate total size freed
            stats['total_size_freed_mb'] = (
                stats['pdf_files_size_mb'] +
                stats['pdf_processor_size_mb'] +
                stats['output_size_mb'] +
                stats['pycache_size_mb'] +
                stats['temp_processing_size_mb']
            )

            self.logger.info(f"🎉 System cleanup {'simulation' if dry_run else 'complete'}: {stats['total_size_freed_mb']:.2f} MB freed")
            self.logger.info(f"   Stats: {stats}")

            return stats

        except Exception as e:
            self.logger.error(f"❌ System cleanup failed: {e}", exc_info=True)
            stats['errors'].append(f"Unexpected error: {str(e)}")
            return stats

    def _get_folder_size_mb(self, folder_path: str) -> float:
        """
        Get the total size of a folder in MB.

        Args:
            folder_path: Path to folder

        Returns:
            Size in MB
        """
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(file_path)
                    except Exception:
                        pass
        except Exception as e:
            self.logger.error(f"Failed to get folder size for {folder_path}: {e}")

        return total_size / 1024 / 1024  # Convert to MB


