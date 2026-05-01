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
            '/tmp/huggingface_cache',
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

    async def rollback_discovered_products(
        self,
        document_id: str,
        product_db_ids: Optional[List[str]] = None,
        supabase_client=None
    ) -> Dict[str, Any]:
        """
        Rollback products created during discovery if subsequent stages fail.

        This ensures we don't leave orphan products in the database when
        processing fails after Stage 0 (discovery).

        Args:
            document_id: Document ID to rollback products for
            product_db_ids: Optional list of specific product IDs to delete
            supabase_client: Supabase client instance

        Returns:
            Dictionary with rollback statistics
        """
        stats = {
            'products_rolled_back': 0,
            'product_progress_cleared': 0,
            'errors': []
        }

        try:
            if not supabase_client:
                from app.services.core.supabase_client import get_supabase_client
                supabase_client = get_supabase_client()

            self.logger.info(f"🔄 Rolling back products for document {document_id}")

            # Delete products by document_id (or specific IDs if provided)
            try:
                if product_db_ids:
                    # Delete specific products
                    for product_id in product_db_ids:
                        supabase_client.client.table('products')\
                            .delete()\
                            .eq('id', product_id)\
                            .execute()
                    stats['products_rolled_back'] = len(product_db_ids)
                else:
                    # Delete all products for this document
                    products_response = supabase_client.client.table('products')\
                        .delete()\
                        .eq('source_document_id', document_id)\
                        .execute()
                    stats['products_rolled_back'] = len(products_response.data) if products_response.data else 0

                self.logger.info(f"✅ Rolled back {stats['products_rolled_back']} products")

            except Exception as e:
                self.logger.error(f"❌ Failed to rollback products: {e}")
                stats['errors'].append(f"Products rollback failed: {str(e)}")

            # Clear product_progress tracking
            try:
                progress_response = supabase_client.client.table('product_progress')\
                    .delete()\
                    .eq('document_id', document_id)\
                    .execute()
                stats['product_progress_cleared'] = len(progress_response.data) if progress_response.data else 0
                self.logger.info(f"✅ Cleared {stats['product_progress_cleared']} product progress records")

            except Exception as e:
                self.logger.warning(f"⚠️ Failed to clear product_progress: {e}")
                # Not critical - progress tracking is informational

            return stats

        except Exception as e:
            self.logger.error(f"❌ Product rollback failed: {e}")
            stats['errors'].append(str(e))
            return stats
    
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
        delete_storage_files: bool = True,
        preserve_outputs: bool = False,
    ) -> Dict[str, Any]:
        """
        Delete a job, with two distinct semantics based on `preserve_outputs`.

        Two modes:

        ──────────────────────────────────────────────────────────────────
        preserve_outputs=False (default — CANCELLATION / FAILURE wipe)
        ──────────────────────────────────────────────────────────────────
        Wipes EVERYTHING tied to the job. Use when the user cancels a job,
        a job fails irrecoverably, or admin wants to remove a stuck/bad job
        and its partial outputs from the catalog. Removes:
          - background_jobs row
          - document row
          - document_chunks
          - document_images
          - VECS embeddings (text + image collections)
          - products + product_layout_regions + product_tables + product_enrichments
          - image_product_associations + chunk_image_relationships +
            image_metafield_values + image_validations
          - product_processing_status
          - storage bucket files (if delete_storage_files=True)
          - server-side temp files in /tmp

        ──────────────────────────────────────────────────────────────────
        preserve_outputs=True (COMPLETED-JOB cleanup)
        ──────────────────────────────────────────────────────────────────
        Removes ONLY the job's tracking state — keeps the produced catalog
        data so it remains queryable / sellable / exportable. Use when the
        user deletes a completed job from the UI: they want the job entry
        gone from "Recent jobs", but the products/images/chunks the job
        produced should stay in the catalog. Removes:
          - background_jobs row (and its embedded stage_history,
            recovery_history, last_checkpoint JSONB columns)
          - product_processing_status (per-product job state — not catalog data)
          - server-side temp files in /tmp (workspace cleanup)
        Preserves:
          - documents, document_chunks, document_images
          - products and all product child tables
          - VECS embeddings
          - storage bucket files

        Args:
            job_id: Job ID to delete
            supabase_client: Supabase client instance
            vecs_service: Optional vecs service for embedding deletion
            delete_storage_files: If True AND preserve_outputs=False, delete
                storage bucket files. Ignored when preserve_outputs=True
                (storage is always preserved in that mode).
            preserve_outputs: If True, keep all produced catalog data
                (documents/products/chunks/images/embeddings/storage). Use this
                for completed-job removal-from-UI. Default False = full wipe.

        Returns:
            Dictionary with deletion statistics
        """
        try:
            mode = "PRESERVE_OUTPUTS" if preserve_outputs else "FULL_WIPE"
            self.logger.info(f"🗑️ Starting deletion for job {job_id} [mode={mode}]")

            stats = {
                'mode': mode,
                'job_deleted': False,
                'document_deleted': False,
                'chunks_deleted': 0,
                'embeddings_deleted': 0,
                'images_deleted': 0,
                'products_deleted': 0,
                'product_processing_status_deleted': 0,
                'product_layout_regions_deleted': 0,
                'product_tables_deleted': 0,
                'product_enrichments_deleted': 0,
                'image_product_associations_deleted': 0,
                'chunk_image_relationships_deleted': 0,
                'image_metafield_values_deleted': 0,
                'image_validations_deleted': 0,
                'storage_files_deleted': 0,
                'checkpoints_deleted': 0,
                'temp_files_deleted': 0,
                # XML import companion tables (cleanup added 2026-05-01)
                'data_import_jobs_deleted': 0,
                'data_import_job_products_deleted': 0,
                'data_import_history_deleted': 0,
                # Web scraping companion tables (cleanup added 2026-05-01)
                'scraping_sessions_deleted': 0,
                'scraping_pages_deleted': 0,
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

            # ──────────────────────────────────────────────────────────────
            # SHORT PATH — preserve_outputs=True (completed-job removal)
            # ──────────────────────────────────────────────────────────────
            # Only remove tracking/job-state rows. Keep documents, products,
            # chunks, images, embeddings, and storage files exactly as the
            # finished job produced them.
            if preserve_outputs:
                # 1. product_processing_status — per-product job-state.
                #    Catalog data lives in `products`; this table just tracks
                #    "did stage X for product Y on job Z succeed?" and is
                #    safe to wipe once the job row is gone.
                try:
                    pps_del = supabase_client.client.table('product_processing_status')\
                        .delete()\
                        .eq('job_id', job_id)\
                        .execute()
                    stats['product_processing_status_deleted'] = len(pps_del.data) if pps_del.data else 0
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to clean product_processing_status: {e}")
                    stats['errors'].append(f"product_processing_status deletion failed: {str(e)}")

                # 2. Delete the job row (also wipes stage_history,
                #    recovery_history, last_checkpoint JSONB columns).
                try:
                    job_del_response = supabase_client.client.table('background_jobs')\
                        .delete()\
                        .eq('id', job_id)\
                        .execute()
                    stats['job_deleted'] = len(job_del_response.data) > 0 if job_del_response.data else False
                    self.logger.info(f"✅ Deleted job tracking row (preserve_outputs)")
                except Exception as e:
                    self.logger.error(f"Failed to delete job row: {e}")
                    stats['errors'].append(f"Job deletion failed: {str(e)}")

                # 3. Server-side temp workspace cleanup (/tmp/pdf_processing/{job_id}, etc.)
                if document_id:
                    try:
                        stats['temp_files_deleted'] = self._clean_temp_directories(job_id, document_id)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Temp file cleanup failed: {e}")
                        stats['errors'].append(f"Temp files cleanup failed: {str(e)}")

                self.logger.info(
                    f"🎉 Job tracking removed (catalog data preserved) for job {job_id}"
                )
                self.logger.info(f"   Stats: {stats}")
                return stats

            # 1b. Resolve the canonical product_id list for this job. Products
            # may be linked via:
            #   - `products.source_job_id = job_id`           (XML import, scraping, PDF stage_4)
            #   - `products.source_document_id = document_id` (PDF, legacy rows)
            #   - `product_processing_status.job_id = job_id` (defensive — per-product progress table)
            # Union all three so we don't leave orphans regardless of pipeline.
            product_ids: List[str] = []
            try:
                pid_set = set()

                pids_by_job = supabase_client.client.table('products')\
                    .select('id')\
                    .eq('source_job_id', job_id)\
                    .execute()
                for row in (pids_by_job.data or []):
                    if row.get('id'):
                        pid_set.add(row['id'])

                if document_id:
                    pids_by_doc = supabase_client.client.table('products')\
                        .select('id')\
                        .eq('source_document_id', document_id)\
                        .execute()
                    for row in (pids_by_doc.data or []):
                        if row.get('id'):
                            pid_set.add(row['id'])

                pps_response = supabase_client.client.table('product_processing_status')\
                    .select('product_id')\
                    .eq('job_id', job_id)\
                    .execute()
                for row in (pps_response.data or []):
                    pid = row.get('product_id')
                    if pid:
                        pid_set.add(pid)

                product_ids = list(pid_set)
                if product_ids:
                    self.logger.info(f"📦 Resolved {len(product_ids)} products tied to job {job_id}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to resolve product list for job {job_id}: {e}")

            # 1c. Resolve the image_id list for those products. Used to clean
            # up image-side child tables (chunk_image_relationships, etc.)
            # that don't have a job_id or document_id of their own.
            image_ids: List[str] = []
            if product_ids:
                try:
                    img_resp = supabase_client.client.table('document_images')\
                        .select('id')\
                        .in_('product_id', product_ids)\
                        .execute()
                    image_ids = [r['id'] for r in (img_resp.data or []) if r.get('id')]
                    if image_ids:
                        self.logger.info(f"🖼️ Resolved {len(image_ids)} images tied to those products")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to resolve image list: {e}")
            if document_id:
                try:
                    img_resp_doc = supabase_client.client.table('document_images')\
                        .select('id')\
                        .eq('document_id', document_id)\
                        .execute()
                    for row in (img_resp_doc.data or []):
                        iid = row.get('id')
                        if iid and iid not in image_ids:
                            image_ids.append(iid)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to resolve images by document_id: {e}")

            # 2. Delete chunks. Delete by document_id (PDF path) AND by
            # product_id (XML / scraping path — chunks live under products
            # and may have no document_id).
            try:
                if document_id:
                    chunks_doc = supabase_client.client.table('document_chunks')\
                        .delete()\
                        .eq('document_id', document_id)\
                        .execute()
                    stats['chunks_deleted'] += len(chunks_doc.data) if chunks_doc.data else 0
                if product_ids:
                    chunks_pid = supabase_client.client.table('document_chunks')\
                        .delete()\
                        .in_('product_id', product_ids)\
                        .execute()
                    stats['chunks_deleted'] += len(chunks_pid.data) if chunks_pid.data else 0
                self.logger.info(f"✅ Deleted {stats['chunks_deleted']} chunks")
            except Exception as e:
                self.logger.error(f"Failed to delete chunks: {e}")
                stats['errors'].append(f"Chunks deletion failed: {str(e)}")

            # 3. Delete embeddings from vecs
            if document_id and vecs_service:
                try:
                    text_emb_deleted = await vecs_service.delete_document_embeddings(document_id)
                    stats['embeddings_deleted'] += text_emb_deleted

                    image_emb_deleted = await vecs_service.delete_image_embeddings_by_document(document_id)
                    stats['embeddings_deleted'] += image_emb_deleted

                    self.logger.info(f"✅ Deleted {stats['embeddings_deleted']} embeddings from vecs")
                except Exception as e:
                    self.logger.error(f"Failed to delete embeddings: {e}")
                    stats['errors'].append(f"Embeddings deletion failed: {str(e)}")

            # 4. Delete image-side child tables BEFORE the images themselves.
            # These tables don't have a job_id of their own; they're keyed off
            # image_id, so we need the resolved image_ids list.
            if image_ids:
                for table_name, stat_key in [
                    ('chunk_image_relationships',  'chunk_image_relationships_deleted'),
                    ('image_metafield_values',     'image_metafield_values_deleted'),
                    ('image_validations',          'image_validations_deleted'),
                    ('image_product_associations', 'image_product_associations_deleted'),
                ]:
                    try:
                        resp = supabase_client.client.table(table_name)\
                            .delete()\
                            .in_('image_id', image_ids)\
                            .execute()
                        stats[stat_key] = len(resp.data) if resp.data else 0
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to clean {table_name} by image_id: {e}")
                        stats['errors'].append(f"{table_name} deletion (by image_id) failed: {str(e)}")

            # 5. Delete the images themselves (by id when known, by document_id
            # as a fallback for legacy rows).
            try:
                deleted_imgs = 0
                if image_ids:
                    img_del = supabase_client.client.table('document_images')\
                        .delete()\
                        .in_('id', image_ids)\
                        .execute()
                    deleted_imgs += len(img_del.data) if img_del.data else 0
                if document_id:
                    img_del_doc = supabase_client.client.table('document_images')\
                        .delete()\
                        .eq('document_id', document_id)\
                        .execute()
                    deleted_imgs += len(img_del_doc.data) if img_del_doc.data else 0
                stats['images_deleted'] = deleted_imgs
                self.logger.info(f"✅ Deleted {deleted_imgs} images")
            except Exception as e:
                self.logger.error(f"Failed to delete images: {e}")
                stats['errors'].append(f"Images deletion failed: {str(e)}")

            # 6. Delete product-side child tables BEFORE products. We don't
            # trust ON DELETE CASCADE here — historical schema migrations
            # have not always added it consistently, and a missed cascade
            # leaves orphans that survive every cleanup pass.
            if product_ids:
                for table_name, stat_key in [
                    ('product_layout_regions',     'product_layout_regions_deleted'),
                    ('product_tables',             'product_tables_deleted'),
                    ('product_enrichments',        'product_enrichments_deleted'),
                    ('image_product_associations', 'image_product_associations_deleted'),
                ]:
                    try:
                        resp = supabase_client.client.table(table_name)\
                            .delete()\
                            .in_('product_id', product_ids)\
                            .execute()
                        deleted = len(resp.data) if resp.data else 0
                        # image_product_associations is counted in stat_key above too
                        # (image_id keyed). Add to whatever is already there.
                        stats[stat_key] = stats.get(stat_key, 0) + deleted
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to clean {table_name} by product_id: {e}")
                        stats['errors'].append(f"{table_name} deletion (by product_id) failed: {str(e)}")

            # 7. Delete the products themselves.
            if product_ids:
                try:
                    prod_del = supabase_client.client.table('products')\
                        .delete()\
                        .in_('id', product_ids)\
                        .execute()
                    stats['products_deleted'] = len(prod_del.data) if prod_del.data else 0
                    self.logger.info(f"✅ Deleted {stats['products_deleted']} products")
                except Exception as e:
                    self.logger.error(f"Failed to delete products: {e}")
                    stats['errors'].append(f"Products deletion failed: {str(e)}")

            # 7b. Delete product_processing_status rows (per-product job state).
            try:
                pps_del = supabase_client.client.table('product_processing_status')\
                    .delete()\
                    .eq('job_id', job_id)\
                    .execute()
                stats['product_processing_status_deleted'] = len(pps_del.data) if pps_del.data else 0
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to clean product_processing_status: {e}")
                stats['errors'].append(f"product_processing_status deletion failed: {str(e)}")

            # 8. Delete files from storage (only if delete_storage_files=True)
            if document_id and delete_storage_files:
                try:
                    stats['storage_files_deleted'] = self.cleanup_storage_bucket('documents', document_id)
                    self.logger.info(f"✅ Deleted {stats['storage_files_deleted']} storage files")
                except Exception as e:
                    self.logger.error(f"Failed to delete storage files: {e}")
                    stats['errors'].append(f"Storage deletion failed: {str(e)}")
            elif document_id and not delete_storage_files:
                self.logger.info(f"⏭️ Skipping storage file deletion (automatic cleanup mode)")

            # Checkpoints now live on background_jobs.stage_history; deleted
            # automatically when the job row is removed in step 10.
            stats['checkpoints_deleted'] = 0

            # 9. Delete document record
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

            # 9b. Delete XML import companion tables.
            # XML jobs maintain three companion tables that the cleanup
            # function ignored before 2026-05-01:
            #   data_import_jobs.background_job_id → background_jobs.id
            #   data_import_job_products.job_id   → data_import_jobs.id
            #   data_import_history.job_id        → data_import_jobs.id
            # Without explicit cleanup, deleting an XML job leaves these
            # rows orphaned forever — admin UIs that read them show ghosts
            # and disk fills with stale operational state.
            try:
                import_jobs_resp = supabase_client.client.table('data_import_jobs')\
                    .select('id')\
                    .eq('background_job_id', job_id)\
                    .execute()
                import_job_ids = [r['id'] for r in (import_jobs_resp.data or []) if r.get('id')]

                if import_job_ids:
                    for table_name, stat_key in [
                        ('data_import_history',      'data_import_history_deleted'),
                        ('data_import_job_products', 'data_import_job_products_deleted'),
                    ]:
                        try:
                            resp = supabase_client.client.table(table_name)\
                                .delete()\
                                .in_('job_id', import_job_ids)\
                                .execute()
                            stats[stat_key] = len(resp.data) if resp.data else 0
                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to clean {table_name}: {e}")
                            stats['errors'].append(f"{table_name} deletion failed: {str(e)}")

                    # Delete the data_import_jobs rows themselves
                    try:
                        dij_resp = supabase_client.client.table('data_import_jobs')\
                            .delete()\
                            .in_('id', import_job_ids)\
                            .execute()
                        stats['data_import_jobs_deleted'] = len(dij_resp.data) if dij_resp.data else 0
                        self.logger.info(
                            f"✅ Deleted {stats['data_import_jobs_deleted']} XML "
                            f"import job(s) + companion rows"
                        )
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to clean data_import_jobs: {e}")
                        stats['errors'].append(f"data_import_jobs deletion failed: {str(e)}")
            except Exception as e:
                self.logger.warning(f"⚠️ XML companion table cleanup skipped: {e}")
                stats['errors'].append(f"XML companion cleanup failed: {str(e)}")

            # 9c. Delete web-scraping companion tables.
            #   scraping_sessions.background_job_id → background_jobs.id
            #   scraping_pages.session_id           → scraping_sessions.id
            try:
                scrape_sessions_resp = supabase_client.client.table('scraping_sessions')\
                    .select('id')\
                    .eq('background_job_id', job_id)\
                    .execute()
                scrape_session_ids = [r['id'] for r in (scrape_sessions_resp.data or []) if r.get('id')]

                if scrape_session_ids:
                    try:
                        sp_resp = supabase_client.client.table('scraping_pages')\
                            .delete()\
                            .in_('session_id', scrape_session_ids)\
                            .execute()
                        stats['scraping_pages_deleted'] = len(sp_resp.data) if sp_resp.data else 0
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to clean scraping_pages: {e}")
                        stats['errors'].append(f"scraping_pages deletion failed: {str(e)}")

                    try:
                        ss_resp = supabase_client.client.table('scraping_sessions')\
                            .delete()\
                            .in_('id', scrape_session_ids)\
                            .execute()
                        stats['scraping_sessions_deleted'] = len(ss_resp.data) if ss_resp.data else 0
                        self.logger.info(
                            f"✅ Deleted {stats['scraping_sessions_deleted']} scraping "
                            f"session(s) + their pages"
                        )
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to clean scraping_sessions: {e}")
                        stats['errors'].append(f"scraping_sessions deletion failed: {str(e)}")
            except Exception as e:
                self.logger.warning(f"⚠️ Scraping companion table cleanup skipped: {e}")
                stats['errors'].append(f"Scraping companion cleanup failed: {str(e)}")

            # 10. Delete job record
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

            # 11. Clean temporary files
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


