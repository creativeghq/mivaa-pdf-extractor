"""
Data Import Service

Orchestrates the processing of XML import jobs:
- Batch processing (10 products at a time)
- Image downloads (5 concurrent)
- Metadata extraction
- Product normalization
- Queue for product creation pipeline
- Checkpoint recovery
- Real-time progress updates
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from app.services.core.supabase_client import get_supabase_client
from app.services.core.async_queue_service import AsyncQueueService
from app.services.tracking.checkpoint_recovery_service import checkpoint_recovery_service, ProcessingStage
from app.services.images.image_download_service import ImageDownloadService
import sentry_sdk  # ✅ NEW: Sentry integration

logger = logging.getLogger(__name__)


class DataImportService:
    """Service for processing data import jobs"""

    def __init__(self):
        supabase_wrapper = get_supabase_client()
        self.supabase = supabase_wrapper.client
        self.async_queue = AsyncQueueService()
        self.image_downloader = ImageDownloadService()
        self.batch_size = 10  # Process 10 products at a time
        self.max_concurrent_images = 5  # Download 5 images concurrently

    async def process_import_job(self, job_id: str, workspace_id: str) -> Dict[str, Any]:
        """
        Process an import job from start to finish.

        Args:
            job_id: Import job ID
            workspace_id: Workspace ID

        Returns:
            Processing result with statistics
        """
        # ✅ NEW: Start Sentry transaction for performance monitoring
        with sentry_sdk.start_transaction(op="xml_import", name="process_import_job") as transaction:
            transaction.set_tag("job_id", job_id)
            transaction.set_tag("workspace_id", workspace_id)

            try:
                logger.info(f"🚀 Starting import job processing: {job_id}")

                # ✅ NEW: Add breadcrumb
                sentry_sdk.add_breadcrumb(
                    category="xml_import",
                    message=f"Starting XML import job {job_id}",
                    level="info"
                )

                # Get job details
                job = await self._get_job(job_id)
                if not job:
                    raise ValueError(f"Job {job_id} not found")

                # Update job status to processing
                await self._update_job_status(job_id, 'processing', started_at=datetime.utcnow())

                # 🆕 Update background_jobs to 'processing'
                await self._update_background_job_status(job_id, 'processing', started_at=datetime.utcnow())

                # Get products from job metadata
                products = job.get('metadata', {}).get('products', [])
                if not products:
                    raise ValueError(f"No products found in job {job_id}")

                total_products = len(products)
                logger.info(f"📦 Processing {total_products} products in batches of {self.batch_size}")

                # ✅ NEW: Set transaction data
                transaction.set_data("total_products", total_products)
                transaction.set_data("batch_size", self.batch_size)

                # Check for existing checkpoint
                last_checkpoint = await checkpoint_recovery_service.get_last_checkpoint(job_id)
                start_batch = 0
                if last_checkpoint:
                    start_batch = last_checkpoint.get('batch_index', 0)
                    logger.info(f"♻️ Resuming from batch {start_batch}")

                # Process in batches
                processed_count = 0
                failed_count = 0

                for batch_index in range(start_batch, (total_products + self.batch_size - 1) // self.batch_size):
                    batch_start = batch_index * self.batch_size
                    batch_end = min(batch_start + self.batch_size, total_products)
                    batch = products[batch_start:batch_end]

                    logger.info(f"🔄 Processing batch {batch_index + 1}: products {batch_start + 1}-{batch_end}")

                    # ✅ NEW: Add breadcrumb for batch processing
                    sentry_sdk.add_breadcrumb(
                        category="xml_import",
                        message=f"Processing batch {batch_index + 1}: products {batch_start + 1}-{batch_end}",
                        level="info",
                        data={"batch_index": batch_index + 1, "batch_size": len(batch)}
                    )

                    # Process batch
                    batch_result = await self._process_batch(
                        job_id=job_id,
                        workspace_id=workspace_id,
                        products=batch,
                        batch_index=batch_index,
                        field_mappings=job.get('field_mappings', {})
                    )

                    processed_count += batch_result['processed']
                    failed_count += batch_result['failed']

                    # Update progress
                    progress = int((batch_end / total_products) * 100)
                    await self._update_job_progress(
                        job_id=job_id,
                        processed=processed_count,
                        failed=failed_count,
                        total=total_products,
                        stage=f'batch_{batch_index + 1}',
                        progress_percent=progress
                    )

                    # Save checkpoint
                    await checkpoint_recovery_service.save_checkpoint(
                        job_id=job_id,
                        stage=ProcessingStage.PRODUCTS_CREATED,
                        data={
                            'batch_index': batch_index + 1,
                            'processed_count': processed_count,
                            'failed_count': failed_count
                        }
                    )

                    logger.info(f"✅ Batch {batch_index + 1} complete: {batch_result['processed']} processed, {batch_result['failed']} failed")

                # Mark job as completed
                await self._update_job_status(
                    job_id=job_id,
                    status='completed',
                    completed_at=datetime.utcnow(),
                    processed_products=processed_count,
                    failed_products=failed_count
                )

                # 🆕 Update background_jobs to 'completed'
                await self._update_background_job_status(
                    job_id=job_id,
                    status='completed',
                    completed_at=datetime.utcnow(),
                    progress_percent=100,
                    metadata={
                        'total_products': total_products,
                        'processed': processed_count,
                        'failed': failed_count,
                        'completion_rate': (processed_count / total_products * 100) if total_products > 0 else 0
                    }
                )

                logger.info(f"🎉 Import job {job_id} completed: {processed_count}/{total_products} products processed")

                # ✅ NEW: Add completion breadcrumb
                sentry_sdk.add_breadcrumb(
                    category="xml_import",
                    message=f"XML import job {job_id} completed successfully",
                    level="info",
                    data={
                        "total_products": total_products,
                        "processed": processed_count,
                        "failed": failed_count,
                        "completion_rate": (processed_count / total_products * 100) if total_products > 0 else 0
                    }
                )

                # ✅ NEW: Set transaction status
                transaction.set_status("ok")

                return {
                    'success': True,
                    'job_id': job_id,
                    'total_products': total_products,
                    'processed': processed_count,
                    'failed': failed_count,
                    'completion_rate': (processed_count / total_products * 100) if total_products > 0 else 0
                }

            except Exception as e:
                logger.error(f"❌ Import job {job_id} failed: {e}", exc_info=True)

                # ✅ NEW: Capture exception in Sentry
                sentry_sdk.capture_exception(e)

                # ✅ NEW: Set transaction status
                transaction.set_status("internal_error")

                # Mark job as failed
                await self._update_job_status(
                    job_id=job_id,
                    status='failed',
                    error_message=str(e),
                    completed_at=datetime.utcnow()
                )

                # 🆕 Update background_jobs to 'failed'
                await self._update_background_job_status(
                    job_id=job_id,
                    status='failed',
                    error_message=str(e),
                    completed_at=datetime.utcnow()
                )

                raise

    async def _process_batch(
        self,
        job_id: str,
        workspace_id: str,
        products: List[Dict],
        batch_index: int,
        field_mappings: Dict[str, str]
    ) -> Dict[str, int]:
        """
        Process a batch of products.
        
        Args:
            job_id: Import job ID
            workspace_id: Workspace ID
            products: List of product data
            batch_index: Current batch index
            field_mappings: Field mapping configuration
            
        Returns:
            Dictionary with processed and failed counts
        """
        processed = 0
        failed = 0
        
        for product in products:
            try:
                # Apply field mappings to normalize product data
                normalized_product = await self._normalize_product(product, field_mappings)
                
                # Download images if present
                if normalized_product.get('images'):
                    image_urls = normalized_product['images']
                    downloaded_images = await self._download_images(image_urls, job_id, workspace_id)
                    normalized_product['downloaded_images'] = downloaded_images
                
                # Queue for product creation pipeline
                await self._queue_product_processing(
                    job_id=job_id,
                    workspace_id=workspace_id,
                    product_data=normalized_product
                )
                
                # Record in import history
                await self._record_import_history(
                    job_id=job_id,
                    source_data=product,
                    normalized_data=normalized_product,
                    status='success'
                )
                
                processed += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to process product: {e}")
                
                # Record failure in import history
                await self._record_import_history(
                    job_id=job_id,
                    source_data=product,
                    normalized_data={},
                    status='failed',
                    error_details=str(e)
                )
                
                failed += 1
        
        return {'processed': processed, 'failed': failed}

    async def _normalize_product(
        self,
        product: Dict[str, Any],
        field_mappings: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Normalize product data using field mappings.
        
        Args:
            product: Raw product data from XML
            field_mappings: Field mapping configuration
            
        Returns:
            Normalized product data
        """
        normalized = {}
        
        # Apply field mappings
        for xml_field, target_field in field_mappings.items():
            if xml_field in product:
                normalized[target_field] = product[xml_field]
        
        # Ensure required fields exist
        if 'name' not in normalized:
            normalized['name'] = product.get('name', 'Unknown Product')
        
        if 'factory_name' not in normalized:
            normalized['factory_name'] = product.get('factory_name', 'Unknown Factory')
        
        if 'material_category' not in normalized:
            normalized['material_category'] = product.get('material_category', 'Unknown Category')
        
        # Add metadata
        normalized['metadata'] = {
            'source_type': 'xml_import',
            'import_date': datetime.utcnow().isoformat(),
            **product.get('metadata', {})
        }
        
        return normalized

    async def _download_images(
        self,
        image_urls: List[str],
        job_id: str,
        workspace_id: str
    ) -> List[Dict[str, Any]]:
        """
        Download images concurrently using ImageDownloadService.

        Args:
            image_urls: List of image URLs
            job_id: Import job ID
            workspace_id: Workspace ID

        Returns:
            List of downloaded image references
        """
        if not image_urls:
            return []

        logger.info(f"📥 Downloading {len(image_urls)} images")

        # Download images using ImageDownloadService
        downloaded_images = await self.image_downloader.download_images(
            urls=image_urls,
            job_id=job_id,
            workspace_id=workspace_id,
            max_concurrent=self.max_concurrent_images
        )

        return downloaded_images

    async def _queue_product_processing(
        self,
        job_id: str,
        workspace_id: str,
        product_data: Dict[str, Any]
    ) -> None:
        """
        Create product directly in database (XML imports don't need full PDF pipeline).

        For XML imports, we create products directly with:
        - Product metadata from field mappings
        - Downloaded images linked to product
        - Text content for chunking/embeddings (async)

        Args:
            job_id: Import job ID
            workspace_id: Workspace ID
            product_data: Normalized product data
        """
        try:
            product_name = product_data.get('name', 'Unknown Product')
            logger.info(f"📤 Creating product: {product_name}")

            # Build product record for database
            product_record = {
                "name": product_name,
                "description": product_data.get('description', '')[:200],
                "long_description": product_data.get('description', '')[:1000],
                "workspace_id": workspace_id,
                "properties": {
                    "factory_name": product_data.get('factory_name'),
                    "factory_group_name": product_data.get('factory_group_name'),
                    "material_category": product_data.get('material_category'),
                    "price": product_data.get('price'),
                    "color": product_data.get('color'),
                    "dimensions": product_data.get('dimensions'),
                    "designer": product_data.get('designer'),
                    "collection": product_data.get('collection'),
                    "finish": product_data.get('finish'),
                    "material": product_data.get('material'),
                    "import_source": "xml_import",
                    "import_job_id": job_id,
                    "auto_generated": True,
                    "generation_timestamp": datetime.utcnow().isoformat()
                },
                "metadata": {
                    "extracted_from": "xml_import",
                    "import_job_id": job_id,
                    "extraction_date": datetime.utcnow().isoformat(),
                    "workspace_id": workspace_id,
                    **product_data.get('metadata', {})
                },
                "status": "draft",
                "created_from_type": "xml_import",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }

            # Add source tracking fields
            product_record['source_type'] = 'xml_import'
            product_record['source_job_id'] = job_id
            product_record['import_batch_id'] = f"xml_{job_id}"

            # Insert product into database
            insert_response = self.supabase.client.table('products').insert(product_record).execute()

            if not insert_response.data:
                raise ValueError("Failed to insert product - no data returned")

            product_id = insert_response.data[0]['id']
            logger.info(f"✅ Created product {product_id}: {product_name}")

            # Link downloaded images to product
            downloaded_images = product_data.get('downloaded_images', [])
            if downloaded_images:
                await self._link_images_to_product(product_id, downloaded_images, workspace_id, job_id)

            # Queue for async text processing (chunking, embeddings)
            # This happens in background - product is already created
            if product_data.get('description'):
                await self._queue_text_processing(product_id, product_data, workspace_id, job_id)

        except Exception as e:
            logger.error(f"❌ Failed to create product {product_data.get('name')}: {e}")
            raise

    async def _link_images_to_product(
        self,
        product_id: str,
        downloaded_images: List[Dict[str, Any]],
        workspace_id: str,
        job_id: str = None
    ) -> None:
        """
        Link downloaded images to product in database.

        Args:
            product_id: Product ID
            downloaded_images: List of downloaded image references
            workspace_id: Workspace ID
        """
        try:
            # Step 1: Create image records in document_images table
            image_records = []
            for img in downloaded_images:
                if not img.get('success'):
                    continue

                # ✅ FIXED: Map to correct document_images columns
                image_record = {
                    "document_id": product_id,  # Use product_id as document_id for XML imports
                    "workspace_id": workspace_id,
                    "image_url": img['storage_url'],  # ✅ Use image_url column
                    "source_type": "xml_import",
                    "source_job_id": job_id,
                    "metadata": {
                        "source": "xml_import",
                        "index": img['index'],
                        "storage_path": img['storage_path'],
                        "original_url": img['original_url'],
                        "filename": img['filename'],
                        "content_type": img['content_type'],
                        "size_bytes": img['size_bytes']
                    }
                }
                image_records.append(image_record)

            if not image_records:
                logger.info(f"⏭️ No images to link for product {product_id}")
                return

            # Insert images into document_images
            insert_response = self.supabase.client.table('document_images').insert(image_records).execute()

            if not insert_response.data:
                logger.warning(f"⚠️ No images were inserted for product {product_id}")
                return

            logger.info(f"✅ Inserted {len(insert_response.data)} images into document_images")

            # Step 2: Create product-image relationships
            relationship_records = []
            for idx, image_data in enumerate(insert_response.data):
                relationship_records.append({
                    "product_id": product_id,
                    "image_id": image_data['id'],
                    "relationship_type": "depicts",
                    "relevance_score": 1.0 - (idx * 0.05)  # First image gets highest score
                })

            if relationship_records:
                self.supabase.client.table('product_image_relationships').insert(relationship_records).execute()
                logger.info(f"✅ Created {len(relationship_records)} product-image relationships")

        except Exception as e:
            logger.error(f"❌ Failed to link images to product {product_id}: {e}")
            logger.exception(e)
            # Don't raise - product is already created

    async def _queue_text_processing(
        self,
        product_id: str,
        product_data: Dict[str, Any],
        workspace_id: str,
        job_id: str = None
    ) -> None:
        """
        Queue product text for async processing (chunking, embeddings).

        This creates chunks and embeddings in the background without blocking
        the import process.

        Args:
            product_id: Product ID
            product_data: Product data with text content
            workspace_id: Workspace ID
        """
        try:
            # Create a text chunk for the product description
            description = product_data.get('description', '')

            if not description or len(description) < 50:
                logger.info(f"⏭️ Skipping text processing for product {product_id} - insufficient content")
                return

            # Create chunk record
            chunk_record = {
                "document_id": product_id,  # ✅ FIXED: Use 'document_id' not 'product_id'
                "workspace_id": workspace_id,
                "content": description,
                "chunk_index": 0,
                "source_type": "xml_import",
                "source_job_id": job_id,
                "metadata": {
                    "source": "xml_import",
                    "product_id": product_id,  # Store product_id in metadata for reference
                    "product_name": product_data.get('name'),
                    "auto_generated": True
                }
            }

            chunk_response = self.supabase.client.table('document_chunks').insert(chunk_record).execute()

            if chunk_response.data:
                chunk_id = chunk_response.data[0]['id']
                logger.info(f"✅ Created chunk {chunk_id} for product {product_id}")

                # Queue for embedding generation (async)
                await self.async_queue.queue_ai_analysis_jobs(
                    document_id=product_id,  # Use product_id as document_id
                    chunks=[{'id': chunk_id}],
                    analysis_type='embedding_generation',
                    priority=0
                )

        except Exception as e:
            logger.error(f"❌ Failed to queue text processing for product {product_id}: {e}")
            # Don't raise - product is already created

    async def _get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job details from database."""
        try:
            response = self.supabase.client.table('data_import_jobs').select('*').eq('id', job_id).single().execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None

    async def _update_job_status(self, job_id: str, status: str, **kwargs) -> None:
        """Update job status in database."""
        try:
            update_data = {'status': status, 'updated_at': datetime.utcnow().isoformat()}
            update_data.update(kwargs)
            
            # Convert datetime objects to ISO strings
            for key, value in update_data.items():
                if isinstance(value, datetime):
                    update_data[key] = value.isoformat()
            
            self.supabase.client.table('data_import_jobs').update(update_data).eq('id', job_id).execute()
            logger.info(f"✅ Updated job {job_id} status to {status}")
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")

    async def _update_job_progress(
        self,
        job_id: str,
        processed: int,
        failed: int,
        total: int,
        stage: str,
        progress_percent: int = None
    ) -> None:
        """Update job progress in database and job_progress table."""
        try:
            if progress_percent is None:
                progress_percent = int((processed / total) * 100) if total > 0 else 0

            # Update data_import_jobs table
            self.supabase.client.table('data_import_jobs').update({
                'processed_products': processed,
                'failed_products': failed,
                'metadata': {
                    'current_stage': stage,
                    'progress_percentage': progress_percent
                }
            }).eq('id', job_id).execute()

            # 🆕 Get background_job_id from data_import_jobs
            job_response = self.supabase.client.table('data_import_jobs').select('background_job_id').eq('id', job_id).single().execute()
            background_job_id = job_response.data.get('background_job_id') if job_response.data else None

            if background_job_id:
                # 🆕 Update job_progress table (upsert)
                self.supabase.client.table('job_progress').upsert({
                    'job_id': background_job_id,
                    'stage': f'xml_import_{stage}',
                    'progress_percent': progress_percent,
                    'current_step': f'Processing: {processed}/{total} products',
                    'details': {
                        'processed': processed,
                        'failed': failed,
                        'total': total,
                        'current_stage': stage
                    }
                }, on_conflict='job_id,stage').execute()

                # 🆕 Update background_jobs heartbeat and progress
                self.supabase.client.table('background_jobs').update({
                    'progress_percent': progress_percent,
                    'last_heartbeat': datetime.utcnow().isoformat(),
                    'metadata': {
                        'processed': processed,
                        'failed': failed,
                        'total': total,
                        'current_stage': stage
                    }
                }).eq('id', background_job_id).execute()

                logger.debug(f"📊 Updated job progress: {progress_percent}% ({processed}/{total} products)")
        except Exception as e:
            logger.error(f"Failed to update job progress: {e}")

    async def _update_background_job_status(
        self,
        job_id: str,
        status: str,
        **kwargs
    ) -> None:
        """
        Update background_jobs status via data_import_jobs.

        Args:
            job_id: data_import_jobs ID
            status: New status
            **kwargs: Additional fields (started_at, completed_at, error_message, progress_percent, metadata)
        """
        try:
            # Get background_job_id from data_import_jobs
            job_response = self.supabase.client.table('data_import_jobs').select('background_job_id').eq('id', job_id).single().execute()
            background_job_id = job_response.data.get('background_job_id') if job_response.data else None

            if not background_job_id:
                logger.warning(f"No background_job_id found for data_import_job {job_id}")
                return

            # Build update data
            update_data = {'status': status}

            # Convert datetime objects to ISO strings
            for key, value in kwargs.items():
                if isinstance(value, datetime):
                    update_data[key] = value.isoformat()
                else:
                    update_data[key] = value

            # Always update last_heartbeat
            update_data['last_heartbeat'] = datetime.utcnow().isoformat()

            # Update background_jobs table
            self.supabase.client.table('background_jobs').update(update_data).eq('id', background_job_id).execute()
            logger.info(f"✅ Updated background_job {background_job_id} status to {status}")

        except Exception as e:
            logger.error(f"Failed to update background_job status: {e}")
            # Don't raise - this is non-critical

    async def _record_import_history(
        self,
        job_id: str,
        source_data: Dict[str, Any],
        normalized_data: Dict[str, Any],
        status: str,
        error_details: Optional[str] = None
    ) -> None:
        """Record import history entry."""
        try:
            self.supabase.client.table('data_import_history').insert({
                'job_id': job_id,
                'source_data': source_data,
                'normalized_data': normalized_data,
                'processing_status': status,
                'error_details': error_details,
                'created_at': datetime.utcnow().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to record import history: {e}")


