"""
CLIP Embedding Background Job Service

Generates CLIP embeddings for images as an async background job.
This separates CLIP generation from the main PDF processing pipeline to:
1. Reduce main pipeline processing time
2. Improve error handling and retry logic
3. Allow parallel CLIP generation across multiple workers
4. Reduce CPU pressure on main pipeline

Architecture:
- Main pipeline saves images to database
- Background job picks up images without CLIP embeddings
- Generates all 5 CLIP embedding types (visual, color, texture, style, material)
- Generates understanding embedding (1024D) if vision_analysis exists
- Saves to VECS collections and document_images table
"""

import logging
import asyncio
import os
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.services.core.supabase_client import get_supabase_client
from app.services.embeddings.vecs_service import VecsService
from app.services.embeddings.real_embeddings_service import RealEmbeddingsService

logger = logging.getLogger(__name__)


class CLIPEmbeddingJobService:
    """Service for generating CLIP embeddings as background jobs."""
    
    def __init__(self):
        """Initialize CLIP embedding job service."""
        self.supabase = get_supabase_client()
        self.vecs_service = VecsService()
        self.embedding_service = RealEmbeddingsService()
        logger.info("CLIPEmbeddingJobService initialized")
    
    async def queue_clip_generation_job(
        self,
        document_id: str,
        workspace_id: str,
        priority: str = "normal"
    ) -> str:
        """
        Queue a CLIP embedding generation job for all images in a document.
        
        Args:
            document_id: Document ID
            workspace_id: Workspace ID
            priority: Job priority (low, normal, high)
            
        Returns:
            Job ID
        """
        try:
            # Create background job record
            job_data = {
                'job_type': 'clip_embedding_generation',
                'status': 'pending',
                'metadata': {
                    'document_id': document_id,
                    'workspace_id': workspace_id,
                    'priority': priority
                },
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            result = self.supabase.client.table('background_jobs').insert(job_data).execute()
            
            if result.data and len(result.data) > 0:
                job_id = result.data[0]['id']
                logger.info(f"✅ Queued CLIP embedding job {job_id} for document {document_id}")
                return job_id
            else:
                raise Exception("Failed to create background job")
                
        except Exception as e:
            logger.error(f"❌ Failed to queue CLIP embedding job: {e}")
            raise
    
    async def process_clip_generation_job(
        self,
        job_id: str,
        document_id: str,
        workspace_id: str,
        batch_size: int = 10,
        max_concurrent: int = 5  # Increased from 3 for faster embedding generation
    ) -> Dict[str, Any]:
        """
        Process CLIP embedding generation for all images in a document.
        
        Args:
            job_id: Background job ID
            document_id: Document ID
            workspace_id: Workspace ID
            batch_size: Number of images to process per batch
            max_concurrent: Maximum concurrent CLIP generations
            
        Returns:
            Dictionary with processing statistics
        """
        try:
            logger.info(f"🎨 Starting CLIP embedding generation for document {document_id}")
            
            # Update job status to processing
            await self._update_job_status(job_id, 'processing', {'started_at': datetime.utcnow().isoformat()})
            
            # Get all images for this document that don't have CLIP embeddings
            images = await self._get_images_without_clip(document_id)
            
            if not images:
                logger.info(f"✅ No images need CLIP embeddings for document {document_id}")
                await self._update_job_status(job_id, 'completed', {
                    'completed_at': datetime.utcnow().isoformat(),
                    'images_processed': 0,
                    'embeddings_generated': 0
                })
                return {'success': True, 'images_processed': 0, 'embeddings_generated': 0}
            
            logger.info(f"📊 Found {len(images)} images without CLIP embeddings")
            
            # Process images in batches
            total_processed = 0
            total_embeddings = 0
            total_failed = 0
            
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(images) + batch_size - 1) // batch_size
                
                logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} images)")
                
                # Process batch with concurrency limit
                semaphore = asyncio.Semaphore(max_concurrent)
                
                async def process_with_semaphore(img):
                    async with semaphore:
                        return await self._generate_clip_for_image(img, document_id, workspace_id)
                
                results = await asyncio.gather(
                    *[process_with_semaphore(img) for img in batch],
                    return_exceptions=True
                )
                
                # Count successes and failures
                for result in results:
                    if isinstance(result, Exception):
                        total_failed += 1
                        logger.error(f"❌ Image processing failed: {result}")
                    elif result and result.get('success'):
                        total_processed += 1
                        total_embeddings += result.get('embeddings_count', 0)
                    else:
                        total_failed += 1
                
                # Update job progress
                progress = int((i + len(batch)) / len(images) * 100)
                await self._update_job_status(job_id, 'processing', {
                    'progress': progress,
                    'images_processed': total_processed,
                    'images_failed': total_failed,
                    'embeddings_generated': total_embeddings
                })
                
                logger.info(f"✅ Batch {batch_num}/{total_batches} complete: {total_processed} processed, {total_failed} failed")
            
            # Mark job as completed
            await self._update_job_status(job_id, 'completed', {
                'completed_at': datetime.utcnow().isoformat(),
                'images_processed': total_processed,
                'images_failed': total_failed,
                'embeddings_generated': total_embeddings
            })
            
            logger.info(f"✅ CLIP embedding generation complete for document {document_id}")
            logger.info(f"   Images processed: {total_processed}/{len(images)}")
            logger.info(f"   Embeddings generated: {total_embeddings}")
            logger.info(f"   Failed: {total_failed}")
            
            return {
                'success': True,
                'images_processed': total_processed,
                'images_failed': total_failed,
                'embeddings_generated': total_embeddings
            }
            
        except Exception as e:
            logger.error(f"❌ CLIP embedding job failed: {e}")
            await self._update_job_status(job_id, 'failed', {
                'error': str(e),
                'failed_at': datetime.utcnow().isoformat()
            })
            return {'success': False, 'error': str(e)}

    async def _get_images_without_clip(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all images for a document that don't have a visual SLIG embedding yet.

        Uses the canonical has_slig_embedding boolean flag on document_images,
        which is maintained by vecs_service after each successful upsert.

        Args:
            document_id: Document ID

        Returns:
            List of image records that need embedding generation
        """
        try:
            images_result = self.supabase.client.table('document_images')\
                .select('id, image_url, storage_path, page_number, metadata, has_slig_embedding')\
                .eq('document_id', document_id)\
                .execute()

            if not images_result.data:
                return []

            images_without_slig = [img for img in images_result.data if not img.get('has_slig_embedding')]

            logger.info(
                f"Found {len(images_without_slig)}/{len(images_result.data)} images without SLIG embeddings"
            )
            return images_without_slig

        except Exception as e:
            logger.error(f"❌ Failed to get images without SLIG: {e}")
            return []

    async def _generate_clip_for_image(
        self,
        image: Dict[str, Any],
        document_id: str,
        workspace_id: str
    ) -> Dict[str, Any]:
        """
        Generate all CLIP embeddings for a single image.

        Args:
            image: Image record from database
            document_id: Document ID
            workspace_id: Workspace ID

        Returns:
            Dictionary with success status and embeddings count
        """
        try:
            image_id = image['id']
            image_url = image.get('image_url')
            storage_path = image.get('storage_path')

            if not image_url:
                logger.warning(f"⚠️ Image {image_id} has no image_url, skipping")
                return {'success': False, 'error': 'No image_url'}

            logger.info(f"🎨 Generating CLIP embeddings for image {image_id}")

            # Download image from Supabase Storage
            image_data = await self._download_image_from_storage(storage_path or image_url)

            if not image_data:
                logger.error(f"❌ Failed to download image {image_id}")
                return {'success': False, 'error': 'Failed to download image'}

            # Convert to base64
            image_base64 = f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"

            # Generate all embeddings using RealEmbeddingsService
            embedding_result = await self.embedding_service.generate_all_embeddings(
                entity_id=image_id,
                entity_type="image",
                text_content="",  # No text for images
                image_data=image_base64,
                material_properties={}
            )

            if not embedding_result or not embedding_result.get('success'):
                logger.error(f"❌ Embedding generation failed for image {image_id}")
                return {'success': False, 'error': 'Embedding generation failed'}

            embeddings = embedding_result.get('embeddings', {})
            embeddings_count = 0

            # Save visual SLIG embedding (768D from SigLIP2 via SLIG cloud endpoint)
            # to VECS — single source of truth. The has_slig_embedding boolean
            # flag on document_images is updated automatically by vecs_service.
            visual_embedding = embeddings.get('visual_768')
            if visual_embedding:
                metadata = {
                    'document_id': document_id,
                    'workspace_id': workspace_id,
                    'page_number': image.get('page_number'),
                    'image_url': image_url
                }
                await self.vecs_service.upsert_image_embedding(
                    image_id=image_id,
                    siglip_embedding=visual_embedding,
                    metadata=metadata
                )
                embeddings_count += 1
                logger.debug(f"✅ Saved visual SLIG embedding (768D) for image {image_id}")
                logger.debug(f"✅ Saved visual SLIG to VECS for image {image_id}")

            # Save specialized embeddings (SLIG 768D, emitted by real_embeddings_service
            # under the keys color_slig_768 / texture_slig_768 / style_slig_768 / material_slig_768).
            specialized_embeddings = {}
            if embeddings.get('color_slig_768'):
                specialized_embeddings['color'] = embeddings.get('color_slig_768')
            if embeddings.get('texture_slig_768'):
                specialized_embeddings['texture'] = embeddings.get('texture_slig_768')
            if embeddings.get('style_slig_768'):
                specialized_embeddings['style'] = embeddings.get('style_slig_768')
            if embeddings.get('material_slig_768'):
                specialized_embeddings['material'] = embeddings.get('material_slig_768')

            if specialized_embeddings:
                metadata = {
                    'document_id': document_id,
                    'workspace_id': workspace_id,
                    'page_number': image.get('page_number')
                }
                await self.vecs_service.upsert_specialized_embeddings(
                    image_id=image_id,
                    embeddings=specialized_embeddings,
                    metadata=metadata
                )
                embeddings_count += len(specialized_embeddings)
                logger.debug(f"✅ Saved {len(specialized_embeddings)} specialized embeddings for image {image_id}")

                # ✨ NEW: Stage 3.5 - Convert visual embeddings to text metadata
                try:
                    from app.services.metadata.visual_metadata_service import VisualMetadataService

                    logger.info(f"🎨 Stage 3.5: Converting visual embeddings to text metadata for {image_id}")
                    visual_metadata_service = VisualMetadataService(workspace_id=workspace_id)

                    # Prepare embeddings for conversion (SLIG 768D — canonical schema).
                    embeddings_for_conversion = {}
                    if embeddings.get('color_slig_768'):
                        embeddings_for_conversion['color_slig_768'] = embeddings.get('color_slig_768')
                    if embeddings.get('texture_slig_768'):
                        embeddings_for_conversion['texture_slig_768'] = embeddings.get('texture_slig_768')
                    if embeddings.get('material_slig_768'):
                        embeddings_for_conversion['material_slig_768'] = embeddings.get('material_slig_768')
                    if embeddings.get('style_slig_768'):
                        embeddings_for_conversion['style_slig_768'] = embeddings.get('style_slig_768')

                    if embeddings_for_conversion:
                        visual_metadata_result = await visual_metadata_service.process_image_visual_metadata(
                            image_id=image_id,
                            embeddings=embeddings_for_conversion
                        )

                        if visual_metadata_result.get('success'):
                            logger.info(f"✅ Visual metadata extracted and saved for {image_id}")
                        else:
                            logger.warning(f"⚠️ Visual metadata extraction failed: {visual_metadata_result.get('error')}")
                    else:
                        logger.debug(f"ℹ️ No SigLIP embeddings available for visual metadata extraction")

                except Exception as visual_meta_error:
                    logger.warning(f"⚠️ Visual metadata extraction failed (non-critical): {visual_meta_error}")

            # Generate understanding embedding if vision_analysis exists in DB
            try:
                vision_result = self.supabase.client.table('document_images')\
                    .select('vision_analysis, material_properties')\
                    .eq('id', image_id)\
                    .single()\
                    .execute()

                if vision_result.data and vision_result.data.get('vision_analysis'):
                    understanding_embedding = await self.embedding_service.generate_understanding_embedding(
                        vision_analysis=vision_result.data['vision_analysis'],
                        material_properties=vision_result.data.get('material_properties')
                    )
                    if understanding_embedding:
                        metadata = {
                            'document_id': document_id,
                            'workspace_id': workspace_id,
                            'page_number': image.get('page_number')
                        }
                        await self.vecs_service.upsert_understanding_embedding(
                            image_id=image_id,
                            embedding=understanding_embedding,
                            metadata=metadata
                        )
                        embeddings_count += 1
                        logger.debug(f"✅ Generated understanding embedding for image {image_id}")
            except Exception as understanding_error:
                logger.warning(f"⚠️ Understanding embedding failed for {image_id} (non-critical): {understanding_error}")

            logger.info(f"✅ Generated {embeddings_count} embeddings for image {image_id}")

            return {
                'success': True,
                'embeddings_count': embeddings_count,
                'image_id': image_id
            }

        except Exception as e:
            logger.error(f"❌ Failed to generate CLIP for image {image.get('id')}: {e}")
            return {'success': False, 'error': str(e)}

    async def _download_image_from_storage(self, storage_path: str) -> Optional[bytes]:
        """
        Download image from Supabase Storage.

        Args:
            storage_path: Storage path or URL

        Returns:
            Image bytes or None if failed
        """
        try:
            # Extract bucket and path from storage_path
            # Format: "document-images/workspace_id/document_id/filename.jpg"
            if storage_path.startswith('http'):
                # It's a URL, download directly
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(storage_path)
                    if response.status_code == 200:
                        return response.content
                    else:
                        logger.error(f"❌ Failed to download image from URL: {response.status_code}")
                        return None
            else:
                # It's a storage path, use Supabase client
                bucket_name = 'document-images'
                result = self.supabase.client.storage.from_(bucket_name).download(storage_path)
                return result

        except Exception as e:
            logger.error(f"❌ Failed to download image from storage: {e}")
            return None

    async def _update_job_status(
        self,
        job_id: str,
        status: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update background job status and metadata.

        Args:
            job_id: Job ID
            status: New status (pending, processing, completed, failed)
            metadata: Additional metadata to merge

        Returns:
            True if successful, False otherwise
        """
        try:
            update_data = {
                'status': status,
                'updated_at': datetime.utcnow().isoformat()
            }

            # Merge metadata
            if metadata:
                # Get current metadata
                result = self.supabase.client.table('background_jobs')\
                    .select('metadata')\
                    .eq('id', job_id)\
                    .single()\
                    .execute()

                current_payload = result.data.get('metadata', {}) if result.data else {}
                current_payload.update(metadata)
                update_data['metadata'] = current_payload

            self.supabase.client.table('background_jobs')\
                .update(update_data)\
                .eq('id', job_id)\
                .execute()

            return True

        except Exception as e:
            logger.error(f"❌ Failed to update job status: {e}")
            return False


# Singleton instance
_clip_job_service = None


def get_clip_job_service() -> CLIPEmbeddingJobService:
    """Get singleton instance of CLIP embedding job service."""
    global _clip_job_service
    if _clip_job_service is None:
        _clip_job_service = CLIPEmbeddingJobService()
    return _clip_job_service


