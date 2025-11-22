"""
Pipeline Orchestrator - Coordinates modular PDF processing pipeline.

This orchestrator calls internal API endpoints for each pipeline stage,
providing clean separation, retry logic, and comprehensive progress tracking.
"""

import logging
import asyncio
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.services.job_tracker import JobTracker
from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the modular PDF processing pipeline."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.supabase_client = get_supabase_client()
    
    async def process_document(
        self,
        job_id: str,
        document_id: str,
        workspace_id: str,
        extracted_images: List[Dict[str, Any]],
        extracted_text: str,
        product_ids: Optional[List[str]] = None,
        confidence_threshold: float = 0.7,
        similarity_threshold: float = 0.5,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> Dict[str, Any]:
        """
        Orchestrate the complete PDF processing pipeline.
        
        Pipeline Stages:
        1. Classify Images (0-20%)
        2. Upload Images (20-40%)
        3. Save Images & Generate CLIP (40-60%)
        4. Create Chunks (60-80%)
        5. Create Relationships (80-100%)
        
        Args:
            job_id: Job ID for tracking
            document_id: Document ID
            workspace_id: Workspace ID
            extracted_images: List of extracted images
            extracted_text: Extracted text from PDF
            product_ids: Optional list of product IDs
            confidence_threshold: Minimum confidence for image classification
            similarity_threshold: Minimum similarity for relationships
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Dict with complete pipeline results
        """
        logger.info(f"🚀 [Orchestrator] Starting pipeline for job {job_id}")
        
        # Initialize tracker
        tracker = JobTracker(job_id)
        await tracker.update_stage("PIPELINE_START", 0, sync_to_db=True)
        
        results = {
            'job_id': job_id,
            'document_id': document_id,
            'stages': {},
            'errors': [],
            'start_time': datetime.utcnow().isoformat()
        }
        
        try:
            # Stage 1: Classify Images (0-20%)
            logger.info(f"📸 [Stage 1/5] Classifying {len(extracted_images)} images...")
            classify_result = await self._call_internal_endpoint(
                endpoint=f"/api/internal/classify-images/{job_id}",
                data={
                    "job_id": job_id,
                    "extracted_images": extracted_images,
                    "confidence_threshold": confidence_threshold
                }
            )
            
            if not classify_result.get('success'):
                raise Exception(f"Image classification failed: {classify_result}")
            
            results['stages']['classify'] = classify_result
            material_images = classify_result.get('material_images', [])
            await tracker.update_stage("IMAGE_CLASSIFICATION", 20, 
                                      metadata={'material_count': len(material_images)},
                                      sync_to_db=True)
            
            logger.info(f"✅ [Stage 1/5] Classified: {len(material_images)} material images")
            
            # Stage 2: Upload Images (20-40%)
            if material_images:
                logger.info(f"📤 [Stage 2/5] Uploading {len(material_images)} material images...")
                upload_result = await self._call_internal_endpoint(
                    endpoint=f"/api/internal/upload-images/{job_id}",
                    data={
                        "job_id": job_id,
                        "material_images": material_images,
                        "document_id": document_id
                    }
                )
                
                if not upload_result.get('success'):
                    raise Exception(f"Image upload failed: {upload_result}")
                
                results['stages']['upload'] = upload_result
                uploaded_images = upload_result.get('uploaded_images', [])
                await tracker.update_stage("IMAGE_UPLOAD", 40,
                                          metadata={'uploaded_count': len(uploaded_images)},
                                          sync_to_db=True)
                
                logger.info(f"✅ [Stage 2/5] Uploaded: {len(uploaded_images)} images")
                
                # Stage 3: Save Images & Generate CLIP (40-60%)
                logger.info(f"💾 [Stage 3/5] Saving images and generating CLIP embeddings...")
                save_result = await self._call_internal_endpoint(
                    endpoint=f"/api/internal/save-images-db/{job_id}",
                    data={
                        "job_id": job_id,
                        "material_images": uploaded_images,
                        "document_id": document_id,
                        "workspace_id": workspace_id
                    }
                )
                
                if not save_result.get('success'):
                    raise Exception(f"Image save and CLIP generation failed: {save_result}")
                
                results['stages']['save_and_clip'] = save_result
                await tracker.update_stage("IMAGE_SAVE_AND_CLIP", 60,
                                          metadata={
                                              'images_saved': save_result.get('images_saved', 0),
                                              'clip_embeddings': save_result.get('clip_embeddings_generated', 0)
                                          },
                                          sync_to_db=True)
                
                logger.info(f"✅ [Stage 3/5] Saved: {save_result.get('images_saved', 0)} images, "
                          f"{save_result.get('clip_embeddings_generated', 0)} CLIP embeddings")
            
            else:
                logger.info("⚠️ [Stages 2-3] No material images to upload/save, skipping...")
                await tracker.update_stage("IMAGE_UPLOAD", 60, sync_to_db=True)

            # Stage 4: Create Chunks (60-80%)
            logger.info(f"📝 [Stage 4/5] Creating chunks from extracted text...")
            chunks_result = await self._call_internal_endpoint(
                endpoint=f"/api/internal/create-chunks/{job_id}",
                data={
                    "job_id": job_id,
                    "document_id": document_id,
                    "workspace_id": workspace_id,
                    "extracted_text": extracted_text,
                    "product_ids": product_ids,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
            )

            if not chunks_result.get('success'):
                raise Exception(f"Chunking failed: {chunks_result}")

            results['stages']['chunks'] = chunks_result
            await tracker.update_stage("CHUNKING", 80,
                                      metadata={
                                          'chunks_created': chunks_result.get('chunks_created', 0),
                                          'embeddings_generated': chunks_result.get('embeddings_generated', 0)
                                      },
                                      sync_to_db=True)

            logger.info(f"✅ [Stage 4/5] Created: {chunks_result.get('chunks_created', 0)} chunks, "
                      f"{chunks_result.get('embeddings_generated', 0)} text embeddings")

            # Stage 5: Create Relationships (80-100%)
            if product_ids:
                logger.info(f"🔗 [Stage 5/5] Creating relationships...")
                relationships_result = await self._call_internal_endpoint(
                    endpoint=f"/api/internal/create-relationships/{job_id}",
                    data={
                        "job_id": job_id,
                        "document_id": document_id,
                        "product_ids": product_ids,
                        "similarity_threshold": similarity_threshold
                    }
                )

                if not relationships_result.get('success'):
                    raise Exception(f"Relationship creation failed: {relationships_result}")

                results['stages']['relationships'] = relationships_result
                await tracker.update_stage("RELATIONSHIPS", 100,
                                          metadata={
                                              'chunk_image_relationships': relationships_result.get('chunk_image_relationships', 0),
                                              'product_image_relationships': relationships_result.get('product_image_relationships', 0)
                                          },
                                          sync_to_db=True)

                logger.info(f"✅ [Stage 5/5] Created: {relationships_result.get('chunk_image_relationships', 0)} chunk-image, "
                          f"{relationships_result.get('product_image_relationships', 0)} product-image relationships")

            else:
                logger.info("⚠️ [Stage 5] No products provided, skipping relationships...")
                await tracker.update_stage("RELATIONSHIPS", 100, sync_to_db=True)

            # Pipeline complete
            results['end_time'] = datetime.utcnow().isoformat()
            results['success'] = True
            await tracker.update_stage("COMPLETED", 100, sync_to_db=True)

            logger.info(f"🎉 [Orchestrator] Pipeline complete for job {job_id}")

            return results

        except Exception as e:
            logger.error(f"❌ [Orchestrator] Pipeline failed for job {job_id}: {e}")
            results['errors'].append(str(e))
            results['success'] = False
            results['end_time'] = datetime.utcnow().isoformat()

            # Update job status to failed
            await tracker.update_stage("FAILED", -1,
                                      metadata={'error': str(e)},
                                      sync_to_db=True)

            raise

    async def _call_internal_endpoint(
        self,
        endpoint: str,
        data: Dict[str, Any],
        max_retries: int = 3,
        timeout: float = 300.0
    ) -> Dict[str, Any]:
        """
        Call an internal API endpoint with retry logic.

        Args:
            endpoint: API endpoint path
            data: Request data
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds

        Returns:
            Response data
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, json=data)
                    response.raise_for_status()
                    return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(f"❌ HTTP error calling {endpoint} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

            except Exception as e:
                logger.error(f"❌ Error calling {endpoint} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

        raise Exception(f"Failed to call {endpoint} after {max_retries} attempts")

