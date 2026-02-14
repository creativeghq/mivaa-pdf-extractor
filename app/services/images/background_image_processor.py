"""
Background Image Processor Service

Processes deferred AI analysis for images that were uploaded during PDF processing.
This service runs asynchronously to avoid blocking the main PDF processing flow.

Features:
- CLIP visual embeddings (512D)
- Qwen3-VL 17B Vision analysis
- Claude Sonnet 4.5 Vision validation
- Understanding embeddings (1024D) - Qwen vision_analysis → Voyage AI
- Color embeddings (256D)
- Texture embeddings (256D)
- Application embeddings (512D)
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BackgroundImageProcessor:
    """
    Service to process deferred image AI analysis in background.
    """
    
    def __init__(self, supabase_client):
        """
        Initialize background image processor.
        
        Args:
            supabase_client: Supabase client for database operations
        """
        self.supabase = supabase_client
        self.logger = logger
        
    async def process_pending_images(
        self,
        document_id: str,
        batch_size: int = 20,
        max_concurrent: int = 8
    ) -> Dict[str, Any]:
        """
        ⚡ OPTIMIZED: Process all images needing AI analysis for a document.

        Default max_concurrent increased from 3 to 8 for better throughput.
        Vision APIs can handle higher concurrency without rate limiting issues.

        Args:
            document_id: Document ID to process images for
            batch_size: Number of images to process in each batch (default: 20)
            max_concurrent: Maximum concurrent image processing tasks (default: 8)

        Returns:
            Dictionary with processing statistics
        """
        try:
            self.logger.info(f"🖼️ Starting background image processing for document {document_id}")
            
            # Query images needing analysis
            images_response = self.supabase.client.table('document_images').select('*').eq(
                'document_id', document_id
            ).is_('vision_analysis', 'null').limit(batch_size).execute()
            
            pending_images = images_response.data or []
            
            if not pending_images:
                self.logger.info(f"✅ No pending images for document {document_id}")
                return {
                    "success": True,
                    "images_processed": 0,
                    "message": "No pending images"
                }
            
            self.logger.info(f"📋 Found {len(pending_images)} images needing AI analysis")
            
            # Process images in batches with concurrency control
            processed_count = 0
            failed_count = 0
            
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(image):
                async with semaphore:
                    return await self._process_single_image(image)
            
            # Process all images concurrently (with semaphore limiting)
            tasks = [process_with_semaphore(image) for image in pending_images]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes and failures
            for result in results:
                if isinstance(result, Exception):
                    failed_count += 1
                    self.logger.error(f"Image processing failed: {result}")
                elif result and result.get('success'):
                    processed_count += 1
                else:
                    failed_count += 1
            
            self.logger.info(f"✅ Background image processing complete: {processed_count} processed, {failed_count} failed")
            
            return {
                "success": True,
                "images_processed": processed_count,
                "images_failed": failed_count,
                "total_images": len(pending_images),
                "message": f"Processed {processed_count}/{len(pending_images)} images"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Background image processing failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "images_processed": 0
            }
    
    async def _process_single_image(self, image: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single image with all AI analysis.
        
        Args:
            image: Image record from database
            
        Returns:
            Processing result dictionary
        """
        try:
            image_id = image['id']
            image_url = image.get('image_url')
            
            self.logger.info(f"🔍 Processing image {image_id}")
            
            # Import services
            from .real_image_analysis_service import RealImageAnalysisService
            from .real_embeddings_service import RealEmbeddingsService
            
            # Initialize services
            analysis_service = RealImageAnalysisService(self.supabase)
            embeddings_service = RealEmbeddingsService(self.supabase)
            
            # Run AI analysis
            analysis_result = await analysis_service.analyze_image(
                image_id=image_id,
                image_url=image_url,
                context={
                    "document_id": image.get('document_id'),
                    "page_number": image.get('page_number')
                }
            )
            
            if not analysis_result.get('success'):
                self.logger.warning(f"⚠️ Image analysis failed for {image_id}: {analysis_result.get('error')}")
                return {"success": False, "error": analysis_result.get('error')}
            
            # Generate specialized embeddings (color, texture, application)
            embeddings_result = await embeddings_service.generate_material_embeddings(
                image_url=image_url,
                material_properties=analysis_result.get('vision_analysis', {})
            )

            # Update database with analysis results (all embeddings saved to document_images)
            update_data = {
                "vision_analysis": analysis_result.get('vision_analysis'),
                "claude_validation": analysis_result.get('claude_validation'),
                "material_properties": analysis_result.get('material_properties'),  # ✅ Save extracted material properties
                "quality_score": analysis_result.get('quality_score'),              # ✅ Save quality score
                "confidence_score": analysis_result.get('confidence_score'),        # ✅ Save confidence score
                "processing_status": "completed",
                "updated_at": datetime.utcnow().isoformat()
            }

            # Add CLIP embedding to document_images (512D)
            clip_embedding = analysis_result.get('clip_embedding')
            if clip_embedding:
                update_data["visual_clip_embedding_512"] = clip_embedding
                self.logger.debug(f"✅ Adding CLIP embedding (512D) to document_images for {image_id}")

            # Add specialized embeddings if available
            if embeddings_result and embeddings_result.get('success'):
                if embeddings_result.get('color_embedding'):
                    update_data["color_embedding_256"] = embeddings_result['color_embedding']
                    self.logger.debug(f"✅ Adding color embedding (256D) to document_images for {image_id}")
                if embeddings_result.get('texture_embedding'):
                    update_data["texture_embedding_256"] = embeddings_result['texture_embedding']
                    self.logger.debug(f"✅ Adding texture embedding (256D) to document_images for {image_id}")
                if embeddings_result.get('application_embedding'):
                    update_data["application_embedding_512"] = embeddings_result['application_embedding']
                    self.logger.debug(f"✅ Adding application embedding (512D) to document_images for {image_id}")

            # Update image record with all data
            self.supabase.client.table('document_images').update(update_data).eq('id', image_id).execute()

            # Generate understanding embedding from vision analysis (Qwen → Voyage AI 1024D)
            has_understanding = False
            vision_analysis_data = analysis_result.get('vision_analysis')
            if vision_analysis_data:
                try:
                    understanding_embedding = await embeddings_service.generate_understanding_embedding(
                        vision_analysis=vision_analysis_data,
                        material_properties=analysis_result.get('material_properties')
                    )
                    if understanding_embedding:
                        from app.services.embeddings.vecs_service import get_vecs_service
                        vecs = get_vecs_service()
                        await vecs.upsert_understanding_embedding(
                            image_id=image_id,
                            embedding=understanding_embedding,
                            metadata={
                                'document_id': image.get('document_id'),
                                'workspace_id': image.get('workspace_id'),
                                'page_number': image.get('page_number', 1)
                            }
                        )
                        has_understanding = True
                        self.logger.info(f"✅ Understanding embedding generated and stored for {image_id}")
                except Exception as understanding_error:
                    self.logger.warning(f"⚠️ Understanding embedding failed for {image_id} (non-critical): {understanding_error}")

            self.logger.info(f"✅ Image {image_id} processed successfully")

            return {
                "success": True,
                "image_id": image_id,
                "has_vision": bool(analysis_result.get('vision_analysis')),
                "has_claude": bool(analysis_result.get('claude_validation')),
                "has_clip": bool(analysis_result.get('clip_embedding')),
                "has_understanding": has_understanding,
                "has_color": bool(embeddings_result.get('color_embedding')),
                "has_texture": bool(embeddings_result.get('texture_embedding')),
                "has_application": bool(embeddings_result.get('application_embedding'))
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process image {image.get('id')}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "image_id": image.get('id')
            }
    
    async def process_all_pending_images_for_document(
        self,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Process ALL pending images for a document (not just one batch).
        Continues processing until no more pending images remain.
        
        Args:
            document_id: Document ID to process images for
            
        Returns:
            Dictionary with total processing statistics
        """
        total_processed = 0
        total_failed = 0
        batch_count = 0
        
        while True:
            batch_count += 1
            self.logger.info(f"📦 Processing batch {batch_count} for document {document_id}")
            
            result = await self.process_pending_images(
                document_id=document_id,
                batch_size=10,
                max_concurrent=3
            )
            
            if not result.get('success'):
                break
            
            processed = result.get('images_processed', 0)
            failed = result.get('images_failed', 0)
            
            total_processed += processed
            total_failed += failed
            
            # If no images were processed, we're done
            if processed == 0:
                break
            
            # Small delay between batches to avoid overwhelming the system
            await asyncio.sleep(1)
        
        self.logger.info(f"✅ All batches complete: {total_processed} processed, {total_failed} failed across {batch_count} batches")
        
        return {
            "success": True,
            "total_processed": total_processed,
            "total_failed": total_failed,
            "batches_processed": batch_count,
            "message": f"Processed {total_processed} images across {batch_count} batches"
        }


# Convenience function to start background processing
async def start_background_image_processing(document_id: str, supabase_client):
    """
    Start background image processing for a document.
    
    Args:
        document_id: Document ID to process images for
        supabase_client: Supabase client instance
    """
    processor = BackgroundImageProcessor(supabase_client)
    return await processor.process_all_pending_images_for_document(document_id)


