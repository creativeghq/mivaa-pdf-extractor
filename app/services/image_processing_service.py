"""
Image Processing Service - Handles image extraction, classification, upload, and CLIP generation.

This service encapsulates all image-related operations in the PDF processing pipeline:
1. Extract images from PDF
2. Classify images (material vs non-material) using Llama/Claude
3. Upload material images to Supabase Storage
4. Save images to database
5. Generate CLIP embeddings
"""

import os
import base64
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from asyncio import Semaphore
import logging

from app.services.supabase_client import get_supabase_client
from app.services.vecs_service import VecsService
from app.services.real_embeddings_service import RealEmbeddingsService
from app.services.pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)


class ImageProcessingService:
    """Service for handling all image processing operations."""
    
    def __init__(self):
        self.supabase_client = get_supabase_client()
        self.vecs_service = VecsService()
        self.embedding_service = RealEmbeddingsService()
        self.pdf_processor = PDFProcessor()
    
    async def classify_images(
        self,
        extracted_images: List[Dict[str, Any]],
        confidence_threshold: float = 0.7,
        primary_model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        validation_model: str = "claude-sonnet-4-20250514"
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Classify images as material or non-material using Llama Vision + Claude validation.

        Args:
            extracted_images: List of extracted image data
            confidence_threshold: Threshold for Claude validation (default: 0.7)
            primary_model: Primary classification model (default: Llama Vision)
            validation_model: Validation model for uncertain cases (default: Claude Sonnet)

        Returns:
            Tuple of (material_images, non_material_images)
        """
        logger.info(f"🤖 Starting AI-based image classification for {len(extracted_images)} images...")
        logger.info(f"   Strategy: {primary_model} (fast filter) → {validation_model} (validation for uncertain cases)")
        
        # Import AI services
        from app.services.ai_client_service import get_ai_client_service
        import httpx
        import json
        
        ai_service = get_ai_client_service()
        together_api_key = os.getenv('TOGETHER_API_KEY')
        
        async def classify_image_with_llama(image_path: str) -> Dict[str, Any]:
            """Fast classification using Llama Vision (TogetherAI)."""
            try:
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
                classification_prompt = """Analyze this image and classify it as:
1. MATERIAL: Shows building/interior materials (tiles, wood, fabric, stone, metal, flooring, wallpaper, etc.) - either close-up texture or in application
2. NOT_MATERIAL: Faces, logos, charts, diagrams, text, decorative graphics, abstract patterns

Respond ONLY with JSON:
{"is_material": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}"""
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "https://api.together.xyz/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {together_api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                            "messages": [{
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                                    {"type": "text", "text": classification_prompt}
                                ]
                            }],
                            "max_tokens": 512,
                            "temperature": 0.1
                        }
                    )
                    
                    result_text = response.json()['choices'][0]['message']['content']
                    result = json.loads(result_text)
                    
                    return {
                        'is_material': result.get('is_material', False),
                        'confidence': result.get('confidence', 0.5),
                        'reason': result.get('reason', 'Unknown'),
                        'model': 'llama'
                    }
            
            except Exception as e:
                logger.warning(f"⚠️ Llama classification failed for {image_path}: {e}")
                return {
                    'is_material': False,
                    'confidence': 0.0,
                    'reason': f'Llama failed: {str(e)}',
                    'model': 'llama_failed'
                }
        
        async def validate_with_claude(image_path: str) -> Dict[str, Any]:
            """Validate uncertain cases with Claude Sonnet."""
            try:
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
                classification_prompt = """Analyze this image and classify it into ONE of these categories:

1. **material_closeup**: Close-up photo showing material texture, surface, pattern, or finish (tiles, wood, fabric, stone, metal, etc.)
2. **material_in_situ**: Material shown in application/context (bathroom with tiles, furniture with fabric, room with flooring, etc.)
3. **non_material**: NOT material-related (faces, logos, decorative graphics, charts, diagrams, text, random images)

Respond ONLY with this JSON format:
{
    "classification": "material_closeup" | "material_in_situ" | "non_material",
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}"""
                
                response = await ai_service.anthropic_async.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                            {"type": "text", "text": classification_prompt}
                        ]
                    }]
                )

                result_text = response.content[0].text
                result = json.loads(result_text)

                is_material = result['classification'] in ['material_closeup', 'material_in_situ']

                return {
                    'is_material': is_material,
                    'confidence': result.get('confidence', 0.9),
                    'reason': result.get('reason', 'Claude validation'),
                    'classification': result['classification'],
                    'model': 'claude'
                }

            except Exception as e:
                logger.warning(f"⚠️ Claude validation failed for {image_path}: {e}")
                return {
                    'is_material': False,
                    'confidence': 0.0,
                    'reason': f'Claude failed: {str(e)}',
                    'model': 'claude_failed'
                }

        # Two-stage classification with semaphores for rate limiting
        llama_semaphore = Semaphore(5)  # 5 concurrent Llama requests
        claude_semaphore = Semaphore(2)  # 2 concurrent Claude requests

        async def classify_with_two_stage(img_data):
            image_path = img_data.get('path')
            if not image_path or not os.path.exists(image_path):
                return None

            # STAGE 1: Fast Llama classification
            async with llama_semaphore:
                llama_result = await classify_image_with_llama(image_path)

            # STAGE 2: If confidence is low (< threshold), validate with Claude
            if llama_result['confidence'] < confidence_threshold:
                logger.debug(f"   🔍 Low confidence ({llama_result['confidence']:.2f}) - validating with Claude: {img_data.get('filename')}")
                async with claude_semaphore:
                    claude_result = await validate_with_claude(image_path)
                img_data['ai_classification'] = claude_result
            else:
                img_data['ai_classification'] = llama_result

            return img_data

        # Classify all images
        classification_tasks = [classify_with_two_stage(img_data) for img_data in extracted_images]
        classified_images = await asyncio.gather(*classification_tasks, return_exceptions=True)

        # Filter into material and non-material
        material_images = []
        non_material_images = []

        for img_data in classified_images:
            if img_data is None or isinstance(img_data, Exception):
                logger.debug(f"   ⚠️ Skipping image due to classification failure")
                continue

            classification = img_data.get('ai_classification', {})
            if classification.get('is_material', False):
                material_images.append(img_data)
            else:
                non_material_images.append(img_data)
                logger.info(f"   🚫 Filtered out: {img_data.get('filename')} - {classification.get('classification')} ({classification.get('reason')})")

        logger.info(f"✅ AI classification complete:")
        logger.info(f"   Material images: {len(material_images)}")
        logger.info(f"   Non-material images filtered out: {len(non_material_images)}")

        total_classified = len(material_images) + len(non_material_images)
        if total_classified > 0:
            logger.info(f"   Classification accuracy: {len(material_images) / total_classified * 100:.1f}% kept")

        return material_images, non_material_images

    async def upload_images_to_storage(
        self,
        material_images: List[Dict[str, Any]],
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Upload material images to Supabase Storage.

        Args:
            material_images: List of material image data
            document_id: Document ID for storage path

        Returns:
            List of uploaded images with storage URLs
        """
        logger.info(f"📤 Uploading {len(material_images)} material images to Supabase Storage...")

        upload_semaphore = Semaphore(10)  # 10 concurrent uploads

        async def upload_single_image(img_data):
            """Upload a single material image to Supabase Storage"""
            try:
                image_path = img_data.get('path')
                if not image_path or not os.path.exists(image_path):
                    logger.warning(f"Image file not found: {image_path}")
                    return None

                # Upload to Supabase Storage
                upload_result = await self.pdf_processor._upload_image_to_storage(
                    image_path,
                    document_id,
                    {
                        'filename': img_data.get('filename'),
                        'size_bytes': img_data.get('size_bytes'),
                        'format': img_data.get('format'),
                        'dimensions': img_data.get('dimensions'),
                        'width': img_data.get('width'),
                        'height': img_data.get('height')
                    },
                    None  # No enhanced path
                )

                if upload_result:
                    img_data['storage_url'] = upload_result.get('storage_url')
                    img_data['storage_path'] = upload_result.get('storage_path')
                    return img_data
                else:
                    logger.warning(f"Failed to upload image: {img_data.get('filename')}")
                    return None

            except Exception as e:
                logger.error(f"Error uploading image {img_data.get('filename')}: {e}")
                return None

        async def upload_with_semaphore(img_data):
            async with upload_semaphore:
                return await upload_single_image(img_data)

        upload_tasks = [upload_with_semaphore(img_data) for img_data in material_images]
        uploaded_images = await asyncio.gather(*upload_tasks, return_exceptions=True)

        # Filter out failed uploads
        successful_uploads = [img for img in uploaded_images if img is not None and not isinstance(img, Exception)]

        logger.info(f"✅ Upload complete: {len(successful_uploads)} material images uploaded to storage")

        return successful_uploads

    async def _get_embedding_checkpoint(self, document_id: str) -> Optional[int]:
        """
        Get the last successfully processed image index for embedding generation.

        Args:
            document_id: Document ID

        Returns:
            Last processed index or None if no checkpoint exists
        """
        try:
            result = self.supabase_client.client.table('document_images')\
                .select('id')\
                .eq('document_id', document_id)\
                .not_.is_('visual_clip_embedding_512', 'null')\
                .execute()

            if result.data:
                return len(result.data)  # Number of images with embeddings
            return 0
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to get embedding checkpoint: {e}")
            return 0

    async def _process_single_image_with_retry(
        self,
        img_data: Dict[str, Any],
        document_id: str,
        workspace_id: str,
        idx: int,
        total: int,
        max_retries: int = 3
    ) -> Tuple[bool, bool, Optional[str]]:
        """
        Process a single image with retry logic.

        Args:
            img_data: Image data
            document_id: Document ID
            workspace_id: Workspace ID
            idx: Image index
            total: Total images
            max_retries: Maximum retry attempts

        Returns:
            Tuple of (image_saved, embedding_generated, error_message)
        """
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                # Save to database
                image_id = await self.supabase_client.save_single_image(
                    image_info=img_data,
                    document_id=document_id,
                    workspace_id=workspace_id,
                    image_index=idx
                )

                if not image_id:
                    last_error = "Failed to save image to database"
                    retry_count += 1
                    if retry_count < max_retries:
                        await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    continue

                img_data['id'] = image_id
                logger.info(f"   ✅ Saved image {idx + 1}/{total} to DB: {image_id}")

                # Generate CLIP embeddings
                image_path = img_data.get('path')
                if not image_path or not os.path.exists(image_path):
                    logger.warning(f"   ⚠️ Image file not found for CLIP generation: {image_path}")
                    return (True, False, "Image file not found")

                logger.info(f"   🎨 Generating CLIP embeddings for image {idx + 1}/{total}")

                # Read image and convert to base64
                with open(image_path, 'rb') as img_file:
                    image_bytes = img_file.read()
                    image_base64 = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

                # Generate all embeddings
                embedding_result = await self.embedding_service.generate_all_embeddings(
                    entity_id=image_id,
                    entity_type="image",
                    text_content="",
                    image_data=image_base64,
                    material_properties={}
                )

                if not embedding_result or not embedding_result.get('success'):
                    last_error = "Failed to generate CLIP embeddings"
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"   ⚠️ Retry {retry_count}/{max_retries} for image {image_id}")
                        await asyncio.sleep(2 ** retry_count)
                    continue

                embeddings = embedding_result.get('embeddings', {})

                # Save visual CLIP embedding
                visual_embedding = embeddings.get('visual_512')
                if visual_embedding:
                    # ✅ CRITICAL: Save to document_images table first
                    try:
                        self.supabase_client.client.table('document_images')\
                            .update({'visual_clip_embedding_512': visual_embedding})\
                            .eq('id', image_id)\
                            .execute()
                        logger.debug(f"   ✅ Saved visual CLIP to document_images table for {image_id}")
                    except Exception as db_error:
                        logger.error(f"   ❌ Failed to save visual CLIP to DB: {db_error}")
                        last_error = f"Failed to save visual CLIP: {db_error}"
                        retry_count += 1
                        if retry_count < max_retries:
                            await asyncio.sleep(2 ** retry_count)
                        continue

                    # Then save to VECS collection
                    await self.vecs_service.upsert_image_embedding(
                        image_id=image_id,
                        clip_embedding=visual_embedding,
                        metadata={
                            'document_id': document_id,
                            'page_number': img_data.get('page_number', 1),
                            'quality_score': img_data.get('quality_score', 0.5)
                        }
                    )

                # Save specialized embeddings
                specialized_embeddings = {}
                if embeddings.get('color_512'):
                    specialized_embeddings['color'] = embeddings.get('color_512')
                if embeddings.get('texture_512'):
                    specialized_embeddings['texture'] = embeddings.get('texture_512')
                if embeddings.get('application_512'):
                    specialized_embeddings['application'] = embeddings.get('application_512')
                if embeddings.get('material_512'):
                    specialized_embeddings['material'] = embeddings.get('material_512')

                if specialized_embeddings:
                    await self.vecs_service.upsert_specialized_embeddings(
                        image_id=image_id,
                        embeddings=specialized_embeddings,
                        metadata={
                            'document_id': document_id,
                            'page_number': img_data.get('page_number', 1)
                        }
                    )

                logger.info(f"   ✅ Generated {1 + len(specialized_embeddings)} CLIP embeddings for image {image_id}")
                return (True, True, None)

            except Exception as e:
                last_error = str(e)
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"   ⚠️ Retry {retry_count}/{max_retries} for image {idx + 1}: {e}")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"   ❌ Failed after {max_retries} retries for image {idx + 1}: {e}")

        return (False, False, last_error)

    async def save_images_and_generate_clips(
        self,
        material_images: List[Dict[str, Any]],
        document_id: str,
        workspace_id: str,
        batch_size: int = 20,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Save images to database and generate CLIP embeddings with batching and retry logic.

        This method implements:
        1. Batch processing (default: 20 images per batch)
        2. Retry logic with exponential backoff (up to 3 retries per image)
        3. Checkpoint recovery (resume from last successful batch)
        4. Detailed error tracking (log which images fail and why)

        Args:
            material_images: List of material image data
            document_id: Document ID
            workspace_id: Workspace ID
            batch_size: Number of images to process per batch (default: 20)
            max_retries: Maximum retry attempts per image (default: 3)

        Returns:
            Dict with counts and failed images: {
                images_saved,
                clip_embeddings_generated,
                failed_images: [{index, path, error}]
            }
        """
        logger.info(f"💾 Saving {len(material_images)} material images to database and generating CLIP embeddings...")
        logger.info(f"   📦 Batch size: {batch_size}, Max retries: {max_retries}")

        images_saved_count = 0
        clip_embeddings_count = 0
        failed_images = []

        # Check checkpoint - get number of images already processed
        checkpoint_index = await self._get_embedding_checkpoint(document_id)
        if checkpoint_index > 0:
            logger.info(f"   ⏭️ Resuming from checkpoint: {checkpoint_index} images already have embeddings")
            # Skip already processed images
            material_images = material_images[checkpoint_index:]
            if not material_images:
                logger.info(f"   ✅ All images already processed!")
                return {
                    'images_saved': checkpoint_index,
                    'clip_embeddings_generated': checkpoint_index,
                    'failed_images': []
                }

        # Process in batches
        total_images = len(material_images)
        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            batch = material_images[batch_start:batch_end]

            logger.info(f"   📦 Processing batch {batch_start // batch_size + 1}/{(total_images + batch_size - 1) // batch_size} ({batch_start + 1}-{batch_end}/{total_images})")

            # Process batch images with retry logic
            for idx, img_data in enumerate(batch):
                global_idx = batch_start + idx + checkpoint_index

                image_saved, embedding_generated, error = await self._process_single_image_with_retry(
                    img_data=img_data,
                    document_id=document_id,
                    workspace_id=workspace_id,
                    idx=global_idx,
                    total=total_images + checkpoint_index,
                    max_retries=max_retries
                )

                if image_saved:
                    images_saved_count += 1
                if embedding_generated:
                    clip_embeddings_count += 1

                if error:
                    failed_images.append({
                        'index': global_idx,
                        'path': img_data.get('path'),
                        'page_number': img_data.get('page_number'),
                        'error': error
                    })

            # Log batch completion
            logger.info(f"   ✅ Batch {batch_start // batch_size + 1} complete: {len(batch)} images processed")

        # Final summary
        logger.info(f"✅ Image processing complete:")
        logger.info(f"   Images saved to DB: {images_saved_count + checkpoint_index}")
        logger.info(f"   CLIP embeddings generated: {clip_embeddings_count + checkpoint_index}")

        if failed_images:
            logger.warning(f"   ⚠️ Failed images: {len(failed_images)}")
            for failed in failed_images[:5]:  # Log first 5 failures
                logger.warning(f"      - Image {failed['index']} (page {failed['page_number']}): {failed['error']}")
            if len(failed_images) > 5:
                logger.warning(f"      ... and {len(failed_images) - 5} more")

        return {
            'images_saved': images_saved_count + checkpoint_index,
            'clip_embeddings_generated': clip_embeddings_count + checkpoint_index,
            'failed_images': failed_images
        }


