"""
Pipeline Orchestrator - Coordinates modular PDF processing pipeline.

This orchestrator calls internal API endpoints for each pipeline stage,
providing clean separation, retry logic, and comprehensive progress tracking.

Integrates with existing infrastructure:
- ProgressTracker for job progress and heartbeat monitoring
- CheckpointRecoveryService for checkpoint creation
- JobTracker for database sync
- ResourceManager for cleanup
- Sentry for error tracking
"""

import logging
import asyncio
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the modular PDF processing pipeline.

    This orchestrator coordinates the complete pipeline by calling internal API endpoints
    for each stage, while properly managing:
    - Progress tracking and heartbeat monitoring
    - Checkpoint creation for recovery
    - Job queue and database sync
    - Resource cleanup
    - Error handling and logging
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    async def process_document(
        self,
        job_id: str,
        document_id: str,
        workspace_id: str,
        extracted_images: List[Dict[str, Any]],
        extracted_text: str,
        tracker: Any,  # ProgressTracker instance from main orchestrator
        job_storage: Dict[str, Any],  # Global job_storage dict
        product_ids: Optional[List[str]] = None,
        confidence_threshold: float = 0.7,
        similarity_threshold: float = 0.5,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        focused_extraction: bool = True,
        extract_categories: List[str] = None,
        ai_config: Optional[Dict[str, Any]] = None  # Dynamic AI model configuration
    ) -> Dict[str, Any]:
        """
        Orchestrate the modular PDF processing pipeline.

        This method coordinates the complete pipeline by calling internal API endpoints
        for each stage. It integrates with existing infrastructure:
        - Uses provided ProgressTracker for progress updates and heartbeat
        - Creates checkpoints via CheckpointRecoveryService
        - Syncs to job_storage dict for in-memory tracking
        - Handles errors with proper logging and Sentry integration

        Pipeline Stages:
        1. Classify Images (50-60%) - Two-stage AI classification (Qwen → Claude)
        2. Upload Images (60-65%) - Upload material images to Supabase Storage
        3. Save Images & Generate CLIP (65-75%) - DB save + 5 CLIP embeddings per image
        4. Create Chunks (75-85%) - Semantic chunking + text embeddings
        5. Create Relationships (85-100%) - Chunk-image and product-image relationships

        Args:
            job_id: Job ID for tracking
            document_id: Document ID
            workspace_id: Workspace ID
            extracted_images: List of extracted images from PDF
            extracted_text: Extracted text from PDF
            tracker: ProgressTracker instance (already initialized with heartbeat)
            job_storage: Global job_storage dict for in-memory tracking
            product_ids: Optional list of product IDs from discovery
            confidence_threshold: Minimum confidence for image classification (default: 0.7)
            similarity_threshold: Minimum similarity for relationships (default: 0.5)
            chunk_size: Size of text chunks (default: 512)
            chunk_overlap: Overlap between chunks (default: 50)
            focused_extraction: If True, only process material images (default: True)
            extract_categories: List of categories to extract (default: ['products'])
            ai_config: Optional AI model configuration dict with parameters:
                - visual_embedding_primary: Primary visual model (default: SigLIP)
                - visual_embedding_fallback: Fallback visual model (default: CLIP)
                - classification_primary_model: Primary classification model (default: Qwen)
                - classification_validation_model: Validation model (default: Claude)
                - classification_confidence_threshold: Confidence threshold (default: 0.7)
                - discovery_model: Product discovery model (default: Claude)
                - metadata_extraction_model: Metadata extraction model (default: Claude)
                - chunking_model: Chunking model (default: GPT-4o)
                - text_embedding_model: Text embedding model (default: text-embedding-3-small)

        Returns:
            Dict with complete pipeline results including:
            - material_images_count: Number of material images classified
            - images_uploaded: Number of images uploaded to storage
            - images_saved: Number of images saved to database
            - clip_embeddings_generated: Number of CLIP embeddings generated
            - chunks_created: Number of chunks created
            - text_embeddings_generated: Number of text embeddings generated
            - chunk_image_relationships: Number of chunk-image relationships
            - product_image_relationships: Number of product-image relationships
            - ai_models_used: Dict of AI models used at each stage
        """
        if extract_categories is None:
            extract_categories = ['products']

        logger.info("=" * 80)
        logger.info(f"🚀 [MODULAR PIPELINE] Starting orchestration")
        logger.info("=" * 80)
        logger.info(f"📋 Job ID: {job_id}")
        logger.info(f"📄 Document ID: {document_id}")
        logger.info(f"🖼️  Total Images Extracted: {len(extracted_images)}")
        logger.info(f"📝 Text Length: {len(extracted_text)} characters")
        logger.info(f"🎯 Focused Extraction: {'ENABLED' if focused_extraction else 'DISABLED'}")
        logger.info(f"📦 Extract Categories: {', '.join(extract_categories).upper()}")
        logger.info(f"🔧 Confidence Threshold: {confidence_threshold}")
        logger.info("=" * 80)

        # Import checkpoint service
        from app.services.checkpoint_recovery_service import checkpoint_recovery_service, ProcessingStage as CheckpointStage
        from app.schemas.jobs import ProcessingStage

        results = {
            'job_id': job_id,
            'document_id': document_id,
            'stages': {},
            'errors': [],
            'start_time': datetime.utcnow().isoformat(),
            'focused_extraction': focused_extraction,
            'extract_categories': extract_categories
        }

        try:
            # ========================================
            # STAGE 1: Classify Images (50-60%)
            # ========================================
            logger.info("🤖 [STAGE 1/5] Image Classification - Starting...")
            logger.info(f"   Total images to classify: {len(extracted_images)}")
            logger.info(f"   Confidence threshold: {confidence_threshold}")
            logger.info(f"   Method: Two-stage (Qwen Vision → Claude validation)")

            await tracker.update_stage(ProcessingStage.EXTRACTING_IMAGES, stage_name="image_classification")

            classify_payload = {
                "job_id": job_id,
                "extracted_images": extracted_images,
                "confidence_threshold": confidence_threshold
            }
            if ai_config:
                classify_payload["ai_config"] = ai_config

            classify_result = await self._call_internal_endpoint(
                endpoint=f"/api/internal/classify-images/{job_id}",
                data=classify_payload
            )

            if not classify_result.get('success'):
                raise Exception(f"Image classification failed: {classify_result}")

            results['stages']['classify'] = classify_result
            material_images = classify_result.get('material_images', [])
            non_material_images = classify_result.get('non_material_images', [])

            # Update job_storage
            job_storage[job_id]["progress"] = 60
            job_storage[job_id]["current_step"] = "Image classification complete"
            job_storage[job_id]["details"] = {
                'material_images': len(material_images),
                'non_material_images': len(non_material_images),
                'classification_accuracy': f"{len(material_images) / len(extracted_images) * 100:.1f}%" if extracted_images else "0%"
            }

            # Create checkpoint with vision-guided metadata
            await checkpoint_recovery_service.create_checkpoint(
                job_id=job_id,
                stage=CheckpointStage.IMAGES_EXTRACTED,
                data={
                    "document_id": document_id,
                    "material_images_count": len(material_images),
                    "non_material_images_count": len(non_material_images),
                    "total_images": len(extracted_images),
                    "vision_guided_count": vision_guided_count,
                    "ai_classified_count": ai_classified_count
                },
                metadata={
                    "confidence_threshold": confidence_threshold,
                    "focused_extraction": focused_extraction,
                    "extraction_method": "vision_guided" if vision_guided_count > 0 else "pymupdf"
                }
            )

            # Log vision-guided extraction statistics
            vision_guided_count = classify_result.get('vision_guided_count', 0)
            ai_classified_count = classify_result.get('ai_classified_count', 0)

            logger.info(f"✅ [STAGE 1/5] Classification Complete:")
            logger.info(f"   Material images: {len(material_images)}")
            logger.info(f"   Non-material images: {len(non_material_images)}")
            logger.info(f"   Classification accuracy: {len(material_images) / len(extracted_images) * 100:.1f}%")
            if vision_guided_count > 0:
                logger.info(f"   Vision-guided (pre-classified): {vision_guided_count}")
                logger.info(f"   AI-classified: {ai_classified_count}")

            # ========================================
            # STAGE 2: Upload Images (60-65%)
            # ========================================
            if material_images:
                logger.info("📤 [STAGE 2/5] Image Upload - Starting...")
                logger.info(f"   Uploading {len(material_images)} material images to Supabase Storage...")

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
                failed_uploads = upload_result.get('failed_count', 0)

                # Update job_storage
                job_storage[job_id]["progress"] = 65
                job_storage[job_id]["current_step"] = "Image upload complete"
                job_storage[job_id]["details"] = {
                    'uploaded_images': len(uploaded_images),
                    'failed_uploads': failed_uploads
                }

                logger.info(f"✅ [STAGE 2/5] Upload Complete:")
                logger.info(f"   Uploaded: {len(uploaded_images)} images")
                logger.info(f"   Failed: {failed_uploads} images")

                # ========================================
                # STAGE 3: Save Images & Generate CLIP (65-75%)
                # ========================================
                logger.info("💾 [STAGE 3/5] Save Images & Generate CLIP - Starting...")
                logger.info(f"   Saving {len(uploaded_images)} images to database...")
                logger.info(f"   Generating 5 CLIP embeddings per image (visual, color, texture, application, material)...")

                save_payload = {
                    "job_id": job_id,
                    "material_images": uploaded_images,
                    "document_id": document_id,
                    "workspace_id": workspace_id
                }
                if ai_config:
                    save_payload["ai_config"] = ai_config

                save_result = await self._call_internal_endpoint(
                    endpoint=f"/api/internal/save-images-db/{job_id}",
                    data=save_payload
                )

                if not save_result.get('success'):
                    raise Exception(f"Image save and CLIP generation failed: {save_result}")

                results['stages']['save_and_clip'] = save_result
                images_saved = save_result.get('images_saved', 0)
                clip_embeddings = save_result.get('clip_embeddings_generated', 0)

                # Update tracker
                tracker.images_stored = images_saved
                await tracker.update_database_stats(
                    images_stored=images_saved,
                    sync_to_db=True
                )

                # Update job_storage
                job_storage[job_id]["progress"] = 75
                job_storage[job_id]["current_step"] = "CLIP embeddings generated"
                job_storage[job_id]["details"] = {
                    'images_saved': images_saved,
                    'clip_embeddings_generated': clip_embeddings,
                    'embeddings_per_image': 5
                }

                # Create checkpoint with vision-guided metadata
                await checkpoint_recovery_service.create_checkpoint(
                    job_id=job_id,
                    stage=CheckpointStage.IMAGE_EMBEDDINGS_GENERATED,
                    data={
                        "document_id": document_id,
                        "images_saved": images_saved,
                        "clip_embeddings_generated": clip_embeddings,
                        "vision_guided_count": vision_guided_count,
                        "pymupdf_fallback_count": pymupdf_fallback_count
                    },
                    metadata={
                        "embeddings_per_image": 5,
                        "embedding_types": ["visual", "color", "texture", "application", "material"],
                        "average_vision_confidence": avg_vision_confidence,
                        "extraction_method": "vision_guided" if vision_guided_count > 0 else "pymupdf"
                    }
                )

                # Log vision-guided extraction statistics
                vision_guided_count = save_result.get('vision_guided_count', 0)
                pymupdf_fallback_count = save_result.get('pymupdf_fallback_count', 0)
                avg_vision_confidence = save_result.get('average_vision_confidence')

                logger.info(f"✅ [STAGE 3/5] Save & CLIP Complete:")
                logger.info(f"   Images saved: {images_saved}")
                logger.info(f"   CLIP embeddings: {clip_embeddings} (5 per image)")
                if vision_guided_count > 0:
                    logger.info(f"   Vision-guided: {vision_guided_count}, PyMuPDF fallback: {pymupdf_fallback_count}")
                    if avg_vision_confidence:
                        logger.info(f"   Average vision confidence: {avg_vision_confidence:.2f}")

            else:
                logger.info("⚠️ [STAGES 2-3] No material images to upload/save, skipping...")
                job_storage[job_id]["progress"] = 75
                job_storage[job_id]["current_step"] = "No material images found"

            # ========================================
            # STAGE 4: Create Chunks (75-85%)
            # ========================================
            logger.info("📝 [STAGE 4/5] Chunking & Text Embeddings - Starting...")
            logger.info(f"   Text length: {len(extracted_text)} characters")
            logger.info(f"   Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
            logger.info(f"   Products: {len(product_ids) if product_ids else 0}")

            await tracker.update_stage(ProcessingStage.CHUNKING, stage_name="chunking")

            chunks_payload = {
                "job_id": job_id,
                "document_id": document_id,
                "workspace_id": workspace_id,
                "extracted_text": extracted_text,
                "product_ids": product_ids,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
            if ai_config:
                chunks_payload["ai_config"] = ai_config

            chunks_result = await self._call_internal_endpoint(
                endpoint=f"/api/internal/create-chunks/{job_id}",
                data=chunks_payload
            )

            if not chunks_result.get('success'):
                raise Exception(f"Chunking failed: {chunks_result}")

            results['stages']['chunks'] = chunks_result
            chunks_created = chunks_result.get('chunks_created', 0)
            text_embeddings = chunks_result.get('embeddings_generated', 0)
            chunk_product_relationships = chunks_result.get('relationships_created', 0)

            # Update tracker
            tracker.chunks_created = chunks_created
            await tracker.update_database_stats(
                chunks_created=chunks_created,
                sync_to_db=True
            )

            # Update job_storage
            job_storage[job_id]["progress"] = 85
            job_storage[job_id]["current_step"] = "Chunking complete"
            job_storage[job_id]["details"] = {
                'chunks_created': chunks_created,
                'text_embeddings_generated': text_embeddings,
                'chunk_product_relationships': chunk_product_relationships
            }

            # Create checkpoint
            await checkpoint_recovery_service.create_checkpoint(
                job_id=job_id,
                stage=CheckpointStage.CHUNKS_CREATED,
                data={
                    "document_id": document_id,
                    "chunks_created": chunks_created,
                    "text_embeddings_generated": text_embeddings,
                    "chunk_product_relationships": chunk_product_relationships
                },
                metadata={
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
            )

            logger.info(f"✅ [STAGE 4/5] Chunking Complete:")
            logger.info(f"   Chunks created: {chunks_created}")
            logger.info(f"   Text embeddings: {text_embeddings}")
            logger.info(f"   Chunk-product relationships: {chunk_product_relationships}")

            # ========================================
            # STAGE 5: Create Relationships (85-100%)
            # ========================================
            if product_ids and material_images:
                logger.info("🔗 [STAGE 5/5] Relationships - Starting...")
                logger.info(f"   Products: {len(product_ids)}")
                logger.info(f"   Similarity threshold: {similarity_threshold}")
                logger.info(f"   Creating chunk-image and product-image relationships...")

                await tracker.update_stage(ProcessingStage.GENERATING_EMBEDDINGS, stage_name="relationships")

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
                chunk_image_rels = relationships_result.get('chunk_image_relationships', 0)
                product_image_rels = relationships_result.get('product_image_relationships', 0)

                # Update job_storage
                job_storage[job_id]["progress"] = 100
                job_storage[job_id]["current_step"] = "Relationships created"
                job_storage[job_id]["details"] = {
                    'chunk_image_relationships': chunk_image_rels,
                    'product_image_relationships': product_image_rels
                }

                # Create RELATIONSHIPS_CREATED checkpoint
                await checkpoint_recovery_service.create_checkpoint(
                    job_id=job_id,
                    stage=CheckpointStage.RELATIONSHIPS_CREATED,
                    data={
                        "document_id": document_id,
                        "chunk_image_relationships": chunk_image_rels,
                        "product_image_relationships": product_image_rels
                    },
                    metadata={
                        "similarity_threshold": similarity_threshold,
                        "relationships": {
                            "chunk_product": results['stages'].get('chunks', {}).get('relationships_created', 0),
                            "product_image": product_image_rels,
                            "chunk_image": chunk_image_rels,
                            "product_document_entities": results['stages'].get('products', {}).get('entity_product_relationships', 0),
                            "total_relationships": (
                                results['stages'].get('chunks', {}).get('relationships_created', 0) +
                                product_image_rels +
                                chunk_image_rels +
                                results['stages'].get('products', {}).get('entity_product_relationships', 0)
                            )
                        }
                    }
                )

                logger.info(f"✅ [STAGE 5/5] Relationships Complete:")
                logger.info(f"   Chunk-image relationships: {chunk_image_rels}")
                logger.info(f"   Product-image relationships: {product_image_rels}")

            else:
                logger.info("⚠️ [STAGE 5] No products or images, skipping relationships...")
                job_storage[job_id]["progress"] = 100
                job_storage[job_id]["current_step"] = "No relationships needed"

            # ========================================
            # PIPELINE COMPLETE
            # ========================================
            results['end_time'] = datetime.utcnow().isoformat()
            results['success'] = True

            # Update tracker to COMPLETED
            await tracker.update_stage(ProcessingStage.COMPLETED, stage_name="completed")

            # Final job_storage update
            job_storage[job_id]["status"] = "completed"
            job_storage[job_id]["progress"] = 100
            job_storage[job_id]["completed_at"] = datetime.utcnow().isoformat()

            # Calculate summary metrics
            processing_time = (datetime.utcnow() - datetime.fromisoformat(results['start_time'])).total_seconds()

            results['summary'] = {
                'material_images': len(material_images) if material_images else 0,
                'images_saved': results['stages'].get('save_and_clip', {}).get('images_saved', 0),
                'clip_embeddings': results['stages'].get('save_and_clip', {}).get('clip_embeddings_generated', 0),
                'chunks_created': chunks_created,
                'text_embeddings': text_embeddings,
                'chunk_image_relationships': results['stages'].get('relationships', {}).get('chunk_image_relationships', 0),
                'product_image_relationships': results['stages'].get('relationships', {}).get('product_image_relationships', 0),
                'processing_time_seconds': processing_time
            }

            logger.info("=" * 80)
            logger.info(f"🎉 [MODULAR PIPELINE] COMPLETE")
            logger.info("=" * 80)
            logger.info(f"📊 Summary:")
            logger.info(f"   Material images: {results['summary']['material_images']}")
            logger.info(f"   Images saved: {results['summary']['images_saved']}")
            logger.info(f"   CLIP embeddings: {results['summary']['clip_embeddings']}")
            logger.info(f"   Chunks created: {results['summary']['chunks_created']}")
            logger.info(f"   Text embeddings: {results['summary']['text_embeddings']}")
            logger.info(f"   Chunk-image relationships: {results['summary']['chunk_image_relationships']}")
            logger.info(f"   Product-image relationships: {results['summary']['product_image_relationships']}")
            logger.info(f"   Processing time: {processing_time:.1f}s")
            logger.info("=" * 80)

            return results

        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"❌ [MODULAR PIPELINE] FAILED")
            logger.error("=" * 80)
            logger.error(f"Error: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")

            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")

            results['errors'].append(str(e))
            results['success'] = False
            results['end_time'] = datetime.utcnow().isoformat()

            # Update tracker to FAILED
            await tracker.fail_job(e)

            # Update job_storage
            job_storage[job_id]["status"] = "failed"
            job_storage[job_id]["error"] = str(e)
            job_storage[job_id]["failed_at"] = datetime.utcnow().isoformat()

            # Create FAILED checkpoint
            await checkpoint_recovery_service.create_checkpoint(
                job_id=job_id,
                stage=CheckpointStage.INITIALIZED,  # Use INITIALIZED as fallback
                data={
                    "document_id": document_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                metadata={
                    "failed_at": datetime.utcnow().isoformat(),
                    "stages_completed": list(results['stages'].keys())
                }
            )

            logger.error("=" * 80)

            # Send to Sentry if available
            try:
                import sentry_sdk
                sentry_sdk.capture_exception(e)
            except:
                pass

            raise

    async def _call_internal_endpoint(
        self,
        endpoint: str,
        data: Dict[str, Any],
        max_retries: int = 3,
        timeout: float = 600.0  # 10 minutes default timeout
    ) -> Dict[str, Any]:
        """
        Call an internal API endpoint with retry logic and exponential backoff.

        This method handles:
        - HTTP requests to internal endpoints
        - Retry logic with exponential backoff
        - Timeout handling
        - Error logging

        Args:
            endpoint: API endpoint path (e.g., "/api/internal/classify-images/job123")
            data: Request data (JSON payload)
            max_retries: Maximum number of retries (default: 3)
            timeout: Request timeout in seconds (default: 600s = 10 minutes)

        Returns:
            Response data as dict

        Raises:
            Exception: If all retries fail or timeout occurs
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(max_retries):
            try:
                logger.info(f"🔌 Calling {endpoint} (attempt {attempt + 1}/{max_retries})...")

                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, json=data)
                    response.raise_for_status()
                    result = response.json()

                    logger.info(f"✅ {endpoint} completed successfully")
                    return result

            except httpx.HTTPStatusError as e:
                logger.error(f"❌ HTTP error calling {endpoint} (attempt {attempt + 1}/{max_retries}):")
                logger.error(f"   Status code: {e.response.status_code}")
                logger.error(f"   Response: {e.response.text[:500]}")

                if attempt == max_retries - 1:
                    logger.error(f"❌ All {max_retries} attempts failed for {endpoint}")
                    raise

                backoff_seconds = 2 ** attempt
                logger.warning(f"⏳ Retrying in {backoff_seconds}s...")
                await asyncio.sleep(backoff_seconds)

            except httpx.TimeoutException as e:
                logger.error(f"⏱️ Timeout calling {endpoint} (attempt {attempt + 1}/{max_retries}):")
                logger.error(f"   Timeout: {timeout}s")

                if attempt == max_retries - 1:
                    logger.error(f"❌ All {max_retries} attempts timed out for {endpoint}")
                    raise

                backoff_seconds = 2 ** attempt
                logger.warning(f"⏳ Retrying in {backoff_seconds}s...")
                await asyncio.sleep(backoff_seconds)

            except Exception as e:
                logger.error(f"❌ Unexpected error calling {endpoint} (attempt {attempt + 1}/{max_retries}):")
                logger.error(f"   Error: {str(e)}")
                logger.error(f"   Error type: {type(e).__name__}")

                if attempt == max_retries - 1:
                    logger.error(f"❌ All {max_retries} attempts failed for {endpoint}")
                    raise

                backoff_seconds = 2 ** attempt
                logger.warning(f"⏳ Retrying in {backoff_seconds}s...")
                await asyncio.sleep(backoff_seconds)

        raise Exception(f"Failed to call {endpoint} after {max_retries} attempts")


