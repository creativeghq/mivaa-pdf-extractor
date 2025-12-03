"""
Stage 3: Image Processing

Consolidated image processing pipeline that performs:
1. Image extraction from PDF
2. AI classification (material vs non-material)
3. Upload to Supabase Storage
4. Save to database
5. CLIP embeddings generation (SigLIP primary, CLIP fallback)
6. Llama Vision analysis for quality scoring

All operations are performed in batches with memory monitoring and cleanup.
"""

import logging
import os
import base64
import gc
from typing import Dict, List, Any, Optional, Set

from app.services.supabase_client import get_supabase_client
from app.services.vecs_service import VecsService
from app.services.real_embeddings_service import RealEmbeddingsService
from app.services.pdf_processor import PDFProcessor
from app.utils.timeout_guard import with_timeout, TimeoutConstants, ProgressiveTimeoutStrategy
from app.utils.circuit_breaker import CircuitBreaker, CircuitBreakerError
from app.utils.memory_monitor import global_memory_monitor as memory_monitor
from app.services.checkpoint_recovery_service import ProcessingStage as CheckpointStage

logger = logging.getLogger(__name__)


def _determine_image_category(page_number: int, catalog: Any) -> str:
    """
    Determine category for an image based on its page number and catalog.

    Args:
        page_number: Page number of the image
        catalog: Product catalog with page classifications

    Returns:
        Category string: 'product', 'certificate', 'logo', 'specification', or 'general'
    """
    if not catalog:
        return 'general'

    try:
        # Check if page is in product pages
        if hasattr(catalog, 'products'):
            for product in catalog.products:
                if hasattr(product, 'page_range') and page_number in product.page_range:
                    return 'product'

        # Check if page is in certificate pages
        if hasattr(catalog, 'certificates'):
            for cert in catalog.certificates:
                if hasattr(cert, 'page_range') and page_number in cert.page_range:
                    return 'certificate'

        # Check if page is in logo pages
        if hasattr(catalog, 'logos'):
            for logo in catalog.logos:
                if hasattr(logo, 'page_range') and page_number in logo.page_range:
                    return 'logo'

        # Check if page is in specification pages
        if hasattr(catalog, 'specifications'):
            for spec in catalog.specifications:
                if hasattr(spec, 'page_range') and page_number in spec.page_range:
                    return 'specification'

        return 'general'

    except Exception:
        return 'general'


async def process_stage_3_images(
    file_content: bytes,
    document_id: str,
    workspace_id: str,
    job_id: str,
    page_count: int,
    product_pages: Set[int],
    focused_extraction: bool,
    extract_categories: List[str],
    image_analysis_model: str,
    component_manager: Any,
    loaded_components: List[str],
    tracker: Any,
    checkpoint_recovery_service: Any,
    logger: Any,
    pdf_result_with_images: Any = None,
    catalog: Any = None
) -> Dict[str, Any]:
    """
    Stage 3: Consolidated Image Processing
    
    Performs all image operations in a single batch-processed flow:
    - Extract images from PDF (with optional page filtering)
    - AI classification (Llama/Claude)
    - Upload to Supabase Storage
    - Save to database
    - Generate CLIP embeddings (SigLIP/CLIP)
    - Llama Vision analysis for quality scoring
    
    Args:
        file_content: PDF file bytes
        document_id: Document UUID
        workspace_id: Workspace UUID
        job_id: Processing job UUID
        page_count: Total pages in PDF
        product_pages: Set of page numbers containing products
        focused_extraction: Whether to extract only from specific pages
        extract_categories: Categories to extract (e.g., ['products'])
        component_manager: Lazy loading component manager
        loaded_components: List to track loaded components
        tracker: Progress tracker
        checkpoint_recovery_service: Checkpoint service
        resource_manager: Resource cleanup manager
        pdf_result_with_images: Optional pre-extracted PDF result
        
    Returns:
        Dict with processing results and statistics
    """
    logger.info("🖼️ [STAGE 3] Image Processing - Starting...")
    await tracker.update_stage(CheckpointStage.IMAGES_EXTRACTED, stage_name="image_processing")
    
    # Initialize services
    supabase_client = get_supabase_client()
    vecs_service = VecsService()
    embedding_service = RealEmbeddingsService()
    pdf_processor = PDFProcessor()
    
    # Load LlamaIndex service for Llama Vision analysis
    logger.info("📦 Loading LlamaIndex service for image analysis...")
    try:
        llamaindex_service = await component_manager.load("llamaindex_service")
        loaded_components.append("llamaindex_service")
        logger.info("✅ LlamaIndex service loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load LlamaIndex service: {e}")
        raise
    
    # Counters for tracking
    images_saved_count = 0
    clip_embeddings_generated = 0
    specialized_embeddings_generated = 0
    images_processed = 0  # Images analyzed with Llama Vision
    vecs_batch_records = []
    VECS_BATCH_SIZE = 20 
    
    # Circuit breakers for API calls
    # CLIP: More lenient settings for transient memory pressure failures
    clip_breaker = CircuitBreaker(failure_threshold=8, timeout_seconds=45, half_open_max_calls=5, name="CLIP")
    llama_breaker = CircuitBreaker(failure_threshold=5, timeout_seconds=60, name="Llama")
    
    # Dynamic batch size calculation - MORE CONSERVATIVE for CLIP/SigLIP models
    # CLIP/SigLIP models use ~2-3GB RAM when loaded, so we need smaller batches
    mem_stats = memory_monitor.get_memory_stats()
    total_memory_gb = mem_stats.total_mb / 1024

    # Adjust batch size based on total system memory
    # CRITICAL FIX: Reduced batch sizes for 16GB systems to prevent OOM crashes
    # SigLIP model uses 2-3GB RAM, so we need much smaller batches
    if total_memory_gb < 10:  # Low memory systems (< 10GB)
        DEFAULT_BATCH_SIZE = 5
        MAX_BATCH_SIZE = 8
    elif total_memory_gb < 16:  # Medium memory systems (10-16GB)
        DEFAULT_BATCH_SIZE = 8   # Reduced from 20 to prevent 8GB+ memory spikes
        MAX_BATCH_SIZE = 12      # Reduced from 30 to prevent OOM crashes
    else:  # High memory systems (> 16GB)
        DEFAULT_BATCH_SIZE = 30
        MAX_BATCH_SIZE = 40

    BATCH_SIZE = memory_monitor.calculate_optimal_batch_size(
        default_batch_size=DEFAULT_BATCH_SIZE,
        min_batch_size=3,
        max_batch_size=MAX_BATCH_SIZE,
        memory_per_item_mb=50.0  # Increased estimate: 50MB per image (CLIP model overhead)
    )

    # CRITICAL FIX: Reduced concurrency for 16GB systems to prevent memory exhaustion
    # Processing 6 images concurrently with SigLIP model caused 8GB memory spikes
    if mem_stats.percent_used < 40:
        CONCURRENT_IMAGES = 3  # Reduced from 6 to prevent memory spikes
    elif mem_stats.percent_used < 60:
        CONCURRENT_IMAGES = 2  # Reduced from 4 for safer processing
    else:
        CONCURRENT_IMAGES = 1  # Reduced from 2 when memory is already high
    
    logger.info(f"   🔧 DYNAMIC BATCH PROCESSING: {BATCH_SIZE} images per batch")
    logger.info(f"   🚀 Concurrency level: {CONCURRENT_IMAGES} images (memory: {mem_stats.percent_used:.1f}%)")
    memory_monitor.log_memory_stats(prefix="   ")

    # Step 1: Extract images from PDF (if not already provided)
    if pdf_result_with_images is None:
        logger.info("🔄 Starting image extraction from PDF...")
        logger.info(f"   PDF size: {len(file_content)} bytes")
        logger.info(f"   Focused extraction: {focused_extraction}")
        logger.info(f"   Extract categories: {extract_categories}")

        # Calculate estimated images for timeout
        if focused_extraction and 'products' in extract_categories and product_pages:
            estimated_images = len(product_pages) * 2
            logger.info(f"   🎯 Focused extraction: Only extracting from {len(product_pages)} product pages")
            logger.info(f"   📄 Product pages: {sorted(product_pages)}")
        else:
            estimated_images = page_count * 2
            logger.info(f"   📄 Full extraction: Extracting from all {page_count} pages")

        # Progressive timeout calculation
        image_extraction_timeout = ProgressiveTimeoutStrategy.calculate_image_processing_timeout(
            image_count=estimated_images,
            concurrent_limit=1
        )
        logger.info(f"📊 Image extraction: ~{estimated_images} estimated images → timeout: {image_extraction_timeout:.0f}s")

        # Build processing options
        processing_options = {
            'extract_images': True,
            'extract_tables': False
            # NOTE: Images are now ALWAYS uploaded to Supabase immediately
            # Non-material images are deleted from Supabase after AI classification
        }

        # Add page_list for focused extraction
        if focused_extraction and 'products' in extract_categories and product_pages:
            processing_options['page_list'] = sorted(list(product_pages))
            logger.info(f"   ✅ Passing page_list to PyMuPDF: {len(processing_options['page_list'])} pages")

        # Extract images with timeout
        try:
            pdf_result_with_images = await with_timeout(
                pdf_processor.process_pdf_from_bytes(
                    pdf_bytes=file_content,
                    document_id=document_id,
                    processing_options=processing_options
                ),
                timeout_seconds=image_extraction_timeout,
                operation_name="Image extraction from PDF"
            )
            logger.info(f"✅ Image extraction complete: {len(pdf_result_with_images.extracted_images)} images extracted")
        except Exception as e:
            logger.error(f"❌ Image extraction failed: {e}")
            raise
    else:
        logger.info(f"✅ Using pre-extracted images: {len(pdf_result_with_images.extracted_images)} images")

    # Get extracted images
    all_images = pdf_result_with_images.extracted_images
    if not all_images:
        logger.warning("⚠️ No images extracted from PDF")
        return {
            "status": "completed",
            "pdf_result_with_images": pdf_result_with_images,
            "material_images": [],
            "images_extracted": 0,
            "images_processed": 0,
            "clip_embeddings_generated": 0,
            "clip_embeddings_expected": 0,
            "clip_completion_rate": 0,
            "specialized_embeddings": 0,
            "images_analyzed": 0,
            "total_images_extracted": 0,
            "non_material_images": 0,
            "quality_flags": {
                "clip_embeddings_complete": True,  # No images = nothing to fail
                "all_images_analyzed": True,
                "specialized_embeddings_complete": True
            }
        }

    logger.info(f"📊 Total images to process: {len(all_images)}")

    # Step 2: AI Classification (Llama/Claude) to filter material images
    logger.info("🤖 Starting AI classification to identify material images...")

    import asyncio
    from asyncio import Semaphore

    # Semaphores for concurrency control
    llama_semaphore = Semaphore(10)  # Max 10 concurrent Llama calls
    claude_semaphore = Semaphore(3)  # Max 3 concurrent Claude calls

    material_images = []
    non_material_count = 0
    classification_errors = 0

    async def classify_single_image(img_data, index):
        """
        Classify a single image as material or non-material.

        NEW ARCHITECTURE:
        - Download image from Supabase URL (not local disk)
        - Convert to base64 on-the-fly
        - Classify with Llama Vision
        - No disk I/O - everything in memory
        """
        nonlocal non_material_count, classification_errors

        try:
            # Get Supabase storage URL
            storage_url = img_data.get('storage_url')
            if not storage_url:
                logger.warning(f"   ⚠️ [{index}/{len(all_images)}] No storage URL for image")
                classification_errors += 1
                return None

            # Download from Supabase URL and convert to base64
            from app.services.pdf_processor import download_image_to_base64
            image_base64 = await download_image_to_base64(storage_url)

            # Call Llama Vision for classification
            async with llama_semaphore:
                classification = await llamaindex_service._classify_image_material(
                    image_base64=image_base64,
                    confidence_threshold=0.6  # Lowered from 0.7 to reduce false negatives
                )

            if classification and classification.get('is_material', False):
                logger.info(f"   ✅ [{index}/{len(all_images)}] Material image (confidence: {classification.get('confidence', 0):.2f})")
                img_data['classification'] = classification
                return img_data
            else:
                logger.info(f"   ⏭️  [{index}/{len(all_images)}] Non-material image (skipped)")
                non_material_count += 1
                return None

        except Exception as e:
            logger.error(f"   ❌ [{index}/{len(all_images)}] Classification failed: {e}")
            classification_errors += 1
            return None

    # MEMORY OPTIMIZATION: Stream classification results instead of accumulating
    # Process all images in parallel with semaphore control
    classification_tasks = [
        classify_single_image(img_data, idx + 1)
        for idx, img_data in enumerate(all_images)
    ]

    classification_results = await asyncio.gather(*classification_tasks, return_exceptions=True)

    # Filter out None results and exceptions, then immediately delete classification_results
    material_images = [
        result for result in classification_results
        if result is not None and not isinstance(result, Exception)
    ]

    # Delete classification_results to free memory
    del classification_results
    gc.collect()

    logger.info(f"✅ AI Classification Complete:")
    logger.info(f"   Material images: {len(material_images)}")
    logger.info(f"   Non-material images: {non_material_count}")
    logger.info(f"   Classification errors: {classification_errors}")

    # Cleanup: Delete all non-material images from Supabase Storage
    # NOTE: Local files were already deleted immediately after upload in batch processing
    logger.info(f"🧹 Cleaning up {non_material_count} non-material images from Supabase...")
    deleted_count = 0
    for img_data in all_images:
        if img_data not in material_images:
            storage_path = img_data.get('storage_path')
            if storage_path:
                try:
                    await supabase_client.delete_image_file(storage_path)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"   Failed to delete from Supabase: {e}")
    logger.info(f"✅ Cleanup complete: {deleted_count} non-material images deleted from Supabase")

    if not material_images:
        logger.warning("⚠️ No material images identified")
        return {
            "status": "completed",
            "pdf_result_with_images": pdf_result_with_images,
            "material_images": [],
            "images_extracted": 0,
            "images_processed": 0,
            "clip_embeddings_generated": 0,
            "clip_embeddings_expected": 0,
            "clip_completion_rate": 0,
            "specialized_embeddings": 0,
            "images_analyzed": 0,
            "total_images_extracted": len(all_images),
            "non_material_images": len(all_images),
            "quality_flags": {
                "clip_embeddings_complete": True,  # No material images = nothing to fail
                "all_images_analyzed": True,
                "specialized_embeddings_complete": True
            }
        }

    # Step 3: Consolidated batch processing (Upload → Save → CLIP → Llama Vision)
    logger.info(f"🚀 Starting consolidated batch processing for {len(material_images)} material images...")
    logger.info(f"   Operations: Upload → Save to DB → CLIP Embeddings → Llama Vision Analysis")

    # CRITICAL FIX: Load CLIP/SigLIP models ONCE before ALL batches (not per batch)
    # This prevents models from being loaded 12 times (once per batch), which:
    # - Wastes 36+ seconds (12 loads × 3s each)
    # - Causes memory fragmentation
    # - Increases GC pressure
    logger.info(f"🔧 Pre-loading CLIP/SigLIP models ONCE for all {len(material_images)} images...")
    models_loaded = await embedding_service.ensure_models_loaded()
    if not models_loaded:
        logger.error(f"❌ Failed to load models, cannot process images")
        return {
            "status": "failed",
            "error": "Failed to load SigLIP models",
            "pdf_result_with_images": pdf_result_with_images,
            "material_images": material_images,
            "images_extracted": 0,
            "images_processed": 0,
            "clip_embeddings_generated": 0,
            "clip_embeddings_expected": len(material_images),
            "clip_completion_rate": 0,
            "specialized_embeddings": 0,
            "images_analyzed": 0,
            "total_images_extracted": len(all_images),
            "non_material_images": non_material_count,
            "quality_flags": {
                "clip_embeddings_complete": False,
                "all_images_analyzed": False,
                "specialized_embeddings_complete": False
            }
        }
    logger.info(f"✅ SigLIP models loaded successfully - ready for batch processing")

    async def process_single_image_complete(img_data, image_index, total_images):
        """
        Complete processing for a single image: Save → CLIP → Llama Vision

        MEMORY OPTIMIZATION:
        - Download image ONCE at the start
        - Reuse base64 data for CLIP and Llama Vision
        - Delete base64 data immediately after use
        - Saves ~66% memory (3 downloads → 1 download)
        """
        nonlocal images_saved_count, clip_embeddings_generated, specialized_embeddings_generated, images_processed, vecs_batch_records

        image_base64 = None
        try:
            # Verify we have Supabase storage URL (uploaded in batch processing)
            storage_url = img_data.get('storage_url')
            if not storage_url:
                logger.error(f"   ❌ [{image_index}/{total_images}] No storage URL - image not uploaded")
                return None

            logger.info(f"   🔄 [{image_index}/{total_images}] Processing: {img_data.get('filename')}")

            # OPTIMIZATION: Download image ONCE and reuse for all operations
            from app.services.pdf_processor import download_image_to_base64
            image_base64 = await download_image_to_base64(storage_url)
            logger.info(f"   📥 [{image_index}/{total_images}] Downloaded image once for reuse")

            # ✅ NEW: Determine image category based on page number
            page_number = img_data.get('page_number', 1)
            image_category = _determine_image_category(page_number, catalog)

            # STEP 1: Save to database
            image_id = await supabase_client.save_single_image(
                image_info=img_data,
                document_id=document_id,
                workspace_id=workspace_id,
                image_index=image_index - 1,
                category=image_category  # ✅ NEW: Pass category
            )

            if not image_id:
                logger.error(f"   ❌ [{image_index}/{total_images}] Failed to save to DB")
                return None

            img_data['id'] = image_id
            images_saved_count += 1
            logger.info(f"   ✅ [{image_index}/{total_images}] Saved to DB: {image_id}")

            # STEP 2: Generate CLIP embeddings - REUSE downloaded base64
            logger.info(f"   🎨 [{image_index}/{total_images}] Generating CLIP embeddings (reusing downloaded image)...")
            try:
                clip_result = await clip_breaker.call(
                    with_timeout,
                    embedding_service.generate_all_embeddings(
                        entity_id=image_id,
                        entity_type="image",
                        text_content="",
                        image_url=None,  # Don't download again
                        image_data=image_base64,  # ✅ Reuse downloaded base64
                        material_properties={}
                    ),
                    timeout_seconds=TimeoutConstants.CLIP_EMBEDDING,
                    operation_name=f"CLIP embedding (image {image_index}/{total_images})"
                )

                if clip_result and clip_result.get('success'):
                    embeddings = clip_result.get('embeddings', {})

                    # Save visual CLIP embedding to VECS (batch)
                    visual_embedding = embeddings.get('visual_512')
                    if visual_embedding:
                        vecs_batch_records.append((
                            image_id,
                            visual_embedding,
                            {
                                'document_id': document_id,
                                'workspace_id': workspace_id,
                                'page_number': img_data.get('page_number', 1),
                                'image_url': img_data.get('storage_url'),
                                'storage_path': img_data.get('storage_path')
                            }
                        ))
                        clip_embeddings_generated += 1

                    # Save specialized embeddings (SigLIP 1152D)
                    specialized_embeddings = {}
                    for emb_type in ['color_siglip_1152', 'texture_siglip_1152', 'style_siglip_1152', 'material_siglip_1152']:
                        if embeddings.get(emb_type):
                            key = emb_type.replace('_siglip_1152', '')  # Extract: color, texture, style, material
                            specialized_embeddings[key] = embeddings.get(emb_type)

                    if specialized_embeddings:
                        await vecs_service.upsert_specialized_embeddings(
                            image_id=image_id,
                            embeddings=specialized_embeddings,
                            metadata={'document_id': document_id, 'page_number': img_data.get('page_number', 1)}
                        )
                        specialized_embeddings_generated += len(specialized_embeddings)

                    logger.info(f"   ✅ [{image_index}/{total_images}] Generated {1 + len(specialized_embeddings)} CLIP embeddings")

                    # CRITICAL: Delete embeddings dict to free memory immediately
                    del embeddings
                    del specialized_embeddings

                    # Batch upsert VECS records
                    if len(vecs_batch_records) >= VECS_BATCH_SIZE:
                        batch_count = await vecs_service.batch_upsert_image_embeddings(vecs_batch_records)
                        logger.info(f"   💾 Batch upserted {batch_count} CLIP embeddings to VECS")
                        vecs_batch_records.clear()
                        # Force GC after VECS batch upsert to free embedding memory
                        gc.collect()

            except CircuitBreakerError as cb_error:
                logger.warning(f"   ⚠️ [{image_index}/{total_images}] CLIP skipped (circuit breaker): {cb_error}")
            except Exception as clip_error:
                logger.error(f"   ❌ [{image_index}/{total_images}] CLIP failed: {clip_error}")

            # STEP 3: Llama Vision analysis - REUSE downloaded base64
            logger.info(f"   🔍 [{image_index}/{total_images}] Analyzing with Llama Vision (reusing downloaded image)...")
            try:
                analysis_result = await llama_breaker.call(
                    with_timeout,
                    llamaindex_service._analyze_image_material(
                        image_base64=image_base64,  # ✅ Reuse downloaded base64
                        image_path=storage_url,  # Just for logging
                        image_id=image_id,
                        document_id=document_id,
                        embedding_service=embedding_service  # Pass loaded models
                    ),
                    timeout_seconds=TimeoutConstants.LLAMA_VISION_CALL,
                    operation_name=f"Llama Vision (image {image_index}/{total_images})"
                )

                if analysis_result:
                    img_data['quality_score'] = analysis_result.get('quality_score', 0.5)
                    img_data['confidence_score'] = analysis_result.get('confidence_score', 0.5)
                    img_data['material_properties'] = analysis_result.get('material_properties', {})
                    images_processed += 1
                    logger.info(f"   ✅ [{image_index}/{total_images}] Llama Vision complete (quality: {img_data['quality_score']:.2f})")

            except CircuitBreakerError as cb_error:
                logger.warning(f"   ⚠️ [{image_index}/{total_images}] Llama Vision skipped (circuit breaker): {cb_error}")
            except Exception as llama_error:
                logger.error(f"   ❌ [{image_index}/{total_images}] Llama Vision failed: {llama_error}")

            return img_data

        except Exception as e:
            logger.error(f"   ❌ [{image_index}/{total_images}] Failed to process image: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return None
        finally:
            # CRITICAL: Delete downloaded base64 data to free memory
            if image_base64 is not None:
                del image_base64

            # AGGRESSIVE PER-IMAGE MEMORY CLEANUP
            # Run GC after each image to prevent memory accumulation
            gc.collect()

    # Process images in batches
    total_batches = (len(material_images) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"📦 Processing {len(material_images)} images in {total_batches} batches of {BATCH_SIZE}")

    processing_semaphore = Semaphore(CONCURRENT_IMAGES)

    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(material_images))
        batch_images = material_images[start_idx:end_idx]

        logger.info(f"🔄 Processing batch {batch_num + 1}/{total_batches} ({len(batch_images)} images)...")

        # Check memory pressure before processing batch
        mem_stats = memory_monitor.get_memory_stats()
        if mem_stats.is_critical_pressure:
            logger.error(f"🔴 CRITICAL memory pressure: {mem_stats.percent_used:.1f}%")
            await memory_monitor.check_memory_pressure()
            # Re-check after cleanup
            mem_stats = memory_monitor.get_memory_stats()
        elif mem_stats.is_high_pressure:
            logger.warning(f"⚠️ High memory pressure: {mem_stats.percent_used:.1f}%")

        # Check memory before batch
        mem_stats = memory_monitor.get_memory_stats()
        logger.info(f"   💾 Memory before batch: {mem_stats.used_mb:.1f} MB ({mem_stats.percent_used:.1f}%)")

        # NOTE: Models already loaded once before all batches (see line 355)
        # No need to reload per batch - this saves 36+ seconds total

        # Process batch in parallel with semaphore control
        async def process_with_semaphore(img_data, idx):
            async with processing_semaphore:
                return await process_single_image_complete(img_data, start_idx + idx + 1, len(material_images))

        batch_tasks = [
            process_with_semaphore(img_data, idx)
            for idx, img_data in enumerate(batch_images)
        ]

        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # AGGRESSIVE MEMORY CLEANUP after batch
        logger.info(f"   🧹 Starting memory cleanup after batch {batch_num + 1}/{total_batches}...")

        # 1. Delete batch data explicitly (downloaded base64 images from Llama Vision)
        del batch_results
        del batch_tasks
        del batch_images
        logger.info("   ✅ Deleted batch data from memory")

        # 2. NOTE: SigLIP model NOT unloaded per batch anymore (loaded once for all batches)
        #    This saves 36+ seconds total (12 batches × 3s load time)
        #    Model will be unloaded ONCE after all batches complete

        # 3. Force garbage collection
        gc.collect()

        # 4. Small delay to allow OS to reclaim memory
        await asyncio.sleep(0.5)

        mem_after = memory_monitor.get_memory_stats()
        mem_freed = mem_stats.used_mb - mem_after.used_mb
        logger.info(f"   💾 Memory after cleanup: {mem_after.used_mb:.1f} MB (freed: {mem_freed:.1f} MB, {mem_freed/mem_stats.used_mb*100:.1f}%)")

        # Check memory pressure
        if mem_after.is_high_pressure:
            logger.warning(f"   ⚠️ High memory pressure detected: {mem_after.percent_used:.1f}%")

        logger.info(f"   ✅ Batch {batch_num + 1}/{total_batches} complete: {clip_embeddings_generated} CLIP embeddings generated so far")

        # Update progress after each batch
        await tracker.update_database_stats(
            images_stored=images_saved_count,
        )
        await tracker._sync_to_database(stage="image_processing")

    # Final VECS batch upsert for remaining records
    if vecs_batch_records:
        batch_count = await vecs_service.batch_upsert_image_embeddings(vecs_batch_records)
        logger.info(f"💾 Final batch upserted {batch_count} CLIP embeddings to VECS")
        vecs_batch_records.clear()

    # CRITICAL FIX: Unload SigLIP model ONCE after ALL batches complete
    # This frees 2-3GB of memory now that we're done with embeddings
    logger.info("🧹 Unloading SigLIP model after all batches complete...")
    try:
        unloaded = embedding_service.unload_siglip_model()
        if unloaded:
            logger.info("✅ Successfully unloaded SigLIP model (freed ~2-3GB)")
            gc.collect()  # Force GC to reclaim memory immediately
    except Exception as e:
        logger.warning(f"⚠️ Failed to unload SigLIP model: {e}")

    # NOTE: No disk cleanup needed - local files were deleted immediately after upload in batch processing
    # All images are now stored in Supabase Storage only

    logger.info(f"✅ [STAGE 3] Image Processing Complete:")
    logger.info(f"   Images saved to DB: {images_saved_count}")
    logger.info(f"   CLIP embeddings generated: {clip_embeddings_generated}")
    logger.info(f"   Specialized embeddings: {specialized_embeddings_generated}")
    logger.info(f"   Llama Vision analyzed: {images_processed}")

    # Sync tracker to database
    await tracker._sync_to_database(stage="image_processing")

    # Create IMAGES_EXTRACTED checkpoint
    await checkpoint_recovery_service.create_checkpoint(
        job_id=job_id,
        stage=CheckpointStage.IMAGES_EXTRACTED,
        data={
            "document_id": document_id,
            "images_saved": images_saved_count,
            "clip_embeddings": clip_embeddings_generated,
            "specialized_embeddings": specialized_embeddings_generated,
            "images_analyzed": images_processed
        },
        metadata={
            "total_images_extracted": len(all_images),
            "material_images": len(material_images),
            "non_material_images": non_material_count,
            "classification_errors": classification_errors
        }
    )

    # Force garbage collection
    gc.collect()
    logger.info("💾 Memory freed after Stage 3 (Image Processing)")

    # Unload SigLIP model to free memory
    logger.info("   🔄 Unloading SigLIP model to free memory...")
    try:
        # Unload model from embedding service
        embedding_service._models_loaded = False
        embedding_service._siglip_model = None
        embedding_service._siglip_processor = None

        # Force garbage collection to free model memory
        gc.collect()
        logger.info("   ✅ SigLIP model unloaded successfully")

    except Exception as e:
        logger.warning(f"   ⚠️ Failed to unload SigLIP model: {e}")

    # Calculate quality metrics
    expected_clip_embeddings = images_saved_count * 5  # 5 types per image
    clip_completion_rate = (clip_embeddings_generated / expected_clip_embeddings) if expected_clip_embeddings > 0 else 0

    quality_flags = {
        "clip_embeddings_complete": clip_completion_rate >= 0.9,  # 90% threshold
        "all_images_analyzed": images_processed == images_saved_count,
        "specialized_embeddings_complete": specialized_embeddings_generated >= (images_saved_count * 4)  # 4 specialized types
    }

    logger.info(f"📊 Quality Metrics:")
    logger.info(f"   CLIP Completion Rate: {clip_completion_rate:.1%} ({clip_embeddings_generated}/{expected_clip_embeddings})")
    logger.info(f"   Images Analyzed: {images_processed}/{images_saved_count}")
    logger.info(f"   Quality Flags: {quality_flags}")

    return {
        "status": "completed",
        "pdf_result_with_images": pdf_result_with_images,
        "material_images": material_images,
        "images_extracted": images_saved_count,
        "images_processed": images_processed,
        "clip_embeddings_generated": clip_embeddings_generated,
        "clip_embeddings_expected": expected_clip_embeddings,
        "clip_completion_rate": clip_completion_rate,
        "specialized_embeddings": specialized_embeddings_generated,
        "images_analyzed": images_processed,
        "total_images_extracted": len(all_images),
        "non_material_images": non_material_count,
        "quality_flags": quality_flags
    }

