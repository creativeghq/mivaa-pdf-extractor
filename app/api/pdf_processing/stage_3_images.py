"""
Stage 3: Image Processing

This module handles image processing for individual products in the product-centric pipeline.
"""

import os
import logging
from typing import Dict, Any, Set


async def process_product_images(
    file_content: bytes,
    document_id: str,
    workspace_id: str,
    job_id: str,
    product: Any,
    product_pages: Set[int],
    catalog: Any,
    config: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Process images for a single product (product-centric pipeline).

    Args:
        file_content: PDF file bytes
        document_id: Document identifier
        workspace_id: Workspace identifier
        job_id: Job identifier
        product: Single product object
        product_pages: Set of page numbers for this product
        catalog: Full catalog (for context)
        config: Processing configuration
        logger: Logger instance

    Returns:
        Dictionary with images_processed, images_material, images_non_material counts
    """
    from app.services.pdf.pdf_processor import PDFProcessor
    from app.services.images.image_processing_service import ImageProcessingService
    from app.utils.page_converter import PageConverter

    logger.info(f"🖼️  Processing images for product: {product.name}")
    logger.info(f"   PDF array indices (0-based): {sorted(product_pages)}")

    # ✅ FIX: Convert 0-based array indices to 1-based catalog pages
    # product_pages contains 0-based array indices from Stage 1
    # PDF processor expects 1-based catalog pages in page_list
    pages_per_sheet = getattr(catalog, 'pages_per_sheet', 1)
    converter = PageConverter(pages_per_sheet=pages_per_sheet)

    catalog_pages = []
    for array_index in product_pages:
        try:
            page = converter.from_array_index(array_index)
            catalog_pages.append(page.catalog_page)
        except ValueError as e:
            logger.warning(f"   ⚠️ Skipping invalid array index {array_index}: {e}")
            continue

    logger.info(f"   Catalog pages (1-based): {sorted(catalog_pages)}")

    pdf_processor = PDFProcessor()
    processing_options = {
        'extract_images': True,
        'extract_tables': False,
        'page_list': catalog_pages,  # ✅ FIX: Use catalog pages, not array indices
        'extract_categories': ['products']
    }

    pdf_result = await pdf_processor.process_pdf_from_bytes(
        pdf_bytes=file_content,
        document_id=document_id,
        processing_options=processing_options
    )

    total_images = len(pdf_result.extracted_images) if pdf_result.extracted_images else 0

    # ✅ DETAILED LOGGING: Track image extraction results
    logger.info(f"   📊 IMAGE EXTRACTION SUMMARY:")
    logger.info(f"      Total images extracted: {total_images}")

    if pdf_result.extracted_images:
        # Log images per page
        images_by_page = {}
        for img in pdf_result.extracted_images:
            page_num = img.get('page_number', 'unknown')
            images_by_page[page_num] = images_by_page.get(page_num, 0) + 1

        logger.info(f"      Images per page: {dict(sorted(images_by_page.items()))}")

        # Log extraction methods
        extraction_methods = {}
        for img in pdf_result.extracted_images:
            method = img.get('extraction_method', 'unknown')
            extraction_methods[method] = extraction_methods.get(method, 0) + 1

        logger.info(f"      Extraction methods: {extraction_methods}")

        # Log image sizes
        sizes = [img.get('size_bytes', 0) for img in pdf_result.extracted_images]
        if sizes:
            logger.info(f"      Image sizes: min={min(sizes)} bytes, max={max(sizes)} bytes, avg={sum(sizes)//len(sizes)} bytes")

    # ✅ FIX: Provide more context when no images are extracted
    if total_images == 0:
        logger.warning(f"   ⚠️ NO IMAGES EXTRACTED!")
        logger.warning(f"      Catalog pages requested: {sorted(catalog_pages)}")
        logger.warning(f"      PDF array indices: {sorted(product_pages)}")
        logger.warning(f"      This could mean:")
        logger.warning(f"      1. Pages are text-only (no embedded images)")
        logger.warning(f"      2. Page number conversion failed")
        logger.warning(f"      3. Images were filtered out by size/quality thresholds")
        return {'images_processed': 0, 'images_material': 0, 'images_non_material': 0}

    logger.info(f"   ✅ Extracted {total_images} images from {len(catalog_pages)} pages")

    # ✅ FIX 8: Log image paths before classification to verify they exist
    logger.info(f"   Verifying extracted image files...")
    missing_files = []
    for i, img in enumerate(pdf_result.extracted_images[:5]):  # Check first 5
        img_path = img.get('path', '')
        exists = os.path.exists(img_path) if img_path else False
        logger.debug(f"     Image {i+1}: {img.get('filename')} - exists: {exists}")
        if not exists:
            missing_files.append(img.get('filename'))

    if missing_files:
        logger.error(f"   ❌ WARNING: {len(missing_files)} image files are missing before classification!")
        logger.error(f"      Missing files: {missing_files[:3]}")

    # Use batch classification instead of individual classification
    image_service = ImageProcessingService()

    logger.info(f"   🤖 Starting AI classification of {total_images} images...")
    logger.info(f"      Primary model: Qwen/Qwen3-VL-32B-Instruct")
    logger.info(f"      Validation model: claude-sonnet-4-20250514")
    logger.info(f"      Confidence threshold: 0.6")

    try:
        material_images, non_material_images = await image_service.classify_images(
            extracted_images=pdf_result.extracted_images,
            confidence_threshold=0.6,
            primary_model="Qwen/Qwen3-VL-32B-Instruct",
            validation_model="claude-sonnet-4-20250514",
            batch_size=15
        )
    except Exception as e:
        logger.error(f"   ❌ Image classification failed completely: {type(e).__name__}: {str(e)}")
        logger.error(f"      This is a critical error - no images will be processed")
        logger.error(f"      Stack trace:", exc_info=True)
        raise

    non_material_count = len(non_material_images)

    logger.info(f"   📊 CLASSIFICATION RESULTS:")
    logger.info(f"      Material images: {len(material_images)}")
    logger.info(f"      Non-material images: {non_material_count}")
    logger.info(f"      Classification rate: {(len(material_images) + non_material_count) / total_images * 100:.1f}%")

    # Log which images were classified as non-material
    if non_material_images:
        logger.info(f"      Non-material image filenames:")
        for img in non_material_images[:5]:  # Show first 5
            logger.info(f"         - {img.get('filename')} (confidence: {img.get('ai_classification', {}).get('confidence', 'N/A')})")

    # ✅ NEW: Use batch processing methods for Better Reliability & Metadata Preservation
    # 1. Upload images to storage
    logger.info(f"   📤 UPLOAD STAGE: Uploading {len(material_images)} material images to cloud storage...")
    uploaded_images = await image_service.upload_images_to_storage(
        material_images=material_images,
        document_id=document_id
    )

    logger.info(f"   📊 UPLOAD RESULTS:")
    logger.info(f"      Successfully uploaded: {len(uploaded_images)}/{len(material_images)}")
    if len(uploaded_images) < len(material_images):
        failed_count = len(material_images) - len(uploaded_images)
        logger.warning(f"      ⚠️ Failed to upload {failed_count} images")
        logger.warning(f"         Check logs above for upload errors")

    # 2. Save to DB and Generate CLIP embeddings
    logger.info(f"   💾 DATABASE STAGE: Saving metadata and generating CLIP embeddings...")
    save_result = await image_service.save_images_and_generate_clips(
        material_images=uploaded_images,
        document_id=document_id,
        workspace_id=workspace_id
    )

    images_processed = save_result.get('images_saved', 0)
    clip_embeddings = save_result.get('clip_embeddings_generated', 0)
    failed_images = save_result.get('failed_images', [])

    logger.info(f"   📊 DATABASE RESULTS:")
    logger.info(f"      Images saved to DB: {images_processed}/{len(uploaded_images)}")
    logger.info(f"      CLIP embeddings generated: {clip_embeddings}/{images_processed}")

    if failed_images:
        logger.warning(f"      ⚠️ Failed to save {len(failed_images)} images:")
        for failed in failed_images[:5]:  # Show first 5 failures
            logger.warning(f"         - {failed.get('path')} - Error: {failed.get('error')}")

    logger.info(f"   ✅ STAGE 3 COMPLETE for {product.name}")
    logger.info(f"      Total extracted: {total_images}")
    logger.info(f"      Material images: {len(material_images)}")
    logger.info(f"      Successfully processed: {images_processed}")
    logger.info(f"      CLIP embeddings: {clip_embeddings}")

    return {
        'images_processed': images_processed,
        'clip_embeddings_generated': clip_embeddings,
        'images_material': len(material_images),
        'images_non_material': non_material_count
    }

