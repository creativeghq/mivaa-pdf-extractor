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
    from app.services.pdf_processor import PDFProcessor
    from app.services.image_processing_service import ImageProcessingService

    logger.info(f"🖼️  Processing images for product: {product.name}")
    logger.info(f"   Pages: {sorted(product_pages)}")

    pdf_processor = PDFProcessor()
    processing_options = {
        'extract_images': True,
        'extract_tables': False,
        'page_list': list(product_pages),
        'extract_categories': ['products']
    }

    pdf_result = await pdf_processor.process_pdf_from_bytes(
        pdf_bytes=file_content,
        document_id=document_id,
        processing_options=processing_options
    )

    total_images = len(pdf_result.extracted_images) if pdf_result.extracted_images else 0

    # ✅ FIX: Provide more context when no images are extracted
    if total_images == 0:
        logger.info(f"   📄 No images extracted from product pages {sorted(product_pages)}")
        logger.debug(f"      This is normal for text-only pages or pages without embedded images")
        return {'images_processed': 0, 'images_material': 0, 'images_non_material': 0}

    logger.info(f"   Extracted {total_images} images from product pages")

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

    try:
        material_images, non_material_images = await image_service.classify_images(
            extracted_images=pdf_result.extracted_images,
            confidence_threshold=0.6,
            primary_model="Qwen/Qwen3-VL-8B-Instruct",
            validation_model="Qwen/Qwen3-VL-32B-Instruct",
            batch_size=15
        )
    except Exception as e:
        logger.error(f"   ❌ Image classification failed completely: {type(e).__name__}: {str(e)}")
        logger.error(f"      This is a critical error - no images will be processed")
        raise
    non_material_count = len(non_material_images)

    logger.info(f"   Material: {len(material_images)}, Non-material: {non_material_count}")

    # ✅ NEW: Use batch processing methods for Better Reliability & Metadata Preservation
    # 1. Upload images to storage
    logger.info(f"   📤 Uploading {len(material_images)} material images to cloud storage...")
    uploaded_images = await image_service.upload_images_to_storage(
        material_images=material_images,
        document_id=document_id
    )

    # 2. Save to DB and Generate CLIP embeddings
    logger.info(f"   💾 Saving metadata and generating CLIP embeddings...")
    save_result = await image_service.save_images_and_generate_clips(
        material_images=uploaded_images,
        document_id=document_id,
        workspace_id=workspace_id
    )

    images_processed = save_result.get('images_saved', 0)
    logger.info(f"   ✅ Processed {images_processed} material images for {product.name}")

    return {
        'images_processed': images_processed,
        'images_material': len(material_images),
        'images_non_material': non_material_count
    }

