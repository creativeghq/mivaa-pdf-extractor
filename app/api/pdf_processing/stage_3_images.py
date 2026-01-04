"""
Stage 3: Image Processing

This module handles image processing for individual products in the product-centric pipeline.
"""

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
    logger.info(f"   Extracted {total_images} images from product pages")

    if total_images == 0:
        return {'images_processed': 0, 'images_material': 0, 'images_non_material': 0}

    # Use batch classification instead of individual classification
    image_service = ImageProcessingService()
    material_images, non_material_images = await image_service.classify_images(
        extracted_images=pdf_result.extracted_images,
        confidence_threshold=0.6,
        primary_model="Qwen/Qwen3-VL-8B-Instruct",
        validation_model="Qwen/Qwen3-VL-32B-Instruct",
        batch_size=15
    )
    non_material_count = len(non_material_images)

    logger.info(f"   Material: {len(material_images)}, Non-material: {non_material_count}")

    images_processed = 0
    for img_data in material_images:
        try:
            await image_service.upload_image(img_data, workspace_id, document_id)
            await image_service.save_image_to_db(img_data, document_id, workspace_id, job_id)
            await image_service.generate_clip_embedding(img_data)
            images_processed += 1
        except Exception as e:
            logger.error(f"   Failed to process image: {e}")

    logger.info(f"   ✅ Processed {images_processed} material images for {product.name}")

    return {
        'images_processed': images_processed,
        'images_material': len(material_images),
        'images_non_material': non_material_count
    }

