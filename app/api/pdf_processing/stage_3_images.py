"""
Stage 3: Image Processing

This module handles image processing for individual products in the product-centric pipeline.
"""

import os
import logging
from typing import Dict, Any, Set, Optional, List

# ============================================================================
# SINGLETON PDF PROCESSOR - Reuse across all products to prevent re-initialization
# ============================================================================
_pdf_processor_instance: Optional[Any] = None


def get_pdf_processor():
    """Get or create singleton PDFProcessor instance."""
    global _pdf_processor_instance
    if _pdf_processor_instance is None:
        from app.services.pdf.pdf_processor import PDFProcessor
        _pdf_processor_instance = PDFProcessor()
        logging.getLogger(__name__).info("♻️ Created singleton PDFProcessor for Stage 3")
    return _pdf_processor_instance


def clear_pdf_processor():
    """Clear the singleton PDFProcessor (call at job completion)."""
    global _pdf_processor_instance
    _pdf_processor_instance = None


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
    from app.services.images.image_processing_service import ImageProcessingService

    logger.info(f"🖼️  Processing images for product: {product.name}")
    logger.info(f"   PDF page indices (0-based): {sorted(product_pages)}")

    # Convert 0-based PDF indices to 1-based PDF page numbers
    # NOTE: product_pages now contains CORRECT PDF indices (already accounting for spread layout)
    pdf_pages_1based = [idx + 1 for idx in product_pages]
    logger.info(f"   PDF pages (1-based): {sorted(pdf_pages_1based)}")

    # Build PDF page -> physical page mapping for image metadata correction
    has_spread_layout = catalog and getattr(catalog, 'has_spread_layout', False)
    physical_to_pdf_map = catalog.physical_to_pdf_map if has_spread_layout else {}

    # Create reverse mapping: PDF page (1-based) -> physical page(s)
    pdf_to_physical_map: Dict[int, List[int]] = {}
    if has_spread_layout and physical_to_pdf_map:
        logger.info(f"   📐 Spread layout detected - building PDF-to-physical page mapping")
        for physical_page, (pdf_idx, position) in physical_to_pdf_map.items():
            pdf_page_1based = pdf_idx + 1
            if pdf_page_1based not in pdf_to_physical_map:
                pdf_to_physical_map[pdf_page_1based] = []
            pdf_to_physical_map[pdf_page_1based].append(physical_page)
        # Sort physical pages for each PDF page
        for pdf_page in pdf_to_physical_map:
            pdf_to_physical_map[pdf_page].sort()
        logger.info(f"   📐 PDF to physical mapping: {pdf_to_physical_map}")

    # ✅ FIX: Use singleton PDFProcessor to prevent repeated SLIG client initialization
    pdf_processor = get_pdf_processor()
    
    # NEW: We need to handle spreads by potentially splitting image extraction
    # and adjusting bounding boxes to physical pages.
    
    # Process images from PDF result
    extracted_images_list = []
    
    for pdf_page_1based in pdf_pages_1based:
        pdf_idx = pdf_page_1based - 1
        
        # Get physical pages for this sheet
        sheet_physical_pages = pdf_to_physical_map.get(pdf_page_1based, [pdf_page_1based])
        
        # Extract images for this specific sheet
        processing_options = {
            'extract_images': True,
            'extract_text': False,
            'extract_tables': False,
            'page_list': [pdf_page_1based],
            'extract_categories': ['products']
        }
        
        page_result = await pdf_processor.process_pdf_from_bytes(
            pdf_bytes=file_content,
            document_id=document_id,
            processing_options=processing_options
        )
        
        if not page_result.extracted_images:
            continue
            
        # For each image, determine which physical page it's on if it's a spread
        if has_spread_layout and len(sheet_physical_pages) > 1:
            # It's a spread (usually 2 physical pages)
            import fitz
            
            # Use catalog's pre-analyzed sheet width (avoids redundant PDF opening)
            sheet_width = 0
            if catalog and hasattr(catalog, 'pdf_page_widths') and catalog.pdf_page_widths:
                sheet_width = catalog.pdf_page_widths.get(pdf_idx, 0)
            
            # Fallback if catalog missing width (unlikely)
            if sheet_width == 0:
                import fitz
                doc = fitz.open(stream=file_content, filetype="pdf")
                sheet = doc[pdf_idx]
                sheet_width = sheet.rect.width
                doc.close()
                logger.debug(f"      ⚠️ Sheet width for page {pdf_idx} missing from catalog, opened PDF")
            
            mid_x = sheet_width / 2
            
            # sheet_physical_pages should be [left_phys, right_phys]
            left_phys = min(sheet_physical_pages)
            right_phys = max(sheet_physical_pages)
            
            for img in page_result.extracted_images:
                # Use bbox or center point to determine side
                bbox = img.get('bbox', [0, 0, 0, 0]) # [x, y, w, h]
                img_width = bbox[2]
                center_x = bbox[0] + (img_width / 2)
                
                # SCENE DETECTION: Wide image bridging the center
                spans_middle = (bbox[0] < mid_x) and (bbox[0] + img_width > mid_x)
                is_scene = spans_middle and (img_width > sheet.rect.width * 0.45)
                
                if is_scene:
                    # Assign to the first physical page of the pair and mark as scene
                    img['pdf_page_number'] = pdf_page_1based
                    img['page_number'] = left_phys
                    img['physical_side'] = 'spread'
                    img['is_scene'] = True
                    # Only add once even if iterating through physical pages
                    if physical_page == left_phys:
                        extracted_images_list.append(img)
                        logger.info(f"      🎞️ Scene image detected on PDF page {pdf_page_1based}")
                else:
                    # Normal side detection
                    is_left = center_x < mid_x
                    target_phys = left_phys if is_left else right_phys
                    
                    if physical_page == target_phys:
                        img['pdf_page_number'] = pdf_page_1based
                        img['page_number'] = target_phys
                        img['physical_side'] = 'left' if is_left else 'right'
                        extracted_images_list.append(img)
        else:
            # Single page or fallback
            target_phys = sheet_physical_pages[0]
            for img in page_result.extracted_images:
                img['pdf_page_number'] = pdf_page_1based
                img['page_number'] = target_phys
                extracted_images_list.append(img)

    total_images = len(extracted_images_list)

    if extracted_images_list:
        # Log images per page (now showing physical pages)
        images_by_page = {}
        for img in extracted_images_list:
            page_num = img.get('page_number', 'unknown')
            images_by_page[page_num] = images_by_page.get(page_num, 0) + 1
 
        logger.info(f"      Images per physical page: {dict(sorted(images_by_page.items()))}")
 
        # Log extraction methods
        extraction_methods = {}
        for img in extracted_images_list:
            method = img.get('extraction_method', 'unknown')
            extraction_methods[method] = extraction_methods.get(method, 0) + 1
 
        logger.info(f"      Extraction methods: {extraction_methods}")
 
        # Log image sizes
        sizes = [img.get('size_bytes', 0) for img in extracted_images_list]
        if sizes:
            logger.info(f"      Image sizes: min={min(sizes)} bytes, max={max(sizes)} bytes, avg={sum(sizes)//len(sizes)} bytes")

    # ✅ FIX: Provide more context when no images are extracted
    if total_images == 0:
        logger.warning(f"   ⚠️ NO IMAGES EXTRACTED!")
        logger.warning(f"      PDF pages requested (1-based): {sorted(pdf_pages_1based)}")
        logger.warning(f"      PDF array indices (0-based): {sorted(product_pages)}")
        logger.warning(f"      This could mean:")
        logger.warning(f"      1. Pages are text-only (no embedded images)")
        logger.warning(f"      2. Page number conversion failed")
        logger.warning(f"      3. Images were filtered out by size/quality thresholds")
        return {'images_processed': 0, 'images_material': 0, 'images_non_material': 0, 'clip_embeddings_generated': 0}

    logger.info(f"   ✅ Extracted {total_images} images from {len(product_pages)} PDF pages")

    # ✅ FIX 8: Log image paths before classification to verify they exist
    logger.info(f"   Verifying extracted image files...")
    missing_files = []
    for i, img in enumerate(extracted_images_list[:5]):  # Check first 5
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

