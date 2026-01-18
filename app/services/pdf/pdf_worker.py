
import logging
import os
import gc
import inspect
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Import core extraction functions from centralized module
from app.services.pdf.extractor_imports import (
    extract_pdf_to_markdown,
    extract_pdf_to_markdown_with_doc,
    EXTRACTOR_AVAILABLE
)

from app.services.pdf.ocr_service import get_ocr_service, OCRConfig
from app.utils.exceptions import PDFExtractionError

# Setup logger for this module
logger = logging.getLogger(__name__)

def execute_pdf_extraction_job(
    pdf_path: str,
    processing_options: Dict[str, Any]
) -> Tuple[str, Dict[str, Any], Optional[List[Dict[str, Any]]]]:  # ✅ NEW: Return page_chunks
    """
    Standalone function to execute PDF extraction in a separate process.
    Matches logic of PDFProcessor._extract_markdown_sync but designed for ProcessPoolExecutor.

    Args:
        pdf_path: Absolute path to the PDF file
        processing_options: Configuration dictionary

    Returns:
        Tuple of (markdown_content, metadata, page_chunks)
        - markdown_content: Combined markdown string
        - metadata: Extraction metadata
        - page_chunks: Optional list of page dicts with metadata (when page_list is provided)
    """
    try:
        # Re-configure logging for worker process if needed
        # logging.basicConfig(...) # implementation specific

        # First, analyze the PDF to determine if it's image-based
        import fitz
        if not os.path.exists(pdf_path):
             raise PDFExtractionError(f"PDF file not found at path: {pdf_path}")
             
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        # Sample pages strategically
        pages_to_sample = []
        pages_to_sample.extend(range(min(3, total_pages)))
        if total_pages > 5:
            mid_page = total_pages // 2
            pages_to_sample.extend([mid_page - 1, mid_page])
        if total_pages > 3:
            pages_to_sample.extend([total_pages - 2, total_pages - 1])
        pages_to_sample = sorted(set(pages_to_sample))
        
        total_text_chars = 0
        total_images = 0

        for page_num in pages_to_sample:
            page = doc[page_num]
            text = page.get_text()
            images = page.get_images()
            total_text_chars += len(text.strip())
            total_images += len(images)

        doc.close()

        # Detection logic
        avg_text_per_page = total_text_chars / len(pages_to_sample)
        avg_images_per_page = total_images / len(pages_to_sample)

        criteria = {
            'low_text': avg_text_per_page < 50,
            'very_low_text': avg_text_per_page < 10,
            'has_images': avg_images_per_page >= 1,
            'many_images': avg_images_per_page >= 3,
            'text_to_image_ratio': (avg_text_per_page / max(avg_images_per_page, 1)) < 30,
            'no_images': avg_images_per_page == 0
        }

        is_image_based = (
            (criteria['very_low_text'] and criteria['has_images']) or
            (criteria['low_text'] and criteria['many_images']) or
            (criteria['many_images'] and criteria['text_to_image_ratio'])
        ) and not criteria['no_images']

        logger.info(f"Worker PDF Analysis: {total_pages} pages. Image-based: {is_image_based}")

        markdown_content = ""
        page_chunks = None  # ✅ NEW: Will store page-aware data if page_list is provided

        # ✅ NEW: Check if page_list is provided for focused extraction
        page_list = processing_options.get('page_list')
        if page_list:
            # Use page_chunks=True to preserve page metadata
            logger.info(f"Worker: Using page-aware extraction for {len(page_list)} pages")
            try:
                import pymupdf4llm
                # Convert 1-indexed to 0-indexed for PyMuPDF4LLM
                page_indices = [p - 1 for p in page_list if p > 0]
                
                # ✅ VALIDATION: Filter out-of-bounds pages
                valid_indices = [p for p in page_indices if p < total_pages]
                if len(valid_indices) < len(page_indices):
                    dropped = [p + 1 for p in page_indices if p >= total_pages]
                    logger.warning(f"Worker: Dropping {len(page_indices) - len(valid_indices)} out-of-bounds pages: {dropped} (Total pages in PDF: {total_pages})")
                    page_indices = valid_indices

                if not page_indices:
                    logger.warning("Worker: No valid pages to extract, skipping page-aware extraction")
                    page_chunks = None
                else:
                    page_chunks = pymupdf4llm.to_markdown(pdf_path, pages=page_indices, page_chunks=True)

                # Create combined markdown from page chunks
                markdown_content = "\n\n-----\n\n".join([page.get('text', '') for page in page_chunks])
                logger.info(f"Worker: Extracted {len(page_chunks)} pages with metadata, {len(markdown_content)} chars total")
            except Exception as e:
                logger.warning(f"Worker: Page-aware extraction failed: {e}, falling back to standard extraction")
                page_chunks = None
                # Fall through to standard extraction

        # Standard extraction (if page_list not provided or page-aware extraction failed)
        if not page_chunks:
            if is_image_based:
                logger.info("Worker: Using OCR-first extraction")
                try:
                    markdown_content = extract_text_with_ocr_standalone(pdf_path, processing_options)
                    if len(markdown_content.strip()) < 100:
                        logger.info("Worker: OCR minimal, trying PyMuPDF fallback")
                        page_number = processing_options.get('page_number')
                        fallback_content = extract_pdf_to_markdown(pdf_path, page_number)
                        if len(fallback_content.strip()) > len(markdown_content.strip()):
                            markdown_content = fallback_content
                except Exception as ocr_error:
                    logger.error(f"Worker OCR failed: {ocr_error}")
                    page_number = processing_options.get('page_number')
                    markdown_content = extract_pdf_to_markdown(pdf_path, page_number)
            else:
                logger.info("Worker: Using PyMuPDF extraction")
                page_number = processing_options.get('page_number')
                try:
                    markdown_content = extract_pdf_to_markdown(pdf_path, page_number)
                except ValueError as e:
                    # Fallback to page-by-page if "not a textpage" error
                    if "not a textpage" in str(e):
                        logger.warning("Worker: PyMuPDF batch failed, switching to page-by-page")
                        markdown_content = _extract_page_by_page_safe(pdf_path, processing_options)
                    else:
                        raise

                # Content verification
                clean_content = markdown_content.replace('-', '').replace('\n', '').strip()
                if len(clean_content) < 100:
                    logger.info("Worker: Text extract minimal, attempting OCR")
                    try:
                        ocr_content = extract_text_with_ocr_standalone(pdf_path, processing_options)
                        if len(ocr_content.strip()) > len(clean_content):
                            markdown_content = ocr_content
                    except Exception as e:
                        logger.warning(f"Worker OCR fallback failed: {e}")

        # Extract metadata
        metadata = _extract_metadata_standalone(pdf_path, len(markdown_content))

        # Explicit GC before returning
        gc.collect()

        return markdown_content, metadata, page_chunks  # ✅ NEW: Return page_chunks

    except Exception as e:
        logger.error(f"Worker process failed: {e}")
        gc.collect()
        raise

def _extract_page_by_page_safe(pdf_path: str, processing_options: Dict) -> str:
    """Helper for safe page-by-page extraction."""
    import fitz
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    content_chunks = []
    failed_pages = []
    
    # Process in chunks of 5 pages
    batch_size = 5
    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)
        for page_num in range(batch_start, batch_end):
            try:
                page_content = extract_pdf_to_markdown_with_doc(doc, page_num)
                content_chunks.append(page_content)
            except Exception as e:
                logger.warning(f"Page {page_num} failed: {e}")
                failed_pages.append(page_num)
        gc.collect()
    
    doc.close()
    
    if failed_pages:
        logger.info(f"Worker: Retrying {len(failed_pages)} failed pages with OCR")
        ocr_content = extract_text_with_ocr_pages_standalone(pdf_path, failed_pages, processing_options)
        content_chunks.append(ocr_content)
        
    return "\n\n".join(content_chunks)

def extract_text_with_ocr_standalone(pdf_path: str, options: Dict) -> str:
    """Standalone OCR extraction for full doc."""
    # Simplified version of _extract_text_with_ocr
    return extract_text_with_ocr_pages_standalone(pdf_path, None, options)

def extract_text_with_ocr_pages_standalone(pdf_path: str, pages: Optional[List[int]], options: Dict) -> str:
    """Standalone OCR extraction for specific pages."""
    import fitz
    from PIL import Image
    import io
    
    ocr_languages = options.get('ocr_languages', ['en'])
    ocr_config = OCRConfig(languages=ocr_languages, confidence_threshold=0.3, preprocessing_enabled=True)
    ocr_service = get_ocr_service(ocr_config)
    
    doc = fitz.open(pdf_path)
    if pages is None:
        pages = list(range(len(doc)))
        
    all_text = []
    for page_num in pages:
        if page_num >= len(doc): continue
        
        try:
            page = doc.load_page(page_num)
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes('png')
            pix = None # free mem
            
            img = Image.open(io.BytesIO(img_data))
            results = ocr_service.extract_text_from_image(img)
            
            page_text = " ".join([r.text.strip() for r in results if r.text.strip() and r.confidence > 0.3])
            if page_text:
                all_text.append(f"## Page {page_num + 1}\n\n{page_text}\n")
            
            img.close()
            img = None
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Worker OCR page {page_num} failed: {e}")
            
    doc.close()
    return "\n".join(all_text)

def _extract_metadata_standalone(pdf_path: str, text_len: int) -> Dict:
    import fitz
    doc = fitz.open(pdf_path)
    meta = {
        'page_count': doc.page_count,
        'title': doc.metadata.get('title'),
        'author': doc.metadata.get('author'),
        'creation_date': doc.metadata.get('creationDate'),
        'text_length': text_len
    }
    doc.close()
    return meta
