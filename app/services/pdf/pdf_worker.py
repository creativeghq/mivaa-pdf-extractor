
import asyncio
import gc
import inspect
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Import core extraction functions from centralized module
from app.services.pdf.extractor_imports import (
    EXTRACTOR_AVAILABLE,
    extract_pdf_to_markdown,
    extract_pdf_to_markdown_with_doc,
)
from app.services.pdf.ocr_service import OCRConfig, get_ocr_service
from app.utils.exceptions import PDFExtractionError

# Setup logger for this module
logger = logging.getLogger(__name__)

# DPI used for both YOLO layout detection and the rendered image fed to
# Chandra. Keeping the two aligned means YOLO's region pixel coordinates
# and Chandra's text-fragment pixel coordinates live in the same space,
# so layout_merge_service can compare them without scaling.
LAYOUT_RENDER_DPI = 250
PDF_POINTS_TO_PIXEL_ZOOM = LAYOUT_RENDER_DPI / 72.0  # PyMuPDF coords are in points (72 DPI)


def _extract_pymupdf_spans_with_bbox(page: Any) -> List[Dict[str, Any]]:
    """Extract every PyMuPDF text span as a Chandra-compatible bbox dict.

    PyMuPDF returns text in PDF points (72 DPI) by default. We rescale to
    LAYOUT_RENDER_DPI pixels so the spans share a coordinate space with
    YOLO regions and can flow through layout_merge_service unchanged.
    """
    spans: List[Dict[str, Any]] = []
    try:
        page_dict = page.get_text("dict")
    except Exception as exc:
        logger.warning(f"PyMuPDF span extraction failed: {exc}")
        return spans

    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                if not text:
                    continue
                bbox = span.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                x0, y0, x1, y1 = bbox
                spans.append({
                    "text": text,
                    "x": x0 * PDF_POINTS_TO_PIXEL_ZOOM,
                    "y": y0 * PDF_POINTS_TO_PIXEL_ZOOM,
                    "w": (x1 - x0) * PDF_POINTS_TO_PIXEL_ZOOM,
                    "h": (y1 - y0) * PDF_POINTS_TO_PIXEL_ZOOM,
                })
    return spans


def _yolo_all_pages_sync(pdf_path: str, page_count: int) -> Dict[int, List[Any]]:
    """Run YOLO layout detection for every page; return regions keyed 1-based.

    Synchronous wrapper around `YoloLayoutDetector.detect_layout_regions`
    suitable for use inside the sync `execute_pdf_extraction_job` worker
    function. YOLO is best-effort - any individual page failure (or YOLO
    being globally unavailable) yields an empty list for that page so
    callers degrade gracefully to text-only output.
    """
    try:
        from app.services.pdf.yolo_layout_detector import YoloLayoutDetector
    except Exception as exc:
        logger.warning(f"YOLO detector import failed; skipping layout detection: {exc}")
        return {}

    try:
        detector = YoloLayoutDetector()
    except Exception as exc:
        logger.warning(f"YOLO detector init failed; skipping layout detection: {exc}")
        return {}

    if not getattr(detector, "enabled", False):
        logger.info("YOLO layout detection disabled in settings; skipping")
        return {}

    async def _run_all():
        tasks = [
            detector.detect_layout_regions(pdf_path, p, dpi=LAYOUT_RENDER_DPI)
            for p in range(page_count)
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    try:
        results = asyncio.run(_run_all())
    except RuntimeError:
        # An event loop is already running in this thread (rare for the
        # ThreadPoolExecutor worker, but possible if called from an async
        # context). Fall through to a fresh loop.
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(_run_all())
        finally:
            loop.close()
    except Exception as exc:
        logger.warning(f"YOLO batch failed; degrading to text-only OCR: {exc}")
        return {}

    regions_by_page: Dict[int, List[Any]] = {}
    yolo_failures = 0
    for page_idx, result in enumerate(results):
        page_num_1_based = page_idx + 1
        if isinstance(result, Exception):
            yolo_failures += 1
            logger.warning(f"YOLO failed for page {page_num_1_based}: {result}")
            continue
        regions = list(getattr(result, "regions", []) or [])
        if regions:
            regions_by_page[page_num_1_based] = regions

    logger.info(
        f"YOLO per-page detection: {len(regions_by_page)}/{page_count} pages "
        f"got regions, {yolo_failures} failed"
    )
    return regions_by_page


def _build_page_layout_with_text(
    page: Any,
    page_num_1_based: int,
    yolo_regions: List[Any],
    fallback_to_chandra: bool = True,
) -> Tuple[List[Any], str]:
    """Merge YOLO regions with text bboxes for a single page.

    Tries PyMuPDF spans first (free, fast). If the page has too few spans
    relative to YOLO's text-region count, treats it as scanned and runs
    Chandra v2 to get bbox-JSON, then merges that.

    Returns (merged_regions, extraction_path) where extraction_path is
    one of "pymupdf_spans", "chandra_v2", "yolo_only", "none".
    """
    from app.services.pdf.layout_merge_service import merge_layout

    if not yolo_regions:
        return [], "none"

    # Page rendered at LAYOUT_RENDER_DPI - capture its pixel size for the
    # merge function so it can scale 0..1 normalized regions if any sneak in.
    try:
        rect = page.rect
        page_size_px = (
            float(rect.width) * PDF_POINTS_TO_PIXEL_ZOOM,
            float(rect.height) * PDF_POINTS_TO_PIXEL_ZOOM,
        )
    except Exception:
        page_size_px = None

    # Stage A: try PyMuPDF spans. Born-digital pages return non-empty spans;
    # scanned pages return zero. Threshold is just "any spans at all" - if
    # merging them against the YOLO regions produces no content (spans don't
    # land inside any region), we fall through to Chandra below.
    spans = _extract_pymupdf_spans_with_bbox(page)

    if spans:
        merged = merge_layout(yolo_regions, spans, page_size=page_size_px)
        if any(r.text_content.strip() for r in merged):
            return merged, "pymupdf_spans"

    # Stage B: scanned page or PyMuPDF span extraction failed - try Chandra v2
    if not fallback_to_chandra:
        merged = merge_layout(yolo_regions, [], page_size=page_size_px)
        return merged, "yolo_only"

    try:
        from PIL import Image
        import io
        import fitz

        from app.services.pdf.chandra_endpoint_manager import ChandraResponseError

        pix = page.get_pixmap(matrix=fitz.Matrix(PDF_POINTS_TO_PIXEL_ZOOM, PDF_POINTS_TO_PIXEL_ZOOM))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_service = get_ocr_service()
        if ocr_service.chandra_manager is None:
            merged = merge_layout(yolo_regions, [], page_size=page_size_px)
            return merged, "yolo_only"
        try:
            chandra_result = ocr_service.chandra_manager.run_inference(img)
            chandra_blocks = chandra_result.get("blocks") or []
            extraction_path = chandra_result.get("extraction_path", "chandra_v2")
        except ChandraResponseError as cre:
            logger.warning(f"Page {page_num_1_based}: Chandra v2 unparseable: {cre}")
            chandra_blocks = []
            extraction_path = "yolo_only"
        finally:
            try:
                img.close()
            except Exception:
                pass

        merged = merge_layout(yolo_regions, chandra_blocks, page_size=page_size_px)
        return merged, extraction_path if chandra_blocks else "yolo_only"
    except Exception as exc:
        logger.warning(f"Page {page_num_1_based}: Chandra fallback failed: {exc}")
        merged = merge_layout(yolo_regions, [], page_size=page_size_px)
        return merged, "yolo_only"


def _build_layout_regions_by_page(
    pdf_path: str,
    yolo_regions_by_page: Dict[int, List[Any]],
) -> Tuple[Dict[int, List[Any]], Dict[str, int]]:
    """Build per-page merged-region map for the entire PDF.

    Iterates pages once with PyMuPDF open, computes merged regions per
    page, and returns both the regions dict and a counter of which
    extraction path was taken for each page. The counter feeds job
    progress telemetry so the user can see how many pages were OCR'd
    vs. text-extracted.
    """
    import fitz

    layout_regions_by_page: Dict[int, List[Any]] = {}
    extraction_path_counts: Dict[str, int] = {
        "pymupdf_spans": 0,
        "chandra_v2": 0,
        "reasoning_bbox_json": 0,
        "yolo_only": 0,
        "none": 0,
    }

    if not yolo_regions_by_page:
        return layout_regions_by_page, extraction_path_counts

    doc = fitz.open(pdf_path)
    try:
        for page_num_1_based, yolo_regions in yolo_regions_by_page.items():
            page_idx = page_num_1_based - 1
            if page_idx < 0 or page_idx >= len(doc):
                continue
            try:
                page = doc[page_idx]
                merged, extraction_path = _build_page_layout_with_text(
                    page, page_num_1_based, yolo_regions
                )
                if merged:
                    layout_regions_by_page[page_num_1_based] = merged
                # Normalize Chandra reasoning-fallback into its own bucket.
                if extraction_path == "content_bbox_json":
                    extraction_path_counts["chandra_v2"] += 1
                elif extraction_path in extraction_path_counts:
                    extraction_path_counts[extraction_path] += 1
                else:
                    extraction_path_counts["chandra_v2"] += 1
            except Exception as exc:
                logger.warning(f"Page {page_num_1_based}: layout build failed: {exc}")
                extraction_path_counts["none"] += 1
    finally:
        doc.close()

    return layout_regions_by_page, extraction_path_counts

def execute_pdf_extraction_job(
    pdf_path: str,
    processing_options: Dict[str, Any],
) -> Tuple[
    str,
    Dict[str, Any],
    Optional[List[Dict[str, Any]]],
    Dict[int, List[Any]],
]:
    """
    Standalone function to execute PDF extraction.
    Matches logic of PDFProcessor._extract_markdown_sync but runs in the
    ThreadPoolExecutor configured by PDFProcessor.

    Pipeline (YOLO-first):
      1. Run YOLO layout detection on every page (parallel).
      2. For each page: merge YOLO regions with text bboxes - PyMuPDF
         spans first, falling back to Chandra v2 OCR for scanned pages.
      3. Produce both joined markdown (legacy contract) and the
         per-page merged region list (new contract for layout-aware
         chunking + downstream caches).

    Args:
        pdf_path: Absolute path to the PDF file.
        processing_options: Configuration dictionary.

    Returns:
        Tuple of (markdown_content, metadata, page_chunks, layout_regions_by_page).
        - markdown_content: Combined markdown string.
        - metadata: Extraction metadata, including
          `extraction_path_counts` for job-progress telemetry.
        - page_chunks: Optional list of per-page dicts with metadata.
        - layout_regions_by_page: dict[page_num_1_based] -> list of
          MergedRegion dataclasses ready to be persisted to
          `document_layout_analysis` and forwarded to the chunker.
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
        page_chunks = None  # Stores page-aware data when page_list is provided

        # Check if page_list is provided for focused extraction
        page_list = processing_options.get('page_list')
        if page_list:
            # Use page_chunks=True to preserve page metadata
            logger.info(f"Worker: Using page-aware extraction for {len(page_list)} pages")
            try:
                import pymupdf4llm
                # Convert 1-indexed to 0-indexed for PyMuPDF4LLM
                page_indices = [p - 1 for p in page_list if p > 0]
                
                # ✅ VALIDATION: Filter out-of-bounds pages.
                # Promoted from WARNING to ERROR when ALL pages drop or when
                # >50% drop — those cases mean a calling-side bug (wrong
                # `total_pages` argument, e.g. PDF-sheet-count vs physical
                # page count for spread layouts) and should not be silently
                # tolerated. See bug 2026-05-01.
                valid_indices = [p for p in page_indices if p < total_pages]
                if len(valid_indices) < len(page_indices):
                    dropped = [p + 1 for p in page_indices if p >= total_pages]
                    drop_pct = 100.0 * (len(page_indices) - len(valid_indices)) / max(len(page_indices), 1)
                    if drop_pct >= 50.0:
                        logger.error(
                            f"Worker: ❌ Dropping {len(page_indices) - len(valid_indices)}/{len(page_indices)} "
                            f"({drop_pct:.0f}%) out-of-bounds pages: {dropped} (total_pages={total_pages}). "
                            f"This usually means the caller passed the wrong page count "
                            f"(e.g. PDF sheet count instead of physical page count for spread layouts)."
                        )
                    else:
                        logger.warning(
                            f"Worker: Dropping {len(page_indices) - len(valid_indices)} out-of-bounds pages: "
                            f"{dropped} (Total pages in PDF: {total_pages})"
                        )
                    page_indices = valid_indices

                if not page_indices:
                    logger.error(
                        "Worker: ❌ No valid pages remain after out-of-bounds filtering — "
                        "skipping page-aware extraction. Caller passed an invalid total_pages."
                    )
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

        # ============================================================
        # YOLO-first layout pass (every page, both born-digital + scanned)
        # ============================================================
        # Run YOLO across all pages, then merge regions with text bboxes
        # (PyMuPDF spans for born-digital pages, Chandra v2 for scanned).
        # Result is per-page merged regions with `text_content` populated -
        # caller persists to `document_layout_analysis` and feeds the
        # layout-aware chunker. Failures here are non-fatal: legacy
        # markdown_content path still ships text downstream.
        layout_regions_by_page: Dict[int, List[Any]] = {}
        extraction_path_counts: Dict[str, int] = {}
        try:
            yolo_regions_by_page = _yolo_all_pages_sync(pdf_path, total_pages)
            if yolo_regions_by_page:
                layout_regions_by_page, extraction_path_counts = _build_layout_regions_by_page(
                    pdf_path, yolo_regions_by_page
                )
            metadata["yolo_pages_with_regions"] = len(yolo_regions_by_page)
            metadata["layout_pages_with_text"] = len(layout_regions_by_page)
            metadata["extraction_path_counts"] = extraction_path_counts
            logger.info(
                f"Worker layout pass: yolo_pages={len(yolo_regions_by_page)} "
                f"layout_pages={len(layout_regions_by_page)} "
                f"path_counts={extraction_path_counts}"
            )
        except Exception as exc:
            # Layout pass is opt-in best-effort. Log and ship plain markdown.
            logger.warning(f"Worker layout pass failed (non-fatal): {exc}")
            metadata["yolo_pages_with_regions"] = 0
            metadata["layout_pages_with_text"] = 0
            metadata["extraction_path_counts"] = {}

        # Explicit GC before returning
        gc.collect()

        return markdown_content, metadata, page_chunks, layout_regions_by_page

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
    return extract_text_with_ocr_pages_standalone(pdf_path, None, options)


def extract_text_with_ocr_pages_standalone(
    pdf_path: str,
    pages: Optional[List[int]],
    options: Dict,
    yolo_regions_by_page: Optional[Dict[int, List[Any]]] = None,
    merged_regions_out: Optional[Dict[int, List[Any]]] = None,
) -> str:
    """Standalone OCR extraction for specific pages.

    When `yolo_regions_by_page` is provided, every page that has YOLO
    regions runs Chandra v2's bbox-JSON output through the layout merge
    service to produce per-region text. Each MergedRegion is appended to
    `merged_regions_out[page_num]` (keyed by 1-based page number) so the
    caller can persist `text_content` per `product_layout_regions` row.

    Pages without YOLO regions fall through to the same plain-text join
    as before, so callers that don't pass YOLO regions see the legacy
    contract unchanged.
    """
    import fitz
    from PIL import Image
    import io

    from app.services.pdf.chandra_endpoint_manager import ChandraResponseError
    from app.services.pdf.layout_merge_service import merge_layout, merged_regions_to_joined_text

    ocr_languages = options.get('ocr_languages', ['en'])
    ocr_config = OCRConfig(languages=ocr_languages, confidence_threshold=0.3, preprocessing_enabled=True)
    ocr_service = get_ocr_service(ocr_config)

    doc = fitz.open(pdf_path)
    if pages is None:
        pages = list(range(len(doc)))

    all_text: List[str] = []
    for page_num in pages:
        if page_num >= len(doc):
            continue

        try:
            page = doc.load_page(page_num)
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            page_size_px = (float(pix.width), float(pix.height))
            img_data = pix.tobytes('png')
            pix = None  # free mem

            img = Image.open(io.BytesIO(img_data))

            page_regions = (yolo_regions_by_page or {}).get(page_num + 1, [])
            chandra_blocks: Optional[List[Dict[str, Any]]] = None
            page_text = ""

            if page_regions and ocr_service.chandra_manager is not None:
                # Direct Chandra v2 call to get bbox blocks; merge with YOLO.
                try:
                    raw = ocr_service.chandra_manager.run_inference(img)
                    chandra_blocks = raw.get("blocks") or []
                except ChandraResponseError as cre:
                    logger.warning(f"Worker OCR page {page_num + 1}: Chandra v2 unparseable: {cre}")
                    chandra_blocks = None
                except Exception as e:
                    logger.warning(f"Worker OCR page {page_num + 1}: Chandra call failed, falling back: {e}")
                    chandra_blocks = None

            if chandra_blocks is not None:
                merged = merge_layout(page_regions, chandra_blocks, page_size=page_size_px)
                page_text = merged_regions_to_joined_text(merged)
                if merged_regions_out is not None:
                    merged_regions_out[page_num + 1] = merged
            else:
                # Legacy path: ocr_service runs full pipeline (Chandra -> EasyOCR -> Tesseract).
                results = ocr_service.extract_text_from_image(img)
                page_text = " ".join(
                    [r.text.strip() for r in results if r.text.strip() and r.confidence > 0.3]
                )

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
