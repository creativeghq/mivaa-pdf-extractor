"""
Stage 1.5: Document-level layout precompute (physical-page-aware).

Iterates **physical pages** 1..total_physical_pages — the same numbering
products and chunks use throughout the rest of the pipeline. For each
physical page:

1. Look up `(pdf_idx, position)` from `PDFLayoutAnalysis.physical_to_pdf_map`.
   `position` is one of `single` / `full` / `left` / `right`. For spread
   layouts a single PDF sheet maps to 2 physical pages; we render only
   the relevant half before sending to YOLO.
2. Render the half-page (or full sheet) to a PIL image at LAYOUT_RENDER_DPI.
3. Run YOLO on that image.
4. Get text fragments aligned to the same clip:
   - PyMuPDF spans clipped to the half-rect (born-digital path, ~free).
   - Falls back to Chandra v2 OCR on scanned pages (no spans inside the
     clip area).
5. `merge_layout(yolo_regions, text_fragments)` produces MergedRegion[]
   with `text_content` populated.
6. Persist to `document_layout_analysis` keyed on
   `(document_id, physical_page_number)` — same key Stage 2 chunking
   reads later via `get_layout_from_document_cache`.

Why this matters: the chunker keys regions by physical page. If we
persisted regions keyed by PDF sheet index (the round-17 P2 patch did
this implicitly), every cache read would miss and the chunker would
fall back to text-based chunking — silently nullifying Stage 1.5's
work. Iterating physical pages with `physical_to_pdf_map`-aware
rendering keeps the cache compatible with downstream consumers.

Resume-safe: pages already in the cache are skipped.
Toggle: `LAYOUT_PRECOMPUTE_ENABLED` env var (defaults True).
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import fitz  # PyMuPDF
from PIL import Image


# Single canonical stage name surfaced to the UI through
# background_jobs.stage_history. Keep stable so the frontend can match.
STAGE_NAME = "stage_1_5_layout_precompute"


def _emit_stage_event(
    supabase: Any,
    job_id: Optional[str],
    status: str,
    data: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Append a Stage 1.5 event to background_jobs.stage_history.

    Audit fix #23: previously a single failed RPC silently lost the event.
    Now retries once before logging at ERROR. Still never propagates —
    observability must not break the pipeline — but the retry catches
    transient network blips and the ERROR level surfaces persistent issues.
    """
    if not job_id:
        return
    payload = {
        "p_job_id": job_id,
        "p_event": {
            "stage": STAGE_NAME,
            "status": status,
            "completed_at": datetime.utcnow().isoformat(),
            "data": data,
            "source": "stage_1_5_layout_precompute",
        },
    }
    last_err: Optional[Exception] = None
    for attempt in range(2):
        try:
            supabase.client.rpc("append_stage_history", payload).execute()
            return
        except Exception as hist_err:
            last_err = hist_err
    logger.error(
        f"❌ [STAGE 1.5] stage_history append failed twice for {job_id} "
        f"(audit-log gap): {last_err}"
    )


# Render DPI for YOLO input. 250 keeps small typography legible while
# capping per-page memory at ~3-5 MB for letter-sized sheets. Matches
# the default `dpi` arg in YoloLayoutDetector.convert_pdf_page_to_image.
LAYOUT_RENDER_DPI = 250

# Stage 1.5 reuses the standard Chandra OCR prompt; named explicitly here
# so the per-page call site doesn't have to know about chandra internals.
DEFAULT_OCR_PROMPT_FOR_STAGE_1_5 = (
    "Extract all text from this image. Return a JSON array where each entry is "
    '{"text": <fragment>, "x": <int>, "y": <int>, "w": <int>, "h": <int>}. '
    "Output JSON only - no commentary, no markdown."
)

# pdf-points → pixel scale for the render DPI above. fitz uses 72 dpi
# baseline. PDF_POINTS_TO_PIXEL_ZOOM = LAYOUT_RENDER_DPI / 72.
PDF_POINTS_TO_PIXEL_ZOOM = LAYOUT_RENDER_DPI / 72.0


def _clip_rect_for_position(page: "fitz.Page", position: str) -> Optional["fitz.Rect"]:
    """Return the fitz.Rect to clip to for a physical page's `position`.

    `position` values come from `PDFLayoutAnalysis.physical_to_pdf_map`:
        single / full → full page (no clip)
        left          → left half of a spread
        right         → right half of a spread

    Anything else returns None (caller falls back to full page).
    """
    if position in ("left", "right"):
        rect = page.rect
        mid_x = rect.width / 2.0
        if position == "left":
            return fitz.Rect(0, 0, mid_x, rect.height)
        return fitz.Rect(mid_x, 0, rect.width, rect.height)
    # single / full / unknown → full page; caller passes None to skip clip.
    return None


def _render_physical_page(
    doc: "fitz.Document",
    pdf_idx: int,
    position: str,
) -> Tuple[Image.Image, "fitz.Rect"]:
    """Render the half-page (or full sheet) for one physical page.

    Returns the PIL image AND the fitz.Rect that bounded the render
    in pdf-point space. The rect is needed by `_pymupdf_spans_in_clip`
    to scope text-extraction to the same area.
    """
    page = doc[pdf_idx]
    clip = _clip_rect_for_position(page, position)
    matrix = fitz.Matrix(PDF_POINTS_TO_PIXEL_ZOOM, PDF_POINTS_TO_PIXEL_ZOOM)
    if clip is not None:
        pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
        bound_rect = clip
    else:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        bound_rect = page.rect

    img = Image.open(io.BytesIO(pix.tobytes("png")))
    pix = None  # free C-side buffer immediately
    return img, bound_rect


def _pymupdf_spans_in_clip(
    page: "fitz.Page",
    clip: Optional["fitz.Rect"],
) -> List[Dict[str, Any]]:
    """Extract PyMuPDF spans intersecting `clip`, in pixel-space bboxes.

    Spans go through the same coordinate scaling as the rendered image
    (PDF_POINTS_TO_PIXEL_ZOOM) so they line up with YOLO regions, which
    are reported in pixel coordinates. For spread layouts we also
    translate x by -clip.x0 so each half-page span is in the half's own
    coordinate frame (matching the rendered image's origin).
    """
    spans: List[Dict[str, Any]] = []
    text_dict = page.get_text("dict")  # blocks → lines → spans
    x_offset = clip.x0 if clip is not None else 0.0
    y_offset = clip.y0 if clip is not None else 0.0

    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                bbox = span.get("bbox")
                text = (span.get("text") or "").strip()
                if not text or not bbox or len(bbox) < 4:
                    continue
                sx0, sy0, sx1, sy1 = bbox
                # Skip spans entirely outside the clip
                if clip is not None:
                    if sx1 < clip.x0 or sx0 > clip.x1 or sy1 < clip.y0 or sy0 > clip.y1:
                        continue
                spans.append({
                    "text": text,
                    "x": int((sx0 - x_offset) * PDF_POINTS_TO_PIXEL_ZOOM),
                    "y": int((sy0 - y_offset) * PDF_POINTS_TO_PIXEL_ZOOM),
                    "w": int(max(1, sx1 - sx0) * PDF_POINTS_TO_PIXEL_ZOOM),
                    "h": int(max(1, sy1 - sy0) * PDF_POINTS_TO_PIXEL_ZOOM),
                })
    return spans


async def precompute_document_layout(
    document_id: str,
    pdf_path: str,
    supabase: Any,
    logger: logging.Logger,
    total_physical_pages_hint: Optional[int] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run YOLO + bbox-text merge for every uncached physical page; persist.

    Args:
        document_id: documents.id (FK target of document_layout_analysis)
        pdf_path: local path to the page-numbered PDF (Stage 0 output)
        supabase: supabase client (writer)
        logger: scoped logger
        total_physical_pages_hint: optional override; otherwise derived
            from `analyze_pdf_layout(pdf_path).total_physical_pages`.
        job_id: when provided, emits `stage_1_5_layout_precompute` events
            (started / completed / failed) to `background_jobs.stage_history`
            so the async-job UI can render Stage 1.5 progress directly from
            the job row. No-op without job_id (e.g. ad-hoc backfills).

    Returns:
        Dict with keys: pages_total, pages_cached_already, pages_processed,
        pages_persisted, extraction_paths (counter dict).
    """
    started_at = time.time()
    summary: Dict[str, Any] = {
        "pages_total": 0,
        "pages_cached_already": 0,
        "pages_processed": 0,
        "pages_persisted": 0,
        "extraction_paths": {},
    }

    # 1. Layout analysis tells us the physical→PDF mapping. This is the
    #    ONLY way Stage 1.5 stays compatible with downstream consumers
    #    that key everything by physical page number.
    try:
        from app.utils.pdf_to_images import analyze_pdf_layout
        layout = await asyncio.to_thread(analyze_pdf_layout, pdf_path)
    except Exception as e:
        logger.warning(f"   ⚠️ [STAGE 1.5] analyze_pdf_layout failed: {e}")
        _emit_stage_event(
            supabase, job_id, "failed",
            {**summary, "error": f"analyze_pdf_layout failed: {e}",
             "duration_ms": int((time.time() - started_at) * 1000)},
            logger,
        )
        return summary

    physical_to_pdf = dict(layout.physical_to_pdf_map or {})
    total_physical = (
        total_physical_pages_hint
        or getattr(layout, "total_physical_pages", 0)
        or len(physical_to_pdf)
    )
    summary["pages_total"] = total_physical
    if total_physical <= 0 or not physical_to_pdf:
        logger.info("⏭️  [STAGE 1.5] No physical pages to process")
        _emit_stage_event(
            supabase, job_id, "completed",
            {**summary, "skipped_reason": "no_physical_pages",
             "duration_ms": int((time.time() - started_at) * 1000)},
            logger,
        )
        return summary

    # 2. Resume-safe: skip pages already in cache, BUT NOT pages whose cached
    #    row is marked `cache_status='ocr_failed'`. Previously a transient
    #    Chandra failure would persist an empty `layout_elements: []` row that
    #    the resume check treated as "cached, skip forever", silently
    #    suppressing OCR for that page on every subsequent run (audit fix #6).
    #    Filtering by cache_status here lets re-runs retry failed pages.
    existing_pages: Set[int] = set()
    try:
        resp = (
            supabase.client.table("document_layout_analysis")
            .select("page_number, analysis_metadata")
            .eq("document_id", document_id)
            .execute()
        )
        for row in (resp.data or []):
            meta = row.get("analysis_metadata") or {}
            if meta.get("cache_status") == "ocr_failed":
                # Allow retry on next run.
                continue
            existing_pages.add(row["page_number"])
    except Exception as e:
        logger.warning(f"   ⚠️ [STAGE 1.5] cache-existence query failed: {e}")

    summary["pages_cached_already"] = len(existing_pages)
    pages_to_process = sorted(
        p for p in range(1, total_physical + 1)
        if p not in existing_pages and p in physical_to_pdf
    )

    # Emit `started` once we know the workload (pages_total + already-cached
    # split). This is the single moment between Stage 0 and Stage 2 the UI
    # needs to surface — earlier than this we don't know the work, later
    # the user can't tell that Stage 1.5 is the active stage.
    _emit_stage_event(
        supabase, job_id, "in_progress",
        {**summary, "pages_to_process": len(pages_to_process)},
        logger,
    )

    if not pages_to_process:
        logger.info(
            f"♻️ [STAGE 1.5] All {total_physical} physical pages already cached — skipping"
        )
        _emit_stage_event(
            supabase, job_id, "completed",
            {**summary, "skipped_reason": "all_cached",
             "duration_ms": int((time.time() - started_at) * 1000)},
            logger,
        )
        return summary

    logger.info(
        f"📐 [STAGE 1.5] Precomputing layout for {len(pages_to_process)} physical page(s); "
        f"{len(existing_pages)} already cached, {total_physical} total"
    )

    # 3. Resolve the YOLO detector + Chandra OCR fallback once, then
    #    iterate physical pages serially. Serial iteration keeps memory
    #    flat (one rendered image at a time) and aligns naturally with
    #    the module-level YOLO warmup lock — no parallel hot-storm.
    from app.services.pdf.yolo_layout_detector import YoloLayoutDetector
    from app.services.pdf.layout_merge_service import merge_layout
    from app.services.pdf.chandra_endpoint_manager import ChandraResponseError

    detector: Optional[YoloLayoutDetector] = None
    try:
        detector = YoloLayoutDetector()
        if not getattr(detector, "enabled", False):
            detector = None
    except Exception as e:
        logger.warning(f"   ⚠️ [STAGE 1.5] YOLO detector init failed: {e}")
        detector = None

    ocr_service = None
    chandra_manager = None
    try:
        from app.services.pdf.ocr_service import get_ocr_service
        ocr_service = get_ocr_service()
        chandra_manager = getattr(ocr_service, "chandra_manager", None)
    except Exception as e:
        logger.debug(f"   Chandra fallback not available: {e}")

    extraction_paths: Dict[str, int] = {
        "pymupdf_spans": 0,
        "chandra_v2": 0,
        "yolo_only": 0,
        "none": 0,
    }
    persisted = 0

    doc = await asyncio.to_thread(fitz.open, pdf_path)
    try:
        for physical_page in pages_to_process:
            pdf_idx, position = physical_to_pdf[physical_page]
            extraction_path = "none"
            merged_regions: List[Any] = []
            # cache_status semantics (audit fix #6, #24):
            #   'success'      — text fragments + region structure both present
            #   'yolo_only'    — YOLO regions present, no text (legitimate empty-text page)
            #   'empty_page'   — neither YOLO regions nor text fragments
            #   'ocr_failed'   — Chandra exhausted retries; row will be retried on next run
            #   'page_failed'  — outer exception; row will be retried on next run
            cache_status = "empty_page"

            try:
                # 3a. Render the relevant half (or full sheet) to a PIL image.
                image, bound_rect = await asyncio.to_thread(
                    _render_physical_page, doc, pdf_idx, position
                )

                # 3b. Run YOLO on the rendered image.
                yolo_regions: List[Any] = []
                if detector is not None:
                    yolo_result = await detector.detect_from_image(
                        image=image, page_num=physical_page
                    )
                    yolo_regions = list(yolo_result.regions)

                # 3c. Get text fragments inside the same clip — PyMuPDF
                #     spans first (free, born-digital), Chandra v2 OCR if
                #     no spans landed in the clip (scanned page).
                clip_rect = _clip_rect_for_position(doc[pdf_idx], position)
                pdf_spans = await asyncio.to_thread(
                    _pymupdf_spans_in_clip, doc[pdf_idx], clip_rect
                )

                fragments: List[Dict[str, Any]] = []
                chandra_failed = False
                if pdf_spans:
                    fragments = pdf_spans
                    extraction_path = "pymupdf_spans"
                elif chandra_manager is not None:
                    try:
                        chandra_result = await asyncio.to_thread(
                            chandra_manager.run_inference,
                            image,
                            None,                                 # parameters
                            DEFAULT_OCR_PROMPT_FOR_STAGE_1_5,     # prompt
                            "stage_1_5_layout_precompute",        # caller
                            None,                                 # image_id (unused at page level)
                            job_id,                               # job_id
                            document_id,                          # document_id
                        )
                        chandra_blocks = chandra_result.get("blocks") or []
                        if chandra_blocks:
                            fragments = chandra_blocks
                            extraction_path = "chandra_v2"
                    except ChandraResponseError as cre:
                        # All retries exhausted — mark cache_status='ocr_failed'
                        # so the resume-skip filter excludes this row and a
                        # future run will retry (audit fix #7).
                        chandra_failed = True
                        logger.warning(
                            f"   ⚠️ [STAGE 1.5] Chandra exhausted retries on physical "
                            f"page {physical_page}: {cre}"
                        )
                    except Exception as ce:
                        chandra_failed = True
                        logger.warning(
                            f"   ⚠️ [STAGE 1.5] Chandra HTTP/endpoint error on physical "
                            f"page {physical_page}: {ce}"
                        )

                # 3d. Page-size hint for merge_layout to scale
                #     0..1-normalized YOLO regions into pixel space.
                page_size_px = (
                    float(bound_rect.width) * PDF_POINTS_TO_PIXEL_ZOOM,
                    float(bound_rect.height) * PDF_POINTS_TO_PIXEL_ZOOM,
                )

                # 3e. Merge YOLO regions + text fragments.
                merged_regions = merge_layout(
                    yolo_regions, fragments, page_size=page_size_px
                )

                # Determine cache_status (audit fix #6, #24).
                if chandra_failed:
                    cache_status = "ocr_failed"
                    if not merged_regions and yolo_regions:
                        extraction_path = "yolo_only"
                elif fragments and merged_regions:
                    has_text = any(
                        getattr(r, "text_content", "").strip()
                        for r in merged_regions
                    )
                    cache_status = "success" if has_text else "yolo_only"
                elif yolo_regions:
                    cache_status = "yolo_only"
                    extraction_path = "yolo_only"
                else:
                    cache_status = "empty_page"

                # Free the render before persisting (cap RSS).
                try:
                    image.close()
                except Exception:
                    pass

            except Exception as e:
                cache_status = "page_failed"
                logger.warning(
                    f"   ⚠️ [STAGE 1.5] page {physical_page} failed: {e}"
                )

            extraction_paths[extraction_path] = (
                extraction_paths.get(extraction_path, 0) + 1
            )

            # 3f. Persist for this physical page (always — empty rows are
            #     valid: they short-circuit a future re-run from doing the
            #     same useless work).
            try:
                # Build layout_elements + reading_order in lock-step so the
                # cache readers (pdf_processor._load_cached_layout_for_pages
                # and stage_1_focused_extraction._load_cached_layout) get
                # both fields. The 2026-04-30 refactor of Stage 1.5 lost
                # `processing_version`, `reading_order`, and
                # `structure_confidence` from the payload — which silently
                # invalidated every cache lookup downstream because the
                # readers gate on `processing_version == 'yolo+chandra-v2'`.
                # Result: Stage 3 Layer 2 (YOLO crop) saw cache=empty and
                # fell back to live YOLO, which got hammered by the
                # per-product loop and produced ~0 images. Restoring the
                # three fields here brings image extraction back to its
                # pre-refactor behavior. See migration df751cb.
                layout_elements_payload = (
                    [r.to_dict() for r in merged_regions]
                    if merged_regions else []
                )
                reading_order_payload = [
                    {
                        "index": idx,
                        "region_type": elem.get("region_type"),
                        "reading_order": elem.get("reading_order"),
                    }
                    for idx, elem in enumerate(layout_elements_payload)
                ]

                payload = {
                    "document_id": document_id,
                    "page_number": physical_page,
                    "layout_elements": layout_elements_payload,
                    "reading_order": reading_order_payload,
                    "structure_confidence": 0.85,
                    "processing_version": "yolo+chandra-v2",
                    "analysis_metadata": {
                        "stage_1_5": True,
                        "extraction_path": extraction_path,
                        "cache_status": cache_status,
                        "pdf_idx": pdf_idx,
                        "position": position,
                        "region_count": len(merged_regions),
                        "has_text_content": any(
                            getattr(r, "text_content", "").strip()
                            for r in merged_regions
                        ),
                    },
                }
                (
                    supabase.client.table("document_layout_analysis")
                    .upsert(payload, on_conflict="document_id,page_number")
                    .execute()
                )
                persisted += 1
            except Exception as e:
                logger.warning(
                    f"   ⚠️ [STAGE 1.5] cache write failed for physical "
                    f"page {physical_page}: {e}"
                )
    finally:
        try:
            doc.close()
        except Exception:
            pass

    summary["pages_processed"] = len(pages_to_process)
    summary["pages_persisted"] = persisted
    summary["extraction_paths"] = extraction_paths

    logger.info(
        f"✅ [STAGE 1.5] Cached {persisted}/{len(pages_to_process)} pages — "
        f"extraction paths: {extraction_paths}"
    )
    _emit_stage_event(
        supabase, job_id, "completed",
        {**summary, "duration_ms": int((time.time() - started_at) * 1000)},
        logger,
    )
    return summary


async def get_layout_from_document_cache(
    document_id: str,
    physical_pages: List[int],
    supabase: Any,
    logger: logging.Logger,
) -> Dict[int, List[Dict[str, Any]]]:
    """Backward-compatible thin wrapper — same shape as before.

    Returns `{physical_page: [regions]}` for pages whose cache row has
    non-empty regions. Use `get_layout_from_document_cache_with_status`
    when the caller needs to distinguish "no cache row" from
    "cached row with empty regions".
    """
    full = await get_layout_from_document_cache_with_status(
        document_id, physical_pages, supabase, logger
    )
    return {p: v["regions"] for p, v in full.items() if v["regions"]}


async def get_layout_from_document_cache_with_status(
    document_id: str,
    physical_pages: List[int],
    supabase: Any,
    logger: logging.Logger,
) -> Dict[int, Dict[str, Any]]:
    """Read cached merged regions WITH cache_status semantics (audit fix #24).

    Returns `{physical_page: {regions: [...], cache_status: 'success'|'yolo_only'|
    'empty_page'|'ocr_failed'|'page_failed'|None}}`. cache_status=None means
    the row exists but predates the 2026-05-01 migration that added the field.

    The chunker uses this to decide:
      - cache_status=='success' → use layout-aware chunking
      - cache_status=='ocr_failed' → log + emit metric, fall back to text-based
        (the failure will be retried by the next Stage 1.5 run because the
         resume-skip query filters out ocr_failed rows)
      - cache_status missing OR no row → cache miss, fall back to text-based
    """
    if not document_id or not physical_pages:
        return {}

    try:
        resp = (
            supabase.client.table("document_layout_analysis")
            .select("page_number, layout_elements, analysis_metadata")
            .eq("document_id", document_id)
            .in_("page_number", physical_pages)
            .execute()
        )
    except Exception as e:
        logger.debug(f"   document_layout_analysis read failed: {e}")
        return {}

    rows = resp.data or []
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        regions = row.get("layout_elements") or []
        if not isinstance(regions, list):
            regions = []
        meta = row.get("analysis_metadata") or {}
        out[row["page_number"]] = {
            "regions": regions,
            "cache_status": meta.get("cache_status"),
            "extraction_path": meta.get("extraction_path"),
        }

    non_empty = sum(1 for v in out.values() if v["regions"])
    if non_empty:
        logger.info(
            f"   ♻️ Loaded {sum(len(v['regions']) for v in out.values())} merged regions "
            f"from document_layout_analysis cache "
            f"({non_empty}/{len(physical_pages)} physical pages)"
        )
    return out
