"""
Stage 1: Document-level structural pass (PaddleOCR-VL, physical-page-aware).

The pipeline's layout+OCR backbone. Iterates **physical pages**
1..total_physical_pages — the same numbering products and chunks use
throughout the rest of the pipeline. For each physical page:

1. Look up `(pdf_idx, position)` from `PDFLayoutAnalysis.physical_to_pdf_map`.
   `position` is one of `single` / `full` / `left` / `right`. For spread
   layouts a single PDF sheet maps to 2 physical pages; we render only
   the relevant half before sending to PaddleOCR.
2. Render the half-page (or full sheet) to a PIL image at LAYOUT_RENDER_DPI.
3. One PaddleOCR structural pass over that image → layout regions (label + bbox)
   + OCR'd text + figure boxes, in a single /v1/chat/completions call.
   PaddleOCR returns region boxes, text, and bucketing all internally.
4. `regions_to_layout_elements()` maps the PaddleOCR regions onto the existing
   `layout_elements` schema (region_type + pixel bbox + text_content).
5. Persist to `document_layout_analysis` keyed on
   `(document_id, physical_page_number)`, `processing_version='paddleocr-vl'` —
   the same key/shape Stage 0 discovery, Stage 2 chunking, and Stage 3 crops
   read later.

This runs BEFORE discovery (structure-first): discovery reads PaddleOCR's
reading-order text instead of raw page.get_text().

Resume-safe: pages already in the cache are skipped (ocr_failed/page_failed
rows are retried).
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


# Render DPI for the PaddleOCR structural pass. 250 keeps small typography
# legible while capping per-page memory at ~3-5 MB for letter-sized sheets.
LAYOUT_RENDER_DPI = 250

# Default OCR prompt for Stage 1.5; named explicitly here so the per-page
# call site doesn't have to know about prompt internals.
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
    (PDF_POINTS_TO_PIXEL_ZOOM) so they line up with the layout regions, which
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
    only_physical_pages: Optional[List[int]] = None,
    tracker: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the PaddleOCR structural pass for every uncached physical page; persist.

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

    # 2. Resume-safe: skip pages already in cache EXCEPT for transient-failure
    #    rows that should retry on next run. The audit fix #6 originally
    #    excluded only `ocr_failed`; `page_failed` (outer-exception path,
    #    written at line ~478 below) was documented as "will be retried" but
    #    the resume-skip filter never excluded it, so any transient render /
    #    structural-pass error permanently skipped that page. Both transient
    #    markers must be excluded.
    _RETRY_CACHE_STATUSES = {"ocr_failed", "page_failed"}
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
            if meta.get("cache_status") in _RETRY_CACHE_STATUSES:
                # Allow retry on next run.
                continue
            existing_pages.add(row["page_number"])
    except Exception as e:
        logger.warning(f"   ⚠️ [STAGE 1.5] cache-existence query failed: {e}")

    summary["pages_cached_already"] = len(existing_pages)

    # Optional per-product scoping (test-mode): when caller provides
    # `only_physical_pages`, restrict the work to those pages instead of
    # the full document range. Used by `test_single_product=True` to skip
    # ~5-6 min of full-document precompute on 140-page catalogs (job
    # 051e1dda). The full-document path remains the default for normal runs.
    if only_physical_pages:
        candidate_pages = sorted(set(only_physical_pages) & set(physical_to_pdf.keys()))
        logger.info(
            f"🧪 [STAGE 1.5] Page-range scoping: limiting to {len(candidate_pages)} "
            f"of {total_physical} pages (caller passed only_physical_pages)"
        )
    else:
        candidate_pages = list(range(1, total_physical + 1))

    pages_to_process = sorted(
        p for p in candidate_pages
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

    # Register this stage as a long-running op so the auto-recovery cron's
    # stuck-job detector doesn't false-positive on large catalogs. Layout
    # precompute on 200+ page documents routinely runs past the default
    # stuck-threshold (5 min) — without this flag the cron would re-dispatch
    # the job mid-stage and Stage 1.5 would start over from the resume point.
    # Budget: 30s/page (worst case with PaddleOCR retries), floor 300s.
    #
    # When a `tracker` is provided, we use its stack-based set_slow_operation
    # which is nest-safe vs Stage 3's parallel per-product markers. Without a
    # tracker (e.g. backfill scripts), fall back to a direct UPDATE — the
    # collision risk only exists when Stage 3 is concurrent, which doesn't
    # happen on the backfill path.
    _slow_op_key = f'stage_1_5_layout_precompute:{len(pages_to_process)}_pages'
    _slow_op_budget = max(300, len(pages_to_process) * 30)
    _stage_1_5_slow_op_set = False
    if tracker is not None:
        try:
            await tracker.set_slow_operation(
                operation=_slow_op_key,
                expected_max_seconds=_slow_op_budget,
            )
            _stage_1_5_slow_op_set = True
        except Exception as _slow_err:
            logger.debug(f"   tracker.set_slow_operation failed (non-fatal): {_slow_err}")
    elif job_id:
        try:
            slow_op_payload = {
                'current_slow_operation': {
                    'operation': _slow_op_key,
                    'started_at': datetime.utcnow().isoformat(),
                    'expected_max_seconds': _slow_op_budget,
                },
                'updated_at': datetime.utcnow().isoformat(),
            }
            # supabase-py is sync; run the update in a thread so we don't block
            # the event loop while the PaddleOCR fan-out runs.
            await asyncio.to_thread(
                lambda: supabase.client.table('background_jobs')
                    .update(slow_op_payload)
                    .eq('id', job_id)
                    .execute()
            )
            _stage_1_5_slow_op_set = True
        except Exception as _slow_err:
            logger.debug(f"   set_slow_operation (fallback) failed (non-fatal): {_slow_err}")

    # 3. Resolve the PaddleOCR structural-pass manager once, then iterate physical
    #    pages serially. Serial iteration keeps memory flat (one rendered image
    #    at a time). PaddleOCR returns layout regions + OCR text + figure boxes in a
    #    single call, so there is no separate detector, OCR fallback, or merge.
    from app.services.embeddings.endpoint_registry import endpoint_registry
    from app.services.pdf.paddleocr_pipeline import regions_to_layout_elements
    from app.services.pdf.paddleocr_endpoint_manager import (
        PaddleOCRResponseError,
        PaddleOCRConfigError,
    )

    paddleocr_manager = endpoint_registry.get_paddleocr_manager()
    if paddleocr_manager is None:
        logger.error("   ❌ [STAGE 1.5] PaddleOCR manager unavailable — cannot precompute layout")
        _emit_stage_event(
            supabase, job_id, "failed",
            {**summary, "error": "paddleocr_manager_unavailable",
             "duration_ms": int((time.time() - started_at) * 1000)},
            logger,
        )
        return summary

    extraction_paths: Dict[str, int] = {
        "paddleocr": 0,
        "paddleocr_no_text": 0,
        "none": 0,
    }
    persisted = 0

    # Circuit breaker: if the first pages all fail with HTTP/endpoint errors, the
    # endpoint is down/misconfigured — abort the whole job instead of grinding
    # every page × retries against a dead endpoint (that was the request-flood).
    consecutive_http_failures = 0
    CIRCUIT_BREAKER_THRESHOLD = 3

    doc = await asyncio.to_thread(fitz.open, pdf_path)
    try:
        for physical_page in pages_to_process:
            pdf_idx, position = physical_to_pdf[physical_page]
            extraction_path = "none"
            layout_elements_payload: List[Dict[str, Any]] = []
            # cache_status semantics:
            #   'success'        — blocks present, at least one carries OCR text
            #   'paddleocr_no_text'  — blocks present but none carry text (e.g. a
            #                      full-bleed image page) — legitimate; cached
            #   'empty_page'     — PaddleOCR returned no regions for the page
            #   'ocr_failed'     — PaddleOCR exhausted retries; row retried next run
            #   'page_failed'    — outer exception (render etc.); retried next run
            cache_status = "empty_page"

            try:
                # 3a. Render the relevant half (or full sheet) to a PIL image.
                image, _bound_rect = await asyncio.to_thread(
                    _render_physical_page, doc, pdf_idx, position
                )
                page_w_px, page_h_px = image.size

                # 3b. One PaddleOCR structural pass → layout regions + OCR text +
                #     figure boxes, in the rendered half-page's own pixel frame.
                try:
                    paddle_result = await asyncio.to_thread(
                        paddleocr_manager.run_structural_pass,
                        image,
                        "stage_1_5_layout_precompute",   # caller
                        physical_page,                   # page_number
                        job_id,                          # job_id
                        document_id,                     # document_id
                    )
                    regions = paddle_result.get("regions") or []
                    layout_elements_payload = regions_to_layout_elements(
                        regions, page_w_px, page_h_px, physical_page
                    )
                    if layout_elements_payload:
                        has_text = any(
                            (el.get("text_content") or "").strip()
                            for el in layout_elements_payload
                        )
                        cache_status = "success" if has_text else "paddleocr_no_text"
                        extraction_path = "paddleocr" if has_text else "paddleocr_no_text"
                    else:
                        cache_status = "empty_page"
                        extraction_path = "none"
                    consecutive_http_failures = 0  # a real response — endpoint is up
                except PaddleOCRConfigError as cfg_err:
                    # NON-retryable (401/403/404) — the endpoint is misconfigured.
                    # Abort the whole job immediately; no point processing more pages.
                    try:
                        image.close()
                    except Exception:
                        pass
                    logger.error(
                        f"   ❌ [STAGE 1.5] PaddleOCR endpoint misconfigured — aborting job: {cfg_err}"
                    )
                    _emit_stage_event(
                        supabase, job_id, "failed",
                        {**summary, "error": f"paddleocr_config_error: {cfg_err}",
                         "duration_ms": int((time.time() - started_at) * 1000)},
                        logger,
                    )
                    raise RuntimeError(f"PaddleOCR endpoint misconfigured: {cfg_err}") from cfg_err
                except PaddleOCRResponseError as sre:
                    # Retries exhausted — mark 'ocr_failed' so the resume-skip
                    # filter excludes this row and a future run retries it.
                    cache_status = "ocr_failed"
                    consecutive_http_failures += 1
                    logger.warning(
                        f"   ⚠️ [STAGE 1.5] PaddleOCR exhausted retries on physical "
                        f"page {physical_page}: {sre}"
                    )
                except Exception as se:
                    cache_status = "ocr_failed"
                    consecutive_http_failures += 1
                    logger.warning(
                        f"   ⚠️ [STAGE 1.5] PaddleOCR HTTP/endpoint error on physical "
                        f"page {physical_page}: {se}"
                    )

                # Circuit breaker: too many consecutive failures = endpoint down.
                if consecutive_http_failures >= CIRCUIT_BREAKER_THRESHOLD:
                    try:
                        image.close()
                    except Exception:
                        pass
                    msg = (
                        f"PaddleOCR endpoint failed {consecutive_http_failures} pages in a "
                        f"row — aborting job (endpoint down/too slow). Last page {physical_page}."
                    )
                    logger.error(f"   ❌ [STAGE 1.5] {msg}")
                    _emit_stage_event(
                        supabase, job_id, "failed",
                        {**summary, "error": f"paddleocr_circuit_breaker: {msg}",
                         "duration_ms": int((time.time() - started_at) * 1000)},
                        logger,
                    )
                    raise RuntimeError(msg)

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
                # layout_elements_payload was built above from the PaddleOCR regions
                # (it is [] on empty/failed pages — still persisted so a future
                # run short-circuits instead of redoing the work). reading_order
                # is derived in lock-step. The cache readers
                # (pdf_processor._load_cached_layout_for_pages,
                # stage_1_focused_extraction._load_cached_layout,
                # stage_2_chunking.get_layout_from_document_cache_with_status)
                # gate on `processing_version`; that gate now accepts 'paddleocr-vl'.
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
                    "processing_version": "paddleocr-vl",
                    "analysis_metadata": {
                        "stage_1_5": True,
                        "extraction_path": extraction_path,
                        "cache_status": cache_status,
                        "pdf_idx": pdf_idx,
                        "position": position,
                        "region_count": len(layout_elements_payload),
                        "has_text_content": any(
                            (el.get("text_content") or "").strip()
                            for el in layout_elements_payload
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
        # Always clear the slow-op marker once Stage 1.5 finishes — even on
        # exception — so the next stage can set its own marker without colliding
        # and so auto-recovery doesn't keep suppressing recovery on a stalled job.
        # Use the stack-based clear (keyed on the operation we pushed) when a
        # tracker is available; fall back to direct UPDATE otherwise.
        if _stage_1_5_slow_op_set:
            if tracker is not None:
                try:
                    await tracker.clear_slow_operation(operation=_slow_op_key)
                except Exception as _clear_err:
                    logger.debug(f"   tracker.clear_slow_operation failed (non-fatal): {_clear_err}")
            elif job_id:
                try:
                    await asyncio.to_thread(
                        lambda: supabase.client.table('background_jobs')
                            .update({
                                'current_slow_operation': None,
                                'updated_at': datetime.utcnow().isoformat(),
                            })
                            .eq('id', job_id)
                            .execute()
                    )
                except Exception as _clear_err:
                    logger.debug(f"   clear_slow_operation (fallback) failed (non-fatal): {_clear_err}")

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


def build_page_text_from_layout_cache(
    document_id: str,
    supabase: Any,
    logger: logging.Logger,
) -> Optional[str]:
    """Build discovery's page-marked text from the PaddleOCR structural cache.

    Structure-first: the PaddleOCR pass runs before discovery and persists each
    page's reading-order text into ``document_layout_analysis``. This joins that
    text into the same ``--- # Page N ---`` page-marked string discovery expects
    (cleaner + multilingual + layout-ordered vs. raw ``page.get_text()``).

    Returns ``None`` when no PaddleOCR rows exist for the document, so the caller
    falls back to the PyMuPDF text path (robustness, not a parallel pipeline).
    """
    try:
        resp = (
            supabase.client.table("document_layout_analysis")
            .select("page_number, layout_elements, processing_version")
            .eq("document_id", document_id)
            .execute()
        )
    except Exception as e:
        logger.debug(f"   layout-cache text build skipped: {e}")
        return None

    rows = [r for r in (resp.data or []) if r.get("processing_version") == "paddleocr-vl"]
    if not rows:
        return None

    rows.sort(key=lambda r: int(r["page_number"]))
    parts: List[str] = []
    for row in rows:
        elements = row.get("layout_elements") or []
        ordered = sorted(
            (e for e in elements if (e.get("text_content") or "").strip()),
            key=lambda e: (
                e.get("reading_order") if e.get("reading_order") is not None else 1_000_000
            ),
        )
        page_text = "\n".join((e.get("text_content") or "").strip() for e in ordered)
        parts.append(f"\n\n--- # Page {int(row['page_number'])} ---\n\n{page_text}")

    return "\n\n".join(parts) if parts else None


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

    Returns `{physical_page: {regions: [...], cache_status: 'success'|
    'paddleocr_no_text'|'empty_page'|'ocr_failed'|'page_failed'|None}}`.
    cache_status=None means
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
