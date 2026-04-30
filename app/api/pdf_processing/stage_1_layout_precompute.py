"""
Stage 1.5: Document-level layout precompute.

Runs YOLO layout detection + bbox-text merge (PyMuPDF spans → Chandra
v2 OCR fallback) ONCE per page, persisting merged regions to
`document_layout_analysis` keyed on (document_id, page_number).

Stage 2 (chunking) reads these cached regions instead of running
per-product YOLO+OCR. This is what unblocks layout-aware chunking
end-to-end (every per-product chunker call now sees regions with
text_content populated).

Resume-safe: pages with cache rows are skipped, so an interrupted
job resumes only the uncached pages.

Toggle: `LAYOUT_PRECOMPUTE_ENABLED` env var (defaults True). When
False, the orchestrator skips this stage and the chunker falls back
to its text-based path (round-14 fix in unified_chunking_service).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Set


async def precompute_document_layout(
    document_id: str,
    pdf_path: str,
    total_pages: int,
    supabase: Any,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run YOLO + merge_layout for every uncached page; persist results.

    Args:
        document_id: documents.id (FK to document_layout_analysis.document_id)
        pdf_path: local path to the PDF on disk
        total_pages: physical page count (1-based range)
        supabase: supabase client (writer)
        logger: scoped logger

    Returns:
        Dict with keys: pages_total, pages_cached_already, pages_processed,
        pages_persisted, extraction_paths (counter dict).
    """
    summary: Dict[str, Any] = {
        "pages_total": total_pages,
        "pages_cached_already": 0,
        "pages_processed": 0,
        "pages_persisted": 0,
        "extraction_paths": {},
    }

    # 1. Look up existing cache entries (resume-safe)
    existing_pages: Set[int] = set()
    try:
        resp = (
            supabase.client.table("document_layout_analysis")
            .select("page_number")
            .eq("document_id", document_id)
            .execute()
        )
        existing_pages = {row["page_number"] for row in (resp.data or [])}
    except Exception as e:
        logger.warning(f"   ⚠️ document_layout_analysis cache lookup failed (continuing): {e}")

    summary["pages_cached_already"] = len(existing_pages)
    pages_to_process = [p for p in range(1, total_pages + 1) if p not in existing_pages]

    if not pages_to_process:
        logger.info(
            f"♻️ [STAGE 1.5] All {total_pages} pages already cached in "
            f"document_layout_analysis — skipping precompute"
        )
        return summary

    logger.info(
        f"📐 [STAGE 1.5] Precomputing layout for {len(pages_to_process)} page(s); "
        f"{len(existing_pages)} already cached"
    )

    # 2. Run YOLO + merge in worker threads (sync code)
    from app.services.pdf.pdf_worker import (
        _yolo_all_pages_sync,
        _build_layout_regions_by_page,
    )

    try:
        yolo_regions_by_page = await asyncio.to_thread(
            _yolo_all_pages_sync, pdf_path, total_pages
        )
    except Exception as e:
        logger.warning(
            f"   ⚠️ [STAGE 1.5] YOLO batch failed ({e}) — chunker will fall back to text-only"
        )
        return summary

    # Filter to only the pages we still need
    yolo_for_uncached = {
        p: yolo_regions_by_page.get(p, []) for p in pages_to_process if p in yolo_regions_by_page
    }

    if not yolo_for_uncached:
        logger.info(
            "   ℹ️ [STAGE 1.5] No YOLO regions detected on uncached pages — "
            "writing empty cache entries to short-circuit future jobs"
        )

    try:
        layout_regions_by_page, extraction_paths = await asyncio.to_thread(
            _build_layout_regions_by_page, pdf_path, yolo_for_uncached
        )
    except Exception as e:
        logger.warning(
            f"   ⚠️ [STAGE 1.5] merge_layout pass failed ({e}) — chunker will fall back to text-only"
        )
        return summary

    summary["extraction_paths"] = extraction_paths
    summary["pages_processed"] = len(pages_to_process)

    # 3. Persist per-page merged regions. Upsert on (document_id, page_number)
    #    is safe against the unique index already declared on the table.
    persisted = 0
    for page_num in pages_to_process:
        regions = layout_regions_by_page.get(page_num, [])
        try:
            payload = {
                "document_id": document_id,
                "page_number": page_num,
                "layout_elements": [r.to_dict() for r in regions] if regions else [],
                "analysis_metadata": {
                    "stage_1_5": True,
                    "region_count": len(regions),
                    "has_text_content": any(getattr(r, "text_content", "").strip() for r in regions),
                },
            }
            (
                supabase.client.table("document_layout_analysis")
                .upsert(payload, on_conflict="document_id,page_number")
                .execute()
            )
            persisted += 1
        except Exception as e:
            logger.warning(f"   ⚠️ [STAGE 1.5] cache write failed for page {page_num}: {e}")

    summary["pages_persisted"] = persisted
    logger.info(
        f"✅ [STAGE 1.5] Cached {persisted}/{len(pages_to_process)} pages — "
        f"extraction paths: {extraction_paths}"
    )
    return summary


async def get_layout_from_document_cache(
    document_id: str,
    physical_pages: List[int],
    supabase: Any,
    logger: logging.Logger,
) -> Dict[int, List[Dict[str, Any]]]:
    """Read cached merged regions for a subset of pages.

    Used by Stage 2 chunking to look up Stage 1.5 results. Returns
    `{page_number: [region_dict, ...]}` where each region_dict has the
    same shape `unified_chunking_service` already accepts (`region_type`,
    `text_content`, `bbox`, `reading_order`).

    Empty dict on cache miss; the caller falls back to in-memory or
    product-level regions.
    """
    if not document_id or not physical_pages:
        return {}

    try:
        resp = (
            supabase.client.table("document_layout_analysis")
            .select("page_number, layout_elements")
            .eq("document_id", document_id)
            .in_("page_number", physical_pages)
            .execute()
        )
    except Exception as e:
        logger.debug(f"   document_layout_analysis read failed: {e}")
        return {}

    rows = resp.data or []
    out: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        regions = row.get("layout_elements") or []
        if isinstance(regions, list) and regions:
            out[row["page_number"]] = regions

    if out:
        logger.info(
            f"   ♻️ Loaded {sum(len(v) for v in out.values())} merged regions "
            f"from document_layout_analysis cache ({len(out)}/{len(physical_pages)} pages)"
        )
    return out
