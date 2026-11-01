from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from app.services.pdf.table_repair import assess_elements_offline
from app.services.pdf.text_layer_extractor import ROUTE_TEXT_LAYER, TEXT_LAYER_VERSION

PAGE_BATCH = 200


def _rows(response: Any) -> List[Dict[str, Any]]:
    return list(getattr(response, "data", None) or [])


def stale_pages(layout_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in layout_rows if "route" not in (r.get("analysis_metadata") or {})]


def decide(route: str, assessment: Dict[str, Any], planned: Optional[List[Dict[str, Any]]]) -> str:
    if route == ROUTE_TEXT_LAYER and planned and assessment.get("truncated"):
        return "replace_with_text_layer"
    return "annotate"


def _clip_origin(clip: Any) -> tuple:
    return (clip.x0, clip.y0) if clip is not None else (0.0, 0.0)


async def upgrade_cached_layout(document_id: str, supabase: Any, *, apply: bool = False) -> Dict[str, Any]:
    from app.api.pdf_processing.stage_1_layout_precompute import PDF_POINTS_TO_PIXEL_ZOOM, _plan_page
    import fitz

    log = logging.getLogger(__name__)
    db = supabase.client
    doc_rows = _rows(await asyncio.to_thread(db.table("documents").select("storage_bucket, storage_object_path").eq("id", document_id).execute))
    if not doc_rows or not doc_rows[0].get("storage_object_path"):
        return {"error": "source_not_kept", "applied": False}
    blob = await asyncio.to_thread(db.storage.from_(doc_rows[0]["storage_bucket"]).download, doc_rows[0]["storage_object_path"])
    layout: List[Dict[str, Any]] = []
    start = 0
    while True:
        batch = _rows(await asyncio.to_thread(
            db.table("document_layout_analysis").select("page_number, layout_elements, analysis_metadata, processing_version")
            .eq("document_id", document_id).order("page_number").range(start, start + PAGE_BATCH - 1).execute
        ))
        layout.extend(batch)
        if len(batch) < PAGE_BATCH:
            break
        start += PAGE_BATCH
    todo = stale_pages(layout)
    report: Dict[str, Any] = {"pages": len(layout), "stale": len(todo), "annotated": 0, "replaced": [], "flagged": [], "applied": apply}
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(blob)
        path = handle.name
    try:
        pdf = fitz.open(path)
        for row in todo:
            meta = dict(row.get("analysis_metadata") or {})
            number = int(row["page_number"])
            pdf_idx, position = meta.get("pdf_idx", number - 1), meta.get("position", "single")
            try:
                decision, content, clip, planned = await asyncio.to_thread(_plan_page, pdf, number, pdf_idx, position)
            except Exception as exc:
                log.warning("layout upgrade: page %s could not be planned: %s", number, exc)
                continue
            elements = list(row.get("layout_elements") or [])
            reference = {"words": content["words"], "origin": _clip_origin(clip)} if content.get("words") else None
            checks = assess_elements_offline(elements, reference, PDF_POINTS_TO_PIXEL_ZOOM)
            notes = {"route": decision.route, "route_reasons": decision.reasons, "text_layer_stats": decision.stats, "table_checks": checks, "upgraded": True}
            action = decide(decision.route, checks, planned)
            needs_review = bool(checks["suspect"] or checks["truncated"])
            if action == "replace_with_text_layer":
                report["replaced"].append(number)
                payload = {"layout_elements": planned, "processing_version": TEXT_LAYER_VERSION,
                           "analysis_metadata": {**meta, **notes, "extraction_path": "text_layer", "cache_status": "success", "needs_review": False, "replaced_truncated_ocr": True}}
            else:
                if needs_review:
                    report["flagged"].append({"page": number, "truncated": checks["truncated"], "suspect": checks["suspect"]})
                payload = {"layout_elements": elements, "analysis_metadata": {**meta, **notes, "needs_review": needs_review}}
            report["annotated"] += 1
            if apply:
                await asyncio.to_thread(db.table("document_layout_analysis").update(payload).eq("document_id", document_id).eq("page_number", number).execute)
        pdf.close()
    finally:
        os.unlink(path)
    return report
