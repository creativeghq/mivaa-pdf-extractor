from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from app.services.pdf.ocr_completeness import (
    COMPLETE,
    HARD_REASONS,
    SUSPECT,
    TRUNCATED,
    assess_table,
    ink_profile_of,
    plan_bands,
    stitch_bands,
    text_line_bands,
)
from app.services.pdf.text_layer_extractor import reference_numbers

STATUS_RANK = {COMPLETE: 0, SUSPECT: 1, TRUNCATED: 2}
REPAIR_MIN_ROWS = 4
REPAIR_MAX_TABLES_PER_PAGE = 3
REPAIR_MAX_BANDS = 6
ADOPT_MIN_ROW_GAIN = 1.1

_budget_lock = threading.Lock()


def table_contents(regions: List[Any]) -> str:
    return "".join(r.content for r in regions if getattr(r, "region_type", "") == "TABLE" and r.content)


def repair_with_bands(manager: Any, crop: Any, physical_page: int, job_id: Optional[str], document_id: str) -> Optional[str]:
    bands = text_line_bands(ink_profile_of(crop))
    cuts = plan_bands(bands, crop.height)
    if len(cuts) < 2 or len(cuts) > REPAIR_MAX_BANDS:
        return None
    parts: List[str] = []
    for y0, y1 in cuts:
        result = manager.run_structural_pass(
            crop.crop((0, y0, crop.width, y1)), "stage_1_5_table_band", physical_page, job_id, document_id,
        )
        parts.append(table_contents(result.get("regions") or []))
    stitched, _reasons = stitch_bands(parts)
    return stitched


def should_adopt(before: Any, after: Any) -> bool:
    if after.rows < before.rows * ADOPT_MIN_ROW_GAIN:
        return False
    if STATUS_RANK[after.status] < STATUS_RANK[before.status]:
        return True
    return (
        before.missing_share is not None
        and after.missing_share is not None
        and after.missing_share < before.missing_share * 0.5
    )


def verify_ocr_tables(
    manager: Any,
    image: Any,
    elements: List[Dict[str, Any]],
    reference: Optional[Dict[str, Any]],
    physical_page: int,
    job_id: Optional[str],
    document_id: str,
    zoom: float,
    repair: bool,
    budget: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"checked": 0, "complete": 0, "suspect": 0, "truncated": 0, "recovered": 0, "repair_attempts": 0}
    words = reference["words"] if reference else None
    ox, oy = reference["origin"] if reference else (0.0, 0.0)
    for el in elements:
        if el.get("region_type") != "TABLE":
            continue
        meta = el.setdefault("metadata", {})
        b = el["bbox"]
        box = (
            max(0, int(b["x"])), max(0, int(b["y"])),
            min(image.width, int(b["x"] + b["width"])), min(image.height, int(b["y"] + b["height"])),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        crop = image.crop(box)
        lines = len(text_line_bands(ink_profile_of(crop)))
        ref = None
        if words:
            ref = reference_numbers(words, (box[0] / zoom + ox, box[1] / zoom + oy, box[2] / zoom + ox, box[3] / zoom + oy)) or None
        content = meta.get("html") or el.get("text_content") or ""
        capped = bool(meta.get("truncation_suspected"))
        verdict = assess_table(content, visible_lines=lines, reference_numbers=ref, generation_capped=capped)
        report["checked"] += 1
        if (
            repair
            and verdict.status == TRUNCATED
            and set(verdict.reasons) & HARD_REASONS
            and verdict.rows >= REPAIR_MIN_ROWS
            and report["repair_attempts"] < REPAIR_MAX_TABLES_PER_PAGE
            and _take_budget(budget)
        ):
            report["repair_attempts"] += 1
            stitched = None
            try:
                stitched = repair_with_bands(manager, crop, physical_page, job_id, document_id)
            except Exception as exc:
                meta["repair_error"] = f"{type(exc).__name__}: {exc}"
            if stitched:
                again = assess_table(stitched, visible_lines=lines, reference_numbers=ref)
                if should_adopt(verdict, again):
                    meta["html"] = stitched
                    el["text_content"] = stitched
                    meta["recovered_by"] = "band_rerun"
                    verdict = again
                    report["recovered"] += 1
        meta.update(verdict.as_metadata())
        meta["needs_review"] = verdict.status != COMPLETE
        report[verdict.status] += 1
    return report


def _take_budget(budget: Optional[Dict[str, int]]) -> bool:
    if budget is None:
        return True
    with _budget_lock:
        if budget.get("left", 0) <= 0:
            return False
        budget["left"] -= 1
        return True
