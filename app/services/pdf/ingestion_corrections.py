"""What the pipeline changed on its own, and how well it went: one `ingestion_corrections` row per kind per run.

`degraded` is done but worse than intended and is never counted as success; `needs_decision` is
something the pipeline would not guess. Both keep the document's audit at needs_review until acknowledged.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

CORRECTED = "corrected"
DEGRADED = "degraded"
NEEDS_DECISION = "needs_decision"
OUTCOMES = (CORRECTED, DEGRADED, NEEDS_DECISION)
MAX_SUBJECTS = 200


@dataclass
class Correction:
    code: str
    outcome: str
    message: str
    count: int
    subjects: List[Any] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)


def _ids(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple, set)) else []


def sourcing_corrections(summary: Dict[str, Any]) -> List[Correction]:
    """Image sourcing's run summary as corrections."""
    out: List[Correction] = []
    matched = summary.get("matched_crops") or 0
    added = summary.get("manufacturer_images") or 0
    if matched or added:
        out.append(Correction("images_from_manufacturer", CORRECTED,
                              f"{added} manufacturer photo(s) added; {matched} low-resolution crop(s) replaced by one",
                              added + matched, detail={"manufacturer_images": added, "matched_crops": matched}))
    restored = _ids(summary.get("restored_ids"))
    without = set(_ids(summary.get("restored_without_gemini")))
    full = [i for i in restored if i not in without]
    if full:
        out.append(Correction("images_repaired", CORRECTED, f"{len(full)} low-resolution crop(s) repaired", len(full), full))
    if without:
        reasons = sorted(set(summary.get("gemini_unavailable_reasons") or [])) or ["unknown"]
        out.append(Correction(
            "images_repaired_without_gemini", DEGRADED,
            f"{len(without)} patterned crop(s) got the plain clean-up because the Gemini repair was unavailable "
            f"({'; '.join(reasons)[:200]}); redo them once it is back", len(without), sorted(without),
            {"reasons": reasons}))
    kept = _ids(summary.get("kept_source_ids"))
    if kept:
        out.append(Correction("images_kept_low_resolution", DEGRADED,
                              f"{len(kept)} crop(s) kept at their low resolution: no manufacturer photo matched and no faithful repair",
                              len(kept), kept))
    shrunk = _ids(summary.get("repairs_downscaled"))
    if shrunk:
        out.append(Correction("repairs_capped_at_upscale_limit", CORRECTED,
                              f"{len(shrunk)} repair(s) were scaled down to the audit's enlargement limit before saving",
                              len(shrunk), shrunk))
    rejected = summary.get("rejected_non_product") or 0
    if rejected:
        out.append(Correction("site_images_rejected", CORRECTED,
                              f"{rejected} image(s) from the maker's site were not this product and were left out", rejected))
    if summary.get("unbilled"):
        out.append(Correction("ai_work_unbilled", NEEDS_DECISION,
                              "Image sourcing ran with nobody to bill: no uploader and no active workspace owner", 1))
    return out


def settle_corrections(summary: Dict[str, Any]) -> List[Correction]:
    """Unlinked-image settling's summary as corrections."""
    out: List[Correction] = []
    by_label = summary.get("linked_by_label") or 0
    by_vision = summary.get("linked_by_vision") or 0
    if by_label or by_vision:
        out.append(Correction("images_linked", CORRECTED,
                              f"{by_label + by_vision} unlinked image(s) linked: {by_label} by printed label, {by_vision} by looking at the page",
                              by_label + by_vision, detail={"by_label": by_label, "by_vision": by_vision}))
    multi = _ids(summary.get("multi_linked"))
    if multi:
        out.append(Correction("room_scenes_linked_to_several_series", CORRECTED,
                              f"{len(multi)} image(s) showing more than one series linked to each of them", len(multi), multi))
    excluded = summary.get("excluded") or {}
    if excluded:
        total = sum(excluded.values())
        out.append(Correction("images_excluded", CORRECTED,
                              f"{total} image(s) excluded from products: " + ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in sorted(excluded.items())),
                              total, detail={"by_reason": dict(excluded)}))
    undecided = summary.get("undecided") or 0
    if undecided:
        out.append(Correction("images_undecided", NEEDS_DECISION,
                              f"{undecided} image(s) could not be linked or excluded" + (
                                  f" ({len(summary.get('failed_pages') or [])} page(s) failed)" if summary.get("failed_pages") else ""),
                              undecided, detail={"failed_pages": (summary.get("failed_pages") or [])[:20]}))
    return out


def outcome_counts(corrections: Sequence[Correction]) -> Dict[str, int]:
    return {o: sum(c.count for c in corrections if c.outcome == o) for o in OUTCOMES}


async def record(db: Any, document_id: str, workspace_id: Optional[str], stage: str,
                 corrections: Sequence[Correction], job_id: Optional[str] = None) -> int:
    """Store the corrections and add their counts to the job. Never raises: reporting must not break the run."""
    db = getattr(db, "client", db)
    rows = [{"workspace_id": workspace_id, "document_id": document_id, "job_id": job_id, "stage": stage,
             "code": c.code, "outcome": c.outcome, "message": c.message[:1000], "count": int(c.count),
             "subjects": list(c.subjects)[:MAX_SUBJECTS], "detail": c.detail} for c in corrections if c.outcome in OUTCOMES]
    if not rows or not workspace_id:
        return 0
    try:
        await asyncio.to_thread(db.table("ingestion_corrections").insert(rows).execute)
    except Exception as exc:
        logger.error("corrections for %s (%s) could not be stored: %s", document_id, stage, exc)
        return 0
    if job_id:
        try:
            await asyncio.to_thread(db.rpc("merge_background_job_metadata", {
                "p_job_id": job_id, "p_metadata": {f"corrections_{stage}": outcome_counts(corrections)}}).execute)
        except Exception as exc:
            logger.warning("corrections for %s could not be added to job %s: %s", document_id, job_id, exc)
    return len(rows)
