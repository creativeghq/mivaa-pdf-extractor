"""Whole-file drawing analysis: per-page kind, scale, fingerprint and quantity proposals."""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence

from app.services.drawings.doors import detect_doors
from app.services.drawings.geometry import PageGeometry, extract_page, fitz
from app.services.drawings.rooms import printed_rooms
from app.services.drawings.scale import find_scale, is_plausible
from app.services.drawings.similar import find_similar, page_reason
from app.services.drawings.walls import detect_walls

VECTOR_MIN_ITEMS = 150
SCAN_MIN_COVERAGE = 0.6
FINGERPRINT_GRID = 16


def classify(geom: PageGeometry) -> str:
    if geom.too_complex:
        return "vector"
    if geom.image_coverage >= SCAN_MIN_COVERAGE:
        return "scan"
    if geom.item_count >= VECTOR_MIN_ITEMS:
        return "vector"
    return "empty"


def fingerprint(geom: PageGeometry) -> List[str]:
    if geom.too_complex:
        return []
    n = FINGERPRINT_GRID
    cells: List[List[str]] = [[] for _ in range(n * n)]
    w, h = geom.width or 1.0, geom.height or 1.0

    def _cell_add(x: float, y: float, token: str) -> None:
        i = min(n - 1, max(0, int(x / w * n)))
        j = min(n - 1, max(0, int(y / h * n)))
        cells[j * n + i].append(token)

    for x0, y0, x1, y1 in geom.segments:
        _cell_add((x0 + x1) / 2, (y0 + y1) / 2, f"l{x0 / w:.3f},{y0 / h:.3f},{x1 / w:.3f},{y1 / h:.3f}")
    for c in geom.curves:
        (x0, y0), (x3, y3) = c[0], c[3]
        pts = ",".join(f"{x / w:.3f},{y / h:.3f}" for x, y in c)
        _cell_add((x0 + x3) / 2, (y0 + y3) / 2, "c" + pts)
    for line in geom.lines:
        b = line.bbox
        _cell_add((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, "t" + line.text.strip())
    digests = geom.image_digests + [""] * (len(geom.image_bboxes) - len(geom.image_digests))
    for b, d in zip(geom.image_bboxes, digests):
        _cell_add((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, f"i{b[0] / w:.2f},{b[1] / h:.2f},{b[2] / w:.2f},{b[3] / h:.2f},{d}")

    out = []
    for tokens in cells:
        if not tokens:
            out.append("0")
            continue
        digest = hashlib.blake2b("\n".join(sorted(tokens)).encode("utf-8"), digest_size=4).hexdigest()
        out.append(digest)
    return out


def analyze_page(page, units_per_page_width: Optional[float]) -> dict:
    geom = extract_page(page)
    kind = classify(geom)
    notes: List[str] = []
    scale, scale_notes = find_scale(geom.lines, geom.width)
    notes.extend(scale_notes)

    upw: Optional[float] = None
    if units_per_page_width and units_per_page_width > 0:
        upw = float(units_per_page_width)
        notes.append("using the calibrated scale")
    elif scale and scale["plausible"]:
        upw = scale["units_per_page_width"]
    elif scale:
        notes.append(f"scale 1:{scale['ratio']} gives an implausible page width; ignored")
    m_per_pt = upw / geom.width if upw and geom.width > 0 and is_plausible(upw) else None

    proposals: List[dict] = []
    if geom.lines:
        proposals.extend(printed_rooms(geom, m_per_pt))
    if geom.too_complex:
        notes.append("too many drawn lines to read on this page; measure it by hand")
    elif kind == "vector":
        proposals.extend(detect_doors(geom, m_per_pt))
        if m_per_pt:
            walls, wall_notes = detect_walls(geom, m_per_pt)
            proposals.extend(walls)
            notes.extend(wall_notes)
        else:
            notes.append("walls need a scale")

    return {
        "page": page.number + 1,
        "width_pt": round(geom.width, 3),
        "height_pt": round(geom.height, 3),
        "aspect": round(geom.width / geom.height, 6) if geom.height else 0.0,
        "kind": kind,
        "path_count": geom.item_count,
        "text_chars": geom.text_chars,
        "image_coverage": round(geom.image_coverage, 4),
        "scale": scale,
        "fingerprint": fingerprint(geom),
        "notes": notes,
        "proposals": proposals,
    }


def _open(data: bytes):
    try:
        return fitz.open(stream=data, filetype="pdf")
    except Exception as e:  # noqa: BLE001 — any parse failure is "not a readable PDF"
        raise ValueError(f"not a readable PDF: {str(e)[:200]}") from e


def analyze_pdf(data: bytes, max_pages: int = 60, units_per_page_width: Optional[float] = None) -> dict:
    doc = _open(data)
    try:
        count = doc.page_count
        limit = max(1, min(int(max_pages), count)) if count else 0
        pages = [analyze_page(doc[i], units_per_page_width) for i in range(limit)]
        return {"page_count": count, "truncated": count > limit, "pages": pages}
    finally:
        doc.close()


def find_similar_in_pdf(data: bytes, page: int, bbox: Sequence[float]) -> dict:
    doc = _open(data)
    try:
        if page < 1 or page > doc.page_count:
            raise ValueError(f"page {page} is out of range; the file has {doc.page_count} pages")
        geom = extract_page(doc[page - 1], with_text=False)
        reason = "too_complex" if geom.too_complex else page_reason(classify(geom))
        if reason:
            return {"matches": [], "example_items": 0, "reason": reason}
        return find_similar(geom, bbox)
    finally:
        doc.close()
