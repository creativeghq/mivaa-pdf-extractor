"""Room areas PRINTED on the drawing ("Ε=12,50 τ.μ."), with the room name and enclosing outline."""

from __future__ import annotations

import math
import re
from typing import List, Optional

from app.services.drawings.geometry import (
    PageGeometry,
    TextLine,
    bbox_corners,
    fold,
    norm_bbox,
    norm_point,
    point_in_polygon,
    polygon_area,
)

AREA_RE = re.compile(
    r"(?<![\d.,])(?:(?<![A-ZΑ-Ω])(?:ΕΜΒΑΔΟΝ|ΕΜΒ\.?|AREA|Ε|E)\s*[=:]?\s*)?"
    r"(\d{1,5}(?:[.,]\d{1,3})?)\s*"
    r"(?:Τ\.?\s?Μ\.?|M\s?²|M\s?2|Μ\s?²|Μ\s?2|SQ\.?\s?M)(?![A-ZΑ-Ω0-9])"
)
LETTER_RE = re.compile(r"[A-Za-zΑ-Ωα-ωΆ-Ώά-ώ]")
DIGIT_RE = re.compile(r"\d")
MIN_AREA_M2 = 0.3
MAX_AREA_M2 = 100000.0
LABEL_RADIUS_LINES = 6.0
MAX_LABEL_CHARS = 40


def _is_name(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > MAX_LABEL_CHARS or not LETTER_RE.search(t):
        return False
    if AREA_RE.search(fold(t)):
        return False
    return len(DIGIT_RE.findall(t)) <= len(t) // 3


def _room_name(line: TextLine, prefix: str, lines: List[TextLine]) -> Optional[str]:
    prefix = prefix.strip().strip(" -:=,").strip()
    if _is_name(prefix):
        return prefix
    x0, y0, x1, y1 = line.bbox
    h = max(y1 - y0, 1.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    best, best_d = None, None
    for other in lines:
        if other is line or not _is_name(other.text):
            continue
        ox0, oy0, ox1, oy1 = other.bbox
        ocx, ocy = (ox0 + ox1) / 2, (oy0 + oy1) / 2
        same_row = abs(ocy - cy) < 0.5 * h
        if same_row and ox0 >= cx:
            continue
        if not same_row and ocy > cy:
            continue
        d = math.hypot(ocx - cx, ocy - cy)
        if d > LABEL_RADIUS_LINES * h:
            continue
        if best_d is None or d < best_d:
            best, best_d = other.text.strip(), d
    return best


def printed_rooms(geom: PageGeometry, m_per_pt: Optional[float]) -> List[dict]:
    out: List[dict] = []
    poly_index = []
    for poly in geom.polygons:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        poly_index.append(((min(xs), min(ys), max(xs), max(ys)), poly, polygon_area(poly)))

    for line in geom.lines:
        folded = fold(line.text)
        for m in AREA_RE.finditer(folded):
            raw = m.group(1).replace(",", ".")
            try:
                value = float(raw)
            except ValueError:
                continue
            if not (MIN_AREA_M2 <= value <= MAX_AREA_M2):
                continue
            label_text = line.text[m.start():m.end()].strip()
            name = _room_name(line, line.text[: m.start()], geom.lines)
            x0, y0, x1, y1 = line.bbox
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            label_area = max((x1 - x0) * (y1 - y0), 1.0)

            candidates = [
                (poly, area) for (bx0, by0, bx1, by1), poly, area in poly_index
                if bx0 <= cx <= bx1 and by0 <= cy <= by1 and area >= 4 * label_area
                and point_in_polygon(cx, cy, poly)
            ]
            # A label often sits on a bed or rug block: with a scale, the room is the outline nearest the printed area.
            if candidates and m_per_pt:
                enclosing = min(candidates, key=lambda c: abs(math.log(c[1] * m_per_pt ** 2 / value)))
            else:
                enclosing = min(candidates, key=lambda c: c[1]) if candidates else None

            detail: dict = {"printed_area": value}
            if enclosing:
                poly, area = enclosing
                detail["polygon_area_norm"] = round(area / geom.width ** 2, 8)
                if m_per_pt:
                    detail["polygon_area_m2"] = round(area * m_per_pt ** 2, 2)
                points = [norm_point(p, geom) for p in poly]
            else:
                points = bbox_corners(line.bbox, geom)

            ref = f'Area label "{label_text}"' + (f' under "{name}"' if name else "")
            out.append({
                "kind": "area",
                "element_type": "room",
                "label": name or "Room",
                "source": "printed",
                "value": value,
                "unit": "m2",
                "value_norm": None,
                "geometry": {"points": points},
                "bbox": norm_bbox(line.bbox, geom),
                "confidence": 0.9,
                "source_ref": ref,
                "detail": detail,
            })
    return out
