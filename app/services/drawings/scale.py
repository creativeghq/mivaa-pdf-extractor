"""The printed drawing scale ("ΚΛΙΜΑΚΑ 1:50", "SCALE 1/100") read from the text layer."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Optional, Tuple

from app.services.drawings.geometry import METRES_PER_PT, TextLine, fold

RATIO_RE = re.compile(r"(?<![\d.,/:])1\s*[:/]\s*(\d{1,4})(?![\d.,/:])")
KEYWORD_RE = re.compile(r"ΚΛΙΜΑΚΑ|ΚΛΙΜΑΞ|ΚΛ\.|SCALE|ΚΛΙΜ\.")
COMMON_SCALES = {1, 2, 5, 10, 20, 25, 50, 75, 100, 200, 250, 500, 1000, 2000}
MAX_RATIO = 2000
MIN_REAL_WIDTH_M = 0.5
MAX_REAL_WIDTH_M = 5000.0


def units_per_page_width(width_pt: float, ratio: float) -> float:
    return width_pt * METRES_PER_PT * ratio


def is_plausible(units: float) -> bool:
    return MIN_REAL_WIDTH_M <= units <= MAX_REAL_WIDTH_M


def find_scale(lines: List[TextLine], width_pt: float) -> Tuple[Optional[dict], List[str]]:
    keyed: List[Tuple[int, str]] = []
    loose: List[Tuple[int, str]] = []
    keyword_boxes = [ln.bbox for ln in lines if KEYWORD_RE.search(fold(ln.text))]

    for ln in lines:
        folded = fold(ln.text)
        for m in RATIO_RE.finditer(folded):
            n = int(m.group(1))
            if n < 1 or n > MAX_RATIO:
                continue
            text = ln.text.strip()[:80]
            if KEYWORD_RE.search(folded) or _near_keyword(ln.bbox, keyword_boxes):
                keyed.append((n, text))
            elif n in COMMON_SCALES and n > 1:
                loose.append((n, text))

    pool = keyed or loose
    if not pool:
        return None, []
    counts = Counter(n for n, _ in pool)
    ratio, _ = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    text = next(t for n, t in pool if n == ratio)
    notes: List[str] = []
    if len(counts) > 1:
        listed = ", ".join(f"1:{n}" for n in sorted(counts))
        notes.append(f"{len(counts)} scale texts disagree: {listed}")
    units = units_per_page_width(width_pt, ratio)
    return (
        {"ratio": ratio, "text": text, "units_per_page_width": round(units, 6), "plausible": is_plausible(units)},
        notes,
    )


def _near_keyword(bbox, keyword_boxes) -> bool:
    h = max(bbox[3] - bbox[1], 1.0)
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    for kb in keyword_boxes:
        kx, ky = (kb[0] + kb[2]) / 2, (kb[1] + kb[3]) / 2
        if abs(cy - ky) <= 3 * h and math.hypot(cx - kx, cy - ky) <= 12 * h + (kb[2] - kb[0]):
            return True
    return False
