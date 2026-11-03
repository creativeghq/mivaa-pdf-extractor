"""Count by example: find every repeat of the symbol inside a user-drawn box."""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from typing import List, Optional, Sequence

import numpy as np

from app.services.drawings.geometry import PageGeometry, bbox_corners, curve_length, norm_bbox

ANCHOR_TOLERANCE = 0.03
RADIUS_FACTOR = 0.75
MIN_SCORE = 0.85
QUANT_STEPS = 20
MAX_ANCHORS = 20_000
MAX_MATCHES = 2000
NMS_IOU = 0.3
INSIDE_EPS = 0.5


def _items(geom: PageGeometry) -> list:
    items = []
    for x0, y0, x1, y1 in geom.segments:
        length = math.hypot(x1 - x0, y1 - y0)
        if length <= 0:
            continue
        items.append(("l", length, (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), ((x0 + x1) / 2, (y0 + y1) / 2)))
    for c in geom.curves:
        xs = [p[0] for p in c]
        ys = [p[1] for p in c]
        bb = (min(xs), min(ys), max(xs), max(ys))
        items.append(("c", curve_length(c), bb, ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)))
    return items


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _bbox_union(boxes) -> tuple:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def find_similar(geom: PageGeometry, box_norm: Sequence[float]) -> dict:
    bx0, bx1 = sorted((box_norm[0] * geom.width, box_norm[2] * geom.width))
    by0, by1 = sorted((box_norm[1] * geom.height, box_norm[3] * geom.height))
    items = _items(geom)
    example = [
        it for it in items
        if it[2][0] >= bx0 - INSIDE_EPS and it[2][1] >= by0 - INSIDE_EPS
        and it[2][2] <= bx1 + INSIDE_EPS and it[2][3] <= by1 + INSIDE_EPS
    ]
    if not example:
        return {"matches": [], "example_items": 0, "reason": "no_vector_content_in_box"}

    ex_bb = _bbox_union([it[2] for it in example])
    diag = math.hypot(ex_bb[2] - ex_bb[0], ex_bb[3] - ex_bb[1]) or 1.0
    ex_extent = max(ex_bb[2] - ex_bb[0], ex_bb[3] - ex_bb[1]) or 1.0
    quant = diag / QUANT_STEPS
    ex_sig = defaultdict(int)
    for kind, length, _, _ in example:
        ex_sig[(kind, int(round(length / quant)))] += 1
    total = len(example)
    anchor = max(example, key=lambda it: it[1])
    radius = diag * RADIUS_FACTOR
    max_len = diag * 1.05

    by_kind = defaultdict(list)
    for idx, it in enumerate(items):
        by_kind[it[0]].append((it[1], idx))
    for lst in by_kind.values():
        lst.sort()

    wanted = set()
    for kind, q in ex_sig:
        wanted.update({(kind, q - 1), (kind, q), (kind, q + 1)})
    code_of: dict = {}
    codes = np.full(len(items), -1, dtype=np.int64)
    for idx, it in enumerate(items):
        if it[1] <= max_len:
            key = (it[0], int(round(it[1] / quant)))
            if key in wanted:
                codes[idx] = code_of.setdefault(key, len(code_of))
    mids = np.array([it[3] for it in items], dtype=np.float64)
    cells = defaultdict(list)
    for idx in np.nonzero(codes >= 0)[0].tolist():
        cells[(int(mids[idx, 0] // radius), int(mids[idx, 1] // radius))].append(idx)
    grid = {k: np.array(v, dtype=np.int64) for k, v in cells.items()}
    wanted_codes = [
        (count, [code_of[k] for k in ((kind, q), (kind, q - 1), (kind, q + 1)) if k in code_of])
        for (kind, q), count in ex_sig.items()
    ]

    lst = by_kind[anchor[0]]
    keys = [v for v, _ in lst]
    lo = bisect.bisect_left(keys, anchor[1] * (1 - ANCHOR_TOLERANCE))
    hi = bisect.bisect_right(keys, anchor[1] * (1 + ANCHOR_TOLERANCE))
    candidates = [idx for _, idx in lst[lo:hi]]
    capped = len(candidates) > MAX_ANCHORS
    candidates = candidates[:MAX_ANCHORS]

    found = [(1.0, ex_bb)]
    example_ids = {id(it) for it in example}
    consumed = {idx for idx, it in enumerate(items) if id(it) in example_ids}
    need = MIN_SCORE * total
    for idx in candidates:
        if idx in consumed:
            continue
        mx, my = mids[idx]
        gx, gy = int(mx // radius), int(my // radius)
        parts = [grid[c] for c in ((gx + dx, gy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)) if c in grid]
        if not parts:
            continue
        near = np.concatenate(parts)
        dist = np.hypot(mids[near, 0] - mx, mids[near, 1] - my)
        inside = dist <= radius
        if inside.sum() < need:
            continue
        near = near[inside][np.argsort(dist[inside], kind="stable")]
        near_codes = codes[near]
        taken: set = set()
        matched = []
        for count, options in wanted_codes:
            got = 0
            for code in options:
                for pos in np.nonzero(near_codes == code)[0].tolist():
                    if got >= count:
                        break
                    if pos not in taken:
                        taken.add(pos)
                        matched.append(int(near[pos]))
                        got += 1
                if got >= count:
                    break
        score = len(matched) / total
        if score < MIN_SCORE:
            continue
        bb = _bbox_union([items[j][2] for j in matched])
        extent = max(bb[2] - bb[0], bb[3] - bb[1])
        if not (0.6 * ex_extent <= extent <= 1.6 * ex_extent):
            continue
        found.append((score, bb))
        consumed.update(matched)
        if len(found) >= MAX_MATCHES:
            capped = True
            break

    kept: List[tuple] = []
    cell = max(ex_extent * 2, 1.0)
    kept_grid = defaultdict(list)
    for score, bb in [found[0]] + sorted(found[1:], key=lambda f: -f[0]):
        gx, gy = int((bb[0] + bb[2]) / 2 // cell), int((bb[1] + bb[3]) / 2 // cell)
        neighbours = [k for dx in (-1, 0, 1) for dy in (-1, 0, 1) for k in kept_grid.get((gx + dx, gy + dy), ())]
        if all(_iou(bb, k) <= NMS_IOU for k in neighbours):
            kept.append((score, bb))
            kept_grid[(gx, gy)].append(bb)

    return {
        "matches": [
            {"bbox": norm_bbox(bb, geom), "geometry": {"points": bbox_corners(bb, geom)}, "score": round(score, 3)}
            for score, bb in kept
        ],
        "example_items": total,
        "reason": "capped" if capped else None,
    }


def page_reason(kind: str) -> Optional[str]:
    return None if kind == "vector" else "not_a_vector_page"
