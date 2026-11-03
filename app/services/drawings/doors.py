"""Door swings: quarter-circle arcs, drawn as one or several cubic beziers."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from app.services.drawings.geometry import PageGeometry, bbox_corners, bbox_of, bezier_point, norm_bbox

RADIUS_TOLERANCE = 0.04
MIN_SWEEP_DEG = 75
MAX_SWEEP_DEG = 105
MIN_RADIUS_M, MAX_RADIUS_M = 0.55, 1.3
MIN_RADIUS_FRAC, MAX_RADIUS_FRAC = 0.002, 0.04
MAX_CURVES = 200_000


def _circle(a, b, c) -> Optional[Tuple[float, float, float]]:
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return ux, uy, math.hypot(ax - ux, ay - uy)


def _arc_piece(curve):
    p0, p3 = curve[0], curve[3]
    mid = bezier_point(curve, 0.5)
    circ = _circle(p0, mid, p3)
    if circ is None:
        return None
    cx, cy, r = circ
    if r < 0.5:
        return None
    samples = [bezier_point(curve, t) for t in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)]
    for x, y in samples:
        if abs(math.hypot(x - cx, y - cy) - r) > RADIUS_TOLERANCE * r:
            return None
    a0 = math.degrees(math.atan2(p0[1] - cy, p0[0] - cx))
    am = math.degrees(math.atan2(mid[1] - cy, mid[0] - cx))
    a3 = math.degrees(math.atan2(p3[1] - cy, p3[0] - cx))
    delta = (a3 - a0 + 540) % 360 - 180
    to_mid = (am - a0 + 540) % 360 - 180
    if delta == 0 or (to_mid > 0) != (delta > 0):
        return None
    start, length = (a0, delta) if delta > 0 else (a3, -delta)
    bins = {int(math.floor(start + k + 0.5)) % 360 for k in range(int(round(length)))}
    return cx, cy, r, bins, samples


def detect_doors(geom: PageGeometry, m_per_pt: Optional[float]) -> List[dict]:
    clusters: list = []
    grid: dict = {}
    cell = 8.0
    for curve in geom.curves[:MAX_CURVES]:
        piece = _arc_piece(curve)
        if piece is None:
            continue
        cx, cy, r, bins, samples = piece
        gx, gy = int(cx // cell), int(cy // cell)
        found = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for idx in grid.get((gx + dx, gy + dy), ()):
                    c = clusters[idx]
                    tol = 0.05 * max(r, c["r"]) + 0.5
                    if math.hypot(c["cx"] - cx, c["cy"] - cy) <= tol and abs(c["r"] - r) <= tol:
                        found = idx
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            clusters.append({"cx": cx, "cy": cy, "r": r, "bins": set(bins), "pts": list(samples)})
            grid.setdefault((gx, gy), []).append(len(clusters) - 1)
        else:
            c = clusters[found]
            c["bins"] |= bins
            c["pts"].extend(samples)

    doors: list = []
    for c in clusters:
        sweep = len(c["bins"])
        if not (MIN_SWEEP_DEG <= sweep <= MAX_SWEEP_DEG):
            continue
        r = c["r"]
        if m_per_pt:
            r_m = r * m_per_pt
            if not (MIN_RADIUS_M <= r_m <= MAX_RADIUS_M):
                continue
        else:
            r_m = None
            if not (MIN_RADIUS_FRAC * geom.width <= r <= MAX_RADIUS_FRAC * geom.width):
                continue
        if any(math.hypot(d["cx"] - c["cx"], d["cy"] - c["cy"]) <= 0.1 * max(r, d["r"]) for d in doors):
            continue
        c["r_m"] = r_m
        c["sweep"] = sweep
        doors.append(c)

    out: List[dict] = []
    for c in doors:
        bb = bbox_of(c["pts"])
        if c["r_m"] is not None:
            label = f"Door ~{round(c['r_m'], 1):.1f} m"
            ref = f"Quarter-circle door swing, radius {c['r_m']:.2f} m"
            confidence = 0.6
        else:
            label, ref, confidence = "Door", "Quarter-circle door swing (no scale)", 0.35
        detail = {"radius_norm": round(c["r"] / geom.width, 6), "sweep_deg": c["sweep"]}
        if c["r_m"] is not None:
            detail["radius_m"] = round(c["r_m"], 3)
        out.append({
            "kind": "count",
            "element_type": "door",
            "label": label,
            "source": "detected",
            "value": 1,
            "unit": "pcs",
            "value_norm": None,
            "geometry": {"points": bbox_corners(bb, geom)},
            "bbox": norm_bbox(bb, geom),
            "confidence": confidence,
            "source_ref": ref,
            "detail": detail,
        })
    return out
