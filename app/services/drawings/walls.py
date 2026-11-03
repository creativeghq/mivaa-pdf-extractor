"""Walls: pairs of parallel lines a wall-thickness apart, bucketed by thickness."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Tuple

import numpy as np

from app.services.drawings.geometry import PageGeometry, norm_bbox, norm_point

MIN_LENGTH_M = 0.3
MIN_THICKNESS_M, MAX_THICKNESS_M = 0.07, 0.45
MAX_ANGLE_DEG = 1.0
MIN_OVERLAP = 0.5
THICKNESS_STEP_M = 0.05
MAX_SEGMENTS = 150_000
MAX_COMPARISONS = 12_000_000
CHUNK_PAIRS = 1_000_000
MAX_OUTPUT_SEGMENTS = 1500
DUPLICATE_EPS_M = 0.01
# A face may pair with several opposite segments, but no stretch of it is counted twice.
SHARED_FACE = 0.1


def _span(seg: np.ndarray, px: float, py: float, qx: float, qy: float) -> tuple:
    """Where a pair's overlap lies along one of its faces, as a (start, end) distance from its first end."""
    dx, dy = seg[2] - seg[0], seg[3] - seg[1]
    n = math.hypot(dx, dy) or 1.0
    a = ((px - seg[0]) * dx + (py - seg[1]) * dy) / n
    b = ((qx - seg[0]) * dx + (qy - seg[1]) * dy) / n
    return (min(a, b), max(a, b))


def _claimed_length(spans: list, sp: tuple) -> float:
    return sum(max(0.0, min(sp[1], e) - max(sp[0], s)) for s, e in spans)


def _candidate_pairs(segs: np.ndarray, angles: np.ndarray, lengths: np.ndarray, tmin: float, tmax: float):
    """Every (i, j, overlap, gap, centreline) pair that passes the wall tests; capped by work done."""
    buckets = angles.astype(np.int64) % 180
    by_bucket = [np.nonzero(buckets == b)[0] for b in range(180)]
    found = []
    budget = MAX_COMPARISONS
    capped = False
    for b in range(180):
        members = np.concatenate([by_bucket[b], by_bucket[(b + 1) % 180]])
        if members.size < 2:
            continue
        ref = math.radians(b + 1)
        ux, uy = math.cos(ref), math.sin(ref)
        nx, ny = -uy, ux
        s = segs[members]
        rho = nx * (s[:, 0] + s[:, 2]) / 2 + ny * (s[:, 1] + s[:, 3]) / 2
        ta, tb = ux * s[:, 0] + uy * s[:, 1], ux * s[:, 2] + uy * s[:, 3]
        order = np.argsort(rho, kind="stable")
        members, rho = members[order], rho[order]
        t0, t1 = np.minimum(ta, tb)[order], np.maximum(ta, tb)[order]
        n = members.size
        lo = np.maximum(np.searchsorted(rho, rho + tmin, "left"), np.arange(1, n + 1))
        hi = np.searchsorted(rho, rho + tmax, "right")
        counts = np.clip(hi - lo, 0, None)
        total = int(counts.sum())
        if total > budget:
            keep = np.cumsum(counts) <= budget
            counts = np.where(keep, counts, 0)
            capped = True
        budget -= int(counts.sum())
        rows = np.nonzero(counts)[0]
        cum = np.cumsum(counts[rows])
        cuts = np.searchsorted(cum, np.arange(CHUNK_PAIRS, int(cum[-1]) + CHUNK_PAIRS, CHUNK_PAIRS), "right") if rows.size else []
        start = 0
        for stop in cuts:
            stop = max(int(stop), start + 1)
            r = rows[start:stop]
            start = stop
            if r.size == 0:
                break
            c = counts[r]
            a = np.repeat(r, c)
            offsets = np.arange(a.size) - np.repeat(np.cumsum(c) - c, c)
            j = np.repeat(lo[r], c) + offsets
            gi, gj = members[a], members[j]
            dang = np.abs(angles[gi] - angles[gj])
            dang = np.minimum(dang, 180.0 - dang)
            low, high = np.maximum(t0[a], t0[j]), np.minimum(t1[a], t1[j])
            overlap = high - low
            ok = (dang < MAX_ANGLE_DEG) & (overlap >= MIN_OVERLAP * np.minimum(lengths[gi], lengths[gj]))
            if not ok.any():
                continue
            sel = np.nonzero(ok)[0]
            ka, kj = a[sel], j[sel]
            mid = (rho[kj] + rho[ka]) / 2
            lo_k, hi_k = low[sel], high[sel]
            found.append(np.column_stack([
                gi[sel], gj[sel], overlap[sel], rho[kj] - rho[ka],
                ux * lo_k + nx * mid, uy * lo_k + ny * mid, ux * hi_k + nx * mid, uy * hi_k + ny * mid,
            ]))
        if capped:
            break
    if not found:
        return np.empty((0, 8)), capped
    return np.concatenate(found), capped


def _drop_covered(segs: np.ndarray, lengths: np.ndarray, eps: float) -> np.ndarray:
    """Indices to keep: a line lying on top of a longer collinear one (a room outline on a wall face) is a copy."""
    angles = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180.0
    rad = np.radians(angles)
    ux, uy = np.cos(rad), np.sin(rad)
    rho = -uy * (segs[:, 0] + segs[:, 2]) / 2 + ux * (segs[:, 1] + segs[:, 3]) / 2
    groups: dict = defaultdict(list)
    for i in np.argsort(-lengths, kind="stable").tolist():
        groups[(int(round(angles[i] * 2)) % 360, int(round(rho[i] / eps)))].append(i)
    keep = []
    for (ab, rb), members in groups.items():
        kept: list = []
        for i in members:
            a = ux[i] * segs[i, 0] + uy[i] * segs[i, 1]
            b = ux[i] * segs[i, 2] + uy[i] * segs[i, 3]
            t0, t1 = min(a, b), max(a, b)
            covered = 0.0
            for k0, k1 in kept:
                covered += max(0.0, min(t1, k1) - max(t0, k0))
            if covered >= 0.9 * (t1 - t0):
                continue
            kept.append((t0, t1))
            keep.append(i)
    return np.asarray(sorted(keep), dtype=np.int64)


def detect_walls(geom: PageGeometry, m_per_pt: float) -> Tuple[List[dict], List[str]]:
    notes: List[str] = []
    if not geom.segments:
        return [], notes
    min_len = MIN_LENGTH_M / m_per_pt
    tmin, tmax = MIN_THICKNESS_M / m_per_pt, MAX_THICKNESS_M / m_per_pt

    segs = np.asarray(geom.segments, dtype=np.float64)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    keep = lengths >= min_len
    segs, lengths = segs[keep], lengths[keep]
    if segs.shape[0] > MAX_SEGMENTS:
        top = np.argsort(-lengths, kind="stable")[:MAX_SEGMENTS]
        segs, lengths = segs[top], lengths[top]
        notes.append(f"wall search used the {MAX_SEGMENTS} longest lines only")
    if segs.shape[0] < 2:
        return [], notes
    keep = _drop_covered(segs, lengths, DUPLICATE_EPS_M / m_per_pt)
    segs, lengths = segs[keep], lengths[keep]
    if segs.shape[0] < 2:
        return [], notes
    angles =np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180.0

    found, capped = _candidate_pairs(segs, angles, lengths, tmin, tmax)
    if capped:
        notes.append("wall search stopped early: too many parallel lines on this page")

    groups: dict = defaultdict(lambda: {"length": 0.0, "segments": []})
    if found.shape[0]:
        i_col, j_col = found[:, 0].astype(np.int64), found[:, 1].astype(np.int64)
        key = np.minimum(i_col, j_col) * segs.shape[0] + np.maximum(i_col, j_col)
        _, first = np.unique(key, return_index=True)
        found = found[first]
        found = found[np.argsort(-found[:, 2], kind="stable")]
        claimed: dict = defaultdict(list)
        for i, j, overlap, gap, px, py, qx, qy in found.tolist():
            spans = [(k, _span(segs[int(k)], px, py, qx, qy)) for k in (i, j)]
            if any(_claimed_length(claimed[k], sp) > SHARED_FACE * (sp[1] - sp[0]) for k, sp in spans):
                continue
            for k, sp in spans:
                claimed[k].append(sp)
            cm = int(round(round(gap * m_per_pt / THICKNESS_STEP_M) * THICKNESS_STEP_M * 100))
            g = groups[cm]
            g["length"] += overlap
            g["segments"].append(((px, py), (qx, qy)))

    out: List[dict] = []
    for cm in sorted(groups, key=lambda k: -groups[k]["length"]):
        g = groups[cm]
        all_pts = [pt for seg in g["segments"] for pt in seg]
        xs = [pt[0] for pt in all_pts]
        ys = [pt[1] for pt in all_pts]
        bb = (min(xs), min(ys), max(xs), max(ys))
        length_norm = g["length"] / geom.width
        out.append({
            "kind": "length",
            "element_type": "wall",
            "label": f"Wall ~{cm} cm",
            "source": "detected",
            "value": None,
            "unit": "m",
            "value_norm": round(length_norm, 6),
            "geometry": {"segments": [[norm_point(p, geom), norm_point(q, geom)] for p, q in g["segments"][:MAX_OUTPUT_SEGMENTS]]},
            "bbox": norm_bbox(bb, geom),
            "confidence": 0.4,
            "source_ref": f"{len(g['segments'])} parallel line pairs about {cm} cm apart",
            "detail": {"thickness_cm": cm, "pairs": len(g["segments"]), "length_m": round(g["length"] * m_per_pt, 2)},
        })
    return out, notes
