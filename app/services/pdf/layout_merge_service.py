"""
Layout Merge Service - merges Chandra v2 bbox fragments with YOLO regions.

YOLO produces typed page regions (TEXT / TABLE / TITLE / CAPTION / IMAGE)
with reading-order indices. Chandra v2 produces word/line-level text
fragments with pixel coordinates. This module unites them: each Chandra
fragment is assigned to the YOLO region that contains it (greater-overlap
wins on ties), and each region's `text_content` is filled in by joining
its assigned fragments in reading order.

Pure function - no I/O, no HTTP, no database. Trivially unit-testable.

Output is consumed by:
- The chunker's `_chunk_with_layout_regions` path (reads `text_content`)
- `product_layout_regions` table writes (each row stores `text_content`)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum fraction of a fragment's area that must lie inside a region for
# it to be considered "contained". Fragments below this threshold for every
# region are treated as orphans and grouped under UNCLASSIFIED.
MIN_CONTAINMENT_RATIO = 0.30

UNCLASSIFIED_REGION_TYPE = "UNCLASSIFIED"


@dataclass
class MergedRegion:
    """A YOLO region enriched with Chandra v2 text fragments."""

    region_type: str
    bbox: Dict[str, float]                       # {"x", "y", "width", "height", "page"}
    confidence: float
    reading_order: Optional[int]
    text_content: str                            # joined fragment text in reading order
    fragments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_type": self.region_type,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "reading_order": self.reading_order,
            "text_content": self.text_content,
            "fragments": self.fragments,
            "metadata": self.metadata,
        }


def _fragment_rect(fragment: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Return (x, y, w, h) for a Chandra fragment, or None if unusable.

    Chandra v2 emits {x, y, w, h} per fragment. Some fragments may omit
    width/height when the model considers the bbox a single point - in
    that case we fall back to a 1x1 rect at (x, y) so the point still
    gets matched by greater-overlap to the enclosing region.
    """
    try:
        x = float(fragment.get("x", 0))
        y = float(fragment.get("y", 0))
    except (TypeError, ValueError):
        return None
    try:
        w = float(fragment.get("w") or fragment.get("width") or 1.0)
        h = float(fragment.get("h") or fragment.get("height") or 1.0)
    except (TypeError, ValueError):
        w, h = 1.0, 1.0
    if w <= 0:
        w = 1.0
    if h <= 0:
        h = 1.0
    return (x, y, w, h)


def _region_rect(region: Any) -> Optional[Tuple[float, float, float, float]]:
    """Return (x, y, w, h) for a YOLO region in pixel space.

    Accepts both `LayoutRegion` pydantic objects and dict-shaped regions
    that have already been serialized (e.g. fetched from the database).
    Coordinates may be either pixel-space or 0-1 normalized; the merge
    function callers must pass `page_size` if regions are normalized so
    we can scale them up to match Chandra's pixel coordinates.
    """
    if hasattr(region, "bbox") and not isinstance(region, dict):
        bb = region.bbox
        return (float(bb.x), float(bb.y), float(bb.width), float(bb.height))
    if isinstance(region, dict):
        if "bbox" in region and isinstance(region["bbox"], dict):
            bb = region["bbox"]
            try:
                return (
                    float(bb["x"]),
                    float(bb["y"]),
                    float(bb.get("width", bb.get("w", 0))),
                    float(bb.get("height", bb.get("h", 0))),
                )
            except (KeyError, TypeError, ValueError):
                return None
        try:
            return (
                float(region["bbox_x"]),
                float(region["bbox_y"]),
                float(region["bbox_width"]),
                float(region["bbox_height"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _region_attr(region: Any, attr: str, default: Any = None) -> Any:
    """Read an attribute from either an object or a dict region."""
    if hasattr(region, attr) and not isinstance(region, dict):
        return getattr(region, attr, default)
    if isinstance(region, dict):
        return region.get(attr, default)
    return default


def _intersection_area(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x_overlap = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    y_overlap = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return x_overlap * y_overlap


def merge_layout(
    yolo_regions: List[Any],
    chandra_blocks: List[Dict[str, Any]],
    page_size: Optional[Tuple[float, float]] = None,
) -> List[MergedRegion]:
    """Merge Chandra v2 bbox fragments into YOLO regions.

    Args:
        yolo_regions: List of LayoutRegion objects or equivalent dicts.
            Each must expose `type` / `region_type`, a bbox, optional
            `reading_order`, and `confidence`.
        chandra_blocks: List of {text, x, y, w, h} dicts from Chandra v2.
        page_size: Optional (page_width_px, page_height_px) tuple. If
            given, YOLO regions whose coordinates appear normalized
            (all <= 1.0) are scaled up to pixel space before matching.

    Returns:
        List of MergedRegion in reading order. Includes one trailing
        UNCLASSIFIED region for any orphan fragments so text is never
        dropped silently.
    """
    if not chandra_blocks and not yolo_regions:
        return []

    # Normalize YOLO regions into (rect_in_pixels, region_obj) pairs.
    region_entries: List[Tuple[Tuple[float, float, float, float], Any]] = []
    for r in yolo_regions:
        rect = _region_rect(r)
        if rect is None:
            continue
        x, y, w, h = rect
        if page_size is not None and max(x, y, w, h) <= 1.0:
            pw, ph = page_size
            rect = (x * pw, y * ph, w * pw, h * ph)
        region_entries.append((rect, r))

    # Bucket fragments per region by greater-overlap area.
    buckets: List[List[Dict[str, Any]]] = [[] for _ in region_entries]
    orphans: List[Dict[str, Any]] = []

    for fragment in chandra_blocks:
        if not isinstance(fragment, dict):
            continue
        text_val = fragment.get("text")
        if not isinstance(text_val, str) or not text_val.strip():
            continue
        frect = _fragment_rect(fragment)
        if frect is None:
            orphans.append(fragment)
            continue
        farea = max(frect[2] * frect[3], 1.0)

        # Pick greater overlap; on ties, prefer the smaller (more specific)
        # region so that nested layouts (e.g. a TITLE box inside a TEXT box)
        # attribute the fragment to the tighter fit.
        best_idx = -1
        best_overlap = 0.0
        best_region_area = float("inf")
        for idx, (rrect, _r) in enumerate(region_entries):
            overlap = _intersection_area(frect, rrect)
            region_area = max(rrect[2] * rrect[3], 1.0)
            if overlap > best_overlap or (
                overlap > 0 and overlap == best_overlap and region_area < best_region_area
            ):
                best_overlap = overlap
                best_region_area = region_area
                best_idx = idx

        if best_idx >= 0 and (best_overlap / farea) >= MIN_CONTAINMENT_RATIO:
            buckets[best_idx].append(fragment)
        else:
            orphans.append(fragment)

    merged: List[MergedRegion] = []

    # Build merged regions in YOLO reading-order.
    # NOTE: `reading_order` may legitimately be 0; use an explicit None check
    # rather than `or 10_000`, which would treat 0 as missing and sort the
    # first reading-order region to the end.
    def _sort_key(item):
        region_obj = item[1][1]
        rect = item[1][0]
        ro = _region_attr(region_obj, "reading_order", default=None)
        if ro is None:
            ro = 10_000
        return (ro, rect[1], rect[0])

    indexed = list(enumerate(region_entries))
    indexed.sort(key=_sort_key)

    for original_idx, (rect, region) in indexed:
        fragments_in_region = sorted(
            buckets[original_idx],
            key=lambda f: (float(f.get("y", 0) or 0), float(f.get("x", 0) or 0)),
        )
        text_content = "\n".join(
            f["text"].strip() for f in fragments_in_region if isinstance(f.get("text"), str) and f["text"].strip()
        )
        bbox_dict = {
            "x": rect[0],
            "y": rect[1],
            "width": rect[2],
            "height": rect[3],
            "page": _region_attr(_region_attr(region, "bbox", default={}), "page", default=0)
                    if hasattr(region, "bbox")
                    else (region.get("bbox", {}).get("page", region.get("page_number", 0)) if isinstance(region, dict) else 0),
        }
        # Carry forward source-region metadata (notably `image_size` set by
        # YoloLayoutDetector) so downstream consumers - especially stage_1's
        # `_store_layout_regions` which normalizes pixel bboxes to 0-1 using
        # `metadata.image_size` - keep working when fed cached merged regions.
        source_metadata = _region_attr(region, "metadata", default={}) or {}
        merged_metadata = dict(source_metadata) if isinstance(source_metadata, dict) else {}
        merged_metadata["chandra_fragment_count"] = len(fragments_in_region)
        merged.append(MergedRegion(
            region_type=str(_region_attr(region, "type", None) or _region_attr(region, "region_type", "TEXT")),
            bbox=bbox_dict,
            confidence=float(_region_attr(region, "confidence", default=0.0) or 0.0),
            reading_order=_region_attr(region, "reading_order", default=None),
            text_content=text_content,
            fragments=fragments_in_region,
            metadata=merged_metadata,
        ))

    if orphans:
        orphans_sorted = sorted(
            orphans,
            key=lambda f: (float(f.get("y", 0) or 0), float(f.get("x", 0) or 0)),
        )
        merged.append(MergedRegion(
            region_type=UNCLASSIFIED_REGION_TYPE,
            bbox={"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0, "page": 0},
            confidence=0.0,
            reading_order=(merged[-1].reading_order + 1) if merged and merged[-1].reading_order is not None else None,
            text_content="\n".join(
                f["text"].strip() for f in orphans_sorted if isinstance(f.get("text"), str) and f["text"].strip()
            ),
            fragments=orphans_sorted,
            metadata={"chandra_fragment_count": len(orphans_sorted), "orphan": True},
        ))
        logger.debug("layout_merge: %d orphan fragments grouped into UNCLASSIFIED", len(orphans_sorted))

    return merged


def merged_regions_to_joined_text(merged: List[MergedRegion]) -> str:
    """Flatten merged regions back to a joined-string for legacy consumers."""
    parts: List[str] = []
    for region in merged:
        if region.text_content.strip():
            parts.append(region.text_content.strip())
    return "\n\n".join(parts)
