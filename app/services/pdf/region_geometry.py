"""Page geometry for photo crops: which embedded raster sits under a layout region, its real
resolution, and how the regions the layout model drew map onto whole photos. Depends on PyMuPDF only."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set

import fitz  # PyMuPDF

# A photo region is re-rendered at the native resolution of the embedded image under it,
# never above this: past it the file grows and the picture does not.
MAX_PHOTO_CROP_DPI = 600
PHOTO_REGION_TYPES = ("IMAGE", "FIGURE")
# A placed photo this large that no layout region touches is cropped on its own: the layout
# model does not box a full-bleed photo, and the page's only other capture is a whole-spread render.
UNBOXED_PHOTO_MIN_PAGE_SHARE = 0.15
# Below this many REAL source pixels on the short side, a photo is too weak to show or embed as-is.
LOW_RES_MIN_SOURCE_PX = int(os.getenv("IMAGE_LOW_RES_MIN_SOURCE_PX", "400"))


def assess_source_quality(region_pdf: "fitz.Rect", native_dpi: Optional[float]) -> Dict[str, Any]:
    """The resolution the PDF actually holds for a photo region, not the render's.

    `sourcing: pending` defers the image's embedding until a better source was looked for.
    """
    if not native_dpi:
        return {"verdict": "vector", "native_dpi": None, "native_px": None}
    px = [round(region_pdf.width * native_dpi / 72), round(region_pdf.height * native_dpi / 72)]
    low = min(px) < LOW_RES_MIN_SOURCE_PX
    out: Dict[str, Any] = {"verdict": "low_res" if low else "ok", "native_dpi": round(native_dpi, 1), "native_px": px}
    if low:
        out["sourcing"] = "pending"
    return out


def _embedded_images(page: "fitz.Page", bound: "fitz.Rect") -> List[Dict[str, Any]]:
    """Placed raster images inside the rendered area, with their native DPI on the page."""
    out: List[Dict[str, Any]] = []
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        return out
    for info in infos:
        full = fitz.Rect(info.get("bbox") or (0, 0, 0, 0))
        rect = full & bound
        w, h = info.get("width") or 0, info.get("height") or 0
        if rect.is_empty or full.width <= 0 or full.height <= 0 or not w or not h:
            continue
        out.append({"rect": rect, "full": full, "dpi": min(w / full.width, h / full.height) * 72})
    return out


def gutter_photo_regions(regions: List[Dict[str, Any]], embedded: List[Dict[str, Any]], bound: "fitz.Rect", to_pdf) -> List[Dict[str, Any]]:
    """A photo printed across the fold of a spread is cropped whole, once.

    The half holding the photo's centre crops the full raster (`clip_pdf`, which crosses the
    fold); the facing half drops its region, which would only be a sliver of the same photo.
    A region qualifies only when it covers a real share of that raster on this half, so a
    page-wide background behind other content is never mistaken for the photo.
    """
    out: List[Dict[str, Any]] = []
    claimed: Set[tuple] = set()
    for r in regions:
        if r["region_type"] not in PHOTO_REGION_TYPES:
            out.append(r)
            continue
        rect = to_pdf(r)
        owners = [e for e in embedded if e.get("full") is not None and _covered_by(rect, e["rect"], 0.9)
                  and (rect & e["rect"]).width * (rect & e["rect"]).height >= 0.3 * e["rect"].width * e["rect"].height]
        if not owners:
            out.append(r)
            continue
        full = min(owners, key=lambda e: e["full"].width * e["full"].height)["full"]
        if full.x0 >= bound.x0 - 1 and full.x1 <= bound.x1 + 1:
            out.append(r)
            continue
        if not (bound.x0 <= (full.x0 + full.x1) / 2 < bound.x1):
            continue
        key = (round(full.x0), round(full.y0), round(full.x1), round(full.y1))
        if key in claimed:
            continue
        claimed.add(key)
        out.append({**r, "clip_pdf": key, "crosses_fold": True})
    return out


def _covered_by(region: "fitz.Rect", image: "fitz.Rect", share: float) -> bool:
    area = region.width * region.height
    inter = region & image
    return area > 0 and not inter.is_empty and (inter.width * inter.height) / area >= share


def merge_split_photo_regions(regions: List[Dict[str, Any]], images: List["fitz.Rect"], to_pdf) -> List[Dict[str, Any]]:
    """One photo the layout model cut into several regions becomes one region again.

    Each region belongs to the SMALLEST embedded image covering it, so two photos placed
    on one page-wide background are not glued together through the background.
    """
    groups: Dict[int, List[int]] = {}
    for i, r in enumerate(regions):
        if r["region_type"] not in PHOTO_REGION_TYPES:
            continue
        rect = to_pdf(r)
        owners = [j for j, image in enumerate(images) if _covered_by(rect, image, 0.9)]
        if owners:
            groups.setdefault(min(owners, key=lambda j: images[j].width * images[j].height), []).append(i)
    used: Set[int] = set()
    merged: List[Dict[str, Any]] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        used.update(group)
        x0 = min(regions[i]["x"] for i in group)
        y0 = min(regions[i]["y"] for i in group)
        x1 = max(regions[i]["x"] + regions[i]["width"] for i in group)
        y1 = max(regions[i]["y"] + regions[i]["height"] for i in group)
        merged.append({**regions[group[0]], "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0, "merged_from": len(group)})
    return [r for i, r in enumerate(regions) if i not in used] + merged


def _trim_printed_bands(rect: "fitz.Rect", text_rects: List["fitz.Rect"]) -> "fitz.Rect":
    """Cut the heading / footer printed over a full-bleed photo off its top and bottom.

    With them left in, the crop is a catalogue page and the classifier rightly rejects it.
    """
    top, bottom = rect.y0, rect.y1
    band = rect.height * 0.25
    for t in text_rects:
        if (t & rect).is_empty:
            continue
        if t.y1 <= rect.y0 + band:
            top = max(top, t.y1 + 2)
        elif t.y0 >= rect.y1 - band:
            bottom = min(bottom, t.y0 - 2)
    return fitz.Rect(rect.x0, top, rect.x1, bottom) if bottom - top > rect.height * 0.5 else rect


def unboxed_photo_regions(regions: List[Dict[str, Any]], embedded: List[Dict[str, Any]], bound: "fitz.Rect", zoom: float, to_pdf,
                          text_rects: Optional[List["fitz.Rect"]] = None) -> List[Dict[str, Any]]:
    """Large placed photos no photo region overlaps, as regions in the render's pixel grid."""
    page_area = bound.width * bound.height
    photos = [to_pdf(r) for r in regions if r["region_type"] in PHOTO_REGION_TYPES]
    out: List[Dict[str, Any]] = []
    for e in embedded:
        rect = e["rect"]
        if page_area <= 0 or (rect.width * rect.height) / page_area < UNBOXED_PHOTO_MIN_PAGE_SHARE:
            continue
        if any(_covered_by(p, rect, 0.5) or _covered_by(rect, p, 0.5) for p in photos):
            continue
        rect = _trim_printed_bands(rect, text_rects or [])
        out.append({
            "x": (rect.x0 - bound.x0) * zoom, "y": (rect.y0 - bound.y0) * zoom,
            "width": rect.width * zoom, "height": rect.height * zoom,
            "region_type": "IMAGE", "confidence": None, "reading_order": None, "source": "embedded_image_rect",
        })
    return out
