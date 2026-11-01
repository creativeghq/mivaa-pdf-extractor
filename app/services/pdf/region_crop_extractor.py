"""Spread-aware region-crop extractor (Stage 3 Layer 2 replacement)."""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional, Set

import fitz  # PyMuPDF
from PIL import Image

from app.api.pdf_processing.stage_1_layout_precompute import LAYOUT_RENDER_DPI, _render_physical_page
from app.services.pdf.text_layer_extractor import LAYOUT_PROCESSING_VERSIONS, TEXT_LAYER_VERSION

# Which layout-region classifications become `extraction_layer='region_crop'` rows.
def _crop_region_types() -> tuple:
    env = os.getenv("PDF_CROP_REGION_TYPES")
    if env:
        parsed = tuple(t.strip().upper() for t in env.split(",") if t.strip())
        if parsed:
            return parsed
    try:
        from app.services.pdf.pdf_processor import PDF_CONSTANTS
        return tuple(str(t).upper() for t in PDF_CONSTANTS.CROP_REGION_TYPES)
    except Exception:
        # Never let a config import failure silently shrink extraction coverage back
        # to the old hardcoded pair — fall back to the documented default.
        return ("IMAGE", "FIGURE", "TABLE")


CROP_REGION_TYPES = _crop_region_types()


class RegionCropCacheReadFailed(RuntimeError):
    """The layout cache could not be read, so whether regions exist is unknown."""

# Degenerate-crop guard: skip crops smaller than this in either dimension (px).
# Render DPI / clip math is owned entirely by `_render_physical_page` (Stage 1.5),
# which renders at LAYOUT_RENDER_DPI=250 — the same grid the cached pixel bboxes
# were measured in, so crops align with no scaling here.
_MIN_CROP_PX = 8
# A photo region is re-rendered at the native resolution of the embedded image under it,
# never above this: past it the file grows and the picture does not.
MAX_PHOTO_CROP_DPI = 600
PHOTO_REGION_TYPES = ("IMAGE", "FIGURE")


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
        out.append({"rect": rect, "dpi": min(w / full.width, h / full.height) * 72})
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


async def extract_region_crops_for_physical_pages(
    *,
    file_content: bytes,
    physical_pages: List[int],
    catalog: Any,
    document_id: str,
    image_dir: str,
    job_id: Optional[str] = None,
    logger: Any = None,
    covered_pages_out: Optional[Set[int]] = None,
) -> List[Dict[str, Any]]:
    """Crop IMAGE/FIGURE regions from the PaddleOCR layout cache, spread-aware.

    Args:
        file_content: PDF file bytes.
        physical_pages: 1-based physical page numbers to crop.
        catalog: catalog object exposing `has_spread_layout` +
            `physical_to_pdf_map[physical_page] = (pdf_idx, position)`.
        document_id: document identifier (layout-cache key).
        image_dir: directory the crop JPEGs are written to (must be the dir
            whose files later get uploaded to storage).
        job_id: optional job id for log context.
        logger: optional logger.
        covered_pages_out: filled with every physical page that HAS a layout analysis,
            whether or not it holds a region; a full-page render is redundant for those.

    Returns:
        List of `image_info` dicts shaped exactly like the old
        `_extract_region_crops` output (extraction_layer='region_crop',
        page_number=physical_page, normalized 0..1 bbox).
    """
    import logging

    log = logger or logging.getLogger(__name__)

    if not file_content or not physical_pages:
        return []

    # ------------------------------------------------------------------
    # Load cached layout for these PHYSICAL pages (NO +1 offset — Stage 1.5
    # keys document_layout_analysis by physical page).
    # ------------------------------------------------------------------
    regions_by_page = await _load_layout_regions_for_physical_pages(
        document_id=document_id,
        physical_pages=physical_pages,
        logger=log,
        covered_pages_out=covered_pages_out,
    )

    if not regions_by_page:
        log.info(
            f"   ℹ️ [Job: {job_id}] No cached IMAGE/FIGURE regions for "
            f"physical pages {sorted(physical_pages)} — no region crops."
        )
        return []

    has_spread_layout = bool(catalog and getattr(catalog, "has_spread_layout", False))
    physical_to_pdf_map = (
        catalog.physical_to_pdf_map
        if catalog and hasattr(catalog, "physical_to_pdf_map")
        else {}
    )

    extracted_images: List[Dict[str, Any]] = []
    rendered_pages = 0

    doc = None
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")

        for physical_page in sorted(set(physical_pages)):
            regions = regions_by_page.get(physical_page)
            if not regions:
                continue

            # Resolve (pdf_idx, position) — same logic Stage 1.5 used.
            if has_spread_layout and physical_page in physical_to_pdf_map:
                pdf_idx, position = physical_to_pdf_map[physical_page]
            else:
                pdf_idx, position = physical_page - 1, "single"

            try:
                if pdf_idx < 0 or pdf_idx >= len(doc):
                    log.warning(
                        f"   ⚠️ [Job: {job_id}] physical page {physical_page} → "
                        f"pdf_idx {pdf_idx} out of bounds (doc has {len(doc)} sheets)"
                    )
                    continue

                # Render the SAME clipped half Stage 1.5 rendered. The returned
                # PIL image's pixel grid matches the cached region bboxes 1:1.
                img, bound_rect = _render_physical_page(doc, pdf_idx, position)
                rendered_pages += 1
                page = doc[pdf_idx]
                zoom = LAYOUT_RENDER_DPI / 72.0

                def to_pdf(r: Dict[str, Any], b: "fitz.Rect" = bound_rect, z: float = zoom) -> "fitz.Rect":
                    return fitz.Rect(b.x0 + r["x"] / z, b.y0 + r["y"] / z,
                                     b.x0 + (r["x"] + r["width"]) / z, b.y0 + (r["y"] + r["height"]) / z)

                embedded = _embedded_images(page, bound_rect)
                regions = merge_split_photo_regions(regions, [e["rect"] for e in embedded], to_pdf)
            except Exception as render_err:
                log.error(
                    f"   ❌ [Job: {job_id}] failed to render physical page "
                    f"{physical_page} (pdf_idx={pdf_idx}, position={position}): {render_err}"
                )
                continue

            try:
                img_w, img_h = img.width, img.height
                for region_idx, region in enumerate(regions):
                    try:
                        x = max(0, int(region["x"]))
                        y = max(0, int(region["y"]))
                        w = int(region["width"])
                        h = int(region["height"])

                        x2 = min(img_w, x + w)
                        y2 = min(img_h, y + h)

                        crop_w = x2 - x
                        crop_h = y2 - y
                        if crop_w < _MIN_CROP_PX or crop_h < _MIN_CROP_PX:
                            # Degenerate crop — skip.
                            continue

                        crop_dpi = LAYOUT_RENDER_DPI
                        if region.get("region_type") in PHOTO_REGION_TYPES:
                            native = [e["dpi"] for e in embedded if _covered_by(to_pdf(region), e["rect"], 0.5)]
                            if native:
                                crop_dpi = max(LAYOUT_RENDER_DPI, min(MAX_PHOTO_CROP_DPI, max(native)))
                        if crop_dpi > LAYOUT_RENDER_DPI * 1.1:
                            pix = page.get_pixmap(
                                matrix=fitz.Matrix(crop_dpi / 72, crop_dpi / 72), clip=to_pdf(region), alpha=False,
                            )
                            cropped = Image.open(io.BytesIO(pix.tobytes("png")))
                            pix = None
                        else:
                            crop_dpi = LAYOUT_RENDER_DPI
                            cropped = img.crop((x, y, x2, y2))
                        filename = f"page_{physical_page}_region_{region_idx}.jpg"
                        image_path = os.path.join(image_dir, filename)
                        cropped.save(image_path, "JPEG", quality=95)
                        cropped_w, cropped_h = cropped.size
                        cropped.close()

                        image_info = {
                            "path": image_path,
                            "vision_input_path": None,
                            "filename": filename,
                            "page_number": physical_page,
                            "width": cropped_w,
                            "height": cropped_h,
                            "format": "JPEG",
                            "detection_method": "layout_guided",
                            "extraction_layer": "region_crop",
                            "region_confidence": region.get("confidence"),
                            "region_type": region.get("region_type"),
                            "region_reading_order": region.get("reading_order"),
                            "render_dpi": round(crop_dpi),
                            "merged_regions": region.get("merged_from"),
                            "caption": None,
                            "bbox": [
                                x / img_w,
                                y / img_h,
                                crop_w / img_w,
                                crop_h / img_h,
                            ],
                        }
                        extracted_images.append(image_info)
                    except Exception as crop_err:
                        log.error(
                            f"   ❌ [Job: {job_id}] failed to crop region {region_idx} "
                            f"on physical page {physical_page}: {crop_err}"
                        )
                        continue
            finally:
                try:
                    img.close()
                except Exception:
                    pass

    except Exception as exc:
        log.error(f"   ❌ [Job: {job_id}] region crop extraction failed: {exc}")
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass

    log.info(
        f"   ✅ [Job: {job_id}] region crops: rendered {rendered_pages} physical "
        f"pages, produced {len(extracted_images)} region crops"
    )
    return extracted_images


async def _load_layout_regions_for_physical_pages(
    document_id: Optional[str],
    physical_pages: List[int],
    logger: Any,
    covered_pages_out: Optional[Set[int]] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """Load IMAGE/FIGURE regions keyed by physical page from the layout cache.

    Queries `document_layout_analysis` by `page_number` IN physical_pages
    directly (NO +1 offset — Stage 1.5 keys rows by physical page), filtering
    to `processing_version == 'paddleocr-vl'`. Returns
    `{physical_page: [{x,y,width,height,region_type,confidence,reading_order}]}`,
    keeping only IMAGE/FIGURE regions.
    """
    if not document_id or not physical_pages:
        return {}

    try:
        from app.services.core.supabase_client import get_supabase_client

        supabase = get_supabase_client()
        wanted = sorted(set(int(p) for p in physical_pages))
        response = (
            supabase.client.table("document_layout_analysis")
            .select("page_number, layout_elements, processing_version")
            .eq("document_id", document_id)
            .in_("page_number", wanted)
            .execute()
        )
    except Exception as exc:
        # S3-6: a cache READ FAILURE is not "this page had no IMAGE/FIGURE regions".
        raise RegionCropCacheReadFailed(
            f"layout cache read failed for document {document_id} pages "
            f"{sorted(set(int(p) for p in physical_pages))[:10]}: {exc}"
        ) from exc

    rows = response.data or []
    by_page: Dict[int, List[Dict[str, Any]]] = {}

    for row in rows:
        if row.get("processing_version") not in LAYOUT_PROCESSING_VERSIONS:
            continue
        try:
            page_num = int(row["page_number"])
        except (TypeError, ValueError, KeyError):
            continue
        if covered_pages_out is not None:
            covered_pages_out.add(page_num)
        elements = row.get("layout_elements") or []
        if not isinstance(elements, list):
            continue

        regions: List[Dict[str, Any]] = []
        for elem in elements:
            try:
                region_type = (elem.get("region_type") or "").upper()
                if region_type not in CROP_REGION_TYPES:
                    continue
                if region_type == "TABLE" and (elem.get("metadata") or {}).get("source") == TEXT_LAYER_VERSION:
                    continue
                bbox_dict = elem.get("bbox") or {}
                width = float(bbox_dict.get("width", 0)) or 0.0
                height = float(bbox_dict.get("height", 0)) or 0.0
                if width <= 0 or height <= 0:
                    continue
                regions.append(
                    {
                        "x": float(bbox_dict.get("x", 0)),
                        "y": float(bbox_dict.get("y", 0)),
                        "width": width,
                        "height": height,
                        "region_type": region_type,
                        "confidence": float(elem.get("confidence") or 0.85),
                        "reading_order": elem.get("reading_order"),
                    }
                )
            except Exception:
                continue

        if regions:
            by_page[page_num] = regions

    if by_page:
        total = sum(len(v) for v in by_page.values())
        logger.info(
            f"   ♻️ Loaded {total} IMAGE/FIGURE regions across {len(by_page)} "
            f"physical pages from document_layout_analysis"
        )
    return by_page
