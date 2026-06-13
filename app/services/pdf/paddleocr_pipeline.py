"""
PaddleOCR-VL structural-pass contract + parser.

PaddleOCR-VL is a **two-stage** document parser run in one process by the
``paddleocr`` package on Modal: PP-DocLayoutV2 (RT-DETR detector + pointer
network) localizes regions, labels them, and predicts reading order; the
PaddleOCR-VL-0.9B VLM recognizes the content inside each region (text,
tables→markdown, formulas→LaTeX, charts). It replaced Surya-2 (2026-06-13) — the
RT-DETR boxes are tighter (→ cleaner product crops) and the reading order is from
a dedicated model.

The Modal app ([modal_app/paddleocr_vl.py](../../../modal_app/paddleocr_vl.py))
returns a JSON ``/parse`` response::

    {"regions": [{"bbox": [x0,y0,x1,y1] px, "label": str,
                  "content": str, "order": int}],
     "width": int, "height": int}

This module is PURE (no I/O). It maps that response onto the platform's existing
``document_layout_analysis`` element schema so every downstream consumer (Stage 2
chunking, Stage 3 crops, focused extraction) stays untouched — exactly the seam
the Surya parser produced.

Coordinate convention: PaddleOCR returns **pixel** bboxes on the image we sent.
We normalize to **0..1** at this boundary (dividing by the sent image's
width/height), then :func:`region_to_layout_element` denormalizes back to the
pixel space of the crop render — keeping the normalize→pixel conversion in one
place, identical to the old Surya flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Label taxonomy — PP-DocLayoutV2 categories → the existing region_type vocab
# (TEXT / TABLE / TITLE / CAPTION / IMAGE / FIGURE / BLANK) used across the
# pipeline. Labels are lowercased before lookup; anything unmapped falls back to
# TEXT (conservative: TEXT is OCR-eligible and never treated as a product crop,
# so an unknown label can't silently become a bogus image crop).
# ---------------------------------------------------------------------------
PADDLE_LABEL_TO_REGION_TYPE: Dict[str, str] = {
    # titles / headers
    "doc_title": "TITLE",
    "title": "TITLE",
    "paragraph_title": "TITLE",
    "chart_title": "TITLE",
    "table_title": "TITLE",
    "figure_title": "CAPTION",
    "abstract": "TEXT",
    # body text
    "text": "TEXT",
    "plain_text": "TEXT",
    "content": "TEXT",
    "reference": "TEXT",
    "footnote": "TEXT",
    "header": "TEXT",
    "footer": "TEXT",
    "page_number": "TEXT",
    "header_image": "TEXT",
    "footer_image": "TEXT",
    "aside_text": "TEXT",
    "number": "TEXT",
    # captions
    "figure_caption": "CAPTION",
    "image_caption": "CAPTION",
    "table_caption": "CAPTION",
    "caption": "CAPTION",
    "chart_caption": "CAPTION",
    # tables / forms (structure preserved as HTML/markdown in metadata)
    "table": "TABLE",
    "form": "TABLE",
    # visual blocks — the product-crop sources
    "image": "IMAGE",
    "figure": "FIGURE",
    "chart": "FIGURE",
    "diagram": "FIGURE",
    "seal": "FIGURE",
    # math / code stay text-eligible
    "formula": "TEXT",
    "formula_number": "TEXT",
    "equation": "TEXT",
    "algorithm": "TEXT",
    "code": "TEXT",
}

#: region_types that are product-image crop sources (Stage 3 cuts crops from
#: these). RT-DETR's chart boxes are a new, welcome crop source vs Surya.
CROP_SOURCE_REGION_TYPES = frozenset({"IMAGE", "FIGURE"})

#: Labels whose recognized content is structured (kept as HTML in metadata for
#: the TABLE region so Stage 2 can preserve table structure).
TABLE_REGION_TYPES = frozenset({"TABLE"})

#: Default per-region confidence (the Modal contract does not surface the
#: detector score per region; if we add it later this becomes per-region).
DEFAULT_REGION_CONFIDENCE = 0.9


@dataclass
class PaddleRegion:
    """One region from a PaddleOCR-VL parse.

    ``bbox`` is ``(x0, y0, x1, y1)`` in **0..1** normalized space. ``content`` is
    the recognized text (markdown for tables, LaTeX for formulas, empty for pure
    visual blocks). ``order`` is the reading-order index from PP-DocLayoutV2.
    """

    label: str
    bbox: Tuple[float, float, float, float]  # 0..1 normalized (x0, y0, x1, y1)
    content: str
    order: int
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def region_type(self) -> str:
        return PADDLE_LABEL_TO_REGION_TYPE.get(self.label.strip().lower(), "TEXT")

    @property
    def is_crop_source(self) -> bool:
        return self.region_type in CROP_SOURCE_REGION_TYPES

    @property
    def is_text_bearing(self) -> bool:
        return self.region_type not in CROP_SOURCE_REGION_TYPES and bool(self.content)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def parse_parse_response(payload: Dict[str, Any]) -> List[PaddleRegion]:
    """Turn the Modal ``/parse`` JSON into :class:`PaddleRegion` list.

    Normalizes pixel bbox → 0..1 using the response's ``width``/``height``,
    clamps + order-corrects. Regions missing a usable bbox are skipped (never
    guessed). Returns regions in reading order.
    """
    width = float(payload.get("width") or 0) or 1.0
    height = float(payload.get("height") or 0) or 1.0
    out: List[PaddleRegion] = []
    for raw in payload.get("regions", []) or []:
        bbox = raw.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = (float(v) for v in bbox)
        nx0, ny0, nx1, ny1 = (
            _clamp01(x0 / width), _clamp01(y0 / height),
            _clamp01(x1 / width), _clamp01(y1 / height),
        )
        if nx1 < nx0:
            nx0, nx1 = nx1, nx0
        if ny1 < ny0:
            ny0, ny1 = ny1, ny0
        out.append(
            PaddleRegion(
                label=str(raw.get("label") or "text"),
                bbox=(nx0, ny0, nx1, ny1),
                content=str(raw.get("content") or ""),
                order=int(raw.get("order", len(out))),
            )
        )
    out.sort(key=lambda r: r.order)
    return out


def region_to_layout_element(
    region: PaddleRegion,
    page_width_px: float,
    page_height_px: float,
    page_number: int,
) -> Dict[str, object]:
    """Convert a :class:`PaddleRegion` (0..1) into a pixel-space layout element
    matching the ``layout_elements[]`` contract Stage 2 chunking, Stage 3 crops,
    and focused extraction already consume (identical shape to the old Surya
    ``block_to_region``).

    ``page_width_px`` / ``page_height_px`` are the rendered page pixel size
    (250 DPI render) used to denormalize.
    """
    x0, y0, x1, y1 = region.bbox
    return {
        "region_type": region.region_type,
        "bbox": {
            "x": x0 * page_width_px,
            "y": y0 * page_height_px,
            "width": (x1 - x0) * page_width_px,
            "height": (y1 - y0) * page_height_px,
            "page": page_number,
        },
        "confidence": DEFAULT_REGION_CONFIDENCE,
        "reading_order": region.order,
        "text_content": region.content if region.region_type not in CROP_SOURCE_REGION_TYPES else "",
        "fragments": [],  # PaddleOCR returns merged regions; no sub-fragment list.
        "metadata": {
            "paddle_label": region.label,
            # Table content is markdown/HTML from the VLM — preserve for Stage 2.
            "html": region.content if region.region_type in TABLE_REGION_TYPES else None,
            "source": "paddleocr-vl",
        },
    }


def regions_to_layout_elements(
    regions: List[PaddleRegion],
    page_width_px: float,
    page_height_px: float,
    page_number: int,
) -> List[Dict[str, object]]:
    """Convert a page's regions into the ``layout_elements[]`` list persisted to
    ``document_layout_analysis`` (``BLANK`` regions dropped — there are none in
    PaddleOCR's taxonomy, but the guard mirrors the old contract)."""
    return [
        region_to_layout_element(r, page_width_px, page_height_px, page_number)
        for r in regions
        if r.region_type != "BLANK"
    ]


def regions_to_reading_text(regions: List[PaddleRegion]) -> str:
    """Join text-bearing regions in reading order — the page's plain text.

    Used as the page text for discovery + chunking input. Regions arrive in
    reading order already, so this is an order-preserving join.
    """
    return "\n\n".join(r.content for r in regions if r.is_text_bearing and r.content).strip()
