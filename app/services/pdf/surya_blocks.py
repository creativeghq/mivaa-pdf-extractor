"""
Surya-2 structural-pass contract + parser.

Surya-2 (``datalab-to/surya-ocr-2``) is a single 650M vision-language model
that, in one ``/v1/chat/completions`` call over a rendered page image, returns
the page's layout as a sequence of top-level ``<div>`` blocks — each carrying a
layout label (Text / Section-Header / Table / Figure / Image / ...), a bounding
box, and the OCR'd inner content as HTML. One call gives layout regions + OCR
text + figure boxes, which is why it replaces the previous two-model split
(YOLO for region/figure boxes + Chandra for text + boxes) AND the
``merge_layout`` step that bucketed Chandra fragments into YOLO regions — Surya
does that bucketing internally.

This module is PURE (no I/O, no network): the verbatim prompt strings (the
model's training-time contract), the label taxonomy, a dependency-free parser
that turns Surya's raw HTML output into :class:`SuryaBlock` objects, and the
mapping onto the platform's existing ``document_layout_analysis`` element
schema so every downstream consumer (Stage 2 chunking, Stage 3 crops, focused
extraction) stays untouched.

Coordinate convention
---------------------
Surya emits bboxes in a **0-1000 normalized** space. We divide by 1000 at this
boundary so a :class:`SuryaBlock` always carries clean ``0..1`` floats and the
0-1000 range never leaks downstream. The pixel form required by the existing
``layout_elements`` contract is produced explicitly by :func:`block_to_region`
using the page render size — keeping the normalized→pixel conversion in exactly
one place (the bbox-as-PDF-points bug class from the 2026-05-01 audit came from
ambiguous coordinate spaces; this module makes the space unambiguous).

Prompt wording is copied verbatim from ``surya/inference/prompts.py``. Per
Surya's own header: "The exact wording is the model's training-time contract —
do not paraphrase without retraining."
"""

from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Prompts — verbatim from surya/inference/prompts.py (training-time contract).
# ---------------------------------------------------------------------------

#: Full-page mode: one div per block with ``data-label`` + ``data-bbox``,
#: inner content OCR'd to HTML. This is the structural-pass prompt.
HIGH_ACCURACY_BBOX_PROMPT = (
    "OCR this image to HTML. Each block is a div with data-label and data-bbox "
    "(x0 y0 x1 y1, normalized 0-1000)."
)

#: Layout-only mode: JSON array of ``{label, bbox, count}`` — no OCR text.
LAYOUT_PROMPT = (
    "Output the layout of this image as JSON. Each entry is a dict with "
    '"label", "bbox", and "count" fields. Bbox is x0 y0 x1 y1, normalized 0-1000.'
)

#: Single-block OCR mode: OCR one cropped block image to HTML.
BLOCK_PROMPT = "OCR this block image to HTML."

#: Table-structure mode: JSON array of ``{label: Row|Col, bbox}``.
TABLE_REC_PROMPT = (
    "Output the table rows then columns as JSON. Each entry is a dict with "
    '"label" ("Row" or "Col") and "bbox" (x0 y0 x1 y1, normalized 0-1000).'
)

PROMPT_TYPE_HIGH_ACCURACY_BBOX = "high_accuracy_bbox"
PROMPT_TYPE_LAYOUT = "layout"
PROMPT_TYPE_BLOCK = "block"
PROMPT_TYPE_TABLE_REC = "table_rec"

PROMPT_MAPPING: Dict[str, str] = {
    PROMPT_TYPE_HIGH_ACCURACY_BBOX: HIGH_ACCURACY_BBOX_PROMPT,
    PROMPT_TYPE_LAYOUT: LAYOUT_PROMPT,
    PROMPT_TYPE_BLOCK: BLOCK_PROMPT,
    PROMPT_TYPE_TABLE_REC: TABLE_REC_PROMPT,
}

# ---------------------------------------------------------------------------
# Label taxonomy — verbatim from Surya, plus the mapping onto the existing
# region_type vocabulary so document_layout_analysis consumers are untouched.
# ---------------------------------------------------------------------------

#: Surya's 19 layout labels (verbatim from prompts.LAYOUT_LABEL_SET).
SURYA_LAYOUT_LABELS = (
    "Caption",
    "Footnote",
    "Equation-Block",
    "List-Group",
    "Page-Header",
    "Page-Footer",
    "Image",
    "Section-Header",
    "Table",
    "Text",
    "Complex-Block",
    "Code-Block",
    "Form",
    "Table-Of-Contents",
    "Figure",
    "Chemical-Block",
    "Diagram",
    "Bibliography",
    "Blank-Page",
)

#: Labels Surya itself does not OCR (returns an empty/near-empty block body).
#: These are the visual blocks; text never lives inside them.
SKIP_OCR_LABELS = frozenset({"Figure", "Image", "Diagram", "Blank-Page"})

#: Surya label -> existing region_type vocabulary used across the pipeline
#: (TEXT / TABLE / TITLE / CAPTION / IMAGE / FIGURE / BLANK). Anything not
#: listed falls back to TEXT (conservative: a TEXT region is OCR-eligible and
#: never treated as a product crop, so an unknown label can't silently become a
#: bogus image crop).
SURYA_LABEL_TO_REGION_TYPE: Dict[str, str] = {
    "Caption": "CAPTION",
    "Footnote": "TEXT",
    "Equation-Block": "TEXT",
    "List-Group": "TEXT",
    "Page-Header": "TEXT",
    "Page-Footer": "TEXT",
    "Image": "IMAGE",
    "Section-Header": "TITLE",
    "Table": "TABLE",
    "Text": "TEXT",
    "Complex-Block": "TEXT",
    "Code-Block": "TEXT",
    "Form": "TABLE",
    "Table-Of-Contents": "TEXT",
    "Figure": "FIGURE",
    "Chemical-Block": "FIGURE",
    "Diagram": "FIGURE",
    "Bibliography": "TEXT",
    "Blank-Page": "BLANK",
}

#: region_types that are product-image crop sources (Stage 3 cuts crops from
#: these). Matches the pre-Surya Stage-3 rule of cropping IMAGE/FIGURE regions.
CROP_SOURCE_REGION_TYPES = frozenset({"IMAGE", "FIGURE"})

#: Bbox scale Surya emits in. Native -> 0..1 is ``value / BBOX_SCALE``.
BBOX_SCALE = 1000.0

#: Default per-block confidence. Surya's full-page mode does not emit a
#: per-block score; mean token probability (when logprobs are requested) is
#: attached by the manager at the page level, not per block.
DEFAULT_BLOCK_CONFIDENCE = 0.9


@dataclass
class SuryaBlock:
    """One top-level block from a Surya structural pass.

    ``bbox`` is ``(x0, y0, x1, y1)`` in **0..1** normalized space (already
    divided out of Surya's 0-1000). ``html`` is the OCR'd inner HTML (empty for
    :data:`SKIP_OCR_LABELS`); ``text`` is its plain-text rendering for
    reading-order joins and embeddings.
    """

    label: str
    bbox: Tuple[float, float, float, float]  # 0..1 normalized (x0, y0, x1, y1)
    html: str
    text: str
    index: int  # position in the page, in the order Surya emitted (reading order)
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def region_type(self) -> str:
        """Map onto the existing region_type vocabulary."""
        return SURYA_LABEL_TO_REGION_TYPE.get(self.label, "TEXT")

    @property
    def is_crop_source(self) -> bool:
        """True when this block is a product-image crop source (IMAGE/FIGURE)."""
        return self.region_type in CROP_SOURCE_REGION_TYPES

    @property
    def is_text_bearing(self) -> bool:
        """True when the block carries OCR text (not a pure visual block)."""
        return self.label not in SKIP_OCR_LABELS and bool(self.text)


# ---------------------------------------------------------------------------
# Parser — dependency-free (no bs4: it is not a backend dependency).
# ---------------------------------------------------------------------------

_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z0-9]*\n?")
_FENCE_CLOSE_RE = re.compile(r"\n?```\s*$")
_DIV_TAG_RE = re.compile(r"<\s*(/?)\s*div\b[^>]*>", re.IGNORECASE)
# Attribute values may be double- or single-quoted.
_ATTR_RE = re.compile(
    r"""([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*(?:"([^"]*)"|'([^']*)')"""
)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE_OPEN_RE.sub("", cleaned)
        cleaned = _FENCE_CLOSE_RE.sub("", cleaned)
    return cleaned.strip()


def _iter_top_level_divs(html: str):
    """Yield ``(opening_tag, inner_html)`` for each top-level ``<div>...</div>``.

    Depth-aware so nested ``<div>``s inside a block (e.g. a complex block or a
    table cell wrapper) stay with their parent rather than being mistaken for
    new top-level blocks. A regex split can't do this; a tag-by-tag depth walk
    can, with no HTML-parser dependency.

    A trailing unclosed top-level div (model truncated near max_tokens) is still
    yielded with whatever inner content was emitted, so a cut-off final block
    isn't silently dropped.
    """
    depth = 0
    open_tag: Optional[str] = None
    content_start = 0
    for m in _DIV_TAG_RE.finditer(html):
        is_close = m.group(1) == "/"
        if not is_close:
            if depth == 0:
                open_tag = m.group(0)
                content_start = m.end()
            depth += 1
        else:
            if depth == 0:
                continue  # stray close tag — ignore
            depth -= 1
            if depth == 0 and open_tag is not None:
                yield open_tag, html[content_start:m.start()]
                open_tag = None
    if depth > 0 and open_tag is not None:
        # Truncated final block — keep what we have.
        yield open_tag, html[content_start:]


def _parse_attrs(tag: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name, dq, sq in _ATTR_RE.findall(tag):
        # Exactly one of the quoted groups matches; the other is "".
        out[name.lower()] = dq or sq
    return out


def _parse_bbox_str(bbox_str: str) -> Optional[Tuple[float, float, float, float]]:
    try:
        parts = [float(p) for p in bbox_str.replace(",", " ").split()]
    except (TypeError, ValueError):
        return None
    if len(parts) != 4:
        return None
    return (parts[0], parts[1], parts[2], parts[3])


def _normalize_bbox(
    parts: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    """Surya 0-1000 -> clamped, order-corrected 0..1 ``(x0, y0, x1, y1)``."""
    x0, y0, x1, y1 = (max(0.0, min(1.0, p / BBOX_SCALE)) for p in parts)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)


def html_to_text(html: str) -> str:
    """Plain-text rendering of a block's inner HTML (tags stripped, entities
    unescaped, whitespace collapsed). The HTML itself is preserved on the block
    for table structure; this is the text used for reading order + embeddings.
    """
    if not html:
        return ""
    text = _TAG_STRIP_RE.sub(" ", html)
    text = _html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def parse_full_page_html(raw: str) -> List[SuryaBlock]:
    """Parse the output of :data:`HIGH_ACCURACY_BBOX_PROMPT`.

    Surya emits a sequence of top-level ``<div data-bbox="x0 y0 x1 y1"
    data-label="...">inner HTML</div>`` blocks (no surrounding root). Returns one
    :class:`SuryaBlock` per top-level div, in emitted (reading) order, with
    bbox already normalized to 0..1. Blocks missing ``data-label`` or a valid
    ``data-bbox`` are skipped (never guessed).
    """
    cleaned = _strip_fences(raw)
    if not cleaned:
        return []

    blocks: List[SuryaBlock] = []
    for idx, (open_tag, inner) in enumerate(_iter_top_level_divs(cleaned)):
        attrs = _parse_attrs(open_tag)
        label = attrs.get("data-label")
        bbox_str = attrs.get("data-bbox")
        if not label or not bbox_str:
            continue
        parts = _parse_bbox_str(bbox_str)
        if parts is None:
            continue
        bbox = _normalize_bbox(parts)
        inner_html = inner.strip()
        text = "" if label in SKIP_OCR_LABELS else html_to_text(inner_html)
        blocks.append(
            SuryaBlock(
                label=str(label),
                bbox=bbox,
                html=inner_html,
                text=text,
                index=idx,
            )
        )
    return blocks


# ---------------------------------------------------------------------------
# Mapping onto the existing document_layout_analysis element schema.
# ---------------------------------------------------------------------------


def block_to_region(
    block: SuryaBlock,
    page_width_px: float,
    page_height_px: float,
    page_number: int,
) -> Dict[str, object]:
    """Convert a :class:`SuryaBlock` (0..1) into a pixel-space layout element
    matching the ``MergedRegion.to_dict()`` / ``layout_elements[]`` contract that
    Stage 2 chunking, Stage 3 crops, and focused extraction already consume.

    ``page_width_px`` / ``page_height_px`` are the rendered page pixel size
    (250 DPI render, ``pix.width`` / ``pix.height``) used to denormalize.
    """
    x0, y0, x1, y1 = block.bbox
    x_px = x0 * page_width_px
    y_px = y0 * page_height_px
    w_px = (x1 - x0) * page_width_px
    h_px = (y1 - y0) * page_height_px
    return {
        "region_type": block.region_type,
        "bbox": {
            "x": x_px,
            "y": y_px,
            "width": w_px,
            "height": h_px,
            "page": page_number,
        },
        "confidence": DEFAULT_BLOCK_CONFIDENCE,
        "reading_order": block.index,
        "text_content": block.text,
        "fragments": [],  # Surya returns merged blocks; no sub-fragment list.
        "metadata": {
            "surya_label": block.label,
            "html": block.html if block.region_type == "TABLE" else None,
            "source": "surya-2",
        },
    }


def blocks_to_layout_elements(
    blocks: List[SuryaBlock],
    page_width_px: float,
    page_height_px: float,
    page_number: int,
) -> List[Dict[str, object]]:
    """Convert a page's blocks into the ``layout_elements[]`` list persisted to
    ``document_layout_analysis``. ``Blank-Page`` blocks are dropped (no signal).
    """
    return [
        block_to_region(b, page_width_px, page_height_px, page_number)
        for b in blocks
        if b.region_type != "BLANK"
    ]


def blocks_to_reading_text(blocks: List[SuryaBlock]) -> str:
    """Join text-bearing blocks in reading order — the page's plain text.

    Surya emits blocks in reading order already, so this is an order-preserving
    join. Used as the page text for chunking input and for the page multimodal
    embedding.
    """
    return "\n\n".join(b.text for b in blocks if b.text).strip()
