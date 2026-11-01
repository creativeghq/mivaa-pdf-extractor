from __future__ import annotations

import html
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.services.pdf.ocr_completeness import numeric_tokens

OCR_VERSION = "paddleocr-vl"
TEXT_LAYER_VERSION = "text-layer"
LAYOUT_PROCESSING_VERSIONS = frozenset({OCR_VERSION, TEXT_LAYER_VERSION})

ROUTE_TEXT_LAYER = "text_layer"
ROUTE_OCR_WITH_REFERENCE = "ocr_with_reference"
ROUTE_OCR = "ocr"

MIN_WORDS = 25
MIN_PRINTABLE_RATIO = 0.97
MAX_GARBAGE_WORD_RATIO = 0.10
MAX_IMAGE_COVERAGE = 0.85
MAX_UNSTRUCTURED_NUMERIC_RATIO = 0.15
MAX_LOST_NUMBER_SHARE = 0.05
MIN_NUMBERS_FOR_STRUCTURE_CHECK = 12
TITLE_SIZE_FACTOR = 1.35
TITLE_MAX_CHARS = 120
TABLE_TITLE_MAX_GAP_PT = 60.0
MIN_IMAGE_AREA_PT2 = 900.0
LARGE_IMAGE_COVERAGE = 0.25
LARGE_SINGLE_IMAGE = 0.15
MIN_COLUMN_BLOCKS = 3
COLUMN_GAP_PT = 40.0
NARROW_BLOCK_SHARE = 0.6

Word = Tuple[float, float, float, float, str]
Rect = Tuple[float, float, float, float]

_GARBAGE = re.compile("[�-]|\\(cid:\\d+\\)")


@dataclass
class TextVerdict:
    usable: bool
    reasons: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteDecision:
    route: str
    reasons: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


def assess_text_layer(words: Sequence[Word], image_coverage: float = 0.0) -> TextVerdict:
    reasons: List[str] = []
    texts = [w[4] for w in words]
    total_chars = sum(len(t) for t in texts)
    printable = sum(sum(1 for c in t if c.isprintable()) for t in texts)
    garbage_words = sum(1 for t in texts if _GARBAGE.search(t))
    stats = {
        "words": len(words),
        "printable_ratio": round(printable / total_chars, 4) if total_chars else 0.0,
        "garbage_word_ratio": round(garbage_words / len(words), 4) if words else 0.0,
        "image_coverage": round(image_coverage, 3),
    }
    if len(words) < MIN_WORDS:
        reasons.append("few_words")
    if total_chars and stats["printable_ratio"] < MIN_PRINTABLE_RATIO:
        reasons.append("unprintable_characters")
    if words and stats["garbage_word_ratio"] > MAX_GARBAGE_WORD_RATIO:
        reasons.append("garbled_text")
    if image_coverage >= MAX_IMAGE_COVERAGE:
        reasons.append("full_page_image")
    return TextVerdict(usable=not reasons, reasons=reasons, stats=stats)


def _center(w: Word) -> Tuple[float, float]:
    return (w[0] + w[2]) / 2.0, (w[1] + w[3]) / 2.0


def _inside(point: Tuple[float, float], rect: Rect) -> bool:
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def words_in_rect(words: Sequence[Word], rect: Rect) -> List[Word]:
    return [w for w in words if _inside(_center(w), rect)]


def reference_numbers(words: Sequence[Word], rect: Optional[Rect] = None) -> set:
    chosen = words if rect is None else words_in_rect(words, rect)
    return numeric_tokens("|".join(w[4] for w in chosen))


def column_count(blocks: Sequence[Dict[str, Any]], page_width: float) -> int:
    narrow = [b["bbox"] for b in blocks if (b["bbox"][2] - b["bbox"][0]) <= page_width * NARROW_BLOCK_SHARE]
    if len(narrow) < MIN_COLUMN_BLOCKS * 2:
        return 1
    starts = sorted(r[0] for r in narrow)
    groups: List[List[float]] = [[starts[0]]]
    for x in starts[1:]:
        if x - groups[-1][-1] > COLUMN_GAP_PT:
            groups.append([x])
        else:
            groups[-1].append(x)
    return sum(1 for g in groups if len(g) >= MIN_COLUMN_BLOCKS)


def route_page(
    verdict: TextVerdict,
    words: Sequence[Word],
    table_rects: Sequence[Rect],
    blocks: Sequence[Dict[str, Any]] = (),
    page_width: float = 0.0,
    largest_image: float = 0.0,
) -> RouteDecision:
    stats: Dict[str, Any] = dict(verdict.stats)
    if not verdict.usable:
        return RouteDecision(ROUTE_OCR, list(verdict.reasons), stats)
    coverage = float(verdict.stats.get("image_coverage", 0.0))
    stats["largest_image"] = round(largest_image, 3)
    if coverage >= LARGE_IMAGE_COVERAGE or largest_image >= LARGE_SINGLE_IMAGE:
        return RouteDecision(ROUTE_OCR_WITH_REFERENCE, ["large_images_may_hold_text"], stats)
    free_blocks = [
        b for b in blocks
        if not any(_inside(((b["bbox"][0] + b["bbox"][2]) / 2.0, (b["bbox"][1] + b["bbox"][3]) / 2.0), r) for r in table_rects)
    ]
    if page_width and column_count(free_blocks, page_width) >= 2:
        return RouteDecision(ROUTE_OCR_WITH_REFERENCE, ["multi_column_layout"], stats)
    numeric = [w for w in words if numeric_tokens(w[4])]
    outside = [w for w in numeric if not any(_inside(_center(w), r) for r in table_rects)]
    stats.update({"numeric_words": len(numeric), "numeric_outside_tables": len(outside), "tables": len(table_rects)})
    if len(numeric) < MIN_NUMBERS_FOR_STRUCTURE_CHECK:
        return RouteDecision(ROUTE_TEXT_LAYER, [], stats)
    ratio = len(outside) / len(numeric)
    stats["unstructured_numeric_ratio"] = round(ratio, 3)
    if ratio <= MAX_UNSTRUCTURED_NUMERIC_RATIO:
        return RouteDecision(ROUTE_TEXT_LAYER, [], stats)
    return RouteDecision(ROUTE_OCR_WITH_REFERENCE, ["numbers_outside_ruled_tables"], stats)


def lost_number_share(words: Sequence[Word], elements: Sequence[Dict[str, Any]]) -> Tuple[float, int]:
    want = reference_numbers(words)
    if not want:
        return 0.0, 0
    got: set = set()
    for el in elements:
        got |= numeric_tokens(el.get("text_content") or "")
    return len(want - got) / len(want), len(want)


def check_table_rows(rows: Sequence[Sequence[Any]]) -> Dict[str, Any]:
    reasons: List[str] = []
    widths = [len(r) for r in rows]
    if len(rows) < 2 or (widths and max(widths) < 2):
        reasons.append("too_small")
    if widths:
        modal = statistics.mode(widths)
        ragged = sum(1 for w in widths if w != modal)
        if ragged / len(widths) > 0.10:
            reasons.append("ragged_rows")
        cells = sum(widths)
        empty = sum(1 for r in rows for c in r if c in (None, ""))
        if cells and empty / cells > 0.6:
            reasons.append("sparse_cells")
    return {"needs_review": bool(reasons), "reasons": reasons, "rows": len(rows)}


def rows_to_html(rows: Sequence[Sequence[Any]]) -> str:
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(c).strip()) if c not in (None, '') else ''}</td>" for c in row)
        body.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(body) + "</table>"


def _cell_text(cell: Any) -> str:
    return "" if cell in (None, "") else " ".join(str(cell).split())


def rows_to_text(rows: Sequence[Sequence[Any]]) -> str:
    return "\n".join(" | ".join(_cell_text(c) for c in row) for row in rows)


def _table_title(table_rect: Rect, blocks: Sequence[Dict[str, Any]]) -> Optional[str]:
    best: Optional[Tuple[float, str]] = None
    for b in blocks:
        x0, y0, x1, y1 = b["bbox"]
        gap = table_rect[1] - y1
        if gap < -2 or gap > TABLE_TITLE_MAX_GAP_PT:
            continue
        if x1 < table_rect[0] or x0 > table_rect[2]:
            continue
        text = " ".join(str(b.get("text", "")).split())
        if not text or len(text) > TITLE_MAX_CHARS:
            continue
        if sum(c.isalpha() for c in text) < 0.4 * len(text):
            continue
        if best is None or gap < best[0]:
            best = (gap, text)
    return best[1] if best else None


def elements_from_content(content: Dict[str, Any], page_number: int, zoom: float, origin: Tuple[float, float] = (0.0, 0.0)) -> List[Dict[str, Any]]:
    ox, oy = origin
    blocks = content.get("blocks", [])
    tables = content.get("tables", [])
    table_rects = [tuple(t["bbox"]) for t in tables]
    sizes = [b["max_size"] for b in blocks if b.get("max_size")]
    median_size = statistics.median(sizes) if sizes else 0.0

    def px(rect: Rect) -> Dict[str, float]:
        return {
            "x": (rect[0] - ox) * zoom,
            "y": (rect[1] - oy) * zoom,
            "width": (rect[2] - rect[0]) * zoom,
            "height": (rect[3] - rect[1]) * zoom,
            "page": page_number,
        }

    items: List[Tuple[float, float, Dict[str, Any]]] = []
    for t in tables:
        rect = tuple(t["bbox"])
        rows = t["rows"]
        checks = check_table_rows(rows)
        title = _table_title(rect, blocks)
        items.append((rect[1], rect[0], {
            "region_type": "TABLE",
            "bbox": px(rect),
            "confidence": 0.99,
            "text_content": rows_to_text(rows),
            "fragments": [],
            "metadata": {
                "paddle_label": None,
                "html": rows_to_html(rows),
                "source": TEXT_LAYER_VERSION,
                "title": title,
                "needs_review": checks["needs_review"],
                "checks": checks,
            },
        }))
    for b in blocks:
        rect = tuple(b["bbox"])
        text = str(b.get("text", "")).strip()
        if not text:
            continue
        cx, cy = (rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0
        if any(_inside((cx, cy), r) for r in table_rects):
            continue
        is_title = bool(median_size) and b.get("max_size", 0) >= median_size * TITLE_SIZE_FACTOR and len(text) <= TITLE_MAX_CHARS and text.count("\n") <= 1
        items.append((rect[1], rect[0], {
            "region_type": "TITLE" if is_title else "TEXT",
            "bbox": px(rect),
            "confidence": 0.99,
            "text_content": text,
            "fragments": [],
            "metadata": {"paddle_label": None, "html": None, "source": TEXT_LAYER_VERSION},
        }))
    for rect in content.get("images", []):
        rect = tuple(rect)
        items.append((rect[1], rect[0], {
            "region_type": "IMAGE",
            "bbox": px(rect),
            "confidence": 0.99,
            "text_content": "",
            "fragments": [],
            "metadata": {"paddle_label": None, "html": None, "source": TEXT_LAYER_VERSION},
        }))
    items.sort(key=lambda it: (round(it[0] / 4.0), it[1]))
    elements = []
    for order, (_, _, element) in enumerate(items):
        element["reading_order"] = order
        elements.append(element)
    return elements


def extract_tables(page: Any, clip: Any = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    tables: List[Dict[str, Any]] = []
    try:
        for t in page.find_tables(clip=clip).tables:
            rows = [[("" if c is None else str(c)) for c in row] for row in t.extract()]
            tables.append({"bbox": tuple(t.bbox), "rows": rows})
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    return tables, None


def _clip_rect(rect: Rect, clip: Any) -> Optional[Rect]:
    if clip is None:
        return rect
    x0, y0, x1, y1 = max(rect[0], clip.x0), max(rect[1], clip.y0), min(rect[2], clip.x1), min(rect[3], clip.y1)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def extract_page_content(page: Any, clip: Any = None, include_tables: bool = True) -> Dict[str, Any]:
    raw_words = page.get_text("words", clip=clip)
    words: List[Word] = [(w[0], w[1], w[2], w[3], w[4]) for w in raw_words]
    blocks: List[Dict[str, Any]] = []
    for b in page.get_text("dict", clip=clip, sort=True).get("blocks", []):
        if b.get("type") != 0:
            continue
        lines, sizes = [], []
        for line in b.get("lines", []):
            spans = line.get("spans", [])
            lines.append("".join(s.get("text", "") for s in spans))
            sizes.extend(s.get("size", 0.0) for s in spans if s.get("text", "").strip())
        text = "\n".join(ln for ln in lines if ln.strip())
        if text.strip():
            blocks.append({"bbox": tuple(b["bbox"]), "text": text, "max_size": max(sizes) if sizes else 0.0})
    width = clip.width if clip is not None else page.rect.width
    height = clip.height if clip is not None else page.rect.height
    area = width * height
    images: List[Rect] = []
    covered = 0.0
    largest = 0.0
    for info in page.get_image_info():
        r = info.get("bbox")
        if not r:
            continue
        rect = _clip_rect((r[0], r[1], r[2], r[3]), clip)
        if rect is None:
            continue
        size = (rect[2] - rect[0]) * (rect[3] - rect[1])
        if size < MIN_IMAGE_AREA_PT2:
            continue
        images.append(rect)
        covered += size
        largest = max(largest, size / area if area else 0.0)
    tables: List[Dict[str, Any]] = []
    table_error: Optional[str] = None
    if include_tables:
        tables, table_error = extract_tables(page, clip)
    return {
        "words": words,
        "blocks": blocks,
        "tables": tables,
        "images": images,
        "image_coverage": min(1.0, covered / area) if area else 0.0,
        "largest_image": largest,
        "page_width": width,
        "table_error": table_error,
    }
