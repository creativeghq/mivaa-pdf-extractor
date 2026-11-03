"""One page's drawing items and text, in the coordinates a viewer shows (rotation applied)."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field
from typing import List, Tuple

try:
    import pymupdf as fitz
except ImportError:  # PyMuPDF older than 1.24 ships only the `fitz` name
    import fitz

Point = Tuple[float, float]
Segment = Tuple[float, float, float, float]
Curve = Tuple[Point, Point, Point, Point]
BBox = Tuple[float, float, float, float]

METRES_PER_PT = 25.4 / 72 / 1000
COVERAGE_GRID = 64
# A compressed page can expand to millions of path operators; past this we refuse to materialise them.
MAX_CONTENT_BYTES = 40_000_000


@dataclass
class TextLine:
    text: str
    bbox: BBox


@dataclass
class PageGeometry:
    width: float
    height: float
    segments: List[Segment] = field(default_factory=list)
    curves: List[Curve] = field(default_factory=list)
    polygons: List[List[Point]] = field(default_factory=list)
    lines: List[TextLine] = field(default_factory=list)
    image_bboxes: List[BBox] = field(default_factory=list)
    image_digests: List[str] = field(default_factory=list)
    item_count: int = 0
    too_complex: bool = False
    image_coverage: float = 0.0

    @property
    def text_chars(self) -> int:
        return sum(len(line.text.strip()) for line in self.lines)


def fold(text: str) -> str:
    """Upper-case and strip accents character by character, so offsets stay aligned with `text`."""
    out = []
    for ch in text:
        base = unicodedata.normalize("NFD", ch)[0].upper()
        out.append(base[0] if base else ch)
    return "".join(out)


def _transformer(page: "fitz.Page"):
    m = page.rotation_matrix
    a, b, c, d, e, f = m.a, m.b, m.c, m.d, m.e, m.f

    def tp(p) -> Point:
        x, y = p[0], p[1]
        return (a * x + c * y + e, b * x + d * y + f)

    def tr(r) -> BBox:
        pts = [tp((r[0], r[1])), tp((r[2], r[1])), tp((r[0], r[3])), tp((r[2], r[3]))]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    return tp, tr


def _close(p: Point, q: Point, eps: float = 0.01) -> bool:
    return abs(p[0] - q[0]) <= eps and abs(p[1] - q[1]) <= eps


def _edges(poly: List[Point]) -> List[Segment]:
    return [(poly[i][0], poly[i][1], poly[(i + 1) % len(poly)][0], poly[(i + 1) % len(poly)][1]) for i in range(len(poly))]


def polygon_area(poly: List[Point]) -> float:
    s = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


def point_in_polygon(x: float, y: float, poly: List[Point]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def bezier_point(c: Curve, t: float) -> Point:
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = c
    u = 1 - t
    a, b, cc, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
    return (a * x0 + b * x1 + cc * x2 + d * x3, a * y0 + b * y1 + cc * y2 + d * y3)


def curve_length(c: Curve, steps: int = 8) -> float:
    total = 0.0
    prev = c[0]
    for k in range(1, steps + 1):
        p = bezier_point(c, k / steps)
        total += math.hypot(p[0] - prev[0], p[1] - prev[1])
        prev = p
    return total


def _image_coverage(bboxes: List[BBox], width: float, height: float) -> float:
    if not bboxes or width <= 0 or height <= 0:
        return 0.0
    n = COVERAGE_GRID
    covered = set()
    for x0, y0, x1, y1 in bboxes:
        i0 = max(0, int(math.floor(x0 / width * n)))
        i1 = min(n, int(math.ceil(x1 / width * n)))
        j0 = max(0, int(math.floor(y0 / height * n)))
        j1 = min(n, int(math.ceil(y1 / height * n)))
        for i in range(i0, i1):
            for j in range(j0, j1):
                covered.add(i * n + j)
    return len(covered) / float(n * n)


def _content_bytes(page: "fitz.Page") -> int:
    """Decompressed size of the page's content streams, read before any path becomes a Python object."""
    try:
        return sum(len(page.parent.xref_stream(x) or b"") for x in page.get_contents())
    except Exception:  # noqa: BLE001 — unreadable streams are treated as empty
        return 0


def extract_page(page: "fitz.Page", with_text: bool = True) -> PageGeometry:
    tp, tr = _transformer(page)
    rect = page.rect
    geom = PageGeometry(width=float(rect.width), height=float(rect.height))

    reader = getattr(page, "get_cdrawings", None) or page.get_drawings
    geom.too_complex = _content_bytes(page) > MAX_CONTENT_BYTES
    for path in ([] if geom.too_complex else reader()):
        items = path.get("items") or []
        geom.item_count += len(items)
        chain: List[Point] = []
        chain_ok = True
        for it in items:
            kind = it[0]
            if kind == "l":
                p, q = tp(it[1]), tp(it[2])
                geom.segments.append((p[0], p[1], q[0], q[1]))
                if not chain:
                    chain = [p, q]
                elif _close(chain[-1], p):
                    chain.append(q)
                else:
                    chain_ok = False
            elif kind == "c":
                geom.curves.append((tp(it[1]), tp(it[2]), tp(it[3]), tp(it[4])))
                chain_ok = False
            elif kind == "re":
                r = it[1]
                poly = [tp((r[0], r[1])), tp((r[2], r[1])), tp((r[2], r[3])), tp((r[0], r[3]))]
                geom.segments.extend(_edges(poly))
                geom.polygons.append(poly)
                chain_ok = False
            elif kind == "qu":
                q = it[1]
                poly = [tp(q[0]), tp(q[1]), tp(q[3]), tp(q[2])]
                geom.segments.extend(_edges(poly))
                geom.polygons.append(poly)
                chain_ok = False
        if chain_ok and len(chain) >= 4 and _close(chain[0], chain[-1], 0.5):
            geom.polygons.append(chain[:-1])

    try:
        infos = page.get_image_info(hashes=True)
    except Exception:  # noqa: BLE001 — a malformed image dictionary is no image
        infos = []
    for info in infos:
        bb = tr(info["bbox"])
        geom.image_bboxes.append(
            (max(0.0, bb[0]), max(0.0, bb[1]), min(geom.width, bb[2]), min(geom.height, bb[3]))
        )
        digest = info.get("digest")
        geom.image_digests.append(digest.hex() if isinstance(digest, (bytes, bytearray)) else "")
    geom.image_coverage = _image_coverage(geom.image_bboxes, geom.width, geom.height)

    if with_text:
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []) or []:
                text = "".join(span.get("text", "") for span in line.get("spans", []))
                if text.strip():
                    geom.lines.append(TextLine(text=text, bbox=tr(line["bbox"])))
    return geom


def norm_point(p: Point, geom: PageGeometry) -> dict:
    return {"x": round(p[0] / geom.width, 5), "y": round(p[1] / geom.height, 5)}


def norm_bbox(b: BBox, geom: PageGeometry) -> List[float]:
    return [
        round(b[0] / geom.width, 5),
        round(b[1] / geom.height, 5),
        round(b[2] / geom.width, 5),
        round(b[3] / geom.height, 5),
    ]


def bbox_corners(b: BBox, geom: PageGeometry) -> List[dict]:
    return [norm_point(p, geom) for p in ((b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3]))]


def bbox_of(points: List[Point]) -> BBox:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))
