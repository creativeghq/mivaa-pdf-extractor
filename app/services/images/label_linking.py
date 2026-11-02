"""Link an image on a page several series share to the series its printed label names.

Stage 3 links every image on a page to every product whose pages hold it. The label is read from
the PDF text layer: the line under the image, else above it, else a caption bar inside it, else the
half page's heading when it names one series. Anything else is undecided and keeps its links. An
image linked to one product moves only when its half page names series and none is that product.
"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

BELOW_GAP = 0.06
ABOVE_GAP = 0.05
SIDE_SLACK = 0.02
FURNITURE_PAGES = 10


@dataclass
class Line:
    x0: float
    y0: float
    x1: float
    y1: float
    tokens: Tuple[str, ...]


def tokens_of(text: str) -> Tuple[str, ...]:
    folded = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return tuple(t for t in re.split(r"[^a-z0-9]+", folded) if t)


def _contains(hay: Sequence[str], needle: Sequence[str]) -> bool:
    n = len(needle)
    return n > 0 and any(tuple(hay[i:i + n]) == tuple(needle) for i in range(len(hay) - n + 1))


def series_on(tokens: Sequence[str], series: Dict[str, Tuple[str, ...]]) -> Set[str]:
    """Series named on a line; one named inside a longer one also named ("Wood" in "Wood Espiga") is dropped."""
    found = {name for name, t in series.items() if _contains(tokens, t)}
    return {n for n in found
            if not any(m != n and len(series[m]) > len(series[n]) and _contains(series[m], series[n]) for m in found)}


def _overlaps(line: Line, x: float, w: float) -> bool:
    lo, hi = x - SIDE_SLACK, x + w + SIDE_SLACK
    width = max(line.x1 - line.x0, 1e-6)
    return (min(line.x1, hi) - max(line.x0, lo)) / width >= 0.5


def label_for(bbox: Sequence[float], lines: Sequence[Line], series: Dict[str, Tuple[str, ...]]) -> Optional[str]:
    """The one series this image's label names, or None. bbox and lines share half-page 0..1 coordinates."""
    x, y, w, h = bbox
    bottom = y + h

    def decide(candidates: Iterable[Line]) -> Tuple[bool, Optional[str]]:
        for line in candidates:
            named = series_on(line.tokens, series)
            if len(named) == 1:
                return True, next(iter(named))
            if named:
                return True, None
        return False, None

    near = [l for l in lines if _overlaps(l, x, w)]
    for group in (
        sorted((l for l in near if bottom - 0.01 <= l.y0 <= bottom + BELOW_GAP), key=lambda l: l.y0),
        sorted((l for l in near if y - ABOVE_GAP <= l.y1 <= y + 0.01), key=lambda l: -l.y1),
        sorted((l for l in near if l.y0 >= y + h * 0.6 and l.y1 <= bottom + 0.005), key=lambda l: l.y0),
    ):
        done, name = decide(group)
        if done:
            return name
    on_half = set().union(*(series_on(l.tokens, series) for l in lines)) if lines else set()
    return next(iter(on_half)) if len(on_half) == 1 else None


def names_only_others(lines: Sequence[Line], series: Dict[str, Tuple[str, ...]], own: Optional[str]) -> bool:
    """The half page prints series names and none of them is the image's only product: another series' page."""
    on_half = set().union(*(series_on(l.tokens, series) for l in lines)) if lines else set()
    return own is not None and bool(on_half) and own not in on_half


def half_lines(words: Sequence[Sequence[Any]], page_width: float, page_height: float, side: str,
               furniture: Set[Tuple[str, ...]]) -> List[Line]:
    """PyMuPDF words -> lines on one half (or the whole page), in 0..1 coordinates of that half."""
    split = side in ("left", "right")
    half_w = page_width / 2 if split else page_width
    offset = page_width / 2 if side == "right" else 0.0
    printed: Dict[Tuple[int, int], List[Sequence[Any]]] = defaultdict(list)
    for wd in words:
        printed[(wd[5], wd[6])].append(wd)
    out = []
    for whole in printed.values():
        # Furniture is judged on the whole printed line: a hidden caption can cross the fold.
        if tokens_of(" ".join(str(wd[4]) for wd in whole)) in furniture:
            continue
        ws = [wd for wd in whole if not split or offset <= (wd[0] + wd[2]) / 2 < offset + half_w]
        toks = tokens_of(" ".join(str(wd[4]) for wd in ws))
        if not toks:
            continue
        out.append(Line((min(wd[0] for wd in ws) - offset) / half_w, min(wd[1] for wd in ws) / page_height,
                        (max(wd[2] for wd in ws) - offset) / half_w, max(wd[3] for wd in ws) / page_height, toks))
    return out


def furniture_lines(pages_words: Sequence[Sequence[Sequence[Any]]]) -> Set[Tuple[str, ...]]:
    """Lines printed on FURNITURE_PAGES or more pages: running heads and hidden repeated captions."""
    seen: Dict[Tuple[str, ...], int] = defaultdict(int)
    for words in pages_words:
        lines: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for wd in words:
            lines[(wd[5], wd[6])].append(str(wd[4]))
        for toks in {tokens_of(" ".join(v)) for v in lines.values()}:
            seen[toks] += 1
    return {t for t, n in seen.items() if t and n >= FURNITURE_PAGES}


def _all(query_for, page: int = 1000) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    while True:
        rows = query_for().range(len(out), len(out) + page - 1).execute().data or []
        out.extend(rows)
        if len(rows) < page:
            return out


def _in_chunks(ids: List[str], size: int = 200):
    for i in range(0, len(ids), size):
        yield ids[i:i + size]


async def relink_by_labels(supabase: Any, document_id: str, pdf_bytes: bytes, apply: bool = True) -> Dict[str, Any]:
    from app.services.images.association_scores import PRINTED_LABEL_SCORE
    import fitz
    from app.utils.pdf_to_images import analyze_pdf_layout

    db = getattr(supabase, "client", supabase)
    images = await asyncio.to_thread(_all, lambda: db.table("document_images").select("id, page_number, bbox, metadata, workspace_id")
                                     .eq("document_id", document_id).order("id"))
    images = [i for i in images if i.get("bbox") and "excluded_from_products" not in (i.get("metadata") or {})]
    links: Dict[str, Set[str]] = defaultdict(set)
    for chunk in _in_chunks([i["id"] for i in images]):
        for r in db.table("image_product_associations").select("image_id, product_id").in_("image_id", chunk).execute().data or []:
            links[r["image_id"]].add(r["product_id"])
    by_page: Dict[int, Set[str]] = defaultdict(set)
    for i in images:
        by_page[i["page_number"]] |= links.get(i["id"], set())
    shared = {p for p, prods in by_page.items() if len(prods) > 1}
    summary = {"shared_pages": len(shared), "images": 0, "confirmed": 0, "moved": 0, "undecided": 0}
    if not by_page:
        return summary

    linked = sorted({pid for i in images for pid in links.get(i["id"], set())})
    own = db.table("products").select("id, name, brand_company_id, workspace_id").eq("source_document_id", document_id).execute().data or []
    seed = own + [r for c in _in_chunks(linked) for r in (db.table("products").select("id, name, brand_company_id, workspace_id").in_("id", c).execute().data or [])]
    brands = sorted({p["brand_company_id"] for p in seed if p.get("brand_company_id")})
    pool = list({p["id"]: p for p in seed}.values())
    if brands:
        pool += db.table("products").select("id, name, brand_company_id, workspace_id").in_("brand_company_id", brands) \
            .eq("workspace_id", seed[0]["workspace_id"]).execute().data or []
    ids_by_name: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)
    for p in pool:
        ids_by_name[tokens_of(p["name"])].add(p["id"])
    series = {" ".join(t): t for t, ids in ids_by_name.items() if t and len(ids) == 1}
    product_of = {" ".join(t): next(iter(ids)) for t, ids in ids_by_name.items() if t and len(ids) == 1}
    name_of = {pid: name for name, pid in product_of.items()}

    with tempfile.NamedTemporaryFile(suffix=".pdf") as fh:
        fh.write(pdf_bytes)
        fh.flush()
        layout = await asyncio.to_thread(analyze_pdf_layout, fh.name)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        words = [doc[i].get_text("words") for i in range(doc.page_count)]
        sizes = [(doc[i].rect.width, doc[i].rect.height) for i in range(doc.page_count)]
    finally:
        doc.close()
    furniture = furniture_lines(words)

    for img in images:
        current = links.get(img["id"], set())
        single = len(current) == 1
        if not current or (img["page_number"] not in shared and not single):
            continue
        where = layout.physical_to_pdf_map.get(img["page_number"])
        if not where or where[0] >= len(words):
            continue
        pdf_idx, side = where
        width, height = sizes[pdf_idx]
        lines = half_lines(words[pdf_idx], width, height, side, furniture)
        # One link is settled (a product shot is often labelled elsewhere) unless its page prints only other series.
        if single and not names_only_others(lines, series, name_of.get(next(iter(current)))):
            continue
        summary["images"] += 1
        name = label_for(img["bbox"], lines, series)
        target = product_of.get(name) if name else None
        if not target:
            summary["undecided"] += 1
            continue
        if current == {target}:
            summary["confirmed"] += 1
            continue
        summary["moved"] += 1
        if apply:
            db.table("image_product_associations").upsert(
                {"workspace_id": img.get("workspace_id"), "product_id": target, "image_id": img["id"],
                 "overall_score": PRINTED_LABEL_SCORE,
                 "metadata": {"association_method": "printed_label", "label": name}},
                on_conflict="image_id,product_id").execute()
            stale = [pid for pid in current if pid != target]
            if stale:
                db.table("image_product_associations").delete().eq("image_id", img["id"]).in_("product_id", stale).execute()
    return summary
