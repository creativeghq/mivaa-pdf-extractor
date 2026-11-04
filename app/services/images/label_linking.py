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
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

BELOW_GAP = 0.06
ABOVE_GAP = 0.05
SIDE_SLACK = 0.02
FURNITURE_PAGES = 10
CROWDED_SERIES = 5
# Photos fetched from the maker's site: their page number is a bookkeeping value, not a place on a page.
OFF_PAGE_LAYERS = ("manufacturer_original",)


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


def shares_series(a: Sequence[str], b: Sequence[str]) -> bool:
    """One name is the other plus a qualifier ("Alchemy" / "Alchemy 30x90"): the catalogue prints both as the shorter."""
    n = min(len(a), len(b))
    return n > 0 and len(a) != len(b) and tuple(a[:n]) == tuple(b[:n])


def series_siblings(names: Dict[str, Tuple[str, ...]], product_of: Dict[str, str], own: Set[str]) -> Dict[str, List[str]]:
    """{printed series: every product of this document it can mean} where more than one shares it.

    Only this document's products have page ranges to tell them apart; others keep their one-name mapping.
    """
    out = {}
    for s, st in names.items():
        if product_of[s] not in own:
            continue
        sharing = [product_of[s]] + [product_of[t] for t, tt in names.items()
                                     if product_of[t] in own and len(tt) > len(st) and shares_series(st, tt)]
        if len(sharing) > 1:
            out[s] = sharing
    return out


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


def sole_series(lines: Sequence[Line], series: Dict[str, Tuple[str, ...]]) -> Optional[str]:
    """The one series a half page names, for an image with no position on it (a raw embedded raster)."""
    on_half = set().union(*(series_on(l.tokens, series) for l in lines)) if lines else set()
    return next(iter(on_half)) if len(on_half) == 1 else None


def names_only_others(lines: Sequence[Line], series: Dict[str, Tuple[str, ...]], own: Optional[str]) -> bool:
    """The half page prints a few series names, none of them the image's only product: another series' page.

    A page naming CROWDED_SERIES or more is an index or overview, not one series' page.
    """
    on_half = set().union(*(series_on(l.tokens, series) for l in lines)) if lines else set()
    return own is not None and 0 < len(on_half) < CROWDED_SERIES and own not in on_half


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


@dataclass
class LabelContext:
    db: Any
    images: List[Dict[str, Any]]
    links: Dict[str, Set[str]]
    series: Dict[str, Tuple[str, ...]]
    product_of: Dict[str, str]
    layout: Any
    words: List[Any]
    sizes: List[Tuple[float, float]]
    furniture: Set[Tuple[str, ...]]
    pages_of: Dict[str, Set[int]] = field(default_factory=dict)
    siblings: Dict[str, List[str]] = field(default_factory=dict)

    def product_for(self, name: Optional[str], page: int) -> Optional[str]:
        """The product a printed series name means on this page; shared by several, the one whose own pages hold it."""
        if not name:
            return None
        sharing = self.siblings.get(name)
        if not sharing:
            return self.product_of.get(name)
        holding = [pid for pid in sharing if page in self.pages_of.get(pid, set())]
        return holding[0] if len(holding) == 1 else None

    def printed_as(self, product_id: str) -> Optional[str]:
        """The series name the catalogue prints for a product: the shared shorter name for a qualified sibling."""
        for s, sharing in self.siblings.items():
            if product_id in sharing:
                return s
        return next((n for n, pid in self.product_of.items() if pid == product_id), None)

    def lines_for(self, page_number: int) -> Optional[Tuple[int, str, List[Line]]]:
        """(pdf page index, side, the half page's lines) for a physical page, or None."""
        where = self.layout.physical_to_pdf_map.get(page_number)
        if not where or where[0] >= len(self.words):
            return None
        pdf_idx, side = where
        width, height = self.sizes[pdf_idx]
        return pdf_idx, side, half_lines(self.words[pdf_idx], width, height, side, self.furniture)


async def label_context(supabase: Any, document_id: str, pdf_bytes: bytes) -> Optional[LabelContext]:
    """The document's live images, their links, the series names in play and the PDF text layer."""
    import fitz
    from app.utils.pdf_to_images import analyze_pdf_layout

    db = getattr(supabase, "client", supabase)
    images = await asyncio.to_thread(_all, lambda: db.table("document_images").select("id, page_number, bbox, metadata, workspace_id, image_type, extraction_layer")
                                     .eq("document_id", document_id).order("id"))
    images = [i for i in images if "excluded_from_products" not in (i.get("metadata") or {})]
    if not images:
        return None
    links: Dict[str, Set[str]] = defaultdict(set)
    for chunk in _in_chunks([i["id"] for i in images]):
        for r in db.table("image_product_associations").select("image_id, product_id").in_("image_id", chunk).execute().data or []:
            links[r["image_id"]].add(r["product_id"])

    linked = sorted({pid for i in images for pid in links.get(i["id"], set())})
    own = db.table("products").select("id, name, brand_company_id, workspace_id, metadata").eq("source_document_id", document_id).execute().data or []
    seed = own + [r for c in _in_chunks(linked) for r in (db.table("products").select("id, name, brand_company_id, workspace_id").in_("id", c).execute().data or [])]
    if not seed:
        return None
    brands = sorted({p["brand_company_id"] for p in seed if p.get("brand_company_id")})
    pool = list({p["id"]: p for p in seed}.values())
    if brands:
        pool += db.table("products").select("id, name, brand_company_id, workspace_id").in_("brand_company_id", brands)             .eq("workspace_id", seed[0]["workspace_id"]).execute().data or []
    ids_by_name: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)
    for p in pool:
        ids_by_name[tokens_of(p["name"])].add(p["id"])
    series = {" ".join(t): t for t, ids in ids_by_name.items() if t and len(ids) == 1}
    product_of = {" ".join(t): next(iter(ids)) for t, ids in ids_by_name.items() if t and len(ids) == 1}
    pages_of = {p["id"]: {int(n) for n in ((p.get("metadata") or {}).get("page_range") or [])
                          if isinstance(n, (int, float)) and not isinstance(n, bool)} for p in own}

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
    return LabelContext(db, images, links, series, product_of, layout, words, sizes, furniture_lines(words),
                        pages_of, series_siblings(series, product_of, set(pages_of)))


async def _decisions(supabase: Any, document_id: str, pdf_bytes: bytes) -> Tuple[Any, int, List[Dict[str, Any]]]:
    """Every image the printed labels have a say on, with the series its label names (or None)."""
    ctx = await label_context(supabase, document_id, pdf_bytes)
    if ctx is None:
        return getattr(supabase, "client", supabase), 0, []
    images, links, series = ctx.images, ctx.links, ctx.series
    by_page: Dict[int, Set[str]] = defaultdict(set)
    for i in images:
        by_page[i["page_number"]] |= links.get(i["id"], set())
    shared = {p for p, prods in by_page.items() if len(prods) > 1}

    out: List[Dict[str, Any]] = []
    for img in images:
        if img.get("extraction_layer") in OFF_PAGE_LAYERS:
            continue
        current = links.get(img["id"], set())
        single = len(current) == 1
        if not current or (img["page_number"] not in shared and not single) or (not img.get("bbox") and not single):
            continue
        half = ctx.lines_for(img["page_number"])
        if not half:
            continue
        lines = half[2]
        own_name = ctx.printed_as(next(iter(current))) if single else None
        # One link is settled (a product shot is often labelled elsewhere) unless its page prints only other series.
        if single and not names_only_others(lines, series, own_name):
            continue
        name = label_for(img["bbox"], lines, series) if img.get("bbox") else sole_series(lines, series)
        out.append({"image": img, "current": current, "name": name, "target": ctx.product_for(name, img["page_number"]),
                    "only_others": single, "product": own_name,
                    "printed": sorted(set().union(*(series_on(l.tokens, series) for l in lines))) if single else []})
    return ctx.db, len(shared), out


async def relink_by_labels(supabase: Any, document_id: str, pdf_bytes: bytes, apply: bool = True) -> Dict[str, Any]:
    from app.services.images.association_scores import PRINTED_LABEL_SCORE

    db, shared, decisions = await _decisions(supabase, document_id, pdf_bytes)
    summary = {"shared_pages": shared, "images": len(decisions), "confirmed": 0, "moved": 0, "undecided": 0}
    for d in decisions:
        img, current, target, name = d["image"], d["current"], d["target"], d["name"]
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


async def other_series_images(supabase: Any, document_id: str, pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Images held by one product on a half page that prints only other series' names."""
    _, _, decisions = await _decisions(supabase, document_id, pdf_bytes)
    return [{"image_id": d["image"]["id"], "page": d["image"]["page_number"], "product": d["product"],
             "printed": d["printed"], "label": d["name"]} for d in decisions if d["only_others"]]



def series_in_file_name(url: str, series: Dict[str, Tuple[str, ...]]) -> Optional[str]:
    """The series a maker's photo is named after: `CHEZ-COCO-STAR-en-x.jpg` is Chez Coco. Longest name wins."""
    slug = re.split(r"[-_]+", tokens_slug(url))
    best: Optional[str] = None
    for name, tokens in series.items():
        if tokens and tuple(slug[:len(tokens)]) == tuple(tokens) and (best is None or len(tokens) > len(series[best])):
            best = name
    return best


def tokens_slug(url: str) -> str:
    stem = (url or "").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    normalized = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9_-]+", "-", normalized)


async def reattribute_site_photos(supabase: Any, document_id: str, *, apply: bool = True, job_id: Optional[str] = None) -> Dict[str, Any]:
    """A maker's photo named after another series of this brand belongs to that series, not where it was linked."""
    from app.services.images.association_scores import PRINTED_LABEL_SCORE
    from app.services.pdf.ingestion_corrections import CORRECTED, Correction, record

    db = getattr(supabase, "client", supabase)
    products = _all(lambda: db.table("products").select("id, name, workspace_id, brand_company_id").eq("source_document_id", document_id).order("id"))
    summary: Dict[str, Any] = {"moved": 0, "per_image": []}
    if not products:
        return summary
    workspace = products[0]["workspace_id"]
    known = list(products)
    for brand in {p["brand_company_id"] for p in products if p.get("brand_company_id")}:
        known += _all(lambda: db.table("products").select("id, name, workspace_id, brand_company_id").eq("workspace_id", workspace).eq("brand_company_id", brand).order("id"))
    ids = {p["name"]: p["id"] for p in known}
    series = {name: tuple(tokens_slug(name).replace("_", "-").split("-")) for name in ids}
    moved_images: List[str] = []
    for product in products:
        links = _all(lambda: db.table("image_product_associations").select("image_id, document_images!inner(id, extraction_layer, metadata)")
                     .eq("product_id", product["id"]).eq("document_images.extraction_layer", "manufacturer_original").order("image_id"))
        for link in links:
            meta = link["document_images"].get("metadata") or {}
            named = series_in_file_name(meta.get("source_url") or "", series)
            own = series.get(product["name"]) or ()
            # BONN-PELDANO on the Peldaño series is a step piece of Bonn: it names both, and stays.
            on_own = _contains(re.split(r"[-_]+", tokens_slug(meta.get("source_url") or "")), own)
            if not named or named == product["name"] or on_own or "excluded_from_products" in meta:
                continue
            summary["moved"] += 1
            summary["per_image"].append({"image": link["image_id"], "from": product["name"], "to": named})
            moved_images.append(link["image_id"])
            if apply:
                db.table("image_product_associations").upsert(
                    {"workspace_id": workspace, "product_id": ids[named], "image_id": link["image_id"], "overall_score": PRINTED_LABEL_SCORE,
                     "metadata": {"association_method": "site_file_name", "label": named}},
                    on_conflict="image_id,product_id").execute()
                db.table("image_product_associations").delete().eq("image_id", link["image_id"]).eq("product_id", product["id"]).execute()
    if apply and summary["moved"]:
        await record(db, document_id, workspace, "image_linking", [Correction(
            "site_photos_moved_to_named_series", CORRECTED,
            f"{summary['moved']} maker photo(s) moved to the series their file names", summary["moved"], moved_images)], job_id)
    return summary
