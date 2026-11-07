"""Deterministic colour swatch sampling from catalogue PDF page geometry.

Hex comes from the median of the rendered swatch region, never from a model. Every refusal
carries a reason so a colour left without hex is explained rather than silently skipped.
Top-level imports are stdlib only; fitz is imported inside sample_page.
"""

import re
import statistics
import unicodedata
from typing import Optional

HEX_SOURCE = "median of catalogue photo (estimate)"
MAX_SPREAD = 35.0
TIE_TOLERANCE = 6
CAPTION_MAX_GAP = 30.0
SHRINK = 0.20
DPI = 100
MIN_SIDE = 10.0
MAX_WIDTH = 270.0
MAX_HEIGHT = 190.0
REFERENCE_LEAF = 595.0
SPREAD_ASPECT = 1.9

_SIZE_TOKEN = re.compile(r"^\d+([x×]\d+)?")

PATTERN_WORDS = (
    "structure", "struttura", "decor", "patch", "peonies", "victorian", "floral", "stardust",
    "wings", "plume", "grove", "blossom", "comp",
)


def median_hex(pixels) -> str:
    chans = list(zip(*pixels))
    if len(chans) != 3:
        raise ValueError("pixels must be RGB triples")
    return "#%02x%02x%02x" % tuple(int(round(statistics.median(c))) for c in chans)


def spread(pixels) -> float:
    chans = list(zip(*pixels))
    if len(chans) != 3:
        raise ValueError("pixels must be RGB triples")
    return sum(statistics.pstdev(c) for c in chans) / 3.0


def is_pattern_name(name: str) -> bool:
    low = (name or "").lower()
    return any(w in low for w in PATTERN_WORDS)


def _center(rect):
    return (rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0


def _inside(point, rect) -> bool:
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def _reading_order(words):
    return sorted(words, key=lambda w: (round(w["rect"][1] / 4.0), w["rect"][0]))


def pair_by_containment(words, images):
    """Pair each image with the words whose centre lies inside its bbox.

    A word inside two images is ambiguous and pairs with neither. Images holding no word are
    omitted. Returns a list of {label, image}.
    """
    owners = {}
    for wi, w in enumerate(words):
        hits = [ii for ii, im in enumerate(images) if _inside(_center(w["rect"]), im["rect"])]
        if len(hits) == 1:
            owners.setdefault(hits[0], []).append(w)
    out = []
    for ii in sorted(owners):
        label = " ".join(w["text"] for w in _reading_order(owners[ii]))
        out.append({"label": label, "image": images[ii]})
    return out


def pair_by_caption_below(words, images, max_gap=CAPTION_MAX_GAP):
    """Pair each image with the nearest line of words below it, centred within its x-range.

    Returns a list of {label, image}; an image with no caption within max_gap is omitted.
    """
    out = []
    for im in images:
        x0, _, x1, y1 = im["rect"]
        cands = []
        for w in words:
            cx, _ = _center(w["rect"])
            gap = w["rect"][1] - y1
            if x0 <= cx <= x1 and -2.0 <= gap <= max_gap:
                cands.append((gap, w))
        if not cands:
            continue
        nearest = min(g for g, _ in cands)
        line = [w for g, w in cands if g - nearest <= 3.0]
        line.sort(key=lambda w: w["rect"][0])
        texts = []
        for w in line:
            if _SIZE_TOKEN.match(w["text"]):
                break
            texts.append(w["text"])
        if texts:
            out.append({"label": " ".join(texts), "image": im})
    return out


def _hex_close(a: str, b: str, tol: int = TIE_TOLERANCE) -> bool:
    ra, rb = int(a[1:], 16), int(b[1:], 16)
    return all(abs(((ra >> s) & 255) - ((rb >> s) & 255)) <= tol for s in (16, 8, 0))


def fold(text: str) -> str:
    """Lowercase, accents stripped, punctuation to spaces, whitespace collapsed."""
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


def _series_caption(label: str, keys, series: str) -> bool:
    """A caption names THIS colour of THIS series: removing the colour leaves nothing or the series.

    On a page holding several series "Ferrara Gris" must never colour Amalfi, and a caption naming
    two colours ("Amalfi Beige | Gris") is one photo of both, so it colours neither.
    """
    low = fold(label)
    for k in sorted({fold(k) for k in keys}, key=len, reverse=True):
        if k and re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", low):
            rest = " ".join(re.sub(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", " ", low).split())
            return rest in ("", fold(series))
    return False


def _candidates(keys, samples, series=None):
    if series:
        return [s for s in samples if _series_caption(s.get("label") or "", keys, series)]
    exact = [s for s in samples if (s.get("label") or "").strip().lower() in keys]
    if exact:
        return exact
    found = []
    for s in samples:
        low = (s.get("label") or "").lower()
        if any(re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", low) for k in keys):
            found.append(s)
    return found


def merge_hex_into_colour_options(options, samples, series=None):
    """Return (new_options, report). Entries that already have a hex are never touched.

    With `series`, a sample counts only when its caption is that series' own caption for the colour.
    """
    new_options = []
    report = []
    for opt in options or []:
        name = opt.get("name") or opt.get("catalogue_name") or ""
        if opt.get("hex"):
            new_options.append(opt)
            report.append({"name": name, "status": "skipped", "reason": "has_hex"})
            continue
        keys = {k.strip().lower() for k in (opt.get("name"), opt.get("catalogue_name")) if k and k.strip()}
        verdict = _decide(opt, keys, samples, series)
        if verdict.get("hex"):
            new_options.append({**opt, "hex": verdict["hex"], "hex_source": HEX_SOURCE})
            report.append({"name": name, "status": "applied", "hex": verdict["hex"],
                           "spread": verdict["spread"], "layout": verdict.get("layout")})
        else:
            new_options.append(opt)
            report.append({"name": name, "status": "skipped", "reason": verdict["reason"]})
    return new_options, report


def _decide(opt, keys, samples, series=None) -> dict:
    if any(is_pattern_name(k) for k in keys):
        return {"reason": "pattern_name"}
    cands = _candidates(keys, samples, series)
    if not cands:
        return {"reason": "no_sample"}
    usable = [c for c in cands if c.get("hex")]
    if not usable:
        return {"reason": cands[0].get("reason") or "no_sample"}
    if any(not _hex_close(usable[0]["hex"], c["hex"]) for c in usable[1:]):
        return {"reason": "ambiguous"}
    best = usable[0]
    if best.get("reason"):
        return {"reason": best["reason"]}
    if best.get("spread", 0.0) > MAX_SPREAD:
        return {"reason": "patterned"}
    return best


def leaf_scale(page_width: float, page_height: float) -> float:
    """How much larger than an A4 leaf this page is; a double-page spread counts as two leaves."""
    leaf = page_width / 2.0 if page_height and page_width > SPREAD_ASPECT * page_height else page_width
    return max(1.0, leaf / REFERENCE_LEAF)


def _swatch_candidates(infos, scale: float = 1.0):
    out = []
    for info in infos:
        x0, y0, x1, y1 = info["bbox"]
        w, h = x1 - x0, y1 - y0
        if MIN_SIDE <= w <= MAX_WIDTH * scale and 8.0 <= h <= MAX_HEIGHT * scale:
            out.append({"rect": (x0, y0, x1, y1)})
    return out


def _dedupe(images):
    seen, out = set(), []
    for im in images:
        key = tuple(round(v) for v in im["rect"])
        if key not in seen:
            seen.add(key)
            out.append(im)
    return out


def _region_pixels(page, rect, fitz):
    x0, y0, x1, y1 = rect
    dx, dy = (x1 - x0) * SHRINK, (y1 - y0) * SHRINK
    clip = fitz.Rect(x0 + dx, y0 + dy, x1 - dx, y1 - dy)
    pix = page.get_pixmap(dpi=DPI, clip=clip, colorspace=fitz.csRGB, alpha=False)
    data = pix.samples
    n = pix.n
    return [(data[i], data[i + 1], data[i + 2]) for i in range(0, len(data) - n + 1, n)]


def sample_page(pdf_path, page_number: int, doc=None) -> list:
    """Sample every labelled swatch on a 1-based page.

    Returns [{label, rect, hex, spread, layout, reason?}]; a pattern name or high spread keeps
    the measured hex but carries a reason.
    """
    import fitz

    own = doc is None
    doc = doc or fitz.open(str(pdf_path))
    try:
        page = doc[page_number - 1]
        scale = leaf_scale(page.rect.width, page.rect.height)
        images = _dedupe(_swatch_candidates(page.get_image_info(), scale))
        words = [{"text": w[4], "rect": (w[0], w[1], w[2], w[3])} for w in page.get_text("words")]
        paired = [(p, "containment") for p in pair_by_containment(words, images)]
        taken = {id(p["image"]) for p, _ in paired}
        rest = [im for im in images if id(im) not in taken]
        paired += [(p, "caption_below") for p in pair_by_caption_below(words, rest)]
        out = []
        for p, layout in paired:
            pixels = _region_pixels(page, p["image"]["rect"], fitz)
            if not pixels:
                continue
            spr = spread(pixels)
            row = {"label": p["label"], "rect": [round(v, 1) for v in p["image"]["rect"]],
                   "hex": median_hex(pixels), "spread": round(spr, 1), "layout": layout}
            reason: Optional[str] = None
            if is_pattern_name(p["label"]):
                reason = "pattern_name"
            elif spr > MAX_SPREAD:
                reason = "patterned"
            if reason:
                row["reason"] = reason
            out.append(row)
        return out
    finally:
        if own:
            doc.close()
