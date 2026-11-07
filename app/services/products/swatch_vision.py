"""Find a colour's tile on its catalogue page when no caption pairs with it, then measure it.

The model only answers WHICH numbered region shows a colour of the series; the hex is still the
median of that region's own pixels. A tile partly covered by a later-drawn one (IZIDA stacks
WHITE over GREY) is measured on its visible part only. Pure helpers here; fitz/PIL stay inside
the functions that need them.
"""

import base64
import io
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.services.products import swatch_sampler as S

TASK = "colour_swatch_locate"
HEX_SOURCE = "median of catalogue tile located by vision (estimate)"
MIN_SIDE = 18.0
MAX_LEAF_SHARE = 0.45
MAX_LEAF_SPAN = 0.8
NESTED_SHARE = 0.7
MAX_REGIONS = 40
RENDER_LONG_SIDE = 1400
SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["colours"],
    "properties": {"colours": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["colour", "regions"],
        "properties": {"colour": {"type": "string"}, "regions": {"type": "array", "items": {"type": "integer"}}}}}},
}

Rect = Tuple[float, float, float, float]


def _area(r: Rect) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _overlap(a: Rect, b: Rect) -> Optional[Rect]:
    o = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return o if o[2] > o[0] and o[3] > o[1] else None


def visible_rect(rect: Rect, occluders: Sequence[Rect]) -> Optional[Rect]:
    """The largest axis-aligned piece of `rect` that no occluder covers, or None."""
    cur = rect
    for occ in occluders:
        o = _overlap(cur, occ)
        if not o:
            continue
        pieces = [(cur[0], cur[1], cur[2], o[1]), (cur[0], o[3], cur[2], cur[3]),
                  (cur[0], cur[1], o[0], cur[3]), (o[2], cur[1], cur[2], cur[3])]
        best = max(pieces, key=_area)
        if _area(best) <= 0:
            return None
        cur = best
    return cur if min(cur[2] - cur[0], cur[3] - cur[1]) >= MIN_SIDE / 2 else None


def candidate_regions(image_rects: Sequence[Rect], leaf: Rect) -> List[Dict[str, Any]]:
    """Image blocks on the leaf that could be a tile, in draw order, each with its visible part.

    A frame drawn round a picture (IZIDA's 214pt border around a 199pt chip) is the same tile, so
    the outer of two nested near-equal rects is dropped. Full-bleed room shots are dropped by size.
    """
    leaf_area = _area(leaf)
    inside = []
    for r in image_rects:
        c = ((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0)
        if not (leaf[0] <= c[0] <= leaf[2] and leaf[1] <= c[1] <= leaf[3]):
            continue
        if min(r[2] - r[0], r[3] - r[1]) < MIN_SIDE or _area(r) > MAX_LEAF_SHARE * leaf_area:
            continue
        if r[2] - r[0] > MAX_LEAF_SPAN * (leaf[2] - leaf[0]) or r[3] - r[1] > MAX_LEAF_SPAN * (leaf[3] - leaf[1]):
            continue
        inside.append(tuple(r))
    kept = []
    for i, r in enumerate(inside):
        nested = any(j != i and _overlap(r, o) == o and _area(o) >= NESTED_SHARE * _area(r) and o != r
                     for j, o in enumerate(inside))
        if not nested and r not in kept:
            kept.append(r)
    out = []
    for i, r in enumerate(kept):
        vis = visible_rect(r, kept[i + 1:])
        if vis:
            out.append({"rect": r, "visible": vis})
    return out[:MAX_REGIONS]


def picked_regions(answer: Dict[str, Any], colours: Sequence[str], n_regions: int) -> Dict[str, List[int]]:
    """{colour: [0-based region]} for answers naming a colour we asked about and a real region."""
    wanted = {S.fold(c): c for c in colours}
    out: Dict[str, List[int]] = {}
    for row in answer.get("colours") or []:
        name = wanted.get(S.fold(row.get("colour") or ""))
        if not name:
            continue
        idx = [i - 1 for i in row.get("regions") or [] if isinstance(i, int) and 1 <= i <= n_regions]
        if idx:
            out[name] = list(dict.fromkeys(out.get(name, []) + idx))
    return out


def annotated_leaf(page: Any, leaf: Rect, regions: Sequence[Dict[str, Any]]) -> bytes:
    """PNG of the leaf with every region outlined and numbered from 1."""
    import fitz
    from PIL import Image, ImageDraw

    zoom = RENDER_LONG_SIDE / max(leaf[2] - leaf[0], leaf[3] - leaf[1])
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=fitz.Rect(*leaf), colorspace=fitz.csRGB, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(img)
    write = draw.text
    for n, reg in enumerate(regions, start=1):
        x0, y0, x1, y1 = [(v - leaf[k % 2]) * zoom for k, v in enumerate(reg["visible"])]
        draw.rectangle((x0, y0, x1, y1), outline=(230, 0, 0), width=3)
        draw.rectangle((x0, y0, x0 + 34, y0 + 24), fill=(230, 0, 0))
        write((x0 + 6, y0 + 5), str(n), fill=(255, 255, 255))
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


def measure(page: Any, rect: Rect) -> Dict[str, Any]:
    import fitz

    pixels = S._region_pixels(page, rect, fitz)
    if not pixels:
        return {}
    return {"hex": S.median_hex(pixels), "spread": round(S.spread(pixels), 1)}


async def locate(page: Any, leaf: Rect, regions: Sequence[Dict[str, Any]], *, series: str, colours: Sequence[str],
                 prompt: str, model: str, workspace_id: Optional[str], product_id: Optional[str],
                 job_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, List[int]]:
    from app.services.core.claude_tool_call import call_with_schema
    from app.services.utilities.prompt_registry import render

    png = annotated_leaf(page, leaf, regions)
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png).decode()}},
        {"type": "text", "text": render(prompt, series=series, colours=", ".join(colours), n_regions=str(len(regions)))},
    ]
    res = await call_with_schema(task=TASK, model=model, max_tokens=2000, workspace_id=workspace_id, user_id=user_id,
                                 product_id=product_id, job_id=job_id,
                                 messages=[{"role": "user", "content": content}],
                                 output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
    return picked_regions(res.data, colours, len(regions))


def apply_located(options: List[Dict[str, Any]], measured: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Give each still-unmeasured colour the hex of its located tile(s); every distinct tile is kept."""
    out = []
    for opt in options:
        hits = [m for m in measured.get(opt.get("name") or "", []) if m.get("hex")]
        if opt.get("hex") or not hits:
            out.append(opt)
            continue
        clean = [m for m in hits if m["spread"] <= S.MAX_SPREAD]
        if not clean:
            out.append({**opt, "hex_status": "patterned"})
            continue
        entry = {k: v for k, v in opt.items() if k not in ("hex_status", "hex_checked_at")}
        entry.update({"hex": clean[0]["hex"], "hex_source": HEX_SOURCE})
        alternates = S.distinct_hexes([m["hex"] for m in clean[1:]], exclude=[clean[0]["hex"]])
        if alternates:
            entry["hex_alternates"] = alternates
        out.append(entry)
    return out
