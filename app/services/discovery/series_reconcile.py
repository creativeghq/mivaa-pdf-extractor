"""Reconcile one series against everything its sources print about it.

A model sorts the words the catalogue and the maker's site use for the series into colours, decors,
finishes and formats; code keeps only the values that appear literally in those sources. The
description is assembled from the kept values, plus the maker's own page text when we have it.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)

TASK = "series_reconcile"
SIZE = re.compile(r"(\d{1,3}(?:[.,] ?\d(?!\d))?)\s*[x×X]\s*(\d{1,3}(?:[.,] ?\d(?!\d))?)")
LATEX_SIZE = re.compile(r"\$\s*([\d.,]+)\s*\\+times\s*([\d.,]+)\s*\$")
CORPUS_LIMIT = 12000
CROWDED_PAGE_SERIES = 5
SLUG_NOISE = r"\d+|\d+x\d+|[a-z]|[a-z0-9]{12,}|en|es|fr|de|it|pt|amb|ambi|ambient|f\d+|copy|scaled|retouch|\d+dpi"
REFUSAL = re.compile(
    r"^(the )?(source|catalogue|catalog)( text)?( for \S+)? (contains|does not|has|provides|offers|consists)"
    r"|no (real |narrative )?description|there is no description|nothing to (write|describe)", re.I)

OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["colours", "decors", "finishes", "formats", "uses"],
    "properties": {
        "colours": {"type": "array", "items": {"type": "string"}},
        "decors": {"type": "array", "items": {"type": "string"}},
        "finishes": {"type": "array", "items": {"type": "string"}},
        "formats": {"type": "array", "items": {"type": "string"}},
        "uses": {"type": "array", "items": {"type": "string", "enum": ["floor", "wall", "outdoor", "stair"]}},
    },
}


def fold(text: Optional[str]) -> str:
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()


def normalize_text(text: str) -> str:
    return LATEX_SIZE.sub(r"\1x\2", text or "")


def size_key(a: str, b: str) -> str:
    x, y = (float(v.replace(" ", "").replace(",", ".")) for v in (a, b))
    lo, hi = sorted((x, y))
    fmt = lambda v: str(int(v)) if v == int(v) else str(v)
    return f"{fmt(lo)}x{fmt(hi)}"


def nominal_of(size: str, other: str) -> bool:
    """"30x60" beside the printed actual "30.3x60.6": the same tile under its rounded name."""
    a, b = [float(v) for v in size.split("x")], [float(v) for v in other.split("x")]
    return size != other and all(v == int(v) for v in a) and all(abs(x - y) < 1 for x, y in zip(a, b))


def sizes_in(text: str) -> Set[str]:
    return {size_key(*m.groups()) for m in SIZE.finditer(normalize_text(text))}


def _word(name: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(fold(name)) + r"\b")


def series_pages(pages: Dict[int, str], name: str, all_names: Sequence[str] = ()) -> Dict[int, str]:
    """Pages that print the series name followed, within a line or the next, by a size: its own pages.

    A page naming CROWDED_PAGE_SERIES or more series (index, packing table) is catalogue-wide, not its own.
    """
    pat = re.compile(r"\b" + re.escape(fold(name)) + r"\b[^\n]{0,60}?(?:\n[^\n]{0,30}?)?\d{1,3}(?:[.,]\d)?\s*[x×]\s*\d{1,3}")
    others = [_word(n) for n in all_names if len(fold(n)) > 3]
    out = {}
    for pg, text in pages.items():
        folded = fold(normalize_text(text))
        if pat.search(folded) and sum(1 for o in others if o.search(folded)) < CROWDED_PAGE_SERIES:
            out[pg] = text
    return out


def own_pages(pages: Dict[int, str], name: str, all_names: Sequence[str], page_range: Iterable[Any] = ()) -> Dict[int, str]:
    """The series' own pages: by name-then-size, else the discovered range's pages that name it.

    A catalogue that sets the name as a heading and the sizes in a table further down prints no size
    beside the name, so the first test finds nothing and every series would be skipped.
    """
    found = series_pages(pages, name, all_names)
    if found:
        return found
    word = _word(name)
    others = [_word(n) for n in all_names if len(fold(n)) > 3]
    out = {}
    for pg in page_range or ():
        try:
            pg = int(pg)
        except (TypeError, ValueError):
            continue
        folded = fold(normalize_text(pages.get(pg, "")))
        if word.search(folded) and sum(1 for o in others if o.search(folded)) < CROWDED_PAGE_SERIES:
            out[pg] = pages[pg]
    return out


SLIP_MIN_LEN = 6


def one_slip(a: str, b: str) -> bool:
    """The same word but for one typed slip (a wrong, missing, extra or swapped letter): "natutal" is "natural".

    Words under SLIP_MIN_LEN letters must match exactly; there a slip is as likely another colour.
    """
    if a == b:
        return True
    if min(len(a), len(b)) < SLIP_MIN_LEN or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    short, long_ = sorted((a, b), key=len)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


ROOM_SCENE_MARKERS = {"amb", "ambi", "ambient", "ambiente", "ambienti"}


def is_room_scene(url: str) -> bool:
    """A room photo TITLED as one (`.../ambienti/AMB_XS_METAFISICO...jpg`): the rest of its name is the scene, not a variant.

    A trailing marker (`SKYATOS-GOLS-60X120-AMB.jpg`) still names the variant first, so it is not this.
    """
    path = fold(url or "").split("?", 1)[0]
    folders = set(re.split(r"[/]+", path.rsplit("/", 1)[0]))
    head = re.split(r"[-_.]+", path.rsplit("/", 1)[-1])[0]
    return head in ROOM_SCENE_MARKERS or bool(folders & ROOM_SCENE_MARKERS)


def site_variant_names(urls: Iterable[str], name: str) -> List[str]:
    """`SABOYA-OLAS-BEIGE-en-t6VT….jpg` -> "olas beige": what the maker calls the variant it photographed."""
    head = fold(name).split()
    out = []
    for url in urls:
        slug = fold((url or "").rsplit("/", 1)[-1])
        slug = re.sub(r"\.\w+$", "", slug)
        slug = re.sub(r"-(en|es|fr|de|it)-[a-z0-9]{8,}$|-\d+x\d+$|-scaled$", "", slug)
        parts = [p for p in re.split(r"[-_.]+", slug) if p and not re.fullmatch(SLUG_NOISE, p)]
        if parts[:len(head)] == head and len(parts) > len(head):
            out.append(" ".join(parts[len(head):]))
    return sorted(set(out))


def verify(result: Dict[str, Any], corpus: str, site_names: Sequence[str]) -> Dict[str, List[str]]:
    """Keep a value only when the sources print it; a format only when its size is printed."""
    hay = fold(corpus) + "\n" + "\n".join(site_names)
    printed_sizes = sizes_in(corpus)

    def present(values: Iterable[str]) -> List[str]:
        kept: List[str] = []
        for v in values or []:
            f = " ".join(fold(v).split())
            if f and re.search(r"\b" + re.escape(f) + r"\b", hay) and f not in kept:
                kept.append(f)
        return kept

    formats: List[str] = []
    for v in result.get("formats") or []:
        for s in sizes_in(v):
            if s in printed_sizes and s not in formats:
                formats.append(s)
    formats = [f for f in formats if not any(nominal_of(f, other) for other in formats)]
    return {"colours": present(result.get("colours")), "decors": present(result.get("decors")),
            "finishes": present(result.get("finishes")), "formats": formats,
            "uses": [u for u in (result.get("uses") or []) if u in ("floor", "wall", "outdoor", "stair")]}


def leaked_values(attributes: Dict[str, Any], other_names: Iterable[str]) -> Dict[str, List[str]]:
    """Attribute list entries that name another series of the same catalogue."""
    pats = [_word(n) for n in other_names if len(fold(n)) > 3]
    out: Dict[str, List[str]] = {}
    for key, value in (attributes or {}).items():
        values = value if isinstance(value, list) else [value]
        bad = [v for v in values if isinstance(v, str) and any(p.search(fold(v)) for p in pats)]
        if bad:
            out[key] = bad
    return out


def _join(items: Sequence[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def factual_line(name: str, material: Optional[str], uses: Sequence[str], formats: Sequence[str],
                 colours: Sequence[str], decors: Sequence[str]) -> str:
    """"Saboya: porcelain floor and wall tile in 33x33 and 25x50 cm, in Crema and Gris, with an Olas decor." """
    surfaces = [u for u in uses if u != "stair"]
    kind = " ".join(x for x in [material, _join(surfaces) if surfaces else None, "tile"] if x)
    parts = [f"{name}: {kind}"]
    if formats:
        parts.append(f"in {_join(formats)} cm")
    if colours:
        parts.append(f"in {_join([c.title() for c in colours])}")
    line = ", ".join(parts)
    if decors:
        line += f", with {'an' if decors[0][0] in 'aeiou' else 'a'} {_join([d.title() for d in decors])} decor" + ("s" if len(decors) > 1 else "")
    return line + "."


def compose_description(line: str, site_text: Optional[str]) -> str:
    return f"{line}\n\n{site_text.strip()}" if site_text and site_text.strip() else line


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _material(product: Dict[str, Any]) -> Optional[str]:
    m, a = product.get("metadata") or {}, product.get("attributes") or {}
    for v in (a.get("material"), m.get("material"), m.get("body_type"), m.get("tile_type")):
        v = v[0] if isinstance(v, list) and v else v
        if isinstance(v, str) and fold(v) in ("porcelain", "ceramic", "stoneware", "porcelain stoneware", "red body", "white body"):
            return fold(v)
    return None


async def site_description(product_name: str, site_root: Optional[str], user_id: Optional[str],
                           workspace_id: Optional[str]) -> Optional[Dict[str, str]]:
    """The maker's own text for the series, from a search result whose page slug is the series name."""
    if not site_root:
        return None
    from urllib.parse import urlparse
    from app.services.integrations.firecrawl_client import get_firecrawl_client
    host = urlparse(site_root).hostname or ""
    try:
        hits = await get_firecrawl_client().search_web(f"site:{host} {product_name}", user_id=user_id,
                                                       workspace_id=workspace_id, limit=5, source_tag=TASK)
    except Exception as e:
        logger.warning(f"site description search failed for {product_name}: {e}")
        return None
    head = fold(product_name).split()
    for h in hits:
        url, text = h.get("url") or "", (h.get("description") or "").strip()
        slug = [p for p in re.split(r"[-_/]+", fold(urlparse(url).path)) if p]
        if host.removeprefix("www.") in url and text and _word(product_name).search(fold(text)) \
                and any(slug[i:i + len(head)] == head for i in range(len(slug))):
            return {"url": url, "text": text}
    return None


async def classify(product: Dict[str, Any], corpus: str, site_names: Sequence[str], *, model: str,
                   prompt: str, job_id: Optional[str], user_id: Optional[str], workspace_id: Optional[str]) -> Dict[str, Any]:
    from app.services.core.claude_tool_call import call_with_schema
    from app.services.utilities.prompt_registry import render
    content = render(prompt, series=product["name"], site_names=json.dumps(list(site_names), ensure_ascii=False),
                     source=corpus[:CORPUS_LIMIT])
    result = await call_with_schema(
        task=TASK, model=model, max_tokens=1500, job_id=job_id, user_id=user_id, workspace_id=workspace_id,
        product_id=product["id"], messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    )
    return result.data


async def reconcile_document(supabase: Any, document_id: str, *, apply: bool = True, model: Optional[str] = None,
                             job_id: Optional[str] = None, user_id: Optional[str] = None,
                             product_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    from app.services.core.claude_models import SONNET
    from app.services.utilities.prompt_registry import load_prompt
    db = getattr(supabase, "client", supabase)
    doc = db.table("documents").select("id, workspace_id").eq("id", document_id).execute().data[0]
    ws = doc["workspace_id"]
    prompt = await load_prompt("extraction", TASK, stage="quality")
    pages: Dict[int, str] = {}
    start = 0
    while True:
        rows = db.table("document_layout_analysis").select("page_number, layout_elements").eq("document_id", document_id) \
            .order("page_number").range(start, start + 39).execute().data or []
        for r in rows:
            pages[r["page_number"]] = "\n".join(e.get("text_content") or "" for e in (r.get("layout_elements") or []))
        if len(rows) < 40:
            break
        start += 40
    products = db.table("products").select("id, name, description, attributes, metadata, brand_company_id") \
        .eq("source_document_id", document_id).execute().data or []
    brands = sorted({p["brand_company_id"] for p in products if p.get("brand_company_id")})
    brand_names = [r["name"] for r in (db.table("products").select("name").in_("brand_company_id", brands)
                                       .eq("workspace_id", ws).execute().data or [])] if brands else []
    names = sorted({p["name"] for p in products} | set(brand_names))
    summary: Dict[str, Any] = {"products": 0, "changed": 0, "colours_added": 0, "pages_added": 0, "leaks_removed": 0,
                               "descriptions": 0, "site_texts": 0, "failed": [], "per_product": []}
    for p in products:
        if product_ids and p["id"] not in product_ids:
            continue
        summary["products"] += 1
        try:
            own = own_pages(pages, p["name"], names, (p.get("metadata") or {}).get("page_range") or [])
            corpus = "\n\n".join(f"[page {pg}]\n{normalize_text(t)}" for pg, t in sorted(own.items()))
            site_urls = [((r["document_images"].get("metadata") or {}).get("source_url"))
                         for r in db.table("image_product_associations").select("document_images!inner(extraction_layer, metadata)")
                         .eq("product_id", p["id"]).execute().data or []
                         if r["document_images"]["extraction_layer"] == "manufacturer_original"
                         and "excluded_from_products" not in (r["document_images"].get("metadata") or {})]
            site_names = site_variant_names([u for u in site_urls if u], p["name"])
            if not corpus and not site_names:
                summary["per_product"].append({"product": p["name"], "skipped": "no source pages"})
                continue
            raw = await classify(p, corpus, site_names, model=model or SONNET, prompt=prompt, job_id=job_id,
                                 user_id=user_id, workspace_id=ws)
            kept = verify(raw, corpus, site_names)
            m, a = dict(p.get("metadata") or {}), dict(p.get("attributes") or {})
            leaks = leaked_values(a, [n for n in names if n != p["name"]])
            for key, bad in leaks.items():
                vals = a[key] if isinstance(a[key], list) else [a[key]]
                left = [v for v in vals if v not in bad]
                a[key] = left if isinstance(a[key], list) else (left[0] if left else None)
                if a[key] in (None, []):
                    a.pop(key)
            old_colours = a.get("color") if isinstance(a.get("color"), list) else []
            if kept["colours"]:
                a["color"] = m["color"] = kept["colours"]
                m["colour_options"] = [{**next((o for o in (m.get("colour_options") or []) if fold(o.get("name")) == c), {}),
                                       "name": c.title()} for c in kept["colours"]]
            if kept["decors"]:
                m["decorative_variant"] = kept["decors"]
            if kept["finishes"]:
                a["finish"] = m["finish"] = kept["finishes"]
            if kept["formats"]:
                m["available_formats"] = list(dict.fromkeys((m.get("available_formats") or []) + kept["formats"]))
            new_pages = sorted(set(own) - set(m.get("page_range") or []))
            m["page_range"] = sorted(set(m.get("page_range") or []) | set(own))
            desc = p.get("description") or ""
            site_root = None
            if p.get("brand_company_id"):
                brand = db.table("crm_companies").select("website").eq("id", p["brand_company_id"]).execute().data
                site_root = (brand[0].get("website") if brand else None)
                if site_root and not site_root.startswith("http"):
                    site_root = "https://" + site_root.strip("/")
            site = await site_description(p["name"], site_root, user_id, ws) if apply else None
            if site:
                m["site_description"] = {**site, "fetched_at": _now()}
                summary["site_texts"] += 1
            if not desc.strip() or REFUSAL.search(desc.strip()) or m.get("description_source") == "reconcile":
                line = factual_line(p["name"], _material(p), kept["uses"], kept["formats"], kept["colours"], kept["decors"])
                desc = compose_description(line, (site or m.get("site_description") or {}).get("text"))
                m["description_source"] = "reconcile"
                summary["descriptions"] += 1
            m["series_reconcile"] = {"at": _now(), "model": model or SONNET, "kept": kept, "dropped_leaks": leaks,
                                     "pages_added": new_pages, "site_variant_names": site_names,
                                     "previous": {"color": old_colours, "description": p.get("description"),
                                                  "page_range": (p.get("metadata") or {}).get("page_range")}}
            summary["colours_added"] += len(set(kept["colours"]) - set(map(fold, old_colours)))
            summary["pages_added"] += len(new_pages)
            summary["leaks_removed"] += sum(len(v) for v in leaks.values())
            summary["per_product"].append({"product": p["name"], **kept, "pages_added": new_pages, "leaks": leaks,
                                           "description": desc[:200]})
            if apply:
                db.table("products").update({"attributes": a, "metadata": m, "description": desc}).eq("id", p["id"]).execute()
            summary["changed"] += 1
        except Exception as e:
            summary["failed"].append(f"{p['name']}: {type(e).__name__}: {str(e)[:160]}")
    return summary
