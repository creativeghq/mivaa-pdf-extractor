"""A product's variant facts, settled after every step that writes them has run.

Sizes, colours and the description each have several writers and none of them removes anything, so a
misread survives beside its correction. A size is real when the verified format list or a packing
table holds it, or the series' own line prints it; colours are the same in metadata and attributes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)

PACKING_HEADER = re.compile(r"\b(kg|m2|m²|boxes|pieces|pcs)\b", re.I)
FORMAT_HEADERS = {"format", "formato", "size"}
OTHER_UNIT = re.compile(r"\b(mm|in|inch|inches)\b|[\"”]", re.I)
PRINTED_GAP = r"[^\n\d]{0,60}(?:\n[^\n\d]{0,30})?"
REEMBED_BATCH = 100
SIZE_FIELD = re.compile(r"(_mm|_cm|_m|_kg|^size|^width|^height|^depth|^length)$", re.I)
WARRANTY = re.compile(r"\b(\d{1,3})\s*[-‐ ]?\s*(years?|yrs?|jahre?|anni|años|χρόνια|έτη|months?|monate?|mesi|meses|μήνες)\b", re.I)
INDEX_SIZE = re.compile(r"^(?:hexa\s*)?(\d{1,3}(?:[.,]\d)?)\s*x\s*(\d{1,3}(?:[.,]\d)?)\s*cm\b")
INDEX_WORD = re.compile(r"\b(indice|index|inhalt|sommario|sommaire|ευρετηριο)\b")
INDEX_MIN_SERIES = 15
MM_SIZE_HEADER = re.compile(r"\b(size|format|dimensions?)\b.*\bmm\b", re.I)
MM_SIZE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)(?:\s*[x×]\s*(\d+(?:[.,]\d+)?))?\s*(?:mm)?\s*$", re.I)


def _reconcile() -> Any:
    from app.services.discovery import series_reconcile
    return series_reconcile


def is_packing_table(table: Dict[str, Any]) -> bool:
    headers = (table.get("table_data") or {}).get("headers") or []
    return bool(headers) and str(headers[0]).strip().lower() in FORMAT_HEADERS and any(PACKING_HEADER.search(str(h)) for h in headers[1:])


def table_formats(tables: Iterable[Dict[str, Any]]) -> List[str]:
    sr = _reconcile()
    out: List[str] = []
    for table in tables:
        if not is_packing_table(table):
            continue
        for row in (table.get("table_data") or {}).get("rows") or []:
            for size in sorted(sr.sizes_in(str((row or [""])[0]))):
                if size not in out:
                    out.append(size)
    return out


def table_mm_sizes(tables: Iterable[Dict[str, Any]]) -> List[str]:
    """Each size a table states under a column headed in millimetres, as `2780×600×9 mm`."""
    out: List[str] = []
    for table in tables:
        data = table.get("table_data") or {}
        columns = [i for i, h in enumerate(data.get("headers") or []) if MM_SIZE_HEADER.search(str(h))]
        for row in data.get("rows") or []:
            for i in columns:
                match = MM_SIZE.match(str(row[i])) if i < len(row or []) and row[i] is not None else None
                size = "×".join(g for g in match.groups() if g) + " mm" if match else None
                if size and size not in out:
                    out.append(size)
    return out


def _entries(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _cm_keys(value: Any) -> List[str]:
    """The centimetre sizes a size field holds; a size in another unit is not ours to judge."""
    sr = _reconcile()
    out: List[str] = []
    for entry in _entries(value):
        text = entry.get("metric_cm") if isinstance(entry, dict) else entry
        if isinstance(text, str) and not OTHER_UNIT.search(text):
            out.extend(s for s in sorted(sr.sizes_in(text)) if s not in out)
    return out


def printed_sizes(name: str, pages: Iterable[str]) -> Set[str]:
    """The first size after the series name on one line: the next one along belongs to a caption or column."""
    sr = _reconcile()
    pattern = re.compile(r"\b" + re.escape(sr.fold(name)) + r"\b" + PRINTED_GAP + sr.SIZE.pattern)
    out: Set[str] = set()
    for page in pages:
        for match in pattern.finditer(sr.fold(sr.normalize_text(page))):
            out.add(sr.size_key(match.group(1), match.group(2)))
    return out


def verified_formats(metadata: Dict[str, Any], tables: Iterable[Dict[str, Any]], printed: Iterable[str] = ()) -> List[str]:
    held = _cm_keys(metadata.get("available_sizes")) + _cm_keys(metadata.get("dimensions"))
    out = _cm_keys(metadata.get("available_formats"))
    out += [s for s in table_formats(tables) if s not in out]
    return out + [s for s in dict.fromkeys(held) if s in set(printed) and s not in out]


def _settle_sizes(m: Dict[str, Any], verified: Sequence[str]) -> List[str]:
    held = set(_cm_keys(m.get("available_sizes"))) | set(_cm_keys(m.get("dimensions")))
    dropped = sorted(held - set(verified))
    others = [e for e in _entries(m.get("available_sizes")) if not _cm_keys(e)]
    m["available_sizes"] = others + [f"{s} cm" for s in verified]
    m["available_formats"] = list(verified)
    if "dimensions" in m:
        kept: Dict[str, Any] = {}
        for entry in _entries(m["dimensions"]):
            for key in _cm_keys(entry):
                if key in verified and key not in kept:
                    kept[key] = entry if isinstance(entry, dict) else {"metric_cm": key, "format_label": None}
        if kept:
            m["dimensions"] = list(kept.values())
        else:
            m.pop("dimensions")
    return dropped


def _settle_mm_sizes(m: Dict[str, Any], sizes: Sequence[str]) -> List[str]:
    """A bare `2750x119x16` in `dimensions` would read as centimetres, so the unit-bearing list replaces it."""
    dropped = [e for e in _entries(m.get("available_sizes")) if e not in sizes]
    m["available_sizes"] = m["available_formats"] = list(sizes)
    m.pop("dimensions", None)
    return dropped


def _colour_names(m: Dict[str, Any], a: Dict[str, Any]) -> Optional[List[str]]:
    """The names to hold, or None when this product's colours are not ours to rewrite."""
    sr = _reconcile()
    options = m.get("colour_options") or []
    has_hex = any(isinstance(o, dict) and o.get("hex") for o in options)
    kept = ((m.get("series_reconcile") or {}).get("kept") or {}).get("colours")
    if kept:
        return list(kept)
    if kept is not None and not has_hex:
        return []
    raw = _entries(m.get("color")) + _entries(m.get("colors"))
    if not raw:
        raw = _entries(a.get("color"))
    if not all(isinstance(v, str) for v in raw):
        return None
    names: List[str] = []
    seen: Set[str] = set()
    for value in raw:
        clean = " ".join(value.lower().split())
        if clean and sr.fold(clean) not in seen:
            seen.add(sr.fold(clean))
            names.append(clean)
    return names


def _settle_colours(m: Dict[str, Any], a: Dict[str, Any]) -> Dict[str, Any]:
    sr = _reconcile()
    names = _colour_names(m, a)
    if names is None:
        return {}
    before = {"metadata": m.get("color"), "attributes": a.get("color")}
    m.pop("colors", None)
    if names:
        options = {sr.fold(o.get("name")): o for o in (m.get("colour_options") or []) if isinstance(o, dict)}
        m["color"] = a["color"] = names
        m["colour_options"] = [{**options.get(sr.fold(n), {}), "name": options.get(sr.fold(n), {}).get("name") or n.title()} for n in names]
    else:
        for bag in (m, a):
            bag.pop("color", None)
        if not any(isinstance(o, dict) and o.get("hex") for o in (m.get("colour_options") or [])):
            m.pop("colour_options", None)
    after = {"metadata": m.get("color"), "attributes": a.get("color")}
    return {"previous": before} if before != after else {}


def _settle_description(product: Dict[str, Any], m: Dict[str, Any], verified: Sequence[str]) -> Optional[str]:
    """The rebuilt description when the current one is ours, empty, or states a size that is not real."""
    sr = _reconcile()
    current = (product.get("description") or "").strip()
    wrong_size = bool(verified) and bool(sr.sizes_in(current) - set(verified))
    if current and m.get("description_source") != "reconcile" and not wrong_size:
        return None
    kept = (m.get("series_reconcile") or {}).get("kept") or {}
    colours, decors = m.get("color") or [], kept.get("decors") or []
    if not (verified or colours or decors):
        return None
    line = sr.factual_line(product["name"], sr._material({**product, "metadata": m}), kept.get("uses") or [], verified,
                           colours if isinstance(colours, list) else [colours], decors)
    site = (m.get("site_description") or {}).get("text")
    return sr.compose_description(line, site)


def _normalise_values(bag: Dict[str, Any]) -> List[str]:
    """`["electric"]` stored as a string becomes a list; a size stored as {value, unit} becomes its number."""
    changed: List[str] = []
    for key, value in list(bag.items()):
        if key.startswith("_"):
            continue
        if isinstance(value, str) and value.strip().startswith("[") and value.strip().endswith("]"):
            try:
                parsed = json.loads(value)
            except ValueError:
                continue
            if isinstance(parsed, list):
                bag[key] = parsed
                changed.append(key)
        elif SIZE_FIELD.search(key) and isinstance(value, dict) and isinstance(value.get("value"), (int, float)):
            bag[key] = value["value"]
            changed.append(key)
    return changed


def warranty_months(value: Any) -> Optional[int]:
    """"5 years" is 60; a warranty stated any other way is not ours to read."""
    match = WARRANTY.search(str(value or ""))
    if not match:
        return None
    number = int(match.group(1))
    return number * 12 if match.group(2).lower().startswith(("y", "j", "a", "χ", "έ")) else number


def settle_product(product: Dict[str, Any], tables: Sequence[Dict[str, Any]], printed: Iterable[str] = (),
                   index_formats: Iterable[str] = ()) -> Dict[str, Any]:
    """The settled metadata, attributes and description, and what changed. Settling twice changes nothing."""
    m, a = dict(product.get("metadata") or {}), dict(product.get("attributes") or {})
    normalised = _normalise_values(m) + _normalise_values(a)
    columns: Dict[str, Any] = {}
    months = warranty_months(product.get("warranty") or m.get("warranty"))
    if months and not product.get("default_warranty_months"):
        columns["default_warranty_months"] = months
    verified = verified_formats(m, tables, printed)
    if not verified:
        verified = list(dict.fromkeys(index_formats))
    before = (m.get("available_sizes"), m.get("available_formats"), m.get("dimensions"))
    mm_sizes = table_mm_sizes(tables) if not verified else []
    dropped = _settle_sizes(m, verified) if verified else _settle_mm_sizes(m, mm_sizes) if mm_sizes else []
    sizes_changed = before != (m.get("available_sizes"), m.get("available_formats"), m.get("dimensions"))
    colours = _settle_colours(m, a)
    description = _settle_description(product, m, verified)
    changed_description = description is not None and description != (product.get("description") or "")
    changes: Dict[str, Any] = {}
    if normalised:
        changes["normalised"] = normalised
    if columns:
        changes["columns"] = columns
    if sizes_changed:
        changes["sizes"] = {"dropped": dropped, "held": list(verified or mm_sizes)}
    if colours:
        changes["colours"] = colours
    if changed_description:
        changes["description"] = {"previous": product.get("description")}
    if changes:
        m["series_facts"] = {**changes, "at": datetime.now(timezone.utc).isoformat()}
        if changed_description:
            m["description_source"] = "reconcile"
    return {"metadata": m, "attributes": a, "description": description if changed_description else product.get("description"),
            "columns": columns, "changes": changes}


def fact_problems(product: Dict[str, Any], tables: Sequence[Dict[str, Any]]) -> List[str]:
    """What the audit reports: each one is something settling should have left impossible."""
    sr = _reconcile()
    m, a = product.get("metadata") or {}, product.get("attributes") or {}
    out: List[str] = []
    mine, theirs = _entries(m.get("color")), _entries(a.get("color"))
    if all(isinstance(v, str) for v in mine + theirs) and {sr.fold(v) for v in mine} != {sr.fold(v) for v in theirs}:
        out.append("colour_fields_disagree")
    verified = set(verified_formats(m, tables))
    if verified and (set(_cm_keys(m.get("available_sizes"))) | set(_cm_keys(m.get("dimensions")))) - verified:
        out.append("sizes_unverified")
    mm_sizes = table_mm_sizes(tables)
    if mm_sizes and not verified and set(_entries(m.get("available_sizes"))) != set(mm_sizes):
        out.append("sizes_unverified")
    description = (product.get("description") or "").strip()
    if not description:
        out.append("description_missing")
    elif verified and sr.sizes_in(description) - verified:
        out.append("description_size_unverified")
    return out


def table_cell_problems(table: Dict[str, Any]) -> List[str]:
    """A packing row with text in a number column, or whose m2 do not follow from its boxes: two printed rows run together."""
    data = table.get("table_data") or {}
    headers = [str(h).strip().lower() for h in data.get("headers") or []]
    if not is_packing_table(table):
        return []

    def number(cell: Any) -> Optional[float]:
        text = str(cell if cell is not None else "").strip().replace(",", ".")
        return float(text) if re.fullmatch(r"\d+(?:\.\d+)?", text) else None

    index = {h: i for i, h in enumerate(headers)}
    numeric = [i for i, h in enumerate(headers) if i and PACKING_HEADER.search(h) and "(cm)" not in h]
    bad: List[str] = []
    for row in data.get("rows") or []:
        if len(row or []) != len(headers):
            bad.append(" | ".join(str(c) for c in row or []))
            continue
        cells = [str(row[i] if row[i] is not None else "").strip() for i in numeric]
        if any(c not in ("", "-") and number(c) is None for c in cells):
            bad.append(" | ".join(str(c) for c in row))
            continue
        for per_box, boxes, total in (("m2 / box", "boxes / pallet", "m2 / pallet"), ("m2 / box", "europallet boxes", "europallet m2")):
            if per_box in index and boxes in index and total in index:
                a, b, c = number(row[index[per_box]]), number(row[index[boxes]]), number(row[index[total]])
                if a and b and c and abs(a * b - c) > max(0.03 * c, 0.5):
                    bad.append(" | ".join(str(x) for x in row))
                    break
    return bad


def _all(query_for, page: int = 1000) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    while True:
        rows = query_for().range(len(out), len(out) + page - 1).execute().data or []
        out.extend(rows)
        if len(rows) < page:
            return out


async def settle_document(supabase: Any, document_id: str, *, apply: bool = True, job_id: Optional[str] = None,
                          embed: bool = True, trust_print: bool = True) -> Dict[str, Any]:
    """Settle every product of a document and record what changed as a correction."""
    from app.services.pdf.ingestion_corrections import CORRECTED, Correction, record

    db = getattr(supabase, "client", supabase)
    document = (db.table("documents").select("id, workspace_id").eq("id", document_id).limit(1).execute().data or [None])[0]
    if not document:
        return {"error": "document not found"}
    pages: List[str] = []
    if trust_print:
        for row in _all(lambda: db.table("document_layout_analysis").select("page_number, layout_elements")
                        .eq("document_id", document_id).order("page_number")):
            pages.append("\n".join(e.get("text_content") or "" for e in (row.get("layout_elements") or [])))
    products = _all(lambda: db.table("products").select("id, name, description, attributes, metadata, category_id, warranty, default_warranty_months")
                    .eq("source_document_id", document_id).order("id"))
    index = formats_on_index_pages(pages, [p["name"] for p in products])
    summary: Dict[str, Any] = {"products": len(products), "changed": 0, "sizes_dropped": 0, "colours_changed": 0,
                               "descriptions": 0, "failed": [], "per_product": []}
    changed_ids: List[str] = []
    for product in products:
        try:
            tables = db.table("product_tables").select("table_data").eq("product_id", product["id"]).execute().data or []
            result = settle_product(product, tables, printed_sizes(product["name"], pages), index.get(product["name"], ()))
            changes = result["changes"]
            if not changes:
                continue
            summary["changed"] += 1
            summary["sizes_dropped"] += len((changes.get("sizes") or {}).get("dropped") or [])
            summary["colours_changed"] += 1 if changes.get("colours") else 0
            summary["descriptions"] += 1 if changes.get("description") else 0
            summary["per_product"].append({"product": product["name"], **changes})
            changed_ids.append(product["id"])
            if apply:
                db.table("products").update({"metadata": result["metadata"], "attributes": result["attributes"],
                                             "description": result["description"], **result["columns"]}).eq("id", product["id"]).execute()
                product["default_warranty_months"] = result["columns"].get("default_warranty_months") or product.get("default_warranty_months")
        except Exception as e:
            summary["failed"].append(f"{product['name']}: {type(e).__name__}: {str(e)[:160]}")
    warranty = brand_warranty(products)
    lacking = [p["id"] for p in products if not p.get("default_warranty_months")] if warranty else []
    summary["brand_warranty_filled"] = len(lacking)
    if apply and lacking:
        db.table("products").update({"default_warranty_months": warranty}).in_("id", lacking).execute()
    if apply and summary["changed"]:
        await record(db, document_id, document.get("workspace_id"), "series_facts", [Correction(
            "series_facts_settled", CORRECTED,
            f"{summary['changed']} product(s) settled: {summary['sizes_dropped']} unsupported size(s) removed, "
            f"{summary['colours_changed']} colour list(s) aligned, {summary['descriptions']} description(s) rebuilt",
            summary["changed"], [p["product"] for p in summary["per_product"]])], job_id)
    if apply and embed and changed_ids:
        from app.services.embeddings.text_embedding_backfill import backfill_product_text_embeddings
        summary["reembedded"] = 0
        for start in range(0, len(changed_ids), REEMBED_BATCH):
            batch = changed_ids[start:start + REEMBED_BATCH]
            await asyncio.to_thread(lambda: db.table("products").update({"text_embedding_1024": None}).in_("id", batch).execute())
            done = await backfill_product_text_embeddings(max_products=REEMBED_BATCH, product_ids=batch)
            summary["reembedded"] += (done or {}).get("embedded", 0)
    return summary



def packing_rows_by_format(tables: Iterable[Dict[str, Any]]) -> Dict[str, List[List[Any]]]:
    """Each format's distinct packing rows: a maker packs a size the same way whichever series it is."""
    sr = _reconcile()
    out: Dict[str, List[List[Any]]] = {}
    for table in tables:
        if not is_packing_table(table):
            continue
        for row in (table.get("table_data") or {}).get("rows") or []:
            for size in sr.sizes_in(str((row or [""])[0])):
                rows = out.setdefault(size, [])
                if row not in rows:
                    rows.append(row)
    return out


def inherited_packing(product: Dict[str, Any], own_tables: Sequence[Dict[str, Any]], by_format: Dict[str, List[List[Any]]],
                      headers: Sequence[str]) -> Optional[Dict[str, Any]]:
    """The packing table a product without one takes from its own formats, or None when no sibling has them."""
    if any(is_packing_table(t) for t in own_tables):
        return None
    formats = verified_formats(product.get("metadata") or {}, own_tables)
    rows = [row for size in formats for row in by_format.get(size, [])]
    if not rows:
        return None
    return {"rows": rows, "headers": list(headers), "formats": [s for s in formats if s in by_format]}


async def inherit_packing(supabase: Any, document_id: str, *, apply: bool = True, job_id: Optional[str] = None, embed: bool = True) -> Dict[str, Any]:
    """Give a series with no packing table the rows of the same format from the brand's other series."""
    from app.services.chunking import table_chunks as tc
    from app.services.pdf.ingestion_corrections import CORRECTED, Correction, record
    from app.services.pdf.post_ingestion_repair import plan_table_chunks

    db = getattr(supabase, "client", supabase)
    columns = "id, name, workspace_id, brand_company_id, metadata"
    products = _all(lambda: db.table("products").select(columns).eq("source_document_id", document_id).order("id"))
    summary: Dict[str, Any] = {"products": len(products), "inherited": 0, "still_without": [], "per_product": []}
    if not products:
        return summary
    workspace = products[0].get("workspace_id")
    siblings = list(products)
    for brand in {p["brand_company_id"] for p in products if p.get("brand_company_id")}:
        siblings += _all(lambda: db.table("products").select(columns).eq("workspace_id", workspace).eq("brand_company_id", brand).order("id"))
    tables_of: Dict[str, List[Dict[str, Any]]] = {}
    for start in range(0, len({p["id"] for p in siblings}), 100):
        batch = sorted({p["id"] for p in siblings})[start:start + 100]
        for row in db.table("product_tables").select("product_id, table_data").in_("product_id", batch).limit(1000).execute().data or []:
            tables_of.setdefault(row["product_id"], []).append(row)
    every = [t for rows in tables_of.values() for t in rows]
    by_format = packing_rows_by_format(every)
    headers = next(((t["table_data"] or {}).get("headers") for t in every if is_packing_table(t)), None)
    for product in products:
        own = tables_of.get(product["id"], [])
        planned = inherited_packing(product, own, by_format, headers or [])
        if planned is None:
            if not any(is_packing_table(t) for t in own):
                summary["still_without"].append(product["name"])
            continue
        summary["inherited"] += 1
        summary["per_product"].append({"product": product["name"], "formats": planned["formats"]})
        if not apply:
            continue
        db.table("product_tables").insert({
            "product_id": product["id"], "page_number": 1, "headers": planned["headers"], "table_type": "specifications",
            "confidence": 0.8, "extractor": "packing-by-format",
            "table_data": {"rows": planned["rows"], "headers": planned["headers"]},
            "metadata": {"title": "Packing Per Format", "source": "Same format in this brand's other series", "inherited_by_format": True},
        }).execute()
        fresh = db.table("product_tables").select("*").eq("product_id", product["id"]).execute().data or []
        ids = await tc.write_table_chunks(supabase, product["workspace_id"], document_id, product["id"], plan_table_chunks(product["name"], fresh)["chunks"])
        if embed:
            await tc.embed_chunks_once(supabase, ids, workspace_id=product["workspace_id"])
    if apply and summary["inherited"]:
        await record(db, document_id, workspace, "series_facts", [Correction(
            "packing_inherited_by_format", CORRECTED,
            f"{summary['inherited']} product(s) took packing from the same format in this brand's other series",
            summary["inherited"], [p["product"] for p in summary["per_product"]])], job_id)
    return summary



def formats_on_index_pages(pages: Iterable[str], names: Iterable[str]) -> Dict[str, List[str]]:
    """Series under each size heading of a catalogue's index. Keros: 222 agree, 10 extra, so it fills a gap and overrules nothing."""
    sr = _reconcile()
    by_key = {sr.fold(n): n for n in names}
    out: Dict[str, List[str]] = {}
    for page in pages:
        text = sr.fold(sr.normalize_text(page))
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not INDEX_WORD.search(text) or sum(1 for line in lines if line in by_key) < INDEX_MIN_SERIES:
            continue
        current = None
        for line in lines:
            size = INDEX_SIZE.match(line)
            if size:
                current = sr.size_key(*size.groups())
            elif line in by_key and current and current not in out.setdefault(by_key[line], []):
                out[by_key[line]].append(current)
    return out


def brand_warranty(products: Sequence[Dict[str, Any]]) -> Optional[int]:
    """A warranty is the brand's when every product stating one agrees."""
    stated = {p.get("default_warranty_months") for p in products if p.get("default_warranty_months")}
    return stated.pop() if len(stated) == 1 else None
