from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.services.pdf.ocr_completeness import COMPLETE, table_rows

SCOPE = "catalogue_wide"
MIN_ROWS = 3
ID_BATCH = 100


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").upper().replace("KORATHERM", "")).strip(" -,")


def heading_names(heading: str) -> List[str]:
    return [n for n in (norm(part) for part in re.split(r"[,/\n]", heading or "")) if n]


def products_for_heading(heading: str, products: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    wanted = set(heading_names(heading))
    return [p for p in products if norm(p["name"]) in wanted]


def owned_pages(products: Sequence[Dict[str, Any]]) -> Set[int]:
    pages: Set[int] = set()
    for p in products:
        for n in ((p.get("metadata") or {}).get("page_range") or []):
            if isinstance(n, int):
                pages.add(n)
    return pages


def _usable(element: Dict[str, Any]) -> bool:
    meta = element.get("metadata") or {}
    if element.get("region_type") != "TABLE":
        return False
    if meta.get("source") == "text-layer":
        return True
    return meta.get("completeness") == COMPLETE


def plan_catalogue_tables(
    layout_rows: Sequence[Dict[str, Any]], products: Sequence[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    owned = owned_pages(products)
    plans: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    for row in layout_rows:
        page = int(row["page_number"])
        if page in owned:
            continue
        elements = row.get("layout_elements") or []
        tables = [e for e in elements if _usable(e)]
        if not tables:
            continue
        heading = next((e.get("text_content") or "" for e in elements if e.get("region_type") == "TITLE"), "")
        targets = products_for_heading(heading, products)
        for el in tables:
            rows = table_rows((el.get("metadata") or {}).get("html") or el.get("text_content") or "")
            if len(rows) < MIN_ROWS:
                continue
            meta = el.get("metadata") or {}
            if not targets:
                unmatched.append({"page": page, "heading": heading[:120], "rows": len(rows)})
                continue
            plans.append({
                "page": page,
                "title": meta.get("title") or heading or "Table",
                "rows": rows,
                "extractor": "pymupdf-text" if meta.get("source") == "text-layer" else "paddleocr-vl",
                "needs_review": bool(meta.get("needs_review")),
                "products": [p["id"] for p in targets],
            })
    return plans, unmatched


def to_row(plan: Dict[str, Any], product_id: str) -> Dict[str, Any]:
    headers, data = plan["rows"][0], plan["rows"][1:]
    metadata: Dict[str, Any] = {"title": plan["title"], "source": f"Catalogue-wide page {plan['page']}", "scope": SCOPE}
    if plan["needs_review"]:
        metadata["needs_review"] = True
    return {
        "product_id": product_id,
        "page_number": plan["page"],
        "table_data": {"headers": headers, "rows": data, "num_rows": len(data), "num_cols": len(headers)},
        "headers": headers,
        "table_type": "specifications",
        "confidence": 0.99,
        "extractor": plan["extractor"],
        "metadata": metadata,
    }


def _rows(response: Any) -> List[Dict[str, Any]]:
    return list(getattr(response, "data", None) or [])


async def persist_catalogue_wide_tables(document_id: str, supabase: Any, *, apply: bool = True) -> Dict[str, Any]:
    db = supabase.client
    products = _rows(await asyncio.to_thread(db.table("products").select("id, name, metadata").eq("source_document_id", document_id).execute))
    layout = _rows(await asyncio.to_thread(
        db.table("document_layout_analysis").select("page_number, layout_elements").eq("document_id", document_id).order("page_number").execute
    ))
    plans, unmatched = plan_catalogue_tables(layout, products)
    summary: Dict[str, Any] = {"tables": len(plans), "unmatched": unmatched, "rows_written": 0, "applied": apply}
    if not apply or not plans:
        return summary
    pairs = {(pid, plan["page"]) for plan in plans for pid in plan["products"]}
    product_ids = sorted({pid for pid, _ in pairs})
    stale: List[str] = []
    for start in range(0, len(product_ids), ID_BATCH):
        existing = _rows(await asyncio.to_thread(
            db.table("product_tables").select("id, product_id, page_number, metadata").in_("product_id", product_ids[start:start + ID_BATCH]).execute
        ))
        stale += [r["id"] for r in existing if (r.get("metadata") or {}).get("scope") == SCOPE and (r["product_id"], r["page_number"]) in pairs]
    written: List[str] = []
    for plan in plans:
        for pid in plan["products"]:
            saved = _rows(await asyncio.to_thread(db.table("product_tables").insert(to_row(plan, pid)).execute))
            written += [r["id"] for r in saved]
    for start in range(0, len(stale), ID_BATCH):
        await asyncio.to_thread(db.table("product_tables").delete().in_("id", stale[start:start + ID_BATCH]).execute)
    summary["rows_written"] = len(written)
    if unmatched:
        logging.getLogger(__name__).warning("catalogue-wide tables without a product on pages %s", sorted({u["page"] for u in unmatched}))
    return summary


def describe(summary: Optional[Dict[str, Any]]) -> str:
    if not summary:
        return "none"
    return f"{summary.get('tables', 0)} table(s), {summary.get('rows_written', 0)} written, {len(summary.get('unmatched', []))} without a product"



CODE_TOKEN = re.compile(r"\b[A-Z][A-Z0-9]{1,7}\b")
CODE_SHARED_BY = 3


def model_codes(product: Dict[str, Any]) -> Set[str]:
    """The codes a product is known by: printed in its name's brackets or listed on it."""
    meta = product.get("metadata") or {}
    listed = meta.get("product_codes") if isinstance(meta.get("product_codes"), list) else []
    out = {str(c).strip().upper() for c in listed if str(c).strip()}
    for bracket in re.findall(r"\(([^)]*)\)", product.get("name") or ""):
        out |= {t for t in CODE_TOKEN.findall(bracket.upper()) if len(t) >= 3}
    return out


def code_index(products: Sequence[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Code to products. A code on more than a few products names a format or a range, not a model."""
    index: Dict[str, Set[str]] = {}
    for product in products:
        for code in model_codes(product):
            index.setdefault(code, set()).add(product["id"])
    return {code: ids for code, ids in index.items() if len(ids) <= CODE_SHARED_BY}


def table_owners(table: Dict[str, Any], index: Dict[str, Set[str]]) -> Set[str]:
    """Products whose codes the table's model column names. A multi-model table belongs to each of them."""
    owners: Set[str] = set()
    for row in (table.get("table_data") or {}).get("rows") or []:
        for cell in (row or [])[:2]:
            for token in CODE_TOKEN.findall(str(cell or "").upper()):
                owners |= index.get(token, set())
    return owners


def chunk_owner(text: str, index: Dict[str, Set[str]]) -> Optional[str]:
    """The one product a text's model codes name. Text naming several is about the range and stays where it is."""
    owners: Set[str] = set()
    for token in CODE_TOKEN.findall(text or ""):
        owners |= index.get(token, set())
    return next(iter(owners)) if len(owners) == 1 else None


async def route_tables_by_code(supabase: Any, document_id: str, *, apply: bool = True, job_id: Optional[str] = None) -> Dict[str, Any]:
    """A table whose model column names other products goes to them: a merged row must not own the range's tables."""
    from app.services.pdf.ingestion_corrections import CORRECTED, Correction, record

    db = getattr(supabase, "client", supabase)
    products = await asyncio.to_thread(lambda: _rows(db.table("products").select("id, name, metadata, workspace_id").eq("source_document_id", document_id).execute()))
    summary: Dict[str, Any] = {"tables": 0, "moved": 0, "copied": 0}
    index = code_index(products)
    if not index:
        return summary
    ids = [p["id"] for p in products]
    for start in range(0, len(ids), ID_BATCH):
        for table in await asyncio.to_thread(lambda: _rows(db.table("product_tables").select("*").in_("product_id", ids[start:start + ID_BATCH]).execute())):
            summary["tables"] += 1
            owners = table_owners(table, index)
            if not owners or owners == {table["product_id"]}:
                continue
            others = sorted(owners - {table["product_id"]})
            if table["product_id"] not in owners:
                summary["moved"] += 1
                if apply:
                    db.table("product_tables").update({"product_id": others[0]}).eq("id", table["id"]).execute()
                others = others[1:]
            summary["copied"] += len(others)
            if apply and others:
                copy = {k: v for k, v in table.items() if k not in ("id", "created_at", "updated_at", "product_id")}
                db.table("product_tables").insert([{**copy, "product_id": other} for other in others]).execute()
    summary["chunks_moved"] = 0
    start = 0
    while True:
        chunks = await asyncio.to_thread(lambda: _rows(db.table("document_chunks").select("id, product_id, content").eq("document_id", document_id).order("id").range(start, start + 999).execute()))
        for chunk in chunks:
            owner = chunk_owner(chunk.get("content") or "", index)
            if owner and owner != chunk.get("product_id"):
                summary["chunks_moved"] += 1
                if apply:
                    db.table("document_chunks").update({"product_id": owner}).eq("id", chunk["id"]).execute()
        if len(chunks) < 1000:
            break
        start += 1000
    if apply and (summary["moved"] or summary["copied"] or summary["chunks_moved"]):
        await record(db, document_id, products[0].get("workspace_id"), "tables", [Correction(
            "tables_routed_by_model_code", CORRECTED,
            f"{summary['moved']} table(s) moved, {summary['copied']} copied and {summary['chunks_moved']} chunk(s) moved to the products their model codes name",
            summary["moved"] + summary["copied"] + summary["chunks_moved"])], job_id)
    return summary
