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
