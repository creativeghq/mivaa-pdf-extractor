"""Store a PDF's text-layer tables on the products named in each page's heading.

Dry run by default. --apply replaces this script's earlier rows (extractor
'pymupdf-text', same page, same product) and touches nothing else. The PDF is
read from the pdf-documents bucket. Reads SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.
"""

import argparse
import importlib.util
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
ZOOM = 250 / 72.0
MIN_ROWS = 3


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for _pkg in ("app", "app.services", "app.services.pdf"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
oc = _load("app.services.pdf.ocr_completeness", "app/services/pdf/ocr_completeness.py")
tl = _load("text_layer_cli", "app/services/pdf/text_layer_extractor.py")


def parse_pages(spec: str) -> List[int]:
    pages: List[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            pages.extend(range(int(a), int(b) + 1))
        elif part.strip():
            pages.append(int(part))
    return pages


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.upper().replace("KORATHERM", "")).strip(" -,")


def heading_names(heading: str) -> List[str]:
    return [norm(p) for p in re.split(r"[,/\n]", heading) if norm(p)]


def products_for_heading(heading: str, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    wanted = set(heading_names(heading))
    return [p for p in products if norm(p["name"]) in wanted]


def plan(pdf_path: str, pages: List[int], products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    import fitz

    doc = fitz.open(pdf_path)
    out: List[Dict[str, Any]] = []
    for page_number in pages:
        content = tl.extract_page_content(doc[page_number - 1])
        elements = tl.elements_from_content(content, page_number, ZOOM)
        headings = [e["text_content"] for e in elements if e["region_type"] == "TITLE"]
        heading = headings[0] if headings else ""
        targets = products_for_heading(heading, products)
        for el in elements:
            if el["region_type"] != "TABLE":
                continue
            rows = oc.table_rows(el["metadata"]["html"])
            if len(rows) < MIN_ROWS:
                continue
            title = el["metadata"].get("title") or heading or "Table"
            out.append({
                "page": page_number,
                "heading": heading,
                "title": title,
                "rows": rows,
                "needs_review": el["metadata"]["needs_review"],
                "products": targets,
            })
    return out


def to_row(item: Dict[str, Any], product_id: str) -> Dict[str, Any]:
    rows = item["rows"]
    headers, data = rows[0], rows[1:]
    metadata: Dict[str, Any] = {
        "title": item["title"],
        "source": f"PDF text layer, page {item['page']}",
        "set_by": "text_layer_backfill",
    }
    if item["needs_review"]:
        metadata["needs_review"] = True
    return {
        "product_id": product_id,
        "page_number": item["page"],
        "table_data": {"headers": headers, "rows": data, "num_rows": len(data), "num_cols": len(headers)},
        "headers": headers,
        "table_type": "specifications",
        "confidence": 0.99,
        "extractor": "pymupdf-text",
        "metadata": metadata,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--document-id", required=True)
    ap.add_argument("--storage-path", required=True)
    ap.add_argument("--pages", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    for name in ("app.services.pdf", "app.services", "app"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(ROOT))
    from app.config import get_settings
    from app.services.core.supabase_client import get_supabase_client, initialize_supabase

    initialize_supabase(get_settings())
    db = get_supabase_client().client
    products = db.table("products").select("id, name").eq("source_document_id", args.document_id).execute().data or []
    blob = db.storage.from_("pdf-documents").download(args.storage_path)
    pdf_path = Path("/tmp") / f"text_layer_{args.document_id}.pdf"
    pdf_path.write_bytes(blob)
    try:
        items = plan(str(pdf_path), parse_pages(args.pages), products)
    finally:
        pdf_path.unlink(missing_ok=True)

    unmatched = 0
    written = 0
    for item in items:
        names = [p["name"] for p in item["products"]]
        print(f"p{item['page']:>3} {len(item['rows']):>3} rows  {item['title'][:60]!r}  heading={item['heading'][:60]!r}  -> {names or 'NO PRODUCT'}")
        if not item["products"]:
            unmatched += 1
    if args.apply:
        pairs = {(p["id"], item["page"]) for item in items for p in item["products"]}
        for product_id, page in sorted(pairs):
            db.table("product_tables").delete().eq("product_id", product_id).eq("extractor", "pymupdf-text").eq("page_number", page).execute()
        for item in items:
            for product in item["products"]:
                db.table("product_tables").insert(to_row(item, product["id"])).execute()
                written += 1
    print(f"tables {len(items)}, without a product {unmatched}, rows written {written}{'' if args.apply else ' (dry run)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
