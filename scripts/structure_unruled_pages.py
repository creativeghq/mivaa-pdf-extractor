"""Structure pages whose tables have no ruling into rows, verify every value against the page, and (optionally) compare with the tables already stored.

Dry run by default: prints the verdict per page and writes nothing. Reads SUPABASE_URL and
SUPABASE_SERVICE_ROLE_KEY from the environment; makes one model call per page (text only, no images).
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def norm(value):
    return " ".join(str(value).lower().split())


def agreement(structured_rows, truth_rows):
    by_code = {r["code"]: {norm(a["value"]) for a in r["attributes"]} for r in structured_rows}
    seen = hit = 0
    for code, cells in truth_rows.items():
        if code not in by_code:
            continue
        for cell in cells:
            if cell in ("", "none") or cell in ("active", "standard range"):
                continue
            seen += 1
            hit += 1 if any(cell == v or cell in v.split() or cell.replace(",", ".") in {x.replace(",", ".") for x in v.split()} for v in by_code[code]) else 0
    return hit, seen


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--document-id", required=True)
    ap.add_argument("--pages", required=True, help="physical page numbers, comma separated")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from app.config import get_settings
    from app.services.core.supabase_client import get_supabase_client, initialize_supabase
    from app.services.pdf.ingestion_audit_runner import source_pages
    from app.services.pdf.text_table_structurer import lines_from_words, structure_page

    settings = get_settings()
    initialize_supabase(settings)
    db = get_supabase_client().client
    doc = db.table("documents").select("workspace_id, storage_bucket, storage_object_path").eq("id", args.document_id).execute().data[0]
    pdf = db.storage.from_(doc["storage_bucket"]).download(doc["storage_object_path"])
    wanted = [int(p) for p in args.pages.split(",")]
    layout = db.table("document_layout_analysis").select("page_number, analysis_metadata, layout_elements").eq("document_id", args.document_id).in_("page_number", wanted).execute().data
    words = source_pages(pdf, layout)
    truth = {}
    if args.compare:
        products = db.table("products").select("id").eq("source_document_id", args.document_id).execute().data
        for product in products:
            for t in db.table("product_tables").select("headers, table_data, extractor").eq("product_id", product["id"]).eq("extractor", "cereuro-prep").execute().data:
                for row in (t["table_data"] or {}).get("rows") or []:
                    if row and isinstance(row[0], str):
                        truth[row[0]] = {norm(c) for c in row[1:] if c is not None}
    total_hit = total_seen = 0
    for row in sorted(layout, key=lambda r: r["page_number"]):
        number = row["page_number"]
        page = words.get(number)
        if not page or not page["words"]:
            print(f"p{number}: no text layer")
            continue
        lines = lines_from_words(page["words"])
        heading = next((e.get("text_content") for e in row["layout_elements"] if e.get("region_type") == "TITLE"), "")
        out = await structure_page(lines, heading=heading or "", workspace_id=doc["workspace_id"], model=args.model)
        verdict = out.get("verdict") or {}
        print(f"p{number}: {out['status']}  rows={len(out['rows'])} inventory={out.get('inventory')} reasons={verdict.get('reasons')} missing={verdict.get('missing_codes', [])[:4]} foreign={verdict.get('foreign_codes', [])[:4]} unsupported={verdict.get('unsupported', [])[:3]}")
        if args.compare and out["rows"]:
            hit, seen = agreement(out["rows"], truth)
            total_hit += hit
            total_seen += seen
            print(f"      agreement with the hand-verified table: {hit}/{seen} cells ({100 * hit / max(seen, 1):.0f}%)")
    if args.compare:
        print(f"TOTAL agreement {total_hit}/{total_seen} ({100 * total_hit / max(total_seen, 1):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
