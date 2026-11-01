"""Rebuild plain-text table chunks for every product of one document.

Dry run by default. --apply deletes this document's earlier table_block chunks per
product, inserts the new ones and embeds only those ids. Reads SUPABASE_URL and
SUPABASE_SERVICE_ROLE_KEY from the environment.
"""

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]


def _load_table_chunks():
    path = ROOT / "app" / "services" / "chunking" / "table_chunks.py"
    spec = importlib.util.spec_from_file_location("table_chunks_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tc = _load_table_chunks()


def refuse_if_price_columns(chunks: List[Dict[str, Any]]) -> None:
    for chunk in chunks:
        header_line = chunk["content"].split("\n")[1]
        for cell in header_line.split(" | "):
            if tc.is_price_header(cell):
                raise RuntimeError(f"price-like column {cell!r} survived into a chunk; refusing")


def plan_product(product_name: str, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunks = tc.build_table_chunks(product_name, tables)
    refuse_if_price_columns(chunks)
    used = {c["metadata"]["table_id"] for c in chunks}
    codes: set = set()
    for chunk in chunks:
        codes |= tc.item_codes(chunk["content"])
    return {"chunks": chunks, "tables_used": len(used), "codes": codes}


def _client():
    sys.path.insert(0, str(ROOT))
    from app.config import get_settings
    from app.services.core.supabase_client import get_supabase_client, initialize_supabase

    initialize_supabase(get_settings())
    return get_supabase_client()


async def run(document_id: str, apply: bool) -> int:
    supabase = _client()
    db = supabase.client
    products = (
        db.table("products").select("id, name, workspace_id")
        .eq("source_document_id", document_id).execute().data or []
    )
    if not products:
        print(f"no products with source_document_id={document_id}")
        return 1
    total_chunks = 0
    all_codes: set = set()
    failures: List[Dict[str, str]] = []
    for product in products:
        tables = (
            db.table("product_tables").select("*").eq("product_id", product["id"]).execute().data or []
        )
        plan = plan_product(product["name"], tables)
        total_chunks += len(plan["chunks"])
        all_codes |= plan["codes"]
        print(
            f"{product['name']} [{product['id']}]: {len(tables)} tables in db, "
            f"{plan['tables_used']} used, {len(plan['chunks'])} chunks, {len(plan['codes'])} item codes"
        )
        if apply and plan["chunks"]:
            ids = await tc.write_table_chunks(
                supabase, product["workspace_id"], document_id, product["id"], plan["chunks"]
            )
            result = await tc.embed_chunks_once(supabase, ids, workspace_id=product["workspace_id"])
            failures.extend(result["failed"])
            print(f"  wrote {len(ids)}, embedded {result['embedded']}, failed {len(result['failed'])}")
    print(f"TOTAL: {len(products)} products, {total_chunks} chunks, {len(all_codes)} distinct item codes")
    if not apply:
        print("DRY RUN - nothing written. Re-run with --apply.")
    for failure in failures:
        print(f"EMBED FAILED {failure['id']}: {failure['reason']}")
    return 2 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    return asyncio.run(run(args.document_id, args.apply))


if __name__ == "__main__":
    sys.exit(main())
