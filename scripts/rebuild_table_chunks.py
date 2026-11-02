"""Rebuild plain-text table chunks for every product of one document.

Dry run by default. --apply replaces this document's earlier table_block chunks per product and
embeds only the new ones. The pipeline runs the same step after every ingestion
(app/services/pdf/post_ingestion_repair.py). Reads SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


async def run(document_id: str, apply: bool) -> int:
    sys.path.insert(0, str(ROOT))
    from app.config import get_settings
    from app.services.core.supabase_client import get_supabase_client, initialize_supabase
    from app.services.pdf.post_ingestion_repair import rebuild_table_chunks

    initialize_supabase(get_settings())
    summary = await rebuild_table_chunks(get_supabase_client(), document_id, apply=apply)
    if not summary["products"]:
        print(f"no products with source_document_id={document_id}")
        return 1
    for p in summary["per_product"]:
        print(f"{p['product']}: {p['tables']} tables in db, {p['used']} used, {p['chunks']} chunks, {p['codes']} item codes")
    print(f"TOTAL: {summary['products']} products, {summary['chunks']} chunks; wrote {summary['written']}, embedded {summary['embedded']}")
    if not apply:
        print("DRY RUN - nothing written. Re-run with --apply.")
    for failure in summary["failed"]:
        print(f"EMBED FAILED {failure['id']}: {failure['reason']}")
    return 2 if summary["failed"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    return asyncio.run(run(args.document_id, args.apply))


if __name__ == "__main__":
    sys.exit(main())
