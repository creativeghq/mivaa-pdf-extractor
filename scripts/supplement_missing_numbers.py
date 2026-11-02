"""Store the printed text lines that carry numbers the layout pass dropped, as embedded chunks.

Takes the pages the audit says are missing numbers and writes one chunk per page (owned by the single
product whose page range holds the page, else catalogue-level). Dry run unless --apply; reruns replace
the earlier ones. The pipeline runs the same step after every ingestion
(app/services/pdf/post_ingestion_repair.py). Reads SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--document-id", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from app.config import get_settings
    from app.services.core.supabase_client import get_supabase_client, initialize_supabase
    from app.services.pdf.ingestion_audit_runner import _collect, _download, build_audit
    from app.services.pdf.post_ingestion_repair import write_page_supplements

    initialize_supabase(get_settings())
    supabase = get_supabase_client()
    db = supabase.client
    document = db.table("documents").select("id, workspace_id, filename, storage_bucket, storage_object_path, content_type").eq("id", args.document_id).execute().data[0]
    collected = await _collect(db, document)
    pdf = await _download(db, document)
    if pdf is None:
        print("source PDF is not in storage; nothing to read")
        return 1
    findings = build_audit(document, collected, pdf, "post_embed", None).as_dict()["findings"]
    summary = await write_page_supplements(supabase, document, collected, pdf, findings, apply=args.apply)
    if not args.apply:
        for p in summary["planned"]:
            print(f"p{p['page']}: {p['missing']} missing -> {len(p['lines'])} line(s), owner={p['owner']}")
            for t in p["lines"][:6]:
                print("      " + t[:140])
        print(f"DRY RUN - {summary['pages']} chunk(s) planned, nothing written.")
        return 0
    print(f"wrote {summary['written']}, replaced {summary['replaced']}, embedded {summary['embedded']}, failed {len(summary['failed'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
