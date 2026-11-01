"""Audit one document (or every completed one): what the source says versus what the platform stored.

Read-only on business data; writes only an ingestion_audits row (use --no-store to skip). Reads
SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY from the environment.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--document-id")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--phase", default="post_embed", choices=["pre_embed", "post_embed"])
    ap.add_argument("--no-store", action="store_true")
    ap.add_argument("--no-notify", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if not args.document_id and not args.all:
        ap.error("pass --document-id or --all")

    sys.path.insert(0, str(ROOT))
    from app.config import get_settings
    from app.services.core.supabase_client import get_supabase_client, initialize_supabase
    from app.services.pdf.ingestion_audit_runner import run_ingestion_audit

    settings = get_settings()
    initialize_supabase(settings)
    supabase = get_supabase_client()
    ids = [args.document_id] if args.document_id else [
        r["id"] for r in supabase.client.table("documents").select("id").eq("processing_status", "completed").execute().data or []
    ]
    worst = 0
    for document_id in ids:
        result = await run_ingestion_audit(
            document_id, supabase, settings, phase=args.phase, trigger="manual" if args.document_id else "backfill",
            persist=not args.no_store, notify=not args.no_notify,
        )
        worst = max(worst, 0 if result["status"] == "passed" else 1)
        if args.json:
            print(json.dumps({"document_id": document_id, **result}, default=str))
            continue
        print(f"{document_id}  {result['status']}  notified={result['notified']}")
        for key in ("number_coverage", "code_coverage", "pages_with_missing_numbers", "tables_needing_review", "untitled_ocr_fragments", "tables_without_chunk", "unembedded_chunks"):
            if key in result["metrics"]:
                print(f"    {key}: {result['metrics'][key]}")
        for f in result["findings"]:
            print(f"    [{f['severity']}] {f['code']}: {f['message']}  {str(f['examples'][:6])[:160]}")
    return worst


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
