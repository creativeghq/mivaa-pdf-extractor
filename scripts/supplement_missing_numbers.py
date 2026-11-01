"""Store the printed text lines that carry numbers the layout pass dropped, as embedded chunks.

Takes the pages the audit says are missing numbers, reads those lines from the PDF text layer and writes
one chunk per page (owned by the single product whose page range holds the page, else catalogue-level).
Dry run unless --apply; reruns replace this script's earlier chunks for the document. Reads SUPABASE_URL
and SUPABASE_SERVICE_ROLE_KEY from the environment.
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
    from app.services.chunking.table_chunks import embed_chunks_once
    from app.services.core.supabase_client import get_supabase_client, initialize_supabase
    from app.services.pdf.ingestion_audit import numeric_tokens
    from app.services.pdf.ingestion_audit_runner import _collect, _download, build_audit, source_pages
    from app.services.pdf.page_supplements import SOURCE, build_supplement, owner_for, supplement_text
    from app.services.pdf.text_table_structurer import lines_from_words

    initialize_supabase(get_settings())
    db = get_supabase_client().client
    document = db.table("documents").select("id, workspace_id, filename, storage_bucket, storage_object_path, content_type").eq("id", args.document_id).execute().data[0]
    collected = await _collect(db, document)
    pdf = await _download(db, document)
    if pdf is None:
        print("source PDF is not in storage; nothing to read")
        return 1
    result = build_audit(document, collected, pdf, "post_embed", None).as_dict()
    gaps = [g for f in result["findings"] if f["code"] in ("numbers_missing", "numbers_missing_few") for g in f["examples"]]
    pages = source_pages(pdf, collected["layout"])
    label = document["filename"].rsplit(".", 1)[0]
    planned = []
    for gap in sorted(gaps, key=lambda g: g["page"]):
        page = pages.get(gap["page"])
        if not page:
            continue
        lines = lines_from_words(page["words"])
        text = supplement_text(lines, gap["all"], numeric_tokens)
        chunk = build_supplement(label, gap["page"], text, gap["all"])
        if chunk:
            planned.append((gap["page"], owner_for(gap["page"], collected["products"]), chunk))
            print(f"p{gap['page']}: {gap['missing']} missing {gap['all'][:8]} -> {len(text)} line(s), owner={planned[-1][1]}")
            for t in text[:6]:
                print("      " + t[:140])
    if not args.apply:
        print(f"DRY RUN - {len(planned)} chunk(s) planned, nothing written.")
        return 0
    old = [r["id"] for r in db.table("document_chunks").select("id").eq("document_id", args.document_id).eq("metadata->>source", SOURCE).execute().data]
    rows = [{
        "document_id": args.document_id, "workspace_id": document["workspace_id"], "content": c["content"], "chunk_index": 200000 + pg,
        "metadata": c["metadata"], "product_id": owner, "chunk_type": "technical_specs", "chunk_type_status": "classified", "has_text_embedding": False,
    } for pg, owner, c in planned]
    ids = [r["id"] for r in (db.table("document_chunks").insert(rows).execute().data or [])] if rows else []
    if old:
        db.table("document_chunks").delete().in_("id", old).execute()
    out = await embed_chunks_once(db, ids) if ids else {"embedded": 0, "failed": []}
    print(f"wrote {len(ids)}, replaced {len(old)}, embedded {out['embedded']}, failed {len(out['failed'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
