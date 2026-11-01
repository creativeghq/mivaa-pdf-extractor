"""Keep a private copy of every page under a website path, so values taken from it can be re-checked later.

Crawls breadth-first inside ROOT only, politely (one request at a time with a pause), through the
shared SSRF-safe fetch. Stores each page body under pdf-documents/source-snapshots/<workspace>/<sha>.html
and one source_snapshots row per page; with --document-id the rows are linked to that document.
Reads SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY from the environment. Dry run unless --apply.
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--workspace-id", required=True)
    ap.add_argument("--document-id", action="append", default=[])
    ap.add_argument("--max-pages", type=int, default=250)
    ap.add_argument("--pause", type=float, default=0.6)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from app.config import get_settings
    from app.services.core.supabase_client import get_supabase_client, initialize_supabase
    from app.services.pdf.source_snapshots import MAX_PAGE_BYTES, parse_page, same_tree_links, sha256, store_snapshot
    from app.utils.ssrf_guard import SSRFError, safe_fetch_bytes

    initialize_supabase(get_settings())
    db = get_supabase_client().client
    queue, seen, kept = [args.root], {args.root}, []
    while queue and len(kept) < args.max_pages:
        url = queue.pop(0)
        try:
            fetched = await safe_fetch_bytes(url, max_bytes=MAX_PAGE_BYTES, timeout=30.0, allow_schemes=("https",))
        except SSRFError as exc:
            print(f"SKIP   {url}  ({exc})")
            continue
        except Exception as exc:
            print(f"ERROR  {url}  ({type(exc).__name__}: {exc})")
            continue
        if not fetched.ok or "html" not in (fetched.content_type or "html").lower():
            print(f"SKIP   {url}  (HTTP {fetched.status_code} {fetched.content_type})")
            continue
        page = parse_page(fetched.content.decode("utf-8", errors="replace"))
        print(f"PAGE   {url}  {len(page['text'])} chars  {sha256(fetched.content)[:10]}")
        kept.append(url)
        if args.apply:
            for document_id in args.document_id or [None]:
                await store_snapshot(db, workspace_id=args.workspace_id, document_id=document_id, url=url, final_url=fetched.final_url,
                                     status=fetched.status_code, content_type=fetched.content_type, body=fetched.content)
        for link in same_tree_links(fetched.final_url or url, page["links"], args.root):
            if link not in seen:
                seen.add(link)
                queue.append(link)
        await asyncio.sleep(args.pause)
    print(f"pages {len(kept)}{'' if args.apply else ' (dry run: nothing stored)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
