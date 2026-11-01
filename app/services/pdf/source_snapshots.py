from __future__ import annotations

import asyncio
import hashlib
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urldefrag, urljoin, urlparse

SNAPSHOT_BUCKET = "pdf-documents"
SNAPSHOT_PREFIX = "source-snapshots"
MAX_PAGE_BYTES = 5 * 1024 * 1024
SKIPPED_TAGS = {"script", "style", "noscript", "template", "svg"}


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.links: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: List[Any]) -> None:
        if tag in SKIPPED_TAGS:
            self._skip += 1
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self.links.append(value)
        if tag in {"br", "p", "div", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIPPED_TAGS and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self.parts.append(data)


def parse_page(html: str) -> Dict[str, Any]:
    parser = _Text()
    parser.feed(html)
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(parser.parts))
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return {"text": text, "links": parser.links}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def same_tree_links(base_url: str, links: List[str], root_prefix: str) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    root_host = urlparse(root_prefix).netloc
    for link in links:
        absolute = urldefrag(urljoin(base_url, link))[0]
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https") or parsed.netloc != root_host or parsed.query:
            continue
        if not absolute.startswith(root_prefix) or absolute in seen or absolute == urldefrag(base_url)[0]:
            continue
        if re.search(r"\.(jpg|jpeg|png|gif|webp|svg|pdf|zip|css|js)$", parsed.path, re.I):
            continue
        seen.add(absolute)
        out.append(absolute)
    return out


def storage_path(workspace_id: str, digest: str) -> str:
    return f"{SNAPSHOT_PREFIX}/{workspace_id}/{digest}.html"


async def store_snapshot(
    db: Any,
    *,
    workspace_id: str,
    document_id: Optional[str],
    url: str,
    final_url: str,
    status: int,
    content_type: str,
    body: bytes,
    method: str = "direct",
) -> Dict[str, Any]:
    digest = sha256(body)
    page = parse_page(body.decode("utf-8", errors="replace"))
    path = storage_path(workspace_id, digest)
    try:
        await asyncio.to_thread(db.storage.from_(SNAPSHOT_BUCKET).upload, path, body, {"content-type": content_type or "text/html", "upsert": "false"})
    except Exception as exc:
        if "already exists" not in str(exc).lower() and "duplicate" not in str(exc).lower():
            raise
    row = {
        "workspace_id": workspace_id,
        "document_id": document_id,
        "url": url,
        "final_url": final_url,
        "http_status": status,
        "fetch_method": method,
        "content_type": content_type,
        "content_sha256": digest,
        "text_sha256": sha256(page["text"].encode("utf-8")),
        "visible_text_chars": len(page["text"]),
        "storage_bucket": SNAPSHOT_BUCKET,
        "storage_object_path": path,
    }
    saved = await asyncio.to_thread(db.table("source_snapshots").upsert(row, on_conflict="workspace_id,url,content_sha256,document_id").execute)
    return {"id": (saved.data or [{}])[0].get("id"), "text": page["text"], "links": page["links"], "sha256": digest}
