"""Plain-text table chunks for product_tables rows: build, write idempotently, embed once."""

import asyncio
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

PRICE_COLUMN = re.compile(
    r"\b(price|prices|cost|costs|margin|markup|discount|rrp|eur|euro|vat|pvp|msrp|retail|listino|prezzo|preis|prix|precio|tarif|τιμή|τιμές|αξία|κόστος|έκπτωση|φπα)\b|[€$£]", re.I
)
CURRENCY_CELL = re.compile(r"[€$£]|\b(eur|euro|usd|gbp)\b", re.I)
CURRENCY_COLUMN_SHARE = 0.5
ITEM_CODE = re.compile(
    r"\b(?=[A-Za-z0-9._/-]*\d)(?=[A-Za-z0-9._/-]*[A-Za-z]|\d{5,})[A-Za-z0-9][A-Za-z0-9._/-]{3,}\b"
)
SOURCE = "table_block"
INDEX_BASE = 100000
EMBED_BATCH = 50
INSERT_BATCH = 100


def is_price_header(header: Any) -> bool:
    return bool(PRICE_COLUMN.search(str(header or "")))


def item_codes(text: str) -> set:
    return set(ITEM_CODE.findall(text or ""))


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def resolve_header(headers: List[Any], rows: List[Any]) -> tuple:
    if sum(1 for h in headers if _cell(h)) >= 2:
        return headers, list(rows)
    for pos, raw in enumerate(rows):
        if sum(1 for c in (raw or []) if _cell(c)) >= 2:
            return list(raw), list(rows[pos + 1:])
    return headers, list(rows)


def is_currency_column(rows: List[Any], index: int) -> bool:
    cells = [_cell(r[index]) for r in rows if r and index < len(r) and _cell(r[index])]
    return bool(cells) and sum(1 for c in cells if CURRENCY_CELL.search(c)) / len(cells) >= CURRENCY_COLUMN_SHARE


def build_table_chunks(
    product_name: str,
    tables: List[Dict[str, Any]],
    rows_per_chunk: int = 25,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    per_page: Dict[int, int] = {}
    for table in tables or []:
        meta = table.get("metadata") or {}
        title = _cell(meta.get("title"))
        if table.get("extractor") == "paddleocr-vl" and not title:
            continue
        title = title or "Table"
        headers, body = resolve_header(list(table.get("headers") or []), (table.get("table_data") or {}).get("rows") or [])
        keep = [i for i, h in enumerate(headers) if not is_price_header(h) and not is_currency_column(body, i)]
        header_cells = [_cell(headers[i]) for i in keep]
        if not any(header_cells):
            continue
        rows: List[str] = []
        for raw in body:
            raw = list(raw or [])
            cells = [_cell(raw[i]) if i < len(raw) else "" for i in keep]
            if any(cells):
                rows.append(" | ".join(cells))
        if not rows:
            continue

        page = int(table.get("page_number") or 0)
        table_index = per_page.get(page, 0)
        per_page[page] = table_index + 1
        groups = [rows[i:i + rows_per_chunk] for i in range(0, len(rows), rows_per_chunk)]
        head = f"{product_name} - {title} (page {page})\n" + " | ".join(header_cells)
        for part, group in enumerate(groups, start=1):
            chunks.append({
                "content": head + "\n" + "\n".join(group),
                "metadata": {
                    "source": SOURCE,
                    "table_id": table.get("id"),
                    "page_number": page,
                    "part": part,
                    "parts": len(groups),
                    "extractor": table.get("extractor"),
                    "table_title": title,
                    "needs_review": bool(meta.get("needs_review", False)),
                    "is_table": True,
                },
                "part": part,
                "parts": len(groups),
                "table_index": table_index,
            })
    return chunks


def assign_chunk_indexes(chunks: List[Dict[str, Any]]) -> List[int]:
    """100000 + page*100 + table_index*10 + part, bumped past any collision."""
    taken: set = set()
    out: List[int] = []
    for chunk in chunks:
        page = int(chunk["metadata"].get("page_number") or 0)
        index = INDEX_BASE + page * 100 + int(chunk.get("table_index", 0)) * 10 + int(chunk.get("part", 1))
        while index in taken:
            index += 1
        taken.add(index)
        out.append(index)
    return out


def _db(supabase: Any) -> Any:
    return getattr(supabase, "client", supabase)


async def write_table_chunks(
    supabase: Any,
    workspace_id: str,
    document_id: str,
    product_id: str,
    chunks: List[Dict[str, Any]],
) -> List[str]:
    db = _db(supabase)
    previous = await asyncio.to_thread(
        db.table("document_chunks").select("id")
        .eq("document_id", document_id)
        .eq("product_id", product_id)
        .eq("metadata->>source", SOURCE)
        .execute
    )
    old_ids = [r["id"] for r in (previous.data or [])]
    rows = [
        {
            "document_id": document_id,
            "workspace_id": workspace_id,
            "content": c["content"],
            "chunk_index": idx,
            "metadata": c["metadata"],
            "product_id": product_id,
            "chunk_type": "technical_specs",
            "chunk_type_status": "classified",
            "has_text_embedding": False,
        }
        for c, idx in zip(chunks, assign_chunk_indexes(chunks))
    ]
    ids: List[str] = []
    for start in range(0, len(rows), INSERT_BATCH):
        result = await asyncio.to_thread(
            db.table("document_chunks").insert(rows[start:start + INSERT_BATCH]).execute
        )
        ids.extend(r["id"] for r in (result.data or []))
    for start in range(0, len(old_ids), INSERT_BATCH):
        await asyncio.to_thread(
            db.table("document_chunks").delete().in_("id", old_ids[start:start + INSERT_BATCH]).execute
        )
    return ids


async def embed_chunks_once(
    supabase: Any,
    chunk_ids: List[str],
    embedder: Optional[Any] = None,
    workspace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Embed exactly `chunk_ids`. Returns {embedded, failed: [{id, reason}]}."""
    ids = list(chunk_ids or [])
    if not ids:
        return {"embedded": 0, "failed": []}
    db = _db(supabase)
    if embedder is None:
        from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
        embedder = RealEmbeddingsService()
    model = getattr(embedder, "voyage_model", None)
    if not model:
        from app.config import get_settings
        model = get_settings().voyage_document_model

    embedded = 0
    failed: List[Dict[str, str]] = []
    for start in range(0, len(ids), EMBED_BATCH):
        batch_ids = ids[start:start + EMBED_BATCH]
        fetched = await asyncio.to_thread(
            db.table("document_chunks").select("id, content").in_("id", batch_ids).execute
        )
        content_by_id = {r["id"]: r.get("content") or "" for r in (fetched.data or [])}
        batch = [i for i in batch_ids if content_by_id.get(i, "").strip()]
        failed.extend({"id": i, "reason": "missing or empty content"} for i in batch_ids if i not in batch)
        if not batch:
            continue
        try:
            vectors = await embedder.generate_batch_embeddings(
                texts=[content_by_id[i] for i in batch],
                dimensions=1024,
                input_type="document",
                workspace_id=workspace_id,
            )
        except Exception as exc:
            failed.extend({"id": i, "reason": f"batch embedding raised: {exc}"} for i in batch)
            continue
        vectors = list(vectors or [])
        now = datetime.utcnow().isoformat()
        for pos, chunk_id in enumerate(batch):
            vector = vectors[pos] if pos < len(vectors) else None
            if not vector:
                failed.append({"id": chunk_id, "reason": "no vector returned"})
                continue
            try:
                await asyncio.to_thread(
                    db.table("document_chunks").update({
                        "text_embedding": vector,
                        "has_text_embedding": True,
                        "embedding_model": model,
                        "embedding_dimension": len(vector),
                        "embedding_generated_at": now,
                    }).eq("id", chunk_id).execute
                )
                embedded += 1
            except Exception as exc:
                failed.append({"id": chunk_id, "reason": f"update failed: {exc}"})
    return {"embedded": embedded, "failed": failed}
