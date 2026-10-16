"""
KB rechunk service (2026-07-06).

Chunk a kb_doc into sections, embed them together (voyage-context-4, each chunk with
its document in view), and replace the doc's rows in kb_doc_chunks. Idempotent: safe to
re-run — it deletes the doc's existing chunks first, so edits/backfills converge.

Called on-write (after a doc's embedding is (re)generated) and by the backfill endpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.services.kb.kb_chunker import chunk_document, embedding_text

logger = logging.getLogger(__name__)

_INSERT_BATCH = 200       # rows per PostgREST insert


def _to_vec_literal(emb) -> Optional[str]:
    # Insert as a pgvector STRING literal, not a JSON array — the array form is parsed
    # unreliably into a vector by PostgREST (the flakiness we hit on kb_match_docs).
    if not emb:
        return None
    return "[" + ",".join(repr(float(x)) for x in emb) + "]"


async def rechunk_doc(
    supabase,
    doc_id: str,
    *,
    embeddings_service=None,
    target: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Dict[str, Any]:
    """Rechunk + re-embed a single kb_doc. Returns a per-doc summary."""
    row = (
        supabase.client.table("kb_docs")
        .select("id, title, content, workspace_id")
        .eq("id", doc_id)
        .single()
        .execute()
    )
    doc = row.data
    if not doc:
        return {"doc_id": doc_id, "error": "not_found", "chunks": 0}

    content = doc.get("content") or ""
    workspace_id = doc.get("workspace_id")
    title = doc.get("title")

    kwargs = {}
    if target is not None:
        kwargs["target"] = target
    if overlap is not None:
        kwargs["overlap"] = overlap
    chunks = chunk_document(content, title=title, **kwargs)  # coverage-invariant asserted inside

    if not chunks:
        supabase.client.table("kb_doc_chunks").delete().eq("kb_doc_id", doc_id).execute()
        return {"doc_id": doc_id, "chunks": 0, "embedded": 0, "failed": 0, "content_len": len(content)}

    if embeddings_service is None:
        from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
        embeddings_service = RealEmbeddingsService()
    model = embeddings_service.voyage_chunk_model

    texts = [embedding_text(title, ch) for ch in chunks]
    embeddings = await embeddings_service.generate_contextualized_embeddings(
        texts, workspace_id=workspace_id,
    )
    failed = sum(1 for e in embeddings if not e)
    if failed:
        # Embed before replacing: the doc's existing rows stay searchable until a full set exists.
        logger.warning("rechunk_doc %s: %d/%d chunks failed to embed; kept existing chunks",
                       doc_id, failed, len(chunks))
        return {"doc_id": doc_id, "error": "embedding_failed", "chunks": 0,
                "embedded": 0, "failed": failed, "content_len": len(content)}

    supabase.client.table("kb_doc_chunks").delete().eq("kb_doc_id", doc_id).execute()

    rows = []
    for ch, emb in zip(chunks, embeddings):
        vec = _to_vec_literal(emb)
        rows.append({
            "kb_doc_id": doc_id,
            "workspace_id": workspace_id,
            "chunk_index": ch.chunk_index,
            "heading": ch.heading or None,
            "content": ch.content,
            "char_start": ch.char_start,
            "char_end": ch.char_end,
            "token_count": len(ch.content) // 4,  # rough
            "text_embedding": vec,
            "embedding_model": model if vec else None,
            "schema_version": 1,
        })

    for i in range(0, len(rows), _INSERT_BATCH):
        supabase.client.table("kb_doc_chunks").insert(rows[i : i + _INSERT_BATCH]).execute()

    return {
        "doc_id": doc_id,
        "chunks": len(chunks),
        "embedded": len(chunks) - failed,
        "failed": failed,
        "content_len": len(content),
        "model": model,
    }
