"""
KB rechunk service.

Chunk a kb_doc into sections, embed them together (voyage-context-4, each chunk with
its document in view), then swap the doc's kb_doc_chunks in ONE transaction
(`replace_kb_doc_chunks`), which also stamps the source hash. A doc whose hash no longer
matches its text is returned by `kb_docs_needing_rechunk`, so a failed or missed run is
retried by the drain rather than leaving stale sections searchable indefinitely.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.services.kb.chunk_source_hash import source_hash
from app.services.kb.kb_chunker import chunk_document, embedding_text

logger = logging.getLogger(__name__)


def _to_vec_literal(emb) -> Optional[str]:
    # A pgvector STRING literal, not a JSON array — the array form is parsed unreliably.
    # Five significant digits: the column is halfvec (fp16), so more is payload, not precision.
    if not emb:
        return None
    return "[" + ",".join(f"{float(x):.5g}" for x in emb) + "]"


def _record_attempt(supabase, doc_id: str, error: Optional[str]) -> None:
    supabase.client.table("kb_docs").update({
        "chunks_attempted_at": datetime.now(timezone.utc).isoformat(),
        "chunks_error": error,
    }).eq("id", doc_id).execute()


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
    digest = source_hash(title, content)
    _record_attempt(supabase, doc_id, None)

    kwargs = {}
    if target is not None:
        kwargs["target"] = target
    if overlap is not None:
        kwargs["overlap"] = overlap
    chunks = chunk_document(content, title=title, **kwargs)  # coverage-invariant asserted inside

    if not chunks:
        supabase.client.rpc("replace_kb_doc_chunks", {
            "p_doc_id": doc_id, "p_source_hash": digest, "p_rows": [],
        }).execute()
        return {"doc_id": doc_id, "chunks": 0, "embedded": 0, "failed": 0, "content_len": len(content)}

    if embeddings_service is None:
        from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
        embeddings_service = RealEmbeddingsService()
    model = getattr(embeddings_service, "voyage_chunk_model", None)
    if not model:
        from app.config import get_settings
        model = get_settings().voyage_chunk_model

    texts = [embedding_text(title, ch) for ch in chunks]
    embeddings = await embeddings_service.generate_contextualized_embeddings(
        texts, workspace_id=workspace_id,
    )
    failed = sum(1 for e in embeddings if not e)
    if failed:
        # Embed before replacing: the existing rows stay until a full set exists, and the
        # unchanged hash keeps the doc in kb_docs_needing_rechunk for the drain to retry.
        logger.warning("rechunk_doc %s: %d/%d chunks failed to embed; kept existing chunks",
                       doc_id, failed, len(chunks))
        _record_attempt(supabase, doc_id, f"{failed}/{len(chunks)} chunks failed to embed")
        return {"doc_id": doc_id, "error": "embedding_failed", "chunks": 0,
                "embedded": 0, "failed": failed, "content_len": len(content)}

    rows = [{
        "chunk_index": ch.chunk_index,
        "heading": ch.heading or None,
        "content": ch.content,
        "char_start": ch.char_start,
        "char_end": ch.char_end,
        "token_count": len(ch.content) // 4,  # rough
        "text_embedding": _to_vec_literal(emb),
        "embedding_model": model,
        "schema_version": 1,
    } for ch, emb in zip(chunks, embeddings)]

    try:
        supabase.client.rpc("replace_kb_doc_chunks", {
            "p_doc_id": doc_id, "p_source_hash": digest, "p_rows": rows,
        }).execute()
    except Exception as e:
        _record_attempt(supabase, doc_id, f"replace failed: {str(e)[:300]}")
        raise

    return {
        "doc_id": doc_id,
        "chunks": len(chunks),
        "embedded": len(chunks),
        "failed": 0,
        "content_len": len(content),
        "model": model,
    }
