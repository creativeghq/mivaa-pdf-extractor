"""
Text-embedding backfill for products and chunks.

Closes the consumer gap on two failure markers that previously had none:

- Products created with `text_embedding_1024 IS NULL` (Voyage retries
  exhausted during Stage 4 / XML import — Stage 0 stamps
  metadata.embedding_failure but nothing retried it). These products are
  invisible to product-level vector search until re-embedded.
- document_chunks with `has_text_embedding` false/NULL (batch embedding
  failed mid-import in rag_service Step 3b). These chunks are invisible
  to chunk-level RAG retrieval.

Embedding text for products is built by stage_4_products.
build_product_embedding_text — the SAME function the inline path uses —
so backfilled vectors live in the same semantic space. Chunks embed their
`content` verbatim, exactly like the inline batch path.

Triggered by POST /admin/text-embeddings/backfill; bounded by
`max_products` / `max_chunks`; safe to call repeatedly.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.services.core.supabase_client import get_supabase_client
from app.services.embeddings.real_embeddings_service import RealEmbeddingsService

logger = logging.getLogger(__name__)


async def _fetch_products_missing_embedding(
    limit: int,
    workspace_id: Optional[str],
    product_ids: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Products with no text embedding. Explicit product_ids override the
    NULL filter so the admin UI can force a re-embed of specific rows."""
    client = get_supabase_client().client
    query = (
        client.table("products")
        .select("id, name, description, long_description, metadata, workspace_id")
        .order("id")
        .limit(limit)
    )
    if product_ids:
        query = query.in_("id", product_ids)
    else:
        query = query.is_("text_embedding_1024", "null")
    if workspace_id:
        query = query.eq("workspace_id", workspace_id)
    response = await asyncio.to_thread(query.execute)
    return response.data or []


async def _fetch_chunks_missing_embedding(
    limit: int,
    workspace_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Chunks whose text embedding never landed (flag false or NULL)."""
    client = get_supabase_client().client
    query = (
        client.table("document_chunks")
        .select("id, content")
        .or_("has_text_embedding.is.null,has_text_embedding.eq.false")
        .order("id")
        .limit(limit)
    )
    if workspace_id:
        query = query.eq("workspace_id", workspace_id)
    response = await asyncio.to_thread(query.execute)
    return [r for r in (response.data or []) if (r.get("content") or "").strip()]


async def backfill_product_text_embeddings(
    max_products: int = 100,
    workspace_id: Optional[str] = None,
    product_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Re-embed products with NULL text_embedding_1024.

    Returns scanned / embedded / failed counts.
    """
    from app.api.pdf_processing.stage_4_products import (
        _fetch_known_spec_fields,
        build_product_embedding_text,
    )

    supabase = get_supabase_client()
    rows = await _fetch_products_missing_embedding(
        limit=max_products, workspace_id=workspace_id, product_ids=product_ids,
    )
    if not rows:
        return {"scanned": 0, "embedded": 0, "failed": 0}

    logger.info(f"🔄 Product embedding backfill: {len(rows)} product(s) missing text_embedding_1024")

    known_spec_fields = await _fetch_known_spec_fields(supabase, logger)
    embeddings_svc = RealEmbeddingsService()

    embedded = failed = 0
    for row in rows:
        try:
            metadata = row.get("metadata") or {}
            embedding_text = build_product_embedding_text(
                name=row.get("name"),
                description=row.get("description") or row.get("long_description"),
                metadata=metadata if isinstance(metadata, dict) else {},
                known_spec_fields=known_spec_fields,
            )
            if not embedding_text.strip():
                failed += 1
                continue

            emb_result = await embeddings_svc.generate_text_embedding(embedding_text)
            text_emb = emb_result.get("embedding") if emb_result.get("success") else None
            if not text_emb or len(text_emb) != 1024:
                logger.warning(
                    f"⚠️ Backfill embedding failed for product {row['id']}: "
                    f"{emb_result.get('error') or f'dim={len(text_emb) if text_emb else 0}'}"
                )
                failed += 1
                continue

            update_payload: Dict[str, Any] = {
                "text_embedding_1024": "[" + ",".join(str(x) for x in text_emb) + "]",
                "text_embedding_schema_version": 1,
            }
            if emb_result.get("model"):
                update_payload["text_embedding_1024_model"] = emb_result["model"]
            # Clear the failure marker now that the vector landed.
            if isinstance(metadata, dict) and "embedding_failure" in metadata:
                meta = dict(metadata)
                meta["embedding_failure_resolved"] = {
                    **(meta.pop("embedding_failure") or {}),
                    "resolved_at": datetime.utcnow().isoformat(),
                    "resolved_by": "text_embedding_backfill",
                }
                update_payload["metadata"] = meta

            await asyncio.to_thread(
                supabase.client.table("products").update(update_payload)
                .eq("id", row["id"]).execute
            )
            embedded += 1
        except Exception as e:
            logger.error(f"❌ Product embedding backfill failed for {row.get('id')}: {e}")
            failed += 1

    return {"scanned": len(rows), "embedded": embedded, "failed": failed}


async def backfill_chunk_text_embeddings(
    max_chunks: int = 500,
    batch_size: int = 50,
    workspace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Re-embed document_chunks whose batch embedding never landed.

    Mirrors the inline path in rag_service Step 3b: batch Voyage call,
    per-chunk update with has_text_embedding + provenance columns.
    """
    supabase = get_supabase_client()
    rows = await _fetch_chunks_missing_embedding(limit=max_chunks, workspace_id=workspace_id)
    if not rows:
        return {"scanned": 0, "embedded": 0, "failed": 0}

    logger.info(f"🔄 Chunk embedding backfill: {len(rows)} chunk(s) missing text_embedding")

    embeddings_svc = RealEmbeddingsService()
    embedded = failed = 0

    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start:batch_start + batch_size]
        texts = [r["content"] for r in batch]
        try:
            try:
                embeddings_svc._last_provider = None
            except Exception:
                pass
            vectors = await embeddings_svc.generate_batch_embeddings(
                texts=texts, dimensions=1024, input_type="document",
            )
        except Exception as batch_err:
            logger.error(f"❌ Chunk backfill batch embedding failed: {batch_err}")
            failed += len(batch)
            continue

        emb_model = (
            getattr(embeddings_svc, "_last_provider", None)
            or getattr(embeddings_svc, "voyage_model", None)
            or "voyage-3.5"
        )
        emb_now = datetime.utcnow().isoformat()
        for row, vector in zip(batch, vectors or []):
            if not vector:
                failed += 1
                continue
            try:
                await asyncio.to_thread(
                    supabase.client.table("document_chunks").update({
                        "text_embedding": vector,
                        "has_text_embedding": True,
                        "embedding_model": emb_model,
                        "embedding_dimension": len(vector),
                        "embedding_generated_at": emb_now,
                    }).eq("id", row["id"]).execute
                )
                embedded += 1
            except Exception as update_err:
                logger.warning(f"⚠️ Chunk backfill update failed for {row['id']}: {update_err}")
                failed += 1
        # Account for a short vectors list (provider returned fewer than asked).
        if vectors is not None and len(vectors) < len(batch):
            failed += len(batch) - len(vectors)

    return {"scanned": len(rows), "embedded": embedded, "failed": failed}


async def backfill_text_embeddings(
    max_products: int = 100,
    max_chunks: int = 500,
    batch_size: int = 50,
    workspace_id: Optional[str] = None,
    product_ids: Optional[List[str]] = None,
    include_products: bool = True,
    include_chunks: bool = True,
) -> Dict[str, Any]:
    """Combined entry point for the admin endpoint."""
    summary: Dict[str, Any] = {"ok": True}
    if include_products:
        summary["products"] = await backfill_product_text_embeddings(
            max_products=max_products,
            workspace_id=workspace_id,
            product_ids=product_ids,
        )
    if include_chunks and not product_ids:
        summary["chunks"] = await backfill_chunk_text_embeddings(
            max_chunks=max_chunks,
            batch_size=batch_size,
            workspace_id=workspace_id,
        )
    logger.info(f"📊 Text-embedding backfill summary: {summary}")
    return summary
