"""The one embed run of a document whose job deferred its vectors (#513): products, chunks, images, pages.

Every part reuses the path the inline pipeline and the backfills already use, so a deferred
vector lives in the same space as an inline one. A part that fails is counted, never skipped.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

PAGE = 1000
ID_BATCH = 100
MAX_CHUNK_ROUNDS = 50


def owes_image_vector(row: Dict[str, Any], linked: Set[str]) -> bool:
    """A live material image some product holds, with no visual vector, not waiting on image sourcing."""
    from app.services.pdf.ingestion_audit import owes_a_product
    quality = (row.get("metadata") or {}).get("image_quality") or {}
    return (row["id"] in linked and owes_a_product(row) and not row.get("has_slig_embedding")
            and quality.get("sourcing") != "pending")


def _all(make: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    while True:
        batch = make().order("id").range(len(rows), len(rows) + PAGE - 1).execute().data or []
        rows.extend(batch)
        if len(batch) < PAGE:
            return rows


async def embed_document(document_id: str, workspace_id: str, job_id: Optional[str] = None,
                         user_id: Optional[str] = None) -> Dict[str, Any]:
    from app.services.core.supabase_client import get_supabase_client
    from app.services.embeddings.classification_backfill import _embed_confirmed_material
    from app.services.embeddings.page_embedding_service import get_page_embedding_service
    from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
    from app.services.embeddings.text_embedding_backfill import (
        backfill_chunk_text_embeddings, backfill_product_text_embeddings,
    )
    from app.services.embeddings.vecs_service import get_vecs_service

    from app.services.embeddings.embedding_deferral import mark_document
    db = get_supabase_client().client
    await asyncio.to_thread(mark_document, db, document_id, False)
    out: Dict[str, Any] = {"document_id": document_id, "products": {"embedded": 0, "failed": 0},
                           "chunks": {"embedded": 0, "failed": 0}, "images": {"embedded": 0, "failed": 0, "failed_ids": []},
                           "pages": None, "errors": []}

    product_ids = [r["id"] for r in await asyncio.to_thread(
        _all, lambda: db.table("products").select("id").eq("source_document_id", document_id))]
    for start in range(0, len(product_ids), ID_BATCH):
        ids = product_ids[start:start + ID_BATCH]
        try:
            r = await backfill_product_text_embeddings(max_products=len(ids), workspace_id=workspace_id, product_ids=ids)
            out["products"]["embedded"] += int(r.get("embedded") or 0)
            out["products"]["failed"] += int(r.get("failed") or 0)
        except Exception as e:
            out["errors"].append(f"products: {type(e).__name__}: {e}"[:300])

    for _ in range(MAX_CHUNK_ROUNDS):
        try:
            r = await backfill_chunk_text_embeddings(max_chunks=500, document_id=document_id)
        except Exception as e:
            out["errors"].append(f"chunks: {type(e).__name__}: {e}"[:300])
            break
        out["chunks"]["embedded"] += int(r.get("embedded") or 0)
        out["chunks"]["failed"] += int(r.get("failed") or 0)
        if not r.get("embedded"):
            break

    images = await asyncio.to_thread(_all, lambda: db.table("document_images").select(
        "id, image_url, workspace_id, document_id, page_number, image_type, metadata, has_slig_embedding, extraction_layer"
    ).eq("document_id", document_id))
    linked: Set[str] = set()
    for start in range(0, len(images), ID_BATCH):
        ids = [i["id"] for i in images[start:start + ID_BATCH]]
        linked |= {a["image_id"] for a in await asyncio.to_thread(
            _all, lambda ids=ids: db.table("image_product_associations").select("id, image_id").in_("image_id", ids))}
    owed = [i for i in images if owes_image_vector(i, linked)]
    svc, vecs = RealEmbeddingsService(), get_vecs_service()
    for row in owed:
        meta = row.get("metadata") or {}
        try:
            data = await asyncio.to_thread(db.storage.from_(meta["storage_bucket"]).download, meta["storage_path"])
            ok = await _embed_confirmed_material({**row, "workspace_id": row.get("workspace_id") or workspace_id}, data,
                                                 os.getenv("ANTHROPIC_API_KEY", ""), svc, vecs)
        except Exception as e:
            ok = False
            out["errors"].append(f"image {row['id']}: {type(e).__name__}: {e}"[:200])
        out["images"]["embedded" if ok else "failed"] += 1
        if not ok:
            out["images"]["failed_ids"].append(row["id"])

    try:
        out["pages"] = await get_page_embedding_service().embed_document_pages(
            document_id=document_id, workspace_id=workspace_id, job_id=job_id, user_id=user_id)
    except Exception as e:
        out["errors"].append(f"pages: {type(e).__name__}: {e}"[:300])
    return out


async def run_embed_step(document_id: str, workspace_id: str, job_id: Optional[str] = None,
                         user_id: Optional[str] = None) -> Dict[str, Any]:
    """Embed, record the outcome as a correction, settle the ready_to_embed decision, then audit post-embed."""
    from datetime import datetime, timezone

    from app.config import get_settings
    from app.services.core.supabase_client import get_supabase_client
    from app.services.pdf.ingestion_audit_runner import run_ingestion_audit
    from app.services.pdf.ingestion_corrections import CORRECTED, DEGRADED, Correction, record

    from app.services.integrations.billing_principal import billing_principal
    payer = user_id or billing_principal(get_supabase_client().client, workspace_id, document_id)
    result = await embed_document(document_id, workspace_id, job_id, payer)
    failed = result["products"]["failed"] + result["chunks"]["failed"] + result["images"]["failed"] + len(result["errors"])
    done = result["products"]["embedded"] + result["chunks"]["embedded"] + result["images"]["embedded"]
    db = get_supabase_client().client
    corrections = [Correction("document_embedded", CORRECTED,
                              f"Embedded {result['products']['embedded']} product(s), {result['chunks']['embedded']} chunk(s), "
                              f"{result['images']['embedded']} image(s)", done, detail={"pages": result.get("pages")})]
    if failed:
        corrections.append(Correction("embedding_incomplete", DEGRADED,
                                      f"{failed} item(s) could not be embedded; re-run the embed step", failed,
                                      result["images"]["failed_ids"][:200], {"errors": result["errors"][:20]}))
    await record(db, document_id, workspace_id, "embedding", corrections, job_id)
    await asyncio.to_thread(lambda: db.table("ingestion_corrections").update({
        "acknowledged_at": datetime.now(timezone.utc).isoformat(), "acknowledged_by": user_id,
    }).eq("document_id", document_id).eq("code", "ready_to_embed").is_("acknowledged_at", "null").execute())
    result["audit"] = await run_ingestion_audit(document_id, get_supabase_client(), get_settings(), job_id=job_id,
                                                phase="post_embed", trigger="pipeline")
    return result
