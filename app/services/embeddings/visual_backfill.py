"""Backfill for images that never got their visual (SLIG 768D) vector.

Every other image repair (understanding, aspects) selects on `has_slig_embedding = true`, so an
image whose visual embed failed at creation was invisible to all of them and stayed out of
visual search for good.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.services.core.supabase_client import get_supabase_client
from app.services.embeddings.classification_backfill import (
    NOT_AWAITING_SOURCING_FILTER,
    NOT_QUARANTINED_FILTER,
    NOT_RETIRED_FILTER,
)

logger = logging.getLogger(__name__)


def owes_visual_vector(row: Dict[str, Any]) -> bool:
    """A live material image. Spec icons are saved for spec extraction and are never embedded."""
    meta = row.get("metadata") or {}
    if not isinstance(meta, dict) or meta.get("excluded_from_products") or meta.get("bundle_skipped_reason"):
        return False
    cls = meta.get("ai_classification") or {}
    if isinstance(cls, dict) and (cls.get("classification_pending") or cls.get("is_material") is False):
        return False
    if ((meta.get("image_quality") or {}).get("sourcing")) == "pending":
        return False
    return bool(row.get("image_url"))


def missing_visual_filter(also: str = "") -> str:
    extra = f",or({also})" if also else ""
    return (f"and(or(has_slig_embedding.is.null,has_slig_embedding.is.false),{NOT_RETIRED_FILTER},"
            f"metadata->>bundle_skipped_reason.is.null,"
            f"or({NOT_QUARANTINED_FILTER}),or({NOT_AWAITING_SOURCING_FILTER}){extra})")


async def _fetch_missing(limit: int, image_ids: Optional[List[str]],
                         skip_settling_documents: bool = False) -> List[Dict[str, Any]]:
    client = get_supabase_client().client
    settled = ""
    if skip_settling_documents:
        from app.services.embeddings.ingestion_settling import outside_documents_clause, settling_document_ids
        settled = outside_documents_clause(await settling_document_ids(client))
    query = (client.table("document_images")
             .select("id, image_url, document_id, workspace_id, page_number, metadata, has_slig_embedding")
             .or_(missing_visual_filter(settled)).order("id").limit(limit))
    if image_ids:
        query = query.in_("id", image_ids)
    rows = (await asyncio.to_thread(query.execute)).data or []
    return [r for r in rows if owes_visual_vector(r)]


async def backfill_visual_embeddings(
    max_images: int = 100,
    image_ids: Optional[List[str]] = None,
    skip_settling_documents: bool = False,
) -> Dict[str, Any]:
    """Embed each image's pixels with SLIG and write VECS + `has_slig_embedding`. A failure is
    counted and named, never written as a vector from anywhere else."""
    from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
    from app.services.embeddings.vecs_service import get_vecs_service

    rows = await _fetch_missing(max_images, image_ids, skip_settling_documents)
    summary: Dict[str, Any] = {"scanned": len(rows), "embedded": 0, "failed": 0, "failed_ids": []}
    if not rows:
        return summary
    svc, vecs = RealEmbeddingsService(), get_vecs_service()
    for row in rows:
        embedding = None
        try:
            embedding, _model, _pil = await svc._generate_visual_embedding(
                image_url=row["image_url"], image_data=None,
                workspace_id=row.get("workspace_id"), image_id=row["id"],
            )
        except Exception as e:
            logger.warning(f"visual backfill: SLIG failed for {row['id']}: {e}")
        stored = bool(embedding) and await vecs.upsert_image_embedding(
            image_id=row["id"], siglip_embedding=embedding,
            metadata={"document_id": row.get("document_id"), "workspace_id": row.get("workspace_id"),
                      "page_number": row.get("page_number") or 1},
        )
        if stored:
            summary["embedded"] += 1
        else:
            summary["failed"] += 1
            summary["failed_ids"].append(row["id"])
    logger.info(f"📊 Visual backfill summary: {summary}")
    return summary
