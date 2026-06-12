"""
Classification backfill for quarantined images.

Targets document_images rows whose metadata.ai_classification
.classification_pending = true — set by Stage 3 when the classifier API
failed (the image was persisted WITHOUT vision analysis or embeddings so
an unverified logo/header can't pollute visual search).

Per image:
  1. Re-run the material classification (Claude Opus vision, same path
     as the per-image /reclassify endpoint).
  2. Clear classification_pending and stamp the fresh verdict.
  3. Non-material → stays embedding-free (correct end state).
  4. Material → run vision analysis (Opus + tool use) and generate the
     FULL embedding set: visual SLIG 768D + understanding 1024D + 4
     aspect vectors, upserted to VECS with provenance — identical wiring
     to the Stage 3 save path.

Triggered by POST /admin/images/classification-backfill; bounded by
`batch_size` / `max_images` so a run can't pin the Anthropic/SLIG/Voyage
clients for hours. Safe to call repeatedly — each processed row either
loses its classification_pending marker or is counted as failed and
retried on the next run.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.services.core.supabase_client import get_supabase_client
from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
from app.services.embeddings.understanding_backfill import (
    _analyze_one,
    _fetch_image_bytes,
    _persist_vision_analysis,
)
from app.services.embeddings.vecs_service import get_vecs_service

logger = logging.getLogger(__name__)


def is_quarantined(row: Dict[str, Any]) -> bool:
    """True when the row carries the classification_pending quarantine marker.

    Shared helper so the understanding/aspect backfills can skip quarantined
    rows — embedding an unverified image would defeat the quarantine.
    """
    meta = row.get("metadata") or {}
    if not isinstance(meta, dict):
        return False
    ai_cls = meta.get("ai_classification") or {}
    if not isinstance(ai_cls, dict):
        return False
    return bool(ai_cls.get("classification_pending"))


async def _fetch_pending_images(
    limit: int,
    workspace_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Find document_images still carrying the quarantine marker."""
    client = get_supabase_client().client
    query = (
        client.table("document_images")
        .select("id, image_url, document_id, workspace_id, page_number, metadata")
        .eq("metadata->ai_classification->>classification_pending", "true")
        .order("id")
        .limit(limit)
    )
    if workspace_id:
        query = query.eq("workspace_id", workspace_id)
    response = await asyncio.to_thread(query.execute)
    return response.data or []


async def _stamp_verdict(
    row: Dict[str, Any],
    classification: Dict[str, Any],
) -> bool:
    """Clear the quarantine marker and persist the fresh verdict.

    Mirrors the column set the per-image /reclassify endpoint writes so
    both paths leave rows in the same shape.
    """
    client = get_supabase_client().client
    is_material = bool(classification.get("is_material", False))
    meta = dict(row.get("metadata") or {})
    meta["ai_classification"] = {
        "is_material": is_material,
        "confidence": classification.get("confidence"),
        "reason": classification.get("reason"),
        "model": classification.get("model"),
        "classification": classification.get("classification"),
        "classification_pending": False,
        "reclassified_at": datetime.utcnow().isoformat(),
        "reclassified_by": "classification_backfill",
    }
    update_data = {
        "classification": "material" if is_material else "non-material",
        "confidence": classification.get("confidence", 0.0),
        "category": "product" if is_material else "general",
        "metadata": meta,
    }
    try:
        result = await asyncio.to_thread(
            client.table("document_images").update(update_data)
            .eq("id", row["id"]).execute
        )
        return bool(result.data)
    except Exception as e:
        logger.error(f"❌ Failed to stamp verdict for image {row['id']}: {e}")
        return False


async def _embed_confirmed_material(
    row: Dict[str, Any],
    image_bytes: bytes,
    anthropic_api_key: str,
    embeddings_svc: RealEmbeddingsService,
    vecs_svc: Any,
) -> bool:
    """Generate the full embedding set for a confirmed-material image.

    Same wiring as the Stage 3 save path: vision analysis → generate_all_embeddings
    → visual/aspect/understanding VECS upserts (flags + provenance are set by
    vecs_service). Returns True when at least the visual vector landed.
    """
    image_id = row["id"]

    va = await _analyze_one(image_bytes, anthropic_api_key, image_id=image_id)
    if va is not None:
        await _persist_vision_analysis(image_id, va)
    else:
        logger.warning(
            f"⚠️ Vision analysis failed for reclassified image {image_id} — "
            f"visual SLIG will still be generated; understanding/aspects need "
            f"a later /admin/understanding-embeddings/backfill run"
        )

    data_url = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    embedding_result = await embeddings_svc.generate_all_embeddings(
        entity_id=image_id,
        entity_type="image",
        text_content="",
        image_data=data_url,
        material_properties={},
        vision_analysis=va.model_dump() if va is not None else None,
    )
    if not embedding_result or not embedding_result.get("success"):
        logger.error(f"❌ Embedding generation failed for reclassified image {image_id}")
        return False

    embeddings = embedding_result.get("embeddings", {})
    meta_versions = embedding_result.get("metadata", {}).get("model_versions", {})
    meta_schemas = embedding_result.get("metadata", {}).get("schema_versions", {})
    vec_meta = {
        "document_id": row.get("document_id"),
        "workspace_id": row.get("workspace_id"),
        "page_number": row.get("page_number") or 1,
    }

    visual_ok = False
    visual_embedding = embeddings.get("visual_768")
    if visual_embedding:
        try:
            await vecs_svc.upsert_image_embedding(
                image_id=image_id,
                siglip_embedding=visual_embedding,
                metadata=vec_meta,
            )
            visual_ok = True
        except Exception as e:
            logger.error(f"❌ Visual VECS upsert failed for {image_id}: {e}")

    specialized = {
        aspect: embeddings.get(f"{aspect}_aspect_1024")
        for aspect in ("color", "texture", "style", "material")
        if embeddings.get(f"{aspect}_aspect_1024")
    }
    if specialized:
        try:
            await vecs_svc.upsert_specialized_embeddings(
                image_id=image_id,
                embeddings=specialized,
                metadata=vec_meta,
                embedding_model=meta_versions.get("specialized_aspect") or "voyage-3",
                schema_version=meta_schemas.get("specialized_aspect"),
            )
        except Exception as e:
            logger.warning(f"⚠️ Aspect VECS upsert failed for {image_id}: {e}")

    understanding = embeddings.get("understanding_1024")
    if understanding:
        try:
            await vecs_svc.upsert_understanding_embedding(
                image_id=image_id,
                embedding=understanding,
                metadata=vec_meta,
                embedding_model=meta_versions.get("understanding") or "voyage-4",
                schema_version=meta_schemas.get("understanding"),
            )
        except Exception as e:
            logger.warning(f"⚠️ Understanding VECS upsert failed for {image_id}: {e}")

    return visual_ok


async def backfill_pending_classifications(
    batch_size: int = 10,
    max_images: int = 100,
    workspace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Re-classify quarantined images and embed the confirmed materials.

    Returns a summary dict with scanned / material / non_material /
    embedded / skipped / failed counts.
    """
    import os
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        return {"ok": False, "error": "ANTHROPIC_API_KEY not configured"}

    rows = await _fetch_pending_images(limit=max_images, workspace_id=workspace_id)
    if not rows:
        return {
            "ok": True, "scanned": 0, "material": 0, "non_material": 0,
            "embedded": 0, "skipped": 0, "failed": 0,
        }

    logger.info(
        f"🔄 Classification backfill: {len(rows)} quarantined image(s) found "
        f"(workspace={workspace_id or 'all'})"
    )

    # RAGService hosts the proven classification path (same as /reclassify).
    from app.services.search.rag_service import RAGService
    rag_service = RAGService()
    embeddings_svc = RealEmbeddingsService()
    vecs_svc = get_vecs_service()

    material = non_material = embedded = skipped = failed = 0

    async def _process(row: Dict[str, Any]) -> str:
        image_bytes = await _fetch_image_bytes(row.get("image_url") or "")
        if not image_bytes:
            return "skipped"

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        classification = await rag_service._classify_image_material(
            image_base64=image_b64,
            confidence_threshold=0.6,
        )
        # A failed re-classification keeps the quarantine marker so the next
        # run retries — do NOT stamp a verdict derived from an API error.
        cls_model = str(classification.get("model") or "")
        if ("error" in classification or "_failed" in cls_model
                or "_empty_response" in cls_model
                or "API key missing" in str(classification.get("reason") or "")):
            return "failed"

        if not await _stamp_verdict(row, classification):
            return "failed"

        if not classification.get("is_material", False):
            return "non_material"

        ok = await _embed_confirmed_material(
            row, image_bytes, anthropic_api_key, embeddings_svc, vecs_svc,
        )
        return "material_embedded" if ok else "material_unembedded"

    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start:batch_start + batch_size]
        outcomes = await asyncio.gather(
            *[_process(r) for r in batch], return_exceptions=True,
        )
        for outcome in outcomes:
            if isinstance(outcome, Exception):
                failed += 1
            elif outcome == "material_embedded":
                material += 1
                embedded += 1
            elif outcome == "material_unembedded":
                material += 1
            elif outcome == "non_material":
                non_material += 1
            elif outcome == "skipped":
                skipped += 1
            else:
                failed += 1

    summary = {
        "ok": True,
        "scanned": len(rows),
        "material": material,
        "non_material": non_material,
        "embedded": embedded,
        "skipped": skipped,
        "failed": failed,
    }
    logger.info(f"📊 Classification backfill summary: {summary}")
    return summary
