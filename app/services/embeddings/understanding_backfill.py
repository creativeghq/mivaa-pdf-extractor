"""
Understanding-embedding backfill.

Re-runs vision_analysis (Claude Opus 4.7 + tool use) on document_images that
either (a) lack an understanding embedding, (b) were embedded under a stale
VisionAnalysis schema_version, or (c) were embedded by the OpenAI fallback
rather than Voyage (audit gap A — embedding-space drift).

Triggered by an admin endpoint or cron; bounded by `batch_size` and
`max_images` so a single run can't pin the Anthropic + Voyage clients for
hours. Concurrency is capped via the existing Voyage semaphore inside
RealEmbeddingsService.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Dict, List, Optional

import httpx

from app.models.vision_analysis import (
    SCHEMA_VERSION,
    VisionAnalysis,
    VISION_ANALYSIS_TOOL,
)
from app.services.core.supabase_client import get_supabase_client
from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
from app.services.embeddings.vecs_service import get_vecs_service

logger = logging.getLogger(__name__)


async def _fetch_stale_images(
    limit: int,
    workspace_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Find document_images whose understanding embedding is stale.

    Stale = (no embedding) OR (schema_version < current SCHEMA_VERSION) OR
    (embedding_model is not 'voyage-4' — i.e. served by OpenAI fallback).
    """
    client = get_supabase_client().client
    query = (
        client.table("document_images")
        .select(
            "id, image_url, has_understanding_embedding, "
            "understanding_embedding_model, understanding_schema_version, "
            "document_id, workspace_id, page_number"
        )
        .order("id")
        .limit(limit)
    )
    if workspace_id:
        query = query.eq("workspace_id", workspace_id)

    response = await asyncio.to_thread(query.execute)
    rows = response.data or []
    stale = []
    for row in rows:
        if not row.get("has_understanding_embedding"):
            stale.append(row)
            continue
        sv = row.get("understanding_schema_version")
        if sv is None or sv < SCHEMA_VERSION:
            stale.append(row)
            continue
        em = row.get("understanding_embedding_model") or ""
        if not em.startswith("voyage"):
            stale.append(row)
            continue
    return stale


async def _fetch_image_bytes(image_url: str) -> Optional[bytes]:
    """Fetch the image bytes for analysis. Best-effort — None on failure."""
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(image_url)
            if resp.status_code != 200:
                return None
            return resp.content
    except Exception:
        return None


async def _analyze_one(
    image_bytes: bytes,
    anthropic_api_key: str,
    image_id: Optional[str] = None,
) -> Optional[VisionAnalysis]:
    """Run Claude Opus 4.7 + tool use to produce a schema-conformant VisionAnalysis.

    Validation failures are logged with the specific field/type the model
    emitted that broke the schema — so when Claude drifts (e.g. starts
    emitting an extra unknown field that `extra='forbid'` rejects, or
    drops a required field), operators see WHICH field failed instead of
    a silent "None returned".
    """
    from pydantic import ValidationError
    image_b64 = base64.b64encode(image_bytes).decode()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-opus-4-7",
                    "max_tokens": 4096,
                    "tools": [VISION_ANALYSIS_TOOL],
                    "tool_choice": {"type": "tool", "name": "emit_vision_analysis"},
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Use the emit_vision_analysis tool to "
                                    "return a structured catalog-grade material "
                                    "analysis for this image."
                                ),
                            },
                        ],
                    }],
                },
            )
            if resp.status_code != 200:
                logger.warning(
                    f"Anthropic non-200 in backfill (image_id={image_id or '<none>'}): "
                    f"status={resp.status_code} body={resp.text[:300]}"
                )
                return None
            payload = resp.json()
            tool_block = next(
                (b for b in payload.get("content", []) if b.get("type") == "tool_use"),
                None,
            )
            if not tool_block:
                logger.warning(
                    f"Anthropic backfill returned no tool_use block "
                    f"(image_id={image_id or '<none>'}, stop_reason={payload.get('stop_reason')})"
                )
                return None
            try:
                return VisionAnalysis(**tool_block["input"])
            except ValidationError as ve:
                # Schema drift from the model — show exactly which field broke
                # so a prompt or schema bump can be planned.
                error_summary = "; ".join(
                    f"{'.'.join(str(x) for x in err.get('loc', ()))}={err.get('type')}"
                    for err in ve.errors()[:3]
                )
                logger.warning(
                    f"⚠️ Anthropic VisionAnalysis validation failed "
                    f"(image_id={image_id or '<none>'}): {error_summary}"
                )
                return None
    except Exception as e:
        logger.warning(
            f"Anthropic call failed in backfill (image_id={image_id or '<none>'}): {e}"
        )
        return None


async def _persist_vision_analysis(image_id: str, va: VisionAnalysis) -> bool:
    """Persist the freshly-extracted VisionAnalysis back to document_images.

    Without this, after backfill the embedding is correct but the stored
    `vision_analysis` JSON on the row is the old (stale) shape — so the
    Stage 4.7 product rollup keeps reading the legacy data.
    """
    try:
        client = get_supabase_client().client
        await asyncio.to_thread(
            lambda: client.table("document_images")
            .update({
                "vision_analysis": va.model_dump(),
                "understanding_schema_version": SCHEMA_VERSION,
            })
            .eq("id", image_id)
            .execute()
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to persist vision_analysis for image {image_id}: {e}")
        return False


async def _trigger_product_rollup_recompute(image_ids: List[str]) -> int:
    """After re-embedding images, recompute the product-level metadata rollup
    for every product that owns any of these images.

    The rollup is per-product and pulls from `document_images.vision_analysis`,
    so the embeddings can be correct while `products.metadata` is still stale
    until this runs. Returns the number of products re-rolled.
    """
    if not image_ids:
        return 0
    try:
        client = get_supabase_client().client
        # Find every product associated with any of these images via
        # image_product_associations.
        assoc_resp = await asyncio.to_thread(
            lambda: client.table("image_product_associations")
            .select("product_id, document_id")
            .in_("image_id", image_ids)
            .execute()
        )
        rows = assoc_resp.data or []
        # Group products by their document_id so we can call the per-document
        # enrich function once per (document, product) pair.
        per_doc_products: Dict[str, set] = {}
        for r in rows:
            doc_id = r.get("document_id")
            pid = r.get("product_id")
            if not doc_id or not pid:
                continue
            per_doc_products.setdefault(doc_id, set()).add(pid)

        if not per_doc_products:
            return 0

        # Lazy import — `stage_4_products` imports things we don't want loaded
        # at module-init time (it pulls in heavy LLM clients).
        from app.api.pdf_processing.stage_4_products import (
            enrich_products_from_chunks_and_vision,
        )

        rerolled = 0
        for doc_id, pids in per_doc_products.items():
            for pid in pids:
                try:
                    await enrich_products_from_chunks_and_vision(
                        document_id=doc_id,
                        supabase=client,
                        logger=logger,
                        target_product_id=pid,
                        # Only the rollup matters here — skip the heavy spec
                        # extractor / description writer on backfill.
                        enable_spec_vision=False,
                        enable_description_writer=False,
                        enable_layout_analyzer=False,
                        enable_legend_extractor=False,
                    )
                    rerolled += 1
                except Exception as inner:
                    logger.warning(f"Rollup recompute failed for product {pid}: {inner}")
        return rerolled
    except Exception as e:
        logger.warning(f"Product rollup recompute failed: {e}")
        return 0


async def backfill_understanding_embeddings(
    batch_size: int = 25,
    max_images: int = 200,
    workspace_id: Optional[str] = None,
    recompute_product_rollup: bool = True,
) -> Dict[str, Any]:
    """Re-run vision_analysis + Voyage on stale rows.

    Returns a summary dict — counts only, not the embeddings themselves
    (those go straight into VECS).

    Args:
        recompute_product_rollup: When True (default), after re-embedding,
            also recompute Stage 4.7 product-level metadata rollup for every
            product owning any of the re-embedded images. Set False to skip
            (e.g. emergency embedding-only backfill that shouldn't touch
            product metadata).
    """
    import os
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        return {"ok": False, "error": "ANTHROPIC_API_KEY not configured"}

    embeddings_svc = RealEmbeddingsService()
    vecs_svc = get_vecs_service()

    rows = await _fetch_stale_images(limit=max_images, workspace_id=workspace_id)
    if not rows:
        return {"ok": True, "scanned": 0, "reembedded": 0, "skipped": 0, "failed": 0, "products_rerolled": 0}

    logger.info(f"🔄 Backfill: {len(rows)} stale image(s) found (workspace={workspace_id or 'all'})")

    reembedded = 0
    skipped = 0
    failed = 0
    successfully_reembedded_image_ids: List[str] = []

    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start:batch_start + batch_size]

        async def _process(row: Dict[str, Any]) -> str:
            image_bytes = await _fetch_image_bytes(row.get("image_url") or "")
            if not image_bytes:
                return "skipped"

            va = await _analyze_one(image_bytes, anthropic_api_key, image_id=row.get("id"))
            if va is None:
                return "failed"

            ue_result = await embeddings_svc.generate_understanding_embedding(
                vision_analysis=va,
                job_id=None,
            )
            if not ue_result or not ue_result.get("embedding"):
                return "failed"

            ok = await vecs_svc.upsert_understanding_embedding(
                image_id=row["id"],
                embedding=ue_result["embedding"],
                metadata={
                    "document_id": row.get("document_id"),
                    "workspace_id": row.get("workspace_id"),
                    "page_number": row.get("page_number") or 1,
                },
                embedding_model=ue_result.get("embedding_model"),
                schema_version=ue_result.get("schema_version"),
            )
            if not ok:
                return "failed"

            # Persist the fresh VA back to document_images so Stage 4.7
            # rollup reads the same data as Voyage embedded.
            await _persist_vision_analysis(row["id"], va)
            return "reembedded"

        outcomes_with_ids = await asyncio.gather(
            *[_process(r) for r in batch], return_exceptions=True,
        )
        for r, o in zip(batch, outcomes_with_ids):
            if isinstance(o, Exception):
                failed += 1
            elif o == "reembedded":
                reembedded += 1
                successfully_reembedded_image_ids.append(r["id"])
            elif o == "skipped":
                skipped += 1
            elif o == "failed":
                failed += 1

    products_rerolled = 0
    if recompute_product_rollup and successfully_reembedded_image_ids:
        logger.info(
            f"🔁 Recomputing product rollup for {len(successfully_reembedded_image_ids)} re-embedded image(s)…"
        )
        products_rerolled = await _trigger_product_rollup_recompute(
            successfully_reembedded_image_ids
        )
        logger.info(f"   → re-rolled {products_rerolled} product(s)")

    return {
        "ok": True,
        "scanned": len(rows),
        "reembedded": reembedded,
        "skipped": skipped,
        "failed": failed,
        "products_rerolled": products_rerolled,
        "schema_version": SCHEMA_VERSION,
    }
