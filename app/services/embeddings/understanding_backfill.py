"""
Understanding-embedding backfill.

Re-runs vision_analysis (Claude Opus + tool use) on document_images that
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
    VISION_MAX_TOKENS,
    vision_call_extra_kwargs,
)
from app.config import get_settings
from app.services.images.vision_prompt import load_material_analyzer_prompt
from app.services.core.supabase_client import get_supabase_client
from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
from app.services.embeddings.vecs_service import get_vecs_service
from app.utils.ssrf_guard import SSRFError, assert_safe_url

logger = logging.getLogger(__name__)


async def _fetch_stale_images(
    limit: int,
    workspace_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Find document_images whose understanding embedding is stale.

    Stale = (no embedding) OR (schema_version < current SCHEMA_VERSION) OR
    (embedding_model is not 'voyage-4' — i.e. served by OpenAI fallback).
    """
    from app.services.embeddings.classification_backfill import is_quarantined

    client = get_supabase_client().client
    query = (
        client.table("document_images")
        .select(
            "id, image_url, has_understanding_embedding, "
            "understanding_embedding_model, understanding_schema_version, "
            "document_id, workspace_id, page_number, metadata"
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
        # Quarantined rows (classification_pending) intentionally have NO
        # embeddings — embedding them here would defeat the quarantine.
        # The classification backfill re-classifies them first.
        if is_quarantined(row):
            continue
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


#: 20 MB. An image that large is already pathological; the point is that the cap is
#: enforced against the bytes DELIVERED, not against a Content-Length the server claims.
_MAX_IMAGE_BYTES = 20 * 1024 * 1024


async def _fetch_image_bytes(image_url: str) -> Optional[bytes]:
    """Fetch the image bytes for analysis. Best-effort — None on failure.

    This explicitly set `follow_redirects=True` with no SSRF guard and no size cap
    (audit #14 MV-8), so a stored image_url could redirect the backfill into
    169.254.169.254 or any RFC1918 address, and `except Exception: return None`
    made a blocked fetch indistinguishable from "no image".
    """
    try:
        safe_url = assert_safe_url(image_url)
    except SSRFError as e:
        # Logged, not silent: "the URL was refused" and "the image is missing" need
        # different fixes, and both used to arrive as None.
        logger.warning("understanding backfill: refusing %s — %s", image_url, e)
        return None
    try:
        # follow_redirects=False is load-bearing: the guard checked the host it was
        # given, and a redirect moves the request to one nothing checked.
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            resp = await client.get(safe_url)
            if resp.status_code != 200:
                return None
            body = resp.content
            if len(body) > _MAX_IMAGE_BYTES:
                logger.warning("understanding backfill: %s exceeded the size cap", image_url)
                return None
            return body
    except Exception:
        return None


async def _analyze_one(
    image_bytes: bytes,
    anthropic_api_key: str,
    image_id: Optional[str] = None,
) -> Optional[VisionAnalysis]:
    """Run Claude Opus + tool use to produce a schema-conformant VisionAnalysis.

    Validation failures are logged with the specific field/type the model
    emitted that broke the schema — so when Claude drifts (e.g. starts
    emitting an extra unknown field that `extra='forbid'` rejects, or
    drops a required field), operators see WHICH field failed instead of
    a silent "None returned".
    """
    from pydantic import ValidationError
    image_b64 = base64.b64encode(image_bytes).decode()
    # Through the tracked helper, not a raw POST (#33 item 2). This is an OPUS vision
    # call per image and it logged NOTHING — the backfill's entire spend was invisible
    # to every cost view. It self-logs no usage, so routing it here adds a cost record
    # rather than duplicating one (see .github/anthropic-bypass-baseline.json for the
    # files where the opposite is true).
    try:
        from app.services.core.claude_tool_call import call_with_tool, ToolCallNotReturned

        try:
            result = await call_with_tool(
                task="understanding_backfill_vision",
                model=get_settings().anthropic_model_validation,
                max_tokens=VISION_MAX_TOKENS,
                extra_kwargs=vision_call_extra_kwargs(),
                messages=[{
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
                            # The SAME prompt row ingestion uses, not a hardcoded twin.
                            # This said "Use the emit_vision_analysis tool to return a
                            # structured catalog-grade material analysis for this image"
                            # — a different instruction from the one every ingested image
                            # received, writing into the same embedding collection. The
                            # backfill exists to make stale rows match current ones; a
                            # private prompt made it produce a third regime instead.
                            "text": load_material_analyzer_prompt()[0],
                        },
                    ],
                }],
                tool=VISION_ANALYSIS_TOOL,
                image_id=image_id,
            )
        except ToolCallNotReturned as e:
            logger.warning(
                f"Anthropic backfill returned no usable tool_use block "
                f"(image_id={image_id or '<none>'}): {e}"
            )
            return None

        try:
            return VisionAnalysis(**result.data)
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
        # `image_product_associations` has no `document_id` (#26 M13-2), so this read
        # errored and the whole rollup recompute raised on every backfill run — leaving
        # `products.metadata` stale behind correct embeddings, which is precisely the
        # state this function exists to prevent.
        #
        # The document a rollup belongs to is the PRODUCT's source document, so it is
        # resolved from `products.source_document_id` in a second read rather than
        # guessed at from the image.
        assoc_resp = await asyncio.to_thread(
            lambda: client.table("image_product_associations")
            .select("product_id")
            .in_("image_id", image_ids)
            .execute()
        )
        product_ids = sorted({
            r.get("product_id") for r in (assoc_resp.data or []) if r.get("product_id")
        })
        if not product_ids:
            return 0

        docs_resp = await asyncio.to_thread(
            lambda: client.table("products")
            .select("id, source_document_id")
            .in_("id", product_ids)
            .execute()
        )
        # Group products by their document so we can call the per-document enrich
        # function once per (document, product) pair.
        per_doc_products: Dict[str, set] = {}
        for r in docs_resp.data or []:
            doc_id = r.get("source_document_id")
            pid = r.get("id")
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
