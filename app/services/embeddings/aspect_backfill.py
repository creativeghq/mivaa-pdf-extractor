"""
Per-aspect embedding backfill (post-2026-05-04).

Re-embeds the four aspect collections (image_color_embeddings,
image_texture_embeddings, image_style_embeddings, image_material_embeddings)
from cached VisionAnalysis JSON on document_images.vision_analysis. The
aspect strings are derived deterministically from the VisionAnalysis
fields by `app.models.vision_analysis.serialize_aspect_*`, then
Voyage-embedded (1024D) and upserted to VECS.

Stale = (any aspect collection missing) OR (any aspect_schema_version <
current SCHEMA_VERSION) OR (aspect_embedding_model not 'voyage-3'). When
VisionAnalysis JSON itself is missing or unparseable on a target row, the
backfill optionally re-runs Claude Opus 4.7 (same path as the
understanding-embedding backfill) to repopulate it before computing aspects.

This module is the cron- and bulk-mode workhorse. The per-image manual
rebuild endpoint (admin.py /admin/images/{id}/rerun-embeddings) calls
into the same primitives but with explicit-target semantics.

Design notes:
  - Reads cached vision_analysis JSONB from document_images. Rows whose
    JSON predates SCHEMA_VERSION = 2 are still usable because the new
    serializers consume the same field set; the schema_version column is
    purely a freshness marker for THIS module's aspect-embedding output.
  - Per-aspect skip rather than all-or-nothing — if a row's
    VisionAnalysis.colors[] is empty the color aspect is skipped but the
    other three are still embedded. The vecs_service.upsert handles the
    presence flag for each independently.
  - Concurrency capped via the existing Voyage semaphore inside
    RealEmbeddingsService — no new semaphore here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.models.vision_analysis import (
    SCHEMA_VERSION,
    VisionAnalysis,
    vision_analysis_from_legacy_dict,
    serialize_aspect_color,
    serialize_aspect_texture,
    serialize_aspect_style,
    serialize_aspect_material,
)
from app.services.core.supabase_client import get_supabase_client
from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
from app.services.embeddings.vecs_service import get_vecs_service

logger = logging.getLogger(__name__)

# All four aspects — used to drive the stale-row query.
ASPECT_NAMES = ("color", "texture", "style", "material")


def _is_aspect_stale(row: Dict[str, Any], aspect: str) -> bool:
    """True if `row`'s aspect-N collection should be re-embedded.

    Three independent triggers, any one of which marks the row stale:
      1. flag false (`has_<aspect>_slig`) — vector missing from VECS.
      2. schema_version < current — aspect serializer changed since last embed.
      3. embedding_model not voyage — was written by the legacy SLIG path
         (or by an earlier provider that has since been replaced).
    """
    flag = f"has_{aspect}_slig"
    if not row.get(flag):
        return True
    sv = row.get(f"{aspect}_aspect_schema_version")
    if sv is None or sv < SCHEMA_VERSION:
        return True
    em = row.get(f"{aspect}_aspect_embedding_model") or ""
    if not em.startswith("voyage"):
        return True
    return False


def _coerce_vision_analysis(va_raw: Any) -> Optional[VisionAnalysis]:
    """Parse cached VisionAnalysis JSON into the strict Pydantic model.

    Tolerates three shapes seen in production:
      - already a VisionAnalysis instance (admin tool path)
      - dict matching the strict schema (post-2026-05-01 ingestion)
      - legacy free-form dict from older rows (handled by
        `vision_analysis_from_legacy_dict`)
    """
    if va_raw is None:
        return None
    if isinstance(va_raw, VisionAnalysis):
        return va_raw
    if isinstance(va_raw, dict):
        try:
            return VisionAnalysis(**va_raw)
        except Exception:
            return vision_analysis_from_legacy_dict(va_raw)
    return None


async def _fetch_stale_aspect_images(
    limit: int,
    workspace_id: Optional[str],
    document_id: Optional[str],
    image_ids: Optional[List[str]],
    product_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Find document_images that need at least one aspect re-embedded.

    Selectors stack with AND. Most callers pass exactly one — the bulk
    admin endpoint accepts {image_ids[]} OR {document_id} OR {product_id}
    OR none-at-all (cron mode → just `limit`).
    """
    client = get_supabase_client().client

    # Two-tier select: full (with provenance columns) first, falling back
    # to base (no provenance) if the migration hasn't been applied yet.
    # The fallback path treats every row as stale on schema_version because
    # the column doesn't exist — which is exactly the right behavior:
    # pre-migration the aspect collections are still 768D SLIG-blend, every
    # row is "stale" relative to v2, and the backfill can produce v2 vectors
    # the moment migration completes.
    full_select = (
        "id, image_url, document_id, workspace_id, page_number, "
        "vision_analysis, "
        "has_color_slig, has_texture_slig, has_style_slig, has_material_slig, "
        "color_aspect_embedding_model, color_aspect_schema_version, "
        "texture_aspect_embedding_model, texture_aspect_schema_version, "
        "style_aspect_embedding_model, style_aspect_schema_version, "
        "material_aspect_embedding_model, material_aspect_schema_version"
    )
    base_select = (
        "id, image_url, document_id, workspace_id, page_number, "
        "vision_analysis, "
        "has_color_slig, has_texture_slig, has_style_slig, has_material_slig"
    )

    def _build_query(select_str: str):
        q = client.table("document_images").select(select_str).order("id").limit(limit)
        if workspace_id:
            q = q.eq("workspace_id", workspace_id)
        if document_id:
            q = q.eq("document_id", document_id)
        if image_ids:
            q = q.in_("id", image_ids)
        return q

    # Resolve product_id → image_ids ONCE before building the query so the
    # narrowing applies to either select-shape we choose below.
    if product_id:
        assoc = await asyncio.to_thread(
            client.table("image_product_associations")
            .select("image_id")
            .eq("product_id", product_id)
            .execute
        )
        product_image_ids = [r["image_id"] for r in (assoc.data or []) if r.get("image_id")]
        if not product_image_ids:
            return []
        image_ids = list(set((image_ids or []) + product_image_ids))

    # Try the full select first; on PostgREST 'column does not exist' (42703)
    # fall back to the base select. Migration is REQUIRED before the v2 path
    # produces useful output, but pre-migration we can still surface "every
    # row is stale" so the operator sees something rather than a 500.
    try:
        response = await asyncio.to_thread(_build_query(full_select).execute)
    except Exception as e:
        err_msg = str(e).lower()
        if "does not exist" in err_msg or "42703" in err_msg:
            logger.warning(
                "⚠️ Aspect provenance columns missing — falling back to base select. "
                "Apply the v2 SQL migration to enable provenance-aware staleness detection. "
                f"({e})"
            )
            response = await asyncio.to_thread(_build_query(base_select).execute)
        else:
            raise

    rows = response.data or []
    stale = [r for r in rows if any(_is_aspect_stale(r, a) for a in ASPECT_NAMES)]
    return stale


async def _generate_aspects_for_row(
    row: Dict[str, Any],
    embeddings_svc: RealEmbeddingsService,
) -> Optional[Dict[str, List[float]]]:
    """Build the aspect text strings, embed each via Voyage 1024D, return dict.

    Returns None when the row's vision_analysis is unusable. Per-aspect
    skip — if a particular aspect's source fields are empty (e.g. no
    colors[]) that aspect is omitted from the returned dict but the others
    still get embedded.
    """
    va = _coerce_vision_analysis(row.get("vision_analysis"))
    if va is None:
        return None

    aspect_texts: Dict[str, Optional[str]] = {
        "color":    serialize_aspect_color(va),
        "texture":  serialize_aspect_texture(va),
        "style":    serialize_aspect_style(va),
        "material": serialize_aspect_material(va),  # always non-None
    }

    embeddings: Dict[str, List[float]] = {}
    for aspect, text in aspect_texts.items():
        if not text:
            continue
        try:
            # allow_openai_fallback=False — never let an OpenAI 1024D vector
            # land in a Voyage 1024D collection. Same audit-gap-B discipline
            # applied to image_understanding_embeddings.
            vec = await embeddings_svc._generate_text_embedding(
                text=text,
                input_type="document",
                allow_openai_fallback=False,
            )
            if vec and len(vec) == 1024:
                embeddings[aspect] = vec
            else:
                logger.warning(
                    f"⚠️ Aspect '{aspect}' embed wrong dim or empty for image {row.get('id')}"
                )
        except Exception as e:
            logger.error(f"❌ Aspect '{aspect}' embed failed for image {row.get('id')}: {e}")
    return embeddings or None


async def backfill_aspect_embeddings(
    batch_size: int = 25,
    max_images: int = 200,
    workspace_id: Optional[str] = None,
    document_id: Optional[str] = None,
    image_ids: Optional[List[str]] = None,
    product_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Re-embed stale aspect collections from cached VisionAnalysis JSON.

    Cheap path: reads vision_analysis JSON straight from document_images,
    Voyage-embeds 4 short strings per image, upserts to VECS. ~$0.0001 per
    image. Does NOT re-run Claude Opus 4.7 — that's the understanding-
    embedding backfill's job. If a row has no usable vision_analysis JSON,
    it's reported as `failed` and the operator can chain
    /admin/understanding-embeddings/backfill (which DOES re-run Opus)
    before re-running this one.

    Args:
        batch_size: Concurrent images per gather batch
        max_images: Hard ceiling on rows scanned per call
        workspace_id: Optional workspace filter (admin scope)
        document_id: Optional document filter
        image_ids: Optional explicit image-id list (bulk-admin selector)
        product_id: Optional product filter — joins through
            image_product_associations to find the linked images
    Returns:
        Dict summary: {ok, scanned, reembedded, skipped, failed,
        partial, schema_version}. `partial` counts rows where some but not
        all aspects were embedded (legitimate when a row's
        VisionAnalysis lacks one of the source fields).
    """
    embeddings_svc = RealEmbeddingsService()
    vecs_svc = get_vecs_service()

    rows = await _fetch_stale_aspect_images(
        limit=max_images,
        workspace_id=workspace_id,
        document_id=document_id,
        image_ids=image_ids,
        product_id=product_id,
    )
    if not rows:
        return {
            "ok": True,
            "scanned": 0,
            "reembedded": 0,
            "skipped": 0,
            "failed": 0,
            "partial": 0,
            "schema_version": SCHEMA_VERSION,
        }

    logger.info(
        f"🔄 Aspect backfill: {len(rows)} stale image(s) found "
        f"(workspace={workspace_id or 'all'}, document={document_id or 'all'}, "
        f"product={product_id or 'all'}, explicit_ids={'yes' if image_ids else 'no'})"
    )

    reembedded = 0
    skipped = 0
    failed = 0
    partial = 0

    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start:batch_start + batch_size]

        async def _process(row: Dict[str, Any]) -> str:
            aspect_embeddings = await _generate_aspects_for_row(row, embeddings_svc)
            if not aspect_embeddings:
                # Either VA missing/unparseable OR every aspect string empty.
                # The understanding-backfill path can repopulate VA from Opus
                # if needed — operator chains the two endpoints when that
                # happens. We don't fan out to Anthropic here on purpose
                # (cost discipline).
                return "skipped"

            # Aspect upsert. embedding_model + schema_version are persisted
            # on document_images so subsequent staleness checks see this row
            # as up-to-date and don't re-process it.
            results = await vecs_svc.upsert_specialized_embeddings(
                image_id=row["id"],
                embeddings=aspect_embeddings,
                metadata={
                    "document_id": row.get("document_id"),
                    "workspace_id": row.get("workspace_id"),
                    "page_number": row.get("page_number") or 1,
                },
                embedding_model="voyage-3",
                schema_version=SCHEMA_VERSION,
            )
            successes = sum(1 for v in results.values() if v)
            attempted = len(aspect_embeddings)
            if successes == 0:
                return "failed"
            if successes < attempted:
                return "partial"
            return "reembedded"

        outcomes = await asyncio.gather(
            *[_process(r) for r in batch],
            return_exceptions=True,
        )
        for o in outcomes:
            if isinstance(o, Exception):
                logger.error(f"Aspect backfill row exception: {o}")
                failed += 1
            elif o == "reembedded":
                reembedded += 1
            elif o == "partial":
                partial += 1
            elif o == "skipped":
                skipped += 1
            elif o == "failed":
                failed += 1

    summary = {
        "ok": True,
        "scanned": len(rows),
        "reembedded": reembedded,
        "partial": partial,
        "skipped": skipped,
        "failed": failed,
        "schema_version": SCHEMA_VERSION,
    }
    logger.info(f"📊 Aspect backfill summary: {summary}")
    return summary


async def rerun_aspect_embeddings_for_image(
    image_id: str,
    aspects: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Per-image manual rebuild — synchronous, used by the admin endpoint.

    Reads the row's cached vision_analysis JSON, generates aspect
    embeddings (only for aspects in `aspects`, or all four if None),
    upserts to VECS. Returns a detailed result so the admin tool can
    show the operator exactly what changed.

    Differs from `backfill_aspect_embeddings` in three ways:
      - Targets exactly one image (no batching, no concurrency)
      - Bypasses the staleness gate (force=true semantics)
      - Returns per-aspect outcome + the source text used so the
        diagnostic UI can render "what was actually embedded for this
        aspect" without a follow-up query
    """
    aspects = aspects or list(ASPECT_NAMES)

    client = get_supabase_client().client
    # Use .maybe_single() rather than .single() — the latter raises on
    # zero rows (PostgREST 406), which would translate into a 500 from
    # the admin endpoint instead of the cleaner 404 we return below.
    try:
        row_result = await asyncio.to_thread(
            client.table("document_images").select(
                "id, document_id, workspace_id, page_number, vision_analysis"
            ).eq("id", image_id).maybe_single().execute
        )
    except Exception as e:
        return {"ok": False, "error": "image_lookup_failed", "image_id": image_id, "detail": str(e)}
    row = row_result.data
    if not row:
        return {"ok": False, "error": "image_not_found", "image_id": image_id}

    va = _coerce_vision_analysis(row.get("vision_analysis"))
    if va is None:
        return {
            "ok": False,
            "error": "vision_analysis_missing_or_unparseable",
            "image_id": image_id,
            "hint": "Run /admin/understanding-embeddings/backfill or "
                    "/admin/images/{id}/rerun-embeddings with rerun_vision_analysis=true first.",
        }

    serializers = {
        "color":    serialize_aspect_color,
        "texture":  serialize_aspect_texture,
        "style":    serialize_aspect_style,
        "material": serialize_aspect_material,
    }
    embeddings_svc = RealEmbeddingsService()

    aspect_embeddings: Dict[str, List[float]] = {}
    aspect_source_texts: Dict[str, Optional[str]] = {}
    aspect_errors: Dict[str, str] = {}

    for aspect in aspects:
        if aspect not in serializers:
            aspect_errors[aspect] = "unknown_aspect"
            continue
        text = serializers[aspect](va)
        aspect_source_texts[aspect] = text
        if not text:
            continue
        try:
            vec = await embeddings_svc._generate_text_embedding(
                text=text,
                input_type="document",
                allow_openai_fallback=False,
            )
            if vec and len(vec) == 1024:
                aspect_embeddings[aspect] = vec
            else:
                aspect_errors[aspect] = f"wrong_dim_or_empty (len={len(vec) if vec else 0})"
        except Exception as e:
            aspect_errors[aspect] = f"embed_failed: {e}"

    upsert_results: Dict[str, bool] = {}
    if aspect_embeddings:
        vecs_svc = get_vecs_service()
        upsert_results = await vecs_svc.upsert_specialized_embeddings(
            image_id=image_id,
            embeddings=aspect_embeddings,
            metadata={
                "document_id": row.get("document_id"),
                "workspace_id": row.get("workspace_id"),
                "page_number": row.get("page_number") or 1,
            },
            embedding_model="voyage-3",
            schema_version=SCHEMA_VERSION,
        )

    return {
        "ok": True,
        "image_id": image_id,
        "schema_version": SCHEMA_VERSION,
        "aspects_requested": aspects,
        "aspects_embedded": list(aspect_embeddings.keys()),
        "aspects_skipped_empty_text": [
            a for a, t in aspect_source_texts.items() if not t and a not in aspect_errors
        ],
        "aspects_errored": aspect_errors,
        "source_texts": aspect_source_texts,
        "upsert_results": upsert_results,
    }
