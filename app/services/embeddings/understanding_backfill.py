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
) -> Optional[VisionAnalysis]:
    """Run Claude Opus 4.7 + tool use to produce a schema-conformant VisionAnalysis."""
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
                logger.warning(f"Anthropic non-200 in backfill: {resp.status_code}")
                return None
            payload = resp.json()
            tool_block = next(
                (b for b in payload.get("content", []) if b.get("type") == "tool_use"),
                None,
            )
            if not tool_block:
                return None
            return VisionAnalysis(**tool_block["input"])
    except Exception as e:
        logger.warning(f"Anthropic call failed in backfill: {e}")
        return None


async def backfill_understanding_embeddings(
    batch_size: int = 25,
    max_images: int = 200,
    workspace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Re-run vision_analysis + Voyage on stale rows.

    Returns a summary dict — counts only, not the embeddings themselves
    (those go straight into VECS).
    """
    import os
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        return {"ok": False, "error": "ANTHROPIC_API_KEY not configured"}

    embeddings_svc = RealEmbeddingsService()
    vecs_svc = get_vecs_service()

    rows = await _fetch_stale_images(limit=max_images, workspace_id=workspace_id)
    if not rows:
        return {"ok": True, "scanned": 0, "reembedded": 0, "skipped": 0, "failed": 0}

    logger.info(f"🔄 Backfill: {len(rows)} stale image(s) found (workspace={workspace_id or 'all'})")

    reembedded = 0
    skipped = 0
    failed = 0

    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start:batch_start + batch_size]

        async def _process(row: Dict[str, Any]) -> str:
            image_bytes = await _fetch_image_bytes(row.get("image_url") or "")
            if not image_bytes:
                return "skipped"

            va = await _analyze_one(image_bytes, anthropic_api_key)
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
            return "reembedded" if ok else "failed"

        outcomes = await asyncio.gather(*[_process(r) for r in batch], return_exceptions=True)
        for o in outcomes:
            if isinstance(o, Exception):
                failed += 1
            elif o == "reembedded":
                reembedded += 1
            elif o == "skipped":
                skipped += 1
            elif o == "failed":
                failed += 1

    return {
        "ok": True,
        "scanned": len(rows),
        "reembedded": reembedded,
        "skipped": skipped,
        "failed": failed,
        "schema_version": SCHEMA_VERSION,
    }
