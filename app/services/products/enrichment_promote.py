"""Promote saved web-enrichment claims into a draft product, then rebuild what is derived from them."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

OPEN_STATUSES = ("extracted", "promoted")


class EnrichmentIO(Protocol):
    async def load_attempt(self, attempt_id: str) -> Optional[Dict[str, Any]]: ...
    async def load_product(self, product_id: str) -> Optional[Dict[str, Any]]: ...
    async def still_empty(self, product_id: str, workspace_id: str) -> List[str]: ...
    async def merge(self, product_id: str, patch: Dict[str, Any], source: Dict[str, Any]) -> List[str]: ...
    async def save_attempt(self, attempt_id: str, fields: Dict[str, Any]) -> None: ...
    async def canonicalize(self, product: Dict[str, Any]) -> bool: ...
    async def embed(self, product: Dict[str, Any]) -> bool: ...


def claims_to_patch(claims: List[Dict[str, Any]], still_empty: List[str]) -> Dict[str, Any]:
    """Only a field that is STILL empty may be written; a claim for a field filled since is dropped."""
    allowed = set(still_empty)
    return {c["field"]: c["value"] for c in claims if c.get("field") in allowed and c.get("value") is not None}


def finished_status(canonicalization: str, embedding: str) -> str:
    return "done" if canonicalization == "ok" and embedding == "ok" else "promoted"


async def promote_attempt(io: EnrichmentIO, attempt_id: str) -> Dict[str, Any]:
    """Resumable: an `extracted` attempt is promoted then derived; a `promoted` one only re-derives.

    Nothing here calls a model or a crawler, so a retry never spends again. The derived steps run
    even when nothing was merged now: a crash between the merge and its checkpoint leaves a retry
    with nothing left to merge and a stale vector, and it cannot tell that from "someone else filled it".
    """
    attempt = await io.load_attempt(attempt_id)
    if not attempt:
        return {"status": "not_found"}
    if attempt["status"] not in OPEN_STATUSES:
        return {"status": attempt["status"], "promoted_fields": attempt.get("promoted_fields") or []}

    product = await io.load_product(attempt["product_id"])
    if not product or product.get("workspace_id") != attempt["workspace_id"]:
        await io.save_attempt(attempt_id, {"status": "failed", "reason": "product not found in this workspace"})
        return {"status": "failed"}

    promoted: List[str] = list(attempt.get("promoted_fields") or [])
    if attempt["status"] == "extracted":
        if product.get("status") != "draft":
            await io.save_attempt(attempt_id, {"status": "failed", "reason": "product is no longer a draft"})
            return {"status": "failed"}
        patch = claims_to_patch(
            attempt.get("claims") or [], await io.still_empty(product["id"], product["workspace_id"]),
        )
        source = {"source": "web_enrichment", "url": attempt.get("source_url"), "confidence": attempt.get("confidence")}
        promoted = await io.merge(product["id"], patch, source) if patch else []
        await io.save_attempt(attempt_id, {"status": "promoted", "promoted_fields": promoted})
        product = await io.load_product(product["id"]) or product

    canon = "ok" if await _safely(io.canonicalize, product) else "degraded"
    embed = "ok" if await _safely(io.embed, product) else "failed"
    status = finished_status(canon, embed)
    await io.save_attempt(attempt_id, {
        "status": status, "canonicalization_state": canon, "embedding_state": embed,
        "reason": None if status == "done" else "a derived step failed; promotion is kept and a retry repeats only that step",
    })
    return {"status": status, "promoted_fields": promoted, "canonicalization_state": canon, "embedding_state": embed}


async def _safely(step: Any, product: Dict[str, Any]) -> bool:
    try:
        return bool(await step(product))
    except Exception as e:
        logger.error("enrichment derived step failed for %s: %s", product.get("id"), e)
        return False


class SupabaseIO:
    """The real thing: every write is a column the promotion owns, never a whole-metadata replace."""

    def __init__(self, supabase: Any) -> None:
        self.sb = supabase

    async def _run(self, query: Any) -> Any:
        return await asyncio.to_thread(query.execute)

    async def load_attempt(self, attempt_id: str) -> Optional[Dict[str, Any]]:
        r = await self._run(self.sb.client.table("product_enrichment_attempts").select("*").eq("id", attempt_id).limit(1))
        return (r.data or [None])[0]

    async def load_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        r = await self._run(
            self.sb.client.table("products")
            .select("id, name, description, workspace_id, source_document_id, metadata, status").eq("id", product_id).limit(1)
        )
        return (r.data or [None])[0]

    async def still_empty(self, product_id: str, workspace_id: str) -> List[str]:
        from app.services.products.enrichment_gaps import product_gaps

        rows = await product_gaps(self.sb, [product_id], workspace_id)
        return rows[0]["gaps"] if rows else []

    async def merge(self, product_id: str, patch: Dict[str, Any], source: Dict[str, Any]) -> List[str]:
        r = await self._run(self.sb.client.rpc(
            "merge_product_metadata_if_empty", {"p_product": product_id, "p_patch": patch, "p_source": source},
        ))
        return list(r.data or [])

    async def save_attempt(self, attempt_id: str, fields: Dict[str, Any]) -> None:
        await self._run(self.sb.client.table("product_enrichment_attempts").update(fields).eq("id", attempt_id))

    async def canonicalize(self, product: Dict[str, Any]) -> bool:
        from app.api.pdf_processing.stage_4_products import recanonicalize_product

        payload = await recanonicalize_product(
            self.sb, product["id"], dict(product.get("metadata") or {}), product["workspace_id"],
            "web_enrichment", logger,
        )
        if not payload:
            return False
        await self._run(self.sb.client.table("products").update(payload).eq("id", product["id"]))
        return True

    async def embed(self, product: Dict[str, Any]) -> bool:
        from app.api.pdf_processing.stage_4_products import _fetch_known_spec_fields, reembed_product_text

        fields = await _fetch_known_spec_fields(self.sb, logger)
        if not fields:
            raise RuntimeError("field registry unreadable; an embedding without spec fields would be a thin vector")
        return await reembed_product_text(
            self.sb, product.get("source_document_id"), product["id"], product.get("name"),
            product.get("description"), product.get("metadata") or {}, fields, logger,
        )
