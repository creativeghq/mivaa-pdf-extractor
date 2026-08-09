"""
Admin facet canonicalization API.

Used by:
  - canonicalize-attributes edge function (proxy for background agents
    invoking the canonicalizer through service-role auth)
  - admin UI / observability for inspecting canonical values + the merge log

The actual canonicalization logic lives in
app.services.facets.facet_canonicalizer; this module is a thin HTTP surface.

Auth: pattern follows admin_linking.py (no explicit auth in this layer — the
edge function proxy enforces JWT before forwarding, and reverse-proxy ACLs gate
the /api/admin/* prefix at the platform boundary).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from app.services.core.supabase_client import get_supabase_client
from app.services.facets import canonicalize_product_attributes
from app.dependencies import require_admin, get_current_user, resolve_workspace_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin/facets", tags=["Admin - Facets"])


# ─────────────────────────────────────────────────────────────────────────────
# Canonicalize endpoint (proxied by edge function for background agents)
# ─────────────────────────────────────────────────────────────────────────────

class CanonicalizeRequest(BaseModel):
    raw_attributes: Dict[str, Any] = Field(
        ...,
        description="Source metadata dict from the agent / promoter. Only whitelisted facets are canonicalized; everything else passes through.",
    )
    source: str = Field(
        ...,
        description="Origin tag for facet_merge_log (e.g. 'agent_material_tagger', 'agent_product_enrichment', 'catalog_extract_promote').",
        max_length=64,
    )
    product_id: Optional[str] = Field(
        None,
        description="If provided, enables diff-before-canonicalize so re-tags skip already-seen values and attributes_raw merges across runs.",
    )
    workspace_id: Optional[str] = Field(
        None,
        description="Workspace to scope canonicalization to. Required for tenant isolation; omitted => canonicalizer fails closed (workspace_required).",
    )


class CanonicalizeResponse(BaseModel):
    attributes: Dict[str, Any]
    attributes_raw: Dict[str, List[str]]
    resolutions_count: int
    actions_by_type: Dict[str, int]


@router.post("/canonicalize", response_model=CanonicalizeResponse)
async def canonicalize(req: CanonicalizeRequest, current_user: dict = Depends(get_current_user)) -> CanonicalizeResponse:
    # Bind the caller-supplied workspace to the authenticated identity (invariant 1).
    workspace_id = await resolve_workspace_id(current_user, req.workspace_id)
    supabase = get_supabase_client()
    try:
        result = await canonicalize_product_attributes(
            supabase,
            req.raw_attributes,
            source=req.source,
            product_id=req.product_id,  # uuid string OK; service stringifies
            workspace_id=workspace_id,
        )
    except Exception as e:
        logger.exception("canonicalize endpoint failed")
        raise HTTPException(status_code=500, detail=f"canonicalize failed: {e}")

    actions: Dict[str, int] = {}
    for r in result.resolutions:
        actions[r.action] = actions.get(r.action, 0) + 1

    return CanonicalizeResponse(
        attributes=result.attributes,
        attributes_raw=result.attributes_raw,
        resolutions_count=len(result.resolutions),
        actions_by_type=actions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bulk re-canonicalization sweep (#316)
# ─────────────────────────────────────────────────────────────────────────────

class RecanonicalizeRequest(BaseModel):
    """Bounded replay of products.attributes_raw through the canonicalizer.

    Without this, a threshold change, a curated alias from /lock, or a golden-set
    correction could only reach products by re-ingesting every source PDF — and
    outage-degraded products (empty `attributes`, flagged in
    metadata.facet_canonicalization, never retried) stayed facet-less forever.
    """
    target_version: Optional[int] = Field(
        None,
        description="Sweep products below this facet_canonicalization_version. Defaults to the "
                    "code's CURRENT_FACET_CANONICALIZATION_VERSION; raise that constant to "
                    "re-sweep the whole catalog after a rules change.",
    )
    max_products: int = Field(200, ge=1, le=5000, description="Hard cap for one call. The response's `remaining` says whether to call again.")
    batch_size: int = Field(25, ge=1, le=200)
    workspace_id: Optional[str] = Field(None, description="Scope to one tenant. Omit to sweep all.")
    degraded_only: bool = Field(
        False,
        description="Only replay products flagged degraded by a failed canonicalization run. "
                    "Use for outage recovery; the default also re-sweeps rules-stale products.",
    )


@router.post("/recanonicalize", dependencies=[Depends(require_admin)])
async def recanonicalize(req: RecanonicalizeRequest, current_user: dict = Depends(get_current_user)):
    """Replay attributes_raw → canonical attributes for stale/degraded products.

    Idempotent and resumable: `products.facet_canonicalization_version` is the cursor, so a
    run that dies halfway resumes where it stopped and a completed run is a no-op.

    A DEGRADED result never bumps the version — the product stays eligible and is picked
    first next time. Bumping it would turn a retryable outage into a permanent empty-facet
    product while reporting success.
    """
    # Bind the caller-supplied workspace to the authenticated identity (invariant 1).
    workspace_id = await resolve_workspace_id(current_user, req.workspace_id)
    from app.services.facets.facet_recanonicalization import (
        recanonicalize_products,
        CURRENT_FACET_CANONICALIZATION_VERSION,
    )

    try:
        summary = await recanonicalize_products(
            target_version=req.target_version or CURRENT_FACET_CANONICALIZATION_VERSION,
            max_products=req.max_products,
            batch_size=req.batch_size,
            workspace_id=workspace_id,
            degraded_only=req.degraded_only,
        )
    except Exception as e:
        logger.exception("recanonicalize sweep failed")
        raise HTTPException(status_code=500, detail=f"recanonicalize failed: {e}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Observability — list canonicals (drift watch + dashboard)
# ─────────────────────────────────────────────────────────────────────────────

class CanonicalRow(BaseModel):
    facet_key: str
    canonical_value: str
    aliases: List[str]
    alias_count: int
    embedding_model: Optional[str]
    is_locked: bool
    first_seen_at: Optional[str]
    last_seen_at: Optional[str]


@router.get("/canonicals", response_model=List[CanonicalRow], dependencies=[Depends(require_admin)])
async def list_canonicals(
    facet_key: Optional[str] = Query(None, description="Filter by facet_key (color, material, finish, ...)"),
    limit: int = Query(200, ge=1, le=1000),
) -> List[CanonicalRow]:
    supabase = get_supabase_client()
    try:
        q = supabase.client.table('facet_canonical_values').select(
            'facet_key, canonical_value, aliases, alias_count, embedding_model, is_locked, first_seen_at, last_seen_at'
        )
        if facet_key:
            q = q.eq('facet_key', facet_key)
        resp = q.order('alias_count', desc=True).limit(limit).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"canonicals query failed: {e}")

    rows: List[CanonicalRow] = []
    for r in (resp.data or []):
        rows.append(CanonicalRow(
            facet_key=r['facet_key'],
            canonical_value=r['canonical_value'],
            aliases=list(r.get('aliases') or []),
            alias_count=int(r.get('alias_count') or 0),
            embedding_model=r.get('embedding_model'),
            is_locked=bool(r.get('is_locked')),
            first_seen_at=r.get('first_seen_at'),
            last_seen_at=r.get('last_seen_at'),
        ))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Observability — merge log (false-merge / drift detection)
# ─────────────────────────────────────────────────────────────────────────────

class MergeLogRow(BaseModel):
    id: int
    facet_key: str
    raw_value: str
    normalized_value: str
    resolved_canonical: str
    action: str
    similarity: Optional[float]
    source: Optional[str]
    product_id: Optional[str]
    occurred_at: str


@router.get("/merge-log", response_model=List[MergeLogRow], dependencies=[Depends(require_admin)])
async def merge_log(
    facet_key: Optional[str] = Query(None),
    action: Optional[str] = Query(None, description="Filter to one of: exact_alias, embedding_merge, new, rejected_non_english"),
    source: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
) -> List[MergeLogRow]:
    """Recent canonicalization events. Use ?action=embedding_merge to surface
    cosine-merged values for false-merge review. Use ?action=rejected_non_english
    to see values the RPC refused (pretranslate failure or no Anthropic key)."""
    supabase = get_supabase_client()
    try:
        q = supabase.client.table('facet_merge_log').select(
            'id, facet_key, raw_value, normalized_value, resolved_canonical, action, similarity, source, product_id, occurred_at'
        )
        if facet_key:
            q = q.eq('facet_key', facet_key)
        if action:
            q = q.eq('action', action)
        if source:
            q = q.eq('source', source)
        resp = q.order('occurred_at', desc=True).limit(limit).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"merge-log query failed: {e}")

    rows: List[MergeLogRow] = []
    for r in (resp.data or []):
        rows.append(MergeLogRow(
            id=int(r['id']),
            facet_key=r['facet_key'],
            raw_value=r['raw_value'],
            normalized_value=r['normalized_value'],
            resolved_canonical=r['resolved_canonical'],
            action=r['action'],
            similarity=r.get('similarity'),
            source=r.get('source'),
            product_id=r.get('product_id'),
            occurred_at=r['occurred_at'],
        ))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Lock / unlock — admin override to prevent embedding_merge into a canonical
# ─────────────────────────────────────────────────────────────────────────────

class LockRequest(BaseModel):
    facet_key: str
    canonical_value: str
    is_locked: bool


@router.post("/lock", dependencies=[Depends(require_admin)])
async def set_lock(req: LockRequest):
    supabase = get_supabase_client()
    try:
        resp = supabase.client.table('facet_canonical_values') \
            .update({'is_locked': req.is_locked}) \
            .eq('facet_key', req.facet_key) \
            .eq('canonical_value', req.canonical_value) \
            .execute()
        if not resp.data:
            raise HTTPException(status_code=404, detail="canonical not found")
        return {"facet_key": req.facet_key, "canonical_value": req.canonical_value, "is_locked": req.is_locked}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"lock toggle failed: {e}")
