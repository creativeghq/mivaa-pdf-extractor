"""The discovered product list an operator reviews before any product, chunk or embedding exists (#513).

Edits act on the cached catalogue dict (`background_jobs.metadata.catalog_cache.catalog`), which
Stage 0 reuses on resume, so what the operator approved is exactly what gets created.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ACTIONS = ("keep", "drop", "rename", "merge_into")
MAX_PRODUCT_LIMIT = 500


class ReviewError(ValueError):
    """An edit that names no discovered product, or contradicts another edit."""


def review_rows(products: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """What the operator sees per discovered product: enough to spot a duplicate, a merge or an accessory."""
    out = []
    for p in products:
        meta = p.get("metadata") or {}
        variants = meta.get("variants") or []
        out.append({
            "name": p.get("name"),
            "page_range": sorted({int(n) for n in (p.get("page_range") or []) if isinstance(n, (int, float))}),
            "category": meta.get("category"),
            "dimensions": (meta.get("dimensions") or [])[:12],
            "variants": len(variants) if isinstance(variants, list) else 0,
            "confidence": p.get("confidence"),
        })
    return out


def apply_review(catalog: Dict[str, Any], edits: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    """The catalogue with the edits applied, and one line per change. Raises ReviewError, applying nothing."""
    products = [dict(p) for p in (catalog.get("products") or [])]
    names = {(p.get("name") or "").strip().casefold(): p.get("name") for p in products}

    def key_of(name: Optional[str]) -> str:
        key = (name or "").strip().casefold()
        if key not in names:
            raise ReviewError(f"no discovered product is named {name!r}")
        return key

    drops, renames, merges = set(), {}, {}
    for e in edits:
        action = e.get("action")
        if action not in ACTIONS:
            raise ReviewError(f"unknown action {action!r}; use one of {', '.join(ACTIONS)}")
        key = key_of(e.get("name"))
        if action == "drop":
            drops.add(key)
        elif action == "rename":
            new = (e.get("new_name") or "").strip()
            if not new:
                raise ReviewError(f"rename of {e.get('name')!r} needs new_name")
            renames[key] = new
        elif action == "merge_into":
            into = key_of(e.get("into"))
            if into == key:
                raise ReviewError(f"{e.get('name')!r} cannot be merged into itself")
            merges[key] = into
    for src, dst in merges.items():
        if dst in drops or dst in merges:
            raise ReviewError(f"{names[src]!r} is merged into {names[dst]!r}, which is itself dropped or merged away")
        if src in drops:
            raise ReviewError(f"{names[src]!r} is both dropped and merged")

    by_key = {(p.get("name") or "").strip().casefold(): p for p in products}
    log: List[str] = []
    for src, dst in merges.items():
        target, source = by_key[dst], by_key[src]
        target["page_range"] = sorted({*(target.get("page_range") or []), *(source.get("page_range") or [])})
        meta = dict(target.get("metadata") or {})
        meta["merged_from"] = sorted({*(meta.get("merged_from") or []), source.get("name")})
        target["metadata"] = meta
        log.append(f"merged {source.get('name')!r} into {target.get('name')!r} (pages now {target['page_range']})")
    kept = []
    for p in products:
        key = (p.get("name") or "").strip().casefold()
        if key in merges:
            continue
        if key in drops:
            log.append(f"dropped {p.get('name')!r}")
            continue
        if key in renames:
            log.append(f"renamed {p.get('name')!r} to {renames[key]!r}")
            p["name"] = renames[key]
        kept.append(p)
    for key, new in renames.items():
        if sum(1 for p in kept if (p.get("name") or "").strip().casefold() == new.casefold()) > 1:
            raise ReviewError(f"renaming {names[key]!r} to {new!r} collides with another product; merge them instead")
    return {**catalog, "products": kept}, log


def product_limit(value: Any) -> Optional[int]:
    """A dry run's product cap: a positive integer up to MAX_PRODUCT_LIMIT, or None for no cap."""
    if value in (None, "", 0, "0"):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ReviewError(f"product_limit must be a whole number, got {value!r}")
    if not 1 <= n <= MAX_PRODUCT_LIMIT:
        raise ReviewError(f"product_limit must be between 1 and {MAX_PRODUCT_LIMIT}")
    return n


def notify_review_waiting(db: Any, job_id: str, document_id: str, workspace_id: Optional[str], discovered: int) -> int:
    """Tell the operators a job is parked: without this it waits silently until someone asks. Never raises."""
    try:
        from app.config import get_settings
        from app.services.flow_events import emit_flow_event

        settings = get_settings()
        role = db.table("roles").select("id").eq("name", "admin").execute().data or []
        if not role:
            logger.warning("review of job %s: no admin role, nobody was told", job_id)
            return 0
        admins = db.table("user_profiles").select("user_id").eq("role_id", role[0]["id"]).execute().data or []
        found = db.table("documents").select("filename").eq("id", document_id).limit(1).execute().data or []
        name = (found[0].get("filename") if found else None) or document_id
        sent = 0
        for admin in admins:
            sent += 1 if emit_flow_event("ingestion_review_waiting", {
                "workspace_id": workspace_id, "user_id": admin["user_id"],
                "title": f"{name} is waiting for your review",
                "body": f"Discovery found {discovered} product(s) and nothing has been created. Review it in the Catalogue Ingestion toolkit: job {job_id}.",
                "type": "ingestion_review", "action_url": "/agent-hub", "job_id": job_id, "document_id": document_id,
            }, settings.supabase_url, settings.supabase_service_role_key) else 0
        return sent
    except Exception as e:
        logger.error("could not announce the review of job %s: %s", job_id, e)
        return 0
