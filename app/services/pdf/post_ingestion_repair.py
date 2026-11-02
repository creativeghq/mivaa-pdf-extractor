"""After an ingestion, apply the fixes the audit's findings have a deterministic answer for.

Table chunks for products whose tables have none, the printed lines behind pages missing numbers,
one row per duplicated image, and printed-label links on shared pages. Every step is idempotent; a failed step is recorded, never raised.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PAGE = 1000
SUPPLEMENT_CHUNK_INDEX = 200000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db(supabase: Any) -> Any:
    return getattr(supabase, "client", supabase)


def _all(query_for, page: int = PAGE) -> List[Dict[str, Any]]:
    """Every row: PostgREST caps a response at 1000."""
    out: List[Dict[str, Any]] = []
    while True:
        rows = query_for().range(len(out), len(out) + page - 1).execute().data or []
        out.extend(rows)
        if len(rows) < page:
            return out


def plan_table_chunks(product_name: str, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    from app.services.chunking import table_chunks as tc
    chunks = tc.build_table_chunks(product_name, tables)
    for chunk in chunks:
        for cell in chunk["content"].split("\n")[1].split(" | "):
            if tc.is_price_header(cell):
                raise RuntimeError(f"price-like column {cell!r} survived into a chunk; refusing")
    codes: set = set()
    for chunk in chunks:
        codes |= tc.item_codes(chunk["content"])
    return {"chunks": chunks, "tables_used": len({c["metadata"]["table_id"] for c in chunks}), "codes": codes}


async def rebuild_table_chunks(supabase: Any, document_id: str, apply: bool = True) -> Dict[str, Any]:
    from app.services.chunking import table_chunks as tc
    db = _db(supabase)
    products = await asyncio.to_thread(lambda: db.table("products").select("id, name, workspace_id")
                                       .eq("source_document_id", document_id).execute().data or [])
    summary: Dict[str, Any] = {"products": len(products), "chunks": 0, "written": 0, "embedded": 0, "failed": [], "per_product": []}
    for product in products:
        tables = await asyncio.to_thread(lambda: db.table("product_tables").select("*").eq("product_id", product["id"]).execute().data or [])
        plan = plan_table_chunks(product["name"], tables)
        summary["chunks"] += len(plan["chunks"])
        summary["per_product"].append({"product": product["name"], "tables": len(tables), "used": plan["tables_used"],
                                       "chunks": len(plan["chunks"]), "codes": len(plan["codes"])})
        if apply and plan["chunks"]:
            ids = await tc.write_table_chunks(supabase, product["workspace_id"], document_id, product["id"], plan["chunks"])
            result = await tc.embed_chunks_once(supabase, ids, workspace_id=product["workspace_id"])
            summary["written"] += len(ids)
            summary["embedded"] += result["embedded"]
            summary["failed"].extend(result["failed"])
    return summary


async def write_page_supplements(supabase: Any, document: Dict[str, Any], collected: Dict[str, Any], pdf: bytes,
                                 findings: List[Dict[str, Any]], apply: bool = True) -> Dict[str, Any]:
    """One chunk per page holding the printed lines whose numbers the layout pass dropped."""
    from app.services.chunking.table_chunks import embed_chunks_once
    from app.services.pdf.ingestion_audit import numeric_tokens
    from app.services.pdf.ingestion_audit_runner import source_pages
    from app.services.pdf.page_supplements import SOURCE, build_supplement, owner_for, supplement_text
    from app.services.pdf.text_table_structurer import lines_from_words

    db = _db(supabase)
    gaps = [g for f in findings if f["code"] in ("numbers_missing", "numbers_missing_few") for g in f["examples"]]
    pages = await asyncio.to_thread(source_pages, pdf, collected["layout"])
    label = document["filename"].rsplit(".", 1)[0]
    planned = []
    for gap in sorted(gaps, key=lambda g: g["page"]):
        page = pages.get(gap["page"])
        if not page:
            continue
        text = supplement_text(lines_from_words(page["words"]), gap["all"], numeric_tokens)
        chunk = build_supplement(label, gap["page"], text, gap["all"])
        if chunk:
            planned.append({"page": gap["page"], "missing": gap["missing"], "owner": owner_for(gap["page"], collected["products"]),
                            "chunk": chunk, "lines": text})
    summary: Dict[str, Any] = {"pages": len(planned), "planned": planned if not apply else None, "written": 0, "replaced": 0,
                               "embedded": 0, "failed": []}
    if not apply:
        return summary
    old = [r["id"] for r in db.table("document_chunks").select("id").eq("document_id", document["id"]).eq("metadata->>source", SOURCE).execute().data or []]
    rows = [{
        "document_id": document["id"], "workspace_id": document["workspace_id"], "content": p["chunk"]["content"],
        "chunk_index": SUPPLEMENT_CHUNK_INDEX + p["page"], "metadata": p["chunk"]["metadata"], "product_id": p["owner"],
        "chunk_type": "technical_specs", "chunk_type_status": "classified", "has_text_embedding": False,
    } for p in planned]
    ids = [r["id"] for r in (db.table("document_chunks").insert(rows).execute().data or [])] if rows else []
    if old:
        db.table("document_chunks").delete().in_("id", old).execute()
    out = await embed_chunks_once(db, ids, workspace_id=document["workspace_id"]) if ids else {"embedded": 0, "failed": []}
    summary.update(written=len(ids), replaced=len(old), embedded=out["embedded"], failed=out["failed"])
    return summary


def duplicate_image_groups(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Live rows that are one cut: same page and perceptual hash and size, or same page, layer and position.

    A page shared by several products is cropped once per product, so it lands here once each, and
    a crop with no hash is still caught by its position. The survivor is the embedded one, then the oldest.
    """
    parent: Dict[str, str] = {}

    def root(i: str) -> str:
        while parent.setdefault(i, i) != i:
            i = parent[i]
        return i

    keyed: Dict[tuple, str] = {}
    live = [r for r in rows if "excluded_from_products" not in (r.get("metadata") or {})]
    for r in live:
        m = r.get("metadata") or {}
        phash = (m.get("layer_info") or {}).get("perceptual_hash")
        bbox = r.get("bbox")
        keys = []
        if phash:
            keys.append(("hash", r.get("page_number"), phash, m.get("width"), m.get("height")))
        if isinstance(bbox, list) and len(bbox) == 4:
            keys.append(("pos", r.get("page_number"), r.get("extraction_layer"), *(round(float(v), 3) for v in bbox)))
        for k in keys:
            if k in keyed:
                parent[root(r["id"])] = root(keyed[k])
            else:
                keyed[k] = r["id"]
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in live:
        groups[root(r["id"])].append(r)
    return [sorted(g, key=lambda r: (not r.get("has_slig_embedding"), r.get("created_at") or ""))
            for g in groups.values() if len(g) > 1]


async def dedupe_document_images(supabase: Any, document_id: str, apply: bool = True) -> Dict[str, Any]:
    from app.services.images.association_scores import PAGE_LINK_SCORE
    from app.services.embeddings.vecs_service import get_vecs_service
    db = _db(supabase)
    rows = await asyncio.to_thread(_all, lambda: db.table("document_images")
                                   .select("id, page_number, bbox, extraction_layer, metadata, has_slig_embedding, created_at")
                                   .eq("document_id", document_id).order("id"))
    groups = duplicate_image_groups(rows)
    summary = {"groups": len(groups), "retired": sum(len(g) - 1 for g in groups), "links_moved": 0}
    if not apply:
        return summary
    for survivor, *losers in groups:
        loser_ids = [r["id"] for r in losers]
        links = db.table("image_product_associations").select("workspace_id, product_id, image_id, overall_score, metadata") \
            .in_("image_id", loser_ids).execute().data or []
        moved = {(l["product_id"]): l for l in links}
        if moved:
            db.table("image_product_associations").upsert(
                [{"workspace_id": l["workspace_id"], "product_id": pid, "image_id": survivor["id"],
                  "overall_score": l.get("overall_score") or PAGE_LINK_SCORE,
                  "metadata": {**(l.get("metadata") or {}), "deduplicated_from": l["image_id"]}} for pid, l in moved.items()],
                on_conflict="image_id,product_id").execute()
            db.table("image_product_associations").delete().in_("image_id", loser_ids).execute()
        summary["links_moved"] += len(moved)
        embedded = [r["id"] for r in losers if r.get("has_slig_embedding")]
        if embedded:
            await get_vecs_service().delete_embeddings_by_image_ids(embedded)
        for r in losers:
            m = dict(r.get("metadata") or {})
            m["excluded_from_products"] = {"at": _now(), "reason": "duplicate_image", "duplicate_of": survivor["id"]}
            db.table("document_images").update({"metadata": m}).eq("id", r["id"]).execute()
    return summary


async def repair_document(document_id: str, supabase: Any, log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    from app.services.pdf.ingestion_audit_runner import _collect, _download, build_audit

    log = log or logger
    db = _db(supabase)
    out: Dict[str, Any] = {}
    document = (db.table("documents").select("id, workspace_id, filename, storage_bucket, storage_object_path, content_type")
                .eq("id", document_id).limit(1).execute().data or [None])[0]
    if not document:
        return {"error": "document not found"}

    async def audit() -> tuple:
        collected = await _collect(db, document)
        pdf = await _download(db, document)
        result = (await asyncio.to_thread(build_audit, document, collected, pdf, "post_embed", None)).as_dict()
        return collected, pdf, {f["code"] for f in result["findings"]}, result["findings"]

    async def step(name: str, coro) -> None:
        try:
            out[name] = {k: v for k, v in (await coro).items() if k not in ("per_product", "planned")}
        except Exception as e:
            log.warning(f"⚠️ Post-ingestion repair step {name} failed: {e}")
            out[name] = {"error": f"{type(e).__name__}: {e}"[:300]}

    try:
        collected, pdf, codes, findings = await audit()
    except Exception as e:
        return {"error": f"audit failed: {type(e).__name__}: {e}"[:300]}
    if codes & {"tables_without_chunks"}:
        await step("table_chunks", rebuild_table_chunks(supabase, document_id))
        try:
            collected, pdf, codes, findings = await audit()
        except Exception as e:
            out["re_audit_error"] = str(e)[:300]
    if pdf is not None and codes & {"numbers_missing", "numbers_missing_few"}:
        await step("page_supplements", write_page_supplements(supabase, document, collected, pdf, findings))
    await step("duplicate_images", dedupe_document_images(supabase, document_id))
    if pdf is not None:
        from app.services.images.label_linking import relink_by_labels
        await step("label_links", relink_by_labels(supabase, document_id, pdf))
    log.info(f"   Post-ingestion repair: {', '.join(f'{k}={v}' for k, v in out.items())[:600]}")
    return out
