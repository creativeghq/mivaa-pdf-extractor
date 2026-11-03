from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.services.flow_events import emit_flow_event
from app.services.pdf.ingestion_audit import AuditResult, audit_document, db_status, platform_numbers_for_page, unsettled_images
from app.services.pdf.catalogue_tables import plan_catalogue_tables
from app.services.pdf.ocr_completeness import numeric_tokens
from app.services.pdf.text_layer_extractor import assess_text_layer, reference_numbers

STAGE = "ingestion_audit"
CODE_VERSION = "ingestion-audit-1"
PAGE_BATCH = 200
ID_BATCH = 100
NOTIFY_STATUSES = {"failed", "needs_review", "unverifiable", "audit_error"}


def _rows(response: Any) -> List[Dict[str, Any]]:
    return list(getattr(response, "data", None) or [])


def _batches(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _element_text(elements: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for el in elements or []:
        parts.append(el.get("text_content") or "")
        parts.append((el.get("metadata") or {}).get("html") or "")
    return "\n".join(parts)


def _table_text(table: Dict[str, Any]) -> str:
    data = table.get("table_data") or {}
    cells = [str(c) for row in (data.get("rows") or []) for c in (row or [])]
    return " | ".join([str(h) for h in (table.get("headers") or [])] + cells)


def source_pages(pdf_bytes: bytes, layout_rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    import fitz

    out: Dict[int, Dict[str, Any]] = {}
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(pdf_bytes)
        path = handle.name
    try:
        doc = fitz.open(path)
        for row in layout_rows:
            meta = row.get("analysis_metadata") or {}
            idx = meta.get("pdf_idx", int(row["page_number"]) - 1)
            if not isinstance(idx, int) or idx < 0 or idx >= len(doc):
                continue
            page = doc[idx]
            position = meta.get("position")
            clip = None
            if position in ("left", "right"):
                half = page.rect.width / 2.0
                clip = fitz.Rect(0, 0, half, page.rect.height) if position == "left" else fitz.Rect(half, 0, page.rect.width, page.rect.height)
            words = [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words", clip=clip)]
            verdict = assess_text_layer(words, 0.0)
            out[int(row["page_number"])] = {
                "words": words,
                "usable": verdict.usable or len(words) >= 25,
                "reference": reference_numbers(words) if (verdict.usable or len(words) >= 25) else set(),
            }
        doc.close()
    finally:
        os.unlink(path)
    return out


async def _collect(db: Any, document: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = document["id"]
    products = _rows(await asyncio.to_thread(db.table("products").select("id, name, metadata, description, attributes").eq("source_document_id", doc_id).execute))
    product_ids = [p["id"] for p in products]
    layout: List[Dict[str, Any]] = []
    start = 0
    while True:
        batch = _rows(await asyncio.to_thread(
            db.table("document_layout_analysis").select("page_number, layout_elements, analysis_metadata")
            .eq("document_id", doc_id).order("page_number").range(start, start + PAGE_BATCH - 1).execute
        ))
        layout.extend(batch)
        if len(batch) < PAGE_BATCH:
            break
        start += PAGE_BATCH
    chunks = _rows(await asyncio.to_thread(
        db.table("document_chunks").select("id, product_id, content, metadata, has_text_embedding").eq("document_id", doc_id).execute
    ))
    snapshots = _rows(await asyncio.to_thread(
        db.table("source_snapshots").select("id, url, storage_bucket, storage_object_path").eq("document_id", doc_id).execute
    ))
    tables: List[Dict[str, Any]] = []
    embedded: set = set()
    images: Dict[str, int] = {}
    names = {p["id"]: p.get("name") or p["id"] for p in products}
    product_images: Dict[str, List[Dict[str, Any]]] = {}
    for ids in _batches(product_ids, ID_BATCH):
        tables.extend(_rows(await asyncio.to_thread(
            db.table("product_tables").select("id, product_id, page_number, headers, table_data, extractor, metadata").in_("product_id", ids).execute
        )))
        embedded |= {r["id"] for r in _rows(await asyncio.to_thread(
            db.table("products").select("id").in_("id", ids).not_.is_("text_embedding_1024", "null").execute
        ))}
        for r in _rows(await asyncio.to_thread(
            db.table("image_product_associations")
            .select("product_id, document_images!inner(id, document_id, page_number, bbox, extraction_layer, has_slig_embedding, metadata)")
            .in_("product_id", ids).execute
        )):
            images[r["product_id"]] = images.get(r["product_id"], 0) + 1
            product_images.setdefault(names.get(r["product_id"], r["product_id"]), []).append(r["document_images"])
    sources = sorted({(i.get("metadata") or {}).get("restored_from") for imgs in product_images.values() for i in imgs
                      if i.get("extraction_layer") == "restored" and not (i.get("metadata") or {}).get("source_native_px")} - {None})
    source_px: Dict[str, Any] = {}
    for ids in _batches(sources, ID_BATCH):
        for r in _rows(await asyncio.to_thread(db.table("document_images").select("id, metadata").in_("id", ids).execute)):
            source_px[r["id"]] = ((r.get("metadata") or {}).get("image_quality") or {}).get("native_px")
    for imgs in product_images.values():
        for i in imgs:
            if (i.get("metadata") or {}).get("restored_from") in source_px:
                i["source_px"] = source_px[i["metadata"]["restored_from"]]
    return {"products": products, "layout": layout, "chunks": chunks, "tables": tables, "embedded": embedded, "images": images,
            "product_images": product_images, "snapshots": snapshots, "unsettled_images": await _unsettled_images(db, doc_id)}


async def _unsettled_images(db: Any, doc_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    while True:
        batch = _rows(await asyncio.to_thread(
            db.table("document_images").select("id, page_number, image_type, metadata").eq("document_id", doc_id).order("id")
            .range(len(rows), len(rows) + PAGE_BATCH - 1).execute
        ))
        rows.extend(batch)
        if len(batch) < PAGE_BATCH:
            break
    linked: set = set()
    for ids in _batches([r["id"] for r in rows], ID_BATCH):
        linked |= {r["image_id"] for r in _rows(await asyncio.to_thread(
            db.table("image_product_associations").select("image_id").in_("image_id", ids).execute))}
    return unsettled_images(rows, linked)


async def _download(db: Any, document: Dict[str, Any]) -> Optional[bytes]:
    bucket, path = document.get("storage_bucket"), document.get("storage_object_path")
    if not bucket or not path:
        return None
    try:
        return await asyncio.to_thread(db.storage.from_(bucket).download, path)
    except Exception as exc:
        logging.getLogger(__name__).warning("audit: could not download %s/%s: %s", bucket, path, exc)
        return None


async def _snapshot_texts(db: Any, snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from app.services.pdf.source_snapshots import parse_page

    texts: List[Dict[str, Any]] = []
    for snap in snapshots:
        try:
            body = await asyncio.to_thread(db.storage.from_(snap["storage_bucket"]).download, snap["storage_object_path"])
            texts.append({"url": snap.get("url"), "text": parse_page(body.decode("utf-8", errors="replace"))["text"]})
        except Exception as exc:
            logging.getLogger(__name__).warning("audit: could not read snapshot %s: %s", snap.get("url"), exc)
    return texts


def series_coverage(collected: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per product: maker-photographed variants it does not offer, and its own pages outside its range."""
    from app.services.discovery import series_reconcile as sr
    from app.services.images.label_linking import shares_series, tokens_of

    pages = {int(r["page_number"]): _element_text(r.get("layout_elements") or []) for r in collected["layout"]}
    names = [p["name"] for p in collected["products"]]
    site_names: Dict[str, List[str]] = {}
    for p in collected["products"]:
        urls = [(i.get("metadata") or {}).get("source_url") for i in (collected.get("product_images") or {}).get(p["name"], [])
                if i.get("extraction_layer") == "manufacturer_original" and "excluded_from_products" not in (i.get("metadata") or {})]
        site_names[p["name"]] = sr.site_variant_names([u for u in urls if u], p["name"])
    # A word in two or more series' photo names is the maker's collection or language word, not a variant.
    spread: Dict[str, int] = {}
    for variants in site_names.values():
        for word in {w for v in variants for w in v.split()}:
            spread[word] = spread.get(word, 0) + 1
    common = {w for w, n in spread.items() if n >= 2}
    out = []
    for p in collected["products"]:
        m, a = p.get("metadata") or {}, p.get("attributes") or {}
        offered = set()
        for v in (a.get("color"), m.get("color"), m.get("decorative_variant"), a.get("finish"), m.get("finish")):
            for x in (v if isinstance(v, list) else [v]):
                if isinstance(x, str):
                    offered |= set(sr.fold(x).split())
        missing = [v for v in site_names[p["name"]] if not set(v.split()) & offered and set(v.split()) - common]
        own = set(sr.series_pages(pages, p["name"], names))
        # A page in the range of a product sharing the printed name ("Alchemy 30x90") is that product's, not outside.
        ranged = [q for q in collected["products"] if q is p or shares_series(tokens_of(p["name"]), tokens_of(q["name"]))]
        outside = sorted(own - {n for q in ranged for n in ((q.get("metadata") or {}).get("page_range") or []) if isinstance(n, int)})
        out.append({"product": p["name"], "missing_variants": missing, "pages_outside": outside,
                    "site_status": (m.get("site_manifest") or {}).get("status")})
    return out


def build_audit(document: Dict[str, Any], collected: Dict[str, Any], pdf_bytes: Optional[bytes], phase: str, web_texts: Optional[List[Dict[str, Any]]] = None) -> AuditResult:
    layout = collected["layout"]
    chunks, tables = collected["chunks"], collected["tables"]
    pages_source = source_pages(pdf_bytes, layout) if pdf_bytes else {}
    tables_by_page: Dict[int, List[str]] = {}
    for t in tables:
        tables_by_page.setdefault(int(t.get("page_number") or 0), []).append(_table_text(t))
    chunks_by_page: Dict[int, List[str]] = {}
    for c in chunks:
        meta = c.get("metadata") or {}
        numbers = [n for n in ([meta.get("page_number")] + list(meta.get("page_numbers") or [])) if isinstance(n, int)]
        for page in set(numbers):
            chunks_by_page.setdefault(page, []).append(c.get("content") or "")
    numbers_by_product: Dict[str, set] = {}
    for t in tables:
        numbers_by_product.setdefault(t["product_id"], set()).update(numeric_tokens(_table_text(t)))
    for c in chunks:
        if c.get("product_id"):
            numbers_by_product.setdefault(c["product_id"], set()).update(numeric_tokens(c.get("content") or ""))
    owners: Dict[int, List[str]] = {}
    for p in collected["products"]:
        for n in ((p.get("metadata") or {}).get("page_range") or []):
            if isinstance(n, int):
                owners.setdefault(n, []).append(p["id"])
    unowned_numbers: set = set()
    for c in chunks:
        if not c.get("product_id"):
            unowned_numbers |= numeric_tokens(c.get("content") or "")
    for t in tables:
        if (t.get("metadata") or {}).get("scope") == "catalogue_wide":
            unowned_numbers |= numeric_tokens(_table_text(t))
    pages: List[Dict[str, Any]] = []
    for row in layout:
        number = int(row["page_number"])
        source = pages_source.get(number) or {}
        meta = row.get("analysis_metadata") or {}
        pages.append({
            "page": number,
            "route": meta.get("route"),
            "reference_numbers": source.get("reference") or set(),
            "platform_numbers": platform_numbers_for_page(_element_text(row.get("layout_elements") or []), chunks_by_page.get(number, []), tables_by_page.get(number, []))
            | set().union(*(numbers_by_product.get(pid, set()) for pid in owners.get(number, [])))
            | (unowned_numbers if number not in owners else set()),
        })
    pdf_text = " ".join(w[4] for p in pages_source.values() for w in p["words"]) if pages_source else None
    web_mode = pdf_bytes is None and bool(web_texts)
    web: Optional[Dict[str, Any]] = None
    platform_text = "\n".join(
        [c.get("content") or "" for c in chunks] + [_table_text(t) for t in tables] + [_element_text(r.get("layout_elements") or []) for r in layout]
    )
    per_product_tables: Dict[str, int] = {}
    for t in tables:
        per_product_tables[t["product_id"]] = per_product_tables.get(t["product_id"], 0) + 1
    per_product_chunks: Dict[str, int] = {}
    for c in chunks:
        if c.get("product_id"):
            per_product_chunks[c["product_id"]] = per_product_chunks.get(c["product_id"], 0) + 1
    product_rows = [
        {
            "name": p["name"],
            "description": p.get("description"),
            "attributes": p.get("attributes"),
            "n_tables": per_product_tables.get(p["id"], 0),
            "n_chunks": per_product_chunks.get(p["id"], 0),
            "has_embedding": p["id"] in collected["embedded"],
            "n_images": collected["images"].get(p["id"], 0),
        }
        for p in collected["products"]
    ]
    if web_mode:
        pdf_text = "\n".join(t["text"] for t in (web_texts or []))
        pages = [{"page": 1, "route": None, "reference_numbers": set(), "platform_numbers": set()}]
        table_numbers_by_product: Dict[str, set] = {}
        for t in tables:
            table_numbers_by_product.setdefault(t["product_id"], set()).update(
                n for row in ((t.get("table_data") or {}).get("rows") or []) for cell in (row or []) for n in numeric_tokens(str(cell)))
        web = {"snapshots": web_texts, "products": [{"name": p["name"], "table_numbers": table_numbers_by_product.get(p["id"], set())} for p in collected["products"]]}
    plans, unmatched = plan_catalogue_tables(layout, collected["products"])
    return audit_document(
        catalogue={"plans": plans, "unmatched": unmatched},
        web=web if web_mode else None,
        source_kept=pdf_bytes is not None or web_mode,
        source_state="missing_object" if (pdf_bytes is None and not web_mode and document.get("storage_object_path")) else "no_path",
        source_kind="web" if web_mode else "pdf",
        pages=pages,
        tables=tables,
        chunks=chunks,
        products=product_rows,
        pdf_text=pdf_text,
        platform_text=platform_text,
        phase=phase,
        product_images=collected.get("product_images"),
        other_series=collected.get("other_series"),
        unsettled_images=collected.get("unsettled_images"),
        series_coverage=series_coverage(collected) if pdf_bytes else None,
    )


def _notify(settings: Any, db: Any, document: Dict[str, Any], status: str, result: Dict[str, Any], audit_id: Optional[str], job_id: Optional[str]) -> int:
    admin_role = _rows(db.table("roles").select("id").eq("name", "admin").execute())
    if not admin_role:
        logging.getLogger(__name__).warning("audit: no admin role found, nobody was notified about %s", document.get("filename"))
        return 0
    admins = _rows(db.table("user_profiles").select("user_id").eq("role_id", admin_role[0]["id"]).execute())
    codes = ", ".join(f["code"] for f in result["findings"][:6]) or "no detail"
    title = f"Ingestion audit: {document.get('filename')} is {status.replace('_', ' ')}"
    body = f"{len(result['findings'])} finding(s): {codes}. Open Operations > Catalog to review."
    sent = 0
    for admin in admins:
        ok = emit_flow_event(
            "ingestion_audit_failed",
            {
                "workspace_id": document.get("workspace_id"),
                "user_id": admin["user_id"],
                "title": title,
                "body": body,
                "type": "ingestion_audit",
                "action_url": "/admin/operations?tab=catalog",
                "document_id": document["id"],
                "job_id": job_id,
                "audit_id": audit_id,
                "status": status,
            },
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
        sent += 1 if ok else 0
    return sent


async def run_ingestion_audit(
    document_id: str,
    supabase: Any,
    settings: Any,
    *,
    job_id: Optional[str] = None,
    phase: str = "post_embed",
    trigger: str = "pipeline",
    persist: bool = True,
    notify: bool = True,
) -> Dict[str, Any]:
    log = logging.getLogger(__name__)
    db = supabase.client
    document: Dict[str, Any] = {"id": document_id}
    try:
        found = _rows(await asyncio.to_thread(
            db.table("documents").select("id, workspace_id, filename, storage_bucket, storage_object_path, content_type").eq("id", document_id).execute
        ))
        if not found:
            raise LookupError(f"document {document_id} not found")
        document = found[0]
        collected = await _collect(db, document)
        pdf_bytes = await _download(db, document)
        if pdf_bytes is not None and phase != "pre_embed":
            from app.services.images.label_linking import other_series_images
            collected["other_series"] = await other_series_images(supabase, document_id, pdf_bytes)
        web_texts = await _snapshot_texts(db, collected["snapshots"]) if pdf_bytes is None and collected["snapshots"] else None
        result = (await asyncio.to_thread(build_audit, document, collected, pdf_bytes, phase, web_texts)).as_dict()
        status = db_status(result["status"])
    except Exception as exc:
        log.warning("audit of %s crashed: %s", document_id, exc, exc_info=True)
        result = {"status": "audit_error", "metrics": {"error": f"{type(exc).__name__}: {exc}"[:400]}, "findings": [
            {"severity": "fail", "code": "audit_crashed", "message": f"The audit itself failed: {type(exc).__name__}: {exc}"[:400], "count": 0, "examples": []}]}
        status = "audit_error"
    audit_id: Optional[str] = None
    if persist and document.get("workspace_id"):
        row = {
            "workspace_id": document["workspace_id"], "document_id": document_id, "job_id": job_id, "status": status, "phase": phase,
            "trigger": trigger, "metrics": result["metrics"], "findings": result["findings"], "code_version": CODE_VERSION,
        }
        try:
            saved = _rows(await asyncio.to_thread(db.table("ingestion_audits").insert(row).execute))
            audit_id = saved[0]["id"] if saved else None
        except Exception as exc:
            log.error("audit of %s could not be stored: %s", document_id, exc)
            status = "audit_error"
    if job_id:
        event = {"stage": STAGE, "status": "completed" if status == "passed" else "completed_with_findings", "completed_at": datetime.utcnow().isoformat(),
                 "data": {"audit_status": status, "audit_id": audit_id, "findings": len(result["findings"])}, "source": STAGE}
        try:
            await asyncio.to_thread(db.rpc("append_stage_history", {"p_job_id": job_id, "p_event": event}).execute)
            await asyncio.to_thread(db.rpc("merge_background_job_metadata", {"p_job_id": job_id, "p_metadata": {"audit": {"status": status, "audit_id": audit_id}}}).execute)
        except Exception as exc:
            log.warning("audit of %s could not be recorded on job %s: %s", document_id, job_id, exc)
    notified = 0
    if notify and status in NOTIFY_STATUSES:
        try:
            notified = await asyncio.to_thread(_notify, settings, db, document, status, result, audit_id, job_id)
        except Exception as exc:
            log.warning("audit of %s could not notify anyone: %s", document_id, exc)
    return {"status": status, "audit_id": audit_id, "notified": notified, "metrics": result["metrics"], "findings": result["findings"]}
