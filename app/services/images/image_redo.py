"""Re-run a document's product images through the current Stage 3 and image sourcing, without
re-running discovery, chunking or product extraction.

Per product: the images it holds from this document are retired (vectors purged, row kept and
marked `excluded_from_products`), Stage 3 re-crops its `page_range`, the new rows are linked
to it, and image sourcing then replaces / repairs / embeds the low-resolution ones.
"""

from __future__ import annotations

import logging
import os
import tempfile
import types
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def run_image_redo(document_id: str, workspace_id: str, user_id: Optional[str],
                         product_ids: Optional[List[str]] = None, allow_restore: bool = True,
                         log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    from app.api.pdf_processing.stage_3_images import process_product_images
    from app.services.embeddings.vecs_service import get_vecs_service
    from app.services.images.image_sourcing import run_image_sourcing
    from app.utils.pdf_to_images import analyze_pdf_layout

    log = log or logger
    db = get_supabase_client().client
    doc = (db.table("documents").select("storage_bucket,storage_object_path,workspace_id")
           .eq("id", document_id).limit(1).execute().data or [None])[0]
    if not doc or not doc.get("storage_object_path"):
        return {"ok": False, "error": "document has no stored PDF"}
    pdf = db.storage.from_(doc["storage_bucket"] or "pdf-documents").download(doc["storage_object_path"])
    job = (db.table("background_jobs").select("id").eq("document_id", document_id)
           .order("created_at", desc=True).limit(1).execute().data or [{}])[0].get("id")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
        fh.write(pdf)
        pdf_path = fh.name
    try:
        layout = analyze_pdf_layout(pdf_path)
    finally:
        os.unlink(pdf_path)
    catalog = types.SimpleNamespace(
        has_spread_layout=layout.has_spread_layout,
        physical_to_pdf_map=layout.physical_to_pdf_map,
        pdf_page_widths={p.pdf_page_num: p.width for p in layout.pages},
        products=[],
    )

    q = db.table("products").select("id,name,category,workspace_id,metadata")
    q = q.in_("id", product_ids) if product_ids else q.eq("source_document_id", document_id)
    products = q.execute().data or []
    vecs = get_vecs_service()
    summary: Dict[str, Any] = {"products": len(products), "retired": 0, "new_images": 0, "per_product": []}
    redo_started = _now()
    pages_of = {p["id"]: [int(x) for x in ((p.get("metadata") or {}).get("page_range") or []) if str(x).isdigit()]
                for p in products}
    # A page shared by several products is extracted, classified and embedded ONCE, then linked to each.
    done_pages: set = set()

    for product in products:
        pages = pages_of[product["id"]]
        if not pages:
            summary["per_product"].append({"product": product["name"], "skipped": "no page_range"})
            continue

        linked = [r["image_id"] for r in (db.table("image_product_associations").select("image_id")
                                          .eq("product_id", product["id"]).execute().data or [])]
        old = (db.table("document_images").select("id,metadata").in_("id", linked).eq("document_id", document_id)
               .execute().data or []) if linked else []
        if old:
            await vecs.delete_embeddings_by_image_ids([r["id"] for r in old])
            for r in old:
                m = dict(r.get("metadata") or {})
                m.setdefault("excluded_from_products", {"at": _now(), "reason": "replaced_by_image_redo"})
                db.table("document_images").update({"metadata": m}).eq("id", r["id"]).execute()
            db.table("image_product_associations").delete().eq("product_id", product["id"]) \
                .in_("image_id", [r["id"] for r in old]).execute()
        summary["retired"] += len(old)

        todo = [p for p in pages if p not in done_pages]
        if todo:
            await process_product_images(
                file_content=pdf, document_id=document_id, workspace_id=product.get("workspace_id") or workspace_id,
                job_id=job, product=types.SimpleNamespace(name=product["name"]), physical_pages=todo, catalog=catalog,
                config={"material_category": product.get("category")}, logger=log, product_db_id=product["id"],
            )
            done_pages.update(todo)
        summary["per_product"].append({"product": product["name"], "pages": len(pages), "processed": len(todo), "retired": len(old)})
        log.info(f"🖼️ Image redo {product['name']}: retired {len(old)}, processed {len(todo)} of {len(pages)} pages")

    new = (db.table("document_images").select("id,page_number,metadata").eq("document_id", document_id)
           .gte("created_at", redo_started).execute().data or [])
    rows = []
    for n in new:
        if ((n.get("metadata") or {}).get("ai_classification") or {}).get("is_material") is False:
            continue
        for product in products:
            if n.get("page_number") in pages_of[product["id"]]:
                rows.append({"workspace_id": product.get("workspace_id") or workspace_id, "product_id": product["id"],
                             "image_id": n["id"], "metadata": {"association_method": "image_redo"}})
    for i in range(0, len(rows), 500):
        db.table("image_product_associations").upsert(rows[i:i + 500], on_conflict="image_id,product_id").execute()
    summary["new_images"] = len(new)
    summary["links"] = len(rows)

    summary["sourcing"] = await run_image_sourcing(
        document_id, workspace_id, user_id, product_ids=[p["id"] for p in products], allow_restore=allow_restore, log=log,
    )
    return summary
