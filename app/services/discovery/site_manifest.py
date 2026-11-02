"""What the maker's own site shows for each discovered series, gathered before the PDF is read in depth.

Stored on the product as `metadata.site_manifest`: the site's photos of the series (page slug = the
series name), the variant names in their file names, and a status. Stage 3 reads it to skip paid
work on catalogue crops the site already shows; the audit names series the site does not have.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MIN_GALLERY = 3
MAX_PHOTOS = 12


def enabled() -> bool:
    return os.getenv("SITE_MANIFEST_ENABLED", "true").lower() not in ("0", "false", "no")


def gallery(product: Dict[str, Any]) -> List[str]:
    """The site photos a series' manifest confirmed, when there are enough to stand in for crops."""
    m = ((product.get("metadata") or {}).get("site_manifest") or {})
    photos = [p for p in (m.get("photos") or []) if isinstance(p, str)]
    return photos if m.get("status") == "found" and len(photos) >= MIN_GALLERY else []


SMALL_CROP_PX = 300


async def drop_crops_the_site_shows(images: List[Dict[str, Any]], product_id: Optional[str], workspace_id: str,
                                    log: Optional[logging.Logger] = None) -> List[Dict[str, Any]]:
    """Leave out small crops whose tile a site photo already shows, before any paid work is done on them.

    Both checks must pass: the vision coverage check, then the colour and size gate. Any failure keeps the crop.
    """
    import io
    import httpx
    from PIL import Image
    from app.services.core.claude_models import SONNET
    from app.services.core.supabase_client import get_supabase_client
    from app.services.images.gallery_coverage import coverage
    from app.services.images.image_matching import gallery_can_replace
    from app.services.utilities.prompt_registry import load_prompt

    log = log or logger
    if not product_id or not images:
        return images
    try:
        product = (get_supabase_client().client.table("products").select("name, metadata").eq("id", product_id)
                   .limit(1).execute().data or [None])[0]
        photos = gallery(product or {})
        small = []
        for img in images:
            try:
                with Image.open(img.get("path") or "") as raw:
                    if max(raw.size) < SMALL_CROP_PX:
                        small.append((img, raw.convert("RGB")))
            except Exception:  # noqa: BLE001 - unreadable crop: keep it, the pipeline decides
                continue
        if not photos or not small:
            return images
        async with httpx.AsyncClient(timeout=40) as http:
            site = []
            for url in photos:
                r = await http.get(url)
                if r.status_code == 200:
                    site.append(Image.open(io.BytesIO(r.content)).convert("RGB"))
        if not site:
            return images
        covered = await coverage([s for _, s in small], site, product_name=product["name"], model=SONNET,
                                 prompt=await load_prompt("extraction", "gallery_coverage", stage="quality"),
                                 workspace_id=workspace_id, product_id=product_id)
        drop = {id(small[i][0]) for i, g in covered.items() if gallery_can_replace(small[i][1], site[g], max(small[i][1].size), False)}
    except Exception as e:  # noqa: BLE001 - keep every crop when the check cannot run
        log.warning(f"   ⚠️ Site-gallery check skipped: {e}")
        return images
    if drop:
        log.info(f"   🌐 {len(drop)} small crop(s) left out: the maker's site photos already show them")
    return [img for img in images if id(img) not in drop]


async def build_manifest(document_id: str, workspace_id: str, user_id: Optional[str],
                         product_ids: List[str]) -> Dict[str, Any]:
    """Search the maker's site once per series and record what it shows. Never raises."""
    from app.services.core.supabase_client import get_supabase_client
    from app.services.discovery.series_reconcile import site_variant_names
    from app.services.images.image_sourcing import ImageSourcing

    summary: Dict[str, Any] = {"series": 0, "found": 0, "not_found": 0, "no_site": 0, "errors": []}
    if not enabled() or not product_ids:
        return summary
    db = get_supabase_client().client
    sourcing = ImageSourcing(document_id, workspace_id, user_id)
    for start in range(0, len(product_ids), 100):
        rows = db.table("products").select("id, name, workspace_id, brand_company_id, product_url, metadata") \
            .in_("id", product_ids[start:start + 100]).execute().data or []
        for p in rows:
            summary["series"] += 1
            record: Dict[str, Any] = {"searched_at": datetime.now(timezone.utc).isoformat(), "photos": [], "pages": [], "variants": []}
            try:
                root, _ = sourcing._site_for(p)
                if not root:
                    record["status"] = "no_site"
                    summary["no_site"] += 1
                else:
                    hits = await sourcing._search(p, root)
                    record.update(site_root=root, photos=[img for img, _ in hits][:MAX_PHOTOS],
                                  pages=sorted({page for _, page in hits}),
                                  variants=site_variant_names([img for img, _ in hits], p["name"]))
                    record["status"] = "found" if record["photos"] else "not_found"
                    summary["found" if record["photos"] else "not_found"] += 1
            except Exception as e:  # noqa: BLE001 - a failed search is recorded, never fatal to ingestion
                record["status"] = "error"
                record["error"] = f"{type(e).__name__}: {e}"[:200]
                summary["errors"].append(f"{p['name']}: {record['error']}")
            meta = dict(p.get("metadata") or {})
            meta["site_manifest"] = record
            db.table("products").update({"metadata": meta}).eq("id", p["id"]).execute()
    logger.info(f"   🌐 Site manifest: {summary['found']}/{summary['series']} series on the maker's site, "
                f"{summary['not_found']} not found, {summary['no_site']} with no known site")
    return summary
