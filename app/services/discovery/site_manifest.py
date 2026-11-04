"""What the maker's own site shows for each discovered series, gathered before the PDF is read in depth.

Stored on the product as `metadata.site_manifest`: the site's photos of the series (page slug = the
series name), the variant names in their file names, and a status. Stage 3 reads it to skip paid
work on catalogue crops the site already shows; the audit names series the site does not have.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

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



def fold_slug(text: str) -> str:
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()


def slug_tokens(url: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", fold_slug(url.rstrip("/").rsplit("/", 1)[-1])) if t]


def series_only_on_site(sitemap_pages: Iterable[str], known_pages: Iterable[str], names: Iterable[str]) -> List[str]:
    """Pages beside the series' own pages, sharing a word with the catalogue, naming no discovered product."""
    known = [[t for t in re.split(r"[^a-z0-9]+", fold_slug(n)) if t] for n in names]
    words = {t for tokens in known for t in tokens if len(t) > 3}
    parents = {page.rstrip("/").rsplit("/", 1)[0] for page in known_pages}
    out: List[str] = []
    for url in sitemap_pages:
        tokens = slug_tokens(url)
        if url.rstrip("/").rsplit("/", 1)[0] not in parents or not set(tokens) & words:
            continue
        if any(t and all(w in tokens for w in t) for t in known) or url in out:
            continue
        out.append(url)
    return out


async def sitemap_pages(root: str, max_files: int = 6) -> List[str]:
    """The site's page URLs, or nothing when it does not publish them to us."""
    from app.utils.ssrf_guard import safe_fetch_bytes

    pending, seen, pages = [root.rstrip("/") + "/sitemap.xml"], set(), []
    while pending and len(seen) < max_files:
        url = pending.pop(0)
        seen.add(url)
        try:
            result = await safe_fetch_bytes(url, max_bytes=5_000_000, timeout=20.0)
            text = result.content.decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001 - a blocked or absent sitemap only means no suggestions
            logger.info(f"sitemap {url} not read: {type(e).__name__}")
            continue
        for loc in re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", text):
            (pending if loc.endswith(".xml") else pages).append(loc)
    return pages


async def report_site_only_series(db: Any, document_id: str, workspace_id: str, job_id: Optional[str] = None) -> Dict[str, Any]:
    """Models the maker lists that the PDF does not print. Only named for a person to decide; never created."""
    from app.services.pdf.ingestion_corrections import NEEDS_DECISION, Correction, record

    rows = db.table("products").select("name, metadata").eq("source_document_id", document_id).execute().data or []
    manifests = [(r.get("metadata") or {}).get("site_manifest") or {} for r in rows]
    roots = {m.get("site_root") for m in manifests if m.get("site_root")}
    pages = [page for m in manifests for page in (m.get("pages") or [])]
    if len(roots) != 1 or not pages:
        return {"site_only": []}
    found = series_only_on_site(await sitemap_pages(next(iter(roots))), pages, [r["name"] for r in rows])
    if found:
        await record(db, document_id, workspace_id, "discovery", [Correction(
            "series_only_on_maker_site", NEEDS_DECISION,
            f"The maker's site lists {len(found)} page(s) beside these series that the PDF does not name", len(found), found[:50])], job_id)
    return {"site_only": found}
