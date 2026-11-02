"""Image sourcing: replace low-resolution PDF crops with the manufacturer's own files, repair
what has no original, then embed each final image once.

Order per product: crops Stage 3 marked `image_quality.sourcing = pending` (embedding deferred)
-> the manufacturer page, CONFIRMED by a visual match of one of those crops -> every product
image on it, classified -> repair for crops nothing matched -> one embed per surviving image.
A replaced crop keeps its row (`excluded_from_products`, `superseded_by`); nothing is deleted.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from PIL import Image

from app.services.core.supabase_client import get_supabase_client
from app.services.images import manufacturer_site as site
from app.services.images.image_matching import (
    MatchScore, classical_cleanup, faces_in_crop, is_patterned, match_score, pick_match, repair_is_faithful,
)
from app.utils.ssrf_guard import safe_fetch_bytes
from app.services.images.association_scores import PAGE_LINK_SCORE, SOURCED_MATCH_SCORE

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 40
DOWNLOAD_CONCURRENCY = 8
MAX_IMAGE_BYTES = 12 * 1024 * 1024
# Swatch files are long strips (Ceramica Euro: 886x296), so the short side alone stays low.
MIN_CANDIDATE_SHORT_SIDE = 200
GEMINI_ASPECTS = (1.0, 16 / 9, 3 / 2, 4 / 3, 9 / 16, 3 / 4, 4 / 5, 5 / 4, 21 / 9, 2 / 3)
RESTORE_PROMPT = (
    "Restore this photo of a ceramic tile product. Remove the JPEG compression blocks, pixelation and colour "
    "blotches and output it at high resolution. Do NOT change the pattern, motif, veining, colours, layout, "
    "framing, crop or aspect ratio. Do not add, remove or redraw anything. The result must be the same image, "
    "only cleaner and sharper."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _open(data: bytes) -> Optional[Image.Image]:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except Exception:
        return None


def _thumb(img: Image.Image, side: int = 720) -> Image.Image:
    t = img.copy()
    t.thumbnail((side, side), Image.LANCZOS)
    return t


def _jpeg(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


async def _download(url: str) -> Optional[bytes]:
    try:
        r = await safe_fetch_bytes(url, max_bytes=MAX_IMAGE_BYTES, timeout=40, headers=site.BROWSER_UA)
        return r.content if r.ok else None
    except Exception:
        return None


def _same_site(url: str) -> str:
    """One spelling per URL: www.example.com and example.com serve the same page and file."""
    p = urlparse(url)
    return p._replace(netloc=(p.netloc or "").removeprefix("www.")).geturl()


def _site_root(url: Optional[str]) -> Optional[str]:
    if not url or not url.lower().startswith("https://"):
        return None
    p = urlparse(url)
    return f"https://{p.hostname}/" if p.hostname else None


class ImageSourcing:
    def __init__(self, document_id: str, workspace_id: str, user_id: Optional[str], allow_restore: bool = True,
                 log: Optional[logging.Logger] = None, dry_run: bool = False, low_res_embedded: bool = False) -> None:
        self.dry_run = dry_run
        # Also take crops already embedded with a low_res verdict, and products holding no image at all.
        self.low_res_embedded = low_res_embedded
        self.document_id = document_id
        self.workspace_id = workspace_id
        self.user_id = user_id
        self.allow_restore = allow_restore
        self.log = log or logger
        self.sb = get_supabase_client()
        self.db = self.sb.client
        self._links: Dict[str, List[Tuple[str, str]]] = {}
        self._downloads: Dict[str, Optional[bytes]] = {}
        self._siblings: Dict[str, List[str]] = {}
        # Image url -> its page, for images found by search on a page whose slug IS the product name.
        self._name_confirmed: Dict[str, str] = {}
        self.summary: Dict[str, Any] = {"products": 0, "pending_crops": 0, "matched_crops": 0,
                                        "manufacturer_images": 0, "restored": 0, "kept_source": 0,
                                        "rejected_non_product": 0, "errors": []}

    # ── data ────────────────────────────────────────────────────────────────

    def _products(self, product_ids: Optional[List[str]]) -> List[Dict[str, Any]]:
        q = self.db.table("products").select("id,name,workspace_id,brand_company_id,product_url,metadata")
        q = q.in_("id", product_ids) if product_ids else q.eq("source_document_id", self.document_id)
        return q.execute().data or []

    def _site_for(self, product: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """(site root, a known product page): brand company website, then product_url, then the PDF's factory site."""
        page = product.get("product_url") if (product.get("product_url") or "").startswith("https://") else None
        if product.get("brand_company_id"):
            rows = self.db.table("crm_companies").select("website").eq("id", product["brand_company_id"]).limit(1).execute().data
            root = _site_root((rows or [{}])[0].get("website"))
            if root:
                return root, page
        if page:
            return _site_root(page), page
        factory = ((product.get("metadata") or {}).get("factory") or {})
        web = factory.get("website") if isinstance(factory, dict) else None
        if isinstance(web, dict):
            web = web.get("value")
        if isinstance(web, str) and web and not web.startswith("http"):
            web = "https://" + web.strip().strip("/")
        return _site_root(web if isinstance(web, str) else None), None

    def _pending_crops(self, product_id: str) -> List[Dict[str, Any]]:
        links = self.db.table("image_product_associations").select("image_id").eq("product_id", product_id).execute().data or []
        ids = [r["image_id"] for r in links]
        if not ids:
            return []
        rows = self.db.table("document_images").select("id,image_url,page_number,document_id,metadata,has_slig_embedding") \
            .in_("id", ids).execute().data or []
        out = []
        for r in rows:
            m = r.get("metadata") or {}
            if "excluded_from_products" in m:
                continue
            quality = m.get("image_quality") or {}
            pending = quality.get("sourcing") == "pending" or (
                self.low_res_embedded and quality.get("verdict") == "low_res" and not quality.get("sourcing"))
            # A dry run also previews crops made before the quality marker existed.
            legacy = self.dry_run and m.get("extraction_layer") == "region_crop" and m.get("region_type") in ("IMAGE", "FIGURE", None)
            if pending or legacy:
                out.append(r)
        return out

    def _has_images(self, product_id: str) -> bool:
        links = self.db.table("image_product_associations").select("image_id").eq("product_id", product_id).execute().data or []
        ids = [r["image_id"] for r in links]
        rows = (self.db.table("document_images").select("metadata").in_("id", ids).execute().data or []) if ids else []
        return any("excluded_from_products" not in (r.get("metadata") or {}) for r in rows)

    def _sibling_names(self, product: Dict[str, Any]) -> List[str]:
        key = product.get("brand_company_id") or f"doc:{self.document_id}"
        if key not in self._siblings:
            q = self.db.table("products").select("name").eq("workspace_id", product.get("workspace_id") or self.workspace_id)
            q = q.eq("brand_company_id", key) if product.get("brand_company_id") else q.eq("source_document_id", self.document_id)
            self._siblings[key] = [r["name"] for r in (q.execute().data or []) if r.get("name")]
        return self._siblings[key]

    async def _search(self, product: Dict[str, Any], root: str) -> List[Tuple[str, str]]:
        """Image search for a site that refuses our HTML fetch (bot walls): its own files, by page slug."""
        from app.services.integrations.firecrawl_client import get_firecrawl_client
        host = (urlparse(root).hostname or "").removeprefix("www.")
        hits: List[Dict[str, Any]] = []
        for query in (f"site:{host} {product['name']}", f"{host.split('.')[0]} {product['name']}"):
            hits += await get_firecrawl_client().search_images(
                query, self.user_id, product.get("workspace_id") or self.workspace_id, limit=MAX_CANDIDATES,
                source_tag="image_sourcing")
        return site.search_hits_for(hits, root, product["name"], self._sibling_names(product))

    # ── candidates ──────────────────────────────────────────────────────────

    async def _candidates(self, product: Dict[str, Any]) -> Tuple[List[str], Dict[str, Tuple[str, bytes, Image.Image]]]:
        root, known_page = self._site_for(product)
        pages: List[str] = [known_page] if known_page else []
        if root:
            if root not in self._links:
                self._links[root] = await site.site_links(root)
            pages += site.candidate_pages(self._links[root], product["name"])
        pages = list(dict.fromkeys(_same_site(p) for p in pages))
        urls: List[str] = []
        for p in pages[:2]:
            for u in await site.page_images(p):
                if _same_site(u) not in {_same_site(x) for x in urls}:
                    urls.append(u)
        if not urls and root:
            for u, page in await self._search(product, root):
                self._name_confirmed[u] = page
                urls.append(u)
        cands: Dict[str, Tuple[str, bytes, Image.Image]] = {}
        seen_hash = set()
        # Supplier sites are slow per connection (cereuro.it ~160 KB/s): fetch in parallel, once per run.
        todo = [u for u in urls[:MAX_CANDIDATES] if u not in self._downloads]
        sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)

        async def _fetch(u: str) -> None:
            async with sem:
                self._downloads[u] = await _download(u)

        await asyncio.gather(*(_fetch(u) for u in todo))
        for u in urls[:MAX_CANDIDATES]:
            data = self._downloads.get(u)
            img = _open(data) if data else None
            if img is None or min(img.size) < MIN_CANDIDATE_SHORT_SIDE:
                continue
            h = hashlib.sha1(data).hexdigest()
            if h in seen_hash:
                continue
            seen_hash.add(h)
            cands[u] = (self._name_confirmed.get(u) or (pages[0] if pages else ""), data, img)
        return pages, cands

    # ── writes ──────────────────────────────────────────────────────────────

    async def _classify(self, data: bytes) -> Optional[Dict[str, Any]]:
        from app.services.images.classification_verdict import classification_failed
        from app.services.pdf.embedded_image_format import sniff_media_type
        from app.services.search.rag_service import RAGService
        verdict = await RAGService()._classify_image_material(
            image_base64=base64.b64encode(data).decode(), confidence_threshold=0.6, media_type=sniff_media_type(data),
        )
        return None if classification_failed(verdict) else verdict

    async def _store(self, product: Dict[str, Any], img: Image.Image, page_number: int, layer: str,
                     source_type: str, extra: Dict[str, Any], classification: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        data = _jpeg(img)
        up = await self.sb.upload_image_file(data, f"{layer}_{uuid.uuid4().hex[:12]}.jpg", self.document_id, page_number)
        if not up.get("success"):
            self.summary["errors"].append(f"upload failed for {product['name']}")
            return None
        cls = classification or {}
        row = {
            "document_id": self.document_id,
            "workspace_id": product.get("workspace_id") or self.workspace_id,
            "image_url": up["public_url"],
            "image_type": cls.get("classification") or "PRODUCT_IMAGE",
            "caption": f"{product['name']}: {'manufacturer image' if layer == 'manufacturer_original' else 'restored image'}",
            "page_number": page_number or 1,
            "confidence": 0.95,
            "processing_status": "completed",
            "category": "product",
            "source_type": source_type,
            "extraction_layer": layer,
            "metadata": {
                "source": "image_sourcing", "extraction_layer": layer,
                "storage_bucket": "pdf-tiles", "storage_path": up["storage_path"], "storage_uploaded": True,
                "width": img.width, "height": img.height, "format": "JPEG",
                "image_quality": {"verdict": "ok", "native_px": [img.width, img.height]},
                "ai_classification": {k: cls.get(k) for k in ("is_material", "confidence", "reason", "model", "classification")} if cls else None,
                "sourced_at": _now(), **extra,
            },
        }
        res = self.db.table("document_images").insert(row).execute()
        if not res.data:
            return None
        saved = res.data[0]
        self.db.table("image_product_associations").upsert(
            {"workspace_id": row["workspace_id"], "product_id": product["id"], "image_id": saved["id"],
             "overall_score": SOURCED_MATCH_SCORE if layer == "manufacturer_original" else PAGE_LINK_SCORE,
             "metadata": {"association_method": layer}},
            on_conflict="image_id,product_id",
        ).execute()
        return saved

    async def _retire_vectors(self, crop: Dict[str, Any]) -> None:
        """A crop embedded before it was superseded loses its vectors once no product holds it."""
        if not crop.get("has_slig_embedding"):
            return
        if self.db.table("image_product_associations").select("image_id").eq("image_id", crop["id"]).limit(1).execute().data:
            return
        from app.services.embeddings.vecs_service import get_vecs_service
        await get_vecs_service().delete_embeddings_by_image_ids([crop["id"]])

    def _supersede(self, crop: Dict[str, Any], product_id: str, by: List[str], reason: str) -> None:
        m = dict(crop.get("metadata") or {})
        q = dict(m.get("image_quality") or {})
        q.update({"sourcing": "replaced", "superseded_by": by})
        m["image_quality"] = q
        m["excluded_from_products"] = {"at": _now(), "reason": reason, "replaced_by": by}
        self.db.table("document_images").update({"metadata": m}).eq("id", crop["id"]).execute()
        self.db.table("image_product_associations").delete().eq("image_id", crop["id"]).eq("product_id", product_id).execute()

    def _mark(self, crop: Dict[str, Any], state: str, **extra: Any) -> None:
        m = dict(crop.get("metadata") or {})
        q = dict(m.get("image_quality") or {})
        q.update({"sourcing": state, **extra})
        m["image_quality"] = q
        self.db.table("document_images").update({"metadata": m}).eq("id", crop["id"]).execute()
        crop["metadata"] = m

    async def _embed(self, row: Dict[str, Any], data: bytes) -> bool:
        from app.services.embeddings.classification_backfill import _embed_confirmed_material
        from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
        from app.services.embeddings.vecs_service import get_vecs_service
        try:
            return await _embed_confirmed_material(
                {**row, "workspace_id": row.get("workspace_id") or self.workspace_id}, data,
                os.getenv("ANTHROPIC_API_KEY", ""), RealEmbeddingsService(), get_vecs_service(),
            )
        except Exception as e:
            self.summary["errors"].append(f"embed {row.get('id')}: {str(e)[:120]}")
            return False

    # ── repair ──────────────────────────────────────────────────────────────

    async def _gemini_restore(self, crop_img: Image.Image) -> Optional[Image.Image]:
        """Through the edge function (its key, its credit debit), padded to a ratio the model keeps."""
        if not self.user_id:
            return None
        ratio = crop_img.width / crop_img.height
        target = min(GEMINI_ASPECTS, key=lambda r: abs(r - ratio) / r)
        padded, box = crop_img, None
        if abs(target - ratio) / target > 0.04:
            w, h = crop_img.size
            W, H = (round(h * target), h) if target > ratio else (w, round(w / target))
            padded = Image.new("RGB", (W, H), (128, 128, 128))
            ox, oy = (W - w) // 2, (H - h) // 2
            padded.paste(crop_img, (ox, oy))
            box = (ox / W, oy / H, (ox + w) / W, (oy + h) / H)
        tmp = await self.sb.upload_image_file(_jpeg(padded, 95), f"restore_in_{uuid.uuid4().hex[:10]}.jpg", self.document_id)
        if not tmp.get("success"):
            return None
        try:
            async with httpx.AsyncClient(timeout=170) as client:
                r = await client.post(
                    f"{os.environ['SUPABASE_URL']}/functions/v1/generate-interior-gemini",
                    headers={"Authorization": f"Bearer {os.environ['SUPABASE_SERVICE_ROLE_KEY']}",
                             "Content-Type": "application/json", "User-Agent": "mivaa-image-sourcing"},
                    json={"mode": "image-edit", "reference_image_url": tmp["public_url"], "edit_instruction": RESTORE_PROMPT,
                          "user_id": self.user_id, "workspace_id": self.workspace_id},
                )
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            url = body.get("image_url") or body.get("imageUrl")
            if r.status_code != 200 or not url:
                self.summary["errors"].append(f"gemini restore {r.status_code}: {str(body.get('error'))[:120]}")
                return None
            out = _open(await _download(url) or b"")
            if out is not None and box:
                W, H = out.size
                out = out.crop((round(box[0] * W), round(box[1] * H), round(box[2] * W), round(box[3] * H)))
            return out
        finally:
            try:
                self.db.storage.from_("pdf-tiles").remove([tmp["storage_path"]])
            except Exception:
                pass

    async def _repair(self, product: Dict[str, Any], crop: Dict[str, Any], crop_img: Image.Image, crop_bytes: bytes) -> None:
        native = (crop.get("metadata") or {}).get("image_quality", {}).get("native_px")
        restored, method, fidelity = None, None, None
        if self.allow_restore and is_patterned(crop_img):
            g = await self._gemini_restore(crop_img)
            if g is not None:
                ok, fidelity = repair_is_faithful(crop_img, g, native)
                if ok:
                    restored, method = g, "gemini-3.1-flash-image"
                else:
                    self.log.info(f"   ✖ Gemini repair redrew {crop['id']} (fidelity {fidelity}) — using the classical clean-up")
        if restored is None and self.allow_restore:
            restored, method = classical_cleanup(crop_img), "classical"
            fidelity = repair_is_faithful(crop_img, restored, native)[1]
        if restored is None:
            self._mark(crop, "kept_source")
            if not crop.get("has_slig_embedding"):
                await self._embed(crop, crop_bytes)
            self.summary["kept_source"] += 1
            return
        cls = (crop.get("metadata") or {}).get("ai_classification")
        saved = await self._store(product, restored, crop.get("page_number") or 1, "restored", "pdf_processing",
                                  {"restored_from": crop["id"], "restore_method": method, "restore_fidelity": fidelity}, cls)
        if not saved:
            self._mark(crop, "kept_source")
            if not crop.get("has_slig_embedding"):
                await self._embed(crop, crop_bytes)
            self.summary["kept_source"] += 1
            return
        self._supersede(crop, product["id"], [saved["id"]], "superseded_by_restored")
        await self._retire_vectors(crop)
        await self._embed(saved, _jpeg(restored))
        self.summary["restored"] += 1

    # ── run ─────────────────────────────────────────────────────────────────

    async def run_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        crops = self._pending_crops(product["id"])
        report: Dict[str, Any] = {"product": product["name"], "pending": len(crops), "matched": 0, "added": 0, "restored_or_kept": 0}
        imageless = self.low_res_embedded and not crops and not self._has_images(product["id"])
        if not crops and not imageless:
            return report
        self.summary["pending_crops"] += len(crops)
        pages, cands = await self._candidates(product)
        report["pages"] = pages
        loaded: List[Tuple[Dict[str, Any], bytes, Image.Image]] = []
        for c in crops:
            data = await _download(c["image_url"])
            img = _open(data) if data else None
            if img is not None:
                loaded.append((c, data, img))

        # The matcher works at <=360 px; shrinking each picture once, not per comparison, is the run time.
        small = {u: _thumb(v[2]) for u, v in cands.items()}
        matches: Dict[str, List[Tuple[str, MatchScore]]] = {}
        for c, _, img in loaded:
            crop_small = _thumb(img)
            scores = [(u, match_score(crop_small, s)) for u, s in small.items()]
            faces = faces_in_crop(scores)
            best = pick_match(scores)
            hit = faces if faces else ([best] if best else [])
            if hit:
                matches[c["id"]] = hit
        # A page found by search is confirmed by its slug being the product's own name.
        confirmed = bool(matches) or any(u in self._name_confirmed for u in cands)
        report["matched"] = len(matches)
        self.summary["matched_crops"] += len(matches)
        if self.dry_run:
            report["candidates"] = len(cands)
            report["matches"] = {cid: [(u.rsplit("/", 1)[-1], round(s.score, 3), s.detail) for u, s in hits]
                                 for cid, hits in matches.items()}
            report["would_repair"] = [{"crop": c["id"], "page": c.get("page_number"),
                                       "method": "gemini" if is_patterned(img) else "classical"}
                                      for c, _, img in loaded if c["id"] not in matches]
            return report

        stored: Dict[str, str] = {}
        if confirmed:
            for u, (page, data, img) in cands.items():
                verdict = await self._classify(data)
                if not verdict or not verdict.get("is_material"):
                    self.summary["rejected_non_product"] += 1
                    continue
                how = next(({"mode": s.mode, "score": round(s.score, 3), "detail": s.detail, "crop": cid}
                            for cid, hits in matches.items() for hu, s in hits if hu == u), None)
                page_no = next((c.get("page_number") for c, _, _ in loaded if how and c["id"] == how["crop"]), None)
                confirmation = {"mode": "confirmed_page", "confirmed_by": sorted(matches)} if matches \
                    else {"mode": "page_slug_is_product_name"}
                saved = await self._store(product, img, page_no or 1, "manufacturer_original", "web_scraping",
                                          {"source_url": u, "source_page": page, "match": how or confirmation},
                                          verdict)
                if saved:
                    stored[u] = saved["id"]
                    await self._embed(saved, data)
                    self.summary["manufacturer_images"] += 1
                    report["added"] += 1

        for c, data, img in loaded:
            hit_ids = [stored[u] for u, _ in matches.get(c["id"], []) if u in stored]
            if hit_ids:
                self._supersede(c, product["id"], hit_ids, "superseded_by_manufacturer_original")
                await self._retire_vectors(c)
            else:
                await self._repair(product, c, img, data)
                report["restored_or_kept"] += 1
        return report

    async def run_all(self, product_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        reports = []
        products = self._products(product_ids)
        for n, product in enumerate(products, 1):
            self.summary["products"] += 1
            try:
                reports.append(await self.run_product(product))
                r = reports[-1]
                self.log.info(f"🖼️ Image sourcing {n}/{len(products)} {product['name']}: {r.get('pending', 0)} low-res, "
                              f"{r.get('matched', 0)} matched, {r.get('added', 0)} manufacturer images added")
            except Exception as e:
                self.log.exception(f"image sourcing failed for {product.get('name')}")
                self.summary["errors"].append(f"{product.get('name')}: {str(e)[:160]}")
            await asyncio.sleep(0)
        self.summary["per_product"] = reports
        return self.summary


async def run_image_sourcing(document_id: str, workspace_id: str, user_id: Optional[str] = None,
                             product_ids: Optional[List[str]] = None, allow_restore: bool = True,
                             log: Optional[logging.Logger] = None, dry_run: bool = False,
                             low_res_embedded: bool = False) -> Dict[str, Any]:
    return await ImageSourcing(document_id, workspace_id, user_id, allow_restore, log, dry_run,
                               low_res_embedded).run_all(product_ids)
