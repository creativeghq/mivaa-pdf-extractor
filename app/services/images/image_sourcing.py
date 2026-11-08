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
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx
from PIL import Image

from app.services.core.supabase_client import get_supabase_client
from app.services.images import manufacturer_site as site
from app.services.images.image_matching import (
    MatchScore, classical_cleanup, faces_in_crop, gallery_can_replace, is_patterned, match_score, pick_match,
    redundant_resized_copies, redundant_scale_renders, repair_is_faithful, within_upscale,
)
from app.utils.ssrf_guard import safe_fetch_bytes
from app.services.images.association_scores import PAGE_LINK_SCORE, SOURCED_MATCH_SCORE

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 40
DOWNLOAD_CONCURRENCY = 8
MAX_IMAGE_BYTES = 12 * 1024 * 1024
# Swatch files are often long strips, so the short side alone stays low.
MIN_CANDIDATE_SHORT_SIDE = 200
#: A crop whose real source is this small is a swatch thumbnail: no repair can be honest at product size.
SWATCH_MAX_SOURCE_PX = 120
#: With the maker's own photos attached, a PDF image this small on its long side adds nothing next to them.
GALLERY_SUPERSEDES_SOURCE_PX = 300
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


def _source_long_side(row: Dict[str, Any]) -> Optional[int]:
    m = row.get("metadata") or {}
    native = (m.get("image_quality") or {}).get("native_px") or []
    if len(native) == 2:
        return max(int(native[0]), int(native[1]))
    try:
        return max(int(m.get("width") or 0), int(m.get("height") or 0)) or None
    except (TypeError, ValueError):
        return None


def _small_beside_gallery(row: Dict[str, Any]) -> bool:
    long_side = _source_long_side(row)
    return long_side is not None and long_side < GALLERY_SUPERSEDES_SOURCE_PX


def _is_swatch(crop: Dict[str, Any]) -> bool:
    native = ((crop.get("metadata") or {}).get("image_quality") or {}).get("native_px") or []
    return len(native) == 2 and min(int(native[0]), int(native[1])) < SWATCH_MAX_SOURCE_PX


class ImageSourcing:
    def __init__(self, document_id: str, workspace_id: str, user_id: Optional[str], allow_restore: bool = True,
                 log: Optional[logging.Logger] = None, dry_run: bool = False, low_res_embedded: bool = False,
                 gallery_backfill: bool = False, job_id: Optional[str] = None) -> None:
        self.dry_run = dry_run
        self.job_id = job_id
        self.gallery_pairs: List[Dict[str, Any]] = []
        # Also look up the maker's gallery for products that already went through sourcing without one.
        self.gallery_backfill = gallery_backfill
        # Also take crops already embedded with a low_res verdict, and products holding no image at all.
        self.low_res_embedded = low_res_embedded
        self.document_id = document_id
        self.workspace_id = workspace_id
        self.allow_restore = allow_restore
        self.log = log or logger
        self.sb = get_supabase_client()
        self.db = self.sb.client
        from app.services.integrations.billing_principal import billing_principal
        self.user_id = user_id or billing_principal(self.db, workspace_id, document_id)
        self._links: Dict[str, List[Tuple[str, str]]] = {}
        self._downloads: Dict[str, Optional[bytes]] = {}
        self._siblings: Dict[str, List[str]] = {}
        # Image url -> its page, for images found by search on a page whose slug IS the product name.
        self._name_confirmed: Dict[str, str] = {}
        self._page_imgs: Dict[str, List[str]] = {}
        self.summary: Dict[str, Any] = {"products": 0, "pending_crops": 0, "matched_crops": 0,
                                        "manufacturer_images": 0, "restored": 0, "kept_source": 0,
                                        "rejected_non_product": 0, "errors": [], "restored_ids": [],
                                        "restored_without_gemini": [], "gemini_unavailable_reasons": [],
                                        "kept_source_ids": [], "repairs_downscaled": [], "unbilled": not self.user_id}
        self._gemini_failure: Optional[str] = None

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
            # A repair is the result of this step, never its input: a classical one is low_res on purpose.
            pending = m.get("extraction_layer") != "restored" and (quality.get("sourcing") == "pending" or (
                self.low_res_embedded and quality.get("verdict") == "low_res" and not quality.get("sourcing")))
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

    def _linked(self, product_id: str) -> List[Dict[str, Any]]:
        ids = [r["image_id"] for r in (self.db.table("image_product_associations").select("image_id")
                                       .eq("product_id", product_id).execute().data or [])]
        rows = (self.db.table("document_images").select("id,image_url,extraction_layer,metadata,has_slig_embedding")
                .in_("id", ids).execute().data or []) if ids else []
        return [r for r in rows if "excluded_from_products" not in (r.get("metadata") or {})]

    async def _gallery_twin(self, row: Dict[str, Any], gallery_thumbs: List[Tuple[str, Image.Image]]) -> Tuple[Optional[str], Optional[Image.Image]]:
        """(the gallery image this one pixel-matches or None, the image itself)."""
        data = await _download(row.get("image_url") or "")
        img = _open(data) if data else None
        if img is None:
            return None, None
        best = pick_match([(gid, match_score(_thumb(img), g)) for gid, g in gallery_thumbs])
        return (best[0] if best else None), img

    async def _gallery_covers(self, product: Dict[str, Any], unmatched: List[Tuple[Dict[str, Any], Image.Image]],
                              gallery: List[Tuple[str, Image.Image]]) -> Dict[str, str]:
        """{image id: gallery id} the vision check says the gallery already shows. A failed check covers nothing."""
        from app.services.core.claude_models import SONNET
        from app.services.images.gallery_coverage import GALLERY_MAX, coverage, split_covered
        from app.services.utilities.prompt_registry import load_prompt
        try:
            prompt = await load_prompt("extraction", "gallery_coverage", stage="quality")
            hits = await coverage([img for _, img in unmatched], [img for _, img in gallery], product_name=product.get("name") or "",
                                  model=SONNET, prompt=prompt, user_id=self.user_id,
                                  workspace_id=product.get("workspace_id") or self.workspace_id, product_id=product["id"])
        except Exception as e:
            logger.warning(f"gallery coverage check failed for {product['id']}, keeping every image: {e}")
            return {}
        covered, _ = split_covered([r["id"] for r, _ in unmatched], hits, [gid for gid, _ in gallery[:GALLERY_MAX]])
        return covered

    def _has_layer(self, product_id: str, layer: str) -> bool:
        return any(r.get("extraction_layer") == layer for r in self._linked(product_id))


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

    @staticmethod
    def _markers(product: Dict[str, Any]) -> List[str]:
        m = product.get("metadata") or {}
        codes = m.get("product_codes") if isinstance(m.get("product_codes"), list) else []
        skus = list((m.get("sku_codes") or {}).values()) if isinstance(m.get("sku_codes"), dict) else []
        return [product.get("name") or ""] + [str(c) for c in codes + skus if c]

    async def _images_on(self, page: str) -> List[str]:
        if page not in self._page_imgs:
            self._page_imgs[page] = await site.page_images(page)
        return self._page_imgs[page]

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
            named = site.page_names_product(p, product["name"])
            own = await self._images_on(p)
            others = [await self._images_on(o) for o in ([root] if root else []) + site.reference_pages(self._links.get(root) or [], p)]
            furniture = site.page_furniture(own, others, self._markers(product))
            self.summary["page_furniture_skipped"] = self.summary.get("page_furniture_skipped", 0) + len(furniture)
            for u in own:
                if site.image_key(u) in furniture:
                    continue
                if _same_site(u) not in {_same_site(x) for x in urls}:
                    urls.append(u)
                    if named:
                        self._name_confirmed[u] = p
        if not urls and root:
            for u, page in await self._search(product, root):
                self._name_confirmed[u] = page
                urls.append(u)
        cands: Dict[str, Tuple[str, bytes, Image.Image]] = {}
        seen_hash = set()
        # Supplier sites can be slow per connection: fetch in parallel, once per run.
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
        # Keep the largest picture of each surface: one photo saved at several sizes, and formats a site
        # draws to scale on a blank canvas, arrive as separate files.
        for u in redundant_resized_copies([(u, v[2]) for u, v in cands.items()]):
            del cands[u]
            self.summary["resized_copies_skipped"] = self.summary.get("resized_copies_skipped", 0) + 1
        for u in redundant_scale_renders([(u, v[2]) for u, v in cands.items()]):
            del cands[u]
            self.summary["scale_renders_skipped"] = self.summary.get("scale_renders_skipped", 0) + 1
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
                     source_type: str, extra: Dict[str, Any], classification: Optional[Dict[str, Any]],
                     document_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """`document_id` is the PDF the page number belongs to: a crop linked from another catalogue keeps its own."""
        document_id = document_id or self.document_id
        data = _jpeg(img)
        up = await self.sb.upload_image_file(data, f"{layer}_{uuid.uuid4().hex[:12]}.jpg", document_id, page_number)
        if not up.get("success"):
            self.summary["errors"].append(f"upload failed for {product['name']}")
            return None
        cls = classification or {}
        row = {
            "document_id": document_id,
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
                # A classical repair adds no detail, so it keeps the source's verdict and real pixels.
                "image_quality": ({"verdict": "low_res", "native_px": extra.get("source_native_px")}
                                  if extra.get("restore_method") == "classical"
                                  else {"verdict": "ok", "native_px": [img.width, img.height]}),
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
        from app.services.embeddings.embedding_deferral import embeddings_deferred
        if embeddings_deferred():
            return False
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
        """Through the edge function (its key, its credit debit), padded to a ratio the model keeps.

        When it cannot run, `self._gemini_failure` says why, so the fallback is reported as degraded.
        """
        self._gemini_failure = None
        if not self.user_id:
            self._gemini_failure = "nobody to bill"
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
            self._gemini_failure = "upload of the crop failed"
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
                self._gemini_failure = f"HTTP {r.status_code}: {str(body.get('error'))[:80]}"
                return None
            out = _open(await _download(url) or b"")
            if out is None:
                self._gemini_failure = "result could not be downloaded"
            if out is not None and box:
                W, H = out.size
                out = out.crop((round(box[0] * W), round(box[1] * H), round(box[2] * W), round(box[3] * H)))
            return out
        except httpx.HTTPError as e:
            self._gemini_failure = type(e).__name__
            return None
        finally:
            try:
                self.db.storage.from_("pdf-tiles").remove([tmp["storage_path"]])
            except Exception:
                pass

    async def _repair(self, product: Dict[str, Any], crop: Dict[str, Any], crop_img: Image.Image, crop_bytes: bytes) -> None:
        native = (crop.get("metadata") or {}).get("image_quality", {}).get("native_px")
        restored, method, fidelity = None, None, None
        # A swatch stays the size the PDF holds; a repair has nothing at that size to recover.
        repairable = self.allow_restore and not _is_swatch(crop)
        gemini_failed = None
        if repairable and is_patterned(crop_img):
            g = await self._gemini_restore(crop_img)
            gemini_failed = self._gemini_failure
            if g is not None:
                ok, fidelity = repair_is_faithful(crop_img, g, native)
                if ok:
                    restored, method = g, "gemini-3.1-flash-image"
                else:
                    self.log.info(f"   ✖ Gemini repair redrew {crop['id']} (fidelity {fidelity}) — using the classical clean-up")
        if restored is None and repairable:
            cleaned = classical_cleanup(crop_img)
            ok, fidelity = repair_is_faithful(crop_img, cleaned, native)
            # Same gate as Gemini: a clean-up that is no longer the same picture is not kept.
            if ok:
                restored, method = cleaned, "classical"
            else:
                self.log.info(f"   ✖ Classical clean-up changed {crop['id']} too much (fidelity {fidelity}) — keeping the source")
        if restored is None:
            self._mark(crop, "kept_source")
            if not crop.get("has_slig_embedding"):
                await self._embed(crop, crop_bytes)
            self.summary["kept_source"] += 1
            self.summary["kept_source_ids"].append(crop["id"])
            return
        from app.services.pdf.ingestion_audit import UPSCALE_LIMIT
        before = restored.size
        restored = within_upscale(restored, native, UPSCALE_LIMIT)
        if restored.size != before:
            self.summary["repairs_downscaled"].append(crop["id"])
        cls = (crop.get("metadata") or {}).get("ai_classification")
        saved = await self._store(product, restored, crop.get("page_number") or 1, "restored", "pdf_processing",
                                  {"restored_from": crop["id"], "restore_method": method, "restore_fidelity": fidelity,
                                   "source_native_px": native}, cls, crop.get("document_id"))
        if not saved:
            self._mark(crop, "kept_source")
            if not crop.get("has_slig_embedding"):
                await self._embed(crop, crop_bytes)
            self.summary["kept_source"] += 1
            self.summary["kept_source_ids"].append(crop["id"])
            return
        self._supersede(crop, product["id"], [saved["id"]], "superseded_by_restored")
        await self._retire_vectors(crop)
        await self._embed(saved, _jpeg(restored))
        self.summary["restored"] += 1
        self.summary["restored_ids"].append(crop["id"])
        if gemini_failed and method == "classical":
            self.summary["restored_without_gemini"].append(crop["id"])
            self.summary["gemini_unavailable_reasons"].append(gemini_failed)

    # ── run ─────────────────────────────────────────────────────────────────

    async def run_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        crops = self._pending_crops(product["id"])
        report: Dict[str, Any] = {"product": product["name"], "pending": len(crops), "matched": 0, "added": 0, "restored_or_kept": 0}
        imageless = self.low_res_embedded and not crops and not self._has_images(product["id"])
        backfill = self.gallery_backfill and not crops and not self._has_layer(product["id"], "manufacturer_original")
        if not crops and not imageless and not backfill:
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
                crop = next((c for c, _, _ in loaded if how and c["id"] == how["crop"]), None)
                page_no, crop_doc = (crop.get("page_number"), crop.get("document_id")) if crop else (None, None)
                confirmation = {"mode": "confirmed_page", "confirmed_by": sorted(matches)} if matches \
                    else {"mode": "page_slug_is_product_name"}
                saved = await self._store(product, img, page_no or 1, "manufacturer_original", "web_scraping",
                                          {"source_url": u, "source_page": page, "match": how or confirmation},
                                          verdict, crop_doc)
                if saved:
                    stored[u] = saved["id"]
                    await self._embed(saved, data)
                    self.summary["manufacturer_images"] += 1
                    report["added"] += 1

        gallery = [(gid, cands[u][2]) for u, gid in stored.items() if u in cands]
        if gallery:
            await self._retire_shown_by_gallery(product, gallery, {c["id"] for c, _, _ in loaded})
        for c, data, img in loaded:
            hit_ids = [stored[u] for u, _ in matches.get(c["id"], []) if u in stored]
            if hit_ids:
                self._supersede(c, product["id"], hit_ids, "superseded_by_manufacturer_original")
                await self._retire_vectors(c)
            else:
                await self._repair(product, c, img, data)
                report["restored_or_kept"] += 1
        return report

    async def _give_way_to_gallery(self, row: Dict[str, Any], product_id: str, gallery_id: str, judged_by: str) -> None:
        self.gallery_pairs.append({"product": product_id, "drop": row["id"], "drop_url": row.get("image_url"),
                                   "keep": gallery_id, "judged_by": judged_by})
        if self.dry_run:
            return
        self._supersede(row, product_id, [gallery_id], "superseded_by_manufacturer_gallery")
        await self._retire_vectors(row)

    async def _retire_shown_by_gallery(self, product: Dict[str, Any], gallery: List[Tuple[str, Image.Image]],
                                       skip_ids: Set[str]) -> int:
        """Retire small or smoothed images only where a gallery photo SHOWS THE SAME THING. Returns the count."""
        gallery_thumbs = [(gid, _thumb(img)) for gid, img in gallery]
        full = dict(gallery)

        def replaceable(r: Dict[str, Any], img: Image.Image, gid: str) -> bool:
            smoothed = r.get("extraction_layer") == "restored"
            return gallery_can_replace(img, full[gid], _source_long_side(r) or max(img.size), smoothed)

        unmatched: List[Tuple[Dict[str, Any], Image.Image]] = []
        retired = 0
        for r in self._linked(product["id"]):
            small_pdf = r.get("extraction_layer") in ("region_crop", "embedded") and _small_beside_gallery(r)
            smoothed = r.get("extraction_layer") == "restored" and (r.get("metadata") or {}).get("restore_method") == "classical"
            if r["id"] in skip_ids or not (small_pdf or smoothed):
                continue
            twin, img = await self._gallery_twin(r, gallery_thumbs)
            if twin and replaceable(r, img, twin):
                await self._give_way_to_gallery(r, product["id"], twin, "pixels")
                retired += 1
            elif img is not None and not twin:
                unmatched.append((r, img))
        # Pixels cannot compare a square swatch with a landscape room photo; a vision check can.
        if unmatched:
            by_id = {r["id"]: (r, img) for r, img in unmatched}
            for rid, gid in (await self._gallery_covers(product, unmatched, gallery)).items():
                r, img = by_id[rid]
                if replaceable(r, img, gid):
                    await self._give_way_to_gallery(r, product["id"], gid, "vision")
                    retired += 1
        return retired

    async def recheck_gallery(self, product_ids: Optional[List[str]] = None) -> Dict[str, int]:
        """For products sourced earlier: retire what their stored gallery already shows."""
        out: Dict[str, int] = {}
        for product in self._products(product_ids):
            rows = (self.db.table("image_product_associations")
                    .select("document_images!inner(id,image_url,extraction_layer,metadata)")
                    .eq("product_id", product["id"]).eq("document_images.extraction_layer", "manufacturer_original")
                    .execute().data or [])
            gallery = []
            for row in rows:
                di = row["document_images"]
                if "excluded_from_products" in (di.get("metadata") or {}):
                    continue
                data = await _download(di.get("image_url") or "")
                img = _open(data) if data else None
                if img is not None:
                    gallery.append((di["id"], img))
            if gallery:
                out[product["id"]] = await self._retire_shown_by_gallery(product, gallery, set())
        return out

    async def run_all(self, product_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        reports = []
        analysed: List[str] = []
        products = self._products(product_ids)
        for n, product in enumerate(products, 1):
            self.summary["products"] += 1
            try:
                reports.append(await self.run_product(product))
                r = reports[-1]
                if r.get("added"):
                    analysed.append(product["id"])
                self.log.info(f"🖼️ Image sourcing {n}/{len(products)} {product['name']}: {r.get('pending', 0)} low-res, "
                              f"{r.get('matched', 0)} matched, {r.get('added', 0)} manufacturer images added")
            except Exception as e:
                self.log.exception(f"image sourcing failed for {product.get('name')}")
                self.summary["errors"].append(f"{product.get('name')}: {str(e)[:160]}")
            await asyncio.sleep(0)
        self.summary["per_product"] = reports
        from app.services.embeddings.embedding_deferral import embeddings_deferred
        if analysed and not self.dry_run and not embeddings_deferred():
            from app.services.embeddings.document_embed_run import reroll_products
            self.summary["rollup"] = await reroll_products(self.document_id, analysed)
        if not self.dry_run:
            from app.services.pdf.ingestion_corrections import record, sourcing_corrections
            await record(self.db, self.document_id, self.workspace_id, "image_sourcing",
                         sourcing_corrections(self.summary), self.job_id)
        return self.summary


async def redo_repairs_with_gemini(document_id: str, workspace_id: str, user_id: Optional[str] = None,
                                   log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Redo a document's plain clean-ups of patterned crops with Gemini, once it is available again.

    Stops at the first failure to run: the next call would fail the same way and nothing is lost.
    A redraw keeps the clean-up. A replaced repair keeps its row, excluded, like any superseded image.
    """
    from app.services.pdf.ingestion_audit import UPSCALE_LIMIT
    from app.services.pdf.ingestion_corrections import CORRECTED, DEGRADED, Correction, record

    s = ImageSourcing(document_id, workspace_id, user_id, log=log)
    db = s.db
    cols = "id, document_id, workspace_id, page_number, image_url, metadata, has_slig_embedding, extraction_layer"
    rows: List[Dict[str, Any]] = []
    while True:
        batch = (db.table("document_images").select(cols).eq("document_id", document_id).eq("extraction_layer", "restored")
                 .order("id").range(len(rows), len(rows) + 999).execute().data or [])
        rows.extend(batch)
        if len(batch) < 1000:
            break
    out: Dict[str, Any] = {"considered": 0, "replaced": [], "kept_redrawn": [], "failed_to_run": None}
    for old in rows:
        meta = old.get("metadata") or {}
        if meta.get("restore_method") != "classical" or "excluded_from_products" in meta:
            continue
        found = db.table("document_images").select(cols).eq("id", meta.get("restored_from")).limit(1).execute().data
        if not found:
            continue
        source = found[0]
        src_meta = source.get("metadata") or {}
        src_img = _open(db.storage.from_(src_meta["storage_bucket"]).download(src_meta["storage_path"]))
        if src_img is None or not is_patterned(src_img):
            continue
        out["considered"] += 1
        repaired = await s._gemini_restore(src_img)
        if repaired is None:
            out["failed_to_run"] = s._gemini_failure or "unknown"
            break
        native = (src_meta.get("image_quality") or {}).get("native_px")
        ok, fidelity = repair_is_faithful(src_img, repaired, native)
        if not ok:
            out["kept_redrawn"].append(old["id"])
            continue
        repaired = within_upscale(repaired, native, UPSCALE_LIMIT)
        links = db.table("image_product_associations").select("product_id").eq("image_id", old["id"]).execute().data or []
        products = (db.table("products").select("id, name, workspace_id").in_("id", [r["product_id"] for r in links]).execute().data or []) if links else []
        if not products:
            continue
        saved = await s._store(products[0], repaired, old.get("page_number") or 1, "restored", "pdf_processing",
                               {"restored_from": source["id"], "restore_method": "gemini-3.1-flash-image", "restore_fidelity": fidelity,
                                "source_native_px": native, "replaces_repair": old["id"]}, meta.get("ai_classification"), old["document_id"])
        if not saved:
            continue
        for extra in products[1:]:
            db.table("image_product_associations").upsert(
                {"workspace_id": extra["workspace_id"], "product_id": extra["id"], "image_id": saved["id"],
                 "overall_score": PAGE_LINK_SCORE, "metadata": {"association_method": "restored"}},
                on_conflict="image_id,product_id").execute()
        for product in products:
            s._supersede(old, product["id"], [saved["id"]], "superseded_by_gemini_repair")
        await s._retire_vectors(old)
        await s._embed(saved, _jpeg(repaired))
        out["replaced"].append(old["id"])
    corrections = []
    if out["replaced"]:
        corrections.append(Correction("images_repaired", CORRECTED, f"{len(out['replaced'])} plain clean-up(s) redone with Gemini",
                                      len(out["replaced"]), out["replaced"]))
    if out["kept_redrawn"]:
        corrections.append(Correction("images_repaired_without_gemini", DEGRADED,
                                      f"{len(out['kept_redrawn'])} Gemini repair(s) redrew the picture, so the plain clean-up stays",
                                      len(out["kept_redrawn"]), out["kept_redrawn"]))
    if out["failed_to_run"]:
        corrections.append(Correction("gemini_redo_stopped", DEGRADED,
                                      f"The Gemini redo stopped: {out['failed_to_run'][:160]}; run it again once Gemini is available", 1))
    await record(db, document_id, workspace_id, "image_sourcing", corrections)
    return out


async def run_image_sourcing(document_id: str, workspace_id: str, user_id: Optional[str] = None,
                             product_ids: Optional[List[str]] = None, allow_restore: bool = True,
                             log: Optional[logging.Logger] = None, dry_run: bool = False,
                             low_res_embedded: bool = False, gallery_backfill: bool = False,
                             job_id: Optional[str] = None) -> Dict[str, Any]:
    return await ImageSourcing(document_id, workspace_id, user_id, allow_restore, log, dry_run,
                               low_res_embedded, gallery_backfill, job_id).run_all(product_ids)
