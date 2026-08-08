"""
Page embeddings (#239) — the 8th fusion vector.

WHAT THIS IS FOR
----------------
PaddleOCR-VL reads a page structurally: it classifies regions and OCRs the textual
ones. `Image` / `Figure` / `chart` regions are treated as CROP SOURCES — they become
`document_images` rows, but the text baked into their pixels is never read. So a
catalog page whose product name is set in type across a photograph produces chunks
that do not contain that name, and no amount of text search will ever find it.

A page embedding closes that specific hole. The whole rendered page — picture and
words together — goes to `voyage-multimodal-3(.5)` as ONE input and comes back as one
1024D vector, so "the blue ceramic tile page" and a product name that exists only as
pixels are both reachable. It lands in `vecs.page_embeddings` and is scored as the
`page` channel in fusion search.

LAYERING (Medallion)
--------------------
The page TEXT comes from `document_chunks` — silver. We do not re-extract text from
the PDF; silver already holds it, and re-deriving from bronze is the layer violation
that has caused every cache bug in this pipeline.

The page IMAGE is necessarily a bronze read: no silver artifact holds a page raster,
because until now nothing needed one. So the render is cached as an artifact
(`pdf-tiles/extracted/<document_id>/pages/`) and recorded in
`document_page_embeddings`, which makes a re-embed (new model, changed weights) a
silver→gold rebuild rather than another trip to the PDF.

BOUNDING
--------
Every page of every catalog is a per-page API call, so all three dimensions are
capped: pages per document (`page_embedding_max_pages`), concurrency
(`page_embedding_concurrency`), and DPI (`page_embedding_dpi`, sized to Voyage's
2M-pixel billing ceiling — see config). At ~$0.0012/page a 500-page catalog is ~$0.62.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Bumped when the page payload's construction changes (what text is included, how the
#: image is rendered) so the backfill can tell a stale vector from a current one. This
#: is NOT the model version — `embedding_model` carries that separately, because the
#: two drift independently.
PAGE_EMBEDDING_SCHEMA_VERSION = 1

#: Rendered pages live beside the other per-document artifacts. The prefix matters:
#: `build_storage_reference_set()` protects `pdf-tiles/extracted/<document_id>/` and
#: `document_page_embeddings` is registered explicitly as well, so these files survive
#: the orphan cron on both counts.
PAGE_RENDER_BUCKET = "pdf-tiles"

#: A page whose render is essentially blank carries no signal, and embedding it would
#: spend a call to store a vector that matches everything weakly. Measured on the
#: PNG's own byte size, which is a good blankness proxy: a uniform page compresses to
#: almost nothing regardless of its pixel dimensions.
MIN_RENDER_BYTES = 3_000

#: Voyage truncates at 32k tokens per input, and the image consumes most of the
#: budget. Cap the text so a text-dense page cannot push the IMAGE out of the payload
#: — losing the image would silently turn this into a worse text embedding.
MAX_PAGE_TEXT_CHARS = 8_000


def page_storage_path(document_id: str, page_number: int) -> str:
    """Storage path for a page render. One definition — see the prefix note above."""
    return f"extracted/{document_id}/pages/page-{int(page_number):04d}.png"


class PageEmbeddingService:
    """Renders catalog pages and embeds them as the `page` fusion channel."""

    def __init__(self, supabase_client=None, embeddings_service=None, vecs_service=None):
        self._supabase = supabase_client
        self._embeddings = embeddings_service
        self._vecs = vecs_service

    # ── lazily-resolved collaborators ───────────────────────────────────────────

    @property
    def supabase(self):
        if self._supabase is None:
            from app.services.core.supabase_client import get_supabase_client
            self._supabase = get_supabase_client()
        return self._supabase

    @property
    def embeddings(self):
        if self._embeddings is None:
            from app.config import settings
            from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
            self._embeddings = RealEmbeddingsService(config=settings)
        return self._embeddings

    @property
    def vecs(self):
        if self._vecs is None:
            from app.services.embeddings.vecs_service import get_vecs_service
            self._vecs = get_vecs_service()
        return self._vecs

    # ── public entry point ──────────────────────────────────────────────────────

    async def embed_document_pages(
        self,
        document_id: str,
        workspace_id: str,
        job_id: Optional[str] = None,
        page_numbers: Optional[List[int]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Render + embed a document's pages.

        Args:
            document_id: the document to process.
            workspace_id: the caller's workspace. Checked AGAINST the document row —
                see below; it is not trusted as the source of tenancy.
            page_numbers: 1-based pages to process. None = all pages, capped.
            force: re-embed pages already marked `embedded` (use after a model change).

        Returns a summary dict. Never raises for per-page failures: a page that fails
        is recorded with cache_status='failed' so the backfill can retry it, and the
        rest of the document still gets embedded.
        """
        from app.config import settings

        summary: Dict[str, Any] = {
            "document_id": document_id,
            "enabled": bool(settings.page_embeddings_enabled),
            "pages_considered": 0,
            "embedded": 0,
            "skipped_blank": 0,
            "skipped_existing": 0,
            "failed": 0,
            "model": settings.voyage_multimodal_model,
        }

        if not settings.page_embeddings_enabled:
            logger.info("⏭️ Page embeddings disabled (PAGE_EMBEDDINGS_ENABLED=false)")
            return summary

        doc = await self._load_document(document_id, workspace_id)
        if not doc:
            summary["error"] = "document_not_found_or_workspace_mismatch"
            return summary

        pdf_bytes = await self._download_pdf(doc)
        if not pdf_bytes:
            summary["error"] = "source_pdf_unavailable"
            return summary

        try:
            import fitz  # PyMuPDF — already a pipeline dependency
        except ImportError:
            logger.error("❌ PyMuPDF unavailable — cannot render pages")
            summary["error"] = "pymupdf_unavailable"
            return summary

        try:
            pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            logger.error(f"❌ Could not open PDF for document {document_id}: {e}")
            summary["error"] = f"invalid_pdf: {e}"
            return summary

        try:
            total_pages = len(pdf)
            targets = self._resolve_target_pages(total_pages, page_numbers, settings.page_embedding_max_pages)
            summary["pages_considered"] = len(targets)
            if not targets:
                return summary

            already = set() if force else await self._already_embedded(document_id)
            todo = [p for p in targets if p not in already]
            summary["skipped_existing"] = len(targets) - len(todo)
            if not todo:
                logger.info(f"✅ All {len(targets)} pages of {document_id} already embedded")
                return summary

            page_texts = await self._page_texts_from_silver(document_id, todo)

            logger.info(
                f"📄 Page embeddings: {len(todo)} page(s) of document {document_id} "
                f"@ {settings.page_embedding_dpi}dpi, model={settings.voyage_multimodal_model}"
            )

            semaphore = asyncio.Semaphore(max(1, settings.page_embedding_concurrency))
            # PyMuPDF is NOT thread-safe: a single fitz.Document must be touched by one
            # thread at a time, and every render here runs in a worker thread against
            # the SAME document. Without this lock, concurrency > 1 races inside the C
            # layer — which does not raise cleanly, it corrupts or segfaults.
            #
            # Serializing renders costs almost nothing: a page rasterizes in tens of
            # milliseconds while the Voyage round-trip takes seconds, so the
            # concurrency that matters (upload + embed) is untouched.
            render_lock = asyncio.Lock()

            async def _one(page_no: int) -> str:
                async with semaphore:
                    return await self._embed_one_page(
                        pdf=pdf,
                        render_lock=render_lock,
                        document_id=document_id,
                        workspace_id=doc["workspace_id"],
                        page_number=page_no,
                        page_text=page_texts.get(page_no),
                        dpi=settings.page_embedding_dpi,
                        job_id=job_id,
                    )

            outcomes = await asyncio.gather(*[_one(p) for p in todo], return_exceptions=True)

            for outcome in outcomes:
                if isinstance(outcome, Exception):
                    # gather() with return_exceptions still surfaces genuine bugs here;
                    # count them as failures rather than letting one kill the batch.
                    logger.error(f"❌ Page embedding raised: {outcome}")
                    summary["failed"] += 1
                elif outcome == "embedded":
                    summary["embedded"] += 1
                elif outcome == "skipped":
                    summary["skipped_blank"] += 1
                else:
                    summary["failed"] += 1

            logger.info(
                f"✅ Page embeddings for {document_id}: {summary['embedded']} embedded, "
                f"{summary['skipped_blank']} blank, {summary['failed']} failed, "
                f"{summary['skipped_existing']} already present"
            )
            return summary

        finally:
            pdf.close()

    # ── steps ───────────────────────────────────────────────────────────────────

    async def _load_document(self, document_id: str, workspace_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the document and verify the caller's workspace owns it.

        Tenancy comes from the ROW, not from the argument (security invariant 1). The
        argument is only used to detect a mismatch — and a mismatch returns None, so a
        caller passing someone else's document_id gets "not found" rather than a
        stamped-with-my-workspace vector over another tenant's catalog.
        """
        try:
            resp = (
                self.supabase.client.table("documents")
                .select("id, workspace_id, storage_bucket, storage_object_path, metadata")
                .eq("id", document_id)
                .maybe_single()
                .execute()
            )
            row = resp.data if resp else None
            if not row:
                logger.warning(f"⚠️ Page embeddings: document {document_id} not found")
                return None

            if workspace_id and str(row.get("workspace_id")) != str(workspace_id):
                logger.error(
                    f"❌ Page embeddings: workspace mismatch for document {document_id} "
                    f"(caller={workspace_id}, owner={row.get('workspace_id')}) — refusing"
                )
                return None

            if not row.get("workspace_id"):
                # Without it the write invariant would reject every vector anyway;
                # failing here says why.
                logger.error(f"❌ Page embeddings: document {document_id} has no workspace_id")
                return None

            return row
        except Exception as e:
            logger.error(f"❌ Page embeddings: failed to load document {document_id}: {e}")
            return None

    async def _download_pdf(self, doc: Dict[str, Any]) -> Optional[bytes]:
        """Download the source PDF from (storage_bucket, storage_object_path)."""
        bucket = doc.get("storage_bucket") or "pdf-documents"
        path = doc.get("storage_object_path")
        if not path:
            logger.warning(
                f"⚠️ Page embeddings: document {doc.get('id')} has no storage_object_path — "
                f"cannot render pages"
            )
            return None
        try:
            return await asyncio.to_thread(
                self.supabase.client.storage.from_(bucket).download, path
            )
        except Exception as e:
            logger.error(f"❌ Page embeddings: PDF download failed ({bucket}/{path}): {e}")
            return None

    @staticmethod
    def _resolve_target_pages(
        total_pages: int,
        requested: Optional[List[int]],
        max_pages: int,
    ) -> List[int]:
        """1-based page list, de-duplicated, in range, and capped.

        The cap is logged rather than applied silently — a document quietly embedding
        its first 500 pages and reporting success is the "no silent caps" rule's exact
        target.
        """
        if requested:
            pages = sorted({int(p) for p in requested if 1 <= int(p) <= total_pages})
        else:
            pages = list(range(1, total_pages + 1))

        if max_pages and len(pages) > max_pages:
            logger.warning(
                f"⚠️ Page embeddings: capping at {max_pages} of {len(pages)} pages "
                f"(PAGE_EMBEDDING_MAX_PAGES). Pages {pages[max_pages]}–{pages[-1]} "
                f"will have NO page vector."
            )
            pages = pages[:max_pages]
        return pages

    async def _already_embedded(self, document_id: str) -> set:
        """Pages with a current vector. 'skipped' counts too — a blank page is a
        decided outcome, not a pending one, and retrying it every run would spend
        money to re-learn the same thing."""
        try:
            resp = (
                self.supabase.client.table("document_page_embeddings")
                .select("page_number, cache_status, schema_version")
                .eq("document_id", document_id)
                .in_("cache_status", ["embedded", "skipped"])
                .execute()
            )
            return {
                int(r["page_number"])
                for r in (resp.data or [])
                # A row written under an older payload schema is stale by definition;
                # let it be re-embedded rather than trusting a vector built differently.
                if int(r.get("schema_version") or 0) >= PAGE_EMBEDDING_SCHEMA_VERSION
            }
        except Exception as e:
            logger.warning(f"⚠️ Page embeddings: could not read existing rows: {e}")
            return set()

    async def _page_texts_from_silver(
        self, document_id: str, pages: List[int]
    ) -> Dict[int, str]:
        """Page text from `document_chunks` — SILVER, not re-extracted from the PDF.

        Chunks carry their page in `metadata.page_number` (1-based). Chunks with no
        page number are dropped rather than guessed at: attaching a chunk to the wrong
        page would put text into a vector that does not describe that page's picture,
        which is worse than having no text at all.
        """
        texts: Dict[int, List[Tuple[int, str]]] = {}
        try:
            resp = (
                self.supabase.client.table("document_chunks")
                .select("content, chunk_index, metadata")
                .eq("document_id", document_id)
                .execute()
            )
            wanted = set(pages)
            for row in (resp.data or []):
                meta = row.get("metadata") or {}
                raw_page = meta.get("page_number")
                try:
                    page_no = int(raw_page)
                except (TypeError, ValueError):
                    continue
                if page_no not in wanted:
                    continue
                content = (row.get("content") or "").strip()
                if not content:
                    continue
                texts.setdefault(page_no, []).append((int(row.get("chunk_index") or 0), content))
        except Exception as e:
            logger.warning(
                f"⚠️ Page embeddings: chunk text lookup failed ({e}); "
                f"embedding pages image-only"
            )
            return {}

        out: Dict[int, str] = {}
        for page_no, parts in texts.items():
            joined = "\n".join(text for _, text in sorted(parts))
            out[page_no] = joined[:MAX_PAGE_TEXT_CHARS]
        return out

    def _render_page(self, pdf, page_number: int, dpi: int) -> Optional[Tuple[bytes, int, int]]:
        """Render one page to PNG. Synchronous — callers run it off the event loop."""
        import fitz

        page = pdf[page_number - 1]
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png"), pix.width, pix.height

    async def _embed_one_page(
        self,
        pdf,
        render_lock: asyncio.Lock,
        document_id: str,
        workspace_id: str,
        page_number: int,
        page_text: Optional[str],
        dpi: int,
        job_id: Optional[str],
    ) -> str:
        """Render → upload → embed → upsert one page. Returns the cache_status written."""
        try:
            # Lock held for the render ONLY (see the note at the call site: PyMuPDF is
            # not thread-safe). Everything after this — upload and embed, the slow
            # parts — runs concurrently.
            async with render_lock:
                rendered = await asyncio.to_thread(self._render_page, pdf, page_number, dpi)
        except Exception as e:
            logger.error(f"❌ Page {page_number} render failed: {e}")
            await self._record(document_id, workspace_id, page_number, "failed", error=str(e))
            return "failed"

        if not rendered:
            await self._record(document_id, workspace_id, page_number, "failed", error="render returned nothing")
            return "failed"

        png_bytes, width, height = rendered

        if len(png_bytes) < MIN_RENDER_BYTES and not (page_text or "").strip():
            # Blank AND textless. 'skipped' rather than 'failed': there is nothing to
            # retry, and marking it failed would make the backfill re-render it forever.
            logger.debug(f"⏭️ Page {page_number} of {document_id} is blank — skipping")
            await self._record(
                document_id, workspace_id, page_number, "skipped",
                width=width, height=height,
            )
            return "skipped"

        storage_path = page_storage_path(document_id, page_number)
        uploaded = await self._upload_render(storage_path, png_bytes)

        result = await self.embeddings.generate_page_embedding(
            image_base64=base64.b64encode(png_bytes).decode("ascii"),
            page_text=page_text,
            job_id=job_id,
        )
        if not result or not result.get("embedding"):
            await self._record(
                document_id, workspace_id, page_number, "failed",
                storage_path=storage_path if uploaded else None,
                width=width, height=height,
                error="embedding provider returned nothing",
            )
            return "failed"

        ok = await self.vecs.upsert_page_embedding(
            document_id=document_id,
            page_number=page_number,
            embedding=result["embedding"],
            metadata={
                "workspace_id": str(workspace_id),
                "storage_bucket": PAGE_RENDER_BUCKET if uploaded else None,
                "storage_object_path": storage_path if uploaded else None,
                "has_text": bool((page_text or "").strip()),
            },
            embedding_model=result.get("embedding_model"),
            schema_version=PAGE_EMBEDDING_SCHEMA_VERSION,
        )
        if not ok:
            # The vector is what makes the page searchable, so the row must not claim
            # 'embedded' when the upsert was refused — that gap is what makes a flag
            # diverge from VECS and a page invisible forever.
            await self._record(
                document_id, workspace_id, page_number, "failed",
                storage_path=storage_path if uploaded else None,
                width=width, height=height,
                error="vecs upsert refused",
            )
            return "failed"

        await self._record(
            document_id, workspace_id, page_number, "embedded",
            storage_path=storage_path if uploaded else None,
            width=width, height=height,
            embedding_model=result.get("embedding_model"),
            text_tokens=result.get("text_tokens"),
            image_pixels=result.get("image_pixels"),
        )
        return "embedded"

    async def _upload_render(self, storage_path: str, png_bytes: bytes) -> bool:
        """Cache the render. Non-fatal: the embedding is the deliverable, and a page we
        embedded but could not cache is still searchable — it just costs a re-render if
        we ever re-embed."""
        try:
            await asyncio.to_thread(
                self.supabase.client.storage.from_(PAGE_RENDER_BUCKET).upload,
                storage_path, png_bytes,
                {"content-type": "image/png", "upsert": "true"},
            )
            return True
        except Exception:
            try:
                await asyncio.to_thread(
                    self.supabase.client.storage.from_(PAGE_RENDER_BUCKET).update,
                    storage_path, png_bytes, {"content-type": "image/png"},
                )
                return True
            except Exception as e:
                logger.warning(f"⚠️ Page render upload failed ({storage_path}): {e}")
                return False

    async def _record(
        self,
        document_id: str,
        workspace_id: str,
        page_number: int,
        cache_status: str,
        storage_path: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        embedding_model: Optional[str] = None,
        text_tokens: Optional[int] = None,
        image_pixels: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Upsert the serving row. Explicit payload, never a spread of caller input."""
        payload: Dict[str, Any] = {
            "document_id": str(document_id),
            "workspace_id": str(workspace_id),
            "page_number": int(page_number),
            "cache_status": cache_status,
            "schema_version": PAGE_EMBEDDING_SCHEMA_VERSION,
            "storage_bucket": PAGE_RENDER_BUCKET if storage_path else None,
            "storage_object_path": storage_path,
            "image_width": width,
            "image_height": height,
            "embedding_model": embedding_model,
            "text_tokens": text_tokens,
            "image_pixels": image_pixels,
            "error_message": (error or None) and str(error)[:500],
        }
        try:
            await asyncio.to_thread(
                lambda: self.supabase.client.table("document_page_embeddings")
                .upsert(payload, on_conflict="document_id,page_number")
                .execute()
            )
        except Exception as e:
            logger.warning(
                f"⚠️ Could not record page-embedding status for {document_id} "
                f"p{page_number} ({cache_status}): {e}"
            )


_page_embedding_service: Optional[PageEmbeddingService] = None


def get_page_embedding_service() -> PageEmbeddingService:
    """Global instance, matching the vecs/embeddings service convention."""
    global _page_embedding_service
    if _page_embedding_service is None:
        _page_embedding_service = PageEmbeddingService()
    return _page_embedding_service
