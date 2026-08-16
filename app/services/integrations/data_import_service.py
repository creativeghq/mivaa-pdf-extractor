"""
Data Import Service — process XML import jobs into products.

- Batch processing (10 products at a time)
- Image downloads (5 concurrent)
- Metadata extraction + product normalization
- Chunking + inline text embeddings (Voyage AI)
- Image classification + SLIG embeddings (understanding embeddings generated
  inline when vision_analysis is available — no queue/worker step)
- Checkpoint recovery + real-time progress
"""

import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer from the environment with a fallback default."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default
from app.services.core.supabase_client import get_supabase_client
from app.services.tracking.checkpoint_recovery_service import checkpoint_recovery_service, ProcessingStage
from app.services.tracking.xml_import_stages import XmlImportStage, get_xml_import_progress
from app.services.images.image_download_service import ImageDownloadService
from app.services.images.image_processing_service import ImageProcessingService
from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
from app.services.chunking.unified_chunking_service import UnifiedChunkingService, ChunkingConfig, ChunkingStrategy
from app.services.metadata.metadata_normalizer import normalize_factory_keys
from app.services.facets import canonicalize_product_attributes
from app.services.products.material_quota import material_quota_remaining_async
import sentry_sdk
from app.utils.text_fold import fold_model_token

# Category → default unit mapping (mirrors material_categories.default_unit)
_CATEGORY_DEFAULT_UNITS = {
    'tiles': 'sqm', 'wood': 'sqm', 'paint_wall_decor': 'sqm',
    'decor': 'pcs', 'furniture': 'pcs', 'general_materials': 'pcs',
    'heating': 'pcs', 'sanitary': 'pcs', 'kitchen': 'pcs', 'lighting': 'pcs',
}

logger = logging.getLogger(__name__)


#: Keys a supplier feed may never set: identity, tenancy, provenance, and the two fields the
#: platform derives rather than accepts. Invariant 8 wants an allowlisted payload; the
#: product_metadata literal below IS that allowlist, and this set is what stops the feed's own
#: blob from being spread over it. Kept narrow on purpose — a supplier sending `finish` or
#: `pei_rating` is exactly what an import is for. (audit #17 M4-5)
_SUPPLIER_RESERVED_METADATA_KEYS = frozenset({
    'workspace_id', 'user_id', 'created_by', 'owner_id',
    'id', 'product_id_internal', 'source_job_id', 'import_job_id', 'import_batch_id',
    'extracted_from', 'extraction_date', 'source_type', 'created_from_type',
    'status', 'is_verified', 'is_locked', 'credits',
    'external_sku_folded',
})


class DataImportService:
    """Service for processing data import jobs"""

    def __init__(self, workspace_id: str = None):
        supabase_wrapper = get_supabase_client()
        self.supabase = supabase_wrapper.client   # sync client (legacy, kept for compatibility)
        self.db = supabase_wrapper.async_client   # async façade — use in all async methods
        self.image_downloader = ImageDownloadService()
        # 5-product batches keep peak memory low (≤25 images in flight per batch)
        # and avoid piling requests against cold HuggingFace endpoints. Tunable
        # via env vars for ops flexibility without redeploy.
        self.batch_size = _env_int('XML_IMPORT_BATCH_SIZE', 5)
        self.max_concurrent_images = _env_int('XML_IMPORT_MAX_CONCURRENT_IMAGES', 5)
        # Retained for back-compat with callers that pass it, and deliberately NOT read as
        # the tenant anywhere: every method that needs a workspace takes one as an argument.
        # Reading this instead is audit #17 M4-8, and it was None at both call sites.
        self.workspace_id = workspace_id

        # ✅ Initialize chunking service for quality scoring and smart chunking
        self.chunking_service = UnifiedChunkingService(ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID,
            max_chunk_size=1000,
            min_chunk_size=100,
            overlap_size=100
        ))

        # Image processor for classification and SLIG embeddings (lazy init)
        self.image_processor = None

        # Embedding service for inline text-embedding generation
        self.embedding_service = RealEmbeddingsService()

    def _get_image_processor(self, workspace_id: str) -> ImageProcessingService:
        """Get or create image processor for workspace."""
        if not self.image_processor or self.image_processor.workspace_id != workspace_id:
            self.image_processor = ImageProcessingService(workspace_id=workspace_id)
        return self.image_processor

    async def _log_stage(
        self,
        job_id: str,
        stage: XmlImportStage,
        status: str = "started",
        duration_ms: int = None,
        items: int = None,
        error: str = None
    ) -> None:
        """
        Log stage progress with structured format for debugging.

        Args:
            job_id: Job ID
            stage: Current processing stage
            status: 'started', 'completed', or 'failed'
            duration_ms: Duration in milliseconds (for completed stages)
            items: Number of items processed (for completed stages)
            error: Error message (for failed stages)
        """
        if status == "started":
            logger.info(f"[{job_id}] Stage: {stage.value} | Status: started")
        elif status == "completed":
            msg = f"[{job_id}] Stage: {stage.value} | Status: completed"
            if duration_ms is not None:
                msg += f" | Duration: {duration_ms}ms"
            if items is not None:
                msg += f" | Items: {items}"
            logger.info(msg)
        elif status == "failed":
            logger.error(f"[{job_id}] Stage: {stage.value} | Status: failed | Error: {error}")

        # Update background_jobs with current stage and progress
        try:
            progress = get_xml_import_progress(stage)
            await self.db.table('background_jobs').update({
                'current_stage': stage.value,
                'progress': progress,
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', job_id).execute()
        except Exception as e:
            logger.warning(f"Failed to update job stage: {e}")

    async def process_import_job(self, job_id: str, workspace_id: str) -> Dict[str, Any]:
        """
        Process an import job from start to finish.

        Args:
            job_id: Import job ID
            workspace_id: Workspace ID

        Returns:
            Processing result with statistics
        """
        # Sentry transaction for performance monitoring
        with sentry_sdk.start_transaction(op="xml_import", name="process_import_job") as transaction:
            transaction.set_tag("job_id", job_id)
            transaction.set_tag("workspace_id", workspace_id)

            try:
                logger.info(f"🚀 Starting import job processing: {job_id}")

                # ✅ Stage: INITIALIZED
                await self._log_stage(job_id, XmlImportStage.INITIALIZED, "started")

                sentry_sdk.add_breadcrumb(
                    category="xml_import",
                    message=f"Starting XML import job {job_id}",
                    level="info"
                )

                # Get job details.
                #
                # audit #17 M4-4 — TENTH instance of "two ids each individually valid, never
                # checked against each other", and the second this session that writes to the
                # gold layer. The route authorizes the caller against `request.workspace_id`
                # (authorize_rag_workspace), which is real protection — for the WORKSPACE side.
                # Nothing checked the JOB. So a member of workspace A could pass workspace A
                # plus a job id belonging to workspace B and import B's supplier feed into
                # their own catalogue: every product, price and image, with the tenancy check
                # passing cleanly the whole way.
                #
                # MIVAA has no RLS, so this fetch IS the boundary.
                job = await self._get_job(job_id, workspace_id)
                if not job:
                    raise ValueError(f"Job {job_id} not found")

                # Update job status to processing
                await self._update_job_status(job_id, 'processing', started_at=datetime.utcnow())

                # 🆕 Update background_jobs to 'processing'
                await self._update_background_job_status(job_id, 'processing', started_at=datetime.utcnow())

                # Products are stored in data_import_job_products (child table)
                # and page-read per batch so we never hold more than `batch_size`
                # products in memory at once. total_products is authoritative
                # from the job row (set at import time by the edge function).
                total_products = int(job.get('total_products') or 0)
                if total_products <= 0:
                    raise ValueError(f"No products found for job {job_id}")

                # ✅ Stage: PRODUCTS_PARSED
                await self._log_stage(job_id, XmlImportStage.PRODUCTS_PARSED, "completed", items=total_products)

                # Materials quota (#214): resolve the workspace's remaining new-material
                # allowance once; create_product_from_payload consumes it and skips NEW
                # inserts at 0. SKU re-imports update in place and stay unaffected — so
                # a job is never refused up front (it may be a pure re-import).
                self._material_quota_remaining = await material_quota_remaining_async(self.db, workspace_id)
                self._quota_skipped = 0
                if self._material_quota_remaining == 0:
                    logger.error(
                        f"⛔ quota_exceeded: workspace {workspace_id} is at its materials plan "
                        f"limit — only SKU re-imports will apply; new products will be skipped"
                    )

                logger.info(f"📦 Processing {total_products} products in batches of {self.batch_size}")

                transaction.set_data("total_products", total_products)
                transaction.set_data("batch_size", self.batch_size)

                # Check for existing checkpoint
                last_checkpoint = await checkpoint_recovery_service.get_last_checkpoint(job_id)
                start_batch = 0
                if last_checkpoint:
                    start_batch = last_checkpoint.get('batch_index', 0)
                    logger.info(f"♻️ Resuming from batch {start_batch}")

                # Process in batches
                processed_count = 0
                failed_count = 0

                for batch_index in range(start_batch, (total_products + self.batch_size - 1) // self.batch_size):
                    batch_start = batch_index * self.batch_size
                    batch_end = min(batch_start + self.batch_size, total_products)
                    batch = await self._fetch_products_batch(job_id, batch_start, batch_end - batch_start)
                    if batch is None:
                        # audit #17 M4-7: `_fetch_products_batch` returned [] on a DB failure
                        # and the loop treated an empty batch as skippable, so a job with
                        # total_products > 0 completed with zero processed and zero failed —
                        # a clean, successful, entirely empty import. It now returns None for
                        # "the read failed" and [] for "there is genuinely nothing here",
                        # which is pipeline convention 1: an explicit failure marker, because
                        # emptiness alone is ambiguous.
                        raise RuntimeError(
                            f"Could not read products {batch_start}-{batch_end} for job "
                            f"{job_id}. Failing the job rather than completing it short."
                        )
                    if not batch:
                        logger.warning(
                            f"   ⚠️ Empty batch at {batch_start}-{batch_end} for job {job_id} — skipping"
                        )
                        continue

                    logger.info(f"🔄 Processing batch {batch_index + 1}: products {batch_start + 1}-{batch_end}")

                    sentry_sdk.add_breadcrumb(
                        category="xml_import",
                        message=f"Processing batch {batch_index + 1}: products {batch_start + 1}-{batch_end}",
                        level="info",
                        data={"batch_index": batch_index + 1, "batch_size": len(batch)}
                    )

                    # Process batch
                    batch_result = await self._process_batch(
                        job_id=job_id,
                        workspace_id=workspace_id,
                        products=batch,
                        batch_index=batch_index,
                        field_mappings=job.get('field_mappings', {})
                    )

                    processed_count += batch_result['processed']
                    failed_count += batch_result['failed']

                    # Update progress
                    progress = int((batch_end / total_products) * 100)
                    await self._update_job_progress(
                        job_id=job_id,
                        processed=processed_count,
                        failed=failed_count,
                        total=total_products,
                        stage=f'batch_{batch_index + 1}',
                        progress_percent=progress
                    )

                    # Save checkpoint
                    await checkpoint_recovery_service.save_checkpoint(
                        job_id=job_id,
                        stage=ProcessingStage.PRODUCTS_CREATED,
                        data={
                            'batch_index': batch_index + 1,
                            'processed_count': processed_count,
                            'failed_count': failed_count
                        }
                    )

                    logger.info(f"✅ Batch {batch_index + 1} complete: {batch_result['processed']} processed, {batch_result['failed']} failed")

                # audit #17 M4-7: every product the job claimed must be accounted for. This
                # single assertion closes the whole silent-zero class on this path — a
                # swallowed batch read, a skipped range, an off-by-one in the batch maths, a
                # quota clamp that lost count — because all of them arrive here as a number
                # that does not add up. Without it, the only signal was a completed job whose
                # processed count nobody compared to anything.
                # A quota-skipped product is counted inside `processed` — _process_batch
                # increments one of the two for EVERY product it is handed, and a quota skip
                # returns None from create_product_from_payload rather than raising. So the
                # invariant is the pair, not a triple; adding _quota_skipped here would fail
                # every clamped job.
                accounted = processed_count + failed_count
                if accounted != total_products:
                    raise RuntimeError(
                        f"Import job {job_id} does not reconcile: {processed_count} processed "
                        f"+ {failed_count} failed = {accounted}, but the job declared "
                        f"{total_products} products. Refusing to mark it completed — "
                        f"{total_products - accounted} product(s) were never seen."
                    )

                # Mark job as completed
                await self._update_job_status(
                    job_id=job_id,
                    status='completed',
                    completed_at=datetime.utcnow(),
                    processed_products=processed_count,
                    failed_products=failed_count
                )

                # Collect chunk dedup metrics from chunking service
                qm = self.chunking_service.quality_metrics
                chunk_metrics = {
                    'total_chunks_created': qm.total_chunks_created,
                    'exact_duplicates_prevented': qm.exact_duplicates_prevented,
                    'low_quality_rejected': qm.low_quality_rejected,
                    'final_chunks': qm.final_chunks,
                }
                if qm.exact_duplicates_prevented > 0 or qm.low_quality_rejected > 0:
                    logger.info(
                        f"   📊 Chunk dedup: {qm.total_chunks_created} created → "
                        f"{qm.final_chunks} final "
                        f"({qm.exact_duplicates_prevented} deduped, {qm.low_quality_rejected} rejected)"
                    )

                # 🆕 Update background_jobs to 'completed'
                await self._update_background_job_status(
                    job_id=job_id,
                    status='completed',
                    completed_at=datetime.utcnow(),
                    progress_percent=100,
                    metadata={
                        'total_products': total_products,
                        'processed': processed_count,
                        'failed': failed_count,
                        'completion_rate': (processed_count / total_products * 100) if total_products > 0 else 0,
                        'quality_metrics': chunk_metrics,
                        # Explicit marker (#214): "quota clamped this import" is
                        # distinguishable from "the feed only had N products".
                        'quota_skipped': getattr(self, '_quota_skipped', 0),
                    }
                )

                if getattr(self, '_quota_skipped', 0):
                    logger.error(
                        f"⛔ quota_exceeded: {self._quota_skipped} new product(s) skipped in job "
                        f"{job_id} — materials plan limit reached for workspace {workspace_id}"
                    )
                logger.info(f"🎉 Import job {job_id} completed: {processed_count}/{total_products} products processed")

                try:
                    from app.services.core.endpoint_controller import endpoint_controller
                    await endpoint_controller.scale_all_to_zero(reason=f"xml_import_completed_{job_id}")
                except Exception as scale_err:
                    logger.warning(f"⚠️ scale_all_to_zero failed on XML import completion: {scale_err}")

                # ── Factory propagation + enrichment trigger ─────────────────
                # Re-use the same helper from the PDF pipeline
                try:
                    import os, httpx as _httpx
                    _supabase_url = os.getenv("SUPABASE_URL", "")
                    _service_key  = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
                    if _supabase_url and _service_key:
                        # Collect all product IDs created in this job
                        _pids_resp = self.supabase.table('products').select('id').eq('source_job_id', job_id).execute()
                        _pids = [r['id'] for r in (_pids_resp.data or [])]
                        if _pids:
                            async with _httpx.AsyncClient(timeout=15) as _hc:
                                _resp = await _hc.post(
                                    f"{_supabase_url}/functions/v1/trigger-factory-enrichment",
                                    json={
                                        "workspace_id": workspace_id,
                                        "product_ids": _pids,
                                        "scope_column": "source_job_id",
                                        "scope_value": job_id,
                                    },
                                    headers={"Authorization": f"Bearer {_service_key}"},
                                )
                                _d = _resp.json()
                                logger.info(
                                    f"🏭 Factory enrichment: propagated={_d.get('propagated', 0)}, "
                                    f"queued_job={_d.get('queued_job_id') or 'none'}"
                                )
                except Exception as _fe:
                    logger.warning(f"⚠️ Factory enrichment trigger (non-blocking): {_fe}")

                # ── Rebuild gold-layer product relationship edges ────────────
                # SQL rule-derived edges (whole workspace) + LLM text-stated edges
                # over this job's products. Both non-blocking.
                try:
                    from app.services.products.product_relationship_service import ProductRelationshipService
                    _rel_service = ProductRelationshipService(self.supabase)
                    _edge_count = await _rel_service.rebuild_edges(workspace_id)
                    logger.info(f"🔗 Product edges rebuilt (rules): {_edge_count} for workspace {workspace_id}")

                    _job_pids_resp = self.supabase.table('products').select('id').eq(
                        'source_job_id', job_id
                    ).execute()
                    _job_pids = [r['id'] for r in (_job_pids_resp.data or [])]
                    if _job_pids:
                        _llm_edges = await _rel_service.extract_llm_edges(
                            workspace_id, product_ids=_job_pids, job_id=job_id
                        )
                        logger.info(f"🔗 Product edges extracted (LLM): {_llm_edges} for {len(_job_pids)} products")
                except Exception as _ee:
                    logger.warning(f"⚠️ Product edge rebuild (non-blocking): {_ee}")

                # ✅ Stage: COMPLETED
                await self._log_stage(job_id, XmlImportStage.COMPLETED, "completed", items=processed_count)

                sentry_sdk.add_breadcrumb(
                    category="xml_import",
                    message=f"XML import job {job_id} completed successfully",
                    level="info",
                    data={
                        "total_products": total_products,
                        "processed": processed_count,
                        "failed": failed_count,
                        "completion_rate": (processed_count / total_products * 100) if total_products > 0 else 0
                    }
                )

                transaction.set_status("ok")

                return {
                    'success': True,
                    'job_id': job_id,
                    'total_products': total_products,
                    'processed': processed_count,
                    'failed': failed_count,
                    'completion_rate': (processed_count / total_products * 100) if total_products > 0 else 0
                }

            except Exception as e:
                logger.error(f"❌ Import job {job_id} failed: {e}", exc_info=True)

                sentry_sdk.capture_exception(e)
                transaction.set_status("internal_error")

                # Mark job as failed
                await self._update_job_status(
                    job_id=job_id,
                    status='failed',
                    error_message=str(e),
                    completed_at=datetime.utcnow()
                )

                # 🆕 Update background_jobs to 'failed'
                await self._update_background_job_status(
                    job_id=job_id,
                    status='failed',
                    error_message=str(e),
                    completed_at=datetime.utcnow()
                )

                try:
                    from app.services.core.endpoint_controller import endpoint_controller
                    await endpoint_controller.scale_all_to_zero(reason=f"xml_import_failed_{job_id}")
                except Exception as scale_err:
                    logger.warning(f"⚠️ scale_all_to_zero failed on XML import failure: {scale_err}")

                raise

    async def _process_batch(
        self,
        job_id: str,
        workspace_id: str,
        products: List[Dict],
        batch_index: int,
        field_mappings: Dict[str, str]
    ) -> Dict[str, int]:
        """
        Process a batch of products.
        
        Args:
            job_id: Import job ID
            workspace_id: Workspace ID
            products: List of product data
            batch_index: Current batch index
            field_mappings: Field mapping configuration
            
        Returns:
            Dictionary with processed and failed counts
        """
        processed = 0
        failed = 0
        
        for product in products:
            try:
                # Apply field mappings to normalize product data
                normalized_product = await self._normalize_product(product, field_mappings)
                
                # Download images if present
                if normalized_product.get('images'):
                    image_urls = normalized_product['images']
                    downloaded_images = await self._download_images(image_urls, job_id, workspace_id)
                    normalized_product['downloaded_images'] = downloaded_images
                
                # Queue for product creation pipeline
                await self._queue_product_processing(
                    job_id=job_id,
                    workspace_id=workspace_id,
                    product_data=normalized_product
                )
                
                # Record in import history
                await self._record_import_history(
                    job_id=job_id,
                    source_data=product,
                    normalized_data=normalized_product,
                    status='success'
                )
                
                processed += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to process product: {e}")
                
                # Record failure in import history
                await self._record_import_history(
                    job_id=job_id,
                    source_data=product,
                    normalized_data={},
                    status='failed',
                    error_details=str(e)
                )
                
                failed += 1
        
        return {'processed': processed, 'failed': failed}

    async def _normalize_product(
        self,
        product: Dict[str, Any],
        field_mappings: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Normalize product data using field mappings.

        Args:
            product: Raw product data from XML
            field_mappings: Field mapping configuration

        Returns:
            Normalized product data
        """
        # The product dict here is ALREADY canonical — the edge-function
        # orchestrator resolved every field_mappings entry (raw XML tag →
        # canonical target) and stored the finished product_data in
        # data_import_job_products. Re-applying field_mappings here was dead
        # (its keys are raw XML tags, absent from the canonical dict) and a
        # latent corruption vector when a raw tag happened to equal a canonical
        # key (it would overwrite the orchestrator's resolved + manual-value
        # fallback). field_mappings is retained in the signature for callers /
        # back-compat but is intentionally no longer re-applied.
        # audit #17 M4-5: this was a DENYLIST of exactly one key ('metadata'), so every other
        # supplier-controlled key rode straight through into the normalized product. It stays
        # a copy of the feed — this function's job is shape, not trust — but the keys the
        # platform owns are removed here too, so they cannot arrive pre-set from outside.
        normalized = {
            k: v for k, v in product.items()
            if k != 'metadata' and k not in _SUPPLIER_RESERVED_METADATA_KEYS
        }

        normalize_factory_keys(normalized)

        # Ensure required fields exist
        if 'name' not in normalized:
            normalized['name'] = product.get('name', 'Unknown Product')

        if 'factory_name' not in normalized:
            normalized['factory_name'] = product.get('factory_name', 'Unknown Factory')

        if 'material_category' not in normalized:
            normalized['material_category'] = product.get('material_category', 'Unknown Category')

        # Add metadata (also normalize any inner metadata blob the caller passed)
        inner_meta = dict(product.get('metadata', {}))
        normalize_factory_keys(inner_meta)
        normalized['metadata'] = {
            'source_type': 'xml_import',
            'import_date': datetime.utcnow().isoformat(),
            **inner_meta,
        }

        return normalized

    async def _download_images(
        self,
        image_urls: List[str],
        job_id: str,
        workspace_id: str
    ) -> List[Dict[str, Any]]:
        """
        Download images concurrently using ImageDownloadService.

        Args:
            image_urls: List of image URLs
            job_id: Import job ID
            workspace_id: Workspace ID

        Returns:
            List of downloaded image references
        """
        if not image_urls:
            return []

        logger.info(f"📥 Downloading {len(image_urls)} images")

        # Download images using ImageDownloadService
        downloaded_images = await self.image_downloader.download_images(
            urls=image_urls,
            job_id=job_id,
            workspace_id=workspace_id,
            max_concurrent=self.max_concurrent_images
        )

        return downloaded_images

    async def _queue_product_processing(
        self,
        job_id: str,
        workspace_id: str,
        product_data: Dict[str, Any]
    ) -> None:
        """Thin XML-import adapter over the shared single-product create core."""
        await self.create_product_from_payload(
            workspace_id, product_data, source='xml_import', job_id=job_id,
        )

    async def create_product_from_payload(
        self,
        workspace_id: str,
        product_data: Dict[str, Any],
        *,
        source: str = 'xml_import',
        job_id: Optional[str] = None,
        status: str = 'draft',
    ) -> Optional[str]:
        """
        Shared single-product create core — the ONE place that turns a structured
        payload into a searchable product row: canonical metadata → facet
        canonicalization → Voyage text_embedding_1024 → image linking → optional
        chunking. Called by the XML importer and the dealer "Add Product" flow (and,
        as a follow-up, PDF Stage 4) so the embedding/canonicalization standard never
        drifts across ingest paths.

        source: provenance tag ('xml_import' | 'dealer_manual' | …) — drives the
                extracted_from / source_type / created_from_type / import_* fields.
        Returns the created/updated product id (or None on failure).
        """
        try:
            product_name = product_data.get('name', 'Unknown Product')
            logger.info(f"📤 Creating product: {product_name}")

            # ── Defensive: normalize any factory aliases on the inbound row ──
            normalize_factory_keys(product_data)

            # ── Build canonical factory nested object ────────────────────
            factory_obj = {}
            _factory_fields = [
                'factory_name', 'factory_group_name', 'address', 'city',
                'country', 'postal_code', 'phone', 'email', 'website',
                'country_of_origin', 'founded_year', 'company_type',
                'linkedin_url', 'employee_count',
            ]
            for _f in _factory_fields:
                _v = product_data.get(_f)
                if _v and str(_v).strip() not in ('', 'Unknown Factory', 'n/a', 'N/A'):
                    factory_obj[_f] = _v

            # ── Build metadata (canonical location for all content fields) ─
            # Normalize the inner metadata blob too, then spread.
            inner_meta = dict(product_data.get('metadata', {}))
            normalize_factory_keys(inner_meta)

            mat_cat = product_data.get('material_category')
            # audit #17 M4-5 / invariant 8. `**inner_meta` used to be spread LAST, over
            # everything above it — so a supplier feed carrying a `workspace_id`,
            # `import_job_id`, `unit` or `material_category` key silently rewrote the
            # provenance and classification of the product being created. (The products row's
            # own workspace_id column was never reachable this way; what a supplier could
            # rewrite is this jsonb, which is what canonicalization, faceting and the admin UI
            # all read.)
            #
            # Supplier content still arrives — it is the point of an import — but it goes in
            # FIRST, and the fields we own are written over it afterwards. Reserved keys are
            # dropped rather than merged, so their loss is logged instead of inferred.
            supplier_meta = dict(inner_meta)
            rejected = [k for k in supplier_meta if k in _SUPPLIER_RESERVED_METADATA_KEYS]
            for k in rejected:
                supplier_meta.pop(k, None)
            if rejected:
                logger.warning(
                    "   WARN supplier feed tried to set reserved metadata key(s) %s on '%s' "
                    "— dropped", rejected, product_name,
                )

            product_metadata = {
                **supplier_meta,
                "extracted_from": source,
                "import_job_id": job_id,
                "extraction_date": datetime.utcnow().isoformat(),
                "workspace_id": workspace_id,
                # Content fields — same schema as PDF/scraping
                "material_category": mat_cat,
                "unit": _CATEGORY_DEFAULT_UNITS.get(mat_cat, 'pcs') if mat_cat else 'pcs',
                "factory_name": product_data.get('factory_name'),
                "factory_group_name": product_data.get('factory_group_name'),
                "color": product_data.get('color'),
                "colors": product_data.get('colors'),
                "designer": product_data.get('designer'),
                "collection": product_data.get('collection'),
                "finish": product_data.get('finish'),
                "material": product_data.get('material'),
                "size": product_data.get('size'),
            }
            if factory_obj:
                product_metadata['factory'] = factory_obj

            # Build product record for database
            product_record = {
                "name": product_name,
                # `or ''` guards against a JSON-null description (.get(...,'')
                # only defaults a MISSING key; an explicit null returns None and
                # None[:200] raises TypeError, failing an otherwise-valid product).
                "description": (product_data.get('description') or '')[:200],
                "long_description": (product_data.get('description') or '')[:5000],
                "workspace_id": workspace_id,
                "properties": {
                    # Non-content / display fields only
                    "price": product_data.get('price'),
                    "dimensions": product_data.get('dimensions'),
                    "size": product_data.get('size'),
                    "import_source": source,
                    "import_job_id": job_id,
                    "auto_generated": True,
                    "generation_timestamp": datetime.utcnow().isoformat()
                },
                "metadata": product_metadata,
                "status": status,
                "created_from_type": source,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }

            # Add source tracking fields
            product_record['source_type'] = source
            product_record['source_job_id'] = job_id
            product_record['import_batch_id'] = f"{source}_{job_id}" if job_id else source

            # SKU-based dedup: if the XML provides a product_id/sku, treat
            # (workspace_id, external_sku) as the unique key. Re-imports of the
            # same SKU update the existing row in place; downstream chunks +
            # image associations for that product are wiped so the regenerate
            # path produces fresh artifacts instead of duplicating them.
            sku_raw = (product_data.get('product_id')
                       or product_data.get('sku')
                       or (inner_meta.get('product_id') if isinstance(inner_meta, dict) else None))
            external_sku = str(sku_raw).strip() if sku_raw else None
            # audit #17 M4-6. `.strip()` was the entire normalization, so identity was decided
            # byte-exactly. Greek capital M (U+039C) and Latin M render identically, so
            # "7012<greek-M><greek-T>" and "7012MT" are one product that compared unequal, and
            # every re-import of that feed minted a duplicate. "7012-MT" and "7012 mt" did the
            # same through separators and case.
            #
            # Raw stays raw (it is what the supplier calls it, and what a PO has to quote);
            # `external_sku_folded` is what identity is decided on, with a unique index behind
            # it. Same helper the CRM company dedupe and mention identity use — one fold, not
            # a fourth one.
            external_sku_folded = fold_model_token(external_sku) if external_sku else None
            if external_sku:
                product_record['external_sku'] = external_sku
                product_record['external_sku_folded'] = external_sku_folded

            existing_id: Optional[str] = None
            if external_sku_folded:
                try:
                    existing = await self.db.table('products') \
                        .select('id') \
                        .eq('workspace_id', workspace_id) \
                        .eq('external_sku_folded', external_sku_folded) \
                        .limit(1) \
                        .execute()
                    if existing.data:
                        existing_id = existing.data[0]['id']
                except Exception as lookup_err:
                    logger.warning(f"   WARN SKU lookup failed for {external_sku}: {lookup_err}")

            # Materials quota (#214): a NEW row consumes the workspace's remaining
            # allowance — skip BEFORE the canonicalize/embedding spend when exhausted.
            # Updates (existing_id) pass untouched. Single-create callers outside a
            # batch job (dealer manual add) fall back to a one-shot lookup; the
            # enforce_material_quota DB trigger stays the hard boundary either way.
            if existing_id is None:
                _remaining = getattr(self, '_material_quota_remaining', None)
                if _remaining is None:
                    _remaining = await material_quota_remaining_async(self.db, workspace_id)
                if _remaining == 0:
                    self._quota_skipped = getattr(self, '_quota_skipped', 0) + 1
                    logger.error(
                        f"   ⛔ quota_exceeded: skipping '{product_name}' — materials plan limit reached"
                    )
                    return None

            # ── Auto-canonicalize descriptive facets (XML supplier feeds) ────
            # XML has no LLM upstream, so this path relies entirely on the
            # canonicalizer's L0.5 (Haiku pretranslate) + L2 (Voyage cosine).
            # Multilingual raw values from supplier feeds (Greek/Italian/German)
            # auto-collapse to English canonicals here. Passing existing_id
            # enables diff-before-canonicalize so re-imports skip already-seen
            # values without re-paying translation/embedding cost.
            try:
                canonical = await canonicalize_product_attributes(
                    self.db, product_metadata, source=source,
                    product_id=existing_id if existing_id else None,
                    # audit #17 M4-8: was `self.workspace_id`, an instance field set only by a
                    # constructor argument that NEITHER call site passes — so it was None on
                    # every import this platform has ever run. With workspace_id=None the
                    # canonicalizer drops its tenancy predicate entirely: `_fetch_existing`
                    # prefetches facet values across ALL workspaces, and
                    # `resolve_facet_values_batch` writes any new canonical value with a NULL
                    # workspace, i.e. into the shared/golden namespace. One supplier's raw
                    # Greek finish name would become everyone's canonical.
                    #
                    # The argument the method was already given is the tenant. A mutable
                    # instance field is not a tenant source of truth.
                    workspace_id=workspace_id,
                )
                product_record['attributes'] = canonical.attributes
                product_record['attributes_raw'] = canonical.attributes_raw
                if canonical.resolutions:
                    logger.info(
                        f"   🏷️  XML canonicalized {len(canonical.resolutions)} facet values "
                        f"across {len(canonical.attributes)} facets"
                    )
            except Exception as canon_err:
                logger.warning(f"   ⚠️ Canonicalization failed (XML), continuing: {canon_err}")

            # Phase 2: resolve maker → per-workspace crm_companies node (flows
            # into both the insert and the update_payload via product_record).
            try:
                _factory_name = (product_metadata.get('factory_name') or '').strip()
                # Same field, and here the None made the whole branch unreachable: guarded on
                # `self.workspace_id`, which is always None, so brand resolution never ran on a
                # single imported product and `brand_company_id` was silently NULL forever.
                # A silent zero hiding inside a tenancy bug (audit #17 M4-8).
                if _factory_name and _factory_name.lower() != 'unknown factory' and workspace_id:
                    _bc = await self.db.rpc('resolve_brand_company', {
                        'p_workspace_id': workspace_id,
                        'p_name': _factory_name,
                    }).execute()
                    product_record['brand_company_id'] = _bc.data if (_bc and _bc.data) else None
            except Exception as _brand_err:
                logger.warning(f"   ⚠️ brand resolve failed (XML): {_brand_err}")

            if existing_id:
                # Update in place; refresh updated_at, preserve created_at
                update_payload = {k: v for k, v in product_record.items() if k != 'created_at'}
                await self.db.table('products').update(update_payload).eq('id', existing_id).execute()
                product_id = existing_id
                logger.info(f"♻️ Updated existing product {product_id} via SKU={external_sku}: {product_name}")

                # Wipe stale chunks + image associations so the rest of the
                # batch regenerates them cleanly. Images themselves stay in
                # document_images (other products may reference them); only
                # the product↔image join is cleared.
                try:
                    await self.db.table('document_chunks').delete().eq('product_id', product_id).execute()
                    await self.db.table('image_product_associations').delete().eq('product_id', product_id).execute()
                except Exception as wipe_err:
                    logger.warning(f"   ⚠️ Failed to clear stale artifacts for {product_id}: {wipe_err}")
            else:
                insert_response = await self.db.table('products').insert(product_record).execute()
                if not insert_response.data:
                    raise ValueError("Failed to insert product - no data returned")
                product_id = insert_response.data[0]['id']
                if isinstance(getattr(self, '_material_quota_remaining', None), int) and self._material_quota_remaining > 0:
                    self._material_quota_remaining -= 1
                logger.info(f"✅ Created product {product_id}: {product_name}")

            # Generate text_embedding_1024 for product-level vector search
            try:
                emb_parts = [product_name or '']
                desc = product_data.get('description', '')
                if desc:
                    emb_parts.append(desc[:500])
                for key in ('factory_name', 'factory_group_name', 'designer', 'material_category', 'collection', 'color', 'colors', 'finish', 'material', 'size'):
                    val = product_data.get(key) or product_metadata.get(key)
                    if val and isinstance(val, str) and val.lower() not in ('not specified', 'not found', 'unknown', 'n/a', ''):
                        emb_parts.append(val.replace('_', ' '))

                from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
                emb_svc = RealEmbeddingsService()
                emb_result = await emb_svc.generate_text_embedding(' | '.join(emb_parts))
                text_emb = emb_result.get('embedding') if emb_result.get('success') else None
                if text_emb:
                    if len(text_emb) != 1024:
                        # Defense-in-depth dim check (mirrors stage_4_products):
                        # OpenAI fallback could in theory return a wrong-dim
                        # vector; refusing to store keeps the column from
                        # holding mixed-dim rows that break vector search.
                        logger.error(
                            f"   ❌ XML import embedding for {product_name} returned "
                            f"wrong dim ({len(text_emb)}D, expected 1024D). Refusing to store."
                        )
                    else:
                        # Persist provenance (model + schema_version) alongside
                        # the vector so the admin UI / backfill cron can detect
                        # Voyage→OpenAI fallback drift and target stale-schema
                        # rows. Without these fields the rows are invisible to
                        # the standard backfill cron — XML-imported products
                        # silently age out of the v2 schema window.
                        emb_str = '[' + ','.join(str(x) for x in text_emb) + ']'
                        emb_model = emb_result.get('model') if isinstance(emb_result, dict) else None
                        update_payload: Dict[str, Any] = {
                            'text_embedding_1024': emb_str,
                            'text_embedding_schema_version': 1,
                        }
                        if emb_model:
                            update_payload['text_embedding_1024_model'] = emb_model
                        await self.db.table('products').update(update_payload)\
                            .eq('id', product_id).execute()
                        logger.info(
                            f"   🧠 Generated text_embedding_1024 for {product_name} "
                            f"(model={emb_model or 'voyage-4'})"
                        )
            except Exception as emb_err:
                logger.warning(f"   ⚠️ Product embedding failed (non-blocking): {emb_err}")

            # Link downloaded images to product
            downloaded_images = product_data.get('downloaded_images', [])
            if downloaded_images:
                await self._link_images_to_product(product_id, downloaded_images, workspace_id, job_id)

            # Queue for async text processing (chunking, embeddings)
            # This happens in background - product is already created
            if product_data.get('description'):
                await self._queue_text_processing(product_id, product_data, workspace_id, job_id)

            return product_id

        except Exception as e:
            logger.error(f"❌ Failed to create product {product_data.get('name')}: {e}")
            raise

    async def _link_images_to_product(
        self,
        product_id: str,
        downloaded_images: List[Dict[str, Any]],
        workspace_id: str,
        job_id: str = None
    ) -> None:
        """
        Link downloaded images to product in database.

        ✅ ENHANCED: Now includes image classification and SLIG embedding generation.

        Args:
            product_id: Product ID
            downloaded_images: List of downloaded image references
            workspace_id: Workspace ID
            job_id: Job ID for tracking
        """
        try:
            # Filter successful downloads
            successful_images = [img for img in downloaded_images if img.get('success')]

            if not successful_images:
                logger.info(f"⏭️ No images to link for product {product_id}")
                return

            logger.info(f"📷 Processing {len(successful_images)} images for product {product_id}")

            # Prepare images for classification and SLIG generation
            images_for_processing = []
            for img in successful_images:
                images_for_processing.append({
                    'storage_url': img.get('storage_url'),
                    'url': img.get('storage_url'),
                    'public_url': img.get('storage_url'),
                    'filename': img.get('filename'),
                    'path': None,  # Will trigger download from storage
                    'content_type': img.get('content_type'),
                    'size_bytes': img.get('size_bytes'),
                    'original_url': img.get('original_url'),
                    'storage_path': img.get('storage_path'),
                    'index': img.get('index'),
                    'product_id': product_id
                })

            image_processor = self._get_image_processor(workspace_id)

            logger.info(f"   🤖 Classifying {len(images_for_processing)} images")

            # Classify images (material vs non-material) - pass job_id for cost tracking
            material_images, non_material_images = await image_processor.classify_images(
                extracted_images=images_for_processing,
                confidence_threshold=0.6,
                job_id=job_id  # Track AI cost per job
            )

            logger.info(f"   ✅ Classification: {len(material_images)} material, {len(non_material_images)} non-material")

            # Generate SLIG embeddings for ALL images
            all_images = material_images + non_material_images

            if all_images:
                logger.info(f"   🎨 Generating SLIG embeddings for ALL {len(all_images)} images")

                result = await image_processor.save_images_and_generate_clips(
                    material_images=all_images,  # Process all images
                    document_id=product_id,
                    workspace_id=workspace_id,
                    batch_size=10,
                    max_retries=3,
                    job_id=job_id  # Track AI cost per job
                )

                logger.info(f"   ✅ Saved {result.get('images_saved', 0)} images with {result.get('clip_embeddings_generated', 0)} embeddings")

                # Get saved image IDs for associations
                saved_response = self.supabase.table('document_images')\
                    .select('id')\
                    .eq('document_id', product_id)\
                    .order('created_at', desc=True)\
                    .limit(len(all_images))\
                    .execute()

                if saved_response.data:
                    # Create product-image relationships
                    relationship_records = []
                    for idx, image_data in enumerate(saved_response.data):
                        score = 1.0 - (idx * 0.05)  # First image gets highest score
                        relationship_records.append({
                            "product_id": product_id,
                            "image_id": image_data['id'],
                            "spatial_score": 0.0,
                            "caption_score": 0.0,
                            "clip_score": 0.0,
                            "overall_score": score,
                            "confidence": score,
                            "reasoning": "xml_import_extracted",
                            "metadata": {
                                "source": "xml_import",
                                "job_id": job_id,
                                "import_index": idx
                            }
                        })

                    if relationship_records:
                        self.supabase.table('image_product_associations').insert(relationship_records).execute()
                        logger.info(f"   ✅ Created {len(relationship_records)} product-image relationships")

        except Exception as e:
            logger.error(f"❌ Failed to process images for product {product_id}: {e}")
            logger.exception(e)
            # Don't raise - product is already created

    async def _queue_text_processing(
        self,
        product_id: str,
        product_data: Dict[str, Any],
        workspace_id: str,
        job_id: str = None
    ) -> None:
        """
        Queue product text for async processing (chunking, embeddings).

        ✅ ENHANCED: Uses smart chunking for long descriptions (>1500 chars)
        to improve retrieval quality with proper semantic boundaries.

        Args:
            product_id: Product ID
            product_data: Product data with text content
            workspace_id: Workspace ID
            job_id: Job ID for tracking
        """
        try:
            description = product_data.get('description', '')
            product_name = product_data.get('name', 'Unknown')

            if not description or len(description) < 50:
                logger.info(f"⏭️ Skipping text processing for product {product_id} - insufficient content")
                return

            chunk_ids = []

            # Use smart chunking for long descriptions (>1500 chars)
            if len(description) > 1500:
                logger.info(f"📚 Long description ({len(description)} chars) - using smart chunking for {product_name}")

                # Use UnifiedChunkingService for smart chunking
                chunks = await self.chunking_service.chunk_text(
                    text=description,
                    document_id=product_id,
                    metadata={
                        "source": "xml_import",
                        "product_id": product_id,
                        "product_name": product_name,
                        "auto_generated": True
                    }
                )

                logger.info(f"   📚 Created {len(chunks)} chunks for long description")

                # Insert all chunks
                for chunk in chunks:
                    chunk_record = {
                        "document_id": product_id,
                        "workspace_id": workspace_id,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "quality_score": chunk.quality_score,
                        "source_type": "xml_import",
                        "source_job_id": job_id,
                        "metadata": {
                            **chunk.metadata,
                            "quality_score": chunk.quality_score
                        }
                    }

                    chunk_response = await self.db.table('document_chunks').insert(chunk_record).execute()
                    if chunk_response.data:
                        chunk_ids.append({'id': chunk_response.data[0]['id']})
            else:
                # Short description - single chunk (original behavior)
                chunk_record = {
                    "document_id": product_id,
                    "workspace_id": workspace_id,
                    "content": description,
                    "chunk_index": 0,
                    "source_type": "xml_import",
                    "source_job_id": job_id,
                    "metadata": {
                        "source": "xml_import",
                        "product_id": product_id,
                        "product_name": product_name,
                        "auto_generated": True
                    }
                }

                chunk_response = await self.db.table('document_chunks').insert(chunk_record).execute()

                if chunk_response.data:
                    chunk_id = chunk_response.data[0]['id']

                    # Calculate quality score
                    from app.services.chunking.unified_chunking_service import Chunk
                    temp_chunk = Chunk(
                        id=chunk_id,
                        content=description,
                        chunk_index=0,
                        total_chunks=1,
                        start_position=0,
                        end_position=len(description),
                        metadata={}
                    )
                    quality_score = self.chunking_service._calculate_chunk_quality(temp_chunk)

                    # Update chunk with quality score
                    await self.db.table('document_chunks').update({
                        'quality_score': quality_score,
                        'metadata': {
                            **chunk_record['metadata'],
                            'quality_score': quality_score
                        }
                    }).eq('id', chunk_id).execute()

                    chunk_ids.append({'id': chunk_id})

            if chunk_ids:
                logger.info(f"✅ Created {len(chunk_ids)} chunks for product {product_name}")

                # Generate text embeddings inline (ai_analysis_queue has no consumer)
                await self._generate_chunk_embeddings(chunk_ids)

        except Exception as e:
            logger.error(f"❌ Failed to queue text processing for product {product_id}: {e}")
            # Don't raise - product is already created

    async def _generate_chunk_embeddings(self, chunk_ids: list) -> None:
        """Generate Voyage AI text embeddings for chunks inline.

        Per-chunk skip diagnostics (promoted from silent `continue` to
        WARNING-with-context on 2026-05-01): silent skips here turned
        into "this product has no searchable text" weeks later in
        production with nothing to grep for. Each skip now logs the
        chunk_id and the trigger condition.
        """
        skipped_no_row = 0
        skipped_empty = 0
        for chunk_ref in chunk_ids:
            chunk_id = chunk_ref['id']
            try:
                chunk_response = await self.db.table('document_chunks').select('content').eq('id', chunk_id).single().execute()
                if not chunk_response.data:
                    skipped_no_row += 1
                    logger.warning(
                        f"⚠️ Chunk {chunk_id} not found when generating embedding "
                        f"(was it deleted between create and embed?). Skipping."
                    )
                    continue
                content = chunk_response.data.get('content', '')
                if not content:
                    skipped_empty += 1
                    logger.warning(
                        f"⚠️ Chunk {chunk_id} has empty content. Skipping embedding."
                    )
                    continue
                # generate_text_embedding returns {success, embedding, model} —
                # unpack the embedding vector; writing the dict into halfvec is
                # the bug fixed 2026-05-23.
                emb_result = await self.embedding_service.generate_text_embedding(content)
                emb_vec = emb_result.get('embedding') if isinstance(emb_result, dict) else None
                if emb_result and emb_result.get('success') and emb_vec:
                    from datetime import datetime as _dt
                    await self.db.table('document_chunks').update({
                        'text_embedding': emb_vec,
                        'embedding_model': emb_result.get('model') or 'voyage-3.5',
                        'embedding_dimension': len(emb_vec),
                        'embedding_generated_at': _dt.utcnow().isoformat(),
                        'has_text_embedding': True,
                    }).eq('id', chunk_id).execute()
                    logger.info(f"✅ Generated text embedding for chunk {chunk_id}")
            except Exception as e:
                logger.error(f"Failed to generate text embedding for chunk {chunk_id}: {e}")

        # Aggregate ERROR if more than half the chunks were silently skipped —
        # that's a structural issue (e.g. chunks getting deleted by a
        # parallel writer) and must be visible at job level, not buried in
        # per-chunk WARNINGs.
        total = len(chunk_ids)
        skipped = skipped_no_row + skipped_empty
        if total > 0 and skipped / total >= 0.5:
            logger.error(
                f"❌ Embedding loop skipped {skipped}/{total} chunks "
                f"({skipped_no_row} missing rows, {skipped_empty} empty content). "
                f"This is unusually high — a concurrent writer may be deleting "
                f"chunks, or the upstream chunker is producing empty rows. "
                f"Investigate before this batch's products go searchable with "
                f"no text-embedding coverage."
            )

    async def _get_job(
        self,
        job_id: str,
        workspace_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get job details, scoped to the workspace the caller was authorized for.

        `workspace_id` is not optional in practice — it defaults to None only so the
        single-product path (which has no job) keeps working. Passing a job id whose row
        belongs to another workspace returns None, so the caller raises "job not found"
        rather than reporting a permissions problem: same reason the platform's rule is
        404-not-403 on an ownership mismatch, since a distinguishable error is an id oracle.
        """
        try:
            q = self.db.table('data_import_jobs').select('*').eq('id', job_id)
            if workspace_id:
                q = q.eq('workspace_id', workspace_id)
            response = await q.single().execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None

    async def _fetch_products_batch(
        self,
        job_id: str,
        offset: int,
        limit: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch a single batch of products for this job, ordered by product_index.

        Returns None when the READ FAILED, [] when the range is genuinely empty.

        Products are stored in data_import_job_products (one row per product).
        Range is inclusive on both ends per PostgREST `.range(start, end)`.
        """
        try:
            end_inclusive = offset + limit - 1
            response = await self.db.table('data_import_job_products') \
                .select('product_data, product_index') \
                .eq('job_id', job_id) \
                .order('product_index') \
                .range(offset, end_inclusive) \
                .execute()
            rows = response.data or []
            return [row.get('product_data') for row in rows if row.get('product_data') is not None]
        except Exception as e:
            logger.error(
                f"Failed to fetch products batch for job {job_id} "
                f"(offset={offset}, limit={limit}): {e}"
            )
            # None, not []. The caller cannot tell a broken read from an empty range
            # otherwise, and it used to skip both (audit #17 M4-7).
            return None

    async def _update_job_status(self, job_id: str, status: str, **kwargs) -> None:
        """Update job status in database."""
        try:
            update_data = {'status': status, 'updated_at': datetime.utcnow().isoformat()}
            update_data.update(kwargs)

            # Convert datetime objects to ISO strings
            for key, value in update_data.items():
                if isinstance(value, datetime):
                    update_data[key] = value.isoformat()

            await self.db.table('data_import_jobs').update(update_data).eq('id', job_id).execute()
            logger.info(f"✅ Updated job {job_id} status to {status}")
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")

    async def _update_job_progress(
        self,
        job_id: str,
        processed: int,
        failed: int,
        total: int,
        stage: str,
        progress_percent: int = None
    ) -> None:
        """Update job progress in database and job_progress table."""
        try:
            if progress_percent is None:
                progress_percent = int((processed / total) * 100) if total > 0 else 0

            now = datetime.utcnow().isoformat()

            await self.db.table('data_import_jobs').update({
                'processed_products': processed,
                'failed_products': failed,
                'last_heartbeat': now,
            }).eq('id', job_id).execute()

            try:
                await self.db.rpc('merge_background_job_metadata', {
                    'p_job_id': job_id,
                    'p_metadata': {
                        'current_stage': stage,
                        'progress_percentage': progress_percent,
                    }
                }).execute()
            except Exception:
                pass

            # Look up background_job_id
            job_response = await self.db.table('data_import_jobs').select('background_job_id').eq('id', job_id).single().execute()
            background_job_id = job_response.data.get('background_job_id') if job_response.data else None

            if background_job_id:
                # Stage event → background_jobs.stage_history (canonical).
                try:
                    await self.db.rpc('append_stage_history', {
                        'p_job_id': background_job_id,
                        'p_event': {
                            'stage': f'xml_import_{stage}',
                            'status': 'in_progress',
                            'progress': progress_percent,
                            'completed_at': now,
                            'data': {
                                'processed': processed,
                                'failed': failed,
                                'total': total,
                                'current_stage': stage,
                            },
                            'source': 'xml_import',
                        },
                    }).execute()
                except Exception as hist_err:
                    logger.warning(f"XML stage_history append failed: {hist_err}")

                # Update background_jobs scalar fields. CORRECTION (2026-05-23):
                # the column is `progress`, not `progress_percent`. PostgREST
                # silently dropped the unknown column → XML jobs' progress was
                # never updated in background_jobs. The stage_history append
                # above is the only thing that worked.
                await self.db.table('background_jobs').update({
                    'progress': progress_percent,
                    'last_heartbeat': now,
                }).eq('id', background_job_id).execute()

                # Merge metadata atomically — no read-modify-write race
                await self.db.rpc('merge_background_job_metadata', {
                    'p_job_id': background_job_id,
                    'p_metadata': {
                        'processed': processed,
                        'failed': failed,
                        'total': total,
                        'current_stage': stage
                    }
                }).execute()

                logger.debug(f"📊 Updated job progress: {progress_percent}% ({processed}/{total} products)")
        except Exception as e:
            logger.error(f"Failed to update job progress: {e}")

    async def _update_background_job_status(
        self,
        job_id: str,
        status: str,
        **kwargs
    ) -> None:
        """
        Update background_jobs status via data_import_jobs.

        Args:
            job_id: data_import_jobs ID
            status: New status
            **kwargs: Additional fields (started_at, completed_at, error_message, progress_percent, metadata)
        """
        try:
            # Get background_job_id from data_import_jobs
            job_response = await self.db.table('data_import_jobs').select('background_job_id').eq('id', job_id).single().execute()
            background_job_id = job_response.data.get('background_job_id') if job_response.data else None

            if not background_job_id:
                logger.warning(f"No background_job_id found for data_import_job {job_id}")
                return

            # Validate status against the JobStatus enum. JobStatus(status)
            # raises ValueError on values not in the DB CHECK constraint —
            # so a typo from a caller fails LOUDLY here instead of being
            # silently rejected by Postgres.
            from app.schemas.jobs import JobStatus as _JS_dis
            validated_status = _JS_dis(status).value

            # Build update data for scalar fields
            update_data = {'status': validated_status, 'last_heartbeat': datetime.utcnow().isoformat()}

            # Separate out 'metadata' — merge atomically; everything else goes in update_data
            extra_metadata = None
            for key, value in kwargs.items():
                if key == 'metadata':
                    extra_metadata = value
                elif isinstance(value, datetime):
                    update_data[key] = value.isoformat()
                else:
                    update_data[key] = value

            await self.db.table('background_jobs').update(update_data).eq('id', background_job_id).execute()

            if extra_metadata:
                await self.db.rpc('merge_background_job_metadata', {
                    'p_job_id': background_job_id,
                    'p_metadata': extra_metadata
                }).execute()

            logger.info(f"✅ Updated background_job {background_job_id} status to {status}")

        except Exception as e:
            logger.error(f"Failed to update background_job status: {e}")
            # Don't raise - this is non-critical

    async def _record_import_history(
        self,
        job_id: str,
        source_data: Dict[str, Any],
        normalized_data: Dict[str, Any],
        status: str,
        error_details: Optional[str] = None
    ) -> None:
        """Record import history entry."""
        try:
            await self.db.table('data_import_history').insert({
                'job_id': job_id,
                'source_data': source_data,
                'normalized_data': normalized_data,
                'processing_status': status,
                'error_details': error_details,
                'created_at': datetime.utcnow().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to record import history: {e}")


