"""Data Import Service — process XML import jobs into products."""

import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple


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
from app.config import get_settings
from app.services.core.supabase_client import get_supabase_client
from app.services.tracking.xml_import_stages import XmlImportStage, get_xml_import_progress
from app.services.images.image_download_service import ImageDownloadService
from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
from app.services.chunking.unified_chunking_service import UnifiedChunkingService, ChunkingConfig, ChunkingStrategy
from app.services.metadata.metadata_normalizer import normalize_factory_keys
from app.services.facets import canonicalize_product_attributes
from app.services.products.material_quota import material_quota_remaining_async
import sentry_sdk
from app.utils.text_fold import fold_model_token
from app.services.products.category_units import (
    CategoryRegistryUnavailable, get_category_registry, resolve_category_id, resolve_default_unit,
)
from app.utils.price_parsing import parse_price


def _default_unit_for(material_category, supabase):
    """Raises CategoryRegistryUnavailable when `material_categories` cannot be read."""
    units, vocab, _ids = get_category_registry(supabase)
    return resolve_default_unit(material_category, units, vocab)


def _category_id_for(material_category, supabase):
    """Raises CategoryRegistryUnavailable when `material_categories` cannot be read."""
    _units, vocab, ids = get_category_registry(supabase)
    return resolve_category_id(material_category, ids, vocab)

logger = logging.getLogger(__name__)


#: Keys a supplier feed may never set: identity, tenancy, provenance, and the fields the
#: platform derives rather than accepts (invariant 8).
_SUPPLIER_RESERVED_METADATA_KEYS = frozenset({
    'workspace_id', 'user_id', 'created_by', 'owner_id',
    'id', 'product_id_internal', 'source_job_id', 'import_job_id', 'import_batch_id',
    'extracted_from', 'extraction_date', 'source_type', 'created_from_type',
    'status', 'is_verified', 'is_locked', 'credits',
    'external_sku_folded', 'import_identity', 'facet_canonicalization',
})

#: Descriptive keys a caller may pass at top level instead of inside `metadata`.
_TOP_LEVEL_FACET_KEYS = ('color', 'designer', 'collection', 'finish', 'material', 'size')

#: `products.source_type` CHECK values. Any other ingest path records its provenance in
#: `created_from_type` only.
_SOURCE_TYPES = frozenset({'pdf_processing', 'xml_import', 'web_scraping', 'csv_import'})

#: Columns a supplier feed owns. A re-import rewrites these and nothing else, so status,
#: curated category and the operator's own edits survive a refresh.
_SUPPLIER_OWNED_COLUMNS = frozenset({
    'name', 'description', 'long_description', 'mpn', 'barcode', 'net_mass_kg',
    'warranty', 'country_of_origin',
    'cost', 'cost_currency', 'cost_source', 'cost_updated_at',
    'external_sku', 'external_sku_folded', 'import_identity',
    'source_job_id', 'import_batch_id', 'source_type', 'updated_at',
})



class ImportRowError(ValueError):
    """A row the importer refuses, with the reason the operator reads."""


def source_type_for(source: str) -> Optional[str]:
    """The CHECK-constrained `products.source_type` for an ingest path, or None."""
    return source if source in _SOURCE_TYPES else None


def import_identity_for(
    factory_name: Optional[str],
    external_sku_folded: Optional[str],
    product_name: Optional[str],
    *,
    allow_name_identity: bool,
) -> Optional[str]:
    """Supplier namespace + folded SKU; else (feeds only) namespace + folded name."""
    namespace = fold_model_token(factory_name or '') or 'UNBRANDED'
    if external_sku_folded:
        return f"{namespace}|sku:{external_sku_folded}"
    if allow_name_identity:
        folded_name = fold_model_token(product_name or '')
        if folded_name:
            return f"{namespace}|name:{folded_name}"
    return None


def resolve_money(raw: Any, hint_currency: Optional[str]) -> Tuple[Decimal, str]:
    """(amount, ISO currency) for a feed price, or ImportRowError when it is not unambiguous."""
    text = str(raw).strip()
    amount, stated = parse_price(text)
    _, hinted = parse_price(None, hint_currency)
    if amount is None or amount < 0:
        raise ImportRowError(f"price '{text}' is not a number")
    if stated and hinted and stated != hinted:
        raise ImportRowError(f"price '{text}' is in {stated} but the feed's currency says {hinted}")
    if '¥' in text and not hinted:
        raise ImportRowError(f"price '{text}': ¥ is JPY or CNY — map or set a currency")
    currency = stated or hinted
    if not currency:
        raise ImportRowError(f"price '{text}' has no currency — map a currency field or set a default")
    return amount, currency


def derive_import_outcome(processed: int, failed: int, skipped: int) -> Tuple[str, str]:
    """(data_import_jobs.status, outcome) from the row results — the one place this is decided."""
    if processed == 0:
        return 'failed', 'failed'
    if failed or skipped:
        return 'completed', 'partial'
    return 'completed', 'complete'


class DataImportService:
    """Service for processing data import jobs"""

    def __init__(self, workspace_id: str = None):
        supabase_wrapper = get_supabase_client()
        self.supabase_wrapper = supabase_wrapper
        self.supabase = supabase_wrapper.client
        self.db = supabase_wrapper.async_client
        self.image_downloader = ImageDownloadService()
        self.batch_size = _env_int('XML_IMPORT_BATCH_SIZE', 5)
        self.max_concurrent_images = _env_int('XML_IMPORT_MAX_CONCURRENT_IMAGES', 5)
        # Never read as the tenant: every method that needs a workspace takes one (audit #17 M4-8).
        self.workspace_id = workspace_id
        self._bg_job_id: Optional[str] = None

        self.chunking_service = UnifiedChunkingService(ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID,
            max_chunk_size=1000,
            min_chunk_size=100,
            overlap_size=100
        ))
        self.embedding_service = RealEmbeddingsService()

    async def _log_stage(
        self,
        job_id: str,
        stage: XmlImportStage,
        status: str = "started",
        duration_ms: int = None,
        items: int = None,
        error: str = None
    ) -> None:
        """Log stage progress and mirror the stage onto the job's background_jobs row."""
        if status == "failed":
            logger.error(f"[{job_id}] Stage: {stage.value} | Status: failed | Error: {error}")
        else:
            msg = f"[{job_id}] Stage: {stage.value} | Status: {status}"
            if duration_ms is not None:
                msg += f" | Duration: {duration_ms}ms"
            if items is not None:
                msg += f" | Items: {items}"
            logger.info(msg)

        if not self._bg_job_id:
            return
        try:
            await self.db.table('background_jobs').update({
                'current_stage': stage.value,
                'progress': get_xml_import_progress(stage),
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', self._bg_job_id).execute()
        except Exception as e:
            logger.warning(f"Failed to update job stage: {e}")

    async def process_import_job(self, job_id: str, workspace_id: str) -> Dict[str, Any]:
        """Process an import job (already claimed by the route) from its checkpoint to the end."""
        # Ownership first and outside the try: a job id from another workspace must not reach
        # the failure handler, which writes to the job.
        job = await self._get_job(job_id, workspace_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        self._bg_job_id = job.get('background_job_id')

        with sentry_sdk.start_transaction(op="xml_import", name="process_import_job") as transaction:
            transaction.set_tag("job_id", job_id)
            transaction.set_tag("workspace_id", workspace_id)

            try:
                await self._log_stage(job_id, XmlImportStage.INITIALIZED, "started")
                await self._update_job_status(job_id, workspace_id, 'processing', started_at=datetime.utcnow())
                await self._update_background_job_status(job_id, 'processing', started_at=datetime.utcnow())

                total_products = int(job.get('total_products') or 0)
                if total_products <= 0:
                    raise ValueError(f"No products found for job {job_id}")
                await self._log_stage(job_id, XmlImportStage.PRODUCTS_PARSED, "completed", items=total_products)

                # Materials quota (#214): consumed by NEW inserts only; re-imports update in place.
                self._material_quota_remaining = await material_quota_remaining_async(self.db, workspace_id)
                self._quota_skipped = 0

                checkpoint = (job.get('metadata') or {}).get('checkpoint') or {}
                start_batch = int(checkpoint.get('batch_index') or 0)
                processed_count = int(checkpoint.get('processed') or 0)
                failed_count = int(checkpoint.get('failed') or 0)
                skipped_count = int(checkpoint.get('skipped') or 0)
                if start_batch:
                    logger.info(
                        f"♻️ Resuming {job_id} at batch {start_batch} "
                        f"({processed_count} processed, {failed_count} failed, {skipped_count} skipped)"
                    )

                for batch_index in range(start_batch, (total_products + self.batch_size - 1) // self.batch_size):
                    batch_start = batch_index * self.batch_size
                    batch_end = min(batch_start + self.batch_size, total_products)
                    batch = await self._fetch_products_batch(job_id, batch_start, batch_end - batch_start)
                    if batch is None:
                        raise RuntimeError(
                            f"Could not read products {batch_start}-{batch_end} for job "
                            f"{job_id}. Failing the job rather than completing it short."
                        )
                    if not batch:
                        logger.warning(f"   ⚠️ Empty batch at {batch_start}-{batch_end} for job {job_id}")
                        continue

                    batch_result = await self._process_batch(
                        job_id=job_id,
                        workspace_id=workspace_id,
                        products=batch,
                        batch_index=batch_index,
                        field_mappings=job.get('field_mappings') or {},
                    )
                    processed_count += batch_result['processed']
                    failed_count += batch_result['failed']
                    skipped_count += batch_result['skipped']

                    await self._save_checkpoint(job_id, {
                        'batch_index': batch_index + 1,
                        'processed': processed_count,
                        'failed': failed_count,
                        'skipped': skipped_count,
                    })
                    await self._update_job_progress(
                        job_id=job_id,
                        workspace_id=workspace_id,
                        processed=processed_count,
                        failed=failed_count,
                        total=total_products,
                        stage=f'batch_{batch_index + 1}',
                        progress_percent=int((batch_end / total_products) * 100),
                    )

                # audit #17 M4-7: every declared product must be accounted for.
                accounted = processed_count + failed_count + skipped_count
                if accounted != total_products:
                    raise RuntimeError(
                        f"Import job {job_id} does not reconcile: {processed_count} processed "
                        f"+ {failed_count} failed + {skipped_count} skipped = {accounted}, but the "
                        f"job declared {total_products} products."
                    )

                final_status, outcome = derive_import_outcome(processed_count, failed_count, skipped_count)
                summary = None
                if outcome != 'complete':
                    summary = (
                        f"{failed_count} of {total_products} rows failed"
                        + (f", {skipped_count} skipped (materials plan limit)" if skipped_count else "")
                        + " — see the row history for each reason."
                    )

                availability = {'applied': False, 'reason': 'run was not complete'}
                dropped = int((job.get('metadata') or {}).get('dropped_count') or 0)
                if outcome == 'complete' and dropped == 0:
                    availability = await self._apply_missing_from_feed_policy(job, workspace_id)

                await self._merge_job_metadata(job_id, {
                    'outcome': outcome,
                    'skipped': skipped_count,
                    'availability_policy': availability,
                })
                await self._update_job_status(
                    job_id, workspace_id,
                    status='completed' if final_status == 'completed' else 'failed',
                    completed_at=datetime.utcnow(),
                    processed_products=processed_count,
                    failed_products=failed_count,
                    error_message=summary,
                )

                qm = self.chunking_service.quality_metrics
                await self._update_background_job_status(
                    job_id=job_id,
                    status=final_status,
                    completed_at=datetime.utcnow(),
                    progress_percent=100,
                    error_message=summary,
                    metadata={
                        'total_products': total_products,
                        'processed': processed_count,
                        'failed': failed_count,
                        'skipped': skipped_count,
                        'outcome': outcome,
                        'completed_with_failures': outcome == 'partial',
                        'quota_skipped': skipped_count,
                        'quality_metrics': {
                            'total_chunks_created': qm.total_chunks_created,
                            'exact_duplicates_prevented': qm.exact_duplicates_prevented,
                            'low_quality_rejected': qm.low_quality_rejected,
                            'final_chunks': qm.final_chunks,
                        },
                    }
                )

                try:
                    from app.services.core.endpoint_controller import endpoint_controller
                    await endpoint_controller.scale_all_to_zero(reason=f"xml_import_completed_{job_id}")
                except Exception as scale_err:
                    logger.warning(f"⚠️ scale_all_to_zero failed on XML import completion: {scale_err}")

                if processed_count and self._bg_job_id:
                    await self._post_import_enrichment(workspace_id)

                await self._log_stage(job_id, XmlImportStage.COMPLETED, "completed", items=processed_count)
                transaction.set_status("ok")
                return {
                    'success': final_status == 'completed',
                    'job_id': job_id,
                    'outcome': outcome,
                    'total_products': total_products,
                    'processed': processed_count,
                    'failed': failed_count,
                    'skipped': skipped_count,
                }

            except Exception as e:
                logger.error(f"❌ Import job {job_id} failed: {e}", exc_info=True)
                sentry_sdk.capture_exception(e)
                transaction.set_status("internal_error")
                await self._update_job_status(
                    job_id, workspace_id,
                    status='failed',
                    error_message=str(e),
                    completed_at=datetime.utcnow()
                )
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

    async def _post_import_enrichment(self, workspace_id: str) -> None:
        """Factory propagation and relationship edges over this run's products (non-blocking)."""
        try:
            pids_resp = await self.db.table('products').select('id') \
                .eq('workspace_id', workspace_id).eq('source_job_id', self._bg_job_id).execute()
            pids = [r['id'] for r in (pids_resp.data or [])]
        except Exception as e:
            logger.warning(f"⚠️ Could not list this run's products: {e}")
            return
        if not pids:
            return
        try:
            import httpx as _httpx
            _supabase_url = os.getenv("SUPABASE_URL", "")
            _service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
            if _supabase_url and _service_key:
                async with _httpx.AsyncClient(timeout=15) as _hc:
                    _resp = await _hc.post(
                        f"{_supabase_url}/functions/v1/trigger-factory-enrichment",
                        json={
                            "workspace_id": workspace_id,
                            "product_ids": pids,
                            "scope_column": "source_job_id",
                            "scope_value": self._bg_job_id,
                        },
                        headers={"Authorization": f"Bearer {_service_key}"},
                    )
                    _d = _resp.json()
                    logger.info(f"🏭 Factory enrichment: propagated={_d.get('propagated', 0)}")
        except Exception as _fe:
            logger.warning(f"⚠️ Factory enrichment trigger (non-blocking): {_fe}")
        try:
            from app.services.products.product_relationship_service import ProductRelationshipService
            _rel_service = ProductRelationshipService(self.supabase)
            await _rel_service.rebuild_edges(workspace_id)
            await _rel_service.extract_llm_edges(workspace_id, product_ids=pids, job_id=self._bg_job_id)
        except Exception as _ee:
            logger.warning(f"⚠️ Product edge rebuild (non-blocking): {_ee}")

    async def _apply_missing_from_feed_policy(self, job: Dict[str, Any], workspace_id: str) -> Dict[str, Any]:
        """Mark products an earlier run of THIS feed created, and this full run did not carry,
        as discontinued by the supplier. Never deletes; refuses when most of the feed vanished."""
        root = job.get('parent_job_id')
        if not root or not self._bg_job_id:
            return {'applied': False, 'reason': 'not a repeat run of a feed'}
        lineage = await self.db.table('data_import_jobs').select('background_job_id') \
            .eq('workspace_id', workspace_id).or_(f"id.eq.{root},parent_job_id.eq.{root}").execute()
        prior_runs = [r['background_job_id'] for r in (lineage.data or [])
                      if r.get('background_job_id') and r['background_job_id'] != self._bg_job_id]
        if not prior_runs:
            return {'applied': False, 'reason': 'no earlier run'}
        missing_resp = await self.db.table('products').select('id') \
            .eq('workspace_id', workspace_id).in_('source_job_id', prior_runs) \
            .is_('discontinued_by_supplier_on', 'null').execute()
        missing = [r['id'] for r in (missing_resp.data or [])]
        if not missing:
            return {'applied': True, 'marked': 0}
        present_resp = await self.db.table('products').select('id', count='exact') \
            .eq('workspace_id', workspace_id).eq('source_job_id', self._bg_job_id).limit(1).execute()
        present = int(present_resp.count or 0)
        if len(missing) > present:
            logger.error(
                f"Availability policy refused for job {job.get('id')}: {len(missing)} products "
                f"missing vs {present} present — a truncated feed is likelier than a delisting."
            )
            return {'applied': False, 'reason': 'more products missing than present', 'missing': len(missing)}
        today = datetime.now(timezone.utc).date().isoformat()
        for i in range(0, len(missing), 200):
            await self.db.table('products').update({'discontinued_by_supplier_on': today}) \
                .eq('workspace_id', workspace_id).in_('id', missing[i:i + 200]).execute()
        return {'applied': True, 'marked': len(missing)}

    async def _process_batch(
        self,
        job_id: str,
        workspace_id: str,
        products: List[Dict],
        batch_index: int,
        field_mappings: Dict[str, str]
    ) -> Dict[str, int]:
        """Process a batch; every row ends processed, failed or skipped, with its history row."""
        processed = failed = skipped = 0
        for product in products:
            try:
                normalized_product = await self._normalize_product(product, field_mappings)
                if normalized_product.get('images'):
                    normalized_product['downloaded_images'] = await self._download_images(
                        normalized_product['images'], job_id, workspace_id,
                    )
                product_id = await self.create_product_from_payload(
                    workspace_id, normalized_product, source='xml_import', job_id=job_id,
                )
                if product_id is None:
                    skipped += 1
                    await self._record_import_history(
                        job_id=job_id, source_data=product, normalized_data={},
                        status='skipped', error_details='materials plan limit reached',
                    )
                    continue
                await self._record_import_history(
                    job_id=job_id, source_data=product, normalized_data=normalized_product,
                    status='success', product_id=product_id,
                )
                processed += 1
            except Exception as e:
                logger.error(f"❌ Failed to process product: {e}")
                await self._record_import_history(
                    job_id=job_id, source_data=product, normalized_data={},
                    status='failed', error_details=str(e),
                )
                failed += 1
        return {'processed': processed, 'failed': failed, 'skipped': skipped}

    async def _normalize_product(
        self,
        product: Dict[str, Any],
        field_mappings: Dict[str, str]
    ) -> Dict[str, Any]:
        """The orchestrator already resolved every mapping; this only drops reserved keys."""
        normalized = {
            k: v for k, v in product.items()
            if k != 'metadata' and k not in _SUPPLIER_RESERVED_METADATA_KEYS
        }
        normalize_factory_keys(normalized)
        inner_meta = dict(product.get('metadata') or {})
        normalize_factory_keys(inner_meta)
        normalized['metadata'] = inner_meta
        return normalized

    async def _download_images(
        self,
        image_urls: List[str],
        job_id: str,
        workspace_id: str
    ) -> List[Dict[str, Any]]:
        """Download images concurrently using ImageDownloadService."""
        if not image_urls:
            return []
        return await self.image_downloader.download_images(
            urls=image_urls,
            job_id=job_id,
            workspace_id=workspace_id,
            max_concurrent=self.max_concurrent_images
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
        """Shared single-product create core: canonical metadata → facet canonicalization →
        Voyage text_embedding_1024 → image linking → optional chunking. Returns None only when
        the materials quota skipped a NEW product; every refusal of the row raises."""
        product_name = product_data.get('name', 'Unknown Product')
        logger.info(f"📤 Creating product: {product_name}")
        normalize_factory_keys(product_data)
        is_feed = source == 'xml_import'

        # Money is decided before anything is written: an ambiguous price refuses the row.
        hint = product_data.get('currency') or (None if is_feed else 'EUR')
        price = None
        if product_data.get('price') not in (None, ''):
            price = resolve_money(product_data['price'], hint)
        cost = None
        if product_data.get('supplier_cost') not in (None, ''):
            cost = resolve_money(product_data['supplier_cost'], hint)

        factory_obj = {}
        for _f in ('factory_name', 'factory_group_name', 'address', 'city', 'country', 'postal_code',
                   'phone', 'email', 'website', 'country_of_origin', 'founded_year', 'company_type',
                   'linkedin_url', 'employee_count'):
            _v = product_data.get(_f)
            if _v and str(_v).strip() not in ('', 'Unknown Factory', 'n/a', 'N/A'):
                factory_obj[_f] = _v

        inner_meta = dict(product_data.get('metadata') or {})
        normalize_factory_keys(inner_meta)
        attributes_in = product_data.get('attributes')
        if isinstance(attributes_in, dict):
            inner_meta.update(attributes_in)
        for k in _TOP_LEVEL_FACET_KEYS:
            v = product_data.get(k)
            if v not in (None, '') and inner_meta.get(k) in (None, ''):
                inner_meta[k] = v

        mat_cat = product_data.get('material_category')
        supplier_meta = dict(inner_meta)
        rejected = [k for k in supplier_meta if k in _SUPPLIER_RESERVED_METADATA_KEYS]
        for k in rejected:
            supplier_meta.pop(k, None)
        if rejected:
            logger.warning(
                "   WARN supplier feed tried to set reserved metadata key(s) %s on '%s' — dropped",
                rejected, product_name,
            )
        feed_unit = str(product_data.get('unit') or supplier_meta.get('unit') or '').strip()
        _cat_id = None
        _registry_error = None
        try:
            if not feed_unit:
                feed_unit = _default_unit_for(mat_cat, self.supabase)
            _cat_id = _category_id_for(mat_cat, self.supabase)
        except CategoryRegistryUnavailable as e:
            # No unit and no category_id rather than `pcs` stated as fact.
            _registry_error = str(e)[:200]
            logger.error("   category registry unavailable for '%s': %s", product_name, e)

        product_metadata = {
            **supplier_meta,
            "extracted_from": source,
            "import_job_id": job_id,
            "extraction_date": datetime.utcnow().isoformat(),
            "workspace_id": workspace_id,
            "material_category": mat_cat,
            "unit": feed_unit or None,
            "factory_name": product_data.get('factory_name'),
            "factory_group_name": product_data.get('factory_group_name'),
        }
        if factory_obj:
            product_metadata['factory'] = factory_obj
        if _registry_error:
            product_metadata['_category_registry'] = {'status': 'unavailable', 'error': _registry_error}

        _hero = next(
            (i.get('storage_url') for i in (product_data.get('downloaded_images') or [])
             if i.get('success') and i.get('storage_url')),
            None,
        )
        if _hero:
            product_metadata['image_url'] = _hero

        columns = await self._route_registry_columns(product_metadata)

        product_record: Dict[str, Any] = {
            "name": product_name,
            "description": (product_data.get('description') or '')[:200],
            "long_description": (product_data.get('description') or '')[:5000],
            "workspace_id": workspace_id,
            "properties": {
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
            "source_type": source_type_for(source),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            **columns,
        }

        if _cat_id:
            product_record['category_id'] = _cat_id
        for _col, _src in (('mpn', 'mpn'), ('barcode', 'barcode')):
            _v = product_data.get(_src)
            if _v and str(_v).strip():
                product_record[_col] = str(_v).strip()
        _weight = product_data.get('weight')
        if _weight is not None and str(_weight).strip():
            try:
                product_record['net_mass_kg'] = float(str(_weight).replace(',', '.').strip())
            except (TypeError, ValueError):
                logger.warning(f"   WARN unparseable weight '{_weight}' for {product_name}")
        if cost is not None:
            product_record['cost'] = float(cost[0])
            product_record['cost_currency'] = cost[1]
            product_record['cost_source'] = 'supplier_feed'
            product_record['cost_updated_at'] = datetime.utcnow().isoformat()

        if self._bg_job_id:
            product_record['source_job_id'] = self._bg_job_id
        product_record['import_batch_id'] = f"{source}_{job_id}" if job_id else source

        sku_raw = (product_data.get('external_sku') or product_data.get('product_id')
                   or product_data.get('sku') or inner_meta.get('product_id'))
        external_sku = str(sku_raw).strip() if sku_raw else None
        # audit #17 M4-6: identity is decided on the folded token (Greek Μ vs Latin M, separators).
        external_sku_folded = fold_model_token(external_sku) if external_sku else None
        if external_sku:
            product_record['external_sku'] = external_sku
            product_record['external_sku_folded'] = external_sku_folded
        import_identity = import_identity_for(
            product_data.get('factory_name'), external_sku_folded, product_name,
            allow_name_identity=is_feed,
        )
        if import_identity:
            product_record['import_identity'] = import_identity

        existing: Optional[Dict[str, Any]] = None
        if import_identity:
            found = await self.db.table('products') \
                .select('id, metadata, attributes, attributes_raw, properties, category_id, brand_company_id') \
                .eq('workspace_id', workspace_id) \
                .eq('import_identity', import_identity) \
                .limit(1) \
                .execute()
            if found.data:
                existing = found.data[0]
        existing_id = existing['id'] if existing else None

        if existing_id is None:
            _remaining = getattr(self, '_material_quota_remaining', None)
            if _remaining is None:
                _remaining = await material_quota_remaining_async(self.db, workspace_id)
            if _remaining == 0:
                self._quota_skipped = getattr(self, '_quota_skipped', 0) + 1
                logger.error(f"   ⛔ quota_exceeded: skipping '{product_name}' — materials plan limit reached")
                return None

        canonical = await canonicalize_product_attributes(
            self.supabase_wrapper, product_metadata, source=source,
            product_id=existing_id,
            workspace_id=workspace_id,
        )
        product_record['attributes'] = canonical.attributes
        product_record['attributes_raw'] = canonical.attributes_raw
        if getattr(canonical, 'status', 'ok') != 'ok':
            product_metadata['facet_canonicalization'] = {
                'status': canonical.status,
                'at': datetime.utcnow().isoformat(),
                'stage': source,
            }
            logger.error(
                f"   ❌ Facet canonicalization DEGRADED for {product_name} — flagged via "
                f"metadata.facet_canonicalization"
            )

        try:
            _factory_name = (product_metadata.get('factory_name') or '').strip()
            if _factory_name and _factory_name.lower() != 'unknown factory':
                _bc = await self.db.rpc('resolve_brand_company', {
                    'p_workspace_id': workspace_id,
                    'p_name': _factory_name,
                }).execute()
                product_record['brand_company_id'] = _bc.data if (_bc and _bc.data) else None
        except Exception as _brand_err:
            logger.warning(f"   ⚠️ brand resolve failed: {_brand_err}")

        if existing:
            product_id = existing_id
            update_payload = {k: v for k, v in product_record.items() if k in _SUPPLIER_OWNED_COLUMNS}
            update_payload['metadata'] = {**(existing.get('metadata') or {}), **product_metadata}
            update_payload['attributes'] = {**(existing.get('attributes') or {}), **canonical.attributes}
            update_payload['attributes_raw'] = {**(existing.get('attributes_raw') or {}), **canonical.attributes_raw}
            update_payload['properties'] = {**(existing.get('properties') or {}), **product_record['properties']}
            update_payload['discontinued_by_supplier_on'] = None
            if not existing.get('category_id') and product_record.get('category_id'):
                update_payload['category_id'] = product_record['category_id']
            if not existing.get('brand_company_id') and product_record.get('brand_company_id'):
                update_payload['brand_company_id'] = product_record['brand_company_id']
            await self.db.table('products').update(update_payload) \
                .eq('id', product_id).eq('workspace_id', workspace_id).execute()
            logger.info(f"♻️ Updated existing product {product_id} ({import_identity}): {product_name}")
            await self.db.table('document_chunks').delete() \
                .eq('product_id', product_id).eq('source_type', 'xml_import').execute()
            await self.db.table('image_product_associations').delete() \
                .eq('product_id', product_id).eq('reasoning', 'xml_import_extracted').execute()
        else:
            insert_response = await self.db.table('products').insert(product_record).execute()
            if not insert_response.data:
                raise ValueError("Failed to insert product - no data returned")
            product_id = insert_response.data[0]['id']
            if isinstance(getattr(self, '_material_quota_remaining', None), int) and self._material_quota_remaining > 0:
                self._material_quota_remaining -= 1
            logger.info(f"✅ Created product {product_id}: {product_name}")

        if price is not None:
            await self._upsert_import_price(workspace_id, product_id, price, product_metadata.get('unit'))

        await self._embed_product(workspace_id, product_id, product_name, product_data, product_metadata)

        downloaded_images = product_data.get('downloaded_images') or []
        if downloaded_images:
            linked = await self._link_images_to_product(
                product_id, downloaded_images, workspace_id, job_id, product_name, source=source,
            )
            if linked:
                await self._embed_linked_images(product_id, workspace_id, linked, mat_cat, job_id)

        if product_data.get('description'):
            await self._queue_text_processing(product_id, product_data, workspace_id, job_id, source=source)

        return product_id

    async def _route_registry_columns(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Move registry `destination='column'` values onto their products column (stage 4's rule)."""
        from app.services.metadata.field_registry import field_registry
        from app.api.pdf_processing.stage_4_products import _route_column_fields

        await field_registry.ensure_loaded()
        columns = _route_column_fields(metadata)
        columns.pop('sku', None)
        return columns

    async def _embed_product(
        self,
        workspace_id: str,
        product_id: str,
        product_name: str,
        product_data: Dict[str, Any],
        product_metadata: Dict[str, Any],
    ) -> None:
        """text_embedding_1024 for product-level vector search, attributed to the product."""
        try:
            emb_parts = [product_name or '']
            desc = product_data.get('description') or ''
            if desc:
                emb_parts.append(desc[:500])
            for key in ('factory_name', 'factory_group_name', 'designer', 'material_category',
                        'collection', 'color', 'finish', 'material', 'size'):
                val = product_metadata.get(key)
                if val and isinstance(val, str) and val.lower() not in ('not specified', 'not found', 'unknown', 'n/a', ''):
                    emb_parts.append(val.replace('_', ' '))
            emb_result = await self.embedding_service.generate_text_embedding(
                ' | '.join(emb_parts),
                job_id=self._bg_job_id,
                workspace_id=workspace_id,
                product_id=product_id,
            )
            text_emb = emb_result.get('embedding') if emb_result.get('success') else None
            if not text_emb:
                logger.warning(f"   ⚠️ No product embedding for {product_name}")
                return
            if len(text_emb) != 1024:
                logger.error(f"   ❌ Embedding for {product_name} is {len(text_emb)}D, expected 1024D. Refusing to store.")
                return
            update_payload: Dict[str, Any] = {
                'text_embedding_1024': '[' + ','.join(str(x) for x in text_emb) + ']',
                'text_embedding_schema_version': 1,
            }
            if emb_result.get('model'):
                update_payload['text_embedding_1024_model'] = emb_result['model']
            await self.db.table('products').update(update_payload) \
                .eq('id', product_id).eq('workspace_id', workspace_id).execute()
        except Exception as emb_err:
            logger.warning(f"   ⚠️ Product embedding failed (non-blocking): {emb_err}")

    async def _upsert_import_price(
        self,
        workspace_id: str,
        product_id: str,
        price: Tuple[Decimal, str],
        unit: Optional[str],
    ) -> None:
        """Write the supplier's list price onto product_prices; a failed write fails the row.

        Upserts on (workspace_id, product_id, variant_key). `storefront_published` is not set:
        publishing is a decision, not a side effect of an import.
        """
        amount, currency = price
        await self.db.table('product_prices').upsert(
            {
                'workspace_id': workspace_id,
                'product_id': product_id,
                'variant_key': None,
                'list_price': float(amount),
                'currency': currency,
                'unit': unit,
                'notes': 'From supplier import',
            },
            on_conflict='workspace_id,product_id,variant_key',
        ).execute()

    async def _link_images_to_product(
        self,
        product_id: str,
        downloaded_images: List[Dict[str, Any]],
        workspace_id: str,
        job_id: Optional[str],
        product_name: Optional[str] = None,
        *,
        source: str = 'xml_import',
    ) -> List[Dict[str, Any]]:
        """Product-scoped image rows (no document) plus their product associations.

        Returns `{'id', 'storage_path'}` per stored row, for `_embed_linked_images`.
        """
        successful = [img for img in downloaded_images if img.get('success') and img.get('storage_url')]
        if not successful:
            return []
        rows = [{
            'document_id': None,
            'workspace_id': workspace_id,
            'image_url': img['storage_url'],
            'image_type': 'product_image',
            'source_type': 'xml_import' if source == 'xml_import' else None,
            'product_name': (product_name or '')[:255] or None,
            'processing_status': 'pending',
            'metadata': {
                'product_id': product_id,
                'original_url': img.get('original_url'),
                'storage_path': img.get('storage_path'),
                'filename': img.get('filename'),
                'content_type': img.get('content_type'),
                'import_job_id': job_id,
            },
        } for img in successful]
        inserted = await self.db.table('document_images').insert(rows).execute()
        image_ids = [r['id'] for r in (inserted.data or [])]
        if len(image_ids) != len(rows):
            raise ImportRowError(f"only {len(image_ids)} of {len(rows)} images were stored")
        associations = []
        for idx, image_id in enumerate(image_ids):
            score = max(0.0, 1.0 - (idx * 0.05))
            associations.append({
                "workspace_id": workspace_id,
                "product_id": product_id,
                "image_id": image_id,
                "spatial_score": 0.0,
                "caption_score": 0.0,
                # NULL, not 0.0: no visual comparison happened.
                "clip_score": None,
                "overall_score": score,
                "confidence": score,
                "reasoning": f"{source}_extracted",
                "metadata": {"source": source, "job_id": job_id, "import_index": idx},
            })
        await self.db.table('image_product_associations').insert(associations).execute()
        return [
            {'id': image_id, 'storage_path': img.get('storage_path')}
            for image_id, img in zip(image_ids, successful)
        ]

    async def _embed_linked_images(
        self,
        product_id: str,
        workspace_id: str,
        linked: List[Dict[str, Any]],
        material_category: Optional[str],
        job_id: Optional[str],
    ) -> None:
        """Vision analysis + image vectors inline. A failure is logged and marked on the row
        (`embedding_metadata.status='failed'`); the rows keep `has_slig_embedding = false`, so
        the backfill can still retry them. It never fails the product."""
        from app.services.images.image_processing_service import ImageProcessingService

        blocked = await self._embed_credit_block(workspace_id)
        if blocked:
            logger.warning("   inline image embedding skipped for product %s: %s", product_id, blocked)
            await self._mark_images(linked, {
                'status': 'skipped_no_credits',
                'reason': blocked,
                'skipped_at': datetime.utcnow().isoformat(),
            })
            return

        try:
            result = await ImageProcessingService(workspace_id=workspace_id).embed_product_images(
                product_id=product_id,
                workspace_id=workspace_id,
                images=linked,
                material_category=material_category,
                job_id=job_id,
            )
        except Exception as e:
            logger.error("   image embedding failed for product %s: %s", product_id, e)
            sentry_sdk.capture_exception(e)
            await self._mark_images(linked, {
                'status': 'failed',
                'error': f"{type(e).__name__}: {e}"[:1000],
                'failed_at': datetime.utcnow().isoformat(),
            })
            return
        failed = result.get('failed_images') or []
        if failed:
            logger.error(
                "   %d of %d images for product %s got no vectors: %s",
                len(failed), len(linked), product_id, failed[:3],
            )

    async def _mark_images(self, linked: List[Dict[str, Any]], marker: Dict[str, Any]) -> None:
        for img in linked:
            try:
                await self.db.table('document_images').update({'embedding_metadata': marker}) \
                    .eq('id', img['id']).execute()
            except Exception as mark_err:
                logger.warning("   could not mark image %s %s: %s", img['id'], marker.get('status'), mark_err)

    async def _billing_user(self, workspace_id: str) -> Optional[str]:
        """The job's user, else the workspace's first active owner (edge `resolveBillingUser`)."""
        cache_key = (workspace_id, self._bg_job_id)
        cache = self.__dict__.setdefault('_billing_users', {})
        if cache_key in cache:
            return cache[cache_key]
        user_id = None
        if self._bg_job_id:
            job = await self.db.table('background_jobs').select('user_id') \
                .eq('id', self._bg_job_id).limit(1).execute()
            user_id = ((job.data or [{}])[0] or {}).get('user_id')
        if not user_id:
            owner = await self.db.table('workspace_members').select('user_id') \
                .eq('workspace_id', workspace_id).eq('status', 'active').eq('role', 'owner') \
                .order('joined_at').limit(1).execute()
            user_id = ((owner.data or [{}])[0] or {}).get('user_id')
        cache[cache_key] = str(user_id) if user_id else None
        return cache[cache_key]

    async def _embed_credit_block(self, workspace_id: str) -> Optional[str]:
        """Why paid inline embedding must not run now (invariant 10), or None. Fails closed."""
        from app.services.integrations.credits_integration_service import get_credits_service

        floor = float(os.getenv('XML_IMPORT_EMBED_MIN_CREDITS', '1'))
        try:
            user_id = await self._billing_user(workspace_id)
        except Exception as e:  # noqa: BLE001
            return f"billing user unresolved: {e}"[:300]
        if not user_id:
            return "no billing user for workspace"
        available = await get_credits_service().get_available_credits(
            user_id=user_id, workspace_id=workspace_id,
        )
        if available is None:
            return "credit balance unavailable"
        if available < floor:
            return f"{available:.2f} credits available, {floor:.2f} required"
        return None

    async def _queue_text_processing(
        self,
        product_id: str,
        product_data: Dict[str, Any],
        workspace_id: str,
        job_id: str = None,
        *,
        source: str = 'xml_import',
    ) -> None:
        """Chunk the product description (product-scoped, no document) and embed the chunks."""
        try:
            description = product_data.get('description', '') or ''
            product_name = product_data.get('name', 'Unknown')
            if len(description) < 50:
                return

            chunk_source_type = source_type_for(source)
            base_meta = {
                "source": source,
                "product_id": product_id,
                "product_name": product_name,
                "auto_generated": True,
            }
            chunk_ids = []
            if len(description) > 1500:
                chunks = await self.chunking_service.chunk_text(
                    text=description, document_id=product_id, metadata=base_meta,
                )
                for chunk in chunks:
                    chunk_response = await self.db.table('document_chunks').insert({
                        "document_id": None,
                        "product_id": product_id,
                        "workspace_id": workspace_id,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "quality_score": chunk.quality_score,
                        "source_type": chunk_source_type,
                        "metadata": {**chunk.metadata, "quality_score": chunk.quality_score},
                    }).execute()
                    if chunk_response.data:
                        chunk_ids.append({'id': chunk_response.data[0]['id']})
            else:
                from app.services.chunking.unified_chunking_service import Chunk
                quality_score = self.chunking_service._calculate_chunk_quality(Chunk(
                    id='pending', content=description, chunk_index=0, total_chunks=1,
                    start_position=0, end_position=len(description), metadata={},
                ))
                chunk_response = await self.db.table('document_chunks').insert({
                    "document_id": None,
                    "product_id": product_id,
                    "workspace_id": workspace_id,
                    "content": description,
                    "chunk_index": 0,
                    "quality_score": quality_score,
                    "source_type": chunk_source_type,
                    "metadata": {**base_meta, "quality_score": quality_score},
                }).execute()
                if chunk_response.data:
                    chunk_ids.append({'id': chunk_response.data[0]['id']})

            if chunk_ids:
                await self._generate_chunk_embeddings(chunk_ids, workspace_id, product_id)
        except Exception as e:
            logger.error(f"❌ Failed to queue text processing for product {product_id}: {e}")

    async def _generate_chunk_embeddings(self, chunk_ids: list, workspace_id: str, product_id: str) -> None:
        """Voyage text embeddings for the chunks, attributed to the job, workspace and product."""
        skipped = 0
        for chunk_ref in chunk_ids:
            chunk_id = chunk_ref['id']
            try:
                chunk_response = await self.db.table('document_chunks').select('content').eq('id', chunk_id).single().execute()
                content = (chunk_response.data or {}).get('content', '')
                if not content:
                    skipped += 1
                    logger.warning(f"⚠️ Chunk {chunk_id} is missing or empty. Skipping embedding.")
                    continue
                emb_result = await self.embedding_service.generate_text_embedding(
                    content, job_id=self._bg_job_id, workspace_id=workspace_id, product_id=product_id,
                )
                emb_vec = emb_result.get('embedding') if isinstance(emb_result, dict) else None
                if emb_result and emb_result.get('success') and emb_vec:
                    await self.db.table('document_chunks').update({
                        'text_embedding': emb_vec,
                        'embedding_model': emb_result.get('model') or get_settings().voyage_document_model,
                        'embedding_dimension': len(emb_vec),
                        'embedding_generated_at': datetime.utcnow().isoformat(),
                        'has_text_embedding': True,
                    }).eq('id', chunk_id).execute()
            except Exception as e:
                logger.error(f"Failed to generate text embedding for chunk {chunk_id}: {e}")
        total = len(chunk_ids)
        if total > 0 and skipped / total >= 0.5:
            logger.error(f"❌ Embedding loop skipped {skipped}/{total} chunks for product {product_id}.")

    async def _get_job(
        self,
        job_id: str,
        workspace_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get job details, scoped to the workspace the caller was authorized for (404 not 403)."""
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
        """One batch of this job's products by product_index. None when the READ FAILED."""
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
            logger.error(f"Failed to fetch products batch for job {job_id} (offset={offset}, limit={limit}): {e}")
            return None

    async def _save_checkpoint(self, job_id: str, checkpoint: Dict[str, Any]) -> None:
        """The resume point and its counters, in one atomic metadata merge. A lost write raises."""
        await self.db.rpc('merge_import_job_metadata', {
            'p_job_id': job_id,
            'p_metadata': {'checkpoint': checkpoint},
        }).execute()

    async def _merge_job_metadata(self, job_id: str, patch: Dict[str, Any]) -> None:
        try:
            await self.db.rpc('merge_import_job_metadata', {'p_job_id': job_id, 'p_metadata': patch}).execute()
        except Exception as e:
            logger.error(f"Failed to merge metadata into import job {job_id}: {e}")

    async def _update_job_status(self, job_id: str, workspace_id: str, status: str, **kwargs) -> None:
        """Update the job row, scoped to its workspace."""
        try:
            update_data = {'status': status, 'updated_at': datetime.utcnow().isoformat()}
            update_data.update(kwargs)
            for key, value in update_data.items():
                if isinstance(value, datetime):
                    update_data[key] = value.isoformat()
            await self.db.table('data_import_jobs').update(update_data) \
                .eq('id', job_id).eq('workspace_id', workspace_id).execute()
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")

    async def _update_job_progress(
        self,
        job_id: str,
        workspace_id: str,
        processed: int,
        failed: int,
        total: int,
        stage: str,
        progress_percent: int = None
    ) -> None:
        """Update job progress on data_import_jobs and its background_jobs row."""
        try:
            if progress_percent is None:
                progress_percent = int((processed / total) * 100) if total > 0 else 0
            now = datetime.utcnow().isoformat()
            await self.db.table('data_import_jobs').update({
                'processed_products': processed,
                'failed_products': failed,
                'last_heartbeat': now,
            }).eq('id', job_id).eq('workspace_id', workspace_id).execute()

            if not self._bg_job_id:
                return
            try:
                await self.db.rpc('append_stage_history', {
                    'p_job_id': self._bg_job_id,
                    'p_event': {
                        'stage': f'xml_import_{stage}',
                        'status': 'in_progress',
                        'progress': progress_percent,
                        'completed_at': now,
                        'data': {'processed': processed, 'failed': failed, 'total': total, 'current_stage': stage},
                        'source': 'xml_import',
                    },
                }).execute()
            except Exception as hist_err:
                logger.warning(f"XML stage_history append failed: {hist_err}")
            await self.db.table('background_jobs').update({
                'progress': progress_percent,
                'last_heartbeat': now,
            }).eq('id', self._bg_job_id).execute()
            await self.db.rpc('merge_background_job_metadata', {
                'p_job_id': self._bg_job_id,
                'p_metadata': {
                    'processed': processed,
                    'failed': failed,
                    'total': total,
                    'current_stage': stage,
                    'progress_percentage': progress_percent,
                }
            }).execute()
        except Exception as e:
            logger.error(f"Failed to update job progress: {e}")

    async def _update_background_job_status(
        self,
        job_id: str,
        status: str,
        **kwargs
    ) -> None:
        """Update the job's background_jobs row (status validated against JobStatus)."""
        try:
            if not self._bg_job_id:
                logger.warning(f"No background_job_id found for data_import_job {job_id}")
                return
            from app.schemas.jobs import JobStatus as _JS_dis
            update_data = {'status': _JS_dis(status).value, 'last_heartbeat': datetime.utcnow().isoformat()}
            extra_metadata = None
            for key, value in kwargs.items():
                if key == 'metadata':
                    extra_metadata = value
                elif isinstance(value, datetime):
                    update_data[key] = value.isoformat()
                else:
                    update_data[key] = value
            await self.db.table('background_jobs').update(update_data).eq('id', self._bg_job_id).execute()
            if extra_metadata:
                await self.db.rpc('merge_background_job_metadata', {
                    'p_job_id': self._bg_job_id,
                    'p_metadata': extra_metadata
                }).execute()
        except Exception as e:
            logger.error(f"Failed to update background_job status: {e}")

    async def _record_import_history(
        self,
        job_id: str,
        source_data: Dict[str, Any],
        normalized_data: Dict[str, Any],
        status: str,
        error_details: Optional[str] = None,
        product_id: Optional[str] = None,
    ) -> None:
        """Record import history entry."""
        try:
            await self.db.table('data_import_history').insert({
                'job_id': job_id,
                'product_id': product_id,
                'source_data': source_data,
                'normalized_data': {k: v for k, v in normalized_data.items() if k != 'downloaded_images'},
                'processing_status': status,
                'error_details': error_details,
                'created_at': datetime.utcnow().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to record import history: {e}")
