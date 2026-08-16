"""
Supabase client initialization and configuration.

This module provides a centralized way to initialize and manage the Supabase client
for database operations and storage management.
"""

import logging
import time
import httpx
import inspect
import functools
import importlib
import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Import-time cost is zero and there is no cycle risk: TYPE_CHECKING is False at
    # runtime. Present so the quoted `-> 'AsyncSupabaseClient'` annotation below
    # actually resolves — without it the name is undefined to every reader, human or
    # tool, which is how genuine undefined-name bugs stayed camouflaged in the noise.
    from app.services.core.async_supabase import AsyncSupabaseClient  # noqa: F401
from supabase import create_client, Client, ClientOptions
from app.config import Settings
from app.utils.exceptions import SupabaseQueryError, TenancyViolation

logger = logging.getLogger(__name__)

# Guard so the central PostgREST .execute() retry patch is installed once per
# process, no matter how many times initialize() runs.
_postgrest_retry_installed = False

#: Set while the LOG SINK is writing its own batch. Thread-local because
#: SupabaseLoggingHandler flushes on its own worker thread, so this can never leak into a
#: request's retry decisions. See `log_sink_write()`.
_log_sink_guard = threading.local()


@contextmanager
def log_sink_write():
    """Mark this thread as being inside the log sink's own Supabase write.

    Everything under here reports retry decisions to stderr instead of through `logging`.
    Without it the sink's failure is logged, which re-enters the sink that just failed and
    (via Sentry's `event_level=ERROR`) raises an alert about the alerting.
    """
    previous = getattr(_log_sink_guard, "active", False)
    _log_sink_guard.active = True
    try:
        yield
    finally:
        _log_sink_guard.active = previous


def _install_postgrest_retry_once(
    max_retries: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 8.0,
) -> None:
    """
    Centralize transient-disconnect retry for EVERY Supabase/PostgREST query.

    PostgREST keeps a pooled keep-alive HTTP connection; after an idle period
    the server closes it, so the next query raises httpx "Server disconnected"
    / ConnectError. We were patching this per call-site (job_monitor_service,
    rag_service, …) — easy to miss. Instead, wrap the sync request-builder's
    `.execute()` at the library boundary so a fresh request is re-issued
    transparently on transient failures.

    This single patch covers BOTH access paths:
      • sync   — `supabase_client.client.table(...).execute()`
      • async  — `supabase_client.async_client.table(...).execute()`, which
                 dispatches the SAME sync `.execute()` via asyncio.to_thread.

    Discovers the builder classes that define their own `execute` (rather than
    hardcoding class names) so it survives postgrest version bumps. Non-retryable
    errors (per should_retry_exception) propagate immediately.
    """
    global _postgrest_retry_installed
    if _postgrest_retry_installed:
        return

    from app.utils.retry_helper import should_retry_exception

    #: Methods whose repetition cannot create a second row. PostgREST maps
    #: select→GET, update→PATCH, delete→DELETE, and insert/upsert/rpc→POST.
    _IDEMPOTENT_METHODS = {"GET", "HEAD", "PUT", "PATCH", "DELETE"}

    def _is_safe_to_repeat(builder) -> bool:
        """Whether re-issuing this request can duplicate an effect.

        A transient "Server disconnected" is AMBIGUOUS: the server may have
        committed the write and died before answering. Re-issuing a SELECT or a
        filtered PATCH/DELETE is free; re-issuing an INSERT creates a second row,
        and re-issuing an RPC runs the function twice — `append_stage_history`
        would append the same event twice, `charge_cron` would debit twice.

        Audit #12: this patch wrapped every builder's execute() unconditionally,
        so the convenience of transparent retry was being bought with silent
        duplicates on exactly the calls where a duplicate matters.

        Upserts are the one POST that IS safe — they carry
        `Prefer: resolution=merge-duplicates` and collapse onto the conflict
        target — so they keep their retry.
        """
        request = getattr(builder, "request", None)
        method = (getattr(request, "http_method", "") or "").upper()
        if method in _IDEMPOTENT_METHODS:
            return True
        if method == "POST":
            try:
                prefer = ",".join(request.headers.get_list("Prefer", split_commas=True))
            except Exception:
                prefer = str(getattr(request, "headers", {}) or "")
            return "resolution=" in prefer.lower()
        # Unknown method: assume unsafe. Being wrong in this direction costs one
        # surfaced error; being wrong the other way costs a duplicate row.
        return False

    try:
        module = importlib.import_module("postgrest._sync.request_builder")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"⚠️ PostgREST retry patch skipped (import failed): {e}")
        return

    def _say(level: str, message: str) -> None:
        """Report a retry decision — to stderr when the failing request IS the log sink.

        `SupabaseLoggingHandler._write_batch` inserts into `system_logs`, which is a POST
        and therefore non-idempotent, so a transient disconnect during a log flush lands
        in the branch below. That branch called `logger.error(...)`, which the root logger
        routes straight back into the handler that just failed — and, because Sentry's
        LoggingIntegration has `event_level=ERROR`, raised a Sentry event for it.

        The handler already prints to stderr precisely to avoid that recursion; this patch
        sits underneath it and defeated it. 27 Sentry events in one day, all of them a
        dropped LOG ROW rather than dropped business data.

        Business callers still get the ERROR and the Sentry event, which is the point.
        """
        if getattr(_log_sink_guard, "active", False):
            print(f"[postgrest-retry] {message}", file=sys.stderr)
            return
        getattr(logger, level)(message)

    def _wrap(original):
        @functools.wraps(original)
        def execute_with_retry(self, *args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries + 1):  # +1 for the initial attempt
                try:
                    result = original(self, *args, **kwargs)
                    if attempt > 0:
                        _say(
                            "info",
                            f"✅ PostgREST query succeeded on attempt "
                            f"{attempt + 1}/{max_retries + 1}",
                        )
                    return result
                except Exception as e:
                    if attempt < max_retries and should_retry_exception(e):
                        if not _is_safe_to_repeat(self):
                            # Surface it instead of silently risking a duplicate.
                            # The caller decides; most of these are inserts whose
                            # duplicate would be far more expensive than the error.
                            _say(
                                "error",
                                f"❌ PostgREST transient failure on a NON-IDEMPOTENT "
                                f"request — not retrying (a repeat could duplicate the "
                                f"write): {e}",
                            )
                            raise
                        _say(
                            "warning",
                            f"⚠️ PostgREST transient failure "
                            f"(attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {delay:.1f}s...",
                        )
                        time.sleep(delay)
                        delay = min(delay * 2.0, max_delay)
                        continue
                    raise
        execute_with_retry._mivaa_retry_wrapped = True  # type: ignore[attr-defined]
        return execute_with_retry

    patched = []
    for name, obj in vars(module).items():
        if not inspect.isclass(obj):
            continue
        # Only patch classes that define their OWN execute (subclasses that
        # inherit it are covered transitively through the patched parent).
        execute = obj.__dict__.get("execute")
        if execute is None or getattr(execute, "_mivaa_retry_wrapped", False):
            continue
        obj.execute = _wrap(execute)
        patched.append(name)

    _postgrest_retry_installed = True
    if patched:
        logger.info(
            f"✅ Installed central PostgREST .execute() retry on: {', '.join(patched)}"
        )
    else:
        logger.warning(
            "⚠️ PostgREST retry patch found no builder classes to wrap "
            "(library layout changed?) — falling back to per-call-site retries."
        )


class SupabaseClient:
    """Singleton class for managing Supabase client instance."""
    
    _instance: Optional['SupabaseClient'] = None
    _httpx_client: Optional[httpx.Client] = None
    _client: Optional[Client] = None
    
    def __new__(cls) -> 'SupabaseClient':
        """Ensure only one instance of SupabaseClient exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the SupabaseClient (called only once due to singleton pattern)."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._settings: Optional[Settings] = None
    
    
    def _create_httpx_client(self) -> httpx.Client:
        """
        Create httpx client with connection pooling and timeout configuration.
        
        Returns:
            Configured httpx.Client instance
        """
        return httpx.Client(
            limits=httpx.Limits(
                max_connections=50,  # Total connection pool size
                max_keepalive_connections=20,  # Reusable connections
                keepalive_expiry=30.0  # Keep connections alive for 30 seconds
            ),
            timeout=httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=30.0,  # Read timeout
                write=30.0,  # Write timeout
                pool=5.0  # Pool timeout
            ),
            http2=True,
            follow_redirects=True
        )
    def initialize(self, settings: Settings) -> None:
        """
        Initialize the Supabase client with configuration settings.
        
        Args:
            settings: Application settings containing Supabase configuration
            
        Raises:
            ValueError: If required Supabase settings are missing
            Exception: If client initialization fails
        """
        try:
            self._settings = settings
            
            # Validate required settings
            if not settings.supabase_url:
                raise ValueError("SUPABASE_URL is required but not provided")
            
            # SUPABASE_ANON_KEY is deliberately NOT required: nothing in MIVAA
            # reads it any more (the only two references were this check and the
            # config field). Requiring a credential the service never uses just
            # blocks startup on a deployment that is actually configured
            # correctly. The key that matters is asserted below.

            # Create httpx client with connection pooling.
            # keepalive_expiry=30s makes the client recycle idle connections
            # BEFORE PostgREST's gateway closes them, which is the root cause of
            # the transient "Server disconnected" errors (the central retry in
            # _install_postgrest_retry_once is the safety net for the residual).
            self._httpx_client = self._create_httpx_client()
            logger.info("✅ Created httpx client with connection pooling (max_connections=50, max_keepalive=20)")

            # Create Supabase client.
            #
            # The service-role key is REQUIRED, not preferred. MIVAA has no RLS
            # backstop by design — the architecture assumes service role — so
            # falling back to the anon key does not degrade gracefully: writes
            # fail and reads come back RLS-filtered, which the helpers below
            # then report as "found nothing" rather than "misconfigured". One
            # missing env var used to present as an empty platform.
            #
            # If an anon-scoped client is ever genuinely needed, build it
            # separately and name it as such; never silently substitute it here.
            supabase_key = settings.supabase_service_role_key
            if not supabase_key:
                raise ValueError(
                    "SUPABASE_SERVICE_ROLE_KEY is required - MIVAA runs every query "
                    "on the service-role client and has no RLS backstop. Refusing to "
                    "start on the anon key, which would silently return empty results."
                )
            # Inject the tuned httpx client so the pool config above actually
            # applies to PostgREST (supabase plumbs options.httpx_client into the
            # postgrest sub-client). Without this the config was dead.
            self._client = create_client(
                supabase_url=settings.supabase_url,
                supabase_key=supabase_key,
                options=ClientOptions(httpx_client=self._httpx_client),
            )

            # Centralize transient-disconnect retry for every query (sync + async)
            # at the PostgREST builder boundary — see _install_postgrest_retry_once.
            _install_postgrest_retry_once()

            logger.info("Supabase client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise
    
    @property
    def client(self) -> Client:
        """
        Get the Supabase client instance.

        Returns:
            Supabase client instance

        Raises:
            RuntimeError: If client is not initialized
        """
        if self._client is None:
            raise RuntimeError(
                "Supabase client not initialized. Call initialize() first."
            )
        return self._client

    @property
    def async_client(self) -> 'AsyncSupabaseClient':
        """
        Async façade over the sync Supabase client.

        Every .execute() call is dispatched to asyncio.to_thread() so it never
        blocks the FastAPI event loop. Use this in all async service methods:

            result = await self.db.table('products').insert(record).execute()
            result = await self.db.rpc('some_fn', {...}).execute()

        The underlying sync client is the same singleton — no extra connections.
        """
        from app.services.core.async_supabase import AsyncSupabaseClient
        return AsyncSupabaseClient(self.client)

    @property
    def settings(self) -> Settings:
        """
        Get the application settings.
        
        Returns:
            Application settings instance
            
        Raises:
            RuntimeError: If settings are not available
        """
        if self._settings is None:
            raise RuntimeError(
                "Settings not available. Call initialize() first."
            )
        return self._settings
    
    def health_check(self) -> bool:
        """
        Perform a health check on the Supabase connection.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            # Simple query to test connection
            self._client.table('processed_documents').select('id').limit(1).execute()
            return True
        except Exception as e:
            logger.warning(f"Supabase health check failed: {str(e)}")
            return False

    def get_connection_pool_stats(self) -> Dict[str, Any]:
        """
        Get current connection pool statistics.

        Returns:
            Dict containing connection pool metrics
        """
        try:
            if not self._httpx_client:
                return {
                    "status": "not_initialized",
                    "max_connections": 0,
                    "max_keepalive": 0,
                    "active_connections": 0,
                    "idle_connections": 0,
                    "pool_utilization_percent": 0
                }

            # httpx doesn't expose pool internals — return configured values
            # Note: Don't call health_check() here to avoid triggering supabase-py
            # internal httpx attribute access errors. Health is checked separately
            # by database_health_service.
            return {
                "status": "configured",
                "max_connections": 50,
                "max_keepalive": 20,
                "keepalive_expiry_seconds": 30.0,
                "pool_timeout_seconds": 5.0,
                "http2_enabled": True,
                "pool_utilization_percent": 0,
                "note": "httpx does not expose active connection count"
            }
        except Exception as e:
            logger.error(f"Failed to get connection pool stats: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def list_documents(
        self,
        workspace_id: str,
        limit: int = 100,
        status_filter: str = None,
    ) -> dict:
        """
        List documents from the documents table, for ONE workspace.

        `workspace_id` is required and first. This ran on the service-role
        client with no tenant predicate, so it listed every document on the
        platform; four search routes used it to build their candidate id set.

        Args:
            workspace_id: Tenant to list within (required)
            limit: Maximum number of documents to return
            status_filter: Filter by document processing status

        Returns:
            Dictionary containing documents list

        Raises:
            TenancyViolation: If workspace_id is missing.
            SupabaseQueryError: If the query fails. An empty list means the
                query ran and matched nothing - it never means it failed.
        """
        if not workspace_id:
            raise TenancyViolation(
                "list_documents requires a workspace_id - an unscoped list "
                "returns every tenant's documents"
            )
        try:
            query = self._client.table('documents')\
                .select('*')\
                .eq('workspace_id', workspace_id)

            # Filter by processing status if provided
            if status_filter:
                query = query.eq('processing_status', status_filter)

            query = query.limit(limit)
            response = query.execute()

            return {
                "documents": response.data,
                "count": len(response.data)
            }
        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            raise SupabaseQueryError(f"list_documents failed: {e}") from e

    async def get_document_by_id(self, document_id: str, workspace_id: str) -> dict:
        """
        Get a specific document by ID, within one workspace.

        An id alone is not an authorisation. Scoped here rather than at each
        call site so a new caller inherits the check instead of having to
        remember it.

        Args:
            document_id: The document ID to retrieve
            workspace_id: Tenant the document must belong to (required)

        Returns:
            Dictionary containing document data, or a not-found result. A
            document owned by another workspace reports "not found" - the same
            answer as a document that does not exist, so the response cannot
            be used to enumerate ids.

        Raises:
            TenancyViolation: If workspace_id is missing.
            SupabaseQueryError: If the query fails - distinct from not found.
        """
        if not workspace_id:
            raise TenancyViolation(
                "get_document_by_id requires a workspace_id - an id alone is "
                "not an authorisation"
            )
        try:
            response = self._client.table('processed_documents')\
                .select('*')\
                .eq('id', document_id)\
                .eq('workspace_id', workspace_id)\
                .execute()

            if response.data:
                return {
                    "success": True,
                    "data": response.data[0]
                }
            else:
                return {
                    "success": False,
                    "error": f"Document {document_id} not found"
                }
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            raise SupabaseQueryError(f"get_document_by_id failed: {e}") from e

    async def get_document_images(self, document_id: str, workspace_id: str) -> list:
        """
        Get images associated with a document, within one workspace.

        Args:
            document_id: The document ID
            workspace_id: Tenant the images must belong to (required)

        Returns:
            List of image records. Empty means the document has no images in
            this workspace - it never means the query failed.

        Raises:
            TenancyViolation: If workspace_id is missing.
            SupabaseQueryError: If the query fails.
        """
        if not workspace_id:
            raise TenancyViolation(
                "get_document_images requires a workspace_id - document_id "
                "alone does not establish who may read the images"
            )
        try:
            response = self._client.table('document_images')\
                .select('*')\
                .eq('document_id', document_id)\
                .eq('workspace_id', workspace_id)\
                .execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Failed to get images for document {document_id}: {str(e)}")
            raise SupabaseQueryError(f"get_document_images failed: {e}") from e

    async def save_single_image(
        self,
        image_info: Dict[str, Any],
        document_id: str,
        workspace_id: str,
        image_index: int = 0,
        category: Optional[str] = None,
        job_id: Optional[str] = None,
        bbox: Optional[list] = None,
        detection_confidence: Optional[float] = None,
        product_name: Optional[str] = None,
        # 4-Layer Extraction Metadata
        layer: Optional[int] = None,
        captures_vector_graphics: Optional[bool] = None,
        is_duplicate: Optional[bool] = None,
        duplicate_of: Optional[str] = None,
        perceptual_hash: Optional[str] = None,
        vision_provider: Optional[str] = None,
        vision_model: Optional[str] = None,
        # Material category from upload (tiles, heatpump, wood, etc.)
        material_category: Optional[str] = None
    ) -> Optional[str]:
        """
        Save a single image to document_images table.

        This is a lightweight method for saving images one at a time during processing
        to avoid memory accumulation.

        Args:
            image_info: Image metadata dict with keys like storage_url, page_number, etc.
            document_id: Document UUID
            workspace_id: Workspace UUID (optional)
            image_index: Index of image in processing sequence
            category: Image category (product, certificate, logo, specification, general)
            job_id: Job ID for source tracking (optional)
            bbox: Bounding box coordinates [x, y, width, height] normalized to 0-1
            detection_confidence: Confidence score (0.0-1.0)
            product_name: Product name

        Returns:
            Image ID if successful, None otherwise

        Raises:
            TenancyViolation: If workspace_id is missing, or names a different
                workspace than the parent document belongs to.
        """
        if not workspace_id:
            raise TenancyViolation(
                "save_single_image requires a workspace_id - a document_images "
                "row with no tenant is invisible to every scoped read afterwards"
            )

        # document_id and workspace_id are each valid on their own; that is not
        # a check. Reconcile them against the source document before writing,
        # or an image lands in a tenant the document does not belong to.
        document = self._client.table('documents')            .select('workspace_id')            .eq('id', document_id)            .maybe_single()            .execute()
        source_workspace = (getattr(document, 'data', None) or {}).get('workspace_id')
        if not source_workspace:
            raise TenancyViolation(
                f"document {document_id} not found or has no workspace - refusing "
                "to save an image against it"
            )
        if str(source_workspace) != str(workspace_id):
            raise TenancyViolation(
                f"document {document_id} does not belong to workspace {workspace_id}"
            )

        try:
            # Extract image URL (try multiple possible keys)
            # Priority: storage_url (cloud) > public_url (cloud) > url > path (local fallback)
            image_url = (
                image_info.get('storage_url') or
                image_info.get('public_url') or
                image_info.get('url') or
                image_info.get('path')
            )

            # Debug logging to track which URL is being used
            if image_info.get('storage_url'):
                logger.debug(f"✅ Using storage_url for image {image_index}: {image_url[:100]}")
            elif image_info.get('public_url'):
                logger.debug(f"✅ Using public_url for image {image_index}: {image_url[:100]}")
            elif image_info.get('path'):
                logger.warning(f"⚠️ Using local path for image {image_index} (upload may have failed): {image_url[:100]}")
                logger.warning(f"   Available keys in image_info: {list(image_info.keys())}")

            if not image_url or image_url.startswith('placeholder_'):
                logger.debug(f"⏭️  Skipping image {image_index} - no valid URL")
                return None

            # Extract metadata
            page_num = image_info.get('page') or image_info.get('page_number')
            if not page_num:
                logger.warning(
                    f"⚠️ Image {image_index} missing page_number - defaulting to 1. "
                    f"Image info keys: {list(image_info.keys())}"
                )
                page_num = 1

            # Extract AI classification results if available
            ai_classification = image_info.get('ai_classification', {})

            # Generate caption: Use AI-generated description/reason if available
            # Priority: explicit caption > explicit description > AI reason > fallback
            caption = image_info.get('caption') or image_info.get('description')
            if not caption:
                # Use AI classification reason as caption if available (it describes what was seen)
                ai_reason = ai_classification.get('reason')
                if ai_reason and ai_reason != 'Unknown' and len(ai_reason) > 10:
                    # Format: "Material image (AI): <reason>"
                    classification_type = ai_classification.get('classification', 'material')
                    caption = f"{classification_type.replace('_', ' ').title()}: {ai_reason}"
                else:
                    caption = f"Image from page {page_num}"
            is_material = ai_classification.get('is_material', False)

            # Determine category: use material_category from upload (tiles, heatpump, etc.)
            # Priority: material_category > explicit category > ai_classification > default 'general'
            # This ensures images are categorized by their extraction category for proper relevancy search
            if material_category:
                final_category = material_category  # e.g., 'tiles', 'heatpump', 'wood'
            elif category:
                final_category = category
            elif is_material:
                final_category = 'product'  # Fallback if no material_category provided
            else:
                final_category = 'general'

            # Determine image_type from AI classification
            # Priority: AI classification type > fallback to 'material_sample'
            image_type = ai_classification.get('classification') or 'material_sample'
            # Valid types: material_closeup, material_in_situ, non_material

            # Validate bbox before saving to prevent constraint violations.
            # bbox must be None OR [x, y, width, height] with values 0-1.
            validated_bbox = None
            if bbox is not None:
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    # Check all values are between 0 and 1
                    try:
                        all_valid = all(0 <= float(v) <= 1 for v in bbox)
                        if all_valid:
                            validated_bbox = list(bbox)
                        else:
                            logger.warning(f"⚠️ bbox values out of range (0-1): {bbox[:4]}, setting to None")
                    except (TypeError, ValueError) as e:
                        logger.warning(f"⚠️ bbox has non-numeric values: {e}, setting to None")
                else:
                    # Log detailed error for debugging (includes embedding detection)
                    bbox_len = len(bbox) if isinstance(bbox, (list, tuple)) else 'N/A'
                    logger.error(f"❌ CRITICAL: Invalid bbox - expected 4 elements, got {bbox_len}. First 5 values: {bbox[:5] if isinstance(bbox, (list, tuple)) else bbox}")
                    logger.error("   This may indicate an embedding was incorrectly assigned to bbox")
            bbox = validated_bbox

            # `extraction_layer` is the canonical column-backed enum
            # ({embedded, region_crop, full_render, vision_guided}). PyMuPDF is
            # the engine for embedded/full_render/region_crop; the column
            # describes what kind of output we got, not which library did it.
            # Producer (pdf_processor) emits canonical values directly since
            # the 2026-05-02 cleanup, so all callers funnel through image_info.
            extraction_layer_val = image_info.get('extraction_layer') or 'embedded'

            # Prepare image entry (same format as batch save)
            image_entry = {
                'document_id': document_id,
                'image_url': image_url,
                'image_type': image_type,
                'caption': caption,
                'page_number': page_num,
                'confidence': 0.95,
                'processing_status': 'completed',
                'category': final_category,  # AI-derived category
                'source_type': 'pdf_processing',
                'source_job_id': job_id,
                # Extraction metadata — `extraction_layer` is canonical.
                # Legacy `extraction_method` column write removed
                # 2026-05-02; the field now lives only in metadata for
                # backwards-compat with existing read sites that still
                # reference it (frontend display, analytics).
                'extraction_layer': extraction_layer_val,
                'bbox': bbox,
                'detection_confidence': detection_confidence,
                'product_name': product_name,
                # 4-Layer Extraction Metadata
                'layer': layer or image_info.get('layer'),
                'captures_vector_graphics': captures_vector_graphics if captures_vector_graphics is not None else image_info.get('captures_vector_graphics'),
                'is_duplicate': is_duplicate if is_duplicate is not None else image_info.get('is_duplicate'),
                'duplicate_of': duplicate_of or image_info.get('duplicate_of'),
                'perceptual_hash': perceptual_hash or image_info.get('perceptual_hash'),
                'vision_provider': vision_provider or image_info.get('vision_provider'),
                'vision_model': vision_model or image_info.get('vision_model'),
                'metadata': {
                    'source': 'mivaa_pdf_extraction',
                    'image_index': image_index,
                    # `extraction_layer` is the canonical enum on the row;
                    # we mirror it here (not `extraction_method`) so any
                    # consumer reading from metadata sees the same value.
                    'extraction_layer': extraction_layer_val,
                    # Layout region type for region_crop images (TABLE/TEXT/TITLE/
                    # CAPTION/IMAGE/FIGURE). The region-crop extractor emits it
                    # top-level on image_info; persist it under metadata so the
                    # Phase-3 OCR text-bearing filter can read it (S3-2). Without
                    # this it was dropped at save → every region crop fell to the
                    # conservative "unknown → OCR" branch, OCRing product photos.
                    'region_type': image_info.get('region_type'),
                    # Invariant marker (2026-06-14): why this image did NOT go
                    # through the SLIG + vision embedding bundle. None = it DID
                    # (regular material image). 'icon_candidate_spec_path' =
                    # routed to OCR/Claude spec extraction. 'classified_non_material'
                    # = dropped as non-material. Guarantees no document_images row
                    # lands with an empty vision_provider AND no recorded reason.
                    'bundle_skipped_reason': image_info.get('bundle_skipped_reason'),
                    'storage_uploaded': image_info.get('storage_uploaded', False),
                    'storage_bucket': image_info.get('storage_bucket', 'pdf-tiles'),
                    'storage_path': image_info.get('storage_path'),
                    'width': image_info.get('width'),
                    'height': image_info.get('height'),
                    'format': image_info.get('format'),
                    'quality_score': image_info.get('quality_score'),
                    'file_size': image_info.get('size_bytes'),
                    'extracted_at': datetime.utcnow().isoformat(),
                    # Store AI classification results
                    'ai_classification': {
                        'is_material': is_material,
                        'confidence': ai_classification.get('confidence'),
                        'reason': ai_classification.get('reason'),
                        'model': ai_classification.get('model'),
                        'classification': ai_classification.get('classification'),
                        # Quarantine marker — set when the classifier API
                        # failed and the image was persisted WITHOUT
                        # embeddings. Re-classification backfills target this.
                        'classification_pending': ai_classification.get('classification_pending', False)
                    } if ai_classification else None,
                    # Store vision-guided metadata
                    'vision_guided': {
                        'bbox': bbox,
                        'confidence': detection_confidence,
                        'provider': vision_provider or image_info.get('vision_provider'),
                        'model': vision_model or image_info.get('vision_model'),
                        'product_name': product_name
                    } if extraction_layer_val == 'vision_guided' else None,
                    # 4-Layer extraction metadata
                    'layer_info': {
                        'layer': layer or image_info.get('layer'),
                        'captures_vector_graphics': captures_vector_graphics if captures_vector_graphics is not None else image_info.get('captures_vector_graphics'),
                        'is_duplicate': is_duplicate if is_duplicate is not None else image_info.get('is_duplicate'),
                        'duplicate_of': duplicate_of or image_info.get('duplicate_of'),
                        'perceptual_hash': perceptual_hash or image_info.get('perceptual_hash')
                    }
                }
            }

            # Always stamped, never conditional. An optional tenant on a write
            # is how child rows ended up with workspace_id NULL - invisible to
            # every workspace-scoped read afterwards, and indistinguishable
            # from a row that legitimately has no tenant. Asserted at the top
            # of the method, so by here it is present.
            image_entry['workspace_id'] = workspace_id

            # Insert into database
            response = self._client.table('document_images').insert(image_entry).execute()

            if response.data and len(response.data) > 0:
                image_id = response.data[0]['id']
                logger.debug(f"✅ Saved image to DB: {image_id} (page {page_num})")
                return image_id
            else:
                logger.warning(f"⚠️ Failed to save image {image_index}: No data returned")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to save image {image_index} to database: {e}")
            return None

    # save_pdf_processing_result removed 2026-05-23 — zero callers (verified
    # via grep). It also persisted a fake `https://example.com/{id}.pdf` URL
    # into `pdf_processing_results.file_url` when called without an explicit
    # value, contradicting the "never persist file_url" rule. The
    # pdf_processing_results table still exists but is now write-orphan; if
    # ever needed, prefer reading from background_jobs + document_chunks +
    # document_images directly.

    async def save_knowledge_base_entries(self, document_id: str, chunks: list, images: list) -> dict:
        """
        Save extracted chunks and images to knowledge base.

        Args:
            document_id: Document identifier
            chunks: List of text chunks
            images: List of image data

        Returns:
            Dictionary with counts of saved entries
        """
        try:
            saved_chunks = 0
            saved_images = 0

            logger.info(f"💾 Starting knowledge base save for document: {document_id}")
            logger.info(f"   Chunks to save: {len(chunks)}")
            logger.info(f"   Images to save: {len(images)}")

            # First, ensure document exists in documents table and get workspace_id
            workspace_id = None
            try:
                # Check if document already exists and get its workspace_id
                doc_check = self._client.table('documents').select('id, workspace_id').eq('id', document_id).execute()

                if not doc_check.data:
                    # Create document record
                    doc_data = {
                        'id': document_id,
                        'filename': f"{document_id}.pdf",
                        'content_type': 'application/pdf',
                        'content': "",  # Will be updated with markdown content
                        'processing_status': 'completed',
                        'metadata': {
                            'source': 'mivaa_processing',
                            'chunks_count': len(chunks),
                            'images_count': len(images)
                        }
                    }

                    doc_response = self._client.table('documents').insert(doc_data).execute()
                    if doc_response.data:
                        logger.info(f"✅ Created document record: {document_id}")
                        workspace_id = doc_response.data[0].get('workspace_id') if doc_response.data else None
                    else:
                        logger.warning(f"⚠️ Failed to create document record: {document_id}")
                else:
                    logger.info(f"✅ Document already exists: {document_id}")
                    workspace_id = doc_check.data[0].get('workspace_id') if doc_check.data else None
                    logger.info(f"   Workspace ID: {workspace_id}")

            except Exception as doc_error:
                # Must not continue. Everything below writes CHILD rows keyed on
                # this document; without its workspace they are written with no
                # tenant at all, invisible to every scoped read afterwards and
                # counted as saved. A failed parent lookup ends the save.
                logger.error(f"❌ Document lookup/creation failed: {doc_error}")
                raise SupabaseQueryError(
                    f"could not establish document {document_id} before saving "
                    f"knowledge base entries: {doc_error}"
                ) from doc_error

            if not workspace_id:
                # Same reasoning as above, for the case where the lookup
                # succeeded but the document carries no tenant.
                raise TenancyViolation(
                    f"document {document_id} has no workspace_id - refusing to "
                    "write chunks and images with no tenant"
                )

            # Save text chunks to document_chunks table
            if chunks:
                chunk_entries = []
                for i, chunk in enumerate(chunks):
                    if isinstance(chunk, str) and chunk.strip():  # Only save non-empty string chunks
                        chunk_entry = {
                            'document_id': document_id,
                            'content': chunk,
                            'chunk_index': i,
                            'metadata': {
                                'source': 'mivaa_pdf_extraction',
                                'chunk_length': len(chunk),
                                'chunk_number': i + 1,
                                'page_number': 1  # Default page number
                            }
                        }
                        # Add workspace_id if available
                        if workspace_id:
                            chunk_entry['workspace_id'] = workspace_id
                        chunk_entries.append(chunk_entry)

                if chunk_entries:
                    logger.info(f"💾 Saving {len(chunk_entries)} chunks to document_chunks table (workspace_id: {workspace_id})")
                    response = self._client.table('document_chunks').insert(chunk_entries).execute()
                    saved_chunks = len(response.data) if response.data else 0
                    logger.info(f"✅ Saved {saved_chunks} chunks to database")
                else:
                    logger.warning("⚠️ No valid chunks to save")

            # Save images to document_images table
            if images:
                image_entries = []
                for i, image in enumerate(images):
                    # Handle different image data formats
                    if isinstance(image, dict):
                        # Try multiple possible keys for image URL
                        image_url = (
                            image.get('storage_url') or  # From _process_extracted_image
                            image.get('url') or
                            image.get('path') or
                            image.get('public_url') or
                            f"placeholder_image_{i}.jpg"
                        )
                        page_num = image.get('page') or image.get('page_number') or 1
                        caption = image.get('caption') or image.get('description') or f"Image {i+1}"

                        # Log if we're using a placeholder
                        if image_url.startswith('placeholder_'):
                            logger.warning(f"⚠️ Image {i} has no valid URL. Available keys: {list(image.keys())}")
                            logger.warning(f"   Image data sample: {str(image)[:200]}")
                    else:
                        # Handle string or other formats
                        image_url = str(image) if image else f"placeholder_image_{i}.jpg"
                        page_num = 1
                        caption = f"Image {i+1}"
                        logger.warning(f"⚠️ Image {i} is not a dict, type: {type(image)}")

                    # Only add images with valid URLs (not placeholders)
                    if not image_url.startswith('placeholder_'):
                        # Producer (pdf_processor) emits canonical
                        # extraction_layer values directly since the
                        # 2026-05-02 cleanup. Default to 'embedded' if the
                        # caller didn't set it (this batch path is used
                        # for legacy embedded-image extraction).
                        _layer_val = (
                            image.get('extraction_layer')
                            if isinstance(image, dict) else None
                        ) or 'embedded'
                        image_entry = {
                            'document_id': document_id,
                            'image_url': image_url,
                            'image_type': 'material_sample',
                            'caption': caption,
                            'page_number': page_num,
                            'confidence': 0.95,  # Default confidence
                            'processing_status': 'completed',
                            'extraction_layer': _layer_val,
                            'metadata': {
                                'source': 'mivaa_pdf_extraction',
                                'image_index': i,
                                'extraction_layer': _layer_val,
                                'storage_uploaded': image.get('storage_uploaded', False) if isinstance(image, dict) else False,
                                'storage_bucket': image.get('storage_bucket', 'pdf-tiles') if isinstance(image, dict) else 'pdf-tiles',
                                'original_data': image if isinstance(image, dict) else {'url': str(image)}
                            }
                        }
                        # Add workspace_id if available
                        if workspace_id:
                            image_entry['workspace_id'] = workspace_id
                        image_entries.append(image_entry)
                    else:
                        logger.warning(f"⚠️ Skipping image {i} - no valid URL found")

                if image_entries:
                    logger.info(f"💾 Saving {len(image_entries)} images to document_images table (out of {len(images)} total, workspace_id: {workspace_id})")
                    logger.info(f"   Sample image URLs: {[img['image_url'][:100] for img in image_entries[:3]]}")
                    try:
                        response = self._client.table('document_images').insert(image_entries).execute()
                        saved_images = len(response.data) if response.data else 0
                        logger.info(f"✅ Saved {saved_images} images to database")

                        if saved_images < len(image_entries):
                            logger.warning(f"⚠️ Only {saved_images}/{len(image_entries)} images were saved")
                    except Exception as insert_error:
                        logger.error(f"❌ Failed to insert images: {str(insert_error)}")
                        logger.error(f"   Error type: {type(insert_error).__name__}")
                        logger.error(f"   Sample image entry: {image_entries[0] if image_entries else 'N/A'}")
                        import traceback
                        logger.error(f"   Traceback: {traceback.format_exc()}")
                        raise
                else:
                    logger.warning(f"⚠️ No valid images to save (0 out of {len(images)} had valid URLs)")

            logger.info(f"✅ Knowledge base save completed: {saved_chunks} chunks, {saved_images} images")
            return {
                'chunks_saved': saved_chunks,
                'images_saved': saved_images,
                'total_saved': saved_chunks + saved_images
            }

        except (TenancyViolation, SupabaseQueryError):
            # Already precise about what went wrong. Flattening these into a
            # zero-count result is exactly the shape being removed here.
            raise
        except Exception as e:
            logger.error(f"❌ Failed to save knowledge base entries: {str(e)}")
            logger.error(f"   Error type: {type(e).__name__}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            # Zeros used to be returned here, which is indistinguishable from a
            # document that genuinely had nothing to save.
            raise SupabaseQueryError(
                f"failed to save knowledge base entries for {document_id}: {e}"
            ) from e

    async def upload_file(self, bucket_name: str, file_path: str, file_data: bytes,
                         content_type: str = None, upsert: bool = False) -> dict:
        """
        Upload file to Supabase Storage.

        Args:
            bucket_name: Name of the storage bucket
            file_path: Path where the file should be stored
            file_data: File content as bytes
            content_type: MIME type of the file
            upsert: Whether to overwrite existing files

        Returns:
            Dictionary with upload result
        """
        try:
            if not self._client:
                raise Exception("Supabase client not initialized")

            # Debug logging
            logger.info(f"🔍 DEBUG - Uploading file: bucket={bucket_name}, path={file_path}, data_type={type(file_data)}, data_len={len(file_data) if isinstance(file_data, bytes) else 'N/A'}")
            logger.info(f"🔍 DEBUG - Content type: {content_type}, Upsert: {upsert}")

            # Upload file to storage
            response = self._client.storage.from_(bucket_name).upload(
                file_path,
                file_data,
                file_options={
                    "content-type": content_type,
                    "upsert": "true" if upsert else "false"
                }
            )

            logger.info(f"🔍 DEBUG - Upload response type: {type(response)}, hasattr error: {hasattr(response, 'error')}")
            # Check if upload was successful
            # Handle httpx.Response (newer supabase-py versions)
            if hasattr(response, 'status_code'):
                if response.status_code not in [200, 201]:
                    error_msg = response.text if hasattr(response, 'text') else str(response)
                    raise Exception(f"Upload failed with status {response.status_code}: {error_msg}")
                response_data = response.json() if hasattr(response, 'json') else {}
            # Handle old-style response objects
            elif hasattr(response, 'error') and response.error:
                raise Exception(f"Upload failed: {response.error}")
            elif isinstance(response, dict):
                if response.get('error'):
                    raise Exception(f"Upload failed: {response.get('error')}")
                response_data = response
            else:
                response_data = {}

            # Get public URL
            url_response = self._client.storage.from_(bucket_name).get_public_url(file_path)

            logger.info(f"File uploaded successfully to {bucket_name}/{file_path}")
            return {
                "success": True,
                "data": response_data,
                "public_url": url_response,
                "bucket": bucket_name,
                "path": file_path
            }

        except Exception as e:
            logger.error(f"Failed to upload file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def upload_pdf_file(self, file_data: bytes, filename: str,
                             document_id: str = None) -> dict:
        """
        Upload PDF file to the pdf-documents bucket.

        Args:
            file_data: PDF file content as bytes
            filename: Original filename
            document_id: Optional document ID for organizing files

        Returns:
            Dictionary with upload result including public URL
        """
        try:
            # Generate unique file path
            import uuid
            from datetime import datetime

            if not document_id:
                document_id = str(uuid.uuid4())

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename.split('.')[-1] if '.' in filename else 'pdf'
            storage_path = f"documents/{document_id}/{timestamp}_{filename}"

            # Upload to pdf-documents bucket
            result = await self.upload_file(
                bucket_name="pdf-documents",
                file_path=storage_path,
                file_data=file_data,
                content_type="application/pdf",
                upsert=False
            )

            if result["success"]:
                result["document_id"] = document_id
                result["storage_path"] = storage_path

            return result

        except Exception as e:
            logger.error(f"Failed to upload PDF file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def upload_image_file(self, image_data: bytes, filename: str,
                               document_id: str, page_number: int = None) -> dict:
        """
        Upload extracted image to the pdf-tiles bucket.

        Args:
            image_data: Image content as bytes
            filename: Image filename
            document_id: Document ID for organizing images
            page_number: Page number where image was extracted

        Returns:
            Dictionary with upload result including public URL
        """
        try:
            # Generate storage path for extracted images
            from datetime import datetime

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            page_suffix = f"_page{page_number}" if page_number else ""
            file_extension = filename.split('.')[-1] if '.' in filename else 'png'
            storage_path = f"extracted/{document_id}/{timestamp}{page_suffix}_{filename}"

            # Determine content type
            content_type = "image/png"
            if file_extension.lower() in ['jpg', 'jpeg']:
                content_type = "image/jpeg"
            elif file_extension.lower() == 'webp':
                content_type = "image/webp"

            # Upload to pdf-tiles bucket
            result = await self.upload_file(
                bucket_name="pdf-tiles",
                file_path=storage_path,
                file_data=image_data,
                content_type=content_type,
                upsert=False
            )

            if result["success"]:
                result["document_id"] = document_id
                result["page_number"] = page_number
                result["storage_path"] = storage_path

            return result

        except Exception as e:
            logger.error(f"Failed to upload image file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def delete_file(self, bucket_name: str, file_path: str) -> dict:
        """
        Delete file from Supabase Storage.

        Args:
            bucket_name: Name of the storage bucket
            file_path: Path of the file to delete

        Returns:
            Dictionary with deletion result
        """
        try:
            if not self._client:
                raise Exception("Supabase client not initialized")

            response = self._client.storage.from_(bucket_name).remove([file_path])

            if response.error:
                raise Exception(f"Delete failed: {response.error}")

            logger.info(f"File deleted successfully from {bucket_name}/{file_path}")
            return {
                "success": True,
                "data": response.data
            }

        except Exception as e:
            logger.error(f"Failed to delete file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def delete_image_file(self, storage_path: str) -> bool:
        """
        Delete image file from Supabase Storage.

        Used to delete non-material images after AI classification.

        Args:
            storage_path: Storage path (e.g., 'pdf-tiles/doc_id/image.jpg')

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Extract bucket name from storage path
            # Format: 'pdf-tiles/doc_id/image.jpg'
            bucket_name = 'pdf-tiles'

            # Remove bucket name from path if present
            if storage_path.startswith(f'{bucket_name}/'):
                file_path = storage_path[len(f'{bucket_name}/'):]
            else:
                file_path = storage_path

            # Delete from Supabase Storage
            self.client.storage.from_(bucket_name).remove([file_path])

            logger.info(f"✅ Deleted image from Supabase: {storage_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete {storage_path} from Supabase: {e}")
            return False

    def close(self) -> None:
        """Close the Supabase client connection."""
        if self._client:
            # Supabase client doesn't require explicit closing
            # but we can reset the instance for cleanup
            self._client = None
            logger.info("Supabase client connection closed")


# Global instance
supabase_client = SupabaseClient()


def get_supabase_client() -> SupabaseClient:
    """
    Get the global Supabase client instance.
    
    Returns:
        SupabaseClient instance
    """
    return supabase_client


def initialize_supabase(settings: Settings) -> None:
    """
    Initialize the global Supabase client.
    
    Args:
        settings: Application settings containing Supabase configuration
    """
    supabase_client.initialize(settings)
