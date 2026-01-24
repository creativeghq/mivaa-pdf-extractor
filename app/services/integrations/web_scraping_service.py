"""
Web Scraping Service - Process Firecrawl scraping sessions and convert to products.

This service:
1. Fetches scraping sessions from Supabase
2. Retrieves scraped markdown content from scraping_pages
3. Uses ProductDiscoveryService.discover_products_from_text() to find products
4. Creates products in the database
5. Links images to products
6. Updates session status

ARCHITECTURE:
- Reuses existing product discovery pipeline (no changes to PDF/XML)
- Integrates with AsyncQueueService for background processing
- Uses same error handling patterns as PDF processing
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from app.services.discovery.product_discovery_service import ProductDiscoveryService
from app.services.core.supabase_client import get_supabase_client
from app.services.core.async_queue_service import AsyncQueueService
from app.services.chunking.unified_chunking_service import UnifiedChunkingService, ChunkingConfig, ChunkingStrategy
from app.services.images.image_download_service import ImageDownloadService
from app.services.images.image_processing_service import ImageProcessingService
from app.utils.retry_utils import retry_async
from app.utils.circuit_breaker import claude_breaker, gpt_breaker, CircuitBreakerError
from app.utils.timeout_guard import with_timeout, TimeoutError, TimeoutConstants
from app.services.tracking.web_scraping_stages import WebScrapingStage, get_web_scraping_progress
import re

# Sentry integration for error tracking and monitoring
try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

logger = logging.getLogger(__name__)


class WebScrapingError(Exception):
    """Base exception for web scraping errors"""
    pass


class WebScrapingTimeoutError(WebScrapingError):
    """Raised when scraping operation times out"""
    pass


class WebScrapingService:
    """
    Service for processing Firecrawl scraping sessions and converting to products.

    Features:
    - Background job processing via AsyncQueueService
    - Progress tracking in background_jobs table
    - Retry logic and error handling
    - Circuit breaker for AI API calls
    """

    # Image URL patterns to extract from markdown
    IMAGE_URL_PATTERNS = [
        r'!\[.*?\]\((https?://[^\s\)]+)\)',  # Markdown: ![alt](url)
        r'<img[^>]+src=["\']?(https?://[^\s"\'>]+)',  # HTML: <img src="url">
        r'(https?://[^\s<>"\']+\.(?:jpg|jpeg|png|gif|webp|bmp|svg))',  # Direct URLs
    ]

    def __init__(self, model: str = "claude", workspace_id: str = None):
        """
        Initialize service.

        Args:
            model: AI model to use for product discovery ("claude", "gpt")
            workspace_id: Workspace ID for image processing
        """
        self.logger = logger
        self.model = model
        self.workspace_id = workspace_id
        self.supabase = get_supabase_client()
        self.discovery_service = ProductDiscoveryService(model=model)
        self.async_queue = AsyncQueueService()

        # ✅ Initialize chunking service for quality scoring and smart chunking
        self.chunking_service = UnifiedChunkingService(ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID,
            max_chunk_size=1000,
            min_chunk_size=100,
            overlap_size=100
        ))

        # ✅ NEW: Initialize image services for full image processing
        self.image_downloader = ImageDownloadService()
        self.image_processor = None  # Lazy init with workspace_id

    def _get_image_processor(self, workspace_id: str) -> ImageProcessingService:
        """Get or create image processor for workspace."""
        if not self.image_processor or self.image_processor.workspace_id != workspace_id:
            self.image_processor = ImageProcessingService(workspace_id=workspace_id)
        return self.image_processor

    async def _log_stage(
        self,
        job_id: str,
        stage: WebScrapingStage,
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
            self.logger.info(f"[{job_id}] Stage: {stage.value} | Status: started")
        elif status == "completed":
            msg = f"[{job_id}] Stage: {stage.value} | Status: completed"
            if duration_ms is not None:
                msg += f" | Duration: {duration_ms}ms"
            if items is not None:
                msg += f" | Items: {items}"
            self.logger.info(msg)
        elif status == "failed":
            self.logger.error(f"[{job_id}] Stage: {stage.value} | Status: failed | Error: {error}")

        # Update background_jobs with current stage and progress
        try:
            progress = get_web_scraping_progress(stage)
            self.supabase.client.table('background_jobs').update({
                'current_stage': stage.value,
                'progress': progress,
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', job_id).execute()
        except Exception as e:
            self.logger.warning(f"Failed to update job stage: {e}")

    def _extract_image_urls_from_markdown(self, markdown_content: str) -> List[str]:
        """
        Extract image URLs from markdown content.

        Handles:
        - Markdown image syntax: ![alt](url)
        - HTML img tags: <img src="url">
        - Direct image URLs: https://...jpg

        Args:
            markdown_content: Markdown text to scan

        Returns:
            List of unique image URLs
        """
        image_urls = set()

        for pattern in self.IMAGE_URL_PATTERNS:
            matches = re.findall(pattern, markdown_content, re.IGNORECASE)
            for match in matches:
                url = match.strip()
                # Clean up URL - remove trailing punctuation
                url = url.rstrip('.,;:!?)')
                if url and len(url) > 10:  # Basic validation
                    image_urls.add(url)

        self.logger.info(f"   📷 Extracted {len(image_urls)} unique image URLs from markdown")
        return list(image_urls)

    async def process_scraping_session(
        self,
        session_id: str,
        workspace_id: str,
        categories: List[str] = None,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a scraping session and create products.

        Args:
            session_id: Scraping session ID
            workspace_id: Workspace ID
            categories: Categories to discover (default: ["products"])
            job_id: Optional background job ID for tracking

        Returns:
            Processing result with product count and status

        Raises:
            ValueError: If session not found or invalid
            RuntimeError: If processing fails
        """
        start_time = datetime.now()

        # Start Sentry transaction for monitoring
        if SENTRY_AVAILABLE:
            with sentry_sdk.start_transaction(
                op="web_scraping.process_session",
                name=f"Process Scraping Session {session_id}"
            ) as transaction:
                transaction.set_tag("session_id", session_id)
                transaction.set_tag("workspace_id", workspace_id)
                transaction.set_tag("model", self.model)
                transaction.set_tag("categories", ",".join(categories or ["products"]))

                return await self._process_session_internal(
                    session_id, workspace_id, categories, job_id, transaction
                )
        else:
            return await self._process_session_internal(
                session_id, workspace_id, categories, job_id, None
            )

    async def _process_session_internal(
        self,
        session_id: str,
        workspace_id: str,
        categories: Optional[List[str]],
        job_id: Optional[str],
        transaction: Optional[Any]
    ) -> Dict[str, Any]:
        """Internal method for processing session with Sentry tracking."""
        try:
            self.logger.info(f"🔍 Processing scraping session: {session_id}")

            # ✅ Stage: INITIALIZED
            if job_id:
                await self._log_stage(job_id, WebScrapingStage.INITIALIZED, "started")

            # Default to products only
            if categories is None:
                categories = ["products"]

            # Update job progress: Starting
            if job_id:
                await self.update_job_progress(job_id, progress=5, status="processing")

            # ============================================================
            # STEP 1: Fetch scraping session from Supabase
            # ============================================================
            session = await self._fetch_scraping_session(session_id)

            if not session:
                raise ValueError(f"Scraping session not found: {session_id}")

            self.logger.info(f"   Session: {session.get('source_url')}")
            self.logger.info(f"   Status: {session.get('status')}")
            self.logger.info(f"   Pages: {session.get('completed_pages', 0)}/{session.get('total_pages', 0)}")

            # Validate session status
            if session.get('status') not in ['completed', 'processing']:
                raise ValueError(f"Session status is '{session.get('status')}', expected 'completed' or 'processing'")

            # Update job progress: Session fetched
            if job_id:
                await self.update_job_progress(job_id, progress=15)

            # ============================================================
            # STEP 2: Fetch scraped markdown content from scraping_pages
            # ============================================================
            markdown_content = await self._fetch_scraped_markdown(session_id)

            if not markdown_content:
                self.logger.warning(f"⚠️ No markdown content found for session {session_id}")
                return {
                    "success": True,
                    "session_id": session_id,
                    "products_created": 0,
                    "message": "No content to process"
                }

            self.logger.info(f"   ✅ Retrieved {len(markdown_content):,} characters of markdown")

            # ✅ Stage: PAGES_SCRAPED
            if job_id:
                await self._log_stage(job_id, WebScrapingStage.PAGES_SCRAPED, "completed", items=session.get('completed_pages', 1))

            # Update job progress: Content fetched
            if job_id:
                await self.update_job_progress(job_id, progress=30)

            # ============================================================
            # STEP 3: Discover products using ProductDiscoveryService
            # ============================================================
            self.logger.info(f"🔍 Discovering products from scraped content...")

            # Use circuit breaker and timeout for AI discovery
            breaker = claude_breaker if self.model == "claude" else gpt_breaker

            try:
                # Wrap with circuit breaker
                catalog = await breaker.call(
                    self._discover_products_with_timeout,
                    markdown_content=markdown_content,
                    categories=categories,
                    workspace_id=workspace_id,
                    job_id=job_id
                )
            except CircuitBreakerError as e:
                self.logger.error(f"❌ Circuit breaker open for {self.model}: {e}")
                raise WebScrapingError(f"AI service temporarily unavailable: {e}") from e
            except TimeoutError as e:
                self.logger.error(f"❌ Product discovery timed out: {e}")
                raise WebScrapingTimeoutError(f"Product discovery timed out: {e}") from e

            self.logger.info(f"   ✅ Discovered {len(catalog.products)} products")

            # ✅ Stage: PRODUCTS_DISCOVERED
            if job_id:
                await self._log_stage(job_id, WebScrapingStage.PRODUCTS_DISCOVERED, "completed", items=len(catalog.products))

            # Update job progress: Products discovered
            if job_id:
                await self.update_job_progress(
                    job_id,
                    progress=70,
                    metadata={'products_discovered': len(catalog.products)}
                )

            # ============================================================
            # STEP 4: Create products in database (with chunks & embeddings)
            # ============================================================
            created_products = await self._create_products_in_database(
                catalog=catalog,
                workspace_id=workspace_id,
                session_id=session_id,
                source_url=session.get('source_url'),
                job_id=job_id
            )

            # Update job progress: Products created
            if job_id:
                await self.update_job_progress(
                    job_id,
                    progress=90,
                    metadata={'products_created': len(created_products)}
                )

            # ============================================================
            # STEP 5: Update session status
            # ============================================================
            await self._update_session_status(
                session_id=session_id,
                status="completed",
                materials_processed=len(created_products)
            )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Update job progress: Completed
            if job_id:
                await self.update_job_progress(
                    job_id,
                    progress=100,
                    status="completed",
                    metadata={
                        'products_created': len(created_products),
                        'processing_time_ms': processing_time,
                        'completed_at': datetime.utcnow().isoformat()
                    }
                )

            self.logger.info(f"✅ Session processing complete in {processing_time:.0f}ms:")
            self.logger.info(f"   Products created: {len(created_products)}")

            # ✅ Stage: COMPLETED
            if job_id:
                await self._log_stage(job_id, WebScrapingStage.COMPLETED, "completed", items=len(created_products))

            # Track success metrics in Sentry
            if SENTRY_AVAILABLE and transaction:
                transaction.set_measurement("processing_time", processing_time, "millisecond")
                transaction.set_measurement("products_created", len(created_products))
                transaction.set_data("markdown_length", len(markdown_content))
                transaction.set_data("pages_processed", session.get('completed_pages', 0))
                transaction.set_tag("status", "success")

                # Add breadcrumb for successful processing
                sentry_sdk.add_breadcrumb(
                    category="web_scraping",
                    message=f"Successfully processed session {session_id}",
                    level="info",
                    data={
                        "products_created": len(created_products),
                        "processing_time_ms": processing_time,
                        "model": self.model
                    }
                )

            return {
                "success": True,
                "session_id": session_id,
                "products_created": len(created_products),
                "product_ids": [p["id"] for p in created_products],
                "processing_time_ms": processing_time
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to process scraping session {session_id}: {e}")

            # Track error in Sentry
            if SENTRY_AVAILABLE:
                with sentry_sdk.configure_scope() as scope:
                    scope.set_tag("session_id", session_id)
                    scope.set_tag("workspace_id", workspace_id)
                    scope.set_tag("model", self.model)
                    scope.set_tag("error_type", type(e).__name__)
                    scope.set_context("scraping_session", {
                        "session_id": session_id,
                        "workspace_id": workspace_id,
                        "categories": categories,
                        "model": self.model,
                        "job_id": job_id
                    })

                # Capture exception with appropriate level
                if isinstance(e, (WebScrapingTimeoutError, CircuitBreakerError)):
                    sentry_sdk.capture_exception(e, level="warning")
                else:
                    sentry_sdk.capture_exception(e, level="error")

                # Mark transaction as failed
                if transaction:
                    transaction.set_tag("status", "failed")
                    transaction.set_tag("error_type", type(e).__name__)

            # Update job status to failed
            if job_id:
                try:
                    await self.update_job_progress(
                        job_id,
                        progress=0,
                        status="failed",
                        metadata={
                            'error': str(e),
                            'error_type': type(e).__name__,
                            'failed_at': datetime.utcnow().isoformat()
                        }
                    )
                except Exception as job_update_error:
                    self.logger.error(f"Failed to update job status: {job_update_error}")

            # Update session status to failed
            try:
                await self._update_session_status(
                    session_id=session_id,
                    status="failed",
                    error_message=str(e)
                )
            except Exception as update_error:
                self.logger.error(f"Failed to update session status: {update_error}")

            raise RuntimeError(f"Scraping session processing failed: {str(e)}") from e

    @retry_async(
        max_attempts=3,
        base_delay=2.0,
        max_delay=30.0,
        exceptions=(TimeoutError, asyncio.TimeoutError, Exception)
    )
    async def _discover_products_with_timeout(
        self,
        markdown_content: str,
        categories: List[str],
        workspace_id: str,
        job_id: Optional[str]
    ):
        """
        Discover products with timeout and retry logic.

        This method wraps the product discovery call with:
        - Timeout protection (5 minutes for web scraping)
        - Automatic retry with exponential backoff
        - Error logging and tracking

        Args:
            markdown_content: Scraped markdown text
            categories: Categories to discover
            workspace_id: Workspace ID
            job_id: Optional job ID for tracking

        Returns:
            ProductCatalog with discovered products

        Raises:
            TimeoutError: If discovery takes too long
            Exception: If discovery fails after retries
        """
        # Calculate timeout based on content size
        # Base: 2 minutes, +30s per 10k characters
        content_size_kb = len(markdown_content) / 1000
        timeout_seconds = min(
            TimeoutConstants.PRODUCT_DISCOVERY_STAGE_0B,  # Max 5 minutes
            120 + (content_size_kb / 10) * 30  # Scale with content size
        )

        self.logger.info(f"   Product discovery timeout: {timeout_seconds:.0f}s (content: {content_size_kb:.1f}KB)")

        # Execute with timeout
        return await with_timeout(
            self.discovery_service.discover_products_from_text(
                markdown_text=markdown_content,
                source_type="web_scraping",
                categories=categories,
                workspace_id=workspace_id,
                job_id=job_id
            ),
            timeout_seconds=timeout_seconds,
            operation_name="Web scraping product discovery"
        )

    async def _fetch_scraping_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch scraping session from Supabase.

        Args:
            session_id: Session ID

        Returns:
            Session data or None if not found
        """
        try:
            response = self.supabase.client.table("scraping_sessions").select("*").eq("id", session_id).single().execute()

            if response.data:
                return response.data
            return None

        except Exception as e:
            self.logger.error(f"Failed to fetch scraping session: {e}")
            return None

    async def _fetch_scraped_markdown(self, session_id: str) -> str:
        """
        Fetch and combine markdown content from all scraping_pages for a session.

        Args:
            session_id: Session ID

        Returns:
            Combined markdown content from all pages
        """
        try:
            # Fetch all pages for this session
            response = self.supabase.client.table("scraping_pages").select("*").eq("session_id", session_id).order("page_index").execute()

            if not response.data:
                return ""

            # Combine markdown from all pages
            # Note: Firecrawl stores markdown in the 'markdown' field or similar
            # We need to check the actual field name in scraping_pages
            markdown_parts = []

            for page in response.data:
                # Try different possible field names for markdown content
                markdown = page.get("markdown") or page.get("content") or page.get("markdown_content") or ""

                if markdown:
                    markdown_parts.append(markdown)
                    markdown_parts.append("\n\n---\n\n")  # Page separator

            combined_markdown = "".join(markdown_parts)

            self.logger.info(f"   Combined markdown from {len(response.data)} pages")

            return combined_markdown

        except Exception as e:
            self.logger.error(f"Failed to fetch scraped markdown: {e}")
            return ""

    @retry_async(
        max_attempts=3,
        base_delay=1.0,
        max_delay=10.0,
        exceptions=(Exception,)
    )
    async def _create_products_in_database(
        self,
        catalog: Any,
        workspace_id: str,
        session_id: str,
        source_url: str,
        job_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Create products in database from discovered catalog.

        Includes retry logic for database failures.

        Args:
            catalog: ProductCatalog from discovery
            workspace_id: Workspace ID
            session_id: Scraping session ID
            source_url: Source URL

        Returns:
            List of created product records
        """
        created_products = []

        try:
            for product in catalog.products:
                # Prepare product data
                product_data = {
                    "workspace_id": workspace_id,
                    "name": product.name,
                    "description": product.description or "",
                    "metadata": product.metadata or {},
                    "source_type": "web_scraping",
                    "source_job_id": job_id,
                    "import_batch_id": f"scraping_{session_id}",
                    "source_url": source_url,
                    "source_reference": session_id,
                    "confidence_score": product.confidence,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }

                # Insert product
                response = self.supabase.client.table("products").insert(product_data).execute()

                if response.data:
                    created_product = response.data[0]
                    product_id = created_product['id']
                    created_products.append(created_product)
                    self.logger.info(f"   ✅ Created product: {product.name} (ID: {product_id})")

                    # Create chunks and queue embedding generation
                    await self._create_product_chunks_and_embeddings(
                        product_id=product_id,
                        product_name=product.name,
                        description=product.description or "",
                        workspace_id=workspace_id,
                        job_id=job_id
                    )

                    # ✅ NEW: Extract and process images from product description
                    if product.description:
                        await self._process_product_images(
                            product_id=product_id,
                            product_name=product.name,
                            content=product.description,
                            workspace_id=workspace_id,
                            job_id=job_id
                        )
                else:
                    self.logger.warning(f"   ⚠️ Failed to create product: {product.name}")

            return created_products

        except Exception as e:
            self.logger.error(f"Failed to create products in database: {e}")
            raise

    async def _create_product_chunks_and_embeddings(
        self,
        product_id: str,
        product_name: str,
        description: str,
        workspace_id: str,
        job_id: str = None
    ) -> None:
        """
        Create document chunks and queue embedding generation for a product.

        ✅ ENHANCED: Uses smart chunking for long descriptions (>1500 chars)
        to improve retrieval quality with proper semantic boundaries.

        Args:
            product_id: Product ID
            product_name: Product name
            description: Product description
            workspace_id: Workspace ID
            job_id: Job ID for source tracking
        """
        try:
            # Skip if description is too short
            if not description or len(description) < 50:
                self.logger.info(f"   ⏭️ Skipping chunk creation for {product_name} - insufficient content")
                return

            chunk_ids = []

            # ✅ NEW: Use smart chunking for long descriptions (>1500 chars)
            if len(description) > 1500:
                self.logger.info(f"   📚 Long description ({len(description)} chars) - using smart chunking for {product_name}")

                # Use UnifiedChunkingService for smart chunking
                chunks = await self.chunking_service.chunk_text(
                    text=description,
                    document_id=product_id,
                    metadata={
                        "source": "web_scraping",
                        "product_id": product_id,
                        "product_name": product_name,
                        "auto_generated": True
                    }
                )

                self.logger.info(f"   📚 Created {len(chunks)} chunks for long description")

                # Insert all chunks
                for chunk in chunks:
                    chunk_record = {
                        "document_id": product_id,
                        "workspace_id": workspace_id,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "quality_score": chunk.quality_score,
                        "source_type": "web_scraping",
                        "source_job_id": job_id,
                        "metadata": {
                            **chunk.metadata,
                            "quality_score": chunk.quality_score
                        }
                    }

                    chunk_response = self.supabase.client.table('document_chunks').insert(chunk_record).execute()
                    if chunk_response.data:
                        chunk_ids.append({'id': chunk_response.data[0]['id']})
            else:
                # Short description - single chunk (original behavior)
                chunk_record = {
                    "document_id": product_id,
                    "workspace_id": workspace_id,
                    "content": description,
                    "chunk_index": 0,
                    "source_type": "web_scraping",
                    "source_job_id": job_id,
                    "metadata": {
                        "source": "web_scraping",
                        "product_id": product_id,
                        "product_name": product_name,
                        "auto_generated": True
                    }
                }

                chunk_response = self.supabase.client.table('document_chunks').insert(chunk_record).execute()

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
                    self.supabase.client.table('document_chunks').update({
                        'quality_score': quality_score,
                        'metadata': {
                            **chunk_record['metadata'],
                            'quality_score': quality_score
                        }
                    }).eq('id', chunk_id).execute()

                    chunk_ids.append({'id': chunk_id})

            if chunk_ids:
                self.logger.info(f"   ✅ Created {len(chunk_ids)} chunks for product {product_name}")

                # Queue for embedding generation (async)
                async_queue = AsyncQueueService()
                await async_queue.queue_ai_analysis_jobs(
                    document_id=product_id,
                    chunks=chunk_ids,
                    analysis_type='embedding_generation',
                    priority=0
                )

        except Exception as e:
            self.logger.error(f"   ❌ Failed to create chunks for product {product_name}: {e}")
            # Don't raise - product is already created

    async def _process_product_images(
        self,
        product_id: str,
        product_name: str,
        content: str,
        workspace_id: str,
        job_id: str = None
    ) -> None:
        """
        Extract, download, classify, and generate CLIP embeddings for product images.

        ✅ NEW: Full image processing pipeline for scraped products:
        1. Extract image URLs from markdown content
        2. Download images to Supabase Storage
        3. Classify images (material vs non-material) using Qwen/Claude
        4. Generate CLIP embeddings for ALL images (for search and agent)
        5. Save to document_images with embeddings

        Args:
            product_id: Product ID
            product_name: Product name
            content: Product description/content with potential image URLs
            workspace_id: Workspace ID
            job_id: Job ID for source tracking
        """
        try:
            # Step 1: Extract image URLs from markdown
            image_urls = self._extract_image_urls_from_markdown(content)

            if not image_urls:
                self.logger.info(f"   📷 No images found in content for {product_name}")
                return

            self.logger.info(f"   📷 Found {len(image_urls)} images for {product_name}")

            # Step 2: Download images to Supabase Storage
            downloaded_images = await self.image_downloader.download_images(
                urls=image_urls,
                job_id=job_id or product_id,
                workspace_id=workspace_id,
                max_concurrent=5
            )

            if not downloaded_images:
                self.logger.warning(f"   ⚠️ No images downloaded for {product_name}")
                return

            self.logger.info(f"   ✅ Downloaded {len(downloaded_images)} images for {product_name}")

            # Step 3: Get image processor and classify images
            image_processor = self._get_image_processor(workspace_id)

            # Prepare images for classification (need 'path' key for ImageProcessingService)
            # Since we downloaded to storage, we need to fetch them temporarily for classification
            images_for_classification = []
            for img in downloaded_images:
                if img.get('success'):
                    images_for_classification.append({
                        'storage_url': img.get('storage_url'),
                        'url': img.get('storage_url'),  # For fallback in classify_images
                        'public_url': img.get('storage_url'),
                        'filename': img.get('filename'),
                        'path': None,  # Will trigger download from storage in classify_images
                        'content_type': img.get('content_type'),
                        'size_bytes': img.get('size_bytes'),
                        'original_url': img.get('original_url'),
                        'storage_path': img.get('storage_path'),
                        'product_id': product_id,
                        'product_name': product_name
                    })

            if not images_for_classification:
                return

            # Step 4: Classify images (material vs non-material)
            # Note: For web scraping, we process ALL images for CLIP embeddings
            # but still run classification for metadata purposes
            self.logger.info(f"   🤖 Classifying {len(images_for_classification)} images for {product_name}")

            material_images, non_material_images = await image_processor.classify_images(
                extracted_images=images_for_classification,
                confidence_threshold=0.6,
                job_id=job_id  # Track AI cost per job
            )

            self.logger.info(f"   ✅ Classification: {len(material_images)} material, {len(non_material_images)} non-material")

            # Step 5: Generate CLIP embeddings for ALL images (as requested)
            # Even non-material images get embeddings for search and agent responses
            all_images = material_images + non_material_images

            if all_images:
                self.logger.info(f"   🎨 Generating CLIP embeddings for ALL {len(all_images)} images")

                result = await image_processor.save_images_and_generate_clips(
                    material_images=all_images,  # Process all images
                    document_id=product_id,
                    workspace_id=workspace_id,
                    batch_size=10,
                    max_retries=3,
                    job_id=job_id  # Track AI cost per job
                )

                self.logger.info(f"   ✅ Saved {result.get('images_saved', 0)} images with {result.get('clip_embeddings_generated', 0)} CLIP embeddings")

                # Create product-image relationships
                await self._link_images_to_product(
                    product_id=product_id,
                    image_count=result.get('images_saved', 0),
                    workspace_id=workspace_id,
                    job_id=job_id
                )

        except Exception as e:
            self.logger.error(f"   ❌ Failed to process images for {product_name}: {e}")
            # Don't raise - product is already created

    async def _link_images_to_product(
        self,
        product_id: str,
        image_count: int,
        workspace_id: str,
        job_id: str = None
    ) -> None:
        """
        Create product-image relationships in image_product_associations.

        Args:
            product_id: Product ID
            image_count: Number of images saved
            workspace_id: Workspace ID
            job_id: Job ID for tracking
        """
        try:
            # Get images that were just saved for this product
            response = self.supabase.client.table('document_images')\
                .select('id')\
                .eq('document_id', product_id)\
                .order('created_at', desc=True)\
                .limit(image_count)\
                .execute()

            if not response.data:
                return

            # Create associations
            associations = []
            for idx, img in enumerate(response.data):
                score = 1.0 - (idx * 0.05)  # First image gets highest score
                associations.append({
                    "product_id": product_id,
                    "image_id": img['id'],
                    "spatial_score": 0.0,
                    "caption_score": 0.0,
                    "clip_score": 0.0,
                    "overall_score": score,
                    "confidence": score,
                    "reasoning": "web_scraping_extracted",
                    "metadata": {
                        "source": "web_scraping",
                        "job_id": job_id,
                        "import_index": idx
                    }
                })

            if associations:
                self.supabase.client.table('image_product_associations').insert(associations).execute()
                self.logger.info(f"   ✅ Created {len(associations)} product-image associations")

        except Exception as e:
            self.logger.error(f"   ❌ Failed to create product-image associations: {e}")

    @retry_async(
        max_attempts=3,
        base_delay=0.5,
        max_delay=5.0,
        exceptions=(Exception,)
    )
    async def _update_session_status(
        self,
        session_id: str,
        status: str,
        materials_processed: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update scraping session status with retry logic.

        Args:
            session_id: Session ID
            status: New status ("processing", "completed", "failed")
            materials_processed: Number of materials processed
            error_message: Error message if failed
        """
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }

            if materials_processed is not None:
                update_data["materials_processed"] = materials_processed

            if error_message:
                # Store error in scraping_config metadata
                update_data["scraping_config"] = {
                    "error": error_message,
                    "failed_at": datetime.utcnow().isoformat()
                }

            self.supabase.client.table("scraping_sessions").update(update_data).eq("id", session_id).execute()

            self.logger.info(f"   ✅ Updated session status to: {status}")

        except Exception as e:
            self.logger.error(f"Failed to update session status: {e}")
            raise


    async def create_background_job(
        self,
        session_id: str,
        workspace_id: str,
        categories: List[str] = None,
        priority: int = 0
    ) -> str:
        """
        Create a background job for processing a scraping session.

        This allows long-running scraping sessions to be processed asynchronously
        without blocking the API response.

        Args:
            session_id: Scraping session ID
            workspace_id: Workspace ID
            categories: Categories to discover (default: ["products"])
            priority: Job priority (0 = normal, higher = more important)

        Returns:
            Job ID

        Raises:
            Exception: If job creation fails
        """
        import uuid

        try:
            job_id = str(uuid.uuid4())

            if categories is None:
                categories = ["products"]

            # Create background job record
            job_data = {
                'id': job_id,
                'job_type': 'web_scraping_processing',
                'status': 'pending',
                'progress': 0,
                'priority': priority,
                'payload': {
                    'session_id': session_id,
                    'workspace_id': workspace_id,
                    'categories': categories,
                    'model': self.model
                },
                'metadata': {
                    'session_id': session_id,
                    'workspace_id': workspace_id,
                    'categories': categories
                },
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }

            result = self.supabase.client.table('background_jobs').insert(job_data).execute()

            if result.data and len(result.data) > 0:
                self.logger.info(f"✅ Created background job {job_id} for scraping session {session_id}")
                return job_id
            else:
                raise Exception("Failed to create background job")

        except Exception as e:
            self.logger.error(f"❌ Failed to create background job: {e}")
            raise

    async def update_job_progress(
        self,
        job_id: str,
        progress: int,
        status: str = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Update background job progress.

        Args:
            job_id: Job ID
            progress: Progress percentage (0-100)
            status: Optional status update ("pending", "processing", "completed", "failed")
            metadata: Optional metadata to merge with existing metadata
        """
        try:
            update_data = {
                'progress': min(100, max(0, progress)),
                'updated_at': datetime.utcnow().isoformat()
            }

            if status:
                update_data['status'] = status

            if metadata:
                # Fetch current metadata and merge
                current_job = self.supabase.client.table('background_jobs').select('metadata').eq('id', job_id).single().execute()
                if current_job.data:
                    current_metadata = current_job.data.get('metadata', {})
                    current_metadata.update(metadata)
                    update_data['metadata'] = current_metadata

            self.supabase.client.table('background_jobs').update(update_data).eq('id', job_id).execute()

        except Exception as e:
            self.logger.error(f"Failed to update job progress: {e}")
            # Don't raise - progress updates are non-critical


