"""
RAG Service - Direct Vector DB Implementation

This service provides multi-vector search capabilities using VECS collections
as the single source of truth for image embeddings.

Search Strategy:
- Multi-Vector Search: Combines 7 vectors in parallel
  1. document_chunks.text_embedding (1024D Voyage AI, 20%) - Semantic text
  2. vecs.image_slig_embeddings (768D SLIG, 20%) - General visual similarity
  3. vecs.image_color_embeddings (768D SLIG, 12.5%) - Color palette matching
  4. vecs.image_texture_embeddings (768D SLIG, 12.5%) - Texture pattern matching
  5. vecs.image_style_embeddings (768D SLIG, 12.5%) - Design style matching
  6. vecs.image_material_embeddings (768D SLIG, 12.5%) - Material type matching
  7. vecs.image_understanding_embeddings (1024D Voyage from Qwen3-VL, 10%) - Vision-understanding
"""

import logging
import os
from typing import Dict, List, Optional, Any
import asyncio
import time
import io
import gc

# Import utilities
from app.utils.circuit_breaker import CircuitBreaker, CircuitBreakerError
from app.utils.memory_monitor import memory_monitor

# Import services
from ..embeddings.real_embeddings_service import RealEmbeddingsService
from ..core.supabase_client import get_supabase_client
from ..embeddings.vecs_service import get_vecs_service
from ..core.ai_client_service import get_ai_client_service
from ..core.ai_call_logger import AICallLogger
from ..chunking.unified_chunking_service import UnifiedChunkingService, ChunkingConfig, ChunkingStrategy


class RAGService:
    """
    RAG Service for multi-vector search using direct vector database queries.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RAG service with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self._available = True  # Service is available after initialization

        # Vision-analysis and image classification both run on Anthropic
        # Claude Opus 4.7 (post-Qwen-removal). No HF endpoint plumbing
        # needed here — the Anthropic API key is read on demand inside
        # _analyze_image_material / _classify_image_material via os.getenv.

        # Initialize services
        try:
            self.embeddings_service = RealEmbeddingsService()
            self.supabase_client = get_supabase_client()
            self.vecs_service = get_vecs_service()
            self.ai_client_service = get_ai_client_service()
            self.ai_logger = AICallLogger()

            # Initialize chunking service
            chunking_config = ChunkingConfig(
                strategy=ChunkingStrategy.HYBRID,
                max_chunk_size=self.config.get('chunk_size', 1000),
                min_chunk_size=100,
                overlap_size=self.config.get('chunk_overlap', 100),
                preserve_structure=True,
                split_on_sentences=True,
                split_on_paragraphs=True
            )
            self.chunking_service = UnifiedChunkingService(chunking_config)

            # Circuit breaker for resilience
            self.circuit_breaker = CircuitBreaker(
                name="RAG Service",
                failure_threshold=5,
                timeout_seconds=60
            )

            # Load classification prompt from database
            self.classification_prompt = self._load_classification_prompt()

            self.logger.info("✅ RAG Service initialized")
        except Exception as e:
            self.logger.error(f"❌ RAG Service initialization failed: {e}")
            self._available = False
            raise

    def _load_classification_prompt(self) -> Optional[str]:
        """Load image classification prompt from database."""
        try:
            result = self.supabase_client.client.table('prompts')\
                .select('prompt_text')\
                .eq('prompt_type', 'classification')\
                .eq('stage', 'image_analysis')\
                .eq('category', 'image_classification')\
                .eq('is_active', True)\
                .order('version', desc=True)\
                .limit(1)\
                .execute()

            if result.data and len(result.data) > 0:
                self.logger.info("✅ Loaded classification prompt from database")
                return result.data[0]['prompt_text']
            else:
                self.logger.warning("⚠️ Classification prompt not found in database. Add via /admin/ai-configs - classification will fail!")
                return None

        except Exception as e:
            self.logger.error(f"❌ Failed to load classification prompt from database: {e}")
            return None

    @property
    def available(self) -> bool:
        """Check if RAG service is available and ready to use."""
        return self._available

    async def health_check(self) -> Dict[str, Any]:
        """Check health of RAG service and dependencies."""
        try:
            return {
                "status": "healthy",
                "embeddings_service": "available",
                "vector_database": "available",
                "timestamp": time.time()
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

    async def index_pdf_content(
        self,
        pdf_content: bytes = None,
        document_id: str = None,
        metadata: Optional[Dict[str, Any]] = None,
        catalog: Optional[Any] = None,
        page_chunks: Optional[List[Dict[str, Any]]] = None,
        progress_callback: Optional[callable] = None,
        layout_regions_by_page: Optional[Dict[int, List[Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        """
        Index PDF content by extracting text, chunking, and generating embeddings.

        Args:
            pdf_content: PDF file content as bytes. Only used if page_chunks is None.
            document_id: Unique document identifier
            metadata: Document metadata including workspace_id, filename, etc.
            catalog: Optional product catalog for category tagging
            page_chunks: Optional page-aware text data (preserves page numbers).
                         When provided, PDF extraction is skipped.
            progress_callback: Optional async callback for progress updates (current, total, step_name)
            layout_regions_by_page: Optional dict mapping 1-based page_number to YOLO layout regions

        Returns:
            Dict containing:
                - chunks_created: Number of chunks created
                - chunk_ids: List of created chunk IDs
                - success: Boolean indicating success
                - quality_metrics: Chunk quality tracking metrics
        """
        try:
            start_time = time.time()
            metadata = metadata or {}
            workspace_id = metadata.get('workspace_id', 'default')

            self.logger.info(f"📝 Indexing PDF content for document {document_id}")

            # Step 1: Get text with page information
            pages = None
            if page_chunks:
                self.logger.info(f"   ✅ Using page-aware data ({len(page_chunks)} pages)")
                pages = page_chunks
            else:
                # Extract text from PDF using PyMuPDF4LLM with page_chunks=True
                try:
                    import pymupdf4llm
                    import tempfile
                    # Do NOT re-import `os` here — it's imported at module level (line 19).
                    # A local `import os` rebinds it function-scoped and makes the
                    # `os.getenv` calls below crash with UnboundLocalError when this
                    # branch is skipped (caller-provided page_chunks path).

                    # Save PDF bytes to temporary file (pymupdf4llm needs a file path)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(pdf_content)
                        tmp_path = tmp_file.name

                    try:
                        # Extract markdown with page metadata
                        self.logger.info(f"   📄 Extracting PDF with page metadata...")
                        pages = pymupdf4llm.to_markdown(tmp_path, page_chunks=True)

                        # Calculate total characters
                        total_chars = sum(len(page.get('text', '')) for page in pages)
                        self.logger.info(f"   ✅ Extracted {len(pages)} pages, {total_chars} characters from PDF")
                    finally:
                        # Clean up temporary file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)

                except Exception as e:
                    self.logger.error(f"   ❌ PDF text extraction failed: {e}")
                    return {
                        "success": False,
                        "error": f"PDF extraction failed: {str(e)}",
                        "chunks_created": 0,
                        "chunk_ids": []
                    }

            # Step 2: Create chunks using UnifiedChunkingService with page awareness
            try:
                # Log layout-aware chunking status
                if layout_regions_by_page:
                    self.logger.info(f"   📐 Layout-aware chunking enabled ({len(layout_regions_by_page)} pages with regions)")

                chunks = await self.chunking_service.chunk_pages(
                    pages=pages,
                    document_id=document_id,
                    metadata=metadata,
                    layout_regions_by_page=layout_regions_by_page  # Pass layout regions
                )

                self.logger.info(f"   ✅ Created {len(chunks)} chunks from {len(pages)} pages")

                # Drop pages now — they can be 100K+ chars and are no longer needed.
                del pages
                gc.collect()
                self.logger.info(f"   🧹 Released pages from memory")

            except Exception as e:
                self.logger.error(f"   ❌ Chunking failed: {e}")
                return {
                    "success": False,
                    "error": f"Chunking failed: {str(e)}",
                    "chunks_created": 0,
                    "chunk_ids": []
                }

            # Step 3: store chunks and generate embeddings in batches to avoid OOM.
            from app.utils.memory_monitor import MemoryPressureMonitor

            memory_monitor = MemoryPressureMonitor()
            # Batch size for chunk embedding — 15 is the proven-optimal default
            # (commit 74e7a73) but can be tuned via env var without redeploy.
            CHUNK_BATCH_SIZE = int(os.getenv('RAG_CHUNK_BATCH_SIZE', '15') or '15')
            MEMORY_THRESHOLD_PERCENT = float(os.getenv('RAG_MEMORY_THRESHOLD_PERCENT', '85.0') or '85.0')
            total_chunks = len(chunks)

            chunk_ids = []
            chunks_created = 0
            total_embeddings_stored = 0  # Track total embeddings across all batches

            try:
                initial_mem = memory_monitor.get_memory_stats()
                self.logger.info(f"🧠 Initial memory: {initial_mem.percent_used:.1f}% ({initial_mem.used_mb:.0f}MB / {initial_mem.total_mb:.0f}MB)")

                if initial_mem.percent_used > MEMORY_THRESHOLD_PERCENT:
                    self.logger.warning(f"⚠️ High memory usage detected ({initial_mem.percent_used:.1f}%) - proceeding with caution")
            except Exception as mem_error:
                self.logger.debug(f"Memory monitoring unavailable: {mem_error}")

            self.logger.info(f"🔄 Processing {total_chunks} chunks in batches of {CHUNK_BATCH_SIZE}")

            for batch_start in range(0, total_chunks, CHUNK_BATCH_SIZE):
                batch_end = min(batch_start + CHUNK_BATCH_SIZE, total_chunks)
                batch_chunks = chunks[batch_start:batch_end]
                batch_num = (batch_start // CHUNK_BATCH_SIZE) + 1
                total_batches = (total_chunks + CHUNK_BATCH_SIZE - 1) // CHUNK_BATCH_SIZE

                # Report progress via callback for UI updates
                if progress_callback:
                    try:
                        await progress_callback(
                            current=batch_start,
                            total=total_chunks,
                            step_name=f"Processing chunks (batch {batch_num}/{total_batches})"
                        )
                    except Exception as callback_error:
                        self.logger.debug(f"Progress callback failed: {callback_error}")

                # Log memory usage before batch and check threshold
                try:
                    mem_stats = memory_monitor.get_memory_stats()
                    self.logger.info(f"🧠 Memory before batch {batch_num}/{total_batches}: {mem_stats.percent_used:.1f}% ({mem_stats.used_mb:.0f}MB / {mem_stats.total_mb:.0f}MB)")

                    if mem_stats.percent_used > MEMORY_THRESHOLD_PERCENT:
                        self.logger.warning(f"⚠️ Memory usage high ({mem_stats.percent_used:.1f}%) - forcing garbage collection")
                        gc.collect()

                        # Re-check after GC
                        mem_stats_after_gc = memory_monitor.get_memory_stats()
                        self.logger.info(f"🧠 Memory after GC: {mem_stats_after_gc.percent_used:.1f}% ({mem_stats_after_gc.used_mb:.0f}MB / {mem_stats_after_gc.total_mb:.0f}MB)")

                        if mem_stats_after_gc.percent_used > MEMORY_THRESHOLD_PERCENT:
                            self.logger.error(f"❌ Memory usage still high after GC ({mem_stats_after_gc.percent_used:.1f}%) - aborting to prevent OOM")
                            return {
                                "success": False,
                                "error": f"Memory threshold exceeded ({mem_stats_after_gc.percent_used:.1f}% > {MEMORY_THRESHOLD_PERCENT}%)",
                                "chunks_created": chunks_created,
                                "chunk_ids": chunk_ids
                            }
                except Exception as mem_error:
                    self.logger.debug(f"Memory monitoring unavailable: {mem_error}")

                # Collect batch data
                batch_chunk_ids = []
                batch_texts = []
                batch_chunk_records = []

                # Step 3a: Store chunks in database
                self.logger.info(f"   📝 Storing {len(batch_chunks)} chunks in database for batch {batch_num}/{total_batches}...")
                for chunk in batch_chunks:
                    try:
                        if not chunk.content or not chunk.content.strip():
                            self.logger.warning(f"   ⚠️ Skipping chunk {chunk.chunk_index} - empty content")
                            continue

                        # Prepare chunk record for database
                        chunk_record = {
                            'document_id': document_id,
                            'workspace_id': workspace_id,
                            'content': chunk.content.strip(),
                            'chunk_index': chunk.chunk_index,
                            'metadata': {
                                **chunk.metadata,
                                'start_position': chunk.start_position,
                                'end_position': chunk.end_position,
                                'quality_score': chunk.quality_score,
                                'total_chunks': chunk.total_chunks
                            },
                            'quality_score': chunk.quality_score  # top-level column drives dashboard metrics
                        }

                        # Add product_id as top-level field if available in metadata
                        if metadata.get('product_id'):
                            chunk_record['product_id'] = metadata.get('product_id')
                            chunk_record['metadata']['product_id'] = metadata.get('product_id')
                            chunk_record['metadata']['product_name'] = metadata.get('product_name')

                        # Add catalog category if available
                        if catalog and hasattr(catalog, 'catalog_factory'):
                            chunk_record['metadata']['catalog_factory'] = catalog.catalog_factory

                        # Insert chunk into database
                        result = self.supabase_client.client.table('document_chunks').insert(chunk_record).execute()

                        if result.data and len(result.data) > 0:
                            chunk_id = result.data[0]['id']
                            batch_chunk_ids.append(chunk_id)
                            batch_texts.append(chunk.content)
                            batch_chunk_records.append(chunk_record)
                            chunk_ids.append(chunk_id)
                            chunks_created += 1

                    except Exception as e:
                        self.logger.warning(f"   ⚠️ Failed to store chunk {chunk.chunk_index}: {e}")
                        continue

                self.logger.info(f"   ✅ Stored {len(batch_chunk_ids)} chunks in database for batch {batch_num}/{total_batches}")

                # Step 3b: Generate embeddings for ENTIRE BATCH in ONE API call
                if batch_chunk_ids:
                    # Initialize embedding_vectors to prevent NameError in cleanup
                    embedding_vectors = []

                    try:
                        self.logger.info(f"   🔄 Generating embeddings for batch {batch_num}/{total_batches} ({len(batch_texts)} chunks)...")

                        # Use batch embedding generation (Voyage AI or OpenAI)
                        embedding_vectors = await self.embeddings_service.generate_batch_embeddings(
                            texts=batch_texts,
                            dimensions=1024,  # matches DB schema vector(1024)
                            input_type="document"
                        )

                        if embedding_vectors:
                            # Prepare batch update records
                            updates = []
                            embeddings_stored = 0

                            for chunk_id, embedding in zip(batch_chunk_ids, embedding_vectors):
                                if embedding:
                                    updates.append({
                                        'id': chunk_id,
                                        'text_embedding': embedding
                                    })
                                    embeddings_stored += 1
                                else:
                                    self.logger.warning(f"   ⚠️ No embedding generated for chunk {chunk_id}")

                            # Use UPDATE per row, not UPSERT — UPSERT falls back to INSERT on
                            # missing IDs and trips the NOT NULL `content` constraint.
                            if updates:
                                embeddings_stored = 0
                                for update in updates:
                                    try:
                                        self.supabase_client.client.table('document_chunks')\
                                            .update({'text_embedding': update['text_embedding'], 'has_text_embedding': True})\
                                            .eq('id', update['id'])\
                                            .execute()
                                        embeddings_stored += 1
                                    except Exception as individual_update_error:
                                        self.logger.warning(f"   ⚠️ Failed to update chunk {update['id']}: {individual_update_error}")

                                total_embeddings_stored += embeddings_stored
                                self.logger.info(f"   ✅ Stored {embeddings_stored}/{len(updates)} embeddings")

                    except Exception as batch_embedding_error:
                        self.logger.error(f"   ❌ Batch embedding generation failed: {batch_embedding_error}", exc_info=True)

                        # Fallback to individual embedding generation
                        self.logger.info(f"   ⚠️ Falling back to individual embedding generation for batch {batch_num}...")
                        embeddings_stored = 0
                        for chunk_id, text in zip(batch_chunk_ids, batch_texts):
                            try:
                                embedding = await self.embeddings_service.generate_embedding(text, dimensions=1024)
                                if embedding:
                                    self.supabase_client.client.table('document_chunks')\
                                        .update({'text_embedding': embedding, 'has_text_embedding': True})\
                                        .eq('id', chunk_id)\
                                        .execute()
                                    embeddings_stored += 1
                            except Exception as individual_error:
                                self.logger.warning(f"   ⚠️ Failed to generate individual embedding for chunk {chunk_id}: {individual_error}")

                        if embeddings_stored > 0:
                            total_embeddings_stored += embeddings_stored
                            self.logger.info(f"   ✅ Stored {embeddings_stored} embeddings via individual fallback")

                # Step 3c: cleanup after batch.
                del batch_chunk_ids, batch_texts, batch_chunk_records, embedding_vectors
                gc.collect()

                # Log memory usage after batch
                try:
                    mem_stats = memory_monitor.get_memory_stats()
                    self.logger.info(f"🧠 Memory after batch {batch_num}/{total_batches}: {mem_stats.percent_used:.1f}% ({mem_stats.used_mb:.0f}MB / {mem_stats.total_mb:.0f}MB)")
                except Exception as mem_error:
                    self.logger.debug(f"Memory monitoring unavailable: {mem_error}")

            # Drop the chunks list now that all batches are persisted.
            del chunks
            gc.collect()
            self.logger.info(f"   🧹 Released chunks list from memory")

            elapsed_time = time.time() - start_time

            # Get quality metrics from chunking service
            quality_metrics = self.chunking_service.get_quality_metrics()

            self.logger.info(
                f"✅ Indexed PDF: {chunks_created} chunks created, {total_embeddings_stored} embeddings in {elapsed_time:.2f}s "
                f"(duplicates: {quality_metrics.exact_duplicates_prevented}, "
                f"low quality: {quality_metrics.low_quality_rejected})"
            )

            return {
                "success": True,
                "chunks_created": chunks_created,
                "chunk_ids": chunk_ids,
                "processing_time": elapsed_time,
                "embeddings_generated": total_embeddings_stored,  # Actual count of embeddings stored
                "quality_metrics": {
                    "total_chunks_created": quality_metrics.total_chunks_created,
                    "exact_duplicates_prevented": quality_metrics.exact_duplicates_prevented,
                    "low_quality_rejected": quality_metrics.low_quality_rejected,
                    "final_chunks": quality_metrics.final_chunks,
                }
            }

        except Exception as e:
            self.logger.error(f"❌ PDF indexing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "chunks_created": 0,
                "chunk_ids": []
            }

    async def multi_vector_search(
        self,
        query: str,
        workspace_id: str,
        top_k: int = 10,
        material_filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.3,
        search_config: Optional[Dict[str, Any]] = None,
        image_base64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        🎯 HYBRID MULTI-SOURCE SEARCH - Combines ALL available data sources.

        **SEARCH SOURCES (All enabled by default):**
        1. **Visual Embeddings** (6 VECS collections, 768D SLIG / 1024D Voyage)
           - vecs.image_slig_embeddings (general visual similarity)
           - vecs.image_color_embeddings (color palette matching)
           - vecs.image_texture_embeddings (texture pattern matching)
           - vecs.image_style_embeddings (design style matching)
           - vecs.image_material_embeddings (material type matching)
           - vecs.image_understanding_embeddings (Qwen3-VL → Voyage 1024D)

        2. **Text Embeddings** (document_chunks) - Semantic text search
           - Searches document_chunks.text_embedding (halfvec(1024) Voyage AI 3.5)
           - Maps chunks to products via chunk_product_relationships

        3. **Direct Product Search** - Product-level embeddings
           - Searches products.text_embedding_1024 (halfvec(1024) Voyage AI 3.5)
           - Direct metadata matching

        4. **Keyword Matching** - Traditional text search
           - Product name/description matching
           - Metadata field matching

        **INTELLIGENT SCORING:**
        - Combines all sources with configurable weights
        - Applies relevance_score from relationships
        - Boosts based on metadata filters
        - Configurable per use-case

        Args:
            query: Search query text
            workspace_id: Workspace ID to filter results
            top_k: Number of results to return
            material_filters: Optional JSONB metadata filters
            similarity_threshold: Minimum similarity score (default 0.3)
            search_config: Optional configuration for search behavior
                {
                    "enable_visual_search": True,
                    "enable_chunk_search": True,
                    "enable_product_search": True,
                    "enable_keyword_search": True,
                    "weights": {
                        "visual": 0.25,
                        "chunk": 0.25,
                        "product": 0.20,
                        "keyword": 0.15,
                        "color": 0.05,
                        "texture": 0.05,
                        "style": 0.03,
                        "material": 0.02
                    }
                }

        Returns:
            Dictionary containing search results with weighted scores from all sources
        """
        try:
            start_time = time.time()

            # ============================================================================
            # CONFIGURATION: Parse search config with sensible defaults
            # ============================================================================
            config = search_config or {}
            enable_visual = config.get('enable_visual_search', True)
            enable_chunk = config.get('enable_chunk_search', True)
            enable_product = config.get('enable_product_search', True)
            enable_keyword = config.get('enable_keyword_search', True)

            # Default weights (can be overridden)
            default_weights = {
                'visual': 0.20,         # Visual SLIG embeddings
                'chunk': 0.20,          # Text chunks
                'understanding': 0.15,  # Vision-understanding (Qwen → Voyage AI)
                'product': 0.15,        # Direct product embeddings
                'keyword': 0.12,        # Keyword matching
                'color': 0.05,          # Color SLIG
                'texture': 0.05,        # Texture SLIG
                'style': 0.04,          # Style SLIG
                'material': 0.04        # Material SLIG
            }
            embedding_weights = config.get('weights', default_weights)

            # ============================================================================
            # STEP 1: Generate query embeddings (visual + text + understanding)
            # All embedding calls run in PARALLEL with 10s timeout each.
            # If any fail or timeout, search continues with remaining sources.
            # ============================================================================
            EMBEDDING_TIMEOUT = 10  # seconds — prevents search from hanging
            self.logger.info(f"🔍 HYBRID SEARCH: '{query[:50]}...'")
            self.logger.info(f"   Sources: Visual={enable_visual}, Chunk={enable_chunk}, Product={enable_product}, Keyword={enable_keyword}")

            visual_embedding = None
            text_embedding = None
            understanding_embedding = None

            # Build embedding tasks to run in parallel
            async def _get_visual_embedding():
                if not enable_visual:
                    return None
                if image_base64:
                    try:
                        import base64 as _b64
                        from io import BytesIO
                        from PIL import Image as _Image
                        img = _Image.open(BytesIO(_b64.b64decode(image_base64)))
                        emb = await self._generate_visual_embedding_for_search(img)
                        if emb:
                            self.logger.info(f"✅ Visual embedding from image: {len(emb)}D")
                            return emb
                    except Exception as _ve:
                        self.logger.warning(f"⚠️ Image visual embedding failed ({_ve}), falling back to text")
                result = await self.embeddings_service.generate_visual_embedding(query)
                if result.get("success"):
                    emb = result.get("embedding", [])
                    self.logger.info(f"✅ Visual embedding from text: {len(emb)}D")
                    return emb
                self.logger.warning("⚠️ Visual embedding generation failed")
                return None

            async def _get_text_embedding():
                if not (enable_chunk or enable_product):
                    return None
                result = await self.embeddings_service.generate_text_embedding(query)
                if result.get("success"):
                    emb = result.get("embedding", [])
                    self.logger.info(f"✅ Text embedding: {len(emb)}D")
                    return emb
                self.logger.warning("⚠️ Text embedding generation failed")
                return None

            async def _get_understanding_embedding():
                result = await self.embeddings_service.generate_understanding_query_embedding(query)
                if result.get("success"):
                    emb = result.get("embedding", [])
                    self.logger.info(f"✅ Understanding embedding: {len(emb)}D")
                    return emb
                return None

            # Run all embedding generations in parallel, each with its own timeout.
            # Per-task timeouts ensure completed tasks are preserved even if one hangs.
            visual_result, text_result, understanding_result = await asyncio.gather(
                asyncio.wait_for(_get_visual_embedding(), timeout=EMBEDDING_TIMEOUT),
                asyncio.wait_for(_get_text_embedding(), timeout=EMBEDDING_TIMEOUT),
                asyncio.wait_for(_get_understanding_embedding(), timeout=EMBEDDING_TIMEOUT),
                return_exceptions=True
            )
            if isinstance(visual_result, Exception):
                self.logger.warning(f"⚠️ Visual embedding failed/timed out: {visual_result}")
            else:
                visual_embedding = visual_result
            if isinstance(text_result, Exception):
                self.logger.warning(f"⚠️ Text embedding failed/timed out: {text_result}")
            else:
                text_embedding = text_result
            if isinstance(understanding_result, Exception):
                self.logger.warning(f"⚠️ Understanding embedding failed/timed out: {understanding_result}")
            else:
                understanding_embedding = understanding_result

            # ============================================================================
            # STEP 2A: Search VISUAL + UNDERSTANDING embeddings (6 VECS collections)
            # ============================================================================
            image_scores = {}  # Maps image_id -> {embedding_type: score}

            if enable_visual and visual_embedding:
                embedding_types = ['color', 'texture', 'style', 'material']
                search_tasks = []

                for emb_type in embedding_types:
                    task = self.vecs_service.search_specialized_embeddings(
                        query_embedding=visual_embedding,
                        embedding_type=emb_type,
                        limit=top_k * 3,
                        workspace_id=workspace_id,
                        include_metadata=True
                    )
                    search_tasks.append(task)

                # Add understanding search in parallel with visual searches
                if understanding_embedding:
                    search_tasks.append(
                        self.vecs_service.search_understanding_embeddings(
                            query_embedding=understanding_embedding,
                            limit=top_k * 3,
                            workspace_id=workspace_id,
                            include_metadata=True
                        )
                    )
                    embedding_types = embedding_types + ['understanding']

                self.logger.info(f"🚀 Searching {len(embedding_types)} embeddings...")
                search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

                for emb_type, results in zip(embedding_types, search_results):
                    if isinstance(results, Exception):
                        self.logger.warning(f"⚠️ {emb_type} failed: {results}")
                        continue

                    if results:
                        for item in results:
                            image_id = item.get('image_id')
                            score = item.get('similarity_score', 0.0)
                            if image_id not in image_scores:
                                image_scores[image_id] = {}
                            image_scores[image_id][emb_type] = score

                        self.logger.info(f"✅ {emb_type}: {len(results)} images")

                self.logger.info(f"📊 Visual search: {len(image_scores)} unique images")

            # ============================================================================
            # STEP 2B: Search TEXT CHUNKS (NEW!)
            # ============================================================================
            chunk_scores = {}  # Maps chunk_id -> score

            if enable_chunk and text_embedding:
                try:
                    # Search document_chunks using text_embedding
                    # Note: We'll need to add a function to search chunks by embedding
                    chunk_results = await self._search_chunks_by_embedding(
                        query_embedding=text_embedding,
                        workspace_id=workspace_id,
                        limit=top_k * 3
                    )

                    for chunk in chunk_results:
                        chunk_id = chunk.get('chunk_id')
                        score = chunk.get('similarity_score', 0.0)
                        chunk_scores[chunk_id] = score

                    self.logger.info(f"✅ Chunk search: {len(chunk_scores)} chunks")

                    # Cross-reference expansion: fetch target chunks referenced by top results
                    if chunk_scores:
                        try:
                            top_chunk_ids = sorted(chunk_scores, key=chunk_scores.get, reverse=True)[:20]
                            xref_response = self.supabase_client.client.table('document_chunks')\
                                .select('id, metadata')\
                                .in_('id', top_chunk_ids)\
                                .execute()
                            if xref_response.data:
                                xref_target_ids = []
                                for row in xref_response.data:
                                    refs = (row.get('metadata') or {}).get('cross_references', [])
                                    for ref in refs:
                                        if ref.get('resolved'):
                                            xref_target_ids.extend(ref.get('target_chunk_ids', []))
                                xref_target_ids = [tid for tid in set(xref_target_ids) if tid not in chunk_scores]
                                if xref_target_ids:
                                    # Give cross-referenced chunks 60% of the referencing chunk's score
                                    avg_ref_score = sum(chunk_scores[cid] for cid in top_chunk_ids if cid in chunk_scores) / max(len(top_chunk_ids), 1)
                                    for tid in xref_target_ids:
                                        chunk_scores[tid] = avg_ref_score * 0.6
                                    self.logger.info(f"   📎 Cross-reference expansion: added {len(xref_target_ids)} referenced chunks")
                        except Exception as xref_e:
                            self.logger.warning(f"Cross-reference expansion failed: {xref_e}")

                except Exception as e:
                    self.logger.warning(f"⚠️ Chunk search failed: {e}")

            # ============================================================================
            # STEP 2C: Search PRODUCTS directly (NEW!)
            # ============================================================================
            direct_product_scores = {}  # Maps product_id -> score

            if enable_product and text_embedding:
                try:
                    # Search products using text_embedding_1024
                    product_results = await self._search_products_by_embedding(
                        query_embedding=text_embedding,
                        workspace_id=workspace_id,
                        limit=top_k * 2
                    )

                    for product in product_results:
                        product_id = product.get('product_id')
                        score = product.get('similarity_score', 0.0)
                        direct_product_scores[product_id] = score

                    self.logger.info(f"✅ Product search: {len(direct_product_scores)} products")

                except Exception as e:
                    self.logger.warning(f"⚠️ Product search failed: {e}")

            # ============================================================================
            # STEP 2D: Fulltext search on products (search_tsv)
            # Catches manufacturer, designer, material_category, colors, textures
            # that embedding search may miss.
            # ============================================================================
            fulltext_product_scores = {}  # Maps product_id -> score

            try:
                fulltext_results = await self._search_products_fulltext(
                    query=query,
                    workspace_id=workspace_id,
                    limit=top_k * 2
                )

                for result in fulltext_results:
                    product_id = result.get('product_id')
                    if not product_id:
                        continue
                    # Normalize ts_rank_cd score to 0-1 range (scores are typically 0-0.5)
                    score = min(1.0, result.get('similarity_score', 0.0) * 2.0)
                    fulltext_product_scores[product_id] = score

                if fulltext_product_scores:
                    self.logger.info(f"✅ Fulltext search: {len(fulltext_product_scores)} products")

            except Exception as e:
                self.logger.warning(f"⚠️ Fulltext search failed: {e}")

            # ============================================================================
            # STEP 3: Map all sources to products
            # ============================================================================
            product_scores = {}  # Maps product_id -> {source_type: score}

            # 3A: Map images to products via image_product_associations
            if image_scores:
                image_ids = list(image_scores.keys())
                batch_size = 100
                all_image_rels = []

                for i in range(0, len(image_ids), batch_size):
                    batch_ids = image_ids[i:i + batch_size]
                    rel_response = self.supabase_client.client.table('image_product_associations')\
                        .select('product_id, image_id, overall_score')\
                        .in_('image_id', batch_ids)\
                        .execute()
                    if rel_response.data:
                        all_image_rels.extend(rel_response.data)

                self.logger.info(f"📎 Image→Product: {len(all_image_rels)} relationships")

                for rel in all_image_rels:
                    product_id = rel.get('product_id')
                    image_id = rel.get('image_id')
                    relevance = rel.get('overall_score', 1.0)

                    if image_id in image_scores:
                        if product_id not in product_scores:
                            product_scores[product_id] = {}

                        for emb_type, score in image_scores[image_id].items():
                            weighted_score = score * relevance
                            if emb_type not in product_scores[product_id] or product_scores[product_id][emb_type] < weighted_score:
                                product_scores[product_id][emb_type] = weighted_score

            # 3B: Map chunks to products via chunk_product_relationships (NEW!)
            if chunk_scores:
                chunk_ids = list(chunk_scores.keys())
                batch_size = 100
                all_chunk_rels = []

                for i in range(0, len(chunk_ids), batch_size):
                    batch_ids = chunk_ids[i:i + batch_size]
                    rel_response = self.supabase_client.client.table('chunk_product_relationships')\
                        .select('product_id, chunk_id, relevance_score')\
                        .in_('chunk_id', batch_ids)\
                        .execute()
                    if rel_response.data:
                        all_chunk_rels.extend(rel_response.data)

                self.logger.info(f"📎 Chunk→Product: {len(all_chunk_rels)} relationships")

                for rel in all_chunk_rels:
                    product_id = rel.get('product_id')
                    chunk_id = rel.get('chunk_id')
                    relevance = rel.get('relevance_score', 1.0)

                    if chunk_id in chunk_scores:
                        if product_id not in product_scores:
                            product_scores[product_id] = {}

                        weighted_score = chunk_scores[chunk_id] * relevance
                        if 'chunk' not in product_scores[product_id] or product_scores[product_id]['chunk'] < weighted_score:
                            product_scores[product_id]['chunk'] = weighted_score

            # 3C: Add direct product scores
            if direct_product_scores:
                self.logger.info(f"📎 Direct product scores: {len(direct_product_scores)}")

                for product_id, score in direct_product_scores.items():
                    if product_id not in product_scores:
                        product_scores[product_id] = {}
                    product_scores[product_id]['product'] = score

            # 3D: Add fulltext product scores (manufacturer, designer, colors, etc.)
            if fulltext_product_scores:
                self.logger.info(f"📎 Fulltext product scores: {len(fulltext_product_scores)}")

                for product_id, score in fulltext_product_scores.items():
                    if product_id not in product_scores:
                        product_scores[product_id] = {}
                    # Use 'keyword' source since fulltext IS keyword matching
                    # Take the max of keyword (Jaccard) and fulltext (PostgreSQL tsrank)
                    existing = product_scores[product_id].get('keyword', 0.0)
                    product_scores[product_id]['keyword'] = max(existing, score)

            self.logger.info(f"🎯 Total products with scores: {len(product_scores)}")

            # ============================================================================
            # STEP 4: Fetch product details and calculate HYBRID weighted scores
            # ============================================================================
            results = []
            if product_scores:
                product_ids = list(product_scores.keys())
                batch_size = 100
                all_products = []

                for i in range(0, len(product_ids), batch_size):
                    batch_ids = product_ids[i:i + batch_size]
                    products_response = self.supabase_client.client.table('products')\
                        .select('id, name, description, metadata, workspace_id, source_document_id')\
                        .in_('id', batch_ids)\
                        .execute()
                    if products_response.data:
                        all_products.extend(products_response.data)

                self.logger.info(f"📦 Fetched {len(all_products)} product details")

                # Calculate HYBRID weighted scores
                for product in all_products:
                    product_id = product['id']
                    scores = product_scores.get(product_id, {})

                    # Add keyword score if enabled
                    if enable_keyword:
                        keyword_score = self._calculate_text_score(query, product)
                        scores['keyword'] = max(scores.get('keyword', 0.0), keyword_score)

                    # Calculate weighted score from ALL sources
                    # IMPORTANT: Normalize by ACTIVE weights only (sources with score > 0).
                    # Otherwise single-source matches (e.g., manufacturer-only fulltext hits)
                    # get unfairly diluted by the unused embedding weights — e.g., a keyword
                    # score of 0.4 would shrink to 0.4*0.12=0.048 and fall below threshold.
                    weighted_score = 0.0
                    active_weight_sum = 0.0
                    score_breakdown = {}

                    for source_type, weight in embedding_weights.items():
                        source_score = scores.get(source_type, 0.0)
                        if source_score > 0:
                            weighted_score += source_score * weight
                            active_weight_sum += weight
                        score_breakdown[f'{source_type}_score'] = source_score

                    # Normalize so the score reflects quality, not source count
                    if active_weight_sum > 0:
                        weighted_score = weighted_score / active_weight_sum

                    # Apply material filters as soft boosts
                    filter_boost = 0.0
                    if material_filters:
                        product_metadata = product.get('metadata') or {}

                        for filter_key, filter_value in material_filters.items():
                            # Handle nested paths like "appearance.colors"
                            if '.' in filter_key:
                                path_parts = filter_key.split('.')
                                product_value = product_metadata
                                for part in path_parts:
                                    if isinstance(product_value, dict):
                                        product_value = product_value.get(part)
                                    else:
                                        product_value = None
                                        break
                            else:
                                product_value = product_metadata.get(filter_key)

                            if product_value is None:
                                continue

                            # Handle dict values with 'value' key
                            if isinstance(product_value, dict) and 'value' in product_value:
                                product_value = product_value['value']

                            # Handle list values
                            if isinstance(product_value, list):
                                product_value = ' '.join(str(v) for v in product_value)

                            # Use contains matching
                            product_value_str = str(product_value).lower()
                            if isinstance(filter_value, list):
                                matched = any(str(v).lower() in product_value_str or product_value_str in str(v).lower()
                                            for v in filter_value)
                                if matched:
                                    filter_boost += 0.15
                            else:
                                filter_value_str = str(filter_value).lower()
                                if filter_value_str in product_value_str or product_value_str in filter_value_str:
                                    filter_boost += 0.15

                    # Add filter boost to weighted score
                    final_score = weighted_score + filter_boost

                    # Adaptive threshold: if most embedding sources failed/timed out,
                    # lower the threshold so fulltext/keyword results can still surface.
                    active_sources = sum(1 for s in ['visual', 'chunk', 'understanding', 'product']
                                        if scores.get(s, 0.0) > 0)
                    effective_threshold = similarity_threshold if active_sources >= 2 else similarity_threshold * 0.3

                    # Only include results above threshold
                    if final_score >= effective_threshold:
                        results.append({
                            "id": product_id,
                            "product_name": product.get('name'),
                            "description": product.get('description'),
                            "metadata": product.get('metadata', {}),
                            "workspace_id": product.get('workspace_id'),
                            "source_document_id": product.get('source_document_id'),
                            "score": final_score,
                            **score_breakdown,  # Include scores from all sources
                            "filter_boost": filter_boost,
                            "search_type": "hybrid_multi_source",
                            "sources_used": {
                                "visual": enable_visual and len(image_scores) > 0,
                                "chunk": enable_chunk and len(chunk_scores) > 0,
                                "product": enable_product and len(direct_product_scores) > 0,
                                "keyword": enable_keyword
                            }
                        })

                # Sort by final score (descending)
                results.sort(key=lambda x: x['score'], reverse=True)

                # Apply MMR re-ranking for diversity (if enabled via search_config)
                enable_mmr = (search_config or {}).get("enable_mmr", False)
                mmr_lambda = (search_config or {}).get("mmr_lambda", 0.7)
                if enable_mmr and len(results) > top_k:
                    from .mmr_reranker import MMRReranker
                    reranker = MMRReranker(lambda_param=mmr_lambda)
                    mmr_result = reranker.rerank(results, top_k=top_k)
                    results = mmr_result.items
                    self.logger.info(
                        f"MMR re-ranked → {len(results)} diverse results (λ={mmr_lambda})"
                    )
                else:
                    results = results[:top_k]

                self.logger.info(f"✅ Returning {len(results)} results (top {top_k})")

            else:
                self.logger.warning("⚠️ No product scores available, returning empty results")

            # Apply metadata prototype validation scoring (if available)
            if material_filters and results:
                try:
                    from .metadata_prototype_validator import get_metadata_validator
                    validator = get_metadata_validator()
                    await validator.load_prototypes()

                    # Enhance results with metadata validation scoring
                    for result in results:
                        metadata_boost = await self._calculate_metadata_validation_boost(
                            product_metadata=result.get("metadata", {}),
                            query_filters=material_filters,
                            validator=validator
                        )

                        # Apply boost to score (up to 20% increase)
                        original_score = result["score"]
                        result["score"] = original_score * (1.0 + metadata_boost * 0.2)
                        result["metadata_validation_boost"] = metadata_boost
                        result["original_score"] = original_score

                    # Re-sort by enhanced score
                    results.sort(key=lambda x: x["score"], reverse=True)

                except Exception as e:
                    self.logger.warning(f"Metadata validation scoring failed: {e}")

            # NOTE: search query tracking is now done in rag_routes.search_documents
            # with full per-stage timings. The old call here would double-track.

            return {
                "results": results,
                "total_results": len(results),
                "processing_time": time.time() - start_time,
                "search_config": {
                    "weights": embedding_weights,
                    "sources_enabled": {
                        "visual": enable_visual,
                        "chunk": enable_chunk,
                        "product": enable_product,
                        "keyword": enable_keyword
                    },
                    "sources_found": {
                        "images": len(image_scores) if enable_visual else 0,
                        "chunks": len(chunk_scores) if enable_chunk else 0,
                        "products": len(direct_product_scores) if enable_product else 0
                    }
                },
                "material_filters_applied": material_filters if material_filters else None,
                "metadata_validation_enabled": True,
                "query": query,
                "search_type": "hybrid_multi_source"
            }

        except Exception as e:
            self.logger.error(f"Multi-vector search failed: {e}", exc_info=True)
            return {
                "results": [],
                "error": str(e),
                "total_results": 0,
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0
            }

    async def _calculate_metadata_validation_boost(
        self,
        product_metadata: Dict[str, Any],
        query_filters: Dict[str, Any],
        validator: Any
    ) -> float:
        """Calculate metadata validation boost score.

        Compares query filters to product metadata using prototype validation.
        Returns boost score between 0.0 and 1.0.

        Args:
            product_metadata: Product's metadata JSONB
            query_filters: User's search filters
            validator: MetadataPrototypeValidator instance

        Returns:
            Boost score (0.0 = no boost, 1.0 = perfect match)
        """
        if not query_filters or not product_metadata:
            return 0.0

        total_score = 0.0
        matched_fields = 0
        total_fields = len(query_filters)

        for filter_key, filter_value in query_filters.items():
            product_value = product_metadata.get(filter_key)

            if not product_value:
                continue

            # Check if this field has prototype validation
            if filter_key in validator._prototype_cache:
                # Check validation metadata
                validation_info = product_metadata.get('_validation', {}).get(filter_key, {})

                if validation_info.get('prototype_matched'):
                    # Both values are validated - check if they match
                    if str(product_value).lower() == str(filter_value).lower():
                        # Exact match → full score
                        total_score += 1.0
                        matched_fields += 1
                    else:
                        # Different prototypes → check semantic similarity
                        try:
                            import numpy as np

                            # Generate embeddings for both values
                            filter_embedding = await validator.embeddings_service._generate_text_embedding(
                                text=str(filter_value),
                                job_id=None,
                                dimensions=512
                            )
                            product_embedding = await validator.embeddings_service._generate_text_embedding(
                                text=str(product_value),
                                job_id=None,
                                dimensions=512
                            )

                            if filter_embedding and product_embedding:
                                similarity = validator._cosine_similarity(
                                    np.array(filter_embedding),
                                    np.array(product_embedding)
                                )

                                if similarity > 0.70:
                                    # Semantically similar → partial score
                                    total_score += similarity
                                    matched_fields += 1
                        except Exception as e:
                            self.logger.warning(f"Similarity calculation failed: {e}")
                else:
                    # Product value not validated → fuzzy match
                    if str(product_value).lower() == str(filter_value).lower():
                        total_score += 0.8  # Penalty for unvalidated
                        matched_fields += 1
            else:
                # No prototype validation → exact match only
                if str(product_value).lower() == str(filter_value).lower():
                    total_score += 1.0
                    matched_fields += 1

        # Normalize score
        if total_fields > 0:
            return total_score / total_fields
        return 0.0

    def _calculate_text_score(self, query: str, product: Dict[str, Any]) -> float:
        """
        Calculate text similarity score based on keyword matching.

        Args:
            query: Search query text
            product: Product dictionary with name, description, metadata

        Returns:
            Text similarity score (0.0 to 1.0)
        """
        if not query:
            return 0.0

        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Collect all text from product
        text_parts = []

        # Product name (highest weight)
        if product.get('name'):
            text_parts.append(('name', product['name'], 3.0))

        # Manufacturer / factory (highest weight — same as name)
        metadata = product.get('metadata') or {}
        if isinstance(metadata, dict) and metadata:
            for mfr_key in ('manufacturer', 'factory_name', 'factory_group_name'):
                mfr_val = metadata.get(mfr_key)
                if mfr_val and isinstance(mfr_val, str) and mfr_val != 'Not specified':
                    text_parts.append(('manufacturer', mfr_val, 3.0))

            # Designer (high weight)
            designer = metadata.get('designer')
            if designer and isinstance(designer, str):
                text_parts.append(('designer', designer, 2.5))

            # Collection name (high weight)
            design = metadata.get('design')
            collection = None
            if isinstance(design, dict):
                coll_obj = design.get('collection')
                if isinstance(coll_obj, dict):
                    collection = coll_obj.get('value')
                elif isinstance(coll_obj, str):
                    collection = coll_obj
            if collection:
                text_parts.append(('collection', collection, 2.5))

            # Material category + colors (medium weight)
            mat_cat = metadata.get('material_category', '')
            if isinstance(mat_cat, str):
                mat_cat = mat_cat.replace('_', ' ')
            if mat_cat:
                text_parts.append(('material_category', mat_cat, 2.0))

            colors = metadata.get('available_colors')
            if isinstance(colors, list):
                text_parts.append(('colors', ' '.join(str(c) for c in colors), 1.5))

        # Product description (medium weight)
        if product.get('description'):
            text_parts.append(('description', product['description'], 2.0))

        # Remaining metadata fields (lower weight)
        if isinstance(metadata, dict) and metadata:
            metadata_text = self._flatten_metadata_to_text(metadata)
            text_parts.append(('metadata', metadata_text, 1.0))

        # Calculate weighted keyword overlap
        total_score = 0.0
        total_weight = 0.0

        for field_name, text, weight in text_parts:
            if not text:
                continue

            text_lower = str(text).lower()
            text_words = set(text_lower.split())

            # Calculate Jaccard similarity
            if text_words:
                intersection = query_words & text_words
                union = query_words | text_words
                jaccard = len(intersection) / len(union) if union else 0.0

                # Also check for substring matches (partial matches)
                substring_bonus = 0.0
                for query_word in query_words:
                    if len(query_word) > 3 and query_word in text_lower:
                        substring_bonus += 0.1

                field_score = min(1.0, jaccard + substring_bonus)
                total_score += field_score * weight
                total_weight += weight

        # Normalize by total weight
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        return min(1.0, final_score)

    def _flatten_metadata_to_text(self, metadata: Dict[str, Any]) -> str:
        """Flatten nested metadata dictionary to searchable text."""
        text_parts = []

        def extract_values(obj, prefix=''):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        extract_values(value, f"{prefix}{key}.")
                    else:
                        text_parts.append(str(value))
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        extract_values(item, prefix)
                    else:
                        text_parts.append(str(item))
            else:
                text_parts.append(str(obj))

        extract_values(metadata)
        return ' '.join(text_parts)

    async def material_property_search(
        self,
        workspace_id: str,
        material_filters: Dict[str, Any],
        top_k: int = 10,
        match_mode: str = "AND"
    ) -> Dict[str, Any]:
        """
        JSONB-based property filtering with AND/OR logic support.

        Uses Supabase client for filtering (no raw SQL).
        Filters products in Python based on metadata JSONB column.

        Args:
            workspace_id: Workspace ID to filter results
            material_filters: Dictionary of property filters
                Example: {
                    "material_type": "fabric",
                    "color": ["red", "blue"]
                }
            top_k: Number of results to return
            match_mode: "AND" or "OR" for combining filters

        Returns:
            Dictionary containing filtered products
        """
        try:
            start_time = time.time()

            if not material_filters:
                return {
                    "results": [],
                    "message": "No filters provided",
                    "total_results": 0,
                    "processing_time": time.time() - start_time
                }

            # Get all products for the workspace
            products_response = self.supabase_client.client.table('products')\
                .select('id, name, description, metadata, workspace_id, source_document_id')\
                .eq('workspace_id', workspace_id)\
                .execute()

            if not products_response.data:
                return {
                    "results": [],
                    "total_results": 0,
                    "processing_time": time.time() - start_time,
                    "filters": material_filters,
                    "match_mode": match_mode
                }

            # Filter products in Python based on material_filters
            results = []
            for product in products_response.data:
                product_metadata = product.get('metadata') or {}

                # Check each filter
                filter_results = []
                for key, value in material_filters.items():
                    product_value = product_metadata.get(key)

                    if product_value is None:
                        filter_results.append(False)
                        continue

                    if isinstance(value, dict):
                        # Handle comparison operators
                        match = True
                        for op, val in value.items():
                            try:
                                product_num = float(product_value)
                                if op == "gte" and not (product_num >= val):
                                    match = False
                                elif op == "lte" and not (product_num <= val):
                                    match = False
                                elif op == "gt" and not (product_num > val):
                                    match = False
                                elif op == "lt" and not (product_num < val):
                                    match = False
                                elif op == "eq" and str(product_value).lower() != str(val).lower():
                                    match = False
                            except (ValueError, TypeError):
                                match = False
                        filter_results.append(match)
                    elif isinstance(value, list):
                        # Handle IN operator
                        filter_results.append(
                            str(product_value).lower() in [str(v).lower() for v in value]
                        )
                    else:
                        # Handle exact match
                        filter_results.append(
                            str(product_value).lower() == str(value).lower()
                        )

                # Apply AND/OR logic
                if match_mode == "AND":
                    matches = all(filter_results) if filter_results else False
                else:  # OR
                    matches = any(filter_results) if filter_results else False

                if matches:
                    results.append({
                        "id": product.get("id"),
                        "product_name": product.get("name"),
                        "description": product.get("description"),
                        "metadata": product_metadata,
                        "workspace_id": product.get("workspace_id"),
                        "source_document_id": product.get("source_document_id"),
                        "search_type": "material_property"
                    })

            # Limit results
            results = results[:top_k]

            return {
                "results": results,
                "total_results": len(results),
                "processing_time": time.time() - start_time,
                "filters": material_filters,
                "match_mode": match_mode
            }

        except Exception as e:
            self.logger.error(f"Material property search failed: {e}", exc_info=True)
            return {
                "results": [],
                "error": str(e),
                "total_results": 0,
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0
            }

    async def image_similarity_search(
        self,
        workspace_id: str,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        ✅ Visual similarity search using VECS with relationship enrichment.

        Searches images using VECS visual embeddings (SigLIP/CLIP) with HNSW indexing.
        Returns enriched results with related products and chunks.

        Args:
            workspace_id: Workspace ID to filter results
            image_url: URL of the query image (optional)
            image_base64: Base64-encoded query image (optional)
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (default 0.7)
            document_id: Optional document ID to filter results

        Returns:
            Dictionary containing visually similar images with products and chunks
        """
        try:
            import base64
            import requests
            from io import BytesIO
            from PIL import Image

            start_time = time.time()

            # Validate input
            if not image_url and not image_base64:
                return {
                    "results": [],
                    "message": "Either image_url or image_base64 must be provided",
                    "total_results": 0,
                    "processing_time": time.time() - start_time
                }

            # Load image
            if image_url:
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
            else:
                image_data = base64.b64decode(image_base64)
                image = Image.open(BytesIO(image_data))

            # Generate visual embedding for the query image
            query_embedding = await self._generate_visual_embedding_for_search(image)

            if not query_embedding:
                return {
                    "results": [],
                    "message": "Failed to generate image embedding",
                    "total_results": 0,
                    "processing_time": time.time() - start_time
                }

            # Build metadata filters
            filters = {"workspace_id": {"$eq": workspace_id}}
            if document_id:
                filters["document_id"] = {"$eq": document_id}

            # Search VECS collection
            vecs_results = await self.vecs_service.search_similar_images(
                query_embedding=query_embedding,
                limit=top_k * 2,  # Get more results to filter by threshold
                filters=filters,
                include_metadata=True
            )

            # Filter by similarity threshold
            filtered_results = [
                r for r in vecs_results
                if r.get('similarity_score', 0) >= similarity_threshold
            ][:top_k]

            # Format results
            results = []
            for item in filtered_results:
                results.append({
                    "image_id": item.get("image_id"),
                    "similarity_score": item.get("similarity_score"),
                    "metadata": item.get("metadata", {}),
                    "search_type": "vecs_image_similarity"
                })

            self.logger.info(f"✅ VECS image search: {len(results)} results in {time.time() - start_time:.2f}s")

            return {
                "results": results,
                "total_results": len(results),
                "processing_time": time.time() - start_time,
                "query_type": "image_url" if image_url else "image_base64",
                "search_method": "vecs_hnsw"
            }

        except Exception as e:
            self.logger.error(f"Image similarity search failed: {e}", exc_info=True)
            return {
                "results": [],
                "error": str(e),
                "total_results": 0,
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0
            }

    async def _generate_visual_embedding_for_search(self, image: 'Image') -> Optional[List[float]]:
        """
        Generate visual embedding for an image using configured model (SigLIP/CLIP).

        Uses RealEmbeddingsService which has visual embedding model loaded.

        Args:
            image: PIL Image object

        Returns:
            List of floats representing the visual embedding
        """
        try:
            import io
            import base64

            # Convert PIL image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Generate embedding using configured visual model
            embedding, model_used, _ = await self.embeddings_service._generate_visual_embedding(
                image_url=None,
                image_data=image_base64
            )

            if embedding:
                self.logger.info(f"✅ Generated visual embedding ({len(embedding)}D) using {model_used}")
                return embedding

            self.logger.warning("⚠️ Visual embedding generation returned None")
            return None

        except Exception as e:
            self.logger.error(f"Failed to generate visual embedding: {e}")
            return None

    # ============================================================================
    # RAG Query Methods (Claude 4.5)
    # ============================================================================

    async def query_document(
        self,
        document_id: str,
        query: str,
        workspace_id: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query a specific document using Claude 4.5 for intelligent synthesis.

        Args:
            document_id: ID of the document to query
            query: Natural language query
            workspace_id: Optional workspace ID for filtering
            top_k: Number of relevant chunks to retrieve

        Returns:
            Dict containing query response and metadata
        """
        try:
            start_time = time.time()

            # Step 1: Retrieve relevant chunks using multi-vector search
            search_results = await self.multi_vector_search(
                query=query,
                workspace_id=workspace_id,
                top_k=top_k,
                material_filters={"document_id": document_id} if document_id else None
            )

            chunks = search_results.get('results', [])

            if not chunks:
                return {
                    "success": False,
                    "error": f"No relevant content found in document {document_id}"
                }

            # Step 2: Build context from retrieved chunks
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                chunk_text = chunk.get('content', chunk.get('text', ''))
                score = chunk.get('score', 0.0)
                context_parts.append(f"[Chunk {i}, Relevance: {score:.2f}]\n{chunk_text}\n")

            context = "\n".join(context_parts)

            # Step 3: Build prompt for Claude 4.5
            prompt = f"""You are an expert document analyst. Answer the following question based ONLY on the provided context.

**Question:** {query}

**Context from Document:**
{context}

**Instructions:**
- Provide a clear, accurate answer based on the context
- If the context doesn't contain enough information, say so
- Cite specific parts of the context when relevant
- Be concise but comprehensive

**Answer:**"""

            # Step 4: Call Claude Opus 4.7
            client = self.ai_client_service.anthropic
            response = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.content[0].text.strip()

            # Step 5: Log AI call
            latency_ms = int((time.time() - start_time) * 1000)
            await self.ai_logger.log_claude_call(
                task="rag_query_document",
                model="claude-opus-4-7",
                response=response,
                latency_ms=latency_ms,
                confidence_score=0.85,
                confidence_breakdown={},
                action="use_ai_result"
            )

            # Step 6: Format sources
            sources = []
            for chunk in chunks:
                sources.append({
                    "chunk_id": chunk.get('id'),
                    "score": chunk.get('score', 0.0),
                    "text_snippet": chunk.get('content', '')[:200] + "...",
                    "metadata": chunk.get('metadata', {})
                })

            return {
                "success": True,
                "document_id": document_id,
                "query": query,
                "response": answer,
                "sources": sources,
                "metadata": {
                    "chunks_retrieved": len(chunks),
                    "processing_time_ms": latency_ms,
                    "model": "claude-opus-4-7"
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to query document {document_id}: {e}")
            return {
                "success": False,
                "document_id": document_id,
                "query": query,
                "error": str(e)
            }

    async def advanced_rag_query(
        self,
        query: str,
        workspace_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        query_type: str = "factual",
        similarity_threshold: float = 0.7,
        max_results: int = 5,
        enable_reranking: bool = True,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Advanced RAG query with Claude 4.5 and conversation context support.

        Args:
            query: Natural language query
            workspace_id: Workspace ID for filtering
            document_ids: Optional list of specific documents to query
            query_type: Type of query ('factual', 'analytical', 'conversational', 'summarization')
            similarity_threshold: Minimum similarity score for retrieved chunks
            max_results: Maximum number of results to return
            enable_reranking: Whether to apply relevance re-ranking (currently uses score filtering)
            conversation_context: Previous conversation turns for context
            metadata_filters: Optional metadata filters for retrieval

        Returns:
            Dict containing comprehensive query results and metadata
        """
        try:
            start_time = time.time()

            # Step 1: Enhance query with conversation context if provided
            enhanced_query = query
            if conversation_context:
                context_summary = "\n".join([
                    f"{turn.get('role', 'user')}: {turn.get('content', '')}"
                    for turn in conversation_context[-3:]  # Last 3 turns
                ])
                enhanced_query = f"Previous conversation:\n{context_summary}\n\nCurrent question: {query}"

            # Step 2: Retrieve relevant chunks
            search_filters = metadata_filters or {}
            if document_ids:
                search_filters['document_id'] = document_ids[0] if len(document_ids) == 1 else document_ids

            search_results = await self.multi_vector_search(
                query=enhanced_query,
                workspace_id=workspace_id,
                top_k=max_results * 2,  # Retrieve more for filtering
                material_filters=search_filters
            )

            all_chunks = search_results.get('results', [])

            # Step 3: Filter by similarity threshold
            filtered_chunks = [
                chunk for chunk in all_chunks
                if chunk.get('score', 0.0) >= similarity_threshold
            ][:max_results]

            if not filtered_chunks:
                return {
                    "success": False,
                    "error": "No relevant content found matching the query"
                }

            # Step 4: Build context from chunks
            context_parts = []
            for i, chunk in enumerate(filtered_chunks, 1):
                chunk_text = chunk.get('content', chunk.get('text', ''))
                score = chunk.get('score', 0.0)
                metadata = chunk.get('metadata', {})
                doc_id = metadata.get('document_id', 'unknown')
                context_parts.append(
                    f"[Source {i} - Document: {doc_id}, Relevance: {score:.2f}]\n{chunk_text}\n"
                )

            context = "\n".join(context_parts)

            # Step 5: Build query-type specific prompt
            query_instructions = {
                "factual": "Provide factual, precise answers based strictly on the context.",
                "analytical": "Analyze the information and provide insights, patterns, and implications.",
                "conversational": "Engage naturally while staying grounded in the provided context.",
                "summarization": "Summarize the key points from the context concisely."
            }

            instruction = query_instructions.get(query_type, query_instructions["factual"])

            # Build conversation context section
            conversation_section = ""
            if conversation_context:
                conversation_section = f"""
**Previous Conversation:**
{context_summary}
"""

            prompt = f"""You are an expert AI assistant specializing in {query_type} queries.

{conversation_section}
**Current Question:** {query}

**Retrieved Context:**
{context}

**Instructions:**
- {instruction}
- Base your answer ONLY on the provided context
- If information is insufficient, clearly state what's missing
- Cite sources when making specific claims
- Maintain conversation continuity if context is provided

**Response:**"""

            # Step 6: Call Claude Opus 4.7
            client = self.ai_client_service.anthropic
            response = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.content[0].text.strip()

            # Step 7: Calculate confidence score
            avg_score = sum(c.get('score', 0.0) for c in filtered_chunks) / len(filtered_chunks)
            confidence = min(avg_score * 1.2, 1.0)  # Boost slightly, cap at 1.0

            # Step 8: Log AI call
            latency_ms = int((time.time() - start_time) * 1000)
            await self.ai_logger.log_claude_call(
                task=f"rag_advanced_query_{query_type}",
                model="claude-opus-4-7",
                response=response,
                latency_ms=latency_ms,
                confidence_score=confidence,
                confidence_breakdown={
                    "retrieval_quality": avg_score,
                    "chunk_count": len(filtered_chunks),
                    "threshold_met": all(c.get('score', 0) >= similarity_threshold for c in filtered_chunks)
                },
                action="use_ai_result"
            )

            # Step 9: Format sources
            sources = []
            for chunk in filtered_chunks:
                sources.append({
                    "chunk_id": chunk.get('id'),
                    "document_id": chunk.get('metadata', {}).get('document_id'),
                    "score": chunk.get('score', 0.0),
                    "text_snippet": chunk.get('content', '')[:200] + "...",
                    "metadata": chunk.get('metadata', {})
                })

            return {
                "success": True,
                "query": {
                    "original": query,
                    "enhanced": enhanced_query if conversation_context else query,
                    "type": query_type
                },
                "response": answer,
                "confidence_score": confidence,
                "sources": sources,
                "retrieval_stats": {
                    "total_chunks_retrieved": len(all_chunks),
                    "chunks_after_filtering": len(filtered_chunks),
                    "similarity_threshold": similarity_threshold,
                    "average_score": avg_score
                },
                "metadata": {
                    "query_type": query_type,
                    "has_conversation_context": conversation_context is not None,
                    "processing_time_ms": latency_ms,
                    "model": "claude-opus-4-7"
                }
            }

        except Exception as e:
            self.logger.error(f"Advanced RAG query failed: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "query_type": query_type
            }

    async def _analyze_image_material(
        self,
        image_base64: str,
        image_path: str = None,
        image_id: str = None,
        document_id: str = None,
        embedding_service: Any = None
    ) -> Dict[str, Any]:
        """
        Analyze material image with Claude Opus 4.7 vision via Anthropic tool use
        to extract material properties, quality, and confidence.

        Tool use forces a schema-conformant JSON response (no parsing/regex
        recovery needed). The schema matches `app.models.vision_analysis.
        VisionAnalysis` so the same shape feeds the Voyage understanding
        embedding pipeline downstream — this is the single point that
        keeps query-side analysis aligned with ingestion-side analysis,
        preventing Voyage embedding-space drift.

        Args:
            image_base64: Base64 encoded image
            image_path: Image path/URL (for logging)
            image_id: Image ID (for logging)
            document_id: Document ID (for logging)
            embedding_service: Embedding service (unused, for compatibility)

        Returns:
            Dict with quality_score, confidence_score, material_properties.
        """
        try:
            import httpx
            from app.models.vision_analysis import VISION_ANALYSIS_TOOL

            anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
            if not anthropic_api_key:
                self.logger.warning("ANTHROPIC_API_KEY not set, skipping analysis")
                return {
                    'quality_score': 0.5,
                    'confidence_score': 0.0,
                    'material_properties': {},
                    'error': 'API key missing'
                }

            analysis_text = (
                "Analyze this building/interior material image. Use the "
                "emit_vision_analysis tool to return a structured material "
                "analysis. Be catalog-grade specific (e.g. 'Calacatta marble' "
                "not 'marble', 'herringbone white oak engineered wood' not "
                "'wood floor'). Confidence must reflect how certain you are "
                "of the material identification."
            )

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        'x-api-key': anthropic_api_key,
                        'anthropic-version': '2023-06-01',
                        'content-type': 'application/json',
                    },
                    json={
                        'model': 'claude-opus-4-7',
                        'max_tokens': 4096,
                        'tools': [VISION_ANALYSIS_TOOL],
                        'tool_choice': {'type': 'tool', 'name': 'emit_vision_analysis'},
                        'messages': [{
                            'role': 'user',
                            'content': [
                                {
                                    'type': 'image',
                                    'source': {
                                        'type': 'base64',
                                        'media_type': 'image/jpeg',
                                        'data': image_base64,
                                    },
                                },
                                {'type': 'text', 'text': analysis_text},
                            ],
                        }],
                    },
                )

                if response.status_code != 200:
                    self.logger.warning(f"Anthropic API error: {response.status_code} - {response.text[:200]}")
                    return {
                        'quality_score': 0.5,
                        'confidence_score': 0.0,
                        'material_properties': {},
                        'error': f'API error {response.status_code}'
                    }

                # Anthropic returns content as a list; the tool_use block has
                # `input` already validated against the schema by the API.
                payload = response.json()
                tool_block = next(
                    (b for b in payload.get('content', []) if b.get('type') == 'tool_use'),
                    None,
                )
                if not tool_block or 'input' not in tool_block:
                    self.logger.warning(f"No tool_use in Anthropic response: {payload}")
                    return {
                        'quality_score': 0.5,
                        'confidence_score': 0.0,
                        'material_properties': {},
                        'error': 'No tool_use block returned'
                    }

                result = tool_block['input']

                material_properties = {
                    'material_type': result.get('material_type', 'unknown'),
                    'category': result.get('category'),
                    'subcategory': result.get('subcategory'),
                    'colors': result.get('colors') or [],
                    'textures': result.get('textures') or [],
                    'finish': result.get('finish'),
                    'surface_pattern': result.get('surface_pattern'),
                    'description': result.get('description'),
                    'applications': result.get('applications') or [],
                    'style': result.get('style'),
                    'detected_text': result.get('detected_text') or [],
                }

                # Quality score: high when confidence is high. We don't have
                # a separate "image quality" field in the locked schema, so
                # we use confidence as a proxy — same value, same call site.
                confidence_score = float(result.get('confidence', 0.85))
                quality_score = confidence_score

                return {
                    'quality_score': quality_score,
                    'confidence_score': confidence_score,
                    'material_properties': material_properties,
                    'vision_analysis': result,  # full schema-locked dict for embedding
                    'model': 'claude-opus-4-7'
                }

        except Exception as e:
            self.logger.warning(f"Image analysis failed: {e}")
            return {
                'quality_score': 0.5,
                'confidence_score': 0.0,
                'material_properties': {},
                'error': f'Error: {str(e)}'
            }

    async def _classify_image_material(
        self,
        image_base64: str,
        confidence_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Classify if an image shows material content using Claude Opus 4.7 vision
        + Anthropic tool use for guaranteed schema adherence.

        Args:
            image_base64: Base64 encoded image
            confidence_threshold: Minimum confidence to classify as material

        Returns:
            Dict with is_material, confidence, reason, classification.
        """
        try:
            import httpx

            anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
            if not anthropic_api_key:
                self.logger.warning("ANTHROPIC_API_KEY not set, skipping classification")
                return {'is_material': False, 'confidence': 0.0, 'reason': 'API key missing'}

            if not self.classification_prompt:
                error_msg = "CRITICAL: Classification prompt not found in database. Add via /admin/ai-configs with prompt_type='classification', stage='image_analysis', category='image_classification'"
                self.logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            # Tool-use schema mirrors the existing database prompt's expected
            # output shape — classification + confidence + reasoning, plus
            # product_indicators array. Force-calling the tool guarantees
            # the API returns this exact structure.
            classify_tool = {
                "name": "emit_classification",
                "description": "Emit the image classification verdict for the building-materials catalog filter.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "classification": {
                            "type": "string",
                            "enum": ["PRODUCT_IMAGE", "TECHNICAL_DIAGRAM", "DECORATIVE", "MIXED"],
                            "description": "The dominant content category for this image.",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "0-1 confidence in the classification.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "One-sentence justification.",
                        },
                        "product_indicators": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Catalog-grade product/material elements observed.",
                        },
                    },
                    "required": ["classification", "confidence", "reasoning"],
                },
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        'x-api-key': anthropic_api_key,
                        'anthropic-version': '2023-06-01',
                        'content-type': 'application/json',
                    },
                    json={
                        'model': 'claude-opus-4-7',
                        'max_tokens': 1024,
                        'tools': [classify_tool],
                        'tool_choice': {'type': 'tool', 'name': 'emit_classification'},
                        'messages': [{
                            'role': 'user',
                            'content': [
                                {
                                    'type': 'image',
                                    'source': {
                                        'type': 'base64',
                                        'media_type': 'image/jpeg',
                                        'data': image_base64,
                                    },
                                },
                                {'type': 'text', 'text': self.classification_prompt, 'cache_control': {'type': 'ephemeral'}},
                            ],
                        }],
                    },
                )

                if response.status_code != 200:
                    error_body = response.text[:500] if hasattr(response, 'text') else 'No response body'
                    self.logger.error(f"❌ Anthropic API error {response.status_code}: {error_body}")
                    return {'is_material': False, 'confidence': 0.0, 'reason': f'API error {response.status_code}'}

                payload = response.json()
                tool_block = next(
                    (b for b in payload.get('content', []) if b.get('type') == 'tool_use'),
                    None,
                )
                if not tool_block or 'input' not in tool_block:
                    self.logger.warning(f"No tool_use in classification response: {payload}")
                    return {'is_material': False, 'confidence': 0.0, 'reason': 'No tool_use block returned'}

                result = tool_block['input']
                classification = result.get('classification', 'DECORATIVE')
                confidence = float(result.get('confidence', 0.5))
                reason = result.get('reasoning', 'Unknown')

                # Map structured classification → is_material binary that
                # legacy callers expect. PRODUCT_IMAGE and MIXED are kept;
                # TECHNICAL_DIAGRAM and DECORATIVE are dropped.
                is_material = (
                    classification in ("PRODUCT_IMAGE", "MIXED")
                    and confidence >= confidence_threshold
                )

                return {
                    'is_material': is_material,
                    'confidence': confidence,
                    'reason': reason,
                    'classification': classification,
                    'product_indicators': result.get('product_indicators') or [],
                    'model': 'claude-opus-4-7'
                }

        except Exception as e:
            self.logger.warning(f"Image classification failed: {e}")
            return {'is_material': False, 'confidence': 0.0, 'reason': f'Error: {str(e)}'}

    async def _search_chunks_by_embedding(
        self,
        query_embedding: List[float],
        workspace_id: str,
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Search document_chunks using text_embedding similarity.

        Uses SQL function: search_chunks_by_embedding(vector, UUID, INT)

        Args:
            query_embedding: Query embedding vector (1024D, matches DB schema vector(1024))
            workspace_id: Workspace ID to filter
            limit: Maximum results to return

        Returns:
            List of chunks with similarity scores
        """
        try:
            # Convert embedding to PostgreSQL vector format
            embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

            # Call SQL function
            result = self.supabase_client.client.rpc(
                'search_chunks_by_embedding',
                {
                    'query_embedding': embedding_str,
                    'p_workspace_id': workspace_id,
                    'p_limit': limit
                }
            ).execute()

            if result.data and len(result.data) > 0:
                self.logger.info(f"✅ Chunk search: {len(result.data)} results")
                return result.data
            else:
                self.logger.debug("⚠️ No chunks with embeddings found")
                return []

        except Exception as e:
            self.logger.warning(f"⚠️ Chunk search failed: {e}")
            # Graceful degradation: return empty list
            return []

    async def _search_products_by_embedding(
        self,
        query_embedding: List[float],
        workspace_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search products directly using text_embedding_1024 similarity.

        Uses SQL function: search_products_by_embedding(vector, UUID, INT)

        Args:
            query_embedding: Query embedding vector (will be truncated to 1024D)
            workspace_id: Workspace ID to filter
            limit: Maximum results to return

        Returns:
            List of products with similarity scores
        """
        try:
            # Truncate to 1024D if needed (products use 1024D embeddings)
            product_embedding = query_embedding[:1024] if len(query_embedding) > 1024 else query_embedding

            # Convert embedding to PostgreSQL vector format
            embedding_str = '[' + ','.join(str(x) for x in product_embedding) + ']'

            # Call SQL function
            result = self.supabase_client.client.rpc(
                'search_products_by_embedding',
                {
                    'query_embedding': embedding_str,
                    'p_workspace_id': workspace_id,
                    'p_limit': limit
                }
            ).execute()

            if result.data and len(result.data) > 0:
                self.logger.info(f"✅ Product search: {len(result.data)} results")
                return result.data
            else:
                self.logger.debug("⚠️ No products with embeddings found")
                return []

        except Exception as e:
            self.logger.warning(f"⚠️ Product search failed: {e}")
            # Graceful degradation: return empty list
            return []

    async def _search_products_fulltext(
        self,
        query: str,
        workspace_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search products using PostgreSQL full-text search on search_tsv.

        search_tsv indexes: name (A), manufacturer/factory (A), designer (B),
        material_category (B), finish (B), description (C), colors (C), textures (C).

        This catches queries by manufacturer name, designer, material type, etc.
        that embedding-based search may miss.
        """
        try:
            result = self.supabase_client.client.rpc(
                'search_products_fulltext',
                {
                    'search_query': query,
                    'p_workspace_id': workspace_id,
                    'p_limit': limit
                }
            ).execute()

            if result.data and len(result.data) > 0:
                self.logger.info(f"✅ Fulltext search: {len(result.data)} results")
                return result.data
            else:
                self.logger.debug("⚠️ No fulltext matches found")
                return []

        except Exception as e:
            self.logger.warning(f"⚠️ Fulltext search failed: {e}")
            return []

