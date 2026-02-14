"""
RAG Service - Direct Vector DB Implementation

This service provides multi-vector search capabilities using direct vector database queries.

Search Strategy:
- Multi-Vector Search: Combines 6 specialized CLIP embeddings in parallel
  1. text_embedding (1024D, 20%) - Semantic text understanding (Voyage AI)
  2. visual_clip_embedding_512 (20%) - General visual similarity
  3. color_clip_embedding_512 (15%) - Color palette matching
  4. texture_clip_embedding_512 (15%) - Texture pattern matching
  5. style_clip_embedding_512 (15%) - Design style matching
  6. material_clip_embedding_512 (15%) - Material type matching
"""

import logging
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

        # Load HuggingFace endpoint configuration from settings
        from app.config import get_settings
        settings = get_settings()
        qwen_config = settings.get_qwen_config()

        self.qwen_endpoint_url = qwen_config["endpoint_url"]
        self.qwen_endpoint_token = qwen_config["endpoint_token"]

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
        pre_extracted_text: Optional[str] = None,
        page_chunks: Optional[List[Dict[str, Any]]] = None,
        progress_callback: Optional[callable] = None,
        layout_regions_by_page: Optional[Dict[int, List[Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        """
        Index PDF content by extracting text, chunking, and generating embeddings.

        This method replaces the old LlamaIndex-based indexing with direct chunking
        and embedding generation using our current services.

        Args:
            pdf_content: PDF file content as bytes (optional if pre_extracted_text or page_chunks is provided)
            document_id: Unique document identifier
            metadata: Document metadata including workspace_id, filename, etc.
            catalog: Optional product catalog for category tagging
            pre_extracted_text: Optional pre-extracted text (skips PDF extraction if provided)
            page_chunks: Optional page-aware text data from PyMuPDF4LLM (preserves page numbers)
            progress_callback: Optional async callback for progress updates (current, total, step_name)
            layout_regions_by_page: Optional dict mapping page_number -> list of YOLO layout regions
                                   for layout-aware chunking

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

            # Step 1: Extract text from PDF with page information
            pages = None
            if page_chunks:
                # ✅ NEW: Use page-aware data from Stage 1 (preserves page numbers)
                self.logger.info(f"   ✅ Using page-aware data ({len(page_chunks)} pages)")
                pages = page_chunks
            elif pre_extracted_text:
                # Use pre-extracted text (from Stage 1 focused extraction)
                # Convert to page format for consistency
                self.logger.info(f"   ✅ Using pre-extracted text ({len(pre_extracted_text)} characters)")
                # Wrap in single page dict (no page info available from pre-extracted text)
                pages = [{"metadata": {"page": 0}, "text": pre_extracted_text}]
            else:
                # Extract text from PDF using PyMuPDF4LLM with page_chunks=True
                try:
                    import pymupdf4llm
                    import tempfile
                    import os

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

                # 🔥 CRITICAL MEMORY FIX: Delete pages immediately after chunking
                # Pages can be 100K+ characters and are no longer needed
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

            # Step 3: Store chunks and generate embeddings with BATCH PROCESSING
            # ✅ MEMORY OPTIMIZATION: Process chunks in batches to avoid OOM
            from app.utils.memory_monitor import MemoryPressureMonitor

            memory_monitor = MemoryPressureMonitor()
            CHUNK_BATCH_SIZE = 15  # Process 15 chunks at a time (proven optimal from commit 74e7a73)
            MEMORY_THRESHOLD_PERCENT = 85.0  # Pause if memory usage exceeds 85%
            total_chunks = len(chunks)

            chunk_ids = []
            chunks_created = 0
            total_embeddings_stored = 0  # Track total embeddings across all batches

            # FIX #4: Log initial memory state before starting indexing
            try:
                initial_mem = memory_monitor.get_memory_stats()
                self.logger.info(f"🧠 Initial memory: {initial_mem.percent_used:.1f}% ({initial_mem.used_mb:.0f}MB / {initial_mem.total_mb:.0f}MB)")

                # Check if we have enough memory to proceed
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

                    # FIX #4: Check memory threshold and pause if needed
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
                        # ✅ CRITICAL FIX: Skip chunks with null/empty content
                        if not chunk.content or not chunk.content.strip():
                            self.logger.warning(f"   ⚠️ Skipping chunk {chunk.chunk_index} - empty content")
                            continue

                        # Prepare chunk record for database
                        chunk_record = {
                            'document_id': document_id,
                            'workspace_id': workspace_id,
                            'content': chunk.content.strip(),  # ✅ FIXED: Use 'content' column, not 'chunk_text', and strip whitespace
                            'chunk_index': chunk.chunk_index,
                            'metadata': {
                                **chunk.metadata,
                                'start_position': chunk.start_position,
                                'end_position': chunk.end_position,
                                'quality_score': chunk.quality_score,
                                'total_chunks': chunk.total_chunks
                            },
                            'quality_score': chunk.quality_score  # ✅ NEW: Add quality_score as top-level column for dashboard metrics
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
                            dimensions=1024,  # ✅ FIXED: Chunks use 1024D embeddings (matches DB schema: vector(1024))
                            input_type="document"
                        )

                        # FIX #2: Batch database updates instead of individual UPDATE queries
                        # This prevents memory accumulation from 100+ individual Supabase responses
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

                            # ✅ CRITICAL FIX: Use individual UPDATE instead of UPSERT to avoid null content errors
                            # UPSERT tries to INSERT if ID doesn't exist, which fails on NOT NULL content constraint
                            # UPDATE only modifies existing rows, which is what we want here
                            if updates:
                                embeddings_stored = 0
                                for update in updates:
                                    try:
                                        self.supabase_client.client.table('document_chunks')\
                                            .update({'text_embedding': update['text_embedding']})\
                                            .eq('id', update['id'])\
                                            .execute()
                                        embeddings_stored += 1
                                    except Exception as individual_update_error:
                                        self.logger.warning(f"   ⚠️ Failed to update chunk {update['id']}: {individual_update_error}")

                                total_embeddings_stored += embeddings_stored  # Accumulate total
                                self.logger.info(f"   ✅ Stored {embeddings_stored}/{len(updates)} embeddings")

                    except Exception as batch_embedding_error:
                        self.logger.error(f"   ❌ Batch embedding generation failed: {batch_embedding_error}", exc_info=True)

                        # Fallback to individual embedding generation
                        self.logger.info(f"   ⚠️ Falling back to individual embedding generation for batch {batch_num}...")
                        embeddings_stored = 0
                        for chunk_id, text in zip(batch_chunk_ids, batch_texts):
                            try:
                                embedding = await self.embeddings_service.generate_embedding(text, dimensions=1024)  # ✅ FIXED: Use 1024D (matches DB schema)
                                if embedding:
                                    self.supabase_client.client.table('document_chunks')\
                                        .update({'text_embedding': embedding})\
                                        .eq('id', chunk_id)\
                                        .execute()
                                    embeddings_stored += 1
                            except Exception as individual_error:
                                self.logger.warning(f"   ⚠️ Failed to generate individual embedding for chunk {chunk_id}: {individual_error}")

                        if embeddings_stored > 0:
                            total_embeddings_stored += embeddings_stored  # Accumulate total
                            self.logger.info(f"   ✅ Stored {embeddings_stored} embeddings via individual fallback")

                # Step 3c: Cleanup after batch (FIX #1: embedding_vectors now always defined)
                del batch_chunk_ids, batch_texts, batch_chunk_records, embedding_vectors
                gc.collect()

                # Log memory usage after batch
                try:
                    mem_stats = memory_monitor.get_memory_stats()
                    self.logger.info(f"🧠 Memory after batch {batch_num}/{total_batches}: {mem_stats.percent_used:.1f}% ({mem_stats.used_mb:.0f}MB / {mem_stats.total_mb:.0f}MB)")
                except Exception as mem_error:
                    self.logger.debug(f"Memory monitoring unavailable: {mem_error}")

            # 🔥 CRITICAL MEMORY FIX: Delete chunks list after all batches complete
            # Chunks can be 100+ objects with large text content
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
                    "semantic_duplicates_prevented": quality_metrics.semantic_duplicates_prevented,
                    "low_quality_rejected": quality_metrics.low_quality_rejected,
                    "final_chunks": quality_metrics.final_chunks
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
        search_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        🎯 HYBRID MULTI-SOURCE SEARCH - Combines ALL available data sources.

        **SEARCH SOURCES (All enabled by default):**
        1. **Visual Embeddings** (5 VECS collections) - Image-based search
           - visual_clip_embedding_512 (general visual similarity)
           - color_clip_embedding_512 (color palette matching)
           - texture_clip_embedding_512 (texture pattern matching)
           - style_clip_embedding_512 (design style matching)
           - material_clip_embedding_512 (material type matching)

        2. **Text Embeddings** (document_chunks) - Semantic text search
           - Searches chunk text_embedding (1024D Voyage AI, stored as vector(1024))
           - Maps chunks to products via chunk_product_relationships

        3. **Direct Product Search** - Product-level embeddings
           - Searches product text_embedding_1024 (1024D, stored as vector(1024))
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
            # STEP 1: Generate query embeddings (visual + text)
            # ============================================================================
            self.logger.info(f"🔍 HYBRID SEARCH: '{query[:50]}...'")
            self.logger.info(f"   Sources: Visual={enable_visual}, Chunk={enable_chunk}, Product={enable_product}, Keyword={enable_keyword}")

            # Generate visual embedding from text (for image searches)
            visual_embedding = None
            if enable_visual:
                embedding_result = await self.embeddings_service.generate_visual_embedding(query)
                if embedding_result.get("success"):
                    visual_embedding = embedding_result.get("embedding", [])
                    self.logger.info(f"✅ Visual embedding: {len(visual_embedding)}D")
                else:
                    self.logger.warning("⚠️ Visual embedding generation failed")

            # Generate text embedding (for chunk/product searches)
            text_embedding = None
            if enable_chunk or enable_product:
                text_result = await self.embeddings_service.generate_text_embedding(query)
                if text_result.get("success"):
                    text_embedding = text_result.get("embedding", [])
                    self.logger.info(f"✅ Text embedding: {len(text_embedding)}D")
                else:
                    self.logger.warning("⚠️ Text embedding generation failed")

            # Generate understanding query embedding (for vision-understanding search)
            understanding_embedding = None
            understanding_result = await self.embeddings_service.generate_understanding_query_embedding(query)
            if understanding_result.get("success"):
                understanding_embedding = understanding_result.get("embedding", [])
                self.logger.info(f"✅ Understanding embedding: {len(understanding_embedding)}D")

            # ============================================================================
            # STEP 2A: Search VISUAL + UNDERSTANDING embeddings (6 VECS collections)
            # ============================================================================
            image_scores = {}  # Maps image_id -> {embedding_type: score}

            if enable_visual and visual_embedding:
                embedding_types = ['visual', 'color', 'texture', 'style', 'material']
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
                    # ✅ UPDATED: Use image_product_associations table
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
                    relevance = rel.get('overall_score', 1.0)  # ✅ UPDATED: Use overall_score

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

            # 3C: Add direct product scores (NEW!)
            if direct_product_scores:
                self.logger.info(f"📎 Direct product scores: {len(direct_product_scores)}")

                for product_id, score in direct_product_scores.items():
                    if product_id not in product_scores:
                        product_scores[product_id] = {}
                    product_scores[product_id]['product'] = score

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
                        scores['keyword'] = keyword_score

                    # Calculate weighted score from ALL sources
                    weighted_score = 0.0
                    score_breakdown = {}

                    for source_type, weight in embedding_weights.items():
                        source_score = scores.get(source_type, 0.0)
                        weighted_score += source_score * weight
                        score_breakdown[f'{source_type}_score'] = source_score

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

                    # Only include results above threshold
                    if final_score >= similarity_threshold:
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

                # Limit to top_k
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

            # Track search query for prototype discovery
            try:
                from .search_query_tracker import get_search_tracker
                tracker = get_search_tracker()
                # Track asynchronously (don't wait)
                asyncio.create_task(tracker.track_query(
                    workspace_id=workspace_id,
                    query_text=query,
                    query_metadata=material_filters,
                    search_type="multi_vector",
                    result_count=len(results),
                    response_time_ms=int((time.time() - start_time) * 1000)
                ))
            except Exception as e:
                self.logger.warning(f"Search tracking failed: {e}")

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

        # Product description (medium weight)
        if product.get('description'):
            text_parts.append(('description', product['description'], 2.0))

        # Metadata fields (lower weight)
        metadata = product.get('metadata', {})
        if metadata:
            # Flatten metadata to text
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

            # Load model if not already loaded
            await self.embeddings_service.ensure_models_loaded()

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
                chunk_text = chunk.get('content', chunk.get('text', ''))  # ✅ FIXED: Use 'content' column
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

            # Step 4: Call Claude 4.5 Sonnet
            client = self.ai_client_service.anthropic
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.content[0].text.strip()

            # Step 5: Log AI call
            latency_ms = int((time.time() - start_time) * 1000)
            await self.ai_logger.log_claude_call(
                task="rag_query_document",
                model="claude-sonnet-4-5-20250929",
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
                    "text_snippet": chunk.get('content', '')[:200] + "...",  # ✅ FIXED: Use 'content' column
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
                    "model": "claude-sonnet-4-5-20250929"
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
                chunk_text = chunk.get('content', chunk.get('text', ''))  # ✅ FIXED: Use 'content' column
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

            # Step 6: Call Claude 4.5 Sonnet
            client = self.ai_client_service.anthropic
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                temperature=0.1 if query_type == "factual" else 0.3,
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
                model="claude-sonnet-4-5-20250929",
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
                    "text_snippet": chunk.get('content', '')[:200] + "...",  # ✅ FIXED: Use 'content' column
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
                    "model": "claude-sonnet-4-5-20250929"
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
        Analyze material image using Qwen Vision to extract properties, quality, and confidence.

        This is the FULL analysis method (not just classification).
        Returns quality_score, confidence_score, and material_properties.

        Args:
            image_base64: Base64 encoded image
            image_path: Image path/URL (for logging)
            image_id: Image ID (for logging)
            document_id: Document ID (for logging)
            embedding_service: Embedding service (unused, for compatibility)

        Returns:
            Dict with quality_score, confidence_score, material_properties
        """
        try:
            import json
            import httpx
            import os

            huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
            if not huggingface_api_key:
                self.logger.warning("HUGGINGFACE_API_KEY not set, skipping analysis")
                return {
                    'quality_score': 0.5,
                    'confidence_score': 0.0,
                    'material_properties': {},
                    'error': 'API key missing'
                }

            analysis_prompt = """Analyze this building/interior material image and extract detailed properties.

Respond with JSON:
{
  "material_type": "tile|wood|fabric|stone|metal|flooring|wallpaper|other",
  "color": "primary color description",
  "finish": "matte|glossy|satin|textured|other",
  "pattern": "solid|striped|geometric|floral|abstract|other",
  "texture": "smooth|rough|embossed|woven|other",
  "quality_assessment": 0.0-1.0,
  "confidence": 0.0-1.0,
  "notes": "brief description"
}"""

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.qwen_endpoint_url,
                    headers={
                        'Authorization': f'Bearer {self.qwen_endpoint_token}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'Qwen/Qwen3-VL-8B-Instruct',
                        'messages': [{
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': analysis_prompt},
                                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_base64}'}}
                            ]
                        }],
                        'max_tokens': 500,
                        'temperature': 0.1
                    }
                )

                if response.status_code != 200:
                    self.logger.warning(f"Qwen API error: {response.status_code}")
                    return {
                        'quality_score': 0.5,
                        'confidence_score': 0.0,
                        'material_properties': {},
                        'error': f'API error {response.status_code}'
                    }

                result_text = response.json()['choices'][0]['message']['content']
                result = json.loads(result_text)

                # Extract material properties
                material_properties = {
                    'material_type': result.get('material_type', 'unknown'),
                    'color': result.get('color'),
                    'finish': result.get('finish'),
                    'pattern': result.get('pattern'),
                    'texture': result.get('texture'),
                    'notes': result.get('notes', '')
                }

                # Use Qwen's assessments for scores
                quality_score = result.get('quality_assessment', 0.7)
                confidence_score = result.get('confidence', 0.7)

                return {
                    'quality_score': quality_score,
                    'confidence_score': confidence_score,
                    'material_properties': material_properties,
                    'model': 'qwen3-vl-8b'
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
        Classify if an image shows material or not using Qwen Vision.

        Args:
            image_base64: Base64 encoded image
            confidence_threshold: Minimum confidence to classify as material

        Returns:
            Dict with is_material, confidence, reason
        """
        try:
            import json
            import httpx
            import os

            huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
            if not huggingface_api_key:
                self.logger.warning("HUGGINGFACE_API_KEY not set, skipping classification")
                return {'is_material': False, 'confidence': 0.0, 'reason': 'API key missing'}

            # Use database prompt - NO FALLBACK
            if self.classification_prompt:
                classification_prompt = self.classification_prompt
            else:
                error_msg = "CRITICAL: Classification prompt not found in database. Add via /admin/ai-configs with prompt_type='classification', stage='image_analysis', category='image_classification'"
                self.logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.qwen_endpoint_url,
                    headers={
                        'Authorization': f'Bearer {self.qwen_endpoint_token}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'Qwen/Qwen3-VL-8B-Instruct',
                        'messages': [{
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': classification_prompt},
                                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_base64}'}}
                            ]
                        }],
                        'max_tokens': 200,
                        'temperature': 0.1
                    }
                )

                if response.status_code != 200:
                    error_body = response.text[:500] if hasattr(response, 'text') else 'No response body'
                    self.logger.error(f"❌ Qwen API error {response.status_code}: {error_body}")
                    return {'is_material': False, 'confidence': 0.0, 'reason': f'API error {response.status_code}'}

                result_text = response.json()['choices'][0]['message']['content']

                # ✅ COMPREHENSIVE LOGGING: Log FULL raw response for debugging
                self.logger.info(f"📝 Qwen RAW Response (length: {len(result_text)} chars):")
                self.logger.info(f"   First 500 chars: {result_text[:500]}")
                if len(result_text) > 500:
                    self.logger.info(f"   Last 200 chars: ...{result_text[-200:]}")

                # Try to parse JSON, with fallback for malformed responses
                try:
                    result = json.loads(result_text)
                    self.logger.info(f"✅ Qwen JSON parsed successfully: {result}")
                except json.JSONDecodeError as json_err:
                    # ✅ ENHANCED ERROR LOGGING: Show exact error location and context
                    self.logger.error(f"❌ Qwen JSON Parse Error: {json_err}")
                    self.logger.error(f"   Error at line {json_err.lineno}, column {json_err.colno}")
                    self.logger.error(f"   Full response text ({len(result_text)} chars):")
                    self.logger.error(f"   {result_text}")

                    # Try to extract JSON from markdown code blocks
                    if '```json' in result_text:
                        self.logger.info("   Attempting to extract JSON from ```json block...")
                        try:
                            json_match = result_text.split('```json')[1].split('```')[0].strip()
                            result = json.loads(json_match)
                            self.logger.info(f"   ✅ Successfully extracted JSON from markdown: {result}")
                        except Exception as extract_err:
                            self.logger.error(f"   ❌ Failed to extract from ```json block: {extract_err}")
                            return {'is_material': False, 'confidence': 0.0, 'reason': f'Invalid JSON in markdown block'}
                    elif '```' in result_text:
                        self.logger.info("   Attempting to extract JSON from ``` block...")
                        try:
                            json_match = result_text.split('```')[1].split('```')[0].strip()
                            result = json.loads(json_match)
                            self.logger.info(f"   ✅ Successfully extracted JSON from code block: {result}")
                        except Exception as extract_err:
                            self.logger.error(f"   ❌ Failed to extract from ``` block: {extract_err}")
                            return {'is_material': False, 'confidence': 0.0, 'reason': f'Invalid JSON in code block'}
                    else:
                        # ✅ LOG FULL RESPONSE when no JSON found
                        self.logger.error(f"   ❌ No JSON or markdown blocks found in response")
                        self.logger.error(f"   Response type: {type(result_text)}")
                        self.logger.error(f"   Response repr: {repr(result_text[:1000])}")
                        return {'is_material': False, 'confidence': 0.0, 'reason': f'Invalid JSON response'}

                return {
                    'is_material': result.get('is_material', False),
                    'confidence': result.get('confidence', 0.5),
                    'reason': result.get('reason', 'Unknown'),
                    'model': 'qwen3-vl-8b'
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


