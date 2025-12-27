"""
RAG Service - Direct Vector DB Implementation

This service provides multi-vector search capabilities using direct vector database queries.

Search Strategy:
- Multi-Vector Search: Combines 6 specialized CLIP embeddings in parallel
  1. text_embedding_1536 (20%) - Semantic text understanding
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

# Import utilities
from ..utils.circuit_breaker import CircuitBreaker, CircuitBreakerError
from ..utils.memory_monitor import memory_monitor

# Import services
from .real_embeddings_service import RealEmbeddingsService
from .supabase_client import get_supabase_client
from .vecs_service import get_vecs_service
from .ai_client_service import get_ai_client_service
from .ai_call_logger import AICallLogger
from .unified_chunking_service import UnifiedChunkingService, ChunkingConfig, ChunkingStrategy


class RAGService:
    """
    RAG Service for multi-vector search using direct vector database queries.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RAG service with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self._available = True  # Service is available after initialization

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

            self.logger.info("✅ RAG Service initialized")
        except Exception as e:
            self.logger.error(f"❌ RAG Service initialization failed: {e}")
            self._available = False
            raise

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
        pdf_content: bytes,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        catalog: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Index PDF content by extracting text, chunking, and generating embeddings.

        This method replaces the old LlamaIndex-based indexing with direct chunking
        and embedding generation using our current services.

        Args:
            pdf_content: PDF file content as bytes
            document_id: Unique document identifier
            metadata: Document metadata including workspace_id, filename, etc.
            catalog: Optional product catalog for category tagging

        Returns:
            Dict containing:
                - chunks_created: Number of chunks created
                - chunk_ids: List of created chunk IDs
                - success: Boolean indicating success
        """
        try:
            start_time = time.time()
            metadata = metadata or {}
            workspace_id = metadata.get('workspace_id', 'default')

            self.logger.info(f"📝 Indexing PDF content for document {document_id}")

            # Step 1: Extract text from PDF using PyMuPDF4LLM
            try:
                import pymupdf4llm
                import fitz

                # Open PDF from bytes
                pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")

                # Extract markdown text
                markdown_text = pymupdf4llm.to_markdown(pdf_doc)
                pdf_doc.close()

                self.logger.info(f"   ✅ Extracted {len(markdown_text)} characters from PDF")

            except Exception as e:
                self.logger.error(f"   ❌ PDF text extraction failed: {e}")
                return {
                    "success": False,
                    "error": f"PDF extraction failed: {str(e)}",
                    "chunks_created": 0,
                    "chunk_ids": []
                }

            # Step 2: Create chunks using UnifiedChunkingService
            try:
                chunks = await self.chunking_service.chunk_text(
                    text=markdown_text,
                    document_id=document_id,
                    metadata=metadata
                )

                self.logger.info(f"   ✅ Created {len(chunks)} chunks")

            except Exception as e:
                self.logger.error(f"   ❌ Chunking failed: {e}")
                return {
                    "success": False,
                    "error": f"Chunking failed: {str(e)}",
                    "chunks_created": 0,
                    "chunk_ids": []
                }

            # Step 3: Store chunks and generate embeddings
            chunk_ids = []
            chunks_created = 0

            for chunk in chunks:
                try:
                    # Prepare chunk record for database
                    chunk_record = {
                        'document_id': document_id,
                        'workspace_id': workspace_id,
                        'chunk_text': chunk.content,
                        'chunk_index': chunk.chunk_index,
                        'metadata': {
                            **chunk.metadata,
                            'start_position': chunk.start_position,
                            'end_position': chunk.end_position,
                            'quality_score': chunk.quality_score,
                            'total_chunks': chunk.total_chunks
                        }
                    }

                    # Add catalog category if available
                    if catalog and hasattr(catalog, 'catalog_factory'):
                        chunk_record['metadata']['catalog_factory'] = catalog.catalog_factory

                    # Insert chunk into database
                    result = self.supabase_client.client.table('chunks').insert(chunk_record).execute()

                    if result.data and len(result.data) > 0:
                        chunk_id = result.data[0]['id']
                        chunk_ids.append(chunk_id)
                        chunks_created += 1

                        # Generate text embedding asynchronously (don't wait)
                        # The embedding will be generated in the background
                        asyncio.create_task(self._generate_chunk_embedding(chunk_id, chunk.content))

                except Exception as e:
                    self.logger.warning(f"   ⚠️ Failed to store chunk {chunk.chunk_index}: {e}")
                    continue

            elapsed_time = time.time() - start_time
            self.logger.info(f"✅ Indexed PDF: {chunks_created} chunks created in {elapsed_time:.2f}s")

            return {
                "success": True,
                "chunks_created": chunks_created,
                "chunk_ids": chunk_ids,
                "processing_time": elapsed_time
            }

        except Exception as e:
            self.logger.error(f"❌ PDF indexing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "chunks_created": 0,
                "chunk_ids": []
            }

    async def _generate_chunk_embedding(self, chunk_id: str, chunk_text: str):
        """Generate and store embedding for a chunk (background task)."""
        try:
            # Generate text embedding
            embedding_result = await self.embeddings_service.generate_text_embedding(chunk_text)

            if embedding_result and 'embedding' in embedding_result:
                # Update chunk with embedding
                self.supabase_client.client.table('chunks')\
                    .update({'text_embedding_1536': embedding_result['embedding']})\
                    .eq('id', chunk_id)\
                    .execute()

                self.logger.debug(f"   ✅ Generated embedding for chunk {chunk_id}")
        except Exception as e:
            self.logger.warning(f"   ⚠️ Failed to generate embedding for chunk {chunk_id}: {e}")

    async def multi_vector_search(
        self,
        query: str,
        workspace_id: str,
        top_k: int = 10,
        material_filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        🎯 Multi-vector search combining ALL 6 specialized embeddings in parallel.

        Combines 6 specialized embeddings with intelligent weighting:
        1. text_embedding_1536 (20%) - Semantic text understanding
        2. visual_clip_embedding_512 (20%) - General visual similarity
        3. color_clip_embedding_512 (15%) - Color palette matching
        4. texture_clip_embedding_512 (15%) - Texture pattern matching
        5. style_clip_embedding_512 (15%) - Design style matching
        6. material_clip_embedding_512 (15%) - Material type matching

        Architecture:
        - Generate query embeddings from text (using RealEmbeddingsService)
        - Search all 6 VECS collections in parallel
        - Map image results to products via product_image_relationships
        - Combine scores with intelligent weighting
        - Apply metadata filters as soft boosts

        Args:
            query: Search query text
            workspace_id: Workspace ID to filter results
            top_k: Number of results to return
            material_filters: Optional JSONB metadata filters
            similarity_threshold: Minimum similarity score (default 0.3)

        Returns:
            Dictionary containing search results with weighted scores from all 6 embeddings
        """
        try:
            start_time = time.time()

            # ============================================================================
            # STEP 1: Generate query embeddings from text (for visual searches)
            # ============================================================================
            self.logger.info(f"🔍 Multi-vector search: Generating embeddings for query: '{query[:50]}...'")

            # Generate visual embedding from text description (will be used for all 6 searches)
            embedding_result = await self.embeddings_service.generate_visual_embedding(query)

            if not embedding_result.get("success"):
                self.logger.warning("Failed to generate visual embedding, falling back to text-only search")
                query_embedding = None
            else:
                query_embedding = embedding_result.get("embedding", [])
                self.logger.info(f"✅ Generated {len(query_embedding)}D visual embedding from query text")

            # ============================================================================
            # STEP 2: Search all 6 embedding collections in PARALLEL
            # ============================================================================
            embedding_scores = {}  # Maps image_id -> {embedding_type: score}

            if query_embedding:
                # Define all 6 embedding types to search
                embedding_types = ['visual', 'color', 'texture', 'style', 'material']

                # Create parallel search tasks for specialized embeddings
                search_tasks = []
                for emb_type in embedding_types:
                    task = self.vecs_service.search_specialized_embeddings(
                        query_embedding=query_embedding,
                        embedding_type=emb_type,
                        limit=top_k * 3,  # Get more candidates for fusion
                        workspace_id=workspace_id,
                        include_metadata=True
                    )
                    search_tasks.append(task)

                # Execute all searches in parallel
                self.logger.info(f"🚀 Executing {len(embedding_types)} specialized embedding searches in parallel...")
                search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

                # Process results from each embedding type
                for emb_type, results in zip(embedding_types, search_results):
                    if isinstance(results, Exception):
                        self.logger.warning(f"⚠️ {emb_type} search failed: {results}")
                        continue

                    if not results:
                        self.logger.debug(f"No results from {emb_type} search")
                        continue

                    # Store scores for each image
                    for item in results:
                        image_id = item.get('image_id')
                        similarity_score = item.get('similarity_score', 0.0)

                        if image_id not in embedding_scores:
                            embedding_scores[image_id] = {}

                        embedding_scores[image_id][emb_type] = similarity_score

                    self.logger.info(f"✅ {emb_type}: {len(results)} results")

                self.logger.info(f"📊 Total unique images found: {len(embedding_scores)}")

            else:
                self.logger.warning("⚠️ No query embedding available, skipping visual searches")

            # ============================================================================
            # STEP 3: Map images to products via product_image_relationships
            # ============================================================================
            product_scores = {}  # Maps product_id -> {embedding_type: score}

            if embedding_scores:
                # Get all image IDs
                image_ids = list(embedding_scores.keys())

                # Fetch product-image relationships in batches
                batch_size = 100
                all_relationships = []

                for i in range(0, len(image_ids), batch_size):
                    batch_ids = image_ids[i:i + batch_size]
                    rel_response = self.supabase_client.client.table('product_image_relationships')\
                        .select('product_id, image_id, relevance_score')\
                        .in_('image_id', batch_ids)\
                        .execute()

                    if rel_response.data:
                        all_relationships.extend(rel_response.data)

                self.logger.info(f"📎 Found {len(all_relationships)} product-image relationships")

                # Map image scores to products
                for rel in all_relationships:
                    product_id = rel.get('product_id')
                    image_id = rel.get('image_id')
                    relevance = rel.get('relevance_score', 1.0)

                    if image_id in embedding_scores:
                        if product_id not in product_scores:
                            product_scores[product_id] = {}

                        # Transfer scores from image to product (weighted by relevance)
                        for emb_type, score in embedding_scores[image_id].items():
                            weighted_score = score * relevance

                            # Keep the highest score for each embedding type
                            if emb_type not in product_scores[product_id] or product_scores[product_id][emb_type] < weighted_score:
                                product_scores[product_id][emb_type] = weighted_score

                self.logger.info(f"🎯 Mapped scores to {len(product_scores)} products")

            # ============================================================================
            # STEP 4: Fetch product details and calculate weighted scores
            # ============================================================================

            # Define weights for each embedding type (must sum to 1.0)
            embedding_weights = {
                'text': 0.20,      # Text semantic understanding (from metadata/description)
                'visual': 0.20,    # General visual similarity
                'color': 0.15,     # Color palette matching
                'texture': 0.15,   # Texture pattern matching
                'style': 0.15,     # Design style matching
                'material': 0.15   # Material type matching
            }

            # Fetch all products that have scores
            results = []
            if product_scores:
                product_ids = list(product_scores.keys())

                # Fetch product details in batches
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

                # Calculate weighted scores for each product
                for product in all_products:
                    product_id = product['id']
                    scores = product_scores.get(product_id, {})

                    # Calculate text score from keyword matching
                    text_score = self._calculate_text_score(query, product)
                    scores['text'] = text_score

                    # Calculate weighted score from all embedding types
                    weighted_score = 0.0
                    score_breakdown = {}

                    for emb_type, weight in embedding_weights.items():
                        emb_score = scores.get(emb_type, 0.0)
                        weighted_score += emb_score * weight
                        score_breakdown[f'{emb_type}_score'] = emb_score

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
                            **score_breakdown,  # Include individual embedding scores
                            "filter_boost": filter_boost,
                            "search_type": "multi_vector_6_embeddings"
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
                "embedding_weights": embedding_weights,
                "embeddings_used": list(embedding_weights.keys()),
                "material_filters_applied": material_filters if material_filters else None,
                "metadata_validation_enabled": True,
                "query": query,
                "search_type": "multi_vector_6_embeddings"
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
                chunk_text = chunk.get('chunk_text', chunk.get('text', ''))
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
                    "text_snippet": chunk.get('chunk_text', '')[:200] + "...",
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
                chunk_text = chunk.get('chunk_text', chunk.get('text', ''))
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
                    "text_snippet": chunk.get('chunk_text', '')[:200] + "...",
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

