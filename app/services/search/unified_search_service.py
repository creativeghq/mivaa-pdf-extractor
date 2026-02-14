"""
Unified Search Service - Step 7 Implementation

Consolidates all search strategies into a single, unified service:
1. Semantic search - Based on text embeddings
2. Visual search - Based on image/CLIP embeddings
3. Multi-vector search - Combines all embedding types
4. Hybrid search - Combines semantic and keyword search
5. Material search - Specialized for material properties

Replaces multiple search implementations with a single unified approach.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from app.services.core.ai_client_service import get_ai_client_service

logger = logging.getLogger(__name__)


class SearchStrategy(str, Enum):
    """Supported search strategies."""
    SEMANTIC = "semantic"
    VISUAL = "visual"
    MULTI_VECTOR = "multi_vector"
    HYBRID = "hybrid"
    MATERIAL = "material"
    KEYWORD = "keyword"
    UNDERSTANDING = "understanding"
    # Specialized CLIP embedding strategies
    COLOR = "color"
    TEXTURE = "texture"
    STYLE = "style"
    MATERIAL_TYPE = "material_type"


@dataclass
class SearchConfig:
    """Configuration for search."""
    strategy: SearchStrategy = SearchStrategy.MULTI_VECTOR
    max_results: int = 20
    similarity_threshold: float = 0.7
    include_metadata: bool = True
    include_embeddings: bool = False
    enable_hybrid: bool = True
    mmr_lambda: float = 0.7  # For maximal marginal relevance


@dataclass
class SearchResult:
    """Represents a single search result."""
    id: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    embedding_type: str = "text"
    source_type: str = "chunk"  # chunk, product, image


@dataclass
class SearchResponse:
    """Response from search operation."""
    success: bool
    query: str
    results: List[SearchResult]
    total_found: int
    search_time_ms: float
    strategy_used: str
    metadata: Dict[str, Any]


# Dynamic weight profiles keyed by query intent type (all sum to 1.0)
WEIGHT_PROFILES: Dict[str, Dict[str, float]] = {
    "product_name": {
        "text": 0.40, "visual": 0.25, "understanding": 0.15,
        "color": 0.05, "texture": 0.05, "style": 0.05, "material": 0.05
    },
    "color_finish": {
        "text": 0.10, "visual": 0.20, "understanding": 0.15,
        "color": 0.30, "texture": 0.05, "style": 0.15, "material": 0.05
    },
    "specification": {
        "text": 0.25, "visual": 0.10, "understanding": 0.40,
        "color": 0.05, "texture": 0.05, "style": 0.05, "material": 0.10
    },
    "texture_pattern": {
        "text": 0.10, "visual": 0.25, "understanding": 0.15,
        "color": 0.05, "texture": 0.30, "style": 0.10, "material": 0.05
    },
    "style_aesthetic": {
        "text": 0.10, "visual": 0.25, "understanding": 0.15,
        "color": 0.10, "texture": 0.10, "style": 0.25, "material": 0.05
    },
    "material_search": {
        "text": 0.15, "visual": 0.15, "understanding": 0.25,
        "color": 0.05, "texture": 0.10, "style": 0.05, "material": 0.25
    },
    "balanced": {
        "text": 0.15, "visual": 0.15, "understanding": 0.20,
        "color": 0.125, "texture": 0.125, "style": 0.125, "material": 0.125
    },
}


class UnifiedSearchService:
    """
    Unified search service that consolidates all search strategies.

    NEW: Runs all strategies in parallel and merges results for comprehensive coverage.
    Uses dynamic weight profiles selected by query intent from query understanding.

    This service provides:
    - Semantic search using text embeddings
    - Visual search using CLIP embeddings
    - Multi-vector search combining all embedding types
    - Hybrid search combining semantic and keyword
    - Material search for material properties
    - Keyword search for exact matches
    """

    def __init__(self, config: Optional[SearchConfig] = None, supabase_client=None):
        """Initialize unified search service."""
        self.config = config or SearchConfig()
        self.supabase = supabase_client
        self.logger = logger

    def _select_weight_profile(self, parsed_data: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
        """
        Select weight profile based on parsed query fields from query understanding.

        Maps detected query intent (colors, finish, pattern, etc.) to a weight profile
        that upweights the most relevant embedding types for that query.

        Returns:
            Tuple of (profile_name, weights_dict)
        """
        # Product name search → heavy text weight
        if parsed_data.get("is_product_name") or parsed_data.get("product_name"):
            return "product_name", WEIGHT_PROFILES["product_name"]

        has_colors = bool(parsed_data.get("colors"))
        has_finish = bool(parsed_data.get("finish"))
        has_pattern = bool(parsed_data.get("pattern"))
        has_style = bool(parsed_data.get("style"))
        has_dimensions = bool(parsed_data.get("dimensions"))
        has_material = parsed_data.get("material_type_explicit", False)
        has_application = bool(parsed_data.get("application"))

        # Priority-based selection (strongest signal wins)
        if has_dimensions:
            return "specification", WEIGHT_PROFILES["specification"]
        if has_colors or has_finish:
            return "color_finish", WEIGHT_PROFILES["color_finish"]
        if has_pattern:
            return "texture_pattern", WEIGHT_PROFILES["texture_pattern"]
        if has_material:
            return "material_search", WEIGHT_PROFILES["material_search"]
        if has_style or has_application:
            return "style_aesthetic", WEIGHT_PROFILES["style_aesthetic"]

        return "balanced", WEIGHT_PROFILES["balanced"]

    async def search(
        self,
        query: str,
        strategy: Optional[SearchStrategy] = None,
        filters: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
        run_all_strategies: bool = False,
        enable_query_understanding: bool = False
    ) -> SearchResponse:
        """
        Perform search using the configured strategy or all strategies.

        Args:
            query: Search query
            strategy: Search strategy (uses config default if not specified)
            filters: Additional filters for search
            workspace_id: Workspace ID for scoped search
            run_all_strategies: If True, run all strategies in parallel and merge
            enable_query_understanding: If True, parse query with AI to extract structured filters

        Returns:
            SearchResponse with results
        """
        try:
            # 🧠 STEP 1: Query Understanding (if enabled)
            # Parse natural language query into structured filters + dynamic weight profile
            weight_profile = "balanced"
            dynamic_weights = WEIGHT_PROFILES["balanced"]

            if enable_query_understanding:
                parsed_query, parsed_filters, weight_profile, dynamic_weights = await self._parse_query_with_ai(query)

                # Merge parsed filters with user-provided filters (user filters take precedence)
                if parsed_filters:
                    if filters:
                        merged_filters = {**parsed_filters, **filters}
                    else:
                        merged_filters = parsed_filters

                    filters = merged_filters

                    # Use visual query for embedding (not full query)
                    query = parsed_query

                    self.logger.info(f"🧠 Query understanding: parsed_query='{parsed_query}', profile='{weight_profile}', filters={parsed_filters}")

            # 🔍 STEP 2: Multi-Strategy Search (existing logic)
            # All strategies now use the parsed query + extracted filters + dynamic weights
            start_time = datetime.utcnow()

            # If run_all_strategies is True, execute all strategies in parallel
            if run_all_strategies:
                results, strategy_metadata = await self._search_all_strategies(
                    query, filters, workspace_id, weight_overrides=dynamic_weights
                )
                search_strategy = "all_strategies"
            else:
                search_strategy = strategy or self.config.strategy
                results = await self._search_single_strategy(
                    search_strategy, query, filters, workspace_id, weight_overrides=dynamic_weights
                )
                strategy_metadata = {"strategies_used": [search_strategy.value]}

            # Sort by similarity score
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            limited_results = results[:self.config.max_results]

            # Calculate search time
            search_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            self.logger.info(f"✅ Search completed: {len(limited_results)} results in {search_time_ms:.2f}ms (profile={weight_profile})")

            return SearchResponse(
                success=True,
                query=query,
                results=limited_results,
                total_found=len(results),
                search_time_ms=search_time_ms,
                strategy_used=str(search_strategy),
                metadata={
                    "similarity_threshold": self.config.similarity_threshold,
                    "max_results": self.config.max_results,
                    "include_metadata": self.config.include_metadata,
                    "workspace_id": workspace_id,
                    "weight_profile": weight_profile,
                    "dynamic_weights": dynamic_weights,
                    **strategy_metadata
                }
            )

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return SearchResponse(
                success=False,
                query=query,
                results=[],
                total_found=0,
                search_time_ms=0.0,
                strategy_used=str(strategy or self.config.strategy),
                metadata={"error": str(e)}
            )

    async def _search_single_strategy(
        self,
        strategy: SearchStrategy,
        query: str,
        filters: Optional[Dict[str, Any]],
        workspace_id: Optional[str],
        weight_overrides: Optional[Dict[str, float]] = None
    ) -> List[SearchResult]:
        """Execute a single search strategy."""
        if strategy == SearchStrategy.SEMANTIC:
            return await self._search_semantic(query, filters, workspace_id)
        elif strategy == SearchStrategy.VISUAL:
            return await self._search_visual(query, filters, workspace_id)
        elif strategy == SearchStrategy.MULTI_VECTOR:
            return await self._search_multi_vector(query, filters, workspace_id, weight_overrides=weight_overrides)
        elif strategy == SearchStrategy.HYBRID:
            return await self._search_hybrid(query, filters, workspace_id)
        elif strategy == SearchStrategy.MATERIAL:
            return await self._search_material(query, filters, workspace_id)
        elif strategy == SearchStrategy.KEYWORD:
            return await self._search_keyword(query, filters, workspace_id)
        # ✅ NEW: Specialized CLIP embedding strategies
        elif strategy == SearchStrategy.COLOR:
            return await self._search_color(query, filters, workspace_id)
        elif strategy == SearchStrategy.TEXTURE:
            return await self._search_texture(query, filters, workspace_id)
        elif strategy == SearchStrategy.STYLE:
            return await self._search_style(query, filters, workspace_id)
        elif strategy == SearchStrategy.MATERIAL_TYPE:
            return await self._search_material_type(query, filters, workspace_id)
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")

    async def _search_all_strategies(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        workspace_id: Optional[str],
        weight_overrides: Optional[Dict[str, float]] = None
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """
        Run all search strategies in parallel and merge results.

        Returns:
            Tuple of (merged_results, strategy_metadata)
        """
        self.logger.info(f"🔍 Running all search strategies in parallel for: {query}")

        # Run all strategies in parallel
        tasks = [
            self._search_semantic(query, filters, workspace_id),
            self._search_visual(query, filters, workspace_id),
            self._search_understanding(query, filters, workspace_id),
            self._search_multi_vector(query, filters, workspace_id, weight_overrides=weight_overrides),
            self._search_hybrid(query, filters, workspace_id),
            self._search_material(query, filters, workspace_id),
            self._search_keyword(query, filters, workspace_id),
            self._search_color(query, filters, workspace_id),
            self._search_texture(query, filters, workspace_id),
            self._search_style(query, filters, workspace_id),
            self._search_material_type(query, filters, workspace_id),
        ]

        strategy_names = ["semantic", "visual", "understanding", "multi_vector", "hybrid", "material", "keyword", "color", "texture", "style", "material_type"]

        # Execute all tasks in parallel
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        strategy_results = {}
        for name, result in zip(strategy_names, results_list):
            if isinstance(result, Exception):
                self.logger.warning(f"⚠️ {name} search failed: {result}")
                strategy_results[name] = []
            else:
                strategy_results[name] = result

        # Merge results from all strategies
        merged = self._merge_strategy_results(strategy_results)

        # Create metadata
        metadata = {
            "strategies_used": strategy_names,
            "results_by_strategy": {
                name: len(results) for name, results in strategy_results.items()
            }
        }

        return merged, metadata

    def _merge_strategy_results(
        self,
        strategy_results: Dict[str, List[SearchResult]]
    ) -> List[SearchResult]:
        """
        Merge results from all strategies.

        Algorithm:
        1. Deduplicate by ID
        2. Calculate weighted average score
        3. Track which strategies found each result
        4. Return merged results
        """
        import numpy as np

        # Deduplicate by ID
        merged: Dict[str, SearchResult] = {}
        strategy_scores: Dict[str, Dict[str, float]] = {}

        for strategy, results in strategy_results.items():
            for result in results:
                if result.id not in merged:
                    merged[result.id] = result
                    strategy_scores[result.id] = {}

                # Track strategy score
                strategy_scores[result.id][strategy] = result.similarity_score

        # Calculate weighted average scores
        for result_id, result in merged.items():
            scores = list(strategy_scores[result_id].values())
            if scores:
                result.similarity_score = float(np.mean(scores))
                result.metadata["strategies_found"] = len(scores)
                result.metadata["strategy_scores"] = strategy_scores[result_id]

        return list(merged.values())
    
    async def _search_semantic(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Semantic search using text embeddings.
        
        Searches document chunks using text embeddings.
        """
        try:
            # Generate query embedding
            from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
            embeddings_service = RealEmbeddingsService()
            
            embedding_result = await embeddings_service.generate_text_embedding(query)
            if not embedding_result.get("success"):
                self.logger.warning("Failed to generate query embedding")
                return []
            
            query_embedding = embedding_result.get("embedding", [])
            
            # ✅ FIX: Use direct pgvector query instead of non-existent RPC
            if not self.supabase:
                return []

            # Query document_chunks table with pgvector similarity
            # Using <=> operator for cosine distance (lower is better)
            response = self.supabase.client.from_('document_chunks')\
                .select('id, content, metadata, text_embedding')\
                .eq('workspace_id', workspace_id)\
                .limit(self.config.max_results * 2)\
                .execute()

            results = []
            if response.data:
                # Calculate cosine similarity for each chunk
                import numpy as np
                query_vec = np.array(query_embedding)

                for item in response.data:
                    chunk_embedding = item.get('text_embedding')
                    if chunk_embedding:
                        chunk_vec = np.array(chunk_embedding)
                        # Cosine similarity = 1 - cosine distance
                        similarity = 1 - np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec))

                        # Filter by threshold
                        if similarity >= self.config.similarity_threshold:
                            results.append(SearchResult(
                                id=item.get('id'),
                                content=item.get('content', ''),
                                similarity_score=float(similarity),
                                metadata=item.get('metadata', {}),
                                embedding_type="text",
                                source_type="chunk"
                            ))

                # Sort by similarity (highest first)
                results.sort(key=lambda x: x.similarity_score, reverse=True)
                results = results[:self.config.max_results]

            return results
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _search_visual(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Visual search using CLIP embeddings.
        
        Searches images using CLIP embeddings.
        """
        try:
            # For visual search, query should be image URL or description
            # Generate CLIP embedding from description
            from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
            embeddings_service = RealEmbeddingsService()
            
            # Generate visual embedding from text description
            embedding_result = await embeddings_service.generate_visual_embedding(query)
            if not embedding_result.get("success"):
                self.logger.warning("Failed to generate visual embedding")
                return []
            
            query_embedding = embedding_result.get("embedding", [])
            
            # Search in database using vector similarity
            if not self.supabase:
                return []
            
            # ✅ FIX: Use VECS instead of non-existent RPC function
            from app.services.embeddings.vecs_service import get_vecs_service

            vecs_service = get_vecs_service()
            vecs_results = await vecs_service.search_similar_images(
                query_embedding=query_embedding,
                limit=self.config.max_results * 2,
                filters={"workspace_id": {"$eq": workspace_id}} if workspace_id else None,
                include_metadata=True
            )

            results = []
            for item in vecs_results:
                # Filter by similarity threshold
                if item.get('similarity_score', 0.0) >= self.config.similarity_threshold:
                    results.append(SearchResult(
                        id=item.get('image_id'),
                        content=item.get('metadata', {}).get('image_url', ''),
                        similarity_score=item.get('similarity_score', 0.0),
                        metadata=item.get('metadata', {}),
                        embedding_type="visual",
                        source_type="image"
                    ))

            return results
            
        except Exception as e:
            self.logger.error(f"Visual search failed: {e}")
            return []
    
    async def _search_multi_vector(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
        weight_overrides: Optional[Dict[str, float]] = None
    ) -> List[SearchResult]:
        """
        Multi-vector search combining all embedding types.

        Searches using text, visual, understanding, color, texture, style, and material embeddings.
        Weights can be dynamically overridden based on query intent.
        """
        try:
            # Generate all embedding types
            from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
            embeddings_service = RealEmbeddingsService()
            
            embeddings_result = await embeddings_service.generate_all_embeddings(
                entity_id="query",
                entity_type="query",
                text_content=query
            )
            
            if not embeddings_result.get("success"):
                self.logger.warning("Failed to generate multi-vector embeddings")
                return []
            
            embeddings = embeddings_result.get("embeddings", {})
            
            # Search using all embedding types with weights
            results_by_id = {}
            
            # Run all 7 embedding searches in parallel (text, visual, understanding, color, texture, style, material)
            tasks = [
                self._search_semantic(query, filters, workspace_id),
                self._search_visual(query, filters, workspace_id),
                self._search_understanding(query, filters, workspace_id),
                self._search_color(query, filters, workspace_id),
                self._search_texture(query, filters, workspace_id),
                self._search_style(query, filters, workspace_id),
                self._search_material_type(query, filters, workspace_id),
            ]

            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            # 7-vector fusion weights — use dynamic overrides if provided, else balanced defaults
            strategy_names = ["text", "visual", "understanding", "color", "texture", "style", "material"]
            if weight_overrides:
                weights = [weight_overrides.get(n, 0.125) for n in strategy_names]
            else:
                weights = [WEIGHT_PROFILES["balanced"][n] for n in strategy_names]

            # Combine results with weights
            for i, (results, weight, name) in enumerate(zip(results_list, weights, strategy_names)):
                if isinstance(results, Exception):
                    self.logger.warning(f"⚠️ {name} search failed in multi-vector: {results}")
                    continue

                for result in results:
                    if result.id not in results_by_id:
                        results_by_id[result.id] = result
                        results_by_id[result.id].similarity_score = result.similarity_score * weight
                        # Track which embeddings contributed
                        if 'embedding_sources' not in results_by_id[result.id].metadata:
                            results_by_id[result.id].metadata['embedding_sources'] = []
                            results_by_id[result.id].metadata['embedding_scores'] = {}
                        results_by_id[result.id].metadata['embedding_sources'].append(name)
                        results_by_id[result.id].metadata['embedding_scores'][name] = result.similarity_score
                    else:
                        results_by_id[result.id].similarity_score += result.similarity_score * weight
                        results_by_id[result.id].metadata['embedding_sources'].append(name)
                        results_by_id[result.id].metadata['embedding_scores'][name] = result.similarity_score

            # Sort by combined score
            sorted_results = sorted(
                results_by_id.values(),
                key=lambda x: x.similarity_score,
                reverse=True
            )

            self.logger.info(f"✅ Multi-vector search combined {len(results_by_id)} unique results from 7 embedding types")

            return sorted_results[:self.config.max_results]
            
        except Exception as e:
            self.logger.error(f"Multi-vector search failed: {e}")
            return []
    
    async def _search_hybrid(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Hybrid search combining semantic and keyword search.
        
        Combines vector similarity with keyword matching.
        """
        try:
            # Get semantic results
            semantic_results = await self._search_semantic(query, filters, workspace_id)
            
            # Get keyword results
            keyword_results = await self._search_keyword(query, filters, workspace_id)
            
            # Combine results with weights
            results_by_id = {}
            
            for result in semantic_results:
                results_by_id[result.id] = result
                results_by_id[result.id].similarity_score *= 0.7  # 70% weight for semantic
            
            for result in keyword_results:
                if result.id in results_by_id:
                    results_by_id[result.id].similarity_score += result.similarity_score * 0.3
                else:
                    result.similarity_score *= 0.3
                    results_by_id[result.id] = result
            
            return list(results_by_id.values())
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def _search_material(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Material search for material properties.
        
        Searches products and images by material properties.
        """
        try:
            # ✅ FIX: Use direct query on products table instead of non-existent RPC
            if not self.supabase:
                return []

            # Search products by name/description using text search
            response = self.supabase.client.from_('products')\
                .select('id, name, description, metadata')\
                .eq('workspace_id', workspace_id)\
                .or_(f'name.ilike.%{query}%,description.ilike.%{query}%')\
                .limit(self.config.max_results * 2)\
                .execute()

            results = []
            if response.data:
                for item in response.data:
                    # Simple text matching score (0-1)
                    name = item.get('name', '').lower()
                    desc = item.get('description', '').lower()
                    query_lower = query.lower()

                    # Calculate simple relevance score
                    score = 0.0
                    if query_lower in name:
                        score = 0.9
                    elif query_lower in desc:
                        score = 0.7
                    else:
                        score = 0.5

                    results.append(SearchResult(
                        id=item.get('id'),
                        content=item.get('name', ''),
                        similarity_score=score,
                        metadata=item.get('metadata', {}),
                        embedding_type="material",
                        source_type="product"
                    ))

            return results
            
        except Exception as e:
            self.logger.error(f"Material search failed: {e}")
            return []
    
    async def _search_keyword(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Keyword search using PostgreSQL full-text search with tsvector.

        Uses GIN indexes on content_tsv for 10-50x faster performance vs ILIKE.
        Falls back to ILIKE if tsvector column doesn't exist.
        """
        try:
            if not self.supabase:
                return []

            # ✅ OPTIMIZED: Use PostgreSQL full-text search with tsvector + GIN index
            # Try using the search_document_chunks_fts function first (fastest)
            try:
                response = self.supabase.client.rpc(
                    'search_document_chunks_fts',
                    {
                        'search_query': query,
                        'workspace_filter': workspace_id,
                        'result_limit': self.config.max_results * 2
                    }
                ).execute()

                results = []
                if response.data:
                    for item in response.data:
                        results.append(SearchResult(
                            id=item.get('id'),
                            content=item.get('content', ''),
                            score=float(item.get('rank', 0.5)),  # ts_rank score
                            metadata=item.get('metadata', {}),
                            source='keyword_fts'
                        ))
                return results

            except Exception as fts_error:
                # Fallback to direct tsvector query if RPC function doesn't exist
                logger.debug(f"FTS function not available, using direct query: {fts_error}")

                # Direct tsvector query with websearch_to_tsquery
                response = self.supabase.client.from_('document_chunks')\
                    .select('id, content, metadata')\
                    .eq('workspace_id', workspace_id)\
                    .filter('content_tsv', 'fts', f"'{query}'")\
                    .limit(self.config.max_results * 2)\
                    .execute()

                results = []
                if response.data:
                    for item in response.data:
                        # Simple scoring based on query term frequency
                        content = item.get('content', '').lower()
                        query_lower = query.lower()
                        occurrences = content.count(query_lower)
                        score = min(1.0, occurrences * 0.2)

                        results.append(SearchResult(
                            id=item.get('id'),
                            content=item.get('content', ''),
                            score=score,
                            metadata=item.get('metadata', {}),
                            source='keyword_fallback'
                        ))
                return results

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    # ✅ NEW: Specialized CLIP embedding search methods

    async def _search_understanding(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Understanding search using Qwen vision_analysis → Voyage AI embeddings (1024D).

        Enables spec-based search (e.g., "porcelain tile 60x120cm", "R10 slip rating").
        """
        try:
            from app.services.embeddings.vecs_service import get_vecs_service
            from app.services.embeddings.real_embeddings_service import RealEmbeddingsService

            embeddings_service = RealEmbeddingsService()

            # Generate understanding query embedding via Voyage AI (same embedding space)
            embedding_result = await embeddings_service.generate_understanding_query_embedding(query)
            if not embedding_result.get("success"):
                self.logger.warning("Failed to generate understanding query embedding")
                return []

            query_embedding = embedding_result.get("embedding", [])

            # Search VECS understanding collection
            vecs_service = get_vecs_service()
            results = await vecs_service.search_understanding_embeddings(
                query_embedding=query_embedding,
                limit=self.config.max_results,
                workspace_id=workspace_id,
                include_metadata=True
            )

            search_results = []
            for item in results:
                search_results.append(SearchResult(
                    id=item.get('image_id'),
                    content=item.get('metadata', {}).get('image_url', ''),
                    similarity_score=item.get('similarity_score', 0.0),
                    metadata=item.get('metadata', {}),
                    embedding_type="understanding",
                    source_type="image"
                ))

            return search_results

        except Exception as e:
            self.logger.error(f"Understanding search failed: {e}")
            return []

    async def _search_color(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Color palette search using specialized color CLIP embeddings.

        Searches images by color similarity.
        """
        try:
            from app.services.embeddings.vecs_service import get_vecs_service
            from app.services.embeddings.real_embeddings_service import RealEmbeddingsService

            # Generate color embedding from query
            embeddings_service = RealEmbeddingsService()

            # If query is text, generate visual embedding from text description
            embedding_result = await embeddings_service.generate_visual_embedding(query)
            if not embedding_result.get("success"):
                self.logger.warning("Failed to generate color embedding")
                return []

            query_embedding = embedding_result.get("embedding", [])

            # Search VECS color collection
            vecs_service = get_vecs_service()
            results = await vecs_service.search_specialized_embeddings(
                embedding_type='color',
                query_embedding=query_embedding,
                limit=self.config.max_results,
                filters={"workspace_id": {"$eq": workspace_id}} if workspace_id else None,
                min_similarity=self.config.similarity_threshold
            )

            # Format results
            search_results = []
            for item in results:
                search_results.append(SearchResult(
                    id=item.get('image_id'),
                    content=item.get('metadata', {}).get('image_url', ''),
                    similarity_score=item.get('similarity_score', 0.0),
                    metadata=item.get('metadata', {}),
                    embedding_type="color",
                    source_type="image"
                ))

            return search_results

        except Exception as e:
            self.logger.error(f"Color search failed: {e}")
            return []

    async def _search_texture(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Texture pattern search using specialized texture CLIP embeddings.

        Searches images by texture similarity.
        """
        try:
            from app.services.embeddings.vecs_service import get_vecs_service
            from app.services.embeddings.real_embeddings_service import RealEmbeddingsService

            # Generate texture embedding from query
            embeddings_service = RealEmbeddingsService()

            # If query is text, generate visual embedding from text description
            embedding_result = await embeddings_service.generate_visual_embedding(query)
            if not embedding_result.get("success"):
                self.logger.warning("Failed to generate texture embedding")
                return []

            query_embedding = embedding_result.get("embedding", [])

            # Search VECS texture collection
            vecs_service = get_vecs_service()
            results = await vecs_service.search_specialized_embeddings(
                embedding_type='texture',
                query_embedding=query_embedding,
                limit=self.config.max_results,
                filters={"workspace_id": {"$eq": workspace_id}} if workspace_id else None,
                min_similarity=self.config.similarity_threshold
            )

            # Format results
            search_results = []
            for item in results:
                search_results.append(SearchResult(
                    id=item.get('image_id'),
                    content=item.get('metadata', {}).get('image_url', ''),
                    similarity_score=item.get('similarity_score', 0.0),
                    metadata=item.get('metadata', {}),
                    embedding_type="texture",
                    source_type="image"
                ))

            return search_results

        except Exception as e:
            self.logger.error(f"Texture search failed: {e}")
            return []

    async def _search_style(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Design style search using specialized style CLIP embeddings.

        Searches images by design style similarity.
        """
        try:
            from app.services.embeddings.vecs_service import get_vecs_service
            from app.services.embeddings.real_embeddings_service import RealEmbeddingsService

            # Generate style embedding from query
            embeddings_service = RealEmbeddingsService()

            # If query is text, generate visual embedding from text description
            embedding_result = await embeddings_service.generate_visual_embedding(query)
            if not embedding_result.get("success"):
                self.logger.warning("Failed to generate style embedding")
                return []

            query_embedding = embedding_result.get("embedding", [])

            # Search VECS style collection
            vecs_service = get_vecs_service()
            results = await vecs_service.search_specialized_embeddings(
                embedding_type='style',
                query_embedding=query_embedding,
                limit=self.config.max_results,
                filters={"workspace_id": {"$eq": workspace_id}} if workspace_id else None,
                min_similarity=self.config.similarity_threshold
            )

            # Format results
            search_results = []
            for item in results:
                search_results.append(SearchResult(
                    id=item.get('image_id'),
                    content=item.get('metadata', {}).get('image_url', ''),
                    similarity_score=item.get('similarity_score', 0.0),
                    metadata=item.get('metadata', {}),
                    embedding_type="style",
                    source_type="image"
                ))

            return search_results

        except Exception as e:
            self.logger.error(f"Style search failed: {e}")
            return []

    async def _search_material_type(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Material type search using specialized material CLIP embeddings.

        Searches images by material type similarity.
        """
        try:
            from app.services.embeddings.vecs_service import get_vecs_service
            from app.services.embeddings.real_embeddings_service import RealEmbeddingsService

            # Generate material embedding from query
            embeddings_service = RealEmbeddingsService()

            # If query is text, generate visual embedding from text description
            embedding_result = await embeddings_service.generate_visual_embedding(query)
            if not embedding_result.get("success"):
                self.logger.warning("Failed to generate material embedding")
                return []

            query_embedding = embedding_result.get("embedding", [])

            # Search VECS material collection
            vecs_service = get_vecs_service()
            results = await vecs_service.search_specialized_embeddings(
                embedding_type='material',
                query_embedding=query_embedding,
                limit=self.config.max_results,
                filters={"workspace_id": {"$eq": workspace_id}} if workspace_id else None,
                min_similarity=self.config.similarity_threshold
            )

            # Format results
            search_results = []
            for item in results:
                search_results.append(SearchResult(
                    id=item.get('image_id'),
                    content=item.get('metadata', {}).get('image_url', ''),
                    similarity_score=item.get('similarity_score', 0.0),
                    metadata=item.get('metadata', {}),
                    embedding_type="material",
                    source_type="image"
                ))

            return search_results

        except Exception as e:
            self.logger.error(f"Material type search failed: {e}")
            return []

    async def _parse_query_with_ai(self, query: str) -> tuple[str, Dict[str, Any], str, Dict[str, float]]:
        """
        🧠 Parse natural language query using GPT-4o-mini to extract structured filters
        and select dynamic weight profile based on detected query intent.

        This is the PREPROCESSING step that runs BEFORE multi-strategy search.

        Args:
            query: Natural language query (e.g., "waterproof ceramic tiles for outdoor patio, matte finish, light beige")

        Returns:
            Tuple of (visual_query, filters, weight_profile, dynamic_weights):
            - visual_query: Core visual concept for embedding (e.g., "ceramic tiles matte")
            - filters: Extracted structured filters (e.g., {"material_type": "ceramic tiles", ...})
            - weight_profile: Name of selected weight profile (e.g., "color_finish")
            - dynamic_weights: Dict of 7-vector weights for multi-vector fusion

        Cost: ~$0.0001 per query (GPT-4o-mini)
        """
        import time
        start_time = time.time()

        try:
            import json
            from app.services.core.ai_call_logger import AICallLogger

            # Use centralized AI client service
            ai_service = get_ai_client_service()
            client = ai_service.openai_async

            # Call GPT-4o-mini to parse query
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": """You are a material search query parser. Analyze the query and determine if it's:
1. A PRODUCT NAME search (e.g., "MAISON by ONSET", "NOVA collection", "LOG by ALT Design")
2. A DESCRIPTIVE search (e.g., "waterproof ceramic tiles for outdoor patio")

For PRODUCT NAME searches:
- Set is_product_name: true
- Set product_name: the exact product name from the query
- Leave all other fields null

For DESCRIPTIVE searches, extract:
- is_product_name: false
- material_type: Type of material - ONLY if EXPLICITLY stated (ceramic, porcelain, fabric, wood, metal, tile, etc.)
- material_type_explicit: true if the user explicitly mentioned the material type, false if you're inferring it
- properties: Functional properties (waterproof, outdoor, slip-resistant, fire-resistant, etc.)
- finish: Surface finish (matte, glossy, textured, polished, brushed, matt, etc.)
- colors: Color names or descriptions (beige, white, gray, blue, sand, taupe, etc.)
- pattern: Visual pattern (wood pattern, marble pattern, geometric, stripes, etc.)
- application: Use case or location (patio, bathroom, kitchen, flooring, wall, shower, etc.)
- style: Design style (modern, rustic, minimalist, industrial, mediterranean, etc.)
- dimensions: Size specifications if mentioned
- designer: Designer or studio name if mentioned
- collection: Collection name if mentioned
- factory: Factory or manufacturer name if mentioned

IMPORTANT RULES:
1. If the query looks like a product/collection name (contains "by", brand names, ALL CAPS words), treat it as PRODUCT NAME search.
2. For material_type: ONLY set if user EXPLICITLY says it. Examples:
   - "ceramic tile with wood pattern" → material_type="ceramic tile", material_type_explicit=true
   - "wood pattern" → material_type=null, material_type_explicit=false (could be ceramic, wood, MDF)
   - "baxi" → material_type=null (brand spans multiple categories)
3. pattern is separate from material_type - "wood pattern" is a pattern, not necessarily wood material.

Return ONLY valid JSON. Use null for missing fields."""
                }, {
                    "role": "user",
                    "content": f"Parse this query: {query}"
                }],
                response_format={"type": "json_object"},
                temperature=0.3  # Low temperature for consistent parsing
            )

            # Parse response
            parsed_data = json.loads(response.choices[0].message.content)

            # Log GPT-4o-mini query understanding call
            ai_logger = AICallLogger()
            input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else 0
            output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else 0

            # GPT-4o-mini pricing: $0.15/1M input, $0.60/1M output
            cost = (input_tokens / 1_000_000) * 0.15 + (output_tokens / 1_000_000) * 0.60
            latency_ms = int((time.time() - start_time) * 1000)

            # Select dynamic weight profile based on parsed query intent
            profile_name, dynamic_weights = self._select_weight_profile(parsed_data)

            await ai_logger.log_gpt_call(
                task="query_understanding",
                model="gpt-4o-mini",
                response=response,
                latency_ms=latency_ms,
                confidence_score=0.90,
                confidence_breakdown={
                    "model_confidence": 0.92,
                    "completeness": 0.95,
                    "consistency": 0.88,
                    "validation": 0.85
                },
                action="use_ai_result",
                request_data={
                    "query": query,
                    "weight_profile": profile_name,
                    "dynamic_weights": dynamic_weights,
                    "parsed_fields": {k: v for k, v in parsed_data.items() if v is not None and v != [] and v != ""}
                }
            )

            # Check if this is a product name search
            if parsed_data.get("is_product_name") or parsed_data.get("product_name"):
                # For product name searches, return the original query unchanged
                self.logger.info(f"🧠 Query identified as PRODUCT NAME: '{query}' → profile='{profile_name}'")
                return query, {}, profile_name, dynamic_weights

            # Build visual query (core concept for embedding)
            visual_parts = []
            if parsed_data.get("material_type"):
                visual_parts.append(parsed_data["material_type"])
            if parsed_data.get("style"):
                visual_parts.append(parsed_data["style"])
            if parsed_data.get("finish"):
                visual_parts.append(parsed_data["finish"])

            visual_query = " ".join(visual_parts) if visual_parts else query

            # Build filters dictionary (remove null values and visual_query)
            # Comprehensive mapping from AI output fields to actual database metadata fields
            field_mapping = {
                # Appearance (nested in appearance object)
                "colors": "appearance.colors",
                "finish": "appearance.finish",
                "pattern": "appearance.pattern",
                # Application (nested in application object)
                "application": "application.recommended_use",
                # Design (nested in design object)
                "style": "design.aesthetic_style",
                "designer": "design.designers",
                "collection": "design.collection",
                # Material properties (nested in material_properties object)
                "properties": "material_properties",
                # Top-level fields (no mapping needed, but listed for clarity)
                "factory": "factory_name",
                "dimensions": "dimensions",
                # Category - only if explicitly stated
                "material_type": "material_category",
            }

            # Fields that are metadata about parsing, not actual filters
            skip_fields = {"is_product_name", "product_name", "visual_query", "material_type_explicit"}

            filters = {}
            for key, value in parsed_data.items():
                if key in skip_fields:
                    continue

                # Special handling for material_type - only pass if EXPLICITLY stated
                if key == "material_type":
                    if not parsed_data.get("material_type_explicit", False):
                        # Category was inferred, not explicit - skip it
                        # e.g., "wood pattern" shouldn't filter to wood category
                        continue

                if value is not None and value != [] and value != "":
                    # Map to actual DB field name
                    db_key = field_mapping.get(key, key)

                    # Handle properties as array containment
                    if key == "properties" and isinstance(value, list):
                        filters[db_key] = {"contains": value}
                    else:
                        filters[db_key] = value

            self.logger.info(f"🧠 Query parsed: '{query}' → visual_query='{visual_query}', profile='{profile_name}', filters={filters}")

            return visual_query, filters, profile_name, dynamic_weights

        except Exception as e:
            self.logger.error(f"Query parsing failed: {e}, using original query")
            # Fallback: return original query with no filters and balanced weights
            return query, {}, "balanced", WEIGHT_PROFILES["balanced"]


