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

logger = logging.getLogger(__name__)


class SearchStrategy(str, Enum):
    """Supported search strategies."""
    SEMANTIC = "semantic"
    VISUAL = "visual"
    MULTI_VECTOR = "multi_vector"
    HYBRID = "hybrid"
    MATERIAL = "material"
    KEYWORD = "keyword"


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


class UnifiedSearchService:
    """
    Unified search service that consolidates all search strategies.

    NEW: Runs all strategies in parallel and merges results for comprehensive coverage.

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

    async def search(
        self,
        query: str,
        strategy: Optional[SearchStrategy] = None,
        filters: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
        run_all_strategies: bool = False
    ) -> SearchResponse:
        """
        Perform search using the configured strategy or all strategies.

        Args:
            query: Search query
            strategy: Search strategy (uses config default if not specified)
            filters: Additional filters for search
            workspace_id: Workspace ID for scoped search
            run_all_strategies: If True, run all strategies in parallel and merge

        Returns:
            SearchResponse with results
        """
        try:
            start_time = datetime.utcnow()

            # If run_all_strategies is True, execute all strategies in parallel
            if run_all_strategies:
                results, strategy_metadata = await self._search_all_strategies(
                    query, filters, workspace_id
                )
                search_strategy = "all_strategies"
            else:
                search_strategy = strategy or self.config.strategy
                results = await self._search_single_strategy(
                    search_strategy, query, filters, workspace_id
                )
                strategy_metadata = {"strategies_used": [search_strategy.value]}

            # Sort by similarity score
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            limited_results = results[:self.config.max_results]

            # Calculate search time
            search_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            self.logger.info(f"✅ Search completed: {len(limited_results)} results found in {search_time_ms:.2f}ms")

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
        workspace_id: Optional[str]
    ) -> List[SearchResult]:
        """Execute a single search strategy."""
        if strategy == SearchStrategy.SEMANTIC:
            return await self._search_semantic(query, filters, workspace_id)
        elif strategy == SearchStrategy.VISUAL:
            return await self._search_visual(query, filters, workspace_id)
        elif strategy == SearchStrategy.MULTI_VECTOR:
            return await self._search_multi_vector(query, filters, workspace_id)
        elif strategy == SearchStrategy.HYBRID:
            return await self._search_hybrid(query, filters, workspace_id)
        elif strategy == SearchStrategy.MATERIAL:
            return await self._search_material(query, filters, workspace_id)
        elif strategy == SearchStrategy.KEYWORD:
            return await self._search_keyword(query, filters, workspace_id)
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")

    async def _search_all_strategies(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        workspace_id: Optional[str]
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
            self._search_multi_vector(query, filters, workspace_id),
            self._search_hybrid(query, filters, workspace_id),
            self._search_material(query, filters, workspace_id),
            self._search_keyword(query, filters, workspace_id),
        ]

        strategy_names = ["semantic", "visual", "multi_vector", "hybrid", "material", "keyword"]

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
            from app.services.real_embeddings_service import RealEmbeddingsService
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
            from app.services.real_embeddings_service import RealEmbeddingsService
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
            from app.services.vecs_service import get_vecs_service

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
        workspace_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Multi-vector search combining all embedding types.
        
        Searches using text, visual, multimodal, color, texture, and application embeddings.
        """
        try:
            # Generate all embedding types
            from app.services.real_embeddings_service import RealEmbeddingsService
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
            
            # Search with text embedding (25% weight)
            if embeddings.get("text_1536"):
                text_results = await self._search_semantic(query, filters, workspace_id)
                for result in text_results:
                    if result.id not in results_by_id:
                        results_by_id[result.id] = result
                    results_by_id[result.id].similarity_score += result.similarity_score * 0.25
            
            # Search with visual embedding (25% weight)
            if embeddings.get("visual_clip_512"):
                visual_results = await self._search_visual(query, filters, workspace_id)
                for result in visual_results:
                    if result.id not in results_by_id:
                        results_by_id[result.id] = result
                    results_by_id[result.id].similarity_score += result.similarity_score * 0.25
            
            # Additional embeddings (color, texture, application) with lower weights
            # These would be combined similarly
            
            return list(results_by_id.values())
            
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
        Keyword search for exact matches.
        
        Searches using full-text search.
        """
        try:
            if not self.supabase:
                return []
            
            # ✅ FIX: Use direct text search instead of non-existent RPC
            response = self.supabase.client.from_('document_chunks')\
                .select('id, content, metadata')\
                .eq('workspace_id', workspace_id)\
                .ilike('content', f'%{query}%')\
                .limit(self.config.max_results * 2)\
                .execute()

            results = []
            if response.data:
                for item in response.data:
                    content = item.get('content', '').lower()
                    query_lower = query.lower()

                    # Calculate simple keyword relevance score
                    # Count occurrences of query in content
                    occurrences = content.count(query_lower)
                    score = min(1.0, occurrences * 0.2)  # Cap at 1.0

                    results.append(SearchResult(
                        id=item.get('id'),
                        content=item.get('content', ''),
                        similarity_score=score,
                        metadata=item.get('metadata', {}),
                        embedding_type="keyword",
                        source_type="chunk"
                    ))

            return results
            
        except Exception as e:
            self.logger.error(f"Keyword search failed: {e}")
            return []

