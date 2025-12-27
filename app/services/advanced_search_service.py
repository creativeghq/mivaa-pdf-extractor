"""
Advanced Search Service for Enhanced RAG Capabilities

This service implements Phase 7 advanced features including:
- Semantic search with MMR (Maximal Marginal Relevance)
- Query optimization and result ranking algorithms
- Advanced filtering and metadata-based search
- Query expansion and synonym handling
- Complex query types and operators
- Performance optimization and relevance scoring
"""

import logging
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    ADVANCED_SEARCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced search dependencies not available: {e}")
    ADVANCED_SEARCH_AVAILABLE = False


class QueryType(Enum):
    """Enumeration of supported query types."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"
    SUMMARIZATION = "summarization"
    BOOLEAN = "boolean"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"


class SearchOperator(Enum):
    """Enumeration of search operators."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    NEAR = "NEAR"
    PHRASE = "PHRASE"


@dataclass
class QueryExpansion:
    """Data class for query expansion results."""
    original_query: str
    expanded_terms: List[str]
    synonyms: Dict[str, List[str]]
    related_concepts: List[str]
    confidence_score: float


@dataclass
class SearchFilter:
    """Data class for search filters."""
    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, contains
    value: Any
    boost: float = 1.0


@dataclass
class MMRResult:
    """Data class for MMR algorithm results."""
    selected_nodes: List[Any]
    diversity_scores: List[float]
    relevance_scores: List[float]
    final_scores: List[float]


class AdvancedSearchService:
    """
    Advanced search service implementing sophisticated RAG capabilities.
    
    Features:
    - MMR (Maximal Marginal Relevance) for diverse result selection
    - Query optimization and expansion
    - Advanced metadata filtering
    - Complex query parsing and execution
    - Performance-optimized retrieval
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced search service."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Service availability
        self.available = ADVANCED_SEARCH_AVAILABLE
        if not self.available:
            self.logger.warning("Advanced search service unavailable - dependencies not installed")
            return
        
        # Configuration parameters
        self.mmr_lambda = self.config.get('mmr_lambda', 0.7)  # Balance between relevance and diversity
        self.max_query_expansion_terms = self.config.get('max_query_expansion_terms', 10)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.enable_query_expansion = self.config.get('enable_query_expansion', True)
        self.enable_synonym_expansion = self.config.get('enable_synonym_expansion', True)
        
        # Initialize NLP components
        self._initialize_nlp_components()
        
        # Cache for query expansions and embeddings
        self.query_expansion_cache = {}
        self.embedding_cache = {}
        
        self.logger.info("Advanced search service initialized successfully")
    
    def _initialize_nlp_components(self):
        """Initialize NLP components for query processing."""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            # Initialize stopwords
            self.stop_words = set(stopwords.words('english'))
            
            # Initialize TF-IDF vectorizer for keyword extraction
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            self.logger.info("NLP components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP components: {e}")
            self.available = False
    
    async def semantic_search_with_mmr(
        self,
        query: str,
        index: VectorStoreIndex,
        top_k: int = 10,
        mmr_top_k: int = 5,
        lambda_param: Optional[float] = None,
        metadata_filters: Optional[List[SearchFilter]] = None,
        query_type: QueryType = QueryType.SEMANTIC
    ) -> MMRResult:
        """
        Perform semantic search with MMR for diverse result selection.
        
        Args:
            query: Search query
            index: Vector store index to search
            top_k: Number of initial candidates to retrieve
            mmr_top_k: Number of final results after MMR
            lambda_param: MMR lambda parameter (relevance vs diversity balance)
            metadata_filters: Optional metadata filters
            query_type: Type of query for optimization
            
        Returns:
            MMRResult containing selected nodes and scores
        """
        if not self.available:
            raise RuntimeError("Advanced search service not available")
        
        try:
            # Step 1: Query optimization and expansion
            optimized_query = await self._optimize_query(query, query_type)
            
            # Step 2: Initial retrieval with expanded candidates
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=max(top_k, 20),  # Retrieve more candidates for MMR
                filters=self._convert_filters(metadata_filters) if metadata_filters else None
            )
            
            initial_nodes = retriever.retrieve(optimized_query)
            
            if not initial_nodes:
                return MMRResult([], [], [], [])
            
            # Step 3: Apply metadata filtering if specified
            if metadata_filters:
                initial_nodes = self._apply_metadata_filters(initial_nodes, metadata_filters)
            
            # Step 4: Apply MMR algorithm
            mmr_result = await self._apply_mmr_algorithm(
                query=optimized_query,
                nodes=initial_nodes[:top_k],
                lambda_param=lambda_param or self.mmr_lambda,
                top_k=mmr_top_k
            )
            
            self.logger.info(f"MMR search completed: {len(mmr_result.selected_nodes)} results selected from {len(initial_nodes)} candidates")
            
            return mmr_result
            
        except Exception as e:
            self.logger.error(f"Semantic search with MMR failed: {e}")
            raise
    
    async def _optimize_query(self, query: str, query_type: QueryType) -> str:
        """
        Optimize and expand the query based on type and context.
        
        Args:
            query: Original query
            query_type: Type of query for optimization
            
        Returns:
            Optimized query string
        """
        try:
            # Check cache first
            cache_key = f"{query}_{query_type.value}"
            if cache_key in self.query_expansion_cache:
                return self.query_expansion_cache[cache_key]
            
            optimized_query = query.strip()
            
            # Apply query type specific optimizations
            if query_type == QueryType.BOOLEAN:
                optimized_query = self._parse_boolean_query(optimized_query)
            elif query_type == QueryType.FUZZY:
                optimized_query = self._apply_fuzzy_matching(optimized_query)
            elif query_type == QueryType.ANALYTICAL:
                optimized_query = f"analyze and provide insights about: {optimized_query}"
            elif query_type == QueryType.SUMMARIZATION:
                optimized_query = f"summarize key information related to: {optimized_query}"
            
            # Apply query expansion if enabled
            if self.enable_query_expansion:
                expansion = await self._expand_query(optimized_query)
                if expansion.expanded_terms:
                    # Add expanded terms with lower weight
                    expanded_terms_str = " ".join(expansion.expanded_terms[:self.max_query_expansion_terms])
                    optimized_query = f"{optimized_query} {expanded_terms_str}"
            
            # Cache the result
            self.query_expansion_cache[cache_key] = optimized_query
            
            return optimized_query
            
        except Exception as e:
            self.logger.error(f"Query optimization failed: {e}")
            return query
    
    async def multi_query_expansion(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generate multiple query variations using LLM for better recall.

        This is a Phase 1 RAG optimization that generates different ways
        to ask the same question, improving retrieval recall.

        Args:
            query: Original query
            num_variations: Number of query variations to generate

        Returns:
            List of query variations including the original
        """
        try:
            from app.services.ai_client_service import AIClientService

            # Check if multi-query is enabled
            if not hasattr(self, 'enable_multi_query') or not self.enable_multi_query:
                return [query]

            # Check cache first
            cache_key = f"multi_query:{query}"
            if cache_key in self.query_expansion_cache:
                return self.query_expansion_cache[cache_key]

            # Generate variations using LLM
            ai_client = AIClientService()
            prompt = f"""Generate {num_variations} different ways to ask this question.
Each variation should preserve the original intent but use different wording.

Original question: {query}

Variations:
1."""

            response = await ai_client.generate_text(
                prompt=prompt,
                model="gpt-4o-mini",  # Use cheaper model for query expansion
                max_tokens=200,
                temperature=0.7  # Higher temperature for diversity
            )

            # Parse variations from response
            variations = [query]  # Always include original
            lines = response.strip().split('\n')
            for line in lines:
                # Extract variation from numbered list
                match = re.match(r'^\d+\.\s*(.+)$', line.strip())
                if match:
                    variation = match.group(1).strip()
                    if variation and variation != query:
                        variations.append(variation)

            # Limit to requested number + original
            variations = variations[:num_variations + 1]

            # Cache the result
            self.query_expansion_cache[cache_key] = variations

            self.logger.info(f"Generated {len(variations)} query variations for: {query}")
            return variations

        except Exception as e:
            self.logger.error(f"Multi-query expansion failed: {e}")
            return [query]  # Fallback to original query

    async def _expand_query(self, query: str) -> QueryExpansion:
        """
        Expand query with synonyms and related terms.

        Args:
            query: Original query

        Returns:
            QueryExpansion object with expanded terms
        """
        try:
            # Tokenize and clean query
            tokens = word_tokenize(query.lower())
            tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]

            expanded_terms = []
            synonyms = {}
            related_concepts = []

            # Get synonyms using WordNet
            if self.enable_synonym_expansion:
                for token in tokens:
                    token_synonyms = []
                    for syn in wordnet.synsets(token):
                        for lemma in syn.lemmas():
                            synonym = lemma.name().replace('_', ' ')
                            if synonym != token and synonym not in token_synonyms:
                                token_synonyms.append(synonym)

                    if token_synonyms:
                        synonyms[token] = token_synonyms[:3]  # Limit to top 3 synonyms
                        expanded_terms.extend(token_synonyms[:2])  # Add top 2 to expanded terms

            # Extract related concepts using simple heuristics
            # In a production system, this could use more sophisticated NLP models
            for token in tokens:
                if len(token) > 3:  # Only consider meaningful tokens
                    related_concepts.append(f"{token}s")  # Plural form
                    related_concepts.append(f"{token}ing")  # Gerund form
            
            # Calculate confidence score based on expansion quality
            confidence_score = min(len(expanded_terms) / 10.0, 1.0)
            
            return QueryExpansion(
                original_query=query,
                expanded_terms=list(set(expanded_terms)),
                synonyms=synonyms,
                related_concepts=list(set(related_concepts)),
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"Query expansion failed: {e}")
            return QueryExpansion(query, [], {}, [], 0.0)
    
    def _parse_boolean_query(self, query: str) -> str:
        """
        Parse and optimize boolean queries.
        
        Args:
            query: Boolean query string
            
        Returns:
            Optimized boolean query
        """
        try:
            # Simple boolean query parsing
            # In production, this would use a proper query parser
            
            # Replace common boolean operators
            query = re.sub(r'\bAND\b', ' AND ', query, flags=re.IGNORECASE)
            query = re.sub(r'\bOR\b', ' OR ', query, flags=re.IGNORECASE)
            query = re.sub(r'\bNOT\b', ' NOT ', query, flags=re.IGNORECASE)
            
            # Handle phrase queries
            query = re.sub(r'"([^"]*)"', r'PHRASE(\1)', query)
            
            # Clean up extra spaces
            query = re.sub(r'\s+', ' ', query).strip()
            
            return query
            
        except Exception as e:
            self.logger.error(f"Boolean query parsing failed: {e}")
            return query
    
    def _apply_fuzzy_matching(self, query: str) -> str:
        """
        Apply fuzzy matching optimizations to the query.
        
        Args:
            query: Original query
            
        Returns:
            Fuzzy-optimized query
        """
        try:
            # Add fuzzy matching indicators
            # This is a simplified implementation
            tokens = query.split()
            fuzzy_tokens = []
            
            for token in tokens:
                if len(token) > 4:  # Only apply fuzzy to longer words
                    fuzzy_tokens.append(f"{token}~")  # Elasticsearch-style fuzzy
                else:
                    fuzzy_tokens.append(token)
            
            return " ".join(fuzzy_tokens)
            
        except Exception as e:
            self.logger.error(f"Fuzzy matching application failed: {e}")
            return query
    
    async def _apply_mmr_algorithm(
        self,
        query: str,
        nodes: List[NodeWithScore],
        lambda_param: float,
        top_k: int
    ) -> MMRResult:
        """
        Apply Maximal Marginal Relevance algorithm for diverse result selection.
        
        Args:
            query: Search query
            nodes: Retrieved nodes with scores
            lambda_param: Balance between relevance and diversity (0-1)
            top_k: Number of results to select
            
        Returns:
            MMRResult with selected diverse nodes
        """
        try:
            if not nodes or top_k <= 0:
                return MMRResult([], [], [], [])
            
            # Extract embeddings and texts
            node_texts = []
            node_embeddings = []
            relevance_scores = []
            
            for node in nodes:
                node_texts.append(node.node.text)
                relevance_scores.append(getattr(node, 'score', 0.0))
                
                # Get or compute embedding for the node
                # In a real implementation, this would use the actual embeddings
                # For now, we'll use a simplified approach
                embedding = self._get_text_embedding(node.node.text)
                node_embeddings.append(embedding)
            
            # Get query embedding
            query_embedding = self._get_text_embedding(query)
            
            # Apply MMR algorithm
            selected_indices = []
            remaining_indices = list(range(len(nodes)))
            diversity_scores = []
            final_scores = []
            
            # Select first document (highest relevance)
            if remaining_indices:
                best_idx = max(remaining_indices, key=lambda i: relevance_scores[i])
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                diversity_scores.append(0.0)  # First document has no diversity penalty
                final_scores.append(relevance_scores[best_idx])
            
            # Select remaining documents using MMR
            while len(selected_indices) < top_k and remaining_indices:
                best_score = -float('inf')
                best_idx = None
                best_diversity = 0.0
                
                for idx in remaining_indices:
                    # Calculate relevance score
                    relevance = relevance_scores[idx]
                    
                    # Calculate diversity score (minimum similarity to selected documents)
                    max_similarity = 0.0
                    if selected_indices:
                        similarities = []
                        for selected_idx in selected_indices:
                            sim = self._calculate_similarity(
                                node_embeddings[idx],
                                node_embeddings[selected_idx]
                            )
                            similarities.append(sim)
                        max_similarity = max(similarities)
                    
                    diversity = 1.0 - max_similarity
                    
                    # Calculate MMR score
                    mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx
                        best_diversity = diversity
                
                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
                    diversity_scores.append(best_diversity)
                    final_scores.append(best_score)
                else:
                    break
            
            # Create result
            selected_nodes = [nodes[i] for i in selected_indices]
            selected_relevance_scores = [relevance_scores[i] for i in selected_indices]
            
            return MMRResult(
                selected_nodes=selected_nodes,
                diversity_scores=diversity_scores,
                relevance_scores=selected_relevance_scores,
                final_scores=final_scores
            )
            
        except Exception as e:
            self.logger.error(f"MMR algorithm failed: {e}")
            # Fallback to top-k by relevance
            top_nodes = sorted(nodes, key=lambda x: getattr(x, 'score', 0.0), reverse=True)[:top_k]
            return MMRResult(
                selected_nodes=top_nodes,
                diversity_scores=[0.0] * len(top_nodes),
                relevance_scores=[getattr(node, 'score', 0.0) for node in top_nodes],
                final_scores=[getattr(node, 'score', 0.0) for node in top_nodes]
            )
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get or compute embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Check cache first
            if text in self.embedding_cache:
                return self.embedding_cache[text]
            
            # For this implementation, we'll use TF-IDF as a simple embedding
            # In production, this would use the actual embedding model
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            embedding = tfidf_matrix.toarray()[0]
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Cache the result
            self.embedding_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Text embedding failed: {e}")
            # Return zero vector as fallback
            return np.zeros(100)
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        try:
            # Ensure embeddings have the same shape
            if embedding1.shape != embedding2.shape:
                min_len = min(len(embedding1), len(embedding2))
                embedding1 = embedding1[:min_len]
                embedding2 = embedding2[:min_len]
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _convert_filters(self, filters: List[SearchFilter]) -> Dict[str, Any]:
        """
        Convert SearchFilter objects to filter format.

        Args:
            filters: List of search filters

        Returns:
            Filter dictionary
        """
        try:
            converted_filters = {}

            for filter_obj in filters:
                field = filter_obj.field
                operator = filter_obj.operator
                value = filter_obj.value

                # Convert to filter format
                if operator == "eq":
                    converted_filters[field] = value
                elif operator == "in":
                    converted_filters[field] = {"$in": value}
                elif operator == "contains":
                    converted_filters[field] = {"$regex": f".*{value}.*"}
                elif operator == "gt":
                    converted_filters[field] = {"$gt": value}
                elif operator == "lt":
                    converted_filters[field] = {"$lt": value}
                elif operator == "gte":
                    converted_filters[field] = {"$gte": value}
                elif operator == "lte":
                    converted_filters[field] = {"$lte": value}
                elif operator == "ne":
                    converted_filters[field] = {"$ne": value}

            return converted_filters

        except Exception as e:
            self.logger.error(f"Filter conversion failed: {e}")
            return {}
    
    def _apply_metadata_filters(
        self,
        nodes: List[NodeWithScore],
        filters: List[SearchFilter]
    ) -> List[NodeWithScore]:
        """
        Apply metadata filters to retrieved nodes.
        
        Args:
            nodes: List of nodes to filter
            filters: List of search filters
            
        Returns:
            Filtered list of nodes
        """
        try:
            filtered_nodes = []
            
            for node in nodes:
                metadata = node.node.metadata or {}
                passes_all_filters = True
                
                for filter_obj in filters:
                    field = filter_obj.field
                    operator = filter_obj.operator
                    value = filter_obj.value
                    
                    if field not in metadata:
                        passes_all_filters = False
                        break
                    
                    field_value = metadata[field]
                    
                    # Apply filter logic
                    if operator == "eq" and field_value != value:
                        passes_all_filters = False
                        break
                    elif operator == "ne" and field_value == value:
                        passes_all_filters = False
                        break
                    elif operator == "gt" and not (field_value > value):
                        passes_all_filters = False
                        break
                    elif operator == "lt" and not (field_value < value):
                        passes_all_filters = False
                        break
                    elif operator == "gte" and not (field_value >= value):
                        passes_all_filters = False
                        break
                    elif operator == "lte" and not (field_value <= value):
                        passes_all_filters = False
                        break
                    elif operator == "in" and field_value not in value:
                        passes_all_filters = False
                        break
                    elif operator == "contains" and str(value).lower() not in str(field_value).lower():
                        passes_all_filters = False
                        break
                
                if passes_all_filters:
                    filtered_nodes.append(node)
            
            return filtered_nodes
            
        except Exception as e:
            self.logger.error(f"Metadata filtering failed: {e}")
            return nodes
    
    async def advanced_query_processing(
        self,
        query: str,
        query_type: QueryType = QueryType.SEMANTIC,
        enable_expansion: bool = True,
        enable_optimization: bool = True
    ) -> Dict[str, Any]:
        """
        Process query with advanced optimization techniques.
        
        Args:
            query: Original query
            query_type: Type of query
            enable_expansion: Whether to enable query expansion
            enable_optimization: Whether to enable query optimization
            
        Returns:
            Dictionary with processed query information
        """
        try:
            result = {
                "original_query": query,
                "query_type": query_type.value,
                "processed_query": query,
                "expansion": None,
                "optimization_applied": False,
                "processing_time_ms": 0
            }
            
            start_time = datetime.now()
            
            # Apply query optimization
            if enable_optimization:
                result["processed_query"] = await self._optimize_query(query, query_type)
                result["optimization_applied"] = True
            
            # Apply query expansion
            if enable_expansion and self.enable_query_expansion:
                expansion = await self._expand_query(query)
                result["expansion"] = {
                    "expanded_terms": expansion.expanded_terms,
                    "synonyms": expansion.synonyms,
                    "related_concepts": expansion.related_concepts,
                    "confidence_score": expansion.confidence_score
                }
            
            # Calculate processing time
            end_time = datetime.now()
            result["processing_time_ms"] = int((end_time - start_time).total_seconds() * 1000)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Advanced query processing failed: {e}")
            return {
                "original_query": query,
                "query_type": query_type.value,
                "processed_query": query,
                "expansion": None,
                "optimization_applied": False,
                "processing_time_ms": 0,
                "error": str(e)
            }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """
        Get service statistics and performance metrics.
        
        Returns:
            Dictionary with service statistics
        """
        return {
            "service_available": self.available,
            "query_expansion_cache_size": len(self.query_expansion_cache),
            "embedding_cache_size": len(self.embedding_cache),
            "configuration": {
                "mmr_lambda": self.mmr_lambda,
                "max_query_expansion_terms": self.max_query_expansion_terms,
                "similarity_threshold": self.similarity_threshold,
                "enable_query_expansion": self.enable_query_expansion,
                "enable_synonym_expansion": self.enable_synonym_expansion
            },
            "supported_query_types": [qt.value for qt in QueryType],
            "supported_operators": [op.value for op in SearchOperator]
        }
    
    def clear_caches(self) -> None:
        """Clear all internal caches."""
        self.query_expansion_cache.clear()
        self.embedding_cache.clear()
        self.logger.info("Advanced search service caches cleared")
    
    def __del__(self):
        """Cleanup resources when the service is destroyed."""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=True)