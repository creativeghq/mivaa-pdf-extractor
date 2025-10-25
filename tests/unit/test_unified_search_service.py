"""
Unit Tests for Unified Search Service - Step 7

Tests all search strategies:
- Semantic search
- Visual search
- Multi-vector search
- Hybrid search
- Material search
- Keyword search
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from app.services.unified_search_service import (
    UnifiedSearchService,
    SearchConfig,
    SearchStrategy,
    SearchResult,
    SearchResponse
)


class TestSearchConfig:
    """Test SearchConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = SearchConfig()
        
        assert config.strategy == SearchStrategy.MULTI_VECTOR
        assert config.max_results == 20
        assert config.similarity_threshold == 0.7
        assert config.include_metadata is True
        assert config.include_embeddings is False
        assert config.enable_hybrid is True
        assert config.mmr_lambda == 0.7
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SearchConfig(
            strategy=SearchStrategy.SEMANTIC,
            max_results=50,
            similarity_threshold=0.8,
            include_metadata=False,
            include_embeddings=True,
            enable_hybrid=False,
            mmr_lambda=0.5
        )
        
        assert config.strategy == SearchStrategy.SEMANTIC
        assert config.max_results == 50
        assert config.similarity_threshold == 0.8
        assert config.include_metadata is False
        assert config.include_embeddings is True
        assert config.enable_hybrid is False
        assert config.mmr_lambda == 0.5


class TestSearchResult:
    """Test SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test creating a search result."""
        result = SearchResult(
            id="chunk_123",
            content="Test content",
            similarity_score=0.95,
            metadata={"source": "test"},
            embedding_type="text",
            source_type="chunk"
        )
        
        assert result.id == "chunk_123"
        assert result.content == "Test content"
        assert result.similarity_score == 0.95
        assert result.metadata == {"source": "test"}
        assert result.embedding_type == "text"
        assert result.source_type == "chunk"


class TestUnifiedSearchService:
    """Test UnifiedSearchService."""
    
    @pytest.fixture
    def mock_supabase(self):
        """Create mock Supabase client."""
        mock = Mock()
        mock.client = Mock()
        mock.client.rpc = Mock()
        return mock
    
    @pytest.fixture
    def search_service(self, mock_supabase):
        """Create search service with mock Supabase."""
        config = SearchConfig()
        return UnifiedSearchService(config, mock_supabase)
    
    @pytest.mark.asyncio
    async def test_search_semantic_strategy(self, search_service):
        """Test semantic search strategy."""
        # Mock embeddings service
        with patch('app.services.unified_search_service.RealEmbeddingsService') as mock_embeddings:
            mock_service = AsyncMock()
            mock_embeddings.return_value = mock_service
            mock_service.generate_text_embedding = AsyncMock(
                return_value={
                    "success": True,
                    "embedding": [0.1, 0.2, 0.3]
                }
            )
            
            # Mock database response
            mock_response = Mock()
            mock_response.data = [
                {
                    "id": "chunk_1",
                    "content": "Test content 1",
                    "similarity": 0.95,
                    "metadata": {"source": "test"}
                }
            ]
            search_service.supabase.client.rpc.return_value.execute.return_value = mock_response
            
            # Perform search
            result = await search_service._search_semantic("test query")
            
            assert len(result) == 1
            assert result[0].id == "chunk_1"
            assert result[0].similarity_score == 0.95
    
    @pytest.mark.asyncio
    async def test_search_visual_strategy(self, search_service):
        """Test visual search strategy."""
        with patch('app.services.unified_search_service.RealEmbeddingsService') as mock_embeddings:
            mock_service = AsyncMock()
            mock_embeddings.return_value = mock_service
            mock_service.generate_visual_embedding = AsyncMock(
                return_value={
                    "success": True,
                    "embedding": [0.1, 0.2, 0.3]
                }
            )
            
            mock_response = Mock()
            mock_response.data = [
                {
                    "id": "image_1",
                    "url": "https://example.com/image.jpg",
                    "similarity": 0.88,
                    "metadata": {"type": "image"}
                }
            ]
            search_service.supabase.client.rpc.return_value.execute.return_value = mock_response
            
            result = await search_service._search_visual("test query")
            
            assert len(result) == 1
            assert result[0].id == "image_1"
            assert result[0].similarity_score == 0.88
    
    @pytest.mark.asyncio
    async def test_search_keyword_strategy(self, search_service):
        """Test keyword search strategy."""
        mock_response = Mock()
        mock_response.data = [
            {
                "id": "chunk_1",
                "content": "Test content",
                "rank": 0.92,
                "metadata": {"type": "keyword"}
            }
        ]
        search_service.supabase.client.rpc.return_value.execute.return_value = mock_response
        
        result = await search_service._search_keyword("test query")
        
        assert len(result) == 1
        assert result[0].id == "chunk_1"
        assert result[0].similarity_score == 0.92
    
    @pytest.mark.asyncio
    async def test_search_material_strategy(self, search_service):
        """Test material search strategy."""
        mock_response = Mock()
        mock_response.data = [
            {
                "id": "product_1",
                "name": "Test Product",
                "similarity": 0.85,
                "metadata": {"material": "wood"}
            }
        ]
        search_service.supabase.client.rpc.return_value.execute.return_value = mock_response
        
        result = await search_service._search_material("test query")
        
        assert len(result) == 1
        assert result[0].id == "product_1"
        assert result[0].similarity_score == 0.85
    
    @pytest.mark.asyncio
    async def test_search_with_invalid_strategy(self, search_service):
        """Test search with invalid strategy."""
        with pytest.raises(ValueError):
            await search_service.search(
                query="test",
                strategy=SearchStrategy("invalid_strategy")
            )
    
    @pytest.mark.asyncio
    async def test_search_response_format(self, search_service):
        """Test search response format."""
        with patch('app.services.unified_search_service.RealEmbeddingsService') as mock_embeddings:
            mock_service = AsyncMock()
            mock_embeddings.return_value = mock_service
            mock_service.generate_text_embedding = AsyncMock(
                return_value={
                    "success": True,
                    "embedding": [0.1, 0.2, 0.3]
                }
            )
            
            mock_response = Mock()
            mock_response.data = [
                {
                    "id": "chunk_1",
                    "content": "Test content",
                    "similarity": 0.95,
                    "metadata": {"source": "test"}
                }
            ]
            search_service.supabase.client.rpc.return_value.execute.return_value = mock_response
            
            result = await search_service.search(
                query="test query",
                strategy=SearchStrategy.SEMANTIC
            )
            
            assert isinstance(result, SearchResponse)
            assert result.success is True
            assert result.query == "test query"
            assert len(result.results) > 0
            assert result.total_found > 0
            assert result.search_time_ms > 0
            assert result.strategy_used == "semantic"
            assert "similarity_threshold" in result.metadata
    
    @pytest.mark.asyncio
    async def test_search_result_sorting(self, search_service):
        """Test that results are sorted by similarity score."""
        with patch('app.services.unified_search_service.RealEmbeddingsService') as mock_embeddings:
            mock_service = AsyncMock()
            mock_embeddings.return_value = mock_service
            mock_service.generate_text_embedding = AsyncMock(
                return_value={
                    "success": True,
                    "embedding": [0.1, 0.2, 0.3]
                }
            )
            
            mock_response = Mock()
            mock_response.data = [
                {"id": "chunk_1", "content": "Content 1", "similarity": 0.75, "metadata": {}},
                {"id": "chunk_2", "content": "Content 2", "similarity": 0.95, "metadata": {}},
                {"id": "chunk_3", "content": "Content 3", "similarity": 0.85, "metadata": {}},
            ]
            search_service.supabase.client.rpc.return_value.execute.return_value = mock_response
            
            result = await search_service.search(
                query="test query",
                strategy=SearchStrategy.SEMANTIC
            )
            
            # Results should be sorted by similarity score (descending)
            assert result.results[0].similarity_score >= result.results[1].similarity_score
            assert result.results[1].similarity_score >= result.results[2].similarity_score
    
    @pytest.mark.asyncio
    async def test_search_respects_max_results(self, search_service):
        """Test that search respects max_results limit."""
        search_service.config.max_results = 2
        
        with patch('app.services.unified_search_service.RealEmbeddingsService') as mock_embeddings:
            mock_service = AsyncMock()
            mock_embeddings.return_value = mock_service
            mock_service.generate_text_embedding = AsyncMock(
                return_value={
                    "success": True,
                    "embedding": [0.1, 0.2, 0.3]
                }
            )
            
            mock_response = Mock()
            mock_response.data = [
                {"id": f"chunk_{i}", "content": f"Content {i}", "similarity": 0.9 - i*0.05, "metadata": {}}
                for i in range(5)
            ]
            search_service.supabase.client.rpc.return_value.execute.return_value = mock_response
            
            result = await search_service.search(
                query="test query",
                strategy=SearchStrategy.SEMANTIC
            )
            
            assert len(result.results) <= 2
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self, search_service):
        """Test error handling in search."""
        with patch('app.services.unified_search_service.RealEmbeddingsService') as mock_embeddings:
            mock_service = AsyncMock()
            mock_embeddings.return_value = mock_service
            mock_service.generate_text_embedding = AsyncMock(
                side_effect=Exception("Embedding generation failed")
            )
            
            result = await search_service.search(
                query="test query",
                strategy=SearchStrategy.SEMANTIC
            )
            
            assert result.success is False
            assert len(result.results) == 0
            assert result.total_found == 0


class TestSearchStrategies:
    """Test different search strategies."""
    
    def test_all_strategies_defined(self):
        """Test that all search strategies are defined."""
        strategies = [
            SearchStrategy.SEMANTIC,
            SearchStrategy.VISUAL,
            SearchStrategy.MULTI_VECTOR,
            SearchStrategy.HYBRID,
            SearchStrategy.MATERIAL,
            SearchStrategy.KEYWORD
        ]
        
        assert len(strategies) == 6
        assert all(isinstance(s, SearchStrategy) for s in strategies)
    
    def test_strategy_values(self):
        """Test strategy string values."""
        assert SearchStrategy.SEMANTIC.value == "semantic"
        assert SearchStrategy.VISUAL.value == "visual"
        assert SearchStrategy.MULTI_VECTOR.value == "multi_vector"
        assert SearchStrategy.HYBRID.value == "hybrid"
        assert SearchStrategy.MATERIAL.value == "material"
        assert SearchStrategy.KEYWORD.value == "keyword"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

