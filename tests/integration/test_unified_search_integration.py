"""
Integration Tests for Unified Search - Step 7

Tests the complete search flow:
- API endpoint integration
- Service integration
- Database integration
- End-to-end search workflows
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from app.services.unified_search_service import (
    UnifiedSearchService,
    SearchConfig,
    SearchStrategy
)


class TestUnifiedSearchIntegration:
    """Integration tests for unified search."""
    
    @pytest.fixture
    def mock_supabase(self):
        """Create mock Supabase client."""
        mock = Mock()
        mock.client = Mock()
        mock.client.rpc = Mock()
        return mock
    
    @pytest.fixture
    def search_service(self, mock_supabase):
        """Create search service."""
        config = SearchConfig()
        return UnifiedSearchService(config, mock_supabase)
    
    @pytest.mark.asyncio
    async def test_semantic_search_workflow(self, search_service):
        """Test complete semantic search workflow."""
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
                    "content": "Material properties: wood, oak, natural finish",
                    "similarity": 0.95,
                    "metadata": {"source": "pdf_1", "page": 5}
                },
                {
                    "id": "chunk_2",
                    "content": "Wood types: oak, maple, walnut",
                    "similarity": 0.88,
                    "metadata": {"source": "pdf_2", "page": 10}
                }
            ]
            search_service.supabase.client.rpc.return_value.execute.return_value = mock_response
            
            result = await search_service.search(
                query="wood materials",
                strategy=SearchStrategy.SEMANTIC
            )
            
            assert result.success is True
            assert len(result.results) == 2
            assert result.results[0].similarity_score == 0.95
            assert result.results[1].similarity_score == 0.88
            assert result.total_found == 2
    
    @pytest.mark.asyncio
    async def test_visual_search_workflow(self, search_service):
        """Test complete visual search workflow."""
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
                    "url": "https://example.com/wood_texture.jpg",
                    "similarity": 0.92,
                    "metadata": {"type": "texture", "material": "wood"}
                }
            ]
            search_service.supabase.client.rpc.return_value.execute.return_value = mock_response
            
            result = await search_service.search(
                query="wood texture",
                strategy=SearchStrategy.VISUAL
            )
            
            assert result.success is True
            assert len(result.results) == 1
            assert result.results[0].source_type == "image"
    
    @pytest.mark.asyncio
    async def test_multi_vector_search_workflow(self, search_service):
        """Test complete multi-vector search workflow."""
        with patch('app.services.unified_search_service.RealEmbeddingsService') as mock_embeddings:
            mock_service = AsyncMock()
            mock_embeddings.return_value = mock_service
            
            # Mock all embedding types
            mock_service.generate_all_embeddings = AsyncMock(
                return_value={
                    "success": True,
                    "embeddings": {
                        "text_1536": [0.1] * 1536,
                        "visual_clip_512": [0.2] * 512,
                        "multimodal_fusion_2048": [0.3] * 2048,
                        "color_256": [0.4] * 256,
                        "texture_256": [0.5] * 256,
                        "application_512": [0.6] * 512
                    }
                }
            )
            
            mock_service.generate_text_embedding = AsyncMock(
                return_value={
                    "success": True,
                    "embedding": [0.1] * 1536
                }
            )
            
            mock_service.generate_visual_embedding = AsyncMock(
                return_value={
                    "success": True,
                    "embedding": [0.2] * 512
                }
            )
            
            mock_response = Mock()
            mock_response.data = [
                {
                    "id": "product_1",
                    "content": "Oak wood furniture",
                    "similarity": 0.90,
                    "metadata": {"type": "product"}
                }
            ]
            search_service.supabase.client.rpc.return_value.execute.return_value = mock_response
            
            result = await search_service.search(
                query="oak furniture",
                strategy=SearchStrategy.MULTI_VECTOR
            )
            
            assert result.success is True
            assert result.strategy_used == "multi_vector"
    
    @pytest.mark.asyncio
    async def test_hybrid_search_workflow(self, search_service):
        """Test complete hybrid search workflow."""
        with patch('app.services.unified_search_service.RealEmbeddingsService') as mock_embeddings:
            mock_service = AsyncMock()
            mock_embeddings.return_value = mock_service
            mock_service.generate_text_embedding = AsyncMock(
                return_value={
                    "success": True,
                    "embedding": [0.1, 0.2, 0.3]
                }
            )
            
            # Mock semantic results
            semantic_response = Mock()
            semantic_response.data = [
                {
                    "id": "chunk_1",
                    "content": "Wood material",
                    "similarity": 0.95,
                    "metadata": {}
                }
            ]
            
            # Mock keyword results
            keyword_response = Mock()
            keyword_response.data = [
                {
                    "id": "chunk_2",
                    "content": "Wood type",
                    "rank": 0.85,
                    "metadata": {}
                }
            ]
            
            search_service.supabase.client.rpc.return_value.execute.side_effect = [
                semantic_response,
                keyword_response
            ]
            
            result = await search_service.search(
                query="wood",
                strategy=SearchStrategy.HYBRID
            )
            
            assert result.success is True
            assert result.strategy_used == "hybrid"
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, search_service):
        """Test search with metadata filters."""
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
                    "content": "Content",
                    "similarity": 0.95,
                    "metadata": {"source": "pdf_1"}
                }
            ]
            search_service.supabase.client.rpc.return_value.execute.return_value = mock_response
            
            filters = {"source": "pdf_1"}
            result = await search_service.search(
                query="test",
                strategy=SearchStrategy.SEMANTIC,
                filters=filters
            )
            
            assert result.success is True
    
    @pytest.mark.asyncio
    async def test_search_with_workspace_id(self, search_service):
        """Test search scoped to workspace."""
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
                    "content": "Content",
                    "similarity": 0.95,
                    "metadata": {}
                }
            ]
            search_service.supabase.client.rpc.return_value.execute.return_value = mock_response
            
            result = await search_service.search(
                query="test",
                strategy=SearchStrategy.SEMANTIC,
                workspace_id="workspace_123"
            )
            
            assert result.success is True
            assert result.metadata["workspace_id"] == "workspace_123"
    
    @pytest.mark.asyncio
    async def test_search_performance_tracking(self, search_service):
        """Test that search time is tracked."""
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
            mock_response.data = []
            search_service.supabase.client.rpc.return_value.execute.return_value = mock_response
            
            result = await search_service.search(
                query="test",
                strategy=SearchStrategy.SEMANTIC
            )
            
            assert result.search_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_search_empty_results(self, search_service):
        """Test search with no results."""
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
            mock_response.data = []
            search_service.supabase.client.rpc.return_value.execute.return_value = mock_response
            
            result = await search_service.search(
                query="nonexistent",
                strategy=SearchStrategy.SEMANTIC
            )
            
            assert result.success is True
            assert len(result.results) == 0
            assert result.total_found == 0
    
    @pytest.mark.asyncio
    async def test_search_similarity_threshold(self, search_service):
        """Test search respects similarity threshold."""
        search_service.config.similarity_threshold = 0.8
        
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
                    "content": "Content",
                    "similarity": 0.85,
                    "metadata": {}
                }
            ]
            search_service.supabase.client.rpc.return_value.execute.return_value = mock_response
            
            result = await search_service.search(
                query="test",
                strategy=SearchStrategy.SEMANTIC
            )
            
            assert result.metadata["similarity_threshold"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

