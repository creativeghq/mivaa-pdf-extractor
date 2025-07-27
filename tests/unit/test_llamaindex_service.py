"""
Unit tests for LlamaIndex RAG service.

Tests the LlamaIndexService class in isolation using mocks for external dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List
import json

from app.services.llamaindex_service import LlamaIndexService


class TestLlamaIndexService:
    """Test suite for LlamaIndexService class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mocked LLM instance."""
        mock_llm = AsyncMock()
        mock_llm.acomplete.return_value = MagicMock(text="Test response")
        return mock_llm

    @pytest.fixture
    def mock_embed_model(self):
        """Create a mocked embedding model."""
        mock_embed = MagicMock()
        mock_embed.get_text_embedding.return_value = [0.1] * 768
        return mock_embed

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mocked vector store."""
        mock_store = MagicMock()
        mock_store.add.return_value = None
        mock_store.query.return_value = MagicMock(
            nodes=[
                MagicMock(text="Sample text", score=0.95),
                MagicMock(text="Another text", score=0.87)
            ]
        )
        return mock_store

    @pytest.fixture
    def llamaindex_service(self, mock_llm, mock_embed_model, mock_vector_store):
        """Create a LlamaIndexService instance with mocked dependencies."""
        with patch('app.services.llamaindex_service.OpenAI', return_value=mock_llm), \
             patch('app.services.llamaindex_service.OpenAIEmbedding', return_value=mock_embed_model), \
             patch('app.services.llamaindex_service.VectorStoreIndex') as mock_index_class:
            
            mock_index = MagicMock()
            mock_index.as_query_engine.return_value = MagicMock()
            mock_index_class.from_vector_store.return_value = mock_index
            
            service = LlamaIndexService()
            service.vector_store = mock_vector_store
            service.index = mock_index
            return service

    @pytest.mark.asyncio
    async def test_initialization(self, llamaindex_service):
        """Test LlamaIndexService initialization."""
        assert llamaindex_service.llm is not None
        assert llamaindex_service.embed_model is not None
        assert llamaindex_service.vector_store is not None
        assert llamaindex_service.index is not None

    @pytest.mark.asyncio
    async def test_health_check_success(self, llamaindex_service):
        """Test successful health check."""
        # Mock successful embedding generation
        llamaindex_service.embed_model.get_text_embedding.return_value = [0.1] * 768
        
        result = await llamaindex_service.health_check()
        
        assert result["healthy"] is True
        assert "operational" in result["status"].lower()

    @pytest.mark.asyncio
    async def test_health_check_failure(self, llamaindex_service):
        """Test health check failure."""
        # Mock embedding generation failure
        llamaindex_service.embed_model.get_text_embedding.side_effect = Exception("Embedding failed")
        
        result = await llamaindex_service.health_check()
        
        assert result["healthy"] is False
        assert "Embedding failed" in result["error"]

    @pytest.mark.asyncio
    async def test_index_document_success(self, llamaindex_service):
        """Test successful document indexing."""
        document_data = {
            "id": "doc_123",
            "title": "Test Document",
            "content": "This is test content for indexing.",
            "metadata": {"source": "test"}
        }
        
        # Mock successful indexing
        with patch('app.services.llamaindex_service.Document') as mock_doc_class:
            mock_doc = MagicMock()
            mock_doc_class.return_value = mock_doc
            
            result = await llamaindex_service.index_document(document_data)
            
            assert result["success"] is True
            assert result["document_id"] == "doc_123"
            assert "indexed successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_index_document_failure(self, llamaindex_service):
        """Test document indexing failure."""
        document_data = {
            "id": "doc_123",
            "title": "Test Document",
            "content": "This is test content."
        }
        
        # Mock indexing failure
        llamaindex_service.index.insert.side_effect = Exception("Indexing failed")
        
        result = await llamaindex_service.index_document(document_data)
        
        assert result["success"] is False
        assert "Indexing failed" in result["error"]

    @pytest.mark.asyncio
    async def test_search_documents_success(self, llamaindex_service):
        """Test successful document search."""
        query = "test query"
        top_k = 5
        
        # Mock successful search
        mock_nodes = [
            MagicMock(text="Result 1", score=0.95, metadata={"doc_id": "1"}),
            MagicMock(text="Result 2", score=0.87, metadata={"doc_id": "2"})
        ]
        llamaindex_service.vector_store.query.return_value = MagicMock(nodes=mock_nodes)
        
        result = await llamaindex_service.search_documents(query, top_k)
        
        assert result["success"] is True
        assert len(result["results"]) == 2
        assert result["results"][0]["score"] == 0.95
        assert result["results"][0]["content"] == "Result 1"

    @pytest.mark.asyncio
    async def test_search_documents_no_results(self, llamaindex_service):
        """Test document search with no results."""
        query = "nonexistent query"
        
        # Mock empty search results
        llamaindex_service.vector_store.query.return_value = MagicMock(nodes=[])
        
        result = await llamaindex_service.search_documents(query)
        
        assert result["success"] is True
        assert len(result["results"]) == 0

    @pytest.mark.asyncio
    async def test_ask_question_success(self, llamaindex_service):
        """Test successful question answering."""
        question = "What is the main topic?"
        
        # Mock successful query engine response
        mock_response = MagicMock()
        mock_response.response = "The main topic is testing."
        mock_response.source_nodes = [
            MagicMock(text="Source 1", metadata={"doc_id": "1"}),
            MagicMock(text="Source 2", metadata={"doc_id": "2"})
        ]
        
        mock_query_engine = MagicMock()
        mock_query_engine.aquery.return_value = mock_response
        llamaindex_service.index.as_query_engine.return_value = mock_query_engine
        
        result = await llamaindex_service.ask_question(question)
        
        assert result["success"] is True
        assert result["answer"] == "The main topic is testing."
        assert len(result["sources"]) == 2

    @pytest.mark.asyncio
    async def test_ask_question_failure(self, llamaindex_service):
        """Test question answering failure."""
        question = "What is the main topic?"
        
        # Mock query engine failure
        mock_query_engine = MagicMock()
        mock_query_engine.aquery.side_effect = Exception("Query failed")
        llamaindex_service.index.as_query_engine.return_value = mock_query_engine
        
        result = await llamaindex_service.ask_question(question)
        
        assert result["success"] is False
        assert "Query failed" in result["error"]

    @pytest.mark.asyncio
    async def test_summarize_document_success(self, llamaindex_service):
        """Test successful document summarization."""
        document_id = "doc_123"
        
        # Mock successful summarization
        mock_response = MagicMock()
        mock_response.response = "This document discusses testing methodologies."
        
        mock_query_engine = MagicMock()
        mock_query_engine.aquery.return_value = mock_response
        llamaindex_service.index.as_query_engine.return_value = mock_query_engine
        
        result = await llamaindex_service.summarize_document(document_id)
        
        assert result["success"] is True
        assert "testing methodologies" in result["summary"]

    @pytest.mark.asyncio
    async def test_extract_entities_success(self, llamaindex_service):
        """Test successful entity extraction."""
        text = "John Smith works at OpenAI in San Francisco."
        
        # Mock successful entity extraction
        mock_response = MagicMock()
        mock_response.response = json.dumps({
            "entities": [
                {"text": "John Smith", "type": "PERSON"},
                {"text": "OpenAI", "type": "ORGANIZATION"},
                {"text": "San Francisco", "type": "LOCATION"}
            ]
        })
        
        llamaindex_service.llm.acomplete.return_value = mock_response
        
        result = await llamaindex_service.extract_entities(text)
        
        assert result["success"] is True
        assert len(result["entities"]) == 3
        assert result["entities"][0]["type"] == "PERSON"

    @pytest.mark.asyncio
    async def test_extract_entities_invalid_json(self, llamaindex_service):
        """Test entity extraction with invalid JSON response."""
        text = "John Smith works at OpenAI."
        
        # Mock invalid JSON response
        mock_response = MagicMock()
        mock_response.response = "Invalid JSON response"
        
        llamaindex_service.llm.acomplete.return_value = mock_response
        
        result = await llamaindex_service.extract_entities(text)
        
        assert result["success"] is False
        assert "JSON" in result["error"]

    @pytest.mark.asyncio
    async def test_compare_documents_success(self, llamaindex_service):
        """Test successful document comparison."""
        doc1_id = "doc_1"
        doc2_id = "doc_2"
        
        # Mock successful comparison
        mock_response = MagicMock()
        mock_response.response = json.dumps({
            "similarity_score": 0.75,
            "common_themes": ["testing", "automation"],
            "differences": ["methodology", "scope"]
        })
        
        llamaindex_service.llm.acomplete.return_value = mock_response
        
        result = await llamaindex_service.compare_documents(doc1_id, doc2_id)
        
        assert result["success"] is True
        assert result["similarity_score"] == 0.75
        assert len(result["common_themes"]) == 2

    @pytest.mark.asyncio
    async def test_get_similar_documents_success(self, llamaindex_service):
        """Test successful similar document retrieval."""
        document_id = "doc_123"
        top_k = 3
        
        # Mock successful similarity search
        mock_nodes = [
            MagicMock(text="Similar doc 1", score=0.85, metadata={"doc_id": "doc_456"}),
            MagicMock(text="Similar doc 2", score=0.78, metadata={"doc_id": "doc_789"})
        ]
        llamaindex_service.vector_store.query.return_value = MagicMock(nodes=mock_nodes)
        
        result = await llamaindex_service.get_similar_documents(document_id, top_k)
        
        assert result["success"] is True
        assert len(result["similar_documents"]) == 2
        assert result["similar_documents"][0]["similarity_score"] == 0.85

    @pytest.mark.asyncio
    async def test_batch_index_documents_success(self, llamaindex_service):
        """Test successful batch document indexing."""
        documents = [
            {"id": "doc_1", "title": "Doc 1", "content": "Content 1"},
            {"id": "doc_2", "title": "Doc 2", "content": "Content 2"},
            {"id": "doc_3", "title": "Doc 3", "content": "Content 3"}
        ]
        
        # Mock successful batch indexing
        with patch('app.services.llamaindex_service.Document') as mock_doc_class:
            mock_docs = [MagicMock() for _ in documents]
            mock_doc_class.side_effect = mock_docs
            
            result = await llamaindex_service.batch_index_documents(documents)
            
            assert result["success"] is True
            assert result["indexed_count"] == 3
            assert len(result["document_ids"]) == 3

    @pytest.mark.asyncio
    async def test_batch_index_documents_partial_failure(self, llamaindex_service):
        """Test batch document indexing with partial failures."""
        documents = [
            {"id": "doc_1", "title": "Doc 1", "content": "Content 1"},
            {"id": "doc_2", "title": "Doc 2", "content": "Content 2"}
        ]
        
        # Mock partial failure
        with patch('app.services.llamaindex_service.Document') as mock_doc_class:
            mock_doc_class.side_effect = [MagicMock(), Exception("Failed to create document")]
            
            result = await llamaindex_service.batch_index_documents(documents)
            
            assert result["success"] is True  # Partial success
            assert result["indexed_count"] == 1
            assert len(result["failed_documents"]) == 1

    @pytest.mark.asyncio
    async def test_delete_document_success(self, llamaindex_service):
        """Test successful document deletion."""
        document_id = "doc_123"
        
        # Mock successful deletion
        llamaindex_service.vector_store.delete.return_value = True
        
        result = await llamaindex_service.delete_document(document_id)
        
        assert result["success"] is True
        assert "deleted successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_delete_document_failure(self, llamaindex_service):
        """Test document deletion failure."""
        document_id = "doc_123"
        
        # Mock deletion failure
        llamaindex_service.vector_store.delete.side_effect = Exception("Deletion failed")
        
        result = await llamaindex_service.delete_document(document_id)
        
        assert result["success"] is False
        assert "Deletion failed" in result["error"]

    @pytest.mark.asyncio
    async def test_update_document_success(self, llamaindex_service):
        """Test successful document update."""
        document_id = "doc_123"
        updated_data = {
            "title": "Updated Title",
            "content": "Updated content",
            "metadata": {"version": 2}
        }
        
        # Mock successful update (delete + re-index)
        llamaindex_service.vector_store.delete.return_value = True
        
        with patch('app.services.llamaindex_service.Document') as mock_doc_class:
            mock_doc = MagicMock()
            mock_doc_class.return_value = mock_doc
            
            result = await llamaindex_service.update_document(document_id, updated_data)
            
            assert result["success"] is True
            assert "updated successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_get_index_statistics_success(self, llamaindex_service):
        """Test successful index statistics retrieval."""
        # Mock statistics
        with patch.object(llamaindex_service, '_get_vector_store_stats') as mock_stats:
            mock_stats.return_value = {
                "total_documents": 150,
                "total_vectors": 1500,
                "index_size": "2.5MB"
            }
            
            result = await llamaindex_service.get_index_statistics()
            
            assert result["success"] is True
            assert result["statistics"]["total_documents"] == 150

    @pytest.mark.asyncio
    async def test_semantic_search_with_filters_success(self, llamaindex_service):
        """Test successful semantic search with metadata filters."""
        query = "machine learning"
        filters = {"category": "research", "year": 2023}
        
        # Mock filtered search
        mock_nodes = [
            MagicMock(
                text="ML research paper",
                score=0.92,
                metadata={"doc_id": "paper_1", "category": "research", "year": 2023}
            )
        ]
        llamaindex_service.vector_store.query.return_value = MagicMock(nodes=mock_nodes)
        
        result = await llamaindex_service.semantic_search_with_filters(query, filters)
        
        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["metadata"]["category"] == "research"

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self, llamaindex_service):
        """Test successful embedding generation."""
        texts = ["Text 1", "Text 2", "Text 3"]
        
        # Mock embedding generation
        llamaindex_service.embed_model.get_text_embedding_batch.return_value = [
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768
        ]
        
        result = await llamaindex_service.generate_embeddings(texts)
        
        assert result["success"] is True
        assert len(result["embeddings"]) == 3
        assert len(result["embeddings"][0]) == 768

    @pytest.mark.asyncio
    async def test_generate_embeddings_failure(self, llamaindex_service):
        """Test embedding generation failure."""
        texts = ["Text 1", "Text 2"]
        
        # Mock embedding failure
        llamaindex_service.embed_model.get_text_embedding_batch.side_effect = Exception("Embedding failed")
        
        result = await llamaindex_service.generate_embeddings(texts)
        
        assert result["success"] is False
        assert "Embedding failed" in result["error"]

    def test_configuration_validation(self, llamaindex_service):
        """Test service configuration validation."""
        config = llamaindex_service.get_configuration()
        
        assert "llm_model" in config
        assert "embedding_model" in config
        assert "vector_store_type" in config
        assert isinstance(config["max_tokens"], int)

    @pytest.mark.asyncio
    async def test_clear_index_success(self, llamaindex_service):
        """Test successful index clearing."""
        # Mock successful index clearing
        llamaindex_service.vector_store.clear.return_value = True
        
        result = await llamaindex_service.clear_index()
        
        assert result["success"] is True
        assert "cleared successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_export_index_success(self, llamaindex_service):
        """Test successful index export."""
        export_path = "/tmp/index_export.json"
        
        # Mock successful export
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            result = await llamaindex_service.export_index(export_path)
            
            assert result["success"] is True
            assert export_path in result["export_path"]

    @pytest.mark.asyncio
    async def test_import_index_success(self, llamaindex_service):
        """Test successful index import."""
        import_path = "/tmp/index_export.json"
        
        # Mock successful import
        with patch('builtins.open', create=True) as mock_open, \
             patch('json.load') as mock_json_load:
            
            mock_json_load.return_value = {"documents": [], "metadata": {}}
            
            result = await llamaindex_service.import_index(import_path)
            
            assert result["success"] is True
            assert "imported successfully" in result["message"]