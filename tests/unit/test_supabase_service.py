"""
Unit tests for Supabase service.

Tests the SupabaseClient class in isolation using mocks for external dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List
import json

from app.services.supabase_client import SupabaseClient


class TestSupabaseClient:
    """Test suite for SupabaseClient class."""

    @pytest.fixture
    def mock_supabase_instance(self):
        """Create a mocked Supabase client instance."""
        mock_client = MagicMock()
        
        # Mock table operations
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        
        # Mock select operations
        mock_select = MagicMock()
        mock_table.select.return_value = mock_select
        mock_select.execute.return_value = MagicMock(data=[], count=0)
        
        # Mock insert operations
        mock_insert = MagicMock()
        mock_table.insert.return_value = mock_insert
        mock_insert.execute.return_value = MagicMock(data=[{"id": 1}], count=1)
        
        # Mock update operations
        mock_update = MagicMock()
        mock_table.update.return_value = mock_update
        mock_update.eq.return_value.execute.return_value = MagicMock(data=[{"id": 1}], count=1)
        
        # Mock delete operations
        mock_delete = MagicMock()
        mock_table.delete.return_value = mock_delete
        mock_delete.eq.return_value.execute.return_value = MagicMock(data=[], count=1)
        
        # Mock storage operations
        mock_storage = MagicMock()
        mock_client.storage = mock_storage
        mock_bucket = MagicMock()
        mock_storage.from_.return_value = mock_bucket
        
        return mock_client

    @pytest.fixture
    def supabase_client(self, mock_supabase_instance):
        """Create a SupabaseClient instance with mocked dependencies."""
        with patch('app.services.supabase_client.create_client', return_value=mock_supabase_instance):
            client = SupabaseClient()
            return client

    def test_singleton_pattern(self, mock_supabase_instance):
        """Test that SupabaseClient follows singleton pattern."""
        with patch('app.services.supabase_client.create_client', return_value=mock_supabase_instance):
            client1 = SupabaseClient()
            client2 = SupabaseClient()
            
            assert client1 is client2

    def test_initialization(self, supabase_client, mock_supabase_instance):
        """Test SupabaseClient initialization."""
        assert supabase_client.client is mock_supabase_instance
        assert hasattr(supabase_client, '_instance')

    @pytest.mark.asyncio
    async def test_insert_document_success(self, supabase_client):
        """Test successful document insertion."""
        document_data = {
            "title": "Test Document",
            "content": "Test content",
            "file_path": "/test/path.pdf",
            "metadata": {"pages": 5}
        }
        
        # Mock successful insertion
        supabase_client.client.table.return_value.insert.return_value.execute.return_value = MagicMock(
            data=[{"id": 123, **document_data}],
            count=1
        )
        
        result = await supabase_client.insert_document(document_data)
        
        assert result["success"] is True
        assert result["data"]["id"] == 123
        assert result["data"]["title"] == "Test Document"
        
        # Verify the insert was called with correct data
        supabase_client.client.table.assert_called_with("documents")
        supabase_client.client.table.return_value.insert.assert_called_with(document_data)

    @pytest.mark.asyncio
    async def test_insert_document_failure(self, supabase_client):
        """Test document insertion failure."""
        document_data = {
            "title": "Test Document",
            "content": "Test content"
        }
        
        # Mock insertion failure
        supabase_client.client.table.return_value.insert.return_value.execute.side_effect = Exception("Database error")
        
        result = await supabase_client.insert_document(document_data)
        
        assert result["success"] is False
        assert "Database error" in result["error"]

    @pytest.mark.asyncio
    async def test_get_document_by_id_success(self, supabase_client):
        """Test successful document retrieval by ID."""
        document_id = 123
        expected_document = {
            "id": 123,
            "title": "Test Document",
            "content": "Test content"
        }
        
        # Mock successful retrieval
        supabase_client.client.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[expected_document],
            count=1
        )
        
        result = await supabase_client.get_document_by_id(document_id)
        
        assert result["success"] is True
        assert result["data"]["id"] == 123
        assert result["data"]["title"] == "Test Document"
        
        # Verify the select was called correctly
        supabase_client.client.table.assert_called_with("documents")
        supabase_client.client.table.return_value.select.assert_called_with("*")

    @pytest.mark.asyncio
    async def test_get_document_by_id_not_found(self, supabase_client):
        """Test document retrieval when document not found."""
        document_id = 999
        
        # Mock empty result
        supabase_client.client.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[],
            count=0
        )
        
        result = await supabase_client.get_document_by_id(document_id)
        
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_update_document_success(self, supabase_client):
        """Test successful document update."""
        document_id = 123
        update_data = {
            "title": "Updated Title",
            "content": "Updated content"
        }
        
        # Mock successful update
        supabase_client.client.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{"id": 123, **update_data}],
            count=1
        )
        
        result = await supabase_client.update_document(document_id, update_data)
        
        assert result["success"] is True
        assert result["data"]["title"] == "Updated Title"
        
        # Verify the update was called correctly
        supabase_client.client.table.assert_called_with("documents")
        supabase_client.client.table.return_value.update.assert_called_with(update_data)

    @pytest.mark.asyncio
    async def test_delete_document_success(self, supabase_client):
        """Test successful document deletion."""
        document_id = 123
        
        # Mock successful deletion
        supabase_client.client.table.return_value.delete.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[],
            count=1
        )
        
        result = await supabase_client.delete_document(document_id)
        
        assert result["success"] is True
        
        # Verify the delete was called correctly
        supabase_client.client.table.assert_called_with("documents")

    @pytest.mark.asyncio
    async def test_search_documents_success(self, supabase_client):
        """Test successful document search."""
        search_query = "test query"
        expected_documents = [
            {"id": 1, "title": "Test Doc 1", "content": "Test content 1"},
            {"id": 2, "title": "Test Doc 2", "content": "Test content 2"}
        ]
        
        # Mock successful search
        supabase_client.client.table.return_value.select.return_value.ilike.return_value.execute.return_value = MagicMock(
            data=expected_documents,
            count=2
        )
        
        result = await supabase_client.search_documents(search_query)
        
        assert result["success"] is True
        assert len(result["data"]) == 2
        assert result["data"][0]["id"] == 1

    @pytest.mark.asyncio
    async def test_get_documents_with_pagination(self, supabase_client):
        """Test document retrieval with pagination."""
        page = 2
        page_size = 10
        expected_documents = [
            {"id": 11, "title": "Doc 11"},
            {"id": 12, "title": "Doc 12"}
        ]
        
        # Mock paginated results
        mock_query = supabase_client.client.table.return_value.select.return_value
        mock_query.range.return_value.execute.return_value = MagicMock(
            data=expected_documents,
            count=2
        )
        
        result = await supabase_client.get_documents(page=page, page_size=page_size)
        
        assert result["success"] is True
        assert len(result["data"]) == 2
        
        # Verify pagination parameters
        start = (page - 1) * page_size
        end = start + page_size - 1
        mock_query.range.assert_called_with(start, end)

    @pytest.mark.asyncio
    async def test_store_vector_embedding_success(self, supabase_client):
        """Test successful vector embedding storage."""
        document_id = 123
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        chunk_text = "Sample chunk text"
        
        embedding_data = {
            "document_id": document_id,
            "embedding": embedding,
            "chunk_text": chunk_text,
            "chunk_index": 0
        }
        
        # Mock successful embedding storage
        supabase_client.client.table.return_value.insert.return_value.execute.return_value = MagicMock(
            data=[{"id": 456, **embedding_data}],
            count=1
        )
        
        result = await supabase_client.store_vector_embedding(document_id, embedding, chunk_text)

        assert result["success"] is True
        assert result["data"]["document_id"] == document_id

        # Verify the embedding was stored in the correct table
        supabase_client.client.table.assert_called_with("document_vectors")

    @pytest.mark.asyncio
    async def test_search_similar_vectors_success(self, supabase_client):
        """Test successful vector similarity search."""
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        limit = 5
        
        expected_results = [
            {"id": 1, "chunk_text": "Similar text 1", "similarity": 0.95},
            {"id": 2, "chunk_text": "Similar text 2", "similarity": 0.87}
        ]
        
        # Mock RPC call for vector search
        supabase_client.client.rpc.return_value.execute.return_value = MagicMock(
            data=expected_results,
            count=2
        )
        
        result = await supabase_client.search_similar_vectors(query_embedding, limit)
        
        assert result["success"] is True
        assert len(result["data"]) == 2
        assert result["data"][0]["similarity"] == 0.95
        
        # Verify RPC was called with correct parameters
        supabase_client.client.rpc.assert_called_with(
            "search_similar_vectors",
            {"query_embedding": query_embedding, "match_count": limit}
        )

    @pytest.mark.asyncio
    async def test_upload_file_success(self, supabase_client):
        """Test successful file upload to storage."""
        bucket_name = "documents"
        file_path = "test/document.pdf"
        file_data = b"PDF file content"
        
        # Mock successful file upload
        supabase_client.client.storage.from_.return_value.upload.return_value = MagicMock(
            data={"path": file_path},
            error=None
        )
        
        result = await supabase_client.upload_file(bucket_name, file_path, file_data)
        
        assert result["success"] is True
        assert result["data"]["path"] == file_path
        
        # Verify storage operations
        supabase_client.client.storage.from_.assert_called_with(bucket_name)

    @pytest.mark.asyncio
    async def test_upload_file_failure(self, supabase_client):
        """Test file upload failure."""
        bucket_name = "documents"
        file_path = "test/document.pdf"
        file_data = b"PDF file content"
        
        # Mock upload failure
        supabase_client.client.storage.from_.return_value.upload.return_value = MagicMock(
            data=None,
            error={"message": "Upload failed"}
        )
        
        result = await supabase_client.upload_file(bucket_name, file_path, file_data)
        
        assert result["success"] is False
        assert "Upload failed" in result["error"]

    @pytest.mark.asyncio
    async def test_get_file_url_success(self, supabase_client):
        """Test successful file URL generation."""
        bucket_name = "documents"
        file_path = "test/document.pdf"
        expected_url = "https://storage.supabase.co/object/public/documents/test/document.pdf"
        
        # Mock successful URL generation
        supabase_client.client.storage.from_.return_value.get_public_url.return_value = expected_url
        
        result = await supabase_client.get_file_url(bucket_name, file_path)
        
        assert result["success"] is True
        assert result["data"]["url"] == expected_url

    @pytest.mark.asyncio
    async def test_batch_insert_documents_success(self, supabase_client):
        """Test successful batch document insertion."""
        documents = [
            {"title": "Doc 1", "content": "Content 1"},
            {"title": "Doc 2", "content": "Content 2"},
            {"title": "Doc 3", "content": "Content 3"}
        ]
        
        # Mock successful batch insertion
        supabase_client.client.table.return_value.insert.return_value.execute.return_value = MagicMock(
            data=[{"id": i+1, **doc} for i, doc in enumerate(documents)],
            count=3
        )
        
        result = await supabase_client.batch_insert_documents(documents)
        
        assert result["success"] is True
        assert len(result["data"]) == 3
        
        # Verify batch insert was called
        supabase_client.client.table.return_value.insert.assert_called_with(documents)

    @pytest.mark.asyncio
    async def test_get_document_statistics_success(self, supabase_client):
        """Test successful document statistics retrieval."""
        expected_stats = {
            "total_documents": 150,
            "total_size": 1024000,
            "avg_processing_time": 2.5
        }
        
        # Mock RPC call for statistics
        supabase_client.client.rpc.return_value.execute.return_value = MagicMock(
            data=[expected_stats],
            count=1
        )
        
        result = await supabase_client.get_document_statistics()
        
        assert result["success"] is True
        assert result["data"]["total_documents"] == 150
        
        # Verify RPC was called
        supabase_client.client.rpc.assert_called_with("get_document_statistics")

    def test_connection_health_check(self, supabase_client):
        """Test database connection health check."""
        # Mock successful health check
        supabase_client.client.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[],
            count=0
        )
        
        result = supabase_client.health_check()
        
        assert result["healthy"] is True
        assert "connection" in result["status"].lower()

    def test_connection_health_check_failure(self, supabase_client):
        """Test database connection health check failure."""
        # Mock health check failure
        supabase_client.client.table.return_value.select.return_value.limit.return_value.execute.side_effect = Exception("Connection failed")
        
        result = supabase_client.health_check()
        
        assert result["healthy"] is False
        assert "Connection failed" in result["error"]